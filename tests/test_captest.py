"""Tests for captest.captest module.

Covers the TEST_SETUPS registry, the validate_test_setup / resolve_test_setup
/ load_config helpers, the three shipped scatter-plot callables, and the
``CapTest`` orchestrator class itself (Units 4 and 6 of the implementation
plan).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from captest import CapTest, captest as ct
from captest.calcparams import e_total, power_temp_correct


class TestTestSetupsRegistry:
    @pytest.mark.parametrize("preset", list(ct.TEST_SETUPS.keys()))
    def test_each_shipped_preset_validates(self, preset):
        """Each shipped preset dict passes validate_test_setup as-is."""
        ct.validate_test_setup(ct.TEST_SETUPS[preset])

    @pytest.mark.parametrize("preset", list(ct.TEST_SETUPS.keys()))
    def test_each_preset_lhs_is_power(self, preset):
        """Naming convention: lhs regression variable is always 'power'."""
        entry = ct.TEST_SETUPS[preset]
        lhs = entry["reg_fml"].split("~")[0].strip()
        assert lhs == "power", f"preset {preset!r} lhs is {lhs!r}, expected 'power'"

    @pytest.mark.parametrize("preset", list(ct.TEST_SETUPS.keys()))
    def test_each_preset_has_rep_conditions_dict(self, preset):
        rc = ct.TEST_SETUPS[preset]["rep_conditions"]
        assert isinstance(rc, dict)
        assert "func" in rc
        assert isinstance(rc["func"], dict)

    def test_e2848_default_shape(self):
        """Sanity-check the default preset's reg_cols keys."""
        entry = ct.TEST_SETUPS["e2848_default"]
        assert set(entry["reg_cols_meas"].keys()) == {"power", "poa", "t_amb", "w_vel"}
        assert set(entry["reg_cols_sim"].keys()) == {"power", "poa", "t_amb", "w_vel"}

    def test_bifi_e2848_etotal_uses_e_total(self):
        """bifi_e2848_etotal preset wraps poa in an e_total calc-tuple."""
        entry = ct.TEST_SETUPS["bifi_e2848_etotal"]
        meas_poa = entry["reg_cols_meas"]["poa"]
        assert isinstance(meas_poa, tuple)
        assert meas_poa[0] is e_total

    def test_bifi_power_tc_uses_power_temp_correct(self):
        entry = ct.TEST_SETUPS["bifi_power_tc"]
        meas_power = entry["reg_cols_meas"]["power"]
        assert isinstance(meas_power, tuple)
        assert meas_power[0] is power_temp_correct

    def test_validate_rejects_unknown_keys(self):
        bad = dict(ct.TEST_SETUPS["e2848_default"])
        bad["bogus"] = 42
        with pytest.raises(KeyError, match="unknown keys"):
            ct.validate_test_setup(bad)

    def test_validate_rejects_missing_keys(self):
        bad = dict(ct.TEST_SETUPS["e2848_default"])
        bad.pop("rep_conditions")
        with pytest.raises(KeyError, match="missing required keys"):
            ct.validate_test_setup(bad)

    def test_validate_rejects_non_callable_scatter_plots(self):
        bad = dict(ct.TEST_SETUPS["e2848_default"])
        bad["scatter_plots"] = "not-a-callable"
        with pytest.raises(ValueError, match="scatter_plots"):
            ct.validate_test_setup(bad)

    def test_validate_rejects_formula_vars_missing_from_reg_cols(self):
        bad = dict(ct.TEST_SETUPS["e2848_default"])
        bad["reg_cols_meas"] = {
            k: v for k, v in bad["reg_cols_meas"].items() if k != "w_vel"
        }
        with pytest.raises(ValueError, match="missing keys required by reg_fml"):
            ct.validate_test_setup(bad)

    def test_validate_rejects_non_dict_rep_conditions(self):
        bad = dict(ct.TEST_SETUPS["e2848_default"])
        bad["rep_conditions"] = "not-a-dict"
        with pytest.raises(ValueError, match="'rep_conditions' must be a dict"):
            ct.validate_test_setup(bad)

    def test_validate_rejects_func_with_non_rhs_keys(self):
        bad = dict(ct.TEST_SETUPS["e2848_default"])
        # Reconstruct rep_conditions with an extra func key.
        rc = dict(bad["rep_conditions"])
        rc["func"] = dict(rc["func"])
        rc["func"]["bogus"] = "mean"
        bad["rep_conditions"] = rc
        with pytest.raises(ValueError, match="rhs variables"):
            ct.validate_test_setup(bad)


class TestResolveTestSetup:
    def test_named_preset_no_overrides(self):
        resolved = ct.resolve_test_setup("e2848_default")
        assert resolved["reg_fml"] == ct.TEST_SETUPS["e2848_default"]["reg_fml"]

    def test_named_preset_with_reg_fml_override(self):
        # Keep the same rhs variables (otherwise rep_conditions.func keys no
        # longer align with rhs and validate_test_setup rejects the result).
        resolved = ct.resolve_test_setup(
            "e2848_default",
            overrides={
                "reg_fml": "power ~ poa + I(poa * poa) + I(poa * t_amb) + I(poa * w_vel)"
            },
        )
        assert resolved["reg_fml"].startswith("power ~ poa")

    def test_named_preset_with_rep_conditions_partial_merge(self):
        resolved = ct.resolve_test_setup(
            "e2848_default",
            overrides={"rep_conditions": {"percent_filter": 10}},
        )
        # Top-level percent_filter replaced, other keys preserved.
        assert resolved["rep_conditions"]["percent_filter"] == 10
        assert resolved["rep_conditions"]["irr_bal"] is False
        # func dict untouched.
        assert set(resolved["rep_conditions"]["func"].keys()) == {
            "poa",
            "t_amb",
            "w_vel",
        }

    def test_named_preset_with_rep_conditions_func_partial_merge(self):
        resolved = ct.resolve_test_setup(
            "e2848_default",
            overrides={"rep_conditions": {"func": {"poa": ct.perc_wrap(55)}}},
        )
        # POA entry swapped; others preserved.
        assert resolved["rep_conditions"]["func"]["t_amb"] == "mean"
        assert resolved["rep_conditions"]["func"]["w_vel"] == "mean"
        assert callable(resolved["rep_conditions"]["func"]["poa"])

    def test_unknown_preset_raises(self):
        with pytest.raises(KeyError, match="Unknown test_setup"):
            ct.resolve_test_setup("nonexistent")

    def test_custom_requires_all_three_overrides(self):
        with pytest.raises(ValueError, match="test_setup='custom'"):
            ct.resolve_test_setup("custom", overrides={"reg_fml": "y ~ x"})

    def test_custom_with_minimal_overrides(self):
        resolved = ct.resolve_test_setup(
            "custom",
            overrides={
                "reg_cols_meas": {"power": "p", "poa": "i"},
                "reg_cols_sim": {"power": "p", "poa": "i"},
                "reg_fml": "power ~ poa",
            },
        )
        assert resolved["reg_fml"] == "power ~ poa"
        # scatter_plots falls back to scatter_default.
        assert resolved["scatter_plots"] is ct.scatter_default
        # rep_conditions defaults to empty dict.
        assert resolved["rep_conditions"] == {}


class TestLoadConfig:
    def test_happy_path(self, tmp_path):
        yaml_text = "captest:\n  test_setup: e2848_default\n  ac_nameplate: 125000\n"
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml_text)
        sub = ct.load_config(p)
        assert sub["test_setup"] == "e2848_default"
        assert sub["ac_nameplate"] == 125000

    def test_missing_key_raises_with_suggestion(self, tmp_path):
        yaml_text = "captset:\n  test_setup: e2848_default\n"
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml_text)
        with pytest.raises(KeyError, match="captset"):
            ct.load_config(p)

    def test_custom_key(self, tmp_path):
        yaml_text = "captest_bifi:\n  test_setup: bifi_e2848_etotal\n"
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml_text)
        sub = ct.load_config(p, key="captest_bifi")
        assert sub["test_setup"] == "bifi_e2848_etotal"

    def test_top_level_not_a_mapping_raises(self, tmp_path):
        p = tmp_path / "cfg.yaml"
        p.write_text("- just a list\n")
        with pytest.raises(ValueError, match="must be a mapping"):
            ct.load_config(p)

    def test_perc_N_string_resolved_in_overrides(self, tmp_path):
        yaml_text = (
            "captest:\n"
            "  test_setup: e2848_default\n"
            "  overrides:\n"
            "    rep_conditions:\n"
            "      func:\n"
            "        poa: perc_55\n"
            "        t_amb: mean\n"
        )
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml_text)
        sub = ct.load_config(p)
        func = sub["overrides"]["rep_conditions"]["func"]
        # 'mean' passes through.
        assert func["t_amb"] == "mean"
        # 'perc_55' resolves to perc_wrap(55).
        sample = pd.Series(np.arange(100))
        expected = ct.perc_wrap(55)(sample)
        actual = func["poa"](sample)
        assert actual == expected

    def test_malformed_perc_string_raises(self, tmp_path):
        yaml_text = (
            "captest:\n"
            "  overrides:\n"
            "    rep_conditions:\n"
            "      func:\n"
            "        poa: perc_xx\n"
        )
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml_text)
        with pytest.raises(ValueError, match="perc_<int>"):
            ct.load_config(p)


class TestPercWrap:
    def test_returns_callable(self):
        f = ct.perc_wrap(60)
        assert callable(f)

    def test_computes_percentile(self):
        f = ct.perc_wrap(60)
        sample = pd.Series(np.arange(100))
        # method='nearest' returns the nearest existing value; compare against
        # np.percentile with the same method to avoid rounding ambiguity.
        expected = np.percentile(sample, 60, method="nearest")
        assert f(sample) == expected

    def test_name_encodes_percentile(self):
        f = ct.perc_wrap(55)
        assert f.__name__ == "perc_wrap(55)"


class TestScatterCallables:
    """Smoke tests for the shipped scatter callables using a synthetic CapData."""

    def _make_synthetic_cd(self, formula, columns):
        from captest.capdata import CapData

        cd = CapData("test")
        idx = pd.date_range("2024-01-01", periods=50, freq="1min")
        data = {col: np.linspace(1, 50, 50) for col in columns}
        cd.data = pd.DataFrame(data, index=idx)
        cd.data_filtered = cd.data.copy()
        cd.column_groups = {col: [col] for col in columns}
        cd.regression_cols = {col: col for col in columns}
        cd.regression_formula = formula
        return cd

    def test_scatter_default_returns_layout(self):
        import holoviews as hv

        cd = self._make_synthetic_cd("power ~ poa - 1", ["power", "poa"])
        layout = ct.scatter_default(cd)
        assert isinstance(layout, hv.Layout)

    def test_scatter_etotal_returns_layout(self):
        import holoviews as hv

        cd = self._make_synthetic_cd("power ~ poa + rpoa", ["power", "poa", "rpoa"])
        layout = ct.scatter_etotal(cd)
        assert isinstance(layout, hv.Layout)

    def test_scatter_bifi_power_tc_has_two_panels(self):
        import holoviews as hv

        cd = self._make_synthetic_cd("power ~ poa + rpoa", ["power", "poa", "rpoa"])
        layout = ct.scatter_bifi_power_tc(cd)
        assert isinstance(layout, hv.Layout)
        assert len(layout) == 2


class TestConstruction:
    """Construction paths for CapTest: bare init, from_params, and kwargs."""

    def test_bare_init_has_defaults(self):
        capt = CapTest()
        assert capt.test_setup == "e2848_default"
        assert capt.meas is None
        assert capt.sim is None
        assert capt.test_tolerance == "- 4"
        assert capt.bifaciality == 0.0
        assert capt.power_temp_coeff == -0.32
        assert capt.base_temp == 25
        assert capt._resolved_setup is None

    def test_bare_init_accepts_kwargs(self):
        capt = CapTest(
            test_setup="bifi_e2848_etotal",
            ac_nameplate=125_000,
            bifaciality=0.15,
        )
        assert capt.test_setup == "bifi_e2848_etotal"
        assert capt.ac_nameplate == 125_000
        assert capt.bifaciality == 0.15

    def test_bare_init_rejects_unknown_kwarg(self):
        with pytest.raises(TypeError):
            CapTest(bogus_kwarg=1)

    def test_class_level_downstream_attrs(self):
        assert CapTest._downstream_attrs == (
            "bifaciality",
            "power_temp_coeff",
            "base_temp",
        )

    def test_from_params_with_capdata_instances_triggers_setup(
        self, meas_cd_default, sim_cd_default
    ):
        capt = CapTest.from_params(
            test_setup="e2848_default",
            meas=meas_cd_default,
            sim=sim_cd_default,
            ac_nameplate=6_000_000,
        )
        assert capt._resolved_setup is not None
        # process_regression_columns should have resolved the aggregated poa
        # column name on both CapData instances.
        assert capt.meas.regression_cols["poa"] == "irr_poa_mean_agg"
        assert capt.sim.regression_cols["poa"] == "GlobInc"

    def test_from_params_partial_leaves_unset_defers_setup(self, meas_cd_default):
        capt = CapTest.from_params(
            test_setup="e2848_default",
            meas=meas_cd_default,
        )
        assert capt._resolved_setup is None
        assert capt.meas is meas_cd_default
        assert capt.sim is None

    def test_from_params_pre_built_meas_wins_over_path(
        self, meas_cd_default, sim_cd_default
    ):
        with pytest.warns(UserWarning, match="pre-built"):
            capt = CapTest.from_params(
                test_setup="e2848_default",
                meas=meas_cd_default,
                meas_path="/nonexistent/should/not/be/opened.csv",
                sim=sim_cd_default,
            )
        assert capt.meas is meas_cd_default

    def test_from_params_loads_data_via_custom_loader(self, meas_cd_default):
        meas_loader = MagicMock(return_value=meas_cd_default)
        capt = CapTest.from_params(
            test_setup="e2848_default",
            meas_path="/fake/path",
            meas_loader=meas_loader,
        )
        meas_loader.assert_called_once_with("/fake/path")
        assert capt.meas is meas_cd_default


class TestFromYaml:
    """Yaml-driven construction."""

    def _write(self, tmp_path, text, name="cfg.yaml"):
        p = tmp_path / name
        p.write_text(text)
        return p

    def test_happy_path(self, captest_yaml):
        capt = CapTest.from_yaml(captest_yaml)
        assert capt.test_setup == "e2848_default"
        assert capt.ac_nameplate == 6_000_000
        assert capt.test_tolerance == "- 4"

    def test_unknown_key_raises_with_suggestion(self, tmp_path):
        p = self._write(
            tmp_path,
            "captest:\n  test_setup: e2848_default\n  ac_namplate: 1\n",
        )
        with pytest.raises(ValueError, match="ac_namplate"):
            CapTest.from_yaml(p)

    def test_missing_test_setup_raises(self, tmp_path):
        p = self._write(tmp_path, "captest:\n  ac_nameplate: 100\n")
        with pytest.raises(ValueError, match="test_setup"):
            CapTest.from_yaml(p)

    def test_custom_setup_requires_overrides(self, tmp_path):
        p = self._write(tmp_path, "captest:\n  test_setup: custom\n")
        with pytest.raises(ValueError, match="custom"):
            CapTest.from_yaml(p)

    def test_conflicting_reg_fml_raises(self, tmp_path):
        text = (
            "captest:\n"
            "  test_setup: e2848_default\n"
            "  reg_fml: power ~ poa\n"
            "  overrides:\n"
            "    reg_fml: power ~ poa + t_amb\n"
        )
        p = self._write(tmp_path, text)
        with pytest.raises(ValueError, match="reg_fml"):
            CapTest.from_yaml(p)

    def test_null_values_equivalent_to_absence(self, tmp_path):
        text = (
            "captest:\n"
            "  test_setup: e2848_default\n"
            "  ac_nameplate: null\n"
            "  reg_fml: null\n"
        )
        p = self._write(tmp_path, text)
        capt = CapTest.from_yaml(p)
        assert capt.ac_nameplate is None
        assert capt.reg_fml is None

    def test_relative_paths_resolve_to_yaml_dir(
        self, tmp_path, monkeypatch, meas_cd_default
    ):
        """Relative meas_path / sim_path in yaml resolve against yaml dir."""
        text = "captest:\n  test_setup: e2848_default\n  meas_path: ./subdir/meas.csv\n"
        p = self._write(tmp_path, text, name="cfg.yaml")
        received = {}

        def fake_load_data(path, **kwargs):
            received["path"] = path
            return meas_cd_default

        # from_yaml pulls the default loader via _default_meas_loader, which
        # returns captest.io.load_data. Patch that here.
        monkeypatch.setattr("captest.io.load_data", fake_load_data)
        CapTest.from_yaml(p)
        assert received["path"] == str(tmp_path / "subdir" / "meas.csv")


class TestSetup:
    """Behavior of CapTest.setup()."""

    def test_setup_requires_meas(self, sim_cd_default):
        capt = CapTest(sim=sim_cd_default)
        with pytest.raises(RuntimeError, match="meas"):
            capt.setup()

    def test_setup_requires_sim(self, meas_cd_default):
        capt = CapTest(meas=meas_cd_default)
        with pytest.raises(RuntimeError, match="sim"):
            capt.setup()

    def test_setup_returns_self(self, ct_default):
        # from_params already invoked setup(); re-run returns self.
        assert ct_default.setup(verbose=False) is ct_default

    def test_setup_propagates_downstream_attrs_to_both_cd(
        self, meas_cd_default, sim_cd_default
    ):
        capt = CapTest.from_params(
            test_setup="bifi_e2848_etotal",
            meas=meas_cd_default,
            sim=sim_cd_default,
            bifaciality=0.22,
            power_temp_coeff=-0.41,
            base_temp=20,
        )
        for attr in CapTest._downstream_attrs:
            assert getattr(capt.meas, attr) == getattr(capt, attr)
            assert getattr(capt.sim, attr) == getattr(capt, attr)

    @pytest.mark.parametrize("preset", list(ct.TEST_SETUPS.keys()))
    def test_setup_wires_regression_formula(
        self, preset, meas_cd_default, sim_cd_default
    ):
        capt = CapTest.from_params(
            test_setup=preset,
            meas=meas_cd_default,
            sim=sim_cd_default,
            bifaciality=0.15,
        )
        expected_fml = ct.TEST_SETUPS[preset]["reg_fml"]
        assert capt.meas.regression_formula == expected_fml
        assert capt.sim.regression_formula == expected_fml

    def test_setup_wires_tolerance(self, ct_default):
        assert ct_default.meas.tolerance == "- 4"
        assert ct_default.sim.tolerance == "- 4"

    def test_setup_assigns_resolved_setup(self, ct_default):
        resolved = ct_default._resolved_setup
        assert resolved is not None
        assert set(resolved.keys()) == {
            "reg_cols_meas",
            "reg_cols_sim",
            "reg_fml",
            "scatter_plots",
            "rep_conditions",
        }

    def test_setup_rerun_resets_data_filtered(self, meas_cd_default, sim_cd_default):
        capt = CapTest.from_params(
            test_setup="e2848_default",
            meas=meas_cd_default,
            sim=sim_cd_default,
        )
        # Simulate a filter step: shrink data_filtered.
        capt.meas.data_filtered = capt.meas.data_filtered.iloc[:100].copy()
        assert capt.meas.data_filtered.shape[0] == 100
        capt.setup(verbose=False)
        # process_regression_columns resets data_filtered = data.copy().
        assert capt.meas.data_filtered.shape[0] == capt.meas.data.shape[0]

    def test_setup_verbose_prints_aggregations(
        self, meas_cd_default, sim_cd_default, capsys
    ):
        CapTest.from_params(
            test_setup="e2848_default",
            meas=meas_cd_default,
            sim=sim_cd_default,
        )  # from_params uses verbose default (True).
        captured = capsys.readouterr()
        assert "Aggregating the below columns" in captured.out

    def test_setup_verbose_false_silent(self, meas_cd_default, sim_cd_default, capsys):
        capt = CapTest(
            test_setup="e2848_default",
            meas=meas_cd_default,
            sim=sim_cd_default,
        )
        capt.setup(verbose=False)
        captured = capsys.readouterr()
        assert "Aggregating the below columns" not in captured.out

    def test_setup_applies_rep_conditions_override(
        self, meas_cd_default, sim_cd_default
    ):
        capt = CapTest.from_params(
            test_setup="e2848_default",
            meas=meas_cd_default,
            sim=sim_cd_default,
            rep_conditions={"percent_filter": 10},
        )
        resolved_rc = capt._resolved_setup["rep_conditions"]
        assert resolved_rc["percent_filter"] == 10
        # Non-overridden preset keys are preserved.
        assert resolved_rc["irr_bal"] is False
        assert set(resolved_rc["func"].keys()) == {"poa", "t_amb", "w_vel"}


class TestDownstreamPropagation:
    """Calc-params scalars on CapTest flow through to calcparams functions."""

    def test_bifaciality_flows_into_e_total(self, meas_cd_default, sim_cd_default):
        capt = CapTest.from_params(
            test_setup="bifi_e2848_etotal",
            meas=meas_cd_default,
            sim=sim_cd_default,
            bifaciality=0.5,
        )
        # Sanity: e_total = poa + rpoa * bifaciality (bifacial_frac=1,
        # rear_shade=0 by default). Extract the first non-zero row.
        meas_df = capt.meas.data
        mask = meas_df["irr_poa_mean_agg"] > 0
        first = meas_df.loc[mask].iloc[0]
        expected = first["irr_poa_mean_agg"] + first["irr_rpoa_mean_agg"] * 0.5
        assert first["e_total"] == pytest.approx(expected)

    def test_power_temp_coeff_flows_into_power_temp_correct(
        self, meas_cd_default, sim_cd_default
    ):
        capt = CapTest.from_params(
            test_setup="bifi_power_tc",
            meas=meas_cd_default,
            sim=sim_cd_default,
            bifaciality=0.15,
            power_temp_coeff=-0.5,
            base_temp=25,
        )
        # The power_temp_correct column lives on sim.data; its formula is
        # power / (1 + (coeff/100) * (cell_temp - base_temp)).
        sim_df = capt.sim.data
        first = sim_df.iloc[10]  # avoid the nighttime zeros at the top
        if first["E_Grid"] == 0:
            # grab a daytime row
            first = sim_df.loc[sim_df["E_Grid"] > 0].iloc[0]
        expected = first["E_Grid"] / (1 + (-0.5 / 100) * (first["TArray"] - 25))
        assert first["power_temp_correct"] == pytest.approx(expected)

    def test_base_temp_flows_into_power_temp_correct(
        self, meas_cd_default, sim_cd_default
    ):
        capt = CapTest.from_params(
            test_setup="bifi_power_tc",
            meas=meas_cd_default,
            sim=sim_cd_default,
            bifaciality=0.15,
            base_temp=35,
        )
        sim_df = capt.sim.data
        first = sim_df.loc[sim_df["E_Grid"] > 0].iloc[0]
        expected = first["E_Grid"] / (1 + (-0.32 / 100) * (first["TArray"] - 35))
        assert first["power_temp_correct"] == pytest.approx(expected)


class TestLoaderInjection:
    """Loader callable defaults, overrides, and kwarg splatting."""

    def test_default_meas_loader_is_load_data(self):
        from captest.io import load_data

        assert ct._default_meas_loader() is load_data

    def test_default_sim_loader_is_load_pvsyst(self):
        from captest.io import load_pvsyst

        assert ct._default_sim_loader() is load_pvsyst

    def test_custom_meas_loader_called_with_path_and_kwargs(self, meas_cd_default):
        loader = MagicMock(return_value=meas_cd_default)
        CapTest.from_params(
            test_setup="e2848_default",
            meas_path="/fake/path",
            meas_loader=loader,
            meas_load_kwargs={"period": "2024-01", "groups": ["a", "b"]},
        )
        loader.assert_called_once_with(
            "/fake/path", period="2024-01", groups=["a", "b"]
        )

    def test_custom_sim_loader_called_with_path_and_kwargs(self, sim_cd_default):
        loader = MagicMock(return_value=sim_cd_default)
        CapTest.from_params(
            test_setup="e2848_default",
            sim_path="/fake/sim/path",
            sim_loader=loader,
            sim_load_kwargs={"egrid_unit_adj_factor": 1000},
        )
        loader.assert_called_once_with("/fake/sim/path", egrid_unit_adj_factor=1000)


class TestRepCondConvenience:
    """CapTest.rep_cond and CapTest.scatter_plots convenience methods."""

    def test_rep_cond_requires_setup(self):
        capt = CapTest()
        with pytest.raises(RuntimeError, match="setup"):
            capt.rep_cond()

    def test_rep_cond_calls_cd_rep_cond_with_resolved_defaults(self, ct_default):
        # Patch cd.rep_cond to capture the kwargs passed through.
        received = {}

        def fake_rep_cond(**kwargs):
            received.update(kwargs)

        ct_default.meas.rep_cond = fake_rep_cond
        ct_default.rep_cond()
        preset_rc = ct.TEST_SETUPS["e2848_default"]["rep_conditions"]
        assert received["percent_filter"] == preset_rc["percent_filter"]
        assert received["irr_bal"] == preset_rc["irr_bal"]
        assert received["front_poa"] == preset_rc["front_poa"]
        assert set(received["func"].keys()) == set(preset_rc["func"].keys())

    def test_rep_cond_partial_merge_overrides(self, ct_default):
        received = {}

        def fake_rep_cond(**kwargs):
            received.update(kwargs)

        ct_default.meas.rep_cond = fake_rep_cond
        ct_default.rep_cond(percent_filter=10)
        assert received["percent_filter"] == 10
        # Preset keys preserved.
        assert received["irr_bal"] is False
        assert set(received["func"].keys()) == {"poa", "t_amb", "w_vel"}

    def test_rep_cond_func_partial_merge(self, ct_default):
        received = {}

        def fake_rep_cond(**kwargs):
            received.update(kwargs)

        new_poa_fn = ct.perc_wrap(55)
        ct_default.meas.rep_cond = fake_rep_cond
        ct_default.rep_cond(func={"poa": new_poa_fn})
        assert received["func"]["poa"] is new_poa_fn
        # Preserved from preset.
        assert received["func"]["t_amb"] == "mean"
        assert received["func"]["w_vel"] == "mean"

    def test_rep_cond_which_sim(self, ct_default):
        received = {}

        def fake_sim_rep_cond(**kwargs):
            received["target"] = "sim"
            received.update(kwargs)

        def fake_meas_rep_cond(**kwargs):
            received["target"] = "meas"
            received.update(kwargs)

        ct_default.meas.rep_cond = fake_meas_rep_cond
        ct_default.sim.rep_cond = fake_sim_rep_cond
        ct_default.rep_cond(which="sim")
        assert received["target"] == "sim"

    def test_rep_cond_which_invalid(self, ct_default):
        with pytest.raises(ValueError, match="must be 'meas' or 'sim'"):
            ct_default.rep_cond(which="bogus")

    def test_rep_conditions_override_from_init_partial_merges(
        self, meas_cd_default, sim_cd_default
    ):
        capt = CapTest.from_params(
            test_setup="e2848_default",
            meas=meas_cd_default,
            sim=sim_cd_default,
            rep_conditions={"percent_filter": 10},
        )
        resolved_rc = capt._resolved_setup["rep_conditions"]
        assert resolved_rc["percent_filter"] == 10
        assert resolved_rc["irr_bal"] is False
        assert "func" in resolved_rc

    @pytest.mark.parametrize("preset", list(ct.TEST_SETUPS.keys()))
    def test_each_preset_rep_conditions_round_trips_through_rep_cond(
        self, preset, meas_cd_default, sim_cd_default
    ):
        capt = CapTest.from_params(
            test_setup=preset,
            meas=meas_cd_default,
            sim=sim_cd_default,
            bifaciality=0.15,
        )
        # Should not raise.
        capt.rep_cond()
        assert isinstance(capt.meas.rc, pd.DataFrame)

    def test_scatter_plots_requires_setup(self):
        capt = CapTest()
        with pytest.raises(RuntimeError, match="setup"):
            capt.scatter_plots()

    def test_scatter_plots_dispatches_to_resolved_callable(self, ct_default):
        import holoviews as hv

        layout = ct_default.scatter_plots()
        assert isinstance(layout, hv.Layout)

    def test_scatter_plots_which_sim(self, ct_default):
        import holoviews as hv

        layout = ct_default.scatter_plots(which="sim")
        assert isinstance(layout, hv.Layout)


class TestResolvedSetupProperty:
    def test_property_requires_setup(self):
        capt = CapTest()
        with pytest.raises(RuntimeError, match="setup"):
            capt.resolved_setup

    def test_property_returns_resolved_dict(self, ct_default):
        resolved = ct_default.resolved_setup
        assert resolved is ct_default._resolved_setup
