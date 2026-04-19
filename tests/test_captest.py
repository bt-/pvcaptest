"""Tests for captest.captest module.

Covers Unit 4 of the CapTest implementation plan: TEST_SETUPS registry,
validate_test_setup / resolve_test_setup / load_config helpers, and the three
shipped scatter-plot callables.

Fixtures and later test classes for the CapTest class itself are added in Unit
6 and onward.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from captest import captest as ct
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
