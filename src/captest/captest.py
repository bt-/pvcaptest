"""Unified test orchestrator and supporting utilities.

This module houses the ``CapTest`` class (added in a later unit), the
``TEST_SETUPS`` registry of named regression presets, and small formatting
helpers used by ``CapTest`` methods that compare a measured + modeled pair of
``CapData`` instances.

The module is intentionally light on imports at module scope to avoid any
circular dependency with ``captest.capdata``. Any function that needs a
``CapData`` instance accepts it as an argument rather than importing the class.
"""

import copy
import difflib
import importlib.util
from pathlib import Path

import numpy as np
import yaml

from captest import util
from captest.calcparams import (
    bom_temp,
    cell_temp,
    e_total,
    power_temp_correct,
    rpoa_pvsyst,
)

_hv_spec = importlib.util.find_spec("holoviews")
if _hv_spec is not None:
    import holoviews as hv
else:  # pragma: no cover - optional dep
    hv = None


def print_results(test_passed, expected, actual, cap_ratio, capacity, bounds):
    """Print formatted results of a capacity test.

    Parameters
    ----------
    test_passed : tuple of (bool, str)
        Pass/fail flag and bounds string produced by
        ``CapTest.determine_pass_or_fail`` (or the legacy module-level
        ``determine_pass_or_fail`` in ``capdata.py`` until Unit 7 removes it).
    expected : float
        Predicted modeled test output at reporting conditions.
    actual : float
        Predicted measured test output at reporting conditions.
    cap_ratio : float
        Capacity test ratio (``actual / expected``).
    capacity : float
        Tested capacity (``nameplate * cap_ratio``).
    bounds : str
        Human-readable bounds string for the test tolerance.
    """
    if test_passed[0]:
        print("{:<30s}{}".format("Capacity Test Result:", "PASS"))
    else:
        print("{:<25s}{}".format("Capacity Test Result:", "FAIL"))

    print(
        "{:<30s}{:0.3f}".format("Modeled test output:", expected)
        + "\n"
        + "{:<30s}{:0.3f}".format("Actual test output:", actual)
        + "\n"
        + "{:<30s}{:0.3f}".format("Tested output ratio:", cap_ratio)
        + "\n"
        + "{:<30s}{:0.3f}".format("Tested Capacity:", capacity)
    )

    print("{:<30s}{}\n\n".format("Bounds:", bounds))


def highlight_pvals(s):
    """Highlight Series entries >= 0.05 with a yellow background.

    Intended for use with ``pandas.io.formats.style.Styler.apply``. Consumed by
    ``CapTest.captest_results_check_pvalues`` (ported in Unit 7).
    """
    is_greaterthan = s >= 0.05
    return ["background-color: yellow" if v else "" for v in is_greaterthan]


def perc_wrap(p):
    """Return a callable that computes the ``p``-th percentile of a Series.

    Used to build ``TEST_SETUPS[...]['rep_conditions']['func']`` dicts for
    percentile-based reporting irradiance (e.g. 60th percentile POA).

    Parameters
    ----------
    p : numeric
        Percentile in [0, 100].

    Returns
    -------
    callable
        Function that takes a pandas Series or array-like and returns the
        p-th percentile using ``method='nearest'``.
    """

    def numpy_percentile(x):
        return np.percentile(x.T, p, method="nearest")

    numpy_percentile.__name__ = f"perc_wrap({p})"
    return numpy_percentile


# --- TEST_SETUPS registry -------------------------------------------------


def _scatter_formula_xy(cd, x_key=None):
    """Resolve (x_col, y_col, df) for a formula-agnostic scatter.

    The y variable is the lhs of ``cd.regression_formula``; the x variable is
    the first rhs by default or ``x_key`` if given. ``df`` is the filtered
    regression-columns DataFrame with columns renamed to the formula-variable
    names.
    """
    lhs, rhs = util.parse_regression_formula(cd.regression_formula)
    y_col = lhs[0]
    x_col = x_key if x_key is not None else rhs[0]
    reg_vars = list({y_col, *rhs})
    df = cd.get_reg_cols(reg_vars=reg_vars)
    return x_col, y_col, df


def scatter_default(cd, **kwargs):
    """Formula-agnostic scatter of regression lhs vs. first rhs variable.

    Parameters
    ----------
    cd : CapData
        Must have ``regression_formula`` set and ``regression_cols`` resolved
        (e.g. via ``CapTest.setup()`` or ``cd.process_regression_columns()``).
    **kwargs
        Forwarded to ``hv.Scatter.opts``.

    Returns
    -------
    hv.Layout
        A single-panel Layout wrapping the scatter plot. Layout (not Scatter)
        is returned for a uniform return type across the shipped callables.
    """
    if hv is None:
        raise ImportError(
            "holoviews is required for scatter_default. Install with "
            "`uv add holoviews` or equivalent."
        )
    x_col, y_col, df = _scatter_formula_xy(cd)
    df = df.reset_index().rename(columns={df.index.name or "index": "index"})
    scatter = hv.Scatter(df, x_col, [y_col, "index"]).opts(size=5, **kwargs)
    return hv.Layout([scatter])


def scatter_etotal(cd, **kwargs):
    """Single scatter of regression lhs vs. the ``e_total`` column.

    Intended for the ``bifi_e2848_etotal`` preset. Resolves the x column from
    ``cd.regression_cols['poa']`` after ``process_regression_columns`` has
    materialized the calculated e_total column.
    """
    if hv is None:
        raise ImportError("holoviews is required for scatter_etotal.")
    return scatter_default(cd, **kwargs)


def scatter_bifi_power_tc(cd, **kwargs):
    """Two-panel layout: lhs vs. ``poa`` and lhs vs. ``rpoa``.

    Intended for the ``bifi_power_tc`` preset whose regression formula is
    ``power ~ poa + rpoa`` (with ``power`` resolved to the temperature-
    corrected calculated column). Each rhs variable gets its own panel.
    """
    if hv is None:
        raise ImportError("holoviews is required for scatter_bifi_power_tc.")
    lhs, rhs = util.parse_regression_formula(cd.regression_formula)
    y_col = lhs[0]
    reg_vars = list({y_col, *rhs})
    df = cd.get_reg_cols(reg_vars=reg_vars).reset_index()
    df = df.rename(columns={df.columns[0]: "index"})
    panels = [
        hv.Scatter(df, x_col, [y_col, "index"]).opts(size=5, **kwargs) for x_col in rhs
    ]
    return hv.Layout(panels)


TEST_SETUPS = {
    "e2848_default": {
        "reg_cols_meas": {
            "power": ("real_pwr_mtr", "sum"),
            "poa": ("irr_poa", "mean"),
            "t_amb": ("temp_amb", "mean"),
            "w_vel": ("wind_speed", "mean"),
        },
        "reg_cols_sim": {
            "power": "E_Grid",
            "poa": "GlobInc",
            "t_amb": "T_Amb",
            "w_vel": "WindVel",
        },
        "reg_fml": "power ~ poa + I(poa * poa) + I(poa * t_amb) + I(poa * w_vel) - 1",
        "scatter_plots": scatter_default,
        "rep_conditions": {
            "irr_bal": False,
            "percent_filter": 20,
            "front_poa": "poa",
            "func": {
                "poa": perc_wrap(60),
                "t_amb": "mean",
                "w_vel": "mean",
            },
        },
    },
    "bifi_e2848_etotal": {
        "reg_cols_meas": {
            "power": ("real_pwr_mtr", "sum"),
            "poa": (
                e_total,
                {
                    "poa": ("irr_poa", "mean"),
                    "rpoa": ("irr_rpoa", "mean"),
                },
            ),
            "t_amb": ("temp_amb", "mean"),
            "w_vel": ("wind_speed", "mean"),
        },
        "reg_cols_sim": {
            "power": "E_Grid",
            "poa": (
                e_total,
                {
                    "poa": "GlobInc",
                    "rpoa": (
                        rpoa_pvsyst,
                        {"globbak": "GlobBak", "backshd": "BackShd"},
                    ),
                },
            ),
            "t_amb": "T_Amb",
            "w_vel": "WindVel",
        },
        "reg_fml": "power ~ poa + I(poa * poa) + I(poa * t_amb) + I(poa * w_vel) - 1",
        "scatter_plots": scatter_etotal,
        "rep_conditions": {
            "irr_bal": False,
            "percent_filter": 20,
            "front_poa": "poa",
            "func": {
                "poa": perc_wrap(60),
                "t_amb": "mean",
                "w_vel": "mean",
            },
        },
    },
    "bifi_power_tc": {
        "reg_cols_meas": {
            "power": (
                power_temp_correct,
                {
                    "power": ("real_pwr_mtr", "sum"),
                    "cell_temp": (
                        cell_temp,
                        {
                            "poa": ("irr_poa", "mean"),
                            "bom": (
                                bom_temp,
                                {
                                    "poa": ("irr_poa", "mean"),
                                    "temp_amb": ("temp_amb", "mean"),
                                    "wind_speed": ("wind_speed", "mean"),
                                },
                            ),
                        },
                    ),
                },
            ),
            "poa": ("irr_poa", "mean"),
            "rpoa": ("irr_rpoa", "mean"),
        },
        "reg_cols_sim": {
            "power": (
                power_temp_correct,
                {
                    "power": "E_Grid",
                    "cell_temp": "TArray",
                },
            ),
            "poa": "GlobInc",
            "rpoa": (rpoa_pvsyst, {"globbak": "GlobBak", "backshd": "BackShd"}),
        },
        "reg_fml": "power ~ poa + rpoa",
        "scatter_plots": scatter_bifi_power_tc,
        "rep_conditions": {
            "irr_bal": False,
            "percent_filter": 20,
            "front_poa": "poa",
            "func": {
                "poa": perc_wrap(60),
                "rpoa": "mean",
            },
        },
    },
}

_TEST_SETUP_REQUIRED_KEYS = frozenset(
    {"reg_cols_meas", "reg_cols_sim", "reg_fml", "scatter_plots", "rep_conditions"}
)


def validate_test_setup(entry):
    """Validate a single ``TEST_SETUPS`` entry dict.

    Raises
    ------
    KeyError
        If required keys are missing or unknown keys are present.
    ValueError
        If ``reg_fml`` does not parse, lhs+rhs are not subsets of both
        ``reg_cols_meas`` and ``reg_cols_sim``, ``scatter_plots`` is not
        callable, or ``rep_conditions`` / ``rep_conditions['func']`` have an
        unexpected shape.
    """
    keys = set(entry.keys())
    missing = _TEST_SETUP_REQUIRED_KEYS - keys
    if missing:
        raise KeyError(f"TEST_SETUPS entry missing required keys: {sorted(missing)}")
    extra = keys - _TEST_SETUP_REQUIRED_KEYS
    if extra:
        raise KeyError(f"TEST_SETUPS entry has unknown keys: {sorted(extra)}")

    lhs, rhs = util.parse_regression_formula(entry["reg_fml"])
    formula_vars = set(lhs) | set(rhs)
    for side in ("reg_cols_meas", "reg_cols_sim"):
        if not isinstance(entry[side], dict):
            raise ValueError(f"{side!r} must be a dict.")
        missing_vars = formula_vars - set(entry[side].keys())
        if missing_vars:
            raise ValueError(
                f"{side!r} is missing keys required by reg_fml: {sorted(missing_vars)}"
            )

    if not callable(entry["scatter_plots"]):
        raise ValueError("'scatter_plots' must be callable.")

    rc = entry["rep_conditions"]
    if not isinstance(rc, dict):
        raise ValueError("'rep_conditions' must be a dict.")
    func = rc.get("func")
    if func is not None and isinstance(func, dict):
        extra_func = set(func.keys()) - set(rhs)
        if extra_func:
            raise ValueError(
                "'rep_conditions[\"func\"]' has keys that are not rhs "
                f"variables of reg_fml: {sorted(extra_func)}"
            )


def _merge_rep_conditions(base, override):
    """Partial-merge ``override`` onto ``base`` rep_conditions dict.

    Top-level keys in ``override`` replace corresponding keys in ``base``.
    If both have ``func`` dicts, the ``override['func']`` is merged one level
    deep (per-variable) onto ``base['func']``.
    """
    merged = copy.deepcopy(base)
    if not override:
        return merged
    for key, val in override.items():
        if (
            key == "func"
            and isinstance(val, dict)
            and isinstance(merged.get("func"), dict)
        ):
            merged_func = copy.deepcopy(merged["func"])
            merged_func.update(val)
            merged["func"] = merged_func
        else:
            merged[key] = copy.deepcopy(val)
    return merged


def resolve_test_setup(name, overrides=None):
    """Resolve a preset by name plus optional overrides.

    Parameters
    ----------
    name : str
        Key into ``TEST_SETUPS`` or the literal ``"custom"``.
    overrides : dict or None
        Optional dict with any of ``reg_cols_meas``, ``reg_cols_sim``,
        ``reg_fml``, ``scatter_plots``, ``rep_conditions`` to override the
        preset. ``rep_conditions`` is partial-merged; other keys replace.
        When ``name == "custom"``, ``reg_cols_meas``, ``reg_cols_sim``, and
        ``reg_fml`` are required in ``overrides``.

    Returns
    -------
    dict
        A fully-validated entry dict suitable for ``CapTest._resolved_setup``.
    """
    overrides = overrides or {}
    if name == "custom":
        required = {"reg_cols_meas", "reg_cols_sim", "reg_fml"}
        missing = required - set(overrides.keys())
        if missing:
            raise ValueError(
                f"test_setup='custom' requires overrides with keys: {sorted(required)}; "
                f"missing: {sorted(missing)}"
            )
        base = {
            "reg_cols_meas": copy.deepcopy(overrides["reg_cols_meas"]),
            "reg_cols_sim": copy.deepcopy(overrides["reg_cols_sim"]),
            "reg_fml": overrides["reg_fml"],
            "scatter_plots": overrides.get("scatter_plots", scatter_default),
            "rep_conditions": copy.deepcopy(overrides.get("rep_conditions", {})),
        }
    else:
        if name not in TEST_SETUPS:
            available = sorted(TEST_SETUPS.keys()) + ["custom"]
            raise KeyError(f"Unknown test_setup={name!r}. Available: {available}")
        base = copy.deepcopy(TEST_SETUPS[name])
        for key in ("reg_cols_meas", "reg_cols_sim", "reg_fml", "scatter_plots"):
            if overrides.get(key) is not None:
                base[key] = copy.deepcopy(overrides[key])
        if overrides.get("rep_conditions"):
            base["rep_conditions"] = _merge_rep_conditions(
                base["rep_conditions"], overrides["rep_conditions"]
            )

    validate_test_setup(base)
    return base


# --- yaml loading ---------------------------------------------------------

_PERC_N_PREFIX = "perc_"


def _resolve_perc_string(val):
    """Resolve a "perc_N" string to ``perc_wrap(N)``.

    Non-matching strings pass through unchanged. Malformed ``perc_*`` strings
    raise ``ValueError``.
    """
    if not isinstance(val, str) or not val.startswith(_PERC_N_PREFIX):
        return val
    suffix = val[len(_PERC_N_PREFIX) :]
    if not suffix:
        raise ValueError(f"Malformed percentile string {val!r}: expected 'perc_<int>'.")
    try:
        n = int(suffix)
    except ValueError as exc:
        raise ValueError(
            f"Malformed percentile string {val!r}: expected 'perc_<int>', "
            f"got suffix {suffix!r}."
        ) from exc
    return perc_wrap(n)


def _resolve_func_strings(func_dict):
    """Resolve ``perc_N`` strings inside a rep_conditions.func dict."""
    if not isinstance(func_dict, dict):
        return func_dict
    return {key: _resolve_perc_string(val) for key, val in func_dict.items()}


def load_config(path, key="captest"):
    """Load and lightly validate the captest sub-mapping from a yaml file.

    Parameters
    ----------
    path : str or Path
        Path to the yaml file. Relative paths in ``meas_path`` / ``sim_path``
        are resolved by callers using ``Path(path).parent`` as the base.
    key : str, default 'captest'
        Top-level key whose value is the CapTest configuration sub-mapping.

    Returns
    -------
    dict
        The sub-mapping at ``key`` with string shorthands resolved. Does NOT
        validate against ``CapTest`` param types; ``CapTest.from_yaml`` does
        that.

    Raises
    ------
    KeyError
        If ``key`` is not present at the top level of the yaml file.
    """
    path = Path(path)
    with path.open("r") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"Top level of yaml file {path!s} must be a mapping; got {type(raw).__name__}."
        )
    if key not in raw:
        available = sorted(raw.keys())
        suggestion = difflib.get_close_matches(key, available, n=1)
        hint = f" Did you mean {suggestion[0]!r}?" if suggestion else ""
        raise KeyError(
            f"Top-level key {key!r} not found in {path!s}. "
            f"Top-level keys present: {available}.{hint}"
        )
    sub = raw[key]
    if not isinstance(sub, dict):
        raise ValueError(
            f"Value at {key!r} must be a mapping; got {type(sub).__name__}."
        )
    # Resolve perc_N shorthand in overrides.rep_conditions.func.
    overrides = sub.get("overrides") or {}
    if isinstance(overrides, dict) and isinstance(
        overrides.get("rep_conditions"), dict
    ):
        func_dict = overrides["rep_conditions"].get("func")
        if isinstance(func_dict, dict):
            overrides["rep_conditions"]["func"] = _resolve_func_strings(func_dict)
    # Also resolve top-level rep_conditions.func if someone put it there.
    rc = sub.get("rep_conditions")
    if isinstance(rc, dict) and isinstance(rc.get("func"), dict):
        rc["func"] = _resolve_func_strings(rc["func"])
    return sub


def _suggest_unknown_key(unknown, known):
    """Return a 'did you mean X?' hint or empty string."""
    matches = difflib.get_close_matches(unknown, list(known), n=1)
    return f" Did you mean {matches[0]!r}?" if matches else ""


# Silence ruff F401: these are public API; re-imported by `capdata.py`.
__all__ = [
    "TEST_SETUPS",
    "highlight_pvals",
    "load_config",
    "perc_wrap",
    "print_results",
    "resolve_test_setup",
    "scatter_bifi_power_tc",
    "scatter_default",
    "scatter_etotal",
    "validate_test_setup",
]
