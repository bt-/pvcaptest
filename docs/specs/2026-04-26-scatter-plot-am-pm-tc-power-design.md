# Scatter Plot Features: AM/PM Split + Temperature-Corrected Power

Status: Draft (awaiting user review)
Author: paired session with Oz
Date: 2026-04-26
Branch context: `captest-class`

## Problem
The shipped `scatter_plots` callables for `TEST_SETUPS` (`scatter_default`,
`scatter_etotal`, `scatter_bifi_power_tc` in `src/captest/captest.py`) are
formula-agnostic but visually flat:

- They cannot show morning vs afternoon points differently, which is a
  common diagnostic for soiling, tracker bias, and dew effects.
- They cannot show `power` plotted as temperature-corrected power on
  presets where `power` is raw (`e2848_default`, `bifi_e2848_etotal`,
  `e2848_spec_corrected_poa`).

A standalone helper (`scatter_hv`) exists in user notebooks today that
splits AM/PM but is not integrated with the `CapTest`/`TEST_SETUPS`
pipeline, doesn't share styling with the rest of `captest`, and has no
support for tc-power.

## Goals
1. Add an opt-in AM/PM glyph split to the scatter plots produced by
   `CapTest.scatter_plots`.
2. Add an opt-in temperature-corrected-power view to those scatter
   plots, isolated from `regression_cols` and `process_regression_columns`.
3. Preserve flexibility for arbitrary `TEST_SETUPS` and user-provided
   regression formulas.
4. Make the new functionality dashboard-ready by encapsulating it in a
   `param.Parameterized` class without forcing all scatter callables to
   become classes.

## Non-goals
- Refactoring `scatter_default` / `scatter_etotal` /
  `scatter_bifi_power_tc` into class-based presets across the board.
- Per-day solar noon detection (single-value detection only).
- Yaml round-trip of the new options (deferred until usage stabilizes).
- Sim/PVsyst tc-power defaults (sim users must pass
  `tc_power_calc` explicitly).

## High-level design
A new public `param.Parameterized` class `ScatterPlot` lives in a new
module `src/captest/scatter_plots.py`. The shipped TEST_SETUPS callables
become 1-line wrappers that instantiate it. Composable module-level
helpers do the AM/PM and tc-power work and are independently usable in
notebooks and dashboards.
### Module layout
- New module: `src/captest/scatter_plots.py`
  - `class ScatterPlot(param.Parameterized)`
  - `class ScatterBifiPowerTc(ScatterPlot)` (2-panel variant)
  - `def detect_solar_noon(data, ghi_col="ghi_mod_csky", default="12:30")`
  - `def add_am_pm_dim(df, split_time)`
  - `def ensure_tc_power_column(cd, tc_power_calc, col_name="power_tc_plot", verbose=False, force_recompute=False)`
  - `DEFAULT_TC_POWER_CALC` module constant
  - `TC_POWER_PLOT_COL = "power_tc_plot"` constant
- `src/captest/captest.py` is modified to:
  - Import the wrapper functions from `scatter_plots.py`.
  - Replace bodies of `scatter_default`, `scatter_etotal`,
    `scatter_bifi_power_tc` with 1-line `ScatterPlot(...).view()` /
    `ScatterBifiPowerTc(...).view()` calls. Public symbols and
    signatures unchanged.
- `src/captest/__init__.py`: re-export `ScatterPlot`,
  `detect_solar_noon`, `add_am_pm_dim`, `ensure_tc_power_column` from
  the new module so existing `from captest import ...` import sites are
  uniform.

### `ScatterPlot` parameter surface
```python
class ScatterPlot(param.Parameterized):
    # Data binding
    cd = param.ClassSelector(class_=CapData, default=None, precedence=-1)
    filtered = param.Boolean(True)

    # AM/PM split
    split_day = param.Boolean(False)
    split_time = param.String(default=None, allow_None=True)
    am_color = param.Color(default="#1f77b4")
    pm_color = param.Color(default="#d62728")
    am_marker = param.Selector(
        objects=["circle", "triangle", "square", "x", "diamond"],
        default="circle",
    )
    pm_marker = param.Selector(
        objects=["circle", "triangle", "square", "x", "diamond"],
        default="triangle",
    )

    # Temperature-corrected power
    tc_power = param.Boolean(False)
    tc_mode = param.Selector(
        objects=["replace", "add_panel", "overlay"], default="replace"
    )
    tc_power_calc = param.Dict(default=None, allow_None=True)
    tc_force_recompute = param.Boolean(False)

    # Timeseries pairing
    timeseries = param.Boolean(False)

    # Sizing
    height = param.Integer(400)
    width = param.Integer(500)
```

### Wrapper integration with TEST_SETUPS
```python
# src/captest/captest.py
from captest.scatter_plots import ScatterPlot, ScatterBifiPowerTc

def scatter_default(cd, **kwargs):
    return ScatterPlot(cd=cd, **kwargs).view()

def scatter_etotal(cd, **kwargs):
    return ScatterPlot(cd=cd, **kwargs).view()

def scatter_bifi_power_tc(cd, **kwargs):
    return ScatterBifiPowerTc(cd=cd, **kwargs).view()
```
TEST_SETUPS values are unchanged. `validate_test_setup`'s
`callable(...)` check still passes. `overlay_scatters` keeps using
`list(layout)[0]` to grab the principal element (a `Scatter` for single
panels, an `Overlay` when `split_day=True`, both relabel-compatible).

## Component contracts
### `detect_solar_noon(data, ghi_col="ghi_mod_csky", default="12:30") -> str`
- If `ghi_col` is in `data.columns`: groups by clock-time-of-day mean of
  `data[ghi_col]`, returns the idxmax formatted as `"HH:MM"`.
- If absent: emits `UserWarning` once and returns `default`.
- If `data.index` is empty: emits `UserWarning` and returns `default`.

### `add_am_pm_dim(df, split_time) -> DataFrame`
- Validates `split_time` matches `r"^\d{1,2}:\d{2}$"`. Raises
  `ValueError` otherwise.
- Returns a copy of `df` with a new categorical column `period` whose
  values are `"am"` for rows with index time `<` `split_time` and
  `"pm"` otherwise.
- DST handling: relies on naive clock time via `between_time` semantics;
  this matches user mental model (split at "wall-clock noon").

### `ensure_tc_power_column(cd, tc_power_calc, col_name="power_tc_plot", verbose=False, force_recompute=False) -> str`
- If `col_name` already exists in `cd.data` and `force_recompute is
  False`, returns `col_name` immediately.
- Wraps `captest.util.transform_calc_params` to evaluate the
  `tc_power_calc` expression. The result Series is written to
  `cd.data[col_name]` AND `cd.data_filtered[col_name]` (preserving the
  filter row index).
- Side effects: writes a new column to BOTH `cd.data` and
  `cd.data_filtered`. Does NOT mutate `cd.regression_cols`,
  `cd.regression_formula`, `cd.summary`, `cd.kept`, or `cd.removed`.
- If a referenced column group is missing, raises a `KeyError`
  including the missing group id and a hint to override
  `tc_power_calc`.

### `DEFAULT_TC_POWER_CALC`
Tuned for measured DAS data following the standard column-group
inference (`captest.columngroups.group_columns`):
```python
DEFAULT_TC_POWER_CALC = {
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
}
```
Sim users must pass `tc_power_calc` explicitly (raw column references
into PVsyst output, e.g. `{"power": "E_Grid", "cell_temp": "TArray"}`).
### `ScatterPlot.view() -> hv.Layout`
Logical flow:
1. Resolve `df = cd.get_reg_cols(filtered_data=filtered)`. Keep the
   DatetimeIndex on `df` for now; the index reset happens in step 4
   AFTER tc-power columns are joined and AFTER `add_am_pm_dim` writes
   its `period` column.
2. If `tc_power`:
   - If `tc_power_calc is None`, use `DEFAULT_TC_POWER_CALC`.
   - If the regression formula already targets a tc-power column
     (`cd.regression_formula` lhs resolves to a column whose name is
     `power_temp_correct.__name__`), emit a `UserWarning` and skip the
     extra calc; treat `tc_mode` as `replace`.
   - Otherwise call `ensure_tc_power_column(cd, tc_power_calc,
     force_recompute=self.tc_force_recompute)` and join the resulting
     column onto `df` by index from `cd.data_filtered` (or `cd.data`
     when `filtered=False`).
3. If `split_day`:
   - Resolve `split_time = self.split_time or detect_solar_noon(cd.data)`.
   - `df = add_am_pm_dim(df, split_time)`.
4. Reset the index to a column named `"index"` on `df`. This is the
   form holoviews expects for hover-friendly tooltips.
5. Build the principal scatter(s):
   - Pick `y_col` based on `tc_mode` (`"power"`, tc col, or both for
     overlay/add_panel).
   - For each y_col, build either a single `hv.Scatter` (no split) or
     `hv.Scatter(am) * hv.Scatter(pm)` overlay with distinct
     `am_color` / `pm_color` and `am_marker` / `pm_marker` applied via
     `hv.opts.Scatter`.
6. Compose into `hv.Layout` per `tc_mode`:
   - `replace`: `Layout([principal])`.
   - `add_panel`: `Layout([raw_principal, tc_principal])`.
   - `overlay`: `Layout([raw_principal * tc_principal])`.
7. If `timeseries`:
   - Raise `ValueError` if `tc_mode == "add_panel"`.
   - Add a linked timeseries panel beneath the principal (port the
     existing `cd.scatter_hv(timeseries=True)` logic, including
     `DataLink`).
8. Return the `hv.Layout`.

### `ScatterBifiPowerTc.view() -> hv.Layout`
Subclass that overrides the principal scatter construction for the
`bifi_power_tc` formula (`power ~ poa + rpoa`):
- Builds two scatters: `power vs poa` and `power vs rpoa`.
- Returns `Layout([poa_panel, rpoa_panel])`.
- Inherits `split_day`, `split_time`, color/marker, `timeseries` (when
  `timeseries=True`, only the first panel is paired with a timeseries
  view to keep the layout sane).
- Does NOT use `tc_power` (warns and ignores) — the regression `power`
  is already tc-corrected for this preset.

### Dashboard support
`ScatterPlot.dashboard()` returns `pn.Row(self.param, self.view)`. All
view-affecting parameters are decorated with `@param.depends(...)` so
the dashboard rerenders on widget changes. Out of scope: panel layout
polish (sliders/tabs/etc.). The `dashboard()` method exists for the
follow-on dashboard work; it doesn't have to look pretty in this PR.

## Backward compatibility
- `scatter_default`, `scatter_etotal`, `scatter_bifi_power_tc` keep
  their public signatures and `hv.Layout` return type.
- `cd.scatter_hv()` and `cd.scatter()` are untouched.
- `CapTest.scatter_plots(which, **kwargs)`,
  `CapTest.overlay_scatters(...)`, and `validate_test_setup` are
  unchanged.
- Default behavior of every shipped call site is identical: opting in
  to the new features requires `split_day=True` or `tc_power=True`.

## Edge cases and error messages
- `tc_power=True` on a preset whose regression power is already
  tc-corrected: warn, ignore.
- `tc_power_calc` references a missing column group: `KeyError` with
  the missing group id and a hint to pass an explicit dict.
- `split_time` doesn't parse: `ValueError`.
- No `ghi_mod_csky` column: `UserWarning` (once per call) and fall back
  to `12:30`.
- `overlay_scatters` with `tc_power=True`: documented as
  measured-data-only; sim users must pass an explicit
  `tc_power_calc`. We will not split kwargs into `meas_kwargs`/
  `sim_kwargs` in this PR.

## Testing strategy
All new tests use pytest (project rule).

- `tests/test_scatter_plots.py`:
  - `detect_solar_noon`: with `ghi_mod_csky` present, with absent
    column (warn + default), with empty index (warn + default).
  - `add_am_pm_dim`: correct categorization, ValueError on bad
    `split_time`.
  - `ensure_tc_power_column`: writes to `cd.data` and
    `cd.data_filtered`, idempotent on second call, recomputes when
    `force_recompute=True`, raises on missing column group.
  - `ScatterPlot.view`: returns `hv.Layout`; overlay shape on
    `split_day=True`; correct y column on `tc_mode='replace'`; correct
    panel count on `tc_mode='add_panel'`; ValueError on
    `tc_mode='add_panel' + timeseries=True`.
  - `ScatterBifiPowerTc.view`: returns 2-panel Layout; `tc_power=True`
    triggers the warn+ignore path.

- Backward-compat regression tests (in existing
  `tests/test_captest.py` if present, otherwise add): assert
  `scatter_default(cd)` returns the same Layout shape as before this
  change (single-element Layout containing a Scatter).

## Implementation steps (for the follow-on plan)
1. Create `src/captest/scatter_plots.py` with helpers and
   `DEFAULT_TC_POWER_CALC`.
2. Implement `ScatterPlot` with `split_day` + tc-power logic.
3. Implement `ScatterBifiPowerTc` subclass.
4. Refactor wrappers in `captest.py`.
5. Re-export from `__init__.py`.
6. Add tests.
7. Update `docs/userguide` (or equivalent) with a short section on the
   new options.
8. Run `just lint`, `just fmt`, `just test`.

## Open questions / deferred
- Per-day solar noon detection.
- Yaml round-trip for `split_day` / `tc_power` / `tc_power_calc`.
- Promoting `tc_power_calc` to a TEST_SETUPS or CapTest param (option
  E in the brainstorming session).
- Sim-side default `tc_power_calc`.
- `overlay_scatters` mixed-source kwargs (`meas_kwargs` / `sim_kwargs`).
