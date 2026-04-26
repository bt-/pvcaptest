.. _captest:

CapTest
=======
:py:class:`~captest.captest.CapTest` is the unified entry point for running a
capacity test against a pair of measured and modeled :py:class:`~captest.capdata.CapData`
instances. It binds both instances together, holds every test-level setting
(regression formula, reporting-irradiance recipe, irradiance / shade filter
bounds, nameplate, tolerance, and calc-params scalars like bifaciality), and
drives a single source of truth from a yaml config file so the same test can be
reproduced from a notebook, GUI, or command-line workflow.

:py:class:`~captest.captest.CapTest` is a config + state container, not a
runner. After ``setup()``, users still call ``ct.meas.filter_*(...)``,
``ct.meas.fit_regression()``, and the other :py:class:`~captest.capdata.CapData`
methods directly; see :ref:`dataload` for that workflow. The two things
:py:class:`~captest.captest.CapTest` takes over are **regression-preset
resolution** (via :py:data:`~captest.captest.TEST_SETUPS`) and all
**cross-``CapData``** comparison and plotting methods that previously lived at
module level.

Regression presets (``TEST_SETUPS``)
------------------------------------
:py:data:`~captest.captest.TEST_SETUPS` is a module-level registry of named
regression-equation presets. Each entry packages together the measured-side
regression columns, the modeled-side regression columns, the regression formula,
the scatter-plot callable that matches the formula, and the default reporting-
conditions kwargs. Three presets ship with pvcaptest:

- ``e2848_default`` — the default ASTM E2848 regression
  ``power ~ poa + I(poa*poa) + I(poa*t_amb) + I(poa*w_vel) - 1``. Uses
  :py:func:`~captest.captest.scatter_default` for plots.
- ``bifi_e2848_etotal`` — the same ASTM formula but the ``poa`` term is the
  calculated ``e_total`` column (front POA plus rear POA times bifaciality).
  Uses :py:func:`~captest.captest.scatter_etotal`. Requires the
  :py:attr:`~captest.captest.CapTest.bifaciality` attribute to be set before
  ``setup()``.
- ``bifi_power_tc`` — a temperature-corrected bifacial regression
  ``power ~ poa + rpoa``, where ``power`` is the calculated
  ``power_temp_correct`` column. Uses
  :py:func:`~captest.captest.scatter_bifi_power_tc`, which returns a
  two-panel layout (one panel per right-hand-side term).
- ``e2848_spec_corrected_poa`` — the default ASTM formula, but the ``poa``
  term is a First Solar spectral-correction-adjusted POA
  (``poa_spec_corrected = poa * spectral_factor_firstsolar``). The meas-side
  tree computes precipitable water from measured humidity and ambient
  temperature; the sim-side tree reads PVsyst ``PrecWat`` and converts it from
  meters to centimeters via :py:func:`~captest.calcparams.scale`, and computes
  the apparent zenith using
  :py:func:`~captest.calcparams.apparent_zenith_pvsyst`, which handles
  PVsyst's half-hour timestamp shift internally. Requires ``humidity`` and
  ``pressure`` column groups on the measured :py:class:`~captest.capdata.CapData`,
  a ``PrecWat`` column on the modeled :py:class:`~captest.capdata.CapData`,
  and ``cd.site`` on the measured instance (see :ref:`spec_corrected_poa`).

.. note::

    The lhs key of the regression formula is always ``"power"`` across
    shipped presets, even when the formula regresses a derived quantity like
    temperature-corrected power. Code that hardcodes ``"power"`` as the lhs
    key keeps working with all shipped presets.

Additional presets can be registered by assigning into
:py:data:`~captest.captest.TEST_SETUPS` at import time. Each entry is validated
against :py:func:`~captest.captest.validate_test_setup` when
:py:meth:`~captest.captest.CapTest.setup` resolves the preset.

Construction
------------
A :py:class:`~captest.captest.CapTest` can be constructed three ways.

From pre-built CapData instances
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:py:meth:`~captest.captest.CapTest.from_params` is the most direct constructor.
When both ``meas`` and ``sim`` are supplied it calls
:py:meth:`~captest.captest.CapTest.setup` automatically:

.. code-block:: Python

    from captest import CapTest, load_data, load_pvsyst

    meas = load_data(path='./data/measured/')
    sim  = load_pvsyst(path='./data/pvsyst_results.csv')

    ct = CapTest.from_params(
        test_setup='e2848_default',
        meas=meas,
        sim=sim,
        ac_nameplate=6_000_000,
        test_tolerance='- 4',
    )

From data paths
~~~~~~~~~~~~~~~
If ``meas`` / ``sim`` are not pre-built,
:py:meth:`~captest.captest.CapTest.from_params` accepts ``meas_path`` /
``sim_path`` and calls :py:func:`~captest.io.load_data` /
:py:func:`~captest.io.load_pvsyst` internally. A custom loader can be injected
by passing ``meas_loader`` / ``sim_loader`` (and any extra kwargs via
``meas_load_kwargs`` / ``sim_load_kwargs``):

.. code-block:: Python

    ct = CapTest.from_params(
        test_setup='bifi_e2848_etotal',
        meas_path='./data/measured/',
        sim_path='./data/pvsyst_results.csv',
        bifaciality=0.15,
        ac_nameplate=6_000_000,
    )

From yaml
~~~~~~~~~
:py:meth:`~captest.captest.CapTest.from_yaml` reads the sub-mapping at a
top-level ``key`` (default ``"captest"``) of a yaml file; relative
``meas_path`` and ``sim_path`` values are resolved against the yaml file's
directory.

.. code-block:: yaml

    # project.yaml
    captest:
      test_setup: bifi_e2848_etotal
      meas_path: ./data/meas/
      sim_path:  ./data/pvsyst.csv
      ac_nameplate: 6_000_000
      test_tolerance: "- 4"
      min_irr: 400
      max_irr: 1400
      fshdbm: 1.0
      bifaciality: 0.15

.. code-block:: Python

    ct = CapTest.from_yaml('./project.yaml')

The yaml ``key`` argument lets one file hold multiple captest sections — for
example ``captest_e2848`` and ``captest_bifi`` — so the same project yaml can
drive multiple flavors of the capacity test against the same data:

.. code-block:: Python

    ct_default = CapTest.from_yaml('./project.yaml', key='captest_e2848')
    ct_bifi    = CapTest.from_yaml('./project.yaml', key='captest_bifi')

Bare + manual setup
~~~~~~~~~~~~~~~~~~~
If you want to assemble the instance yourself, construct
:py:class:`~captest.captest.CapTest` with just the preset and scalars, assign
``meas`` / ``sim`` directly, and call
:py:meth:`~captest.captest.CapTest.setup` when ready:

.. code-block:: Python

    ct = CapTest(test_setup='bifi_power_tc', bifaciality=0.15)
    ct.meas = my_meas
    ct.sim  = my_sim
    ct.setup()

What ``setup()`` does
---------------------
:py:meth:`~captest.captest.CapTest.setup` resolves the active preset (with any
user overrides), propagates the scalar calc-params
(:py:attr:`~captest.captest.CapTest.bifaciality`,
:py:attr:`~captest.captest.CapTest.power_temp_coeff`,
:py:attr:`~captest.captest.CapTest.base_temp`) onto both
:py:class:`~captest.capdata.CapData` instances, wires the preset's
:py:attr:`~captest.capdata.CapData.regression_cols` and
:py:attr:`~captest.capdata.CapData.regression_formula` onto both sides, and
runs :py:meth:`~captest.capdata.CapData.process_regression_columns` on both to
materialize aggregated and calculated columns (e.g. the ``e_total`` column for
``bifi_e2848_etotal`` or the ``power_temp_correct`` column for
``bifi_power_tc``).

The method is re-runnable. Calling ``setup()`` a second time resets
``data_filtered`` back to ``data.copy()`` on both instances, which is
intentional — it lets users iterate on configuration without carrying stale
filter state across runs.

Running a capacity test
-----------------------
After ``setup()``, drive the filter sequence, reporting conditions, and fit
directly on the :py:class:`~captest.capdata.CapData` instances. The attributes
held on :py:class:`~captest.captest.CapTest`
(:py:attr:`~captest.captest.CapTest.min_irr`,
:py:attr:`~captest.captest.CapTest.max_irr`,
:py:attr:`~captest.captest.CapTest.fshdbm`, etc.) can be forwarded directly:

.. code-block:: Python

    ct.meas.filter_irr(ct.min_irr, ct.max_irr)
    ct.sim.filter_irr(ct.min_irr, ct.max_irr)
    ct.sim.filter_shade(fshdbm=ct.fshdbm)
    ct.sim.filter_time(start='2026-03-26', end='2026-04-12')

    ct.rep_cond()                # reporting conditions on ct.meas.rc
    ct.rep_cond(which='sim')     # reporting conditions on ct.sim.rc

    ct.meas.fit_regression()
    ct.sim.fit_regression()

    cap_ratio = ct.captest_results()

After computing reporting conditions, a second narrow irradiance filter around
the reporting irradiance is common. The read-only properties
:py:attr:`~captest.captest.CapTest.rep_irr_filter_low` and
:py:attr:`~captest.captest.CapTest.rep_irr_filter_high` provide the fractional
bounds derived from :py:attr:`~captest.captest.CapTest.rep_irr_filter`
(``1 - rep_irr_filter`` and ``1 + rep_irr_filter`` respectively) for direct
use with :py:meth:`~captest.capdata.CapData.filter_irr`:

.. code-block:: Python

    ct.meas.filter_irr(
        ct.rep_irr_filter_low,
        ct.rep_irr_filter_high,
        ref_val='self_val',   # uses ct.meas.rc['poa']
    )
    ct.sim.filter_irr(
        ct.rep_irr_filter_low,
        ct.rep_irr_filter_high,
        ref_val='self_val',   # uses ct.sim.rc['poa']
    )

:py:meth:`~captest.captest.CapTest.rep_cond` is a convenience method that
calls :py:meth:`~captest.capdata.CapData.rep_cond` using the active preset's
``rep_conditions`` dict as default kwargs. Keyword overrides are partial-merged
on top: top-level keys replace, and the nested ``func`` dict is merged one
level deep so a contract-specific percentile can be plugged in for a single
variable without disturbing the others:

.. code-block:: Python

    from captest.captest import perc_wrap

    # 55th-percentile POA for a project under contract; t_amb and w_vel
    # still fall back to the preset's 'mean' aggregations.
    ct.rep_cond(func={'poa': perc_wrap(55)})

Results and comparison
----------------------
The cross-CapData helpers previously at module level are now methods on
:py:class:`~captest.captest.CapTest`. All of them use
:py:attr:`~captest.captest.CapTest.meas`,
:py:attr:`~captest.captest.CapTest.sim`,
:py:attr:`~captest.captest.CapTest.rep_cond_source`,
:py:attr:`~captest.captest.CapTest.ac_nameplate`, and
:py:attr:`~captest.captest.CapTest.test_tolerance` directly — no more passing
CapData pairs around by hand.

- :py:meth:`~captest.captest.CapTest.captest_results` — predicts measured and
  modeled capacities at the reporting conditions (picked from ``meas.rc`` or
  ``sim.rc`` per :py:attr:`~captest.captest.CapTest.rep_cond_source`), returns
  the capacity ratio, and optionally prints a pass / fail summary.
- :py:meth:`~captest.captest.CapTest.captest_results_check_pvalues` — runs
  :py:meth:`~captest.captest.CapTest.captest_results` twice (with and without
  p-value zeroing) and returns a Styler highlighting any coefficient whose
  p-value is above 0.05.
- :py:meth:`~captest.captest.CapTest.overlay_scatters` — builds the preset's
  scatter for both :py:class:`~captest.capdata.CapData` instances and overlays
  them with labels. The scatter callable is picked automatically from the
  resolved preset.
- :py:meth:`~captest.captest.CapTest.residual_plot` — overlays regression
  residuals vs. each exogenous variable for both instances. Replaces the
  former ``plotting.residual_plot``.
- :py:meth:`~captest.captest.CapTest.get_summary` — concatenates
  ``self.meas.get_summary()`` and ``self.sim.get_summary()`` so filter history
  for both sides is visible in one frame.
- :py:meth:`~captest.captest.CapTest.determine_pass_or_fail` — returns a
  ``(bool, bounds)`` tuple for a given capacity ratio using the instance's
  tolerance and nameplate.

:py:meth:`~captest.captest.CapTest.scatter_plots` dispatches to the preset's
scatter callable on either side:

.. code-block:: Python

    ct.scatter_plots()              # measured scatter (hv.Layout)
    ct.scatter_plots(which='sim')   # modeled scatter

Overrides and the ``"custom"`` setup
------------------------------------
Every preset-level value can be overridden per instance without redefining the
preset. The two most common overrides are ``reg_fml`` (regression formula) and
``rep_conditions``. In a yaml file, overrides go under an ``overrides:``
sub-mapping:

.. code-block:: yaml

    captest:
      test_setup: e2848_default
      overrides:
        rep_conditions:
          percent_filter: 10        # replaces preset percent_filter
          func:
            poa: perc_55            # resolved to perc_wrap(55) at load time
                                    # t_amb and w_vel preserved from the preset

The ``"perc_N"`` string shorthand in yaml is resolved to the equivalent
:py:func:`~captest.captest.perc_wrap` callable when the file is loaded, and
written back as a ``"perc_N"`` string when
:py:meth:`~captest.captest.CapTest.to_yaml` serializes the instance. Any
string not matching the ``perc_<int>`` pattern (e.g. ``"mean"``) passes through
unchanged as a pandas aggregation name.

When ``test_setup == "custom"``, three overrides are required:
``reg_cols_meas``, ``reg_cols_sim``, and ``reg_fml``. The scatter callable
falls back to :py:func:`~captest.captest.scatter_default` unless supplied
explicitly, and ``rep_conditions`` defaults to ``{}`` (which lets
:py:meth:`~captest.capdata.CapData.rep_cond` use its own ``func=None`` fallback
of ``{var: 'mean' for var in rhs}``).

Serialization round-trip
------------------------
:py:meth:`~captest.captest.CapTest.to_yaml` writes a curated subset of the
instance's state back to a yaml file: every scalar ``param.*`` attribute,
``test_setup``, non-default overrides for ``reg_fml`` /
``reg_cols_meas`` / ``reg_cols_sim`` / ``rep_conditions``, ``meas_path`` /
``sim_path`` (when the instance was constructed from paths), and non-empty
``meas_load_kwargs`` / ``sim_load_kwargs``. Data, fitted regression results,
the resolved preset, and loader callables are never written.

By default ``to_yaml`` merges into an existing file on disk: other top-level
keys (e.g. ``client``, ``loc``, ``system``) are preserved and only the sub-tree
at ``key`` is overwritten. Pass ``merge_into_existing=False`` to replace the
file unconditionally.

.. note::

    Programmatic-only attributes — ``meas_loader``, ``sim_loader``, and any
    user-mutated ``scatter_plots`` callable on the resolved preset — cannot
    round-trip through yaml. ``to_yaml`` emits a single
    :py:class:`UserWarning` listing the attributes that were skipped and
    writes the rest of the configuration as normal.

.. _spec_corrected_poa:

Spectrally corrected POA (``e2848_spec_corrected_poa``)
-------------------------------------------------------
The ``e2848_spec_corrected_poa`` preset multiplies the regression-driving POA
irradiance by a First Solar spectral-correction factor computed from pvlib
per the `McCarthy 2024 PVPMC poster
<https://pvpmc.sandia.gov/download/7822/?tmstv=1776198191>`_ and the
`pvlib.spectrum.spectral_factor_firstsolar
<https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.spectrum.spectral_factor_firstsolar.html>`_
reference.

Inputs required on the measured :py:class:`~captest.capdata.CapData`:

- A ``humidity`` column group (relative humidity in percent).
- A ``pressure`` column group (station pressure in hPa / mbar).
- A ``cd.site`` attribute with ``{'loc': {...}, 'sys': {...}}`` sub-dicts.
  The :py:func:`~captest.io.load_data` function populates this when called
  with the ``site`` kwarg.

Inputs required on the modeled :py:class:`~captest.capdata.CapData`:

- A ``PrecWat`` column in the PVsyst output (configure the PVsyst export to
  include precipitable water).

At :py:meth:`~captest.captest.CapTest.setup` time the
:py:class:`~captest.captest.CapTest` class auto-propagates ``meas.site`` onto
``sim.site`` and converts the tz to the nearest fixed-offset ``Etc/GMT±N``
string (PVsyst timestamps are not DST-aware). A
:py:class:`UserWarning` describes the conversion. To use a different tz or
site dict for the sim side, assign ``ct.sim.site = {...}`` before calling
``setup()``.

The module type passed to
:py:func:`~captest.calcparams.spectral_factor_firstsolar` is controlled by the
:py:attr:`~captest.captest.CapTest.spectral_module_type` parameter (default
``'cdte'``). It is named ``spectral_module_type`` — not ``module_type`` —
to avoid collision with the ``module_type`` kwarg of
:py:func:`~captest.calcparams.bom_temp` and
:py:func:`~captest.calcparams.cell_temp`.

.. code-block:: Python

    from captest import CapTest, load_data, load_pvsyst

    site = {
        'loc': {'latitude': 33.01, 'longitude': -99.56,
                'altitude': 500, 'tz': 'America/Chicago'},
        'sys': {'surface_tilt': 0, 'surface_azimuth': 180, 'albedo': 0.2},
    }
    meas = load_data(path='./data/measured/', site=site)
    sim  = load_pvsyst(path='./data/pvsyst_results.csv')

    ct = CapTest.from_params(
        test_setup='e2848_spec_corrected_poa',
        meas=meas,
        sim=sim,
        ac_nameplate=6_000_000,
        test_tolerance='- 4',
        spectral_module_type='cdte',   # default; override for non-CdTe plants
    )

.. note::

    The spectrally corrected POA column is named ``poa_spec_corrected`` and
    is added to both ``ct.meas.data`` and ``ct.sim.data`` by
    :py:meth:`~captest.capdata.CapData.process_regression_columns`. The
    regression then uses this column in place of raw POA irradiance.
