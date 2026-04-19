"""Unified test orchestrator and supporting utilities.

This module houses the ``CapTest`` class (added in a later unit), the
``TEST_SETUPS`` registry of named regression presets, and small formatting
helpers used by ``CapTest`` methods that compare a measured + modeled pair of
``CapData`` instances.

The module is intentionally light on imports at module scope to avoid any
circular dependency with ``captest.capdata``. Any function that needs a
``CapData`` instance accepts it as an argument rather than importing the class.
"""


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
