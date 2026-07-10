"""`tracefork.ci_calibration` tests — pure Monte Carlo math, offline/$0,
no fork/agent/API calls anywhere in this file."""

from dataclasses import replace

from tracefork.blame import CIMethod
from tracefork.ci_calibration import (
    DEFAULT_CONFIDENCE,
    CalibrationReport,
    CoverageResult,
    monte_carlo_error,
    run_calibration,
    simulate_coverage,
)


def test_wilson_coverage_at_p_half_is_within_tolerance_of_nominal():
    """Brown-Cai-DasGupta-style calibration check: fix a KNOWN true_p, draw
    many replicate Bernoulli(n_trials) samples, and confirm Wilson's empirical
    coverage lands within its documented tolerance band of the nominal 95%."""
    result = simulate_coverage(
        true_p=0.5, n_trials=20, method=CIMethod.WILSON, n_repeats=2000, seed=1234
    )
    assert result.within_tolerance, (
        f"coverage {result.coverage} not within {result.tolerance} of {result.confidence}"
    )
    assert 0.0 <= result.coverage <= 1.0


def test_coverage_deterministic_given_seed():
    """Same seed -> byte-identical CoverageResult across two independent runs."""
    a = simulate_coverage(true_p=0.5, n_trials=20, n_repeats=2000, seed=42)
    b = simulate_coverage(true_p=0.5, n_trials=20, n_repeats=2000, seed=42)
    assert a == b


def test_calibration_report_deterministic_given_seed():
    """Same seed -> byte-identical CalibrationReport across two independent runs."""
    a = run_calibration(true_ps=(0.1, 0.5, 0.9), n_trials_grid=(10, 20), n_repeats=500, seed=7)
    b = run_calibration(true_ps=(0.1, 0.5, 0.9), n_trials_grid=(10, 20), n_repeats=500, seed=7)
    assert a == b


def test_boundary_true_p_zero_and_one_stay_sane():
    """true_p=0.0 and true_p=1.0 -> coverage stays sane (never regresses below
    the nominal confidence) given wilson_ci's documented boundary-snapping
    (successes==0 -> lo pinned to 0.0; successes==n -> hi pinned to 1.0), which
    makes the true boundary probability always fall inside the interval."""
    for true_p in (0.0, 1.0):
        result = simulate_coverage(
            true_p=true_p, n_trials=20, method=CIMethod.WILSON, n_repeats=500, seed=99
        )
        assert result.coverage == 1.0
        assert result.within_tolerance


def test_clopper_pearson_is_conservative_relative_to_wilson_at_small_n():
    """Clopper-Pearson is conservative BY CONSTRUCTION (Brown-Cai-DasGupta 2001):
    its empirical coverage should meet-or-exceed nominal, while Wilson may dip
    slightly below nominal at small n -- the qualitative ordering the module's
    (and blame.py's) docstrings already claim, not a bug in either backend."""
    n_trials, true_p, seed = 10, 0.3, 2024
    cp = simulate_coverage(
        true_p=true_p, n_trials=n_trials, method=CIMethod.CLOPPER_PEARSON, n_repeats=4000, seed=seed
    )
    wilson = simulate_coverage(
        true_p=true_p, n_trials=n_trials, method=CIMethod.WILSON, n_repeats=4000, seed=seed
    )
    assert cp.coverage >= wilson.coverage
    assert cp.coverage >= cp.confidence - monte_carlo_error(cp.confidence, cp.n_repeats)


def test_monte_carlo_error_shrinks_with_more_repeats():
    small = monte_carlo_error(DEFAULT_CONFIDENCE, 100)
    large = monte_carlo_error(DEFAULT_CONFIDENCE, 10_000)
    assert large < small


def test_run_calibration_covers_full_grid():
    report = run_calibration(
        true_ps=(0.0, 0.5, 1.0),
        n_trials_grid=(10, 20),
        methods=(CIMethod.WILSON, CIMethod.CLOPPER_PEARSON),
        n_repeats=200,
        seed=3,
    )
    assert isinstance(report, CalibrationReport)
    assert len(report.results) == 3 * 2 * 2
    assert all(isinstance(r, CoverageResult) for r in report.results)


def test_calibration_report_regressions_lists_only_out_of_tolerance_results():
    report = run_calibration(
        true_ps=(0.5,), n_trials_grid=(20,), methods=(CIMethod.WILSON,), n_repeats=2000, seed=1234
    )
    assert report.regressions() == []
    # Force a synthetic regression to prove the filter is real, not vacuous.
    bad = replace(report.results[0], coverage=0.0)
    forced = replace(report, results=[bad])
    assert forced.regressions() == [bad]
