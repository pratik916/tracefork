"""Monte Carlo coverage-calibration harness for `blame.py`'s proportion CIs.

Nobody has ever proven that a `blame.py` flip-rate confidence interval
nominally labeled 95% actually *achieves* ~95% coverage. This module closes
that gap using the canonical calibration methodology of Brown, Cai &
DasGupta, "Interval Estimation for a Binomial Proportion" (Statistical
Science, 2001): fix a KNOWN ground-truth success probability ``true_p``,
draw many independent Bernoulli(``n_trials``) replicates, compute the
candidate CI for each replicate, and measure the empirical fraction of
replicates whose interval actually contains ``true_p`` ("empirical
coverage"). Their central finding — that Wald intervals have erratic,
often badly undercovering behavior while Wilson/Jeffreys track nominal
coverage closely for small-to-moderate ``n`` and Clopper-Pearson is
conservative (coverage >= nominal) by construction — is exactly what the
qualitative assertions in this module's test suite check for, using
`blame.py`'s REAL `proportion_ci`/`wilson_ci`/`CIMethod` backend (never a
parallel reimplementation of the interval math itself).

Because the reported coverage is itself an estimate (a sample proportion
over ``n_repeats`` replicates), it carries its own Monte Carlo sampling
error of ``sqrt(nominal * (1 - nominal) / n_repeats)`` (`monte_carlo_error`)
— that quantity sets both the tolerance band a coverage value is judged
against and the minimum replicate count worth trusting; this is why the
default `DEFAULT_N_REPEATS` is 2000 (MC error ≈0.49pp at the 95% level).

Pure math over `random.Random(seed)` — no fork/agent re-execution, no
tape, no network, no API key. Deterministic given a seed. Mirrors
`bench.py`'s dataclass-report shape (a `list[CoverageResult]` wrapped in a
report with a `.regressions()`-style query) for console/JSON printability.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field

from .blame import CIMethod, proportion_ci

#: Replicate count whose Monte Carlo error at the default 95% confidence
#: level is ~0.0049 (0.49 percentage points) — small enough that a coverage
#: value outside the documented tolerance band is a genuine signal, not
#: sampling noise. See Brown-Cai-DasGupta (2001) and the module docstring.
DEFAULT_N_REPEATS = 2000

#: Nominal confidence level calibrated against, matching `blame.py`'s default.
DEFAULT_CONFIDENCE = 0.95

#: Tolerance multiplier applied to the Monte Carlo error of the coverage
#: estimate itself: a coverage value more than this many MC-error units away
#: from nominal is flagged as a regression rather than noise. 4 sigma keeps
#: the false-alarm rate at a fixed seed effectively zero while still catching
#: a genuinely miscalibrated backend (which misses nominal by many, many
#: multiples of the Monte Carlo error, not a handful of sigma).
TOLERANCE_SIGMA = 4.0

#: Default grid of known ground-truth flip probabilities to calibrate against,
#: spanning the 0/1 boundary blame trials cluster at plus interior values.
DEFAULT_TRUE_PS: tuple[float, ...] = (0.0, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0)

#: Default grid of trial counts, matching the small-n regime blame's
#: per-step fork budgets actually run at (`BudgetGovernor`).
DEFAULT_N_TRIALS_GRID: tuple[int, ...] = (5, 10, 20, 50)

#: Every CI backend `blame.proportion_ci` supports, calibrated by default.
DEFAULT_METHODS: tuple[CIMethod, ...] = (
    CIMethod.WILSON,
    CIMethod.JEFFREYS,
    CIMethod.CLOPPER_PEARSON,
    CIMethod.AGRESTI_COULL,
)


def monte_carlo_error(confidence: float, n_repeats: int) -> float:
    """Standard error of the empirical coverage estimate itself.

    A measured coverage is a sample proportion over ``n_repeats`` independent
    replicates, each either "covered" or "not covered" with true probability
    ``confidence`` if the CI backend is exactly calibrated. Its own standard
    error is therefore the binomial-proportion standard error evaluated at
    ``confidence`` — this is what sets the minimum trustworthy replicate
    count and the tolerance band a coverage value is judged against.
    """
    if n_repeats <= 0:
        raise ValueError("n_repeats must be positive")
    return math.sqrt(confidence * (1.0 - confidence) / n_repeats)


@dataclass(frozen=True)
class CoverageResult:
    """Empirical coverage of one (method, true_p, n_trials) calibration cell."""

    method: CIMethod
    true_p: float
    n_trials: int
    n_repeats: int
    confidence: float
    coverage: float
    tolerance: float

    @property
    def within_tolerance(self) -> bool:
        """Whether ``coverage`` lands within ``tolerance`` of ``confidence``.

        Only flags UNDER-coverage as a failure: a CI backend that covers
        *more* than nominal (Clopper-Pearson's documented conservatism, or
        the trivial always-covers case at the ``true_p`` in {0, 1} boundary)
        is honest, not miscalibrated, and must never fail this check.
        """
        return self.coverage >= self.confidence - self.tolerance


@dataclass(frozen=True)
class CalibrationReport:
    """A grid of `CoverageResult`s over (method, true_p, n_trials), mirroring
    `bench.py`'s `BenchReport` dataclass-report shape for console/JSON
    printability."""

    results: list[CoverageResult] = field(default_factory=list)
    seed: int = 0

    def regressions(self) -> list[CoverageResult]:
        """Cells whose empirical coverage fell outside its tolerance band --
        an empty list is the expected, healthy state."""
        return [r for r in self.results if not r.within_tolerance]

    def all_within_tolerance(self) -> bool:
        """True iff every calibration cell is within its tolerance band."""
        return not self.regressions()


def simulate_coverage(
    true_p: float,
    n_trials: int,
    *,
    method: CIMethod = CIMethod.WILSON,
    confidence: float = DEFAULT_CONFIDENCE,
    n_repeats: int = DEFAULT_N_REPEATS,
    seed: int = 0,
) -> CoverageResult:
    """Empirical coverage of ``method`` at a KNOWN ``true_p`` over ``n_trials``
    Bernoulli draws, replicated ``n_repeats`` times.

    For each replicate, draws ``n_trials`` independent Bernoulli(``true_p``)
    outcomes from `random.Random(seed)`, counts successes, computes the
    candidate CI via `blame.py`'s real `proportion_ci` (never a parallel
    reimplementation), and checks whether ``true_p`` falls inside it.
    ``coverage`` is the fraction of replicates where it did. Deterministic
    given ``seed`` — the SAME seed always drives the SAME sequence of draws.
    """
    if not 0.0 <= true_p <= 1.0:
        raise ValueError("true_p must be in [0, 1]")
    if n_trials <= 0:
        raise ValueError("n_trials must be positive")

    rng = random.Random(seed)
    covered = 0
    for _ in range(n_repeats):
        successes = sum(1 for _ in range(n_trials) if rng.random() < true_p)
        lo, hi = proportion_ci(successes, n_trials, method=method, confidence=confidence)
        if lo <= true_p <= hi:
            covered += 1

    coverage = covered / n_repeats
    tolerance = TOLERANCE_SIGMA * monte_carlo_error(confidence, n_repeats)
    return CoverageResult(
        method=method,
        true_p=true_p,
        n_trials=n_trials,
        n_repeats=n_repeats,
        confidence=confidence,
        coverage=coverage,
        tolerance=tolerance,
    )


def run_calibration(
    *,
    true_ps: Sequence[float] = DEFAULT_TRUE_PS,
    n_trials_grid: Sequence[int] = DEFAULT_N_TRIALS_GRID,
    methods: Sequence[CIMethod] = DEFAULT_METHODS,
    confidence: float = DEFAULT_CONFIDENCE,
    n_repeats: int = DEFAULT_N_REPEATS,
    seed: int = 0,
) -> CalibrationReport:
    """Calibrate every ``method`` over the full grid of ``true_ps`` x
    ``n_trials_grid``, each cell reusing the SAME ``seed`` for the same
    ``(true_p, n_trials)`` pair across methods so their coverage figures are
    directly comparable (same simulated draws, different CI backend applied
    to the same success counts).
    """
    results: list[CoverageResult] = []
    for n_trials in n_trials_grid:
        for true_p in true_ps:
            for method in methods:
                results.append(
                    simulate_coverage(
                        true_p=true_p,
                        n_trials=n_trials,
                        method=method,
                        confidence=confidence,
                        n_repeats=n_repeats,
                        seed=seed,
                    )
                )
    return CalibrationReport(results=results, seed=seed)
