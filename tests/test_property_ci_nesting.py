"""Property-based (Hypothesis) proofs that `blame.py`'s "exact" Clopper-Pearson
proportion CI genuinely nests the other three backends `proportion_ci`
supports (`CIMethod`) — not just on the handful of fixed cases `test_blame.py`
pins, but over arbitrary generated `(successes, n, confidence)` inputs.

The bead this module answers asked for one property: "Clopper-Pearson always
contains Wilson/Jeffreys/Agresti-Coull." That literal claim is **false** for
this codebase's hand-rolled math — verified by EXHAUSTIVE grid search (every
`n` from 1..300, every `successes` in `[0, n]`, confidence in
`{0.80, 0.90, 0.95, 0.99}`) before this module was written, not assumed:

* Clopper-Pearson contains Jeffreys **unconditionally** — 0 counterexamples
  over the full grid, at all four confidence levels. Both are Beta-quantile
  "exact" intervals built on the same `_beta_ppf` primitive
  (`a=successes[+.5], b=n-successes[+.5]` vs. `a=successes, b=n-successes+1`
  shifted by one success), so this generalizes safely.
* Clopper-Pearson does **not** always contain Wilson or Agresti-Coull.
  Wilson breaches CP at extreme boundary counts (`successes` in `{0, n}`)
  once `n` gets large enough — first observed at `n=46, successes=0,
  confidence=0.95`. Agresti-Coull breaches even at `blame.py`'s own default
  `n=10` trial count, at boundary `successes` in `{0, 1, 9, 10}` and
  `confidence=0.95` — exactly the common all-flip/no-flip case the real
  fault-injection suite hits.
* Restricting to a **moderate-proportion band** (`successes / n` in
  `[0.15, 0.85]`) at confidence in `{0.80, 0.90, 0.95}` (0.95 is
  `proportion_ci`'s own default; **0.99 is deliberately excluded** — it is
  the one level that still breaks even inside the band, verified via
  thousands of in-band counterexamples at that level alone) eliminates every
  counterexample for both Wilson and Agresti-Coull, checked exhaustively for
  every `n` from 2..300.

So this module ships the one property that IS universally true
(`test_clopper_pearson_contains_jeffreys_unconditionally`) plus the property
that IS true in the realistic regime this codebase actually operates in
(`test_clopper_pearson_contains_wilson_and_agresti_coull_in_moderate_regime`),
each scoped honestly rather than asserting the broader, false claim. A future
reader who is tempted to widen the second test's band or add `confidence=0.99`
back in should re-run the grid search first — it will fail, by design, on
exactly the boundary cases cited above.

Deterministic and offline: `derandomize=True` derives every example from a
fixed seed (no network, no on-disk example database — see `test_property_
tape.py`'s identical comment), and `max_examples` is bounded so this stays
well within CI's time budget. Pure in-process math (no scipy), $0.
"""

from __future__ import annotations

import math

from hypothesis import given, settings
from hypothesis import strategies as st

from tracefork.blame import CIMethod, proportion_ci

# See `test_property_tape.py` for why `derandomize=True` + `deadline=None` is
# the right, reproducible-without-a-committed-cache choice here too.
_SETTINGS = settings(max_examples=100, derandomize=True, deadline=None)

# Float-dust tolerance: both the Beta-quantile bisection (`_beta_ppf`, 128
# iterations) and the closed-form Wilson/Agresti-Coull algebra are exact only
# up to floating-point rounding, not bit-for-bit.
_TOL = 1e-9

_CONFIDENCE_ALL_LEVELS = st.sampled_from([0.80, 0.90, 0.95, 0.99])
_CONFIDENCE_MODERATE_LEVELS = st.sampled_from([0.80, 0.90, 0.95])


@st.composite
def _any_proportion_case(draw: st.DrawFn) -> tuple[int, int, float]:
    """An arbitrary `(n, successes, confidence)` triple spanning the FULL
    range `proportion_ci` accepts, including the `successes in {0, n}`
    boundary counts and all four confidence levels."""
    n = draw(st.integers(min_value=1, max_value=200))
    successes = draw(st.integers(min_value=0, max_value=n))
    confidence = draw(_CONFIDENCE_ALL_LEVELS)
    return n, successes, confidence


@st.composite
def _moderate_proportion_case(draw: st.DrawFn) -> tuple[int, int, float]:
    """An `(n, successes, confidence)` triple restricted to the
    moderate-proportion band (`successes / n` in `[0.15, 0.85]`) at the
    confidence levels this codebase's real callers use — see the module
    docstring for why both restrictions are load-bearing, not arbitrary."""
    n = draw(st.integers(min_value=2, max_value=300))
    lo_k = math.ceil(0.15 * n)
    hi_k = math.floor(0.85 * n)
    successes = draw(st.integers(min_value=lo_k, max_value=hi_k))
    confidence = draw(_CONFIDENCE_MODERATE_LEVELS)
    return n, successes, confidence


@_SETTINGS
@given(_any_proportion_case())
def test_clopper_pearson_contains_jeffreys_unconditionally(
    case: tuple[int, int, float],
) -> None:
    """Clopper-Pearson's interval contains Jeffreys' for ANY `(successes, n,
    confidence)` — both are Beta-quantile exact intervals sharing
    `_beta_ppf`, so no proportion band or confidence restriction is needed."""
    n, successes, confidence = case
    cp_lo, cp_hi = proportion_ci(
        successes, n, method=CIMethod.CLOPPER_PEARSON, confidence=confidence
    )
    j_lo, j_hi = proportion_ci(successes, n, method=CIMethod.JEFFREYS, confidence=confidence)
    assert cp_lo <= j_lo + _TOL, (n, successes, confidence, cp_lo, j_lo)
    assert j_hi <= cp_hi + _TOL, (n, successes, confidence, j_hi, cp_hi)


@_SETTINGS
@given(_moderate_proportion_case())
def test_clopper_pearson_contains_wilson_and_agresti_coull_in_moderate_regime(
    case: tuple[int, int, float],
) -> None:
    """Clopper-Pearson's interval contains both Wilson's and Agresti-Coull's
    intervals, restricted to `successes / n` in `[0.15, 0.85]` at confidence
    in `{0.80, 0.90, 0.95}` — the regime this restriction is scoped to.
    Outside it (e.g. `n=10, successes=0, confidence=0.95` for Agresti-Coull,
    or any `n, successes in {0, n}` at `confidence=0.99`) the nesting
    genuinely fails; see the module docstring."""
    n, successes, confidence = case
    cp_lo, cp_hi = proportion_ci(
        successes, n, method=CIMethod.CLOPPER_PEARSON, confidence=confidence
    )
    w_lo, w_hi = proportion_ci(successes, n, method=CIMethod.WILSON, confidence=confidence)
    ac_lo, ac_hi = proportion_ci(successes, n, method=CIMethod.AGRESTI_COULL, confidence=confidence)
    assert cp_lo <= w_lo + _TOL, (n, successes, confidence, cp_lo, w_lo)
    assert w_hi <= cp_hi + _TOL, (n, successes, confidence, w_hi, cp_hi)
    assert cp_lo <= ac_lo + _TOL, (n, successes, confidence, cp_lo, ac_lo)
    assert ac_hi <= cp_hi + _TOL, (n, successes, confidence, ac_hi, cp_hi)
