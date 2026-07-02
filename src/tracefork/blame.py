"""Blame engine: rank each exchange by causal flip-rate with confidence intervals.

The causal question is "if step *i* had gone differently, how often would the
run's *outcome* change?" Answering it honestly requires re-running the agent,
not just rewriting the tape: perturbing step *i*'s response changes every
request the agent makes afterward. So for each candidate step we:

  1. fork the recorded run at *i* with a perturbed response — the prefix is
     replayed from the parent tape for $0 and the agent is re-run from there
     (`ForkEngine.fork`);
  2. grade the resulting outcome with an `Oracle`;
  3. classify the trial as **FLIP** (graded outcome differs from the parent
     run's), **NO_FLIP** (same, determinate outcome), or **UNDEFINED** (the fork
     diverged, errored, or produced an ungradeable outcome).

`flip_rate = flips / valid_trials` is computed over *valid* trials only — a fork
that diverged or errored is **not** a silent non-flip, because divergence
probability grows with tape depth and would otherwise under-blame deep steps.
Each step surfaces its per-step divergence/UNDEFINED rate as a trust flag.

Confidence intervals for the flip-rate proportion are pluggable
(`proportion_ci`): Wilson (default), Jeffreys, Clopper-Pearson, and
Agresti-Coull, all with correct 0-flip / all-flip boundary handling and a
configurable confidence level. Instead of a raw argmax over noisy proportions,
the "responsible set" is chosen by a Benjamini-Hochberg FDR-controlled,
one-sided binomial test of each step's flip-rate against a chance-flip null.

`BudgetGovernor` estimates the fork count and dollar cost before any spend.

The engine is agent- and domain-agnostic: the caller supplies `agent_fn` (the
same agent that produced the tape) and a `perturb_factory(step) -> (response,
tail_transport)`. In tests and the offline validation suite, `tail_transport`
is a scripted fake (zero cost); for a live run it is `None`, so the
counterfactual tail hits the real API under the budget cap.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from statistics import NormalDist
from typing import Protocol, cast

import httpx

from . import pricing
from .constants import SONNET
from .fork import BranchSpec, CoalitionSpec, ForkEngine, StepIntervention
from .nondet import find_divergence
from .observability import instrument
from .plugins import ORACLE_GROUP, Registry
from .providers import get_adapter
from .tape import Tape

# ── Statistical primitives (pure-python, no scipy) ──────────────────────────


def z_from_confidence(confidence: float) -> float:
    """Two-sided critical z for a confidence level, e.g. 0.95 → ≈1.95996."""
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    return NormalDist().inv_cdf(1.0 - (1.0 - confidence) / 2.0)


def _betacf(a: float, b: float, x: float) -> float:
    """Continued-fraction expansion for the incomplete beta (Lentz's method)."""
    maxit = 300
    eps = 1e-15
    fpmin = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, maxit + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _reg_incomplete_beta(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b) ∈ [0, 1]."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _beta_ppf(q: float, a: float, b: float) -> float:
    """Inverse CDF (quantile) of a Beta(a, b) via monotone bisection."""
    if q <= 0.0:
        return 0.0
    if q >= 1.0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(128):
        mid = 0.5 * (lo + hi)
        if _reg_incomplete_beta(a, b, mid) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def binom_sf_ge(k: int, n: int, p: float) -> float:
    """Upper-tail binomial probability P(X ≥ k) for X ~ Binomial(n, p).

    Exact summation — blame trial counts (``n`` = valid trials) are small.
    """
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    total = 0.0
    for j in range(k, n + 1):
        total += math.comb(n, j) * (p**j) * ((1.0 - p) ** (n - j))
    return min(1.0, max(0.0, total))


def benjamini_hochberg(pvalues: list[float], q: float) -> tuple[set[int], list[float]]:
    """Benjamini-Hochberg FDR procedure.

    Returns ``(selected_indices, qvalues)`` where ``qvalues`` are the
    step-up-adjusted p-values (monotone) and ``selected_indices`` is the blamed
    set at false-discovery-rate ``q`` (equivalently ``{i : qvalue_i ≤ q}``).
    """
    m = len(pvalues)
    if m == 0:
        return (set(), [])
    order = sorted(range(m), key=lambda i: pvalues[i])
    qvals = [1.0] * m
    prev = 1.0
    for rank in range(m, 0, -1):
        idx = order[rank - 1]
        adj = min(prev, pvalues[idx] * m / rank)
        qvals[idx] = min(1.0, adj)
        prev = qvals[idx]
    selected = {i for i in range(m) if qvals[i] <= q}
    return (selected, qvals)


# ── Proportion confidence intervals (pluggable) ─────────────────────────────


class CIMethod(StrEnum):
    """Backend for the flip-rate proportion confidence interval."""

    WILSON = "wilson"
    JEFFREYS = "jeffreys"
    CLOPPER_PEARSON = "clopper_pearson"
    AGRESTI_COULL = "agresti_coull"


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion (default 95%)."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    spread = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    # Snap the analytically-exact boundaries (x=0 -> lo=0, x=n -> hi=1); the
    # closed form yields ~1e-17 float dust there that differs across platforms.
    lo = 0.0 if successes == 0 else max(0.0, centre - spread)
    hi = 1.0 if successes == n else min(1.0, centre + spread)
    return (lo, hi)


def proportion_ci(
    successes: int,
    n: int,
    *,
    method: CIMethod = CIMethod.WILSON,
    confidence: float = 0.95,
    z: float | None = None,
) -> tuple[float, float]:
    """Confidence interval for a binomial proportion.

    ``method`` selects the backend; ``confidence`` (or an explicit critical
    ``z``) sets the level. All backends handle the 0-flip / all-flip boundary
    counts where blame trials cluster: Wilson/Agresti-Coull shrink toward 1/2,
    while Jeffreys/Clopper-Pearson pin the touched end to 0 or 1 exactly.
    """
    if n <= 0:
        return (0.0, 1.0)
    if successes < 0 or successes > n:
        raise ValueError(f"successes {successes} out of range [0, {n}]")

    if z is not None:
        zz = z
        conf = 2.0 * NormalDist().cdf(zz) - 1.0
    else:
        conf = confidence
        zz = z_from_confidence(conf)
    alpha = 1.0 - conf

    if method is CIMethod.WILSON:
        return wilson_ci(successes, n, zz)

    if method is CIMethod.AGRESTI_COULL:
        n_t = n + zz * zz
        p_t = (successes + zz * zz / 2.0) / n_t
        margin = zz * math.sqrt(p_t * (1.0 - p_t) / n_t)
        return (max(0.0, p_t - margin), min(1.0, p_t + margin))

    if method is CIMethod.JEFFREYS:
        a = successes + 0.5
        b = n - successes + 0.5
        lo = 0.0 if successes == 0 else _beta_ppf(alpha / 2.0, a, b)
        hi = 1.0 if successes == n else _beta_ppf(1.0 - alpha / 2.0, a, b)
        return (lo, hi)

    # CLOPPER_PEARSON (exact, conservative)
    lo = 0.0 if successes == 0 else _beta_ppf(alpha / 2.0, successes, n - successes + 1)
    hi = 1.0 if successes == n else _beta_ppf(1.0 - alpha / 2.0, successes + 1, n - successes)
    return (lo, hi)


def _normal_ci(values: list[float], confidence: float) -> tuple[float, float]:
    """Mean ± z·(sample standard error) confidence interval over repeated
    estimates of a quantity bounded to ``[-1, 1]`` (a Shapley marginal
    contribution is the difference of two flip-rate proportions, each in
    ``[0, 1]``). Degenerates to a point interval at ``[mean, mean]`` when there
    are fewer than two samples or the samples have zero variance — an honest
    reflection of a deterministic estimator (e.g. a fully-scripted offline
    trial), not a bug.
    """
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    mean = sum(values) / n
    if n < 2:
        return (max(-1.0, mean), min(1.0, mean))
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    stderr = math.sqrt(variance / n)
    if stderr == 0.0:
        return (max(-1.0, mean), min(1.0, mean))
    z = z_from_confidence(confidence)
    return (max(-1.0, mean - z * stderr), min(1.0, mean + z * stderr))


# ── Oracle protocol ─────────────────────────────────────────────────────────


class Oracle(Protocol):
    def grade(self, output: str) -> bool | None: ...


class StringMatchOracle:
    """Grades by regex match: True=success, False=failure, None=ambiguous."""

    def __init__(self, *, success_re: str, failure_re: str) -> None:
        import re

        self._success = re.compile(success_re)
        self._failure = re.compile(failure_re)

    def grade(self, output: str) -> bool | None:
        if self._success.search(output):
            return True
        if self._failure.search(output):
            return False
        return None


# ── Oracle registry ─────────────────────────────────────────────────────────
#
# Stores *classes* (factories), not ready instances: unlike a provider adapter
# or a request matcher, every Oracle implementation so far (`StringMatchOracle`)
# needs domain-specific constructor arguments (the success/failure regexes),
# so there is no sensible zero-arg default instance to register. A caller
# does `get_oracle("string_match")(success_re=..., failure_re=...)`.

ORACLE_REGISTRY: Registry[type[Oracle]] = Registry(ORACLE_GROUP, kind="oracle")
ORACLE_REGISTRY.register("string_match", StringMatchOracle)


def register_oracle(name: str, oracle_cls: type[Oracle]) -> None:
    """Register an ``Oracle`` factory (a class) under ``name``."""
    ORACLE_REGISTRY.register(name, oracle_cls)


def get_oracle(name: str = "string_match") -> type[Oracle]:
    """Look up a registered ``Oracle`` factory by name (default: ``StringMatchOracle``)."""
    return ORACLE_REGISTRY.get_or_raise(name)


def registered_oracles() -> list[str]:
    """Sorted names of all registered ``Oracle`` factories."""
    return ORACLE_REGISTRY.names()


def load_oracle_entry_points(
    *, allow: frozenset[str] | set[str] | None = None, allow_all: bool = False
) -> list[str]:
    """Opt-in: discover third-party oracles advertised under the
    ``tracefork.oracles`` entry-point group (see ``plugins.py`` for the
    security-gating contract — nothing loads unless explicitly allowlisted).
    """
    return ORACLE_REGISTRY.load_entry_points(allow=allow, allow_all=allow_all)


# ── Trial outcomes ───────────────────────────────────────────────────────────


class TrialOutcome(Enum):
    """Three-valued result of a single blame fork trial."""

    FLIP = "flip"
    NO_FLIP = "no_flip"
    UNDEFINED = "undefined"


# ── Result types ────────────────────────────────────────────────────────────


@dataclass
class FlipRateResult:
    step_index: int
    flip_rate: float
    ci_lo: float
    ci_hi: float
    flips: int
    trials: int  # total trials attempted (k) — kept for back-compat
    interpretation: str = ""
    # Three-valued accounting: the CI/flip-rate denominator is ``valid_trials``.
    valid_trials: int = 0
    undefined: int = 0
    divergences: int = 0
    divergence_rate: float = 0.0  # undefined / trials — the per-step trust flag
    trustworthy: bool = True
    # FDR responsible-set membership.
    p_value: float = 1.0
    q_value: float = 1.0
    responsible: bool = False


@dataclass
class BlameReport:
    results: list[FlipRateResult]
    k: int
    total_forks: int
    parent_outcome: bool | None = None
    est_cost_usd: float = 0.0
    ci_method: CIMethod = CIMethod.WILSON
    confidence: float = 0.95
    null_flip_rate: float = 0.05
    fdr_q: float = 0.10
    responsible_set: list[int] = field(default_factory=list)

    def top(self) -> FlipRateResult | None:
        """Highest-flip-rate step (back-compat argmax accessor)."""
        if not self.results:
            return None
        return max(self.results, key=lambda r: r.flip_rate)

    def responsible(self) -> list[FlipRateResult]:
        """FDR-controlled blamed set, ordered by ascending q-value."""
        chosen = [r for r in self.results if r.responsible]
        chosen.sort(key=lambda r: (r.q_value, -r.flip_rate))
        return chosen


@dataclass
class ShapleyResult:
    """One step's temporal-Shapley causal credit, plus necessity/sufficiency."""

    step_index: int
    shapley_value: float
    ci_lo: float
    ci_hi: float
    n_samples: int
    # v({0..i}) and v({0..i-1}) — the "full"/"base" coalition flip-rates the
    # marginal contribution is the difference of (averaged over n_samples).
    coalition_flip_rate: float = 0.0
    base_flip_rate: float = 0.0
    interpretation: str = ""
    # Necessity: would reverting step i's perturbation (dropping the coalition
    # back to the smaller prefix) restore the parent outcome? Read off the same
    # marginal jump: necessity_score IS the shapley_value (how much the
    # coalition's flip-rate falls if step i is reverted).
    necessity: bool = False
    necessity_score: float = 0.0
    # Sufficiency: does injecting ONLY step i into an otherwise-clean run flip
    # it? This is exactly the classic single-step flip-rate (`rank()`'s
    # per-step trial, reused rather than recomputed).
    sufficiency: bool = False
    sufficiency_score: float = 0.0


@dataclass
class ShapleyReport:
    results: list[ShapleyResult]
    n_permutation_samples: int
    k: int
    total_forks: int
    parent_outcome: bool | None = None
    est_cost_usd: float = 0.0
    confidence: float = 0.95

    def top(self) -> ShapleyResult | None:
        """Highest-Shapley-value step (argmax accessor, mirrors BlameReport.top())."""
        if not self.results:
            return None
        return max(self.results, key=lambda r: r.shapley_value)


@dataclass
class BlameEstimate:
    n_candidates: int
    n_forks: int
    est_usd: float


# ── BudgetGovernor ──────────────────────────────────────────────────────────


class BudgetExceededError(RuntimeError):
    """Raised when a blame run's estimated cost exceeds the caller's budget."""


def _detect_model(tape: Tape) -> str:
    """Best-effort model id from the first recorded request (defaults to Sonnet)."""
    adapter = get_adapter()
    for req, _ in tape.exchanges:
        m = adapter.detect_model(req)
        if m:
            return m
    return SONNET


def _avg_tokens(tape: Tape) -> tuple[float, float]:
    """Average (input, output) tokens per exchange — from recorded ``usage`` when
    present, else a ~4-bytes-per-token estimate from the raw bytes."""
    if not tape.exchanges:
        return (0.0, 0.0)
    adapter = get_adapter()
    ins: list[float] = []
    outs: list[float] = []
    for req, resp in tape.exchanges:
        try:
            norm = adapter.parse_response(resp)
            in_tok, out_tok = norm.input_tokens, norm.output_tokens
        except Exception:
            in_tok = out_tok = None
        ins.append(in_tok or max(1, len(req) // 4))
        outs.append(out_tok or max(1, len(resp) // 4))
    n = len(tape.exchanges)
    return (sum(ins) / n, sum(outs) / n)


class BudgetGovernor:
    @staticmethod
    def estimate(
        tape: Tape,
        *,
        k: int,
        model: str | None = None,
        cost_per_fork_usd: float | None = None,
        coalition_samples: int = 0,
    ) -> BlameEstimate:
        """Estimate the dollar cost of a blame run.

        Only the counterfactual *tail* hits the API — the replayed prefix and the
        mutated step itself cost $0. Forking step ``i`` records ``n-1-i`` tail
        calls, so total billed calls = ``sum_i (n-1-i) * k``. Each call is priced
        with the model's real per-token rates (``pricing.get_rates``, backed by the
        bundled offline snapshot) against the tape's recorded token usage. Pass
        ``cost_per_fork_usd`` to override with a flat per-fork figure instead.

        ``coalition_samples`` prices ``BlameEngine.shapley_rank``'s temporal
        Shapley walk: each of its ``m_samples`` repeats re-evaluates a prefix
        coalition ``{0..i}`` for every step ``i`` — the SAME ``sum_i (n-1-i)``
        tail-cost shape as a single-step ``rank()`` pass — plus one more pass for
        the single-step sufficiency check, so the total multiplies the base
        estimate by ``(1 + coalition_samples)``. The default ``0`` reproduces the
        exact pre-existing estimate (``rank()``'s callers are unaffected).
        """
        n_candidates = len(tape.exchanges)
        multiplier = 1 + max(0, coalition_samples)
        n_forks = n_candidates * k * multiplier
        if cost_per_fork_usd is not None:
            est_usd = n_forks * cost_per_fork_usd
        else:
            billed_calls = sum(n_candidates - 1 - i for i in range(n_candidates)) * k * multiplier
            in_rate, out_rate = pricing.get_rates(model or _detect_model(tape))
            avg_in, avg_out = _avg_tokens(tape)
            est_usd = billed_calls * (avg_in * in_rate + avg_out * out_rate)
        return BlameEstimate(n_candidates=n_candidates, n_forks=n_forks, est_usd=est_usd)


# ── outcome extraction ────────────────────────────────────────────────────────


def _outcome_text(resp_bytes: bytes) -> str:
    """Extract the assistant's text from a recorded response via the provider
    adapter, falling back to the decoded raw bytes when it is not parseable JSON."""
    try:
        norm = get_adapter().parse_response(resp_bytes)
    except Exception:
        return resp_bytes.decode(errors="replace")
    return norm.first_text()


def _interpret(flip_rate: float) -> str:
    if flip_rate >= 0.7:
        return "decisive — this step caused it"
    if flip_rate >= 0.3:
        return "suggestive"
    return "diffuse — not the cause"


# ── BlameEngine ─────────────────────────────────────────────────────────────


@dataclass
class _StepTally:
    """Per-step three-valued trial accounting."""

    flips: int = 0
    no_flips: int = 0
    undefined: int = 0
    divergences: int = 0

    @property
    def valid(self) -> int:
        return self.flips + self.no_flips


class BlameEngine:
    """Ranks exchanges by causal flip-rate."""

    @staticmethod
    @instrument("tracefork.blame.rank")
    def rank(
        tape: Tape,
        agent_fn,  # Callable[[anthropic.Anthropic], Any] — the SAME agent
        oracle: Oracle,
        *,
        perturb_factory: Callable[[int], tuple[bytes, object]],
        k: int = 10,
        budget_usd: float = 5.0,
        api_key: str = "sk-ant-blame",
        ci_method: CIMethod = CIMethod.WILSON,
        confidence: float = 0.95,
        ci_z: float | None = None,
        null_flip_rate: float = 0.05,
        fdr_q: float = 0.10,
        min_valid_fraction: float = 0.5,
    ) -> BlameReport:
        """Fork each exchange `k` times with a perturbed response and measure how
        often the graded outcome flips relative to the parent run.

        `perturb_factory(step_idx)` returns `(mutated_response_bytes,
        tail_transport)`, where `tail_transport` serves the counterfactual tail
        (a scripted fake offline, or `None` to use the real API).

        Each trial is FLIP / NO_FLIP / UNDEFINED; the flip-rate and its
        confidence interval are computed over *valid* (non-UNDEFINED) trials
        only. ``ci_method``/``confidence``/``ci_z`` pick the proportion-CI
        backend and level. The FDR-controlled responsible set is chosen by a
        one-sided binomial test of each step's flip-rate against
        ``null_flip_rate`` at false-discovery-rate ``fdr_q``.
        """
        est = BudgetGovernor.estimate(tape, k=k)
        if est.est_usd > budget_usd:
            raise BudgetExceededError(
                f"estimated blame cost ${est.est_usd:.2f} exceeds budget "
                f"${budget_usd:.2f} ({est.n_forks} forks at k={k}); raise the budget "
                f"or lower k"
            )

        parent_outcome: bool | None = None
        if tape.exchanges:
            parent_outcome = oracle.grade(_outcome_text(tape.exchanges[-1][1]))

        results: list[FlipRateResult] = []
        total_forks = 0

        for step_idx in range(len(tape.exchanges)):
            tally = _StepTally()
            for _trial in range(k):
                outcome, diverged = BlameEngine._run_trial(
                    tape, step_idx, perturb_factory, agent_fn, oracle, parent_outcome, api_key
                )
                total_forks += 1
                if outcome is TrialOutcome.FLIP:
                    tally.flips += 1
                elif outcome is TrialOutcome.NO_FLIP:
                    tally.no_flips += 1
                else:
                    tally.undefined += 1
                    if diverged:
                        tally.divergences += 1

            valid = tally.valid
            flip_rate = tally.flips / valid if valid > 0 else 0.0
            ci_lo, ci_hi = proportion_ci(
                tally.flips, valid, method=ci_method, confidence=confidence, z=ci_z
            )
            div_rate = tally.undefined / k if k > 0 else 0.0
            results.append(
                FlipRateResult(
                    step_index=step_idx,
                    flip_rate=flip_rate,
                    ci_lo=ci_lo,
                    ci_hi=ci_hi,
                    flips=tally.flips,
                    trials=k,
                    interpretation=_interpret(flip_rate),
                    valid_trials=valid,
                    undefined=tally.undefined,
                    divergences=tally.divergences,
                    divergence_rate=div_rate,
                    trustworthy=(valid > 0 and (k == 0 or valid / k >= min_valid_fraction)),
                    p_value=binom_sf_ge(tally.flips, valid, null_flip_rate),
                )
            )

        # Benjamini-Hochberg FDR over the per-step one-sided binomial p-values.
        pvals = [r.p_value for r in results]
        selected, qvals = benjamini_hochberg(pvals, fdr_q)
        for i, r in enumerate(results):
            r.q_value = qvals[i]
            r.responsible = i in selected
        responsible_set = sorted(r.step_index for r in results if r.responsible)

        results.sort(key=lambda r: (-r.flip_rate, r.step_index))
        return BlameReport(
            results=results,
            k=k,
            total_forks=total_forks,
            parent_outcome=parent_outcome,
            est_cost_usd=est.est_usd,
            ci_method=ci_method,
            confidence=confidence,
            null_flip_rate=null_flip_rate,
            fdr_q=fdr_q,
            responsible_set=responsible_set,
        )

    @staticmethod
    def _run_trial(
        tape: Tape,
        step_idx: int,
        perturb_factory: Callable[[int], tuple[bytes, object]],
        agent_fn,
        oracle: Oracle,
        parent_outcome: bool | None,
        api_key: str,
    ) -> tuple[TrialOutcome, bool]:
        """Run one fork trial; return ``(outcome, diverged)``.

        A diverged or errored fork is UNDEFINED (not a silent non-flip); the
        caller counts it. ``diverged`` is True only when the failure is a genuine
        `DivergenceError` (recovered from the SDK's exception wrapping), so the
        per-step divergence rate is surfaced rather than swallowed.
        """
        mutated_resp, tail_transport_obj = perturb_factory(step_idx)
        tail_transport = cast("httpx.BaseTransport | None", tail_transport_obj)
        spec = BranchSpec(divergence_step=step_idx, mutated_response=mutated_resp)
        try:
            branch = ForkEngine.fork(
                tape,
                spec,
                agent_fn,
                post_fork_transport=tail_transport,
                api_key=api_key,
            )
        except Exception as exc:
            # A diverged (agent not deterministic up to the step) or otherwise
            # errored fork is UNDEFINED — never counted as a non-flip. Divergence
            # probability grows with depth, so folding it into NO_FLIP would
            # systematically under-blame deep steps.
            return TrialOutcome.UNDEFINED, find_divergence(exc) is not None

        if branch.delta_tape.exchanges:
            graded = oracle.grade(_outcome_text(branch.delta_tape.exchanges[-1][1]))
        else:
            graded = None
        if graded is None or parent_outcome is None:
            return TrialOutcome.UNDEFINED, False
        outcome = TrialOutcome.FLIP if graded != parent_outcome else TrialOutcome.NO_FLIP
        return outcome, False

    @staticmethod
    def _run_coalition_trials(
        tape: Tape,
        steps: tuple[int, ...],
        perturb_factory: Callable[[int], tuple[bytes, object]],
        agent_fn,
        oracle: Oracle,
        parent_outcome: bool | None,
        api_key: str,
        k: int,
    ) -> _StepTally:
        """Run ``k`` coalition-fork trials for ``steps`` (a joint intervention
        set) and tally FLIP/NO_FLIP/UNDEFINED exactly like ``_run_trial``,
        generalized to a coalition — this is what computes ``v(S)``.

        ``perturb_factory`` is called once per coalition member per trial (the
        same one-call-per-trial contract ``_run_trial`` uses for a single step),
        so stateful factories (e.g. a counter-based script) see exactly the calls
        they would for an equivalent sequence of single-step trials. The tail
        transport is the one associated with the *highest-indexed* coalition
        member — for a one-element coalition this reduces to exactly
        ``_run_trial``'s behaviour.
        """
        tally = _StepTally()
        ordered = tuple(sorted(steps))
        top_step = ordered[-1]
        for _trial in range(k):
            per_step = {s: perturb_factory(s) for s in ordered}
            interventions = tuple(StepIntervention(s, per_step[s][0]) for s in ordered)
            tail_transport = cast("httpx.BaseTransport | None", per_step[top_step][1])
            spec = CoalitionSpec(interventions=interventions)
            try:
                branch = ForkEngine.fork_coalition(
                    tape, spec, agent_fn, post_fork_transport=tail_transport, api_key=api_key
                )
            except Exception as exc:
                tally.undefined += 1
                if find_divergence(exc) is not None:
                    tally.divergences += 1
                continue

            if branch.delta_tape.exchanges:
                graded = oracle.grade(_outcome_text(branch.delta_tape.exchanges[-1][1]))
            else:
                graded = None
            if graded is None or parent_outcome is None:
                tally.undefined += 1
                continue
            if graded != parent_outcome:
                tally.flips += 1
            else:
                tally.no_flips += 1
        return tally

    @staticmethod
    @instrument("tracefork.blame.shapley_rank")
    def shapley_rank(
        tape: Tape,
        agent_fn,  # Callable[[anthropic.Anthropic], Any] — the SAME agent
        oracle: Oracle,
        *,
        perturb_factory: Callable[[int], tuple[bytes, object]],
        k: int = 5,
        m_samples: int = 3,
        budget_usd: float = 5.0,
        api_key: str = "sk-ant-shapley",
        confidence: float = 0.95,
        necessity_threshold: float = 0.5,
        sufficiency_threshold: float = 0.5,
    ) -> ShapleyReport:
        """Temporal (order-restricted) Shapley blame — additive to `rank()`.

        Independent single-step flip-rate can't separate a true root cause from
        a downstream step that merely re-expresses it: if step *i*'s fault
        echoes forward, forking *i+1* alone (with an unperturbed prefix) can look
        just as causal as forking *i*, because removing *either* one breaks the
        propagation chain. Coalition/Shapley attribution fixes this by asking a
        joint question instead: given that step *i*'s perturbation is ALREADY in
        play, how much MORE does the outcome flip when step *i+1* is added?

        Each step's Shapley value is its expected marginal contribution
        `v(S∪{i}) − v(S)` to the coalition flip-rate, estimated by PERMUTATION
        SAMPLING but restricted to orderings consistent with temporal order (a
        step is only added after its predecessors — asymmetric/causal Shapley,
        Frye et al. 2020) so a downstream echo isn't credited for its upstream
        cause. Because every tape step's predecessor set is exactly the full
        prefix before it (the tape is a total causal chain: any earlier response
        can in principle be echoed into any later request), that restriction
        admits exactly ONE valid ordering — `(0, 1, ..., n-1)` — collapsing the
        usual multi-permutation Monte-Carlo estimator to a telescoping walk:
        `phi_i = v({0..i}) − v({0..i-1})`. `m_samples` repeats that walk
        independently (a fresh k-trial estimate of every `v(S)` each time) to
        build a genuine permutation-sampling confidence interval on each step's
        value from real trial-to-trial noise, rather than reporting one point
        estimate. (`sum_i phi_i = v({0..n-1}) − v(∅) = v(full) − 0`, the Shapley
        efficiency axiom, since `v(∅)` is defined to be `0` — no intervention
        anywhere is definitionally the parent run.)

        Necessity ("would reverting step i's perturbation restore the parent
        outcome?") is read off the SAME walk: it's exactly the marginal jump
        `phi_i`, since dropping the coalition from `{0..i}` back to `{0..i-1}`
        *is* reverting step i specifically (steps `0..i-1` are held fixed either
        way). Sufficiency ("does injecting only step i into an otherwise-clean
        run flip it?") reuses `rank()`'s existing, independently-tested
        single-step trial — computed once, shared across every step, rather than
        re-derived.

        Cost is `(1 + m_samples)` times a single `rank()` pass (one pass for
        sufficiency, `m_samples` for the coalition walk); `BudgetGovernor.estimate(
        ..., coalition_samples=m_samples)` prices the whole run before any spend,
        and this raises the same `BudgetExceededError` `rank()` does if it's over
        `budget_usd`.
        """
        n = len(tape.exchanges)
        if n == 0:
            return ShapleyReport(results=[], n_permutation_samples=m_samples, k=k, total_forks=0)

        est = BudgetGovernor.estimate(tape, k=k, coalition_samples=m_samples)
        if est.est_usd > budget_usd:
            raise BudgetExceededError(
                f"estimated shapley blame cost ${est.est_usd:.2f} exceeds budget "
                f"${budget_usd:.2f} ({est.n_forks} forks at k={k}, m_samples={m_samples}); "
                f"raise the budget or lower k/m_samples"
            )

        # Sufficiency reuses the existing, independently-tested single-step path.
        # budget_usd=inf: the combined cost was already cleared above, so this
        # inner call must not re-raise on its own (smaller) slice of the budget.
        single_step = BlameEngine.rank(
            tape,
            agent_fn,
            oracle,
            perturb_factory=perturb_factory,
            k=k,
            budget_usd=math.inf,
            api_key=f"{api_key}-sufficiency",
        )
        sufficiency_by_step = {r.step_index: r for r in single_step.results}
        parent_outcome = single_step.parent_outcome

        marginal_samples: list[list[float]] = [[] for _ in range(n)]
        full_samples: list[list[float]] = [[] for _ in range(n)]
        base_samples: list[list[float]] = [[] for _ in range(n)]
        coalition_forks = 0

        for _repeat in range(m_samples):
            base_flip_rate = 0.0  # v(∅) — no intervention, axiomatically the parent
            for i in range(n):
                steps = tuple(range(i + 1))
                tally = BlameEngine._run_coalition_trials(
                    tape, steps, perturb_factory, agent_fn, oracle, parent_outcome, api_key, k
                )
                coalition_forks += k
                valid = tally.valid
                # An all-UNDEFINED coalition trial carries no information about
                # this repeat's marginal step; hold the walk at its prior value
                # rather than injecting a spurious 0-flip-rate reading.
                full_flip_rate = tally.flips / valid if valid > 0 else base_flip_rate
                marginal_samples[i].append(full_flip_rate - base_flip_rate)
                full_samples[i].append(full_flip_rate)
                base_samples[i].append(base_flip_rate)
                base_flip_rate = full_flip_rate

        results: list[ShapleyResult] = []
        for i in range(n):
            m_vals = marginal_samples[i]
            shapley_value = sum(m_vals) / len(m_vals) if m_vals else 0.0
            ci_lo, ci_hi = _normal_ci(m_vals, confidence)
            suff = sufficiency_by_step.get(i)
            sufficiency_score = suff.flip_rate if suff is not None else 0.0
            results.append(
                ShapleyResult(
                    step_index=i,
                    shapley_value=shapley_value,
                    ci_lo=ci_lo,
                    ci_hi=ci_hi,
                    n_samples=len(m_vals),
                    coalition_flip_rate=(
                        sum(full_samples[i]) / len(full_samples[i]) if full_samples[i] else 0.0
                    ),
                    base_flip_rate=(
                        sum(base_samples[i]) / len(base_samples[i]) if base_samples[i] else 0.0
                    ),
                    interpretation=_interpret(shapley_value),
                    necessity=shapley_value >= necessity_threshold,
                    necessity_score=shapley_value,
                    sufficiency=sufficiency_score >= sufficiency_threshold,
                    sufficiency_score=sufficiency_score,
                )
            )

        results.sort(key=lambda r: (-r.shapley_value, r.step_index))
        return ShapleyReport(
            results=results,
            n_permutation_samples=m_samples,
            k=k,
            total_forks=single_step.total_forks + coalition_forks,
            parent_outcome=parent_outcome,
            est_cost_usd=est.est_usd,
            confidence=confidence,
        )
