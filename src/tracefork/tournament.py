"""Tournament engine: rank N candidate continuations at ONE fixed step.

`blame.py` asks "how much does the outcome flip if step *i* had gone
differently, across every step in the tape?" — a comparison ACROSS steps of a
single run. A tournament asks a different-axis question at a single, already-
chosen step: "which of these N pre-specified candidate responses is best?" —
best-of-N argmax, but statistically validated rather than a bare comparison of
noisy point estimates.

Each `Variant` is forked `k` times at the same `step_index` via
`ForkEngine.fork` (the SAME three-phase transport `blame.py` uses — prefix
replayed for $0, the fixed step forced to the variant's response, any tail
recorded fresh), graded by an `Oracle`, and ranked by its own success rate
(not a flip-rate against a baseline — a tournament has no baseline run to
flip away from). `TournamentEngine.estimate` prices the run via
`BudgetGovernor.estimate` (never a parallel cost model) before any trial
runs; `TournamentEngine.run` raises `blame.py`'s own `BudgetExceededError` if
that estimate exceeds `budget_usd`.

When `step_index` is the tape's LAST exchange, forking it has an EMPTY tail —
`ForkTransport` never calls its inner transport — so comparing final-answer
candidates there is genuinely $0 and needs no scripted tail transport at all;
this is the common "best-of-N final answers" case. When `step_index` is
earlier, each variant's `tail_transport` (scripted offline, or `None` for the
real API) determines how its candidate's downstream continuation plays out.

A winner is declared only when the top-scoring variant is significantly
better than EVERY other variant: each runner-up's own trial counts are tested
one-sided (`binom_sf_ge`, reused from `blame.py`) against the top variant's
observed rate as the null, and the resulting p-values are corrected jointly
via Benjamini-Hochberg (`benjamini_hochberg`, also reused from `blame.py`) at
`fdr_q` — the "BH across pairwise/vs-baseline comparisons for >2 candidates"
idea, rather than an uncorrected per-arm check that inflates the false-winner
rate as N grows. Two variants with the same underlying success probability
will not, in general, all clear that bar, so no spurious winner is declared.

(SOTA ideas noted, not implemented here: Sequential-Halving / LUCB fixed-
budget or fixed-confidence racers that stop early instead of spending a fixed
`k` on every variant, and reporting a *selection-adjusted* ("infer-and-
widen") CI on the argmax winner alongside the naive same-sample Wilson CI to
avoid the winner's-curse optimism bias. Both are genuine future upgrades to
`run()`'s sampling/CI strategy, not reflected in this version's arithmetic.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .blame import (
    BlameEstimate,
    BudgetExceededError,
    BudgetGovernor,
    CIMethod,
    Oracle,
    _outcome_text,
    benjamini_hochberg,
    binom_sf_ge,
    proportion_ci,
)
from .fork import BranchSpec, ForkEngine
from .nondet import find_divergence
from .observability import instrument
from .tape import Tape

if TYPE_CHECKING:
    import httpx

__all__ = [
    "BudgetExceededError",
    "TournamentEngine",
    "TournamentReport",
    "Variant",
    "VariantResult",
]


@dataclass
class Variant:
    """One candidate continuation to try at a tournament's fixed step.

    `response` is the forced response `ForkEngine.fork` serves at
    `step_index` (`BranchSpec.mutated_response`). `tail_transport` serves any
    exchanges AFTER `step_index` (a scripted fake offline, or `None` to hit
    the real API); it is never consulted when `step_index` is the tape's last
    exchange, since that fork has no tail.
    """

    name: str
    response: bytes
    tail_transport: httpx.BaseTransport | None = None
    mutation_desc: str = ""


@dataclass
class VariantResult:
    """One variant's scored, ranked outcome."""

    name: str
    score: float
    ci_lo: float
    ci_hi: float
    successes: int
    trials: int  # total trials attempted (k)
    valid_trials: int = 0
    undefined: int = 0
    divergences: int = 0
    # Set only for non-top results: the BH-adjusted q-value of "this variant
    # is at least as good as the top variant" (small => significantly worse).
    q_value: float = 1.0
    # Set only on the top (index 0) result: True iff EVERY other variant
    # cleared the FDR bar above.
    significant_winner: bool = False


@dataclass
class TournamentReport:
    step_index: int
    results: list[VariantResult]  # sorted descending by score
    k: int
    total_forks: int
    est_cost_usd: float = 0.0
    ci_method: CIMethod = CIMethod.WILSON
    confidence: float = 0.95
    fdr_q: float = 0.10

    def winner(self) -> VariantResult | None:
        """The significantly-best variant, or `None` if no variant cleared
        the FDR bar against every runner-up (including a tie for the top
        score, or too few valid trials to tell)."""
        if self.results and self.results[0].significant_winner:
            return self.results[0]
        return None


@dataclass
class _VariantTally:
    successes: int = 0
    failures: int = 0
    undefined: int = 0
    divergences: int = 0

    @property
    def valid(self) -> int:
        return self.successes + self.failures


class TournamentEngine:
    """Ranks pre-specified candidate continuations at one fixed step."""

    @staticmethod
    def estimate(
        tape: Tape,
        *,
        step_index: int,
        n_variants: int,
        k: int,
        model: str | None = None,
        cost_per_fork_usd: float | None = None,
    ) -> BlameEstimate:
        """Estimate the dollar cost of a tournament run — always via
        `BudgetGovernor.estimate` (never a parallel cost model), adapted to
        this engine's "N variants at ONE step" shape rather than blame's
        "every step once" shape.

        A fork at the tape's LAST exchange has an empty tail (no downstream
        API call is ever made), so that case is priced at exactly `$0`
        without calling the governor at all.

        Otherwise, when `cost_per_fork_usd` is a flat per-fork fee (not a
        per-downstream-call fee), a single-exchange probe tape collapses
        `BudgetGovernor.estimate`'s own `n_forks = n_candidates * k *
        multiplier` to exactly `n_variants * k` — reused verbatim. For the
        real per-token pricing model, a 2-exchange probe (the fixed step plus
        its immediate successor) contributes exactly ONE nonzero `(n-1-i)`
        term per trial to the governor's billed-call sum;
        `coalition_samples=tail_length - 1` (the same `(1 + coalition_samples)`
        multiplier `shapley_rank` already uses to reprice a repeated sweep)
        scales that single term up to this tournament's real `tail_length`
        downstream calls per trial, without re-deriving the dollar-per-call
        math a second time.
        """
        n = len(tape.exchanges)
        if step_index < 0 or step_index >= n:
            raise ValueError(f"step_index {step_index} out of range [0, {n})")
        if n_variants < 1:
            raise ValueError("n_variants must be >= 1")

        trials = n_variants * k
        tail_length = max(0, n - 1 - step_index)

        if tail_length == 0:
            return BlameEstimate(n_candidates=n_variants, n_forks=trials, est_usd=0.0)

        if cost_per_fork_usd is not None:
            probe = Tape(
                exchanges=[tape.exchanges[step_index]],
                boundary=tape.boundary,
                agent_name=tape.agent_name,
            )
            probe_est = BudgetGovernor.estimate(
                probe, k=trials, model=model, cost_per_fork_usd=cost_per_fork_usd
            )
        else:
            probe = Tape(
                exchanges=[
                    tape.exchanges[step_index],
                    tape.exchanges[min(step_index + 1, n - 1)],
                ],
                boundary=tape.boundary,
                agent_name=tape.agent_name,
            )
            probe_est = BudgetGovernor.estimate(
                probe, k=trials, model=model, coalition_samples=tail_length - 1
            )

        return BlameEstimate(n_candidates=n_variants, n_forks=trials, est_usd=probe_est.est_usd)

    @staticmethod
    @instrument("tracefork.tournament.run")
    def run(
        tape: Tape,
        *,
        step_index: int,
        variants: list[Variant],
        agent_fn,  # Callable[[anthropic.Anthropic], Any] — the SAME agent
        oracle: Oracle,
        k: int = 10,
        budget_usd: float = 5.0,
        api_key: str = "sk-ant-tournament",
        ci_method: CIMethod = CIMethod.WILSON,
        confidence: float = 0.95,
        ci_z: float | None = None,
        fdr_q: float = 0.10,
        cost_per_fork_usd: float | None = None,
        model: str | None = None,
        boundary_guard: bool = False,
    ) -> TournamentReport:
        """Fork each of `variants` `k` times at `step_index` and rank them by
        success rate.

        Gates on `TournamentEngine.estimate` BEFORE any trial runs, raising
        `BudgetExceededError` (reused from `blame.py`) if the estimate
        exceeds `budget_usd` — exactly like `BlameEngine.rank`.
        """
        n = len(tape.exchanges)
        if step_index < 0 or step_index >= n:
            raise ValueError(f"step_index {step_index} out of range [0, {n})")
        if not variants:
            raise ValueError("TournamentEngine.run requires at least one variant")
        names = [v.name for v in variants]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate variant names: {names}")

        est = TournamentEngine.estimate(
            tape,
            step_index=step_index,
            n_variants=len(variants),
            k=k,
            model=model,
            cost_per_fork_usd=cost_per_fork_usd,
        )
        if est.est_usd > budget_usd:
            raise BudgetExceededError(
                f"estimated tournament cost ${est.est_usd:.2f} exceeds budget "
                f"${budget_usd:.2f} ({est.n_forks} forks at k={k} across "
                f"{len(variants)} variants); raise the budget or lower k"
            )

        results: list[VariantResult] = []
        total_forks = 0

        for variant in variants:
            tally = _VariantTally()
            for _trial in range(k):
                graded, diverged = TournamentEngine._run_variant_trial(
                    tape, step_index, variant, agent_fn, oracle, api_key, boundary_guard
                )
                total_forks += 1
                if graded is True:
                    tally.successes += 1
                elif graded is False:
                    tally.failures += 1
                else:
                    tally.undefined += 1
                    if diverged:
                        tally.divergences += 1

            valid = tally.valid
            score = tally.successes / valid if valid > 0 else 0.0
            ci_lo, ci_hi = proportion_ci(
                tally.successes, valid, method=ci_method, confidence=confidence, z=ci_z
            )
            results.append(
                VariantResult(
                    name=variant.name,
                    score=score,
                    ci_lo=ci_lo,
                    ci_hi=ci_hi,
                    successes=tally.successes,
                    trials=k,
                    valid_trials=valid,
                    undefined=tally.undefined,
                    divergences=tally.divergences,
                )
            )

        results.sort(key=lambda r: (-r.score, r.name))
        TournamentEngine._mark_significant_winner(results, fdr_q)

        return TournamentReport(
            step_index=step_index,
            results=results,
            k=k,
            total_forks=total_forks,
            est_cost_usd=est.est_usd,
            ci_method=ci_method,
            confidence=confidence,
            fdr_q=fdr_q,
        )

    @staticmethod
    def _mark_significant_winner(results: list[VariantResult], fdr_q: float) -> None:
        """Test the top result against every runner-up, BH-corrected.

        Each runner-up's p-value is the one-sided binomial probability
        (`binom_sf_ge`, reused from `blame.py`) that its OWN observed success
        count is as low as it is, or lower, if its true rate equalled the
        top's observed rate — small means "this variant's own data is
        inconsistent with matching the top", i.e. significant evidence it is
        worse. A runner-up with zero valid trials never occupies a
        correction slot (mirrors `blame.py`'s untrustworthy-step exclusion)
        and can never be counted as significantly worse, so its presence
        blocks a winner from being declared.
        """
        if not results:
            return
        top = results[0]
        others = results[1:]
        if not others:
            top.significant_winner = top.valid_trials > 0
            return
        if top.valid_trials == 0:
            return

        trustworthy_idx = [i for i, r in enumerate(others) if r.valid_trials > 0]
        pvalues = [
            1.0 - binom_sf_ge(others[i].successes + 1, others[i].valid_trials, top.score)
            for i in trustworthy_idx
        ]
        selected, qvals = benjamini_hochberg(pvalues, fdr_q)
        for local_i, global_i in enumerate(trustworthy_idx):
            others[global_i].q_value = qvals[local_i]
        all_significantly_worse = (
            len(trustworthy_idx) == len(others) and set(range(len(trustworthy_idx))) == selected
        )
        top.significant_winner = all_significantly_worse

    @staticmethod
    def _run_variant_trial(
        tape: Tape,
        step_index: int,
        variant: Variant,
        agent_fn,
        oracle: Oracle,
        api_key: str,
        boundary_guard: bool,
    ) -> tuple[bool | None, bool]:
        """Run one fork trial for `variant`; return `(graded, diverged)`.

        A diverged or errored fork, or an ungradeable outcome, is `(None,
        ...)` — never silently counted as a failure. `diverged` is True only
        for a genuine `DivergenceError` (recovered from the SDK's exception
        wrapping), mirroring `blame.py`'s `_run_trial`.
        """
        spec = BranchSpec(
            divergence_step=step_index,
            mutated_response=variant.response,
            mutation_desc=variant.mutation_desc,
        )
        try:
            branch = ForkEngine.fork(
                tape,
                spec,
                agent_fn,
                post_fork_transport=variant.tail_transport,
                api_key=api_key,
                boundary_guard=boundary_guard,
            )
        except Exception as exc:
            return None, find_divergence(exc) is not None

        if branch.delta_tape.exchanges:
            graded = oracle.grade(_outcome_text(branch.delta_tape.exchanges[-1][1]))
        else:
            graded = None
        return graded, False
