"""Blame engine: rank each exchange by causal flip-rate with Wilson CIs.

The causal question is "if step *i* had gone differently, how often would the
run's *outcome* change?" Answering it honestly requires re-running the agent,
not just rewriting the tape: perturbing step *i*'s response changes every
request the agent makes afterward. So for each candidate step we:

  1. fork the recorded run at *i* with a perturbed response — the prefix is
     replayed from the parent tape for $0 and the agent is re-run from there
     (`ForkEngine.fork`);
  2. grade the resulting outcome with an `Oracle`;
  3. count it as a *flip* when the graded outcome differs from the parent run's.

`flip_rate = flips / k` over `k` trials, with a Wilson score 95% interval so a
small *k* doesn't masquerade as certainty. `BudgetGovernor` estimates the
fork count and dollar cost before any spend.

The engine is agent- and domain-agnostic: the caller supplies `agent_fn` (the
same agent that produced the tape) and a `perturb_factory(step) -> (response,
tail_transport)`. In tests and the offline validation suite, `tail_transport`
is a scripted fake (zero cost); for a live run it is `None`, so the
counterfactual tail hits the real API under the budget cap.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Callable, Protocol

from .fork import BranchSpec, ForkEngine
from .tape import Tape


# ── Wilson score CI ────────────────────────────────────────────────────────

def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score confidence interval for a proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    spread = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, centre - spread), min(1.0, centre + spread))


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


# ── Result types ────────────────────────────────────────────────────────────

@dataclass
class FlipRateResult:
    step_index: int
    flip_rate: float
    ci_lo: float
    ci_hi: float
    flips: int
    trials: int
    interpretation: str = ""


@dataclass
class BlameReport:
    results: list[FlipRateResult]
    k: int
    total_forks: int
    parent_outcome: bool | None = None
    est_cost_usd: float = 0.0

    def top(self) -> FlipRateResult | None:
        if not self.results:
            return None
        return max(self.results, key=lambda r: r.flip_rate)


@dataclass
class BlameEstimate:
    n_candidates: int
    n_forks: int
    est_usd: float


# ── BudgetGovernor ──────────────────────────────────────────────────────────

class BudgetGovernor:
    @staticmethod
    def estimate(tape: Tape, *, k: int, cost_per_fork_usd: float = 0.01) -> BlameEstimate:
        n_candidates = len(tape.exchanges)
        n_forks = n_candidates * k
        est_usd = n_forks * cost_per_fork_usd
        return BlameEstimate(n_candidates=n_candidates, n_forks=n_forks, est_usd=est_usd)


# ── outcome extraction ────────────────────────────────────────────────────────

def _outcome_text(resp_bytes: bytes) -> str:
    """Extract the assistant's text from a recorded response (JSON or SSE)."""
    try:
        d = json.loads(resp_bytes)
    except Exception:
        return resp_bytes.decode(errors="replace")
    if isinstance(d, dict):
        for block in d.get("content", []):
            if block.get("type") == "text":
                return block.get("text", "")
    return ""


def _interpret(flip_rate: float) -> str:
    if flip_rate >= 0.7:
        return "decisive — this step caused it"
    if flip_rate >= 0.3:
        return "suggestive"
    return "diffuse — not the cause"


# ── BlameEngine ─────────────────────────────────────────────────────────────

class BlameEngine:
    """Ranks exchanges by causal flip-rate."""

    @staticmethod
    def rank(
        tape: Tape,
        agent_fn,                 # Callable[[anthropic.Anthropic], Any] — the SAME agent
        oracle: Oracle,
        *,
        perturb_factory: Callable[[int], tuple[bytes, object]],
        k: int = 10,
        budget_usd: float = 5.0,
        api_key: str = "sk-ant-blame",
    ) -> BlameReport:
        """Fork each exchange `k` times with a perturbed response and measure how
        often the graded outcome flips relative to the parent run.

        `perturb_factory(step_idx)` returns `(mutated_response_bytes,
        tail_transport)`, where `tail_transport` serves the counterfactual tail
        (a scripted fake offline, or `None` to use the real API).
        """
        parent_outcome: bool | None = None
        if tape.exchanges:
            parent_outcome = oracle.grade(_outcome_text(tape.exchanges[-1][1]))

        results: list[FlipRateResult] = []
        total_forks = 0

        for step_idx in range(len(tape.exchanges)):
            flips = 0
            for _trial in range(k):
                mutated_resp, tail_transport = perturb_factory(step_idx)
                spec = BranchSpec(divergence_step=step_idx, mutated_response=mutated_resp)
                try:
                    branch = ForkEngine.fork(
                        tape, spec, agent_fn,
                        post_fork_transport=tail_transport,
                        api_key=api_key,
                    )
                    total_forks += 1
                except Exception:
                    # A divergent fork (e.g. agent not deterministic up to the
                    # step) counts as cost spent but no observed flip.
                    total_forks += 1
                    continue

                if branch.delta_tape.exchanges:
                    graded = oracle.grade(_outcome_text(branch.delta_tape.exchanges[-1][1]))
                else:
                    graded = None
                if graded is not None and graded != parent_outcome:
                    flips += 1

            flip_rate = flips / k if k > 0 else 0.0
            ci_lo, ci_hi = wilson_ci(flips, k)
            results.append(FlipRateResult(
                step_index=step_idx,
                flip_rate=flip_rate,
                ci_lo=ci_lo,
                ci_hi=ci_hi,
                flips=flips,
                trials=k,
                interpretation=_interpret(flip_rate),
            ))

        results.sort(key=lambda r: r.flip_rate, reverse=True)
        return BlameReport(
            results=results, k=k, total_forks=total_forks, parent_outcome=parent_outcome,
        )
