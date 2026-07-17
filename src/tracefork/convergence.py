"""Reconvergence detection: did two sibling forks from the SAME parent tape at
the SAME divergence step end up producing byte-identical continuations again?

`fork.py`'s `compute_divergence_exchange_digest(request_bytes, response_bytes)`
already IS this repo's per-exchange fingerprint primitive (sha256 of the exact
request+response byte pair at a fork's first divergence point) — this module
reuses it verbatim as the per-step fingerprint, walked over a RANGE of steps
instead of just the one divergence point. No new hash logic is invented here.

Scoped deliberately to the well-defined case: two branches that share ONE
`divergence_step` — exactly what `BlameEngine.rank`'s k trials per step, or
`TournamentEngine`'s k forks per variant, already produce (siblings forked
from the same parent at the same point). Comparing branches with DIFFERENT
divergence steps, or N-way (>2) convergence, has no well-defined shared
step-index alignment and is out of scope: `find_reconvergence` raises
`ValueError` rather than silently comparing misaligned steps.

`find_reconvergence` walks offset-by-offset from 0 to
`min(len(delta_tape_a.exchanges), len(delta_tape_b.exchanges))` — mirroring
`diff.py`'s `tape_diff` same-index-only contract, not `branch_diff`'s
`MISSING`-sentinel one: there is no well-defined "did it converge" answer
against a side that has no exchange there at all, so the comparison silently
truncates to the shorter tape rather than raising or inventing a sentinel
match/mismatch.

`ConvergenceResult.stable` is the genuine-reconvergence signal: `True` only
when EVERY step from the first convergent step onward matched, as opposed to
`reconverged` (any match at all), which a single coincidental fingerprint
collision that immediately reverts would also satisfy.
"""

from __future__ import annotations

from dataclasses import dataclass

from .fork import compute_divergence_exchange_digest
from .tape import Tape

__all__ = ["StepFingerprint", "ConvergenceResult", "find_reconvergence"]


@dataclass(frozen=True)
class StepFingerprint:
    """One absolute step's per-exchange fingerprint on each side of a
    reconvergence comparison."""

    step_index: int
    fingerprint_a: str
    fingerprint_b: str

    @property
    def matched(self) -> bool:
        """`True` when both sides fingerprint identically at this step."""
        return self.fingerprint_a == self.fingerprint_b


@dataclass(frozen=True)
class ConvergenceResult:
    """The step-by-step fingerprint comparison of two same-divergence-step
    sibling branches' `delta_tape`s."""

    steps: tuple[StepFingerprint, ...]

    @property
    def matched_steps(self) -> tuple[int, ...]:
        """Absolute step indices where both sides fingerprinted identically,
        in order."""
        return tuple(s.step_index for s in self.steps if s.matched)

    @property
    def reconverged(self) -> bool:
        """`True` if ANY step matched — includes a coincidental single-step
        collision that later reverts; see `stable` for the genuine signal."""
        return len(self.matched_steps) > 0

    @property
    def first_convergent_step(self) -> int | None:
        """The absolute step index of the first match, or `None` if no step
        matched."""
        matched = self.matched_steps
        return matched[0] if matched else None

    @property
    def stable(self) -> bool:
        """`True` only when EVERY step from `first_convergent_step` onward
        matched — the genuine-reconvergence signal, as opposed to
        `reconverged` (any match), which a coincidental single-step
        fingerprint collision that immediately reverts would also satisfy."""
        first = self.first_convergent_step
        if first is None:
            return False
        return all(s.matched for s in self.steps if s.step_index >= first)


def find_reconvergence(
    delta_tape_a: Tape,
    divergence_step_a: int,
    delta_tape_b: Tape,
    divergence_step_b: int,
) -> ConvergenceResult:
    """Compare two sibling branches' `delta_tape`s — forked from the SAME
    parent tape at the SAME `divergence_step` — offset-by-offset, using
    `fork.py`'s `compute_divergence_exchange_digest(request_bytes,
    response_bytes)` as the per-exchange fingerprint.

    Requires `divergence_step_a == divergence_step_b`: the well-defined case
    is two siblings forked from the same parent at the same point (e.g. two
    of `BlameEngine.rank`'s k trials at one step, or two `TournamentEngine`
    variants) — comparing branches with different divergence steps has no
    well-defined shared step-index alignment and raises `ValueError` instead
    of silently comparing misaligned offsets.

    Walks to `min(len(delta_tape_a.exchanges), len(delta_tape_b.exchanges))`
    — a `delta_tape` shorter than the other (its counterfactual agent stopped
    early) silently truncates the comparison to the shorter tail rather than
    raising or inventing a sentinel match.
    """
    if divergence_step_a != divergence_step_b:
        raise ValueError(
            f"divergence_step_a ({divergence_step_a}) != divergence_step_b "
            f"({divergence_step_b}) — reconvergence is only well-defined for "
            "two siblings forked from the same parent at the same step"
        )

    n = min(len(delta_tape_a.exchanges), len(delta_tape_b.exchanges))
    steps: list[StepFingerprint] = []
    for i in range(n):
        req_a, resp_a = delta_tape_a.exchange(i)
        req_b, resp_b = delta_tape_b.exchange(i)
        steps.append(
            StepFingerprint(
                step_index=divergence_step_a + i,
                fingerprint_a=compute_divergence_exchange_digest(req_a, resp_a),
                fingerprint_b=compute_divergence_exchange_digest(req_b, resp_b),
            )
        )
    return ConvergenceResult(steps=tuple(steps))
