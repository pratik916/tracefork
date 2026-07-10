"""Generalized point-to-point / fork-branch diff.

`divergence.py`'s `diff_json`/`diff_request_bytes`/`MISSING` already implement
the exact structural-diff primitive needed at a single step (one recorded
request body vs one live/counterfactual body). This module is purely a
higher-level orchestration layer on top of that primitive: it walks a RANGE of
exchange steps rather than one, for two distinct comparisons:

* `branch_diff` — a `Branch`'s `delta_tape` against its parent, from the
  divergence step (or a later `from_step`) onward. Decoupled from
  `TapeStore`: it takes plain `Tape` objects (+ an int divergence step),
  never a store or a run_id, so it works identically whether the branch is a
  live in-memory `fork.Branch` (as returned by `ForkEngine.fork()`/
  `fork_coalition()`) or a store-reloaded `delta_tape` (via
  `TapeStore.load_branch()`).
* `tape_diff` — two independent tapes compared at ONE step index, with no
  parent/child relationship assumed (e.g. two separate recordings of "the
  same" agent).

Nothing here changes `divergence.py`'s public surface (`diff_json`,
`diff_request_bytes`, `diagnose`, `DivergenceDiagnostic`, `MISSING`) — this
module only calls into it. A step present on only one side of a comparison
(most commonly a `delta_tape` shorter than the parent's tail, because the
counterfactual agent stopped early) is reported via the same `MISSING`
sentinel `divergence.py` already uses for a missing dict key/list index,
applied to the whole missing exchange side.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .divergence import MISSING, FieldDiff, _json_or_b64, diff_request_bytes
from .tape import Tape

if TYPE_CHECKING:
    from .fork import Branch

__all__ = ["MISSING", "FieldDiff", "StepDiff", "RangeDiff", "branch_diff", "tape_diff"]


@dataclass(frozen=True)
class StepDiff:
    """The structural diff of one step's request AND response between two tapes."""

    step_index: int
    request_diffs: tuple[FieldDiff, ...]
    response_diffs: tuple[FieldDiff, ...]

    @property
    def changed(self) -> bool:
        """`True` if either side's request or response differs at this step."""
        return bool(self.request_diffs or self.response_diffs)


@dataclass(frozen=True)
class RangeDiff:
    """The step-by-step diff over a contiguous range of steps."""

    steps: tuple[StepDiff, ...]

    @property
    def changed_steps(self) -> tuple[int, ...]:
        """Absolute step indices where `StepDiff.changed` is `True`, in order."""
        return tuple(s.step_index for s in self.steps if s.changed)

    @property
    def identical(self) -> bool:
        """`True` when every step in the range is unchanged."""
        return len(self.changed_steps) == 0


def _diff_optional_bytes(a: bytes | None, b: bytes | None) -> tuple[FieldDiff, ...]:
    """Structural diff of two optional exchange bodies. `None` on either side
    means "no exchange there at all" (the range extends past one tape's
    length) — reported as a single top-level `FieldDiff` against the
    `MISSING` sentinel, reusing `divergence.py`'s own JSON-or-base64 view so a
    present raw/non-JSON body still renders losslessly rather than as raw
    bytes repr."""
    if a is None:
        if b is None:
            return ()
        return (FieldDiff("$", MISSING, _json_or_b64(b)),)
    if b is None:
        return (FieldDiff("$", _json_or_b64(a), MISSING),)
    return tuple(diff_request_bytes(a, b))


def _step_diff(
    step_index: int,
    req_a: bytes | None,
    resp_a: bytes | None,
    req_b: bytes | None,
    resp_b: bytes | None,
) -> StepDiff:
    return StepDiff(
        step_index=step_index,
        request_diffs=_diff_optional_bytes(req_a, req_b),
        response_diffs=_diff_optional_bytes(resp_a, resp_b),
    )


def tape_diff(tape_a: Tape, tape_b: Tape, step: int) -> StepDiff:
    """Structural diff of two independent tapes' request+response at `step`.

    No parent/child relationship is assumed — this is a plain same-step
    comparison between any two tapes (e.g. two separately recorded runs of
    "the same" agent). A `step` out of range on either tape diffs that side
    against the `MISSING` sentinel rather than raising.
    """
    req_a, resp_a = tape_a.exchange(step) if 0 <= step < len(tape_a.exchanges) else (None, None)
    req_b, resp_b = tape_b.exchange(step) if 0 <= step < len(tape_b.exchanges) else (None, None)
    return _step_diff(step, req_a, resp_a, req_b, resp_b)


def branch_diff(
    parent_tape: Tape,
    branch: Branch | Tape,
    from_step: int | None = None,
    *,
    divergence_step: int | None = None,
) -> RangeDiff:
    """Structural diff of a branch's `delta_tape` against its parent, walked
    from the divergence step (or a later `from_step`) to the end of whichever
    tail — parent or delta — extends further.

    `branch` is either:

    * a live `fork.Branch` (exactly what `ForkEngine.fork()`/
      `fork_coalition()` return) — its `.delta_tape`/`.divergence_step` are
      read directly, or
    * a plain `Tape` (a store-reloaded branch's `delta_tape`, e.g. from
      `TapeStore.load_branch()["delta_tape"]`) — in which case
      `divergence_step` must be passed explicitly, since a bare `Tape` alone
      carries no record of where it diverged from its parent.

    Either way `branch_diff` never touches a store or a run_id — just `Tape`
    objects and ints — so it works identically for both cases.

    A step present on only one side (typically the parent's tail extending
    past a `delta_tape` whose agent stopped short once the mutation changed
    its trajectory) is reported via the `MISSING` sentinel, never a crash.
    """
    if isinstance(branch, Tape):
        if divergence_step is None:
            raise ValueError("divergence_step is required when branch is a plain Tape")
        delta_tape = branch
        d_step = divergence_step
    else:
        delta_tape = branch.delta_tape
        d_step = branch.divergence_step

    start = d_step if from_step is None else from_step
    if start < d_step:
        raise ValueError(f"from_step {start} is before the branch's divergence_step {d_step}")

    n_parent = len(parent_tape.exchanges)
    n_delta = len(delta_tape.exchanges)
    # Last absolute step index present on either side of the comparison.
    last_step = max(n_parent, d_step + n_delta) - 1

    steps: list[StepDiff] = []
    for abs_step in range(start, last_step + 1):
        i = abs_step - d_step
        req_a, resp_a = parent_tape.exchange(abs_step) if abs_step < n_parent else (None, None)
        req_b, resp_b = delta_tape.exchange(i) if 0 <= i < n_delta else (None, None)
        steps.append(_step_diff(abs_step, req_a, resp_a, req_b, resp_b))
    return RangeDiff(steps=tuple(steps))
