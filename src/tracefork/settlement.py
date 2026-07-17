"""Settlement-adjacent diff export: a winning fork's tool-call side effects,
rendered as a portable, framework-agnostic artifact for an external
apply/settlement layer to consume.

`fork.py`'s tail-record phase re-runs `agent_fn` fresh against a `delta_tape`
that starts EMPTY (`Tape(boundary=..., agent_name=...)`, `tool_exchanges`
defaulting to `[]`). If that tail-record `agent_fn` wires a `tools.py`
`ToolTransport`/`ToolForkTransport`/`NativeToolSeam` onto that same
`delta_tape`, every entry in `delta_tape.tool_exchanges` (a `list[tuple[bytes,
bytes]]` of JSON-RPC request/response frame pairs, see `tape.py`) is, without
further step-slicing, exactly this winning branch's own implied real-world
side effects -- tool calls the parent tape never made.

This module decodes those frames into `SettlementOp`s and renders them as
`to_settlement_json`: an in-toto-Statement-shaped dict (`kind` +
subject-by-digest + predicate) any external system's own apply/settlement
layer can read and choose to act on for real. **TraceFork itself never
applies/settles anything** -- this is export-only (see the gap-analysis
doc's framing: "No settlement/apply layer... TraceFork's fork produces an
analytic Branch/delta-tape").

Decoding is done INLINE here (`json.loads` on the request frame's
`params.name`/`params.arguments`, `tools.py`'s own `decode_result` on the
response frame) rather than through `effects.py`'s `extract_effects`. That
module already landed (tracefork-bge.46) but its `Effect` shape resolves a
tool's arguments down to one comparable `resource` STRING for cross-branch
conflict-detection -- it deliberately never preserves the full `arguments`
dict or the call's `result`, which is exactly what a `SettlementOp` needs to
export. The two modules therefore normalize the SAME underlying
`tools/call` JSON-RPC frames for two different shapes on purpose, not as
duplicated logic pending a merge.

`branch_settlement_diff(parent_tape, branch, *, divergence_step=None,
branch_digest="")` follows `diff.py`'s dual-input contract: `branch` is
either a live `fork.Branch` (its `.delta_tape`/`.divergence_step`/
`.branch_digest` read directly) or a plain, store-reloaded `Tape` (a
`ValueError` unless `divergence_step` is also passed; `branch_digest` may
optionally be supplied too, since a bare `Tape` carries neither -- ignored
when `branch` is a live `Branch`, whose own `.branch_digest` always wins).
Either way this never touches a store or a run_id -- just `Tape` objects,
ints, and strings.

Purely additive, read-only, offline/$0: only reads
`delta_tape.tool_exchanges` and calls `parent_tape.digest()` /
`Tape.digest()` methods that already exist. Zero-diff over
`tape.py`/`fork.py`/`store.py`/`diff.py`/`effects.py` -- no new fields, no
`digest()` changes, no existing caller touched.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .tape import Tape
from .tools import decode_result

if TYPE_CHECKING:
    from .fork import Branch

__all__ = [
    "SETTLEMENT_DIFF_KIND",
    "SettlementDiff",
    "SettlementOp",
    "branch_settlement_diff",
    "to_settlement_json",
]

#: `to_settlement_json`'s `kind` marker. Bumped only on a breaking shape
#: change; a consumer should tolerate unknown keys within a major version,
#: same convention as `receipt.py`'s `SCHEMA_VERSION`.
SETTLEMENT_DIFF_KIND = "tracefork.settlement_diff/v1"


@dataclass(frozen=True)
class SettlementOp:
    """One tool-call side effect from a branch's post-divergence tail,
    decoded straight off a `tools/call` JSON-RPC request frame
    (`tool_name`/`arguments`) paired with its response frame's `result`
    (see `tools.py`'s `make_tool_call_frame`/`make_result_frame`/
    `decode_result`).

    `step_index` is this call's position within `delta_tape.tool_exchanges`
    -- encounter order within that list ALONE, mirroring `effects.py`'s
    `Effect.index` convention for its `SOURCE_TOOL_FRAME` kind. It is NOT an
    absolute LLM-exchange step index: `tool_exchanges` and `exchanges` are
    two independent lists on `Tape` (see `tape.py`).
    """

    tool_name: str
    arguments: dict[str, Any]
    result: Any
    step_index: int


@dataclass(frozen=True)
class SettlementDiff:
    """A winning fork's post-divergence tool-call ops, ready for an
    external apply/settlement layer to read and act on. `parent_tape_digest`
    + `branch_digest` are the Merkle-DAG identity of the fork this diff was
    computed from (see `fork.py`'s `compute_branch_digest`) -- the
    in-toto-Statement `subject` `to_settlement_json` renders. An empty
    `ops` tuple (no tool exchanges at all, or none that decoded as a
    `tools/call` frame) is a normal, valid diff, never an error.
    """

    parent_tape_digest: str
    branch_digest: str
    divergence_step: int
    ops: tuple[SettlementOp, ...] = field(default_factory=tuple)


def _decode_tool_frame(
    request_frame: bytes, response_frame: bytes, step_index: int
) -> SettlementOp | None:
    """Decode one `tool_exchanges` request+response frame pair into a
    `SettlementOp`, or `None` if `request_frame` isn't a well-formed
    `tools/call` request -- skipped, never a crash (mirrors `effects.py`'s
    own best-effort JSON-RPC frame decode)."""
    try:
        request = json.loads(request_frame)
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(request, dict) or request.get("method") != "tools/call":
        return None
    params = request.get("params")
    if not isinstance(params, dict):
        return None
    tool_name = params.get("name")
    if not isinstance(tool_name, str):
        return None
    arguments = params.get("arguments")
    if not isinstance(arguments, dict):
        arguments = {}
    try:
        result = decode_result(response_frame)
    except (ValueError, UnicodeDecodeError):
        result = None
    return SettlementOp(
        tool_name=tool_name, arguments=arguments, result=result, step_index=step_index
    )


def branch_settlement_diff(
    parent_tape: Tape,
    branch: Branch | Tape,
    *,
    divergence_step: int | None = None,
    branch_digest: str = "",
) -> SettlementDiff:
    """Decode a branch's `delta_tape.tool_exchanges` into a `SettlementDiff`.

    `branch` is either:

    * a live `fork.Branch` (exactly what `ForkEngine.fork()`/
      `fork_coalition()` return) -- its `.delta_tape`/`.divergence_step`/
      `.branch_digest` are read directly, or
    * a plain `Tape` (a store-reloaded branch's `delta_tape`, e.g. from
      `TapeStore.load_branch()["delta_tape"]`) -- in which case
      `divergence_step` must be passed explicitly, since a bare `Tape` alone
      carries no record of where it diverged from its parent; `branch_digest`
      may optionally be supplied too (a bare `Tape` also carries no
      Merkle-DAG identity of its own) and is used verbatim.

    Either way this never touches a store or a run_id -- just `Tape`
    objects, ints, and strings, exactly like `diff.py`'s `branch_diff`.

    `parent_tape` is read only via `.digest()` (never mutated); computing a
    `SettlementDiff` never feeds anything into ANY tape's `digest()` --
    verified by a regression test in `tests/test_settlement.py`.

    An empty (or entirely non-`tools/call`) `tool_exchanges` list produces
    `ops=()`, never a crash.
    """
    if isinstance(branch, Tape):
        if divergence_step is None:
            raise ValueError("divergence_step is required when branch is a plain Tape")
        delta_tape = branch
        d_step = divergence_step
        resolved_branch_digest = branch_digest
    else:
        delta_tape = branch.delta_tape
        d_step = branch.divergence_step
        resolved_branch_digest = branch.branch_digest

    ops = tuple(
        op
        for i, (request_frame, response_frame) in enumerate(delta_tape.tool_exchanges)
        if (op := _decode_tool_frame(request_frame, response_frame, i)) is not None
    )

    return SettlementDiff(
        parent_tape_digest=parent_tape.digest(),
        branch_digest=resolved_branch_digest,
        divergence_step=d_step,
        ops=ops,
    )


def to_settlement_json(diff: SettlementDiff) -> dict[str, Any]:
    """Render `diff` as a JSON-safe, in-toto-Statement-shaped dict: `kind`
    (`SETTLEMENT_DIFF_KIND`), a digest-keyed `subject` (`parent_tape_digest`
    + `branch_digest`), and a `predicate` carrying the fork's
    `divergence_step` plus its ordered `ops` list. Every field round-trips
    losslessly through `json.dumps`/`json.loads` -- `arguments`/`result`
    are already JSON-safe (they came from `json.loads`-ing a recorded
    frame), so no further conversion is needed.
    """
    return {
        "kind": SETTLEMENT_DIFF_KIND,
        "subject": {
            "parent_tape_digest": diff.parent_tape_digest,
            "branch_digest": diff.branch_digest,
        },
        "predicate": {
            "divergence_step": diff.divergence_step,
            "ops": [
                {
                    "tool_name": op.tool_name,
                    "arguments": op.arguments,
                    "result": op.result,
                    "step_index": op.step_index,
                }
                for op in diff.ops
            ],
        },
    }
