"""Pure query-language layer over already-shipped, already-tested read
primitives.

`report._tape_to_data` (one exchange's role/preview/request/response_preview,
indexed by step), `diff.branch_diff`/`diff.tape_diff` (already power `cli.py
diff`), and `store.py`'s `causal_edges_for_run`/`cited_by`/`causal_closure`
(already power `cli.py blame`, proven in `tests/test_causal_edges.py`) plus
`store.list_branches` (already feeds `report.py`'s fork-tree panel) each ship
individually. This module adds ZERO new engine logic on top of them: it
parses one line of a small query grammar and calls straight through,
formatting their existing return shapes as text — mirroring `cli.py`'s own
`_print_diff_receipt` presentation style, kept independent so it stays a
pure string-returning function testable with no I/O.

Grammar (one verb per line):

* ``state <run_id> <step>`` — the shaped exchange at ``step`` (see
  ``report._tape_to_data``).
* ``diff <a> <b> [--step N]`` — a branch against its parent (default mode,
  ``diff.branch_diff``), or two independent tapes at one step
  (``--step N``, ``diff.tape_diff``).
* ``causes <run_id> <step|--closure>`` — causal edges for one step (plus the
  branches that cite it, via ``store.cited_by``), or the full fork-graph
  closure of responsible edges (``store.causal_closure``).
* ``tree <run_id>`` — ``run_id``'s direct branches (``store.list_branches``).

`QueryError` wraps bad syntax, an out-of-range step, an unknown verb, or a
`KeyError` from `store.load_tape`/`load_branch` (the existing not-found
convention) re-raised with the same message rather than propagated raw.

No `cmd`/`readline` import, no `typer` import here — see `repl.py` for the
thin interactive-loop wrapper built on top of `dispatch()`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .diff import branch_diff, tape_diff
from .report import _tape_to_data

if TYPE_CHECKING:
    from .diff import RangeDiff, StepDiff
    from .store import TapeStore

__all__ = [
    "QueryError",
    "format_state",
    "format_diff",
    "format_causes",
    "format_tree",
    "dispatch",
]

_VERBS = ("state", "diff", "causes", "tree")


class QueryError(Exception):
    """Bad query syntax, an out-of-range step, an unknown verb, or an unknown
    run_id/branch_id -- always a clean, printable message, never a raw
    `KeyError`/`ValueError`/`IndexError` escaping `dispatch()`."""


def format_state(store: TapeStore, run_id: str, step: int) -> str:
    """The shaped exchange at ``step`` of ``run_id``'s tape -- the same dict
    ``report._tape_to_data(tape)["exchanges"][step]`` produces, rendered as
    text."""
    try:
        tape = store.load_tape(run_id)
    except KeyError as exc:
        raise QueryError(str(exc)) from exc

    exchanges = _tape_to_data(tape)["exchanges"]
    if not (0 <= step < len(exchanges)):
        raise QueryError(
            f"step {step} out of range for run_id {run_id!r}: {len(exchanges)} exchange(s) recorded"
        )
    exchange = exchanges[step]
    return (
        f"state {run_id} step {step}\n"
        f"  role              {exchange['role']}\n"
        f"  preview           {exchange['preview']!r}\n"
        f"  request           {exchange['request']}\n"
        f"  response_preview  {exchange['response_preview']}"
    )


def _format_diff_receipt(heading: str, step_diffs: tuple[StepDiff, ...]) -> str:
    """Text rendering of a sequence of `diff.StepDiff`s -- mirrors `cli.py`'s
    `_print_diff_receipt` presentation (PASS/FAIL per step, a trailing
    changed/total summary), kept independent as a pure string builder."""
    lines = [heading, "─" * 60]
    for s in step_diffs:
        if s.changed:
            n = len(s.request_diffs) + len(s.response_diffs)
            lines.append(f"[FAIL] step {s.step_index} {n} field(s) differ")
            for d in s.request_diffs:
                lines.append(f"  request  {d.path}: {d.recorded!r} -> {d.live!r}")
            for d in s.response_diffs:
                lines.append(f"  response {d.path}: {d.recorded!r} -> {d.live!r}")
        else:
            lines.append(f"[PASS] step {s.step_index} identical")
    n_changed = sum(1 for s in step_diffs if s.changed)
    if n_changed == 0:
        lines.append(f"{len(step_diffs)}/{len(step_diffs)} step(s) identical")
    else:
        lines.append(f"{n_changed}/{len(step_diffs)} step(s) changed")
    return "\n".join(lines)


def format_diff(store: TapeStore, id_a: str, id_b: str, *, step: int | None = None) -> str:
    """``diff.tape_diff`` at one step (``step`` given) or ``diff.branch_diff``
    of a branch against its parent (default), rendered as text. Mirrors
    `cli.py diff`'s own dual-mode dispatch."""
    step_diffs: tuple[StepDiff, ...]
    if step is not None:
        try:
            tape_a = store.load_tape(id_a)
            tape_b = store.load_tape(id_b)
        except KeyError as exc:
            raise QueryError(str(exc)) from exc
        step_diffs = (tape_diff(tape_a, tape_b, step),)
        heading = f"diff {id_a} {id_b} --step {step}"
    else:
        try:
            parent_tape = store.load_tape(id_a)
            branch_row = store.load_branch(id_b)
        except KeyError as exc:
            raise QueryError(str(exc)) from exc
        range_diff: RangeDiff = branch_diff(
            parent_tape,
            branch_row["delta_tape"],
            divergence_step=branch_row["divergence_step"],
        )
        step_diffs = range_diff.steps
        heading = f"diff {id_a} {id_b}"
    return _format_diff_receipt(heading, step_diffs)


def format_causes(
    store: TapeStore, run_id: str, *, step: int | None = None, closure: bool = False
) -> str:
    """Causal edges for one step of ``run_id`` (plus the branches that cite
    it, via ``store.cited_by``), or -- with ``closure=True`` -- the full
    fork-graph closure of responsible edges (``store.causal_closure``)."""
    if closure:
        edges = store.causal_closure(run_id)
        lines = [f"causes {run_id} --closure"]
        if not edges:
            lines.append("  (no responsible edges in the fork-graph closure)")
        for e in edges:
            lines.append(
                f"  {e['run_id']}:{e['step_index']} method={e['method']} "
                f"flip_rate={e['flip_rate']} ci=[{e['ci_lo']}, {e['ci_hi']}] "
                f"q_value={e['q_value']} responsible={e['responsible']}"
            )
        return "\n".join(lines)

    assert step is not None  # enforced by dispatch()'s parsing
    edges = [e for e in store.causal_edges_for_run(run_id) if e["step_index"] == step]
    citers = store.cited_by(run_id, step)
    lines = [f"causes {run_id} {step}"]
    if not edges:
        lines.append("  (no causal edges recorded for this step)")
    for e in edges:
        lines.append(
            f"  method={e['method']} flip_rate={e['flip_rate']} "
            f"ci=[{e['ci_lo']}, {e['ci_hi']}] responsible={e['responsible']} "
            f"q_value={e['q_value']}"
        )
    lines.append(f"  cited_by: {citers if citers else '(none)'}")
    return "\n".join(lines)


def format_tree(store: TapeStore, run_id: str) -> str:
    """``run_id``'s direct branches -- every field ``store.list_branches``
    returns, rendered one line per branch."""
    branches = store.list_branches(run_id)
    lines = [f"tree {run_id}"]
    if not branches:
        lines.append("  (no branches)")
    for b in branches:
        lines.append(
            f"  branch_id={b['branch_id']} divergence_step={b['divergence_step']} "
            f"mutation_desc={b['mutation_desc']!r} created_at={b['created_at']!r} "
            f"branch_digest={b['branch_digest']}"
        )
    return "\n".join(lines)


def dispatch(store: TapeStore, line: str) -> str:
    """Parse one query-language ``line`` and route it to the matching
    ``format_*`` function above.

    Bad syntax, an unknown verb, an unparseable step index, or an unknown
    run_id/branch_id all raise `QueryError` with a clean, printable message
    -- never a raw `KeyError`/`ValueError`/`IndexError`.
    """
    tokens = line.split()
    if not tokens:
        raise QueryError(f"empty query; valid verbs: {', '.join(_VERBS)}")
    verb, args = tokens[0], tokens[1:]

    if verb == "state":
        if len(args) != 2:
            raise QueryError("usage: state <run_id> <step>")
        run_id, step_str = args
        try:
            step = int(step_str)
        except ValueError as exc:
            raise QueryError(f"step must be an integer, got {step_str!r}") from exc
        return format_state(store, run_id, step)

    if verb == "diff":
        if len(args) == 2:
            return format_diff(store, args[0], args[1])
        if len(args) == 4 and args[2] == "--step":
            try:
                step = int(args[3])
            except ValueError as exc:
                raise QueryError(f"--step must be an integer, got {args[3]!r}") from exc
            return format_diff(store, args[0], args[1], step=step)
        raise QueryError("usage: diff <a> <b> [--step N]")

    if verb == "causes":
        if len(args) == 2 and args[1] == "--closure":
            return format_causes(store, args[0], closure=True)
        if len(args) == 2:
            try:
                step = int(args[1])
            except ValueError as exc:
                raise QueryError(f"step must be an integer, got {args[1]!r}") from exc
            return format_causes(store, args[0], step=step)
        raise QueryError("usage: causes <run_id> <step|--closure>")

    if verb == "tree":
        if len(args) != 1:
            raise QueryError("usage: tree <run_id>")
        return format_tree(store, args[0])

    raise QueryError(f"unknown verb {verb!r}; valid verbs: {', '.join(_VERBS)}")
