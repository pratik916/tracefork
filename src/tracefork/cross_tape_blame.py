"""cross_tape_blame.py — a read-only, session-scoped aggregation VIEW over
already-persisted per-tape causal data (tracefork-bge.58).

This module is deliberately NOT a joint cross-tape coalition-execution engine.
Full-scope tracefork-bge.58 asks for jointly re-executing perturbed
coalitions across independently-recorded sub-agent tapes (each with its own
`agent_fn`, possibly different), graded by a NEW cross-tape `Oracle` over a
joint outcome — genuine distributed-agent execution semantics that don't
exist anywhere in this codebase yet (`fork.py`'s `ForkEngine`/
`CoalitionForkTransport` re-run ONE tape's own recorded agent; `blame.py`'s
`shapley_rank` computes `v(S)` over ONE tape). Inventing that here, ahead of
a real validated use case, is exactly what `session_ops.py`'s own docstring
(see `docs/session-cross-tape-design-spike.md`) already says not to do.

What this module builds instead — real, immediately useful progress, not a
placeholder:

- :class:`RunRef` — a `(run_id, step_index)` pair identifying one step of
  one tape within a session, hashable/usable as a dict key. This is the
  vocabulary a future joint `v(S)` primitive would need to name a step
  across tapes; introducing it now, tied to data that already exists, lets
  that future work build on a stable name rather than inventing one under
  pressure later.
- :func:`session_topological_order` — interleaves every tape reachable
  within a session (`store.py`'s `session_tapes` BFS) into ONE ordered
  sequence of `RunRef`s, splicing each spawned child's entire step range in
  at its recorded `spawn_step_index` (the gap `store.py`'s
  `spawn_edges.spawn_step_index` column, added by this same bead, fills —
  see that column's docstring). A child whose spawn edge has no recorded
  `spawn_step_index` (`None` — every edge recorded before this bead, or any
  caller that still omits the parameter) falls back to "this child's steps
  in full, entirely after its parent's own steps" — a documented,
  deliberately-visible approximation, never a silent misordering.
- :func:`cross_tape_causal_edges` — aggregates every tape's ALREADY-PERSISTED
  `causal_edges` rows (written by every past `blame`/`shapley_rank` CLI run
  via `store.py`'s `save_blame_report`/`save_shapley_report`) into one list,
  ordered by `session_topological_order`'s cross-tape position. Zero new
  execution: this only reads rows a caller already computed and stored.

A future full-scope cross-tape blame engine would still need, on top of
this: (1) a `RunRef -> (Tape, agent_fn)` mapping generalizing `shapley_rank`'s
single-tape `v(S)` primitive to a joint coalition spanning several tapes, and
(2) a new cross-tape `Oracle` capable of grading a JOINT outcome across those
tapes, not just one tape's own final response. Neither exists here — this
module's job is the honest, currently-buildable slice: a view over data the
system already has.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .store import TapeStore

__all__ = [
    "RunRef",
    "session_topological_order",
    "cross_tape_causal_edges",
]


@dataclass(frozen=True)
class RunRef:
    """One cross-tape step identity: ``(run_id, step_index)``.

    Frozen (hashable, equality by value) so a ``RunRef`` is directly usable
    as a dict key — e.g. a future ``RunRef -> (Tape, agent_fn)`` mapping (see
    the module docstring's stated deferral) — without depending on this
    module's own ordering logic. A ``RunRef``'s identity is the step itself,
    not its position within a session's topological order: that position is
    encoded by where it sits in :func:`session_topological_order`'s returned
    list, not by any field on the ``RunRef`` itself.
    """

    run_id: str
    step_index: int


def session_topological_order(store: TapeStore, session_id: str) -> list[RunRef]:
    """Interleave every tape reachable within ``session_id`` into one
    cross-tape step order.

    Starting at the session's root run_id, each tape's own steps
    (``0..len(tape.exchanges)-1``) are walked in order; whenever a spawned
    child's edge recorded a ``spawn_step_index`` matching the step just
    emitted, that child's ENTIRE step range is recursively spliced in
    immediately afterward (a nested/multi-level session interleaves
    correctly — a grandchild spawned from a child is spliced into the
    child's own range the same way). Children whose edge recorded no
    ``spawn_step_index`` (``None``) are appended, each fully expanded, after
    all of their parent's own steps and after every step-anchored child —
    the documented fallback for a session recorded before this bead, or any
    caller that omits the parameter: "this child's steps entirely after its
    parent's", never a silently-wrong interleaving.

    A run reached via more than one spawn edge (a diamond — the same
    edge-case :meth:`TapeStore.session_tapes` already dedups) is placed
    exactly once, at the first point its expansion is reached.
    """
    edges = store.spawn_edges_for_session(session_id)
    children_by_parent: dict[str, list[dict[str, Any]]] = {}
    for edge in edges:
        children_by_parent.setdefault(edge["parent_run_id"], []).append(edge)

    root_run_id = store.get_session(session_id)["root_run_id"]
    placed: set[str] = set()

    def expand(run_id: str) -> list[RunRef]:
        placed.add(run_id)
        step_count = len(store.load_tape(run_id).exchanges)
        by_step: dict[int, list[dict[str, Any]]] = {}
        deferred: list[dict[str, Any]] = []
        for edge in children_by_parent.get(run_id, []):
            if edge["child_run_id"] in placed:
                continue
            if edge["spawn_step_index"] is None:
                deferred.append(edge)
            else:
                by_step.setdefault(edge["spawn_step_index"], []).append(edge)

        result: list[RunRef] = []
        for step_index in range(step_count):
            result.append(RunRef(run_id, step_index))
            for edge in by_step.get(step_index, []):
                child = edge["child_run_id"]
                if child not in placed:
                    result.extend(expand(child))
        for edge in deferred:
            child = edge["child_run_id"]
            if child not in placed:
                result.extend(expand(child))
        return result

    return expand(root_run_id)


def cross_tape_causal_edges(store: TapeStore, session_id: str) -> list[dict]:
    """Every already-persisted ``causal_edges`` row across ``session_id``'s
    tapes, ordered by :func:`session_topological_order`'s cross-tape
    position.

    Pure aggregation over :meth:`TapeStore.causal_edges_for_run` — no new
    causal computation happens here. Each returned dict is exactly
    ``causal_edges_for_run``'s own shape (already carrying its own
    ``run_id``, so no extra tagging is needed here); a ``(run_id,
    step_index)`` pair that doesn't appear in the topological order (should
    never happen for a step within a session's own tapes) sorts last rather
    than raising, so one surprising row can't crash the whole aggregation.
    """
    order = session_topological_order(store, session_id)
    position: dict[str, dict[int, int]] = {}
    for i, ref in enumerate(order):
        position.setdefault(ref.run_id, {})[ref.step_index] = i

    edges: list[dict] = []
    for run_id in store.session_tapes(session_id):
        edges.extend(store.causal_edges_for_run(run_id))

    def _position(edge: dict) -> int:
        return position.get(edge["run_id"], {}).get(edge["step_index"], len(order))

    edges.sort(key=_position)
    return edges
