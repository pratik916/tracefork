"""session_cost.py — cost model for a minimal-recompute session fork
(``tracefork session cost``), additive on top of the already-landed
orchestration-session model (``store.py``'s ``sessions``/``spawn_edges``,
``TapeStore.session_tapes``/``spawn_children``/``get_session``/
``load_tape``) and ``blame.py``'s existing ``BudgetGovernor.estimate``
(reused verbatim — zero diff to ``blame.py``).

``plan_session_fork`` answers: if I fork ``target_run_id``, which OTHER
tapes in its session could a counterfactual at that tape possibly have
touched, and what does re-pricing only that "recompute set" save over the
naive "re-price the whole session" baseline? It walks the spawn-edge DAG
via ``_spawn_descendants`` (a BFS over ``store.spawn_children``, the same
BFS-dedup shape as ``session_tapes``, just rooted at an arbitrary run
instead of the session root) to find ``target_run_id``'s full transitive
spawn subtree. Since ``spawn_edges`` has no per-step association between a
parent tape and the child it spawned (a distinct, later ``spawn_step_index``
bead), this is deliberately CONSERVATIVE: any fork of ``target_run_id`` is
assumed to potentially invalidate its ENTIRE spawn subtree, not just
descendants spawned at-or-after the specific forked step. Everything else
in the session — ancestors and siblings reached via a different DAG path —
is genuinely independent upstream: spawn edges are directed delegation-only,
so nothing can flow backward into them, and they are safely skipped.

Scoped to the estimator/planner: this module prices the skip-vs-recompute
partition, it does NOT re-execute anything. Actually threading a target
fork's counterfactual output into each recompute-set descendant (looping
``ForkEngine.fork()``/``rebase()`` over the DAG) needs a prompt/data-flow
channel ``spawn_edges`` doesn't have today (it records delegation existence
and a free-text reason only) — a separate, larger follow-on bead.
"""

from __future__ import annotations

from dataclasses import dataclass

from .blame import BudgetGovernor
from .store import TapeStore


@dataclass
class SessionForkPlan:
    """The skip-vs-recompute partition and $ savings for forking
    ``target_run_id`` within ``session_id``."""

    session_id: str
    target_run_id: str
    recompute_run_ids: list[str]
    skip_run_ids: list[str]
    est_usd: float
    est_usd_naive: float
    savings_usd: float
    savings_pct: float


def _spawn_descendants(store: TapeStore, run_id: str) -> list[str]:
    """BFS ``store.spawn_children`` reachable from ``run_id`` (excluding
    ``run_id`` itself) — the same BFS-dedup shape as ``TapeStore.session_tapes``
    (a run reached via more than one path, e.g. a diamond, appears once),
    just rooted at an arbitrary run instead of a session's root."""
    order: list[str] = []
    seen = {run_id}
    frontier = [run_id]
    while frontier:
        current = frontier.pop(0)
        for child in store.spawn_children(current):
            if child not in seen:
                seen.add(child)
                order.append(child)
                frontier.append(child)
    return order


def plan_session_fork(
    store: TapeStore,
    session_id: str,
    target_run_id: str,
    *,
    k: int = 1,
    model: str | None = None,
    cost_per_fork_usd: float | None = None,
) -> SessionForkPlan:
    """Plan a minimal-recompute fork of ``target_run_id`` within
    ``session_id``.

    ``recompute_run_ids`` is ``[target_run_id, *its transitive spawn
    descendants]`` (conservative — see module docstring); ``skip_run_ids``
    is every other tape in ``session_tapes(session_id)``. Both sets are
    priced by calling the existing public ``BudgetGovernor.estimate``
    once per tape (summed) — no new pricing math, no ``blame.py`` edit —
    ``recompute_run_ids`` for ``est_usd``, all of ``session_tapes`` for
    ``est_usd_naive`` (the naive "recompute everything in the session"
    baseline). ``savings_usd = est_usd_naive - est_usd``; ``savings_pct``
    is that as a percentage of ``est_usd_naive``, guarded against
    divide-by-zero (``0.0`` when ``est_usd_naive`` is ``0``).

    Raises ``KeyError`` (via ``TapeStore.get_session``, through
    ``session_tapes``) for an unknown ``session_id``, and ``ValueError``
    for a ``target_run_id`` not reachable in the session.

    Never mutates any tape: only calls ``store.load_tape`` (a read) and
    ``BudgetGovernor.estimate`` (pure), never ``Tape.to_bytes()``/
    ``from_bytes()``.
    """
    all_tapes = store.session_tapes(session_id)  # raises KeyError for unknown session_id
    if target_run_id not in all_tapes:
        raise ValueError(f"target_run_id {target_run_id!r} not reachable in session {session_id!r}")

    recompute_run_ids = [target_run_id, *_spawn_descendants(store, target_run_id)]
    recompute_set = set(recompute_run_ids)
    skip_run_ids = [rid for rid in all_tapes if rid not in recompute_set]

    def _priced(run_ids: list[str]) -> float:
        return sum(
            BudgetGovernor.estimate(
                store.load_tape(rid), k=k, model=model, cost_per_fork_usd=cost_per_fork_usd
            ).est_usd
            for rid in run_ids
        )

    est_usd = _priced(recompute_run_ids)
    est_usd_naive = _priced(all_tapes)
    savings_usd = est_usd_naive - est_usd
    savings_pct = (savings_usd / est_usd_naive * 100.0) if est_usd_naive > 0 else 0.0

    return SessionForkPlan(
        session_id=session_id,
        target_run_id=target_run_id,
        recompute_run_ids=recompute_run_ids,
        skip_run_ids=skip_run_ids,
        est_usd=est_usd,
        est_usd_naive=est_usd_naive,
        savings_usd=savings_usd,
        savings_pct=savings_pct,
    )
