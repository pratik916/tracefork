"""session_chaos.py — session-scope schedule DERIVATION, generalizing
`transport.chaos_release_order` to a session's multi-tape spawn-lineage
graph (tracefork-bge.64).

Ships schedule derivation only — an honest generalization of what
`chaos_release_order` itself already is: a pure seeded-permutation producer,
never a replay driver. `session_chaos_release_orders`/
`session_sibling_chaos_order` compute per-tape and cross-sibling reorderings;
a caller (or a future bead) is responsible for actually feeding a per-tape
order into `AsyncTraceforkTransport(..., release_order=...)`, exactly as a
caller already must with `chaos_release_order`'s own return value today.

This module deliberately does NOT build a session-level multi-tape replay
DRIVER that re-executes several sub-agent `AsyncTraceforkTransport` instances
in lockstep under the derived interleaving — no such multi-tape
orchestration-replay harness exists anywhere in this codebase (replay is
strictly per-tape), and building one is a materially larger, separate effort:
it would need a shared cross-tape release gate, wall-clock-equivalent
ordering semantics between independently-recorded tapes, and almost
certainly new `Tape`/session metadata declaring which spawn edges were
genuinely concurrent versus sequential delegation. This module gives the
analysis artifact a future driver would consume — the same role
`chaos_release_order`'s own return value plays today.

Two axes are covered:

- :func:`session_chaos_release_orders` — PER-TAPE reordering. For every tape
  reachable within a session (`store.py`'s `session_tapes` BFS), calls the
  REAL, UNMODIFIED `transport.chaos_release_order` with a seed derived from
  the base seed and that tape's own `run_id` (:func:`_derive_seed`) — so
  every sub-agent's own recorded async fan-out gets exactly the reordering
  analysis it would get evaluated in isolation, zero-diff over
  `transport.py`.
- :func:`session_sibling_chaos_order` — the NEW axis: reordering completion
  ACROSS sub-agents. For each parent with two or more spawn children (a
  session-scoped fan-out — the delegation-graph analogue of one tape's
  fully-overlapping `async_batches` entry), a seed-shuffled permutation of
  those children's `run_id`s, mirroring `chaos_release_order`'s own
  within-batch shuffle one level up. A parent with 0 or 1 children is
  omitted, mirroring that function's identity-order-for-non-concurrent case.
"""

from __future__ import annotations

import hashlib
import random
from typing import TYPE_CHECKING

from .transport import chaos_release_order

if TYPE_CHECKING:
    from .store import TapeStore

__all__ = [
    "session_chaos_release_orders",
    "session_sibling_chaos_order",
]


def _derive_seed(seed: int, run_id: str) -> int:
    """Mix a session-wide base ``seed`` with a sha256 of ``run_id`` so each
    tape's derived per-tape seed is stable regardless of BFS discovery
    order — a given ``run_id`` always derives the SAME seed, whichever
    position it's visited at within `session_tapes`'s traversal, so results
    are reproducible independent of iteration order."""
    digest = hashlib.sha256(f"{seed}:{run_id}".encode()).hexdigest()
    return seed ^ int(digest[:16], 16)


def session_chaos_release_orders(
    store: TapeStore, session_id: str, seed: int
) -> dict[str, list[int]]:
    """Per-tape chaos release orders for every tape reachable within
    ``session_id``, keyed by ``run_id``.

    For each ``run_id`` in ``store.session_tapes(session_id)``, loads the
    tape and calls the REAL ``transport.chaos_release_order(tape,
    _derive_seed(seed, run_id))`` — zero-diff reuse, so this is exactly what
    a caller would get analyzing that tape alone, not a reimplementation.
    A tape with no recorded concurrent batches yields its own identity
    order, same as calling ``chaos_release_order`` directly.
    """
    return {
        run_id: chaos_release_order(store.load_tape(run_id), _derive_seed(seed, run_id))
        for run_id in store.session_tapes(session_id)
    }


def session_sibling_chaos_order(
    store: TapeStore, session_id: str, seed: int
) -> dict[str, list[str]]:
    """Cross-sub-agent completion-order permutation within ``session_id``.

    For each ``parent_run_id`` reachable in the session with two or more
    spawn children (via :meth:`TapeStore.session_spawn_children`, scoped to
    THIS session so a sibling group can never leak in a child spawned by the
    same ``parent_run_id`` under a different session), returns a
    seed-shuffled permutation of those children's ``run_id``s — the
    delegation-graph analogue of ``chaos_release_order``'s within-batch
    shuffle, one level up (across sibling AGENTS rather than within one
    agent's own async batch). A parent with 0 or 1 children is omitted
    entirely, mirroring ``chaos_release_order``'s own
    identity-order-for-non-concurrent case (nothing to reorder).
    """
    result: dict[str, list[str]] = {}
    for run_id in store.session_tapes(session_id):
        children = store.session_spawn_children(session_id, run_id)
        if len(children) < 2:
            continue
        rng = random.Random(_derive_seed(seed, run_id))
        shuffled = list(children)
        rng.shuffle(shuffled)
        result[run_id] = shuffled
    return result
