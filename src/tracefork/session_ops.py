"""session_ops.py — small, directly-testable helpers backing `cli.py`'s
`session` sub-app verb family (tracefork-bge.66: `record`/`replay`/`fork`/
`blame`/`serve`), additive on top of the already-landed orchestration-session
model (`store.py`'s `sessions`/`spawn_edges`, `TapeStore.create_session`/
`add_spawn_edge`/`session_tapes`/`get_session`).

Every function here is pure CLI-level looping/validation over calls that
already exist and are already tested -- zero engine-module
(`fork.py`/`blame.py`/`replay.py`/`store.py`) changes. `session fork`/
`session blame` guard session membership via `ensure_run_in_session` and
then call `cli.py`'s own top-level `fork`/`blame` command functions
directly (Typer's `@app.command()` decorator returns the callback
unmodified, so this is in-process composition, not a second engine
implementation). `session replay` reuses tracefork-bge.65's already-shipped
`session_replay.session_divergence_rollup` unchanged, via
`build_uniform_agent_manifest` (mapping every tape in the session to the
SAME `--agent`, the common case that verb covers) rather than
reimplementing a second rollup loop. `session serve` needs no helper here
beyond `TapeStore.get_session`'s existing `KeyError` contract plus
`session_deep_link_path`'s trivial path formatting.

See `docs/session-cross-tape-design-spike.md` for why genuine cross-tape
fork/blame (mutating one tape's response and asking what a SPAWNED
sibling/child tape would then say, or attributing a parent's bad outcome to
a specific delegated sub-agent call) is explicitly NOT attempted here --
that needs `fork.py`'s `CoalitionForkTransport` to relax its
single-linear-causal-ordering assumption first, a real prerequisite
refactor, not a missing CLI verb.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .store import TapeStore

__all__ = [
    "SpawnEdgeSpec",
    "parse_spawn_spec",
    "record_session",
    "ensure_run_in_session",
    "build_uniform_agent_manifest",
    "session_deep_link_path",
]


@dataclass(frozen=True)
class SpawnEdgeSpec:
    """One parsed `--spawn PARENT:CHILD[:REASON]` edge (see
    `parse_spawn_spec`) -- field names mirror `TapeStore.add_spawn_edge`'s
    own keyword names (minus `session_id`, supplied separately by
    `record_session`)."""

    parent_run_id: str
    child_run_id: str
    spawn_reason: str = ""


def parse_spawn_spec(spec: str) -> SpawnEdgeSpec:
    """Parse one `--spawn` value: `'PARENT:CHILD[:REASON]'` -- the same
    repeatable `'locus:payload'` option DSL `cli.py`'s `coalition-fork`
    command already establishes for `--intervene` (split at most into 3
    parts, so a REASON may itself contain `:` but PARENT/CHILD may not).

    Raises `ValueError` on malformed input (no `:` separator, or an empty
    PARENT/CHILD) -- the CLI wraps this as a `typer.BadParameter`, exactly
    like `coalition-fork`'s own `--intervene` parsing already does.
    """
    parts = spec.split(":", 2)
    if len(parts) < 2 or not parts[0] or not parts[1]:
        raise ValueError(f"--spawn must be 'PARENT:CHILD[:REASON]', got {spec!r}")
    reason = parts[2] if len(parts) == 3 else ""
    return SpawnEdgeSpec(parent_run_id=parts[0], child_run_id=parts[1], spawn_reason=reason)


def record_session(
    db: TapeStore, root_run_id: str, spawn_specs: list[str]
) -> tuple[str, list[SpawnEdgeSpec]]:
    """Batch-create a session rooted at ROOT_RUN_ID, then register every
    parsed `--spawn` edge in one call -- looping `TapeStore.create_session`/
    `add_spawn_edge`, the SAME calls `session create`/`session spawn`
    already make one at a time, so a caller doesn't need N+1 separate CLI
    invocations to stand up a session with its spawn manifest.

    No new engine logic: every edge is still FK-validated by
    `add_spawn_edge` itself, so a dangling parent/child run_id still raises
    `sqlite3.IntegrityError`, propagated unchanged (never swallowed here).
    A malformed `--spawn` spec raises `ValueError` (via `parse_spawn_spec`)
    before any write happens.
    """
    edges = [parse_spawn_spec(spec) for spec in spawn_specs]
    session_id = db.create_session(root_run_id=root_run_id)
    for edge in edges:
        db.add_spawn_edge(
            session_id=session_id,
            parent_run_id=edge.parent_run_id,
            child_run_id=edge.child_run_id,
            spawn_reason=edge.spawn_reason,
        )
    return session_id, edges


def ensure_run_in_session(db: TapeStore, session_id: str, run_id: str) -> list[str]:
    """Session-membership guard shared by `session fork`/`session blame`:
    returns SESSION_ID's reachable run_ids (`TapeStore.session_tapes`'s BFS
    order) when RUN_ID is among them.

    Raises `ValueError` if RUN_ID is a stored tape but not reachable within
    this session -- rejected BEFORE any fork/blame attempt, never a partial
    one. Propagates `KeyError` (raised by `session_tapes` -> `get_session`)
    unchanged for an unknown session_id, the same "not found" failure mode
    every other session verb already uses.
    """
    run_ids = db.session_tapes(session_id)
    if run_id not in run_ids:
        raise ValueError(f"run_id {run_id!r} is not reachable within session {session_id!r}")
    return run_ids


def build_uniform_agent_manifest(run_ids: list[str], agent_fn: Any) -> dict[str, Any]:
    """`session replay --agent <path>`'s manifest-construction helper: maps
    every run_id in RUN_IDS to the SAME `agent_fn`, so
    `session_replay.py`'s already-shipped `session_divergence_rollup`
    (tracefork-bge.65's per-run_id `agent_fns` mapping) can be reused
    UNCHANGED for the common one-agent-for-the-whole-session case
    `session replay --agent` covers, instead of a second rollup loop.
    """
    return dict.fromkeys(run_ids, agent_fn)


def session_deep_link_path(session_id: str) -> str:
    """The `server.py`-hosted deep-link path for SESSION_ID -- `session
    serve`'s only new surface. `GET /api/session/{id}` is an EXISTING route
    on `server.py`, not a new one."""
    return f"/api/session/{session_id}"
