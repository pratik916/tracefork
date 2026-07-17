"""Locate a value inside a tape (or its fork lineage) and prove exactly
where it lives, using ONLY the hashes `Tape.digest()` itself already folds
in — no new hashing scheme.

`locate_value(tape, value)` scans `tape.exchanges` then `tape.tool_exchanges`
— the same request-then-response, index-ascending order `Tape.digest()`
chains (see `tape.py`) — for `value` as a UTF-8 substring, returning the
FIRST hit's kind/index/side plus `tape.sha256_hex`-computed blob hash and the
tape's own `digest()`. Any reader can independently re-hash the exact raw
bytes at `tape.exchange(i)`/`tape.tool_exchange(i)` themselves and compare
against `TapeHit.blob_sha256` — an offline-checkable receipt, the property
this module is named for.

`locate_in_lineage(store, root_run_id, value, follow_lineage=True)` wraps
`locate_value` with a BFS over a run's fork lineage: the root tape (depth 0),
then `store.list_branches(parent_id)` -> `store.load_branch(branch_id)
["delta_tape"]` for each branch found (depth 1, 2, ...). This is the exact
promotion-convention traversal `store.py`'s `causal_closure`/
`branches_forked_from` already document: a branch only has further branches
of its own once its `delta_tape` has itself been promoted to a tape via
`save_tape(delta_tape, run_id=branch_id)` — a branch that was never promoted
this way simply has no rows under `list_branches(branch_id)`, not an error,
so the BFS needs no special-case check for promotion; it falls out of the
`branches` table's own `parent_run_id` foreign key.

Entirely read-only: never touches `Tape.digest()`/`to_bytes()`/`from_bytes()`
or any store schema. `follow_lineage=False` restricts the search to the root
tape only, skipping the BFS entirely.
"""

from __future__ import annotations

from dataclasses import dataclass

from .store import TapeStore
from .tape import Tape, sha256_hex

__all__ = ["TapeHit", "LocateHit", "locate_value", "locate_in_lineage"]


@dataclass(frozen=True)
class TapeHit:
    """Where `value` was found inside a single tape: `kind` is `"exchange"`
    or `"tool_exchange"` (`tape.exchanges` vs `tape.tool_exchanges`), `index`
    is the position within that log, `side` is `"request"` or `"response"`.
    `blob_sha256` is `sha256_hex` of the exact matched raw bytes and
    `tape_digest` is `tape.digest()` — both independently re-derivable by
    anyone reading the raw bytes themselves, no new hash scheme."""

    kind: str
    index: int
    side: str
    blob_sha256: str
    tape_digest: str


@dataclass(frozen=True)
class LocateHit:
    """A `TapeHit` plus where in the fork lineage it was found. `depth` 0 is
    the root tape itself; 1 is a direct branch's `delta_tape`; 2 a
    fork-of-fork; and so on. `branch_id` is `None` at depth 0 (the root tape
    has no branch id of its own)."""

    depth: int
    branch_id: str | None
    hit: TapeHit


def locate_value(tape: Tape, value: str) -> TapeHit | None:
    """Scan `tape.exchanges` then `tape.tool_exchanges`, in the same
    request-then-response, index-ascending order `Tape.digest()` itself
    chains, for `value` as a UTF-8 substring. Returns the FIRST hit, or
    `None` if `value` never occurs anywhere on the tape.
    """
    needle = value.encode("utf-8")
    logs: tuple[tuple[str, list[tuple[bytes, bytes]]], ...] = (
        ("exchange", tape.exchanges),
        ("tool_exchange", tape.tool_exchanges),
    )
    for kind, exchanges in logs:
        for index, (request_body, response_body) in enumerate(exchanges):
            for side, blob in (("request", request_body), ("response", response_body)):
                if needle in blob:
                    return TapeHit(
                        kind=kind,
                        index=index,
                        side=side,
                        blob_sha256=sha256_hex(blob),
                        tape_digest=tape.digest(),
                    )
    return None


def locate_in_lineage(
    store: TapeStore, root_run_id: str, value: str, *, follow_lineage: bool = True
) -> list[LocateHit]:
    """Check the root tape (`store.load_tape(root_run_id)`, depth 0), then —
    unless `follow_lineage=False` — BFS every branch reachable from it
    (`store.list_branches(parent_id)` -> `store.load_branch(branch_id)
    ["delta_tape"]`, depth 1, 2, ...), collecting a `LocateHit` for every
    tape along the way where `value` actually occurs, in BFS
    (depth-ascending) order. Raises `KeyError` if `root_run_id` isn't a
    stored tape (from `store.load_tape`).
    """
    hits: list[LocateHit] = []

    root_tape = store.load_tape(root_run_id)
    root_hit = locate_value(root_tape, value)
    if root_hit is not None:
        hits.append(LocateHit(depth=0, branch_id=None, hit=root_hit))

    if not follow_lineage:
        return hits

    depth = 1
    frontier = [root_run_id]
    while frontier:
        next_frontier: list[str] = []
        for parent_id in frontier:
            for row in store.list_branches(parent_id):
                branch_id = row["branch_id"]
                delta_tape = store.load_branch(branch_id)["delta_tape"]
                branch_hit = locate_value(delta_tape, value)
                if branch_hit is not None:
                    hits.append(LocateHit(depth=depth, branch_id=branch_id, hit=branch_hit))
                next_frontier.append(branch_id)
        frontier = next_frontier
        depth += 1

    return hits
