"""Crash-safe incremental (checkpointed) recording.

A crash mid-recording, before ``Tape.save()``/``to_bytes()`` ever runs, loses the
entire in-memory recording. ``CheckpointWriter`` is an opt-in WAL-style companion
that durably commits each recorded exchange to a local SQLite file the instant it
happens, so a crash loses at most the exchange currently in flight, never the
prefix already recorded.

Design (SQLite WAL's rolling/chained-checksum recovery model, and rr/FoundationDB's
"capture minimal nondeterminism, make everything else replayable" lesson, applied
to persistence): each committed exchange is durable and atomic (its own ``BEGIN
IMMEDIATE``/``COMMIT``, reusing ``tape.open_sqlite``'s hardened connection
factory), so recovering from a crash is always an honest linear prefix of what
was recorded — never a torn write, and never a resurrected record from a future
that never completed. ``recover_checkpoint`` returns that prefix alongside a
``was_finalized`` flag; a recovered-but-not-finalized tape is explicitly marked
incomplete rather than silently treated as a clean, complete recording.

**Scope**: only ``exchanges`` are checkpointed — not nondeterminism draws
(``Tape.draws``) or tool exchanges. This is a narrower-than-ideal but honest
boundary: a crash-recovered tape has an accurate exchange prefix but no draw log,
so it is best used as forensic evidence of "how far did recording get" rather than
a bit-exact-replayable artifact in its own right. A clean ``finalize()`` (normal,
non-crash exit) writes the *complete* tape — draws included — via ``Tape.save``,
so a finalized checkpoint is exactly as replayable as any other saved tape.

Usage::

    writer = CheckpointWriter(path, agent_name="my-agent")
    # ... pass writer.append_exchange as an `on_exchange` hook to a transport ...
    writer.finalize(tape)  # on clean completion

    tape, was_finalized = recover_checkpoint(path)  # after a crash, or to inspect

``read_new_exchanges``/``checkpoint_status`` are the read-only, incremental
counterpart to ``recover_checkpoint``'s full-prefix reconstruction: a live-tail
consumer (see ``checkpoint_stream.py``'s SSE endpoint) polls them directly
against the same file a ``CheckpointWriter`` is actively appending to, without
paying to rebuild a ``Tape`` on every poll.
"""

from __future__ import annotations

import os
from typing import Any

from .constants import BOUNDARY_V1
from .tape import Tape, open_sqlite

_CREATE_CHECKPOINT_SCHEMA = """
    CREATE TABLE IF NOT EXISTS checkpoint_exchanges (
        seq INTEGER PRIMARY KEY AUTOINCREMENT,
        req BLOB NOT NULL,
        resp BLOB NOT NULL
    );
    CREATE TABLE IF NOT EXISTS checkpoint_meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );
"""


# ── path confinement (security fix) ─────────────────────────────────────────
#
# `open_sqlite` runs `PRAGMA journal_mode=WAL` on whatever file it opens, and
# every helper below runs `executescript(_CREATE_CHECKPOINT_SCHEMA)` (a
# `CREATE TABLE IF NOT EXISTS`) before reading — so pointing any of these at
# an arbitrary path WRITES to it (new tables gained, journal_mode flipped)
# even though the caller only wanted to read. `server.py`'s
# `GET /api/checkpoint/tail?path=...` is exactly that: an unauthenticated,
# CSRF-reachable GET taking a caller-supplied path. `resolve_confined_checkpoint_path`
# is the gate: nothing is tailable unless the operator names an allowed
# directory (the server wires it in via `init_checkpoint_dirs`) — the same
# opt-in-only posture `fork_allowlist.py`'s `TRACEFORK_FORK_AGENTS` already
# establishes for the click-to-fork endpoints (merely running the server must
# never be enough, by itself, to let a request touch an arbitrary file).
CHECKPOINT_DIRS_ENV = "TRACEFORK_CHECKPOINT_DIRS"


class CheckpointPathNotAllowedError(RuntimeError):
    """Raised when a checkpoint path isn't confined to an allowlisted directory."""


def parse_checkpoint_dirs_env(raw: str | None = None) -> list[str]:
    """Parse a comma-separated list of allowed checkpoint directories.

    Reads `TRACEFORK_CHECKPOINT_DIRS` when `raw` is `None` (mirrors
    `fork_allowlist.py`'s `parse_allowlist_env` env-var pattern); an
    unset/empty value parses to `[]` — opt-in only, never a default-open
    allowlist.
    """
    text = os.environ.get(CHECKPOINT_DIRS_ENV, "") if raw is None else raw
    return [entry.strip() for entry in text.split(",") if entry.strip()]


def resolve_confined_checkpoint_path(path: str, allowed_dirs: list[str]) -> str:
    """Resolve `path` and verify it lives inside one of `allowed_dirs`.

    Both `path` and each of `allowed_dirs` are resolved via `os.path.realpath`
    (following symlinks) before comparing, so a symlink can't be used to
    escape a confined directory. Raises `CheckpointPathNotAllowedError`
    (naming what IS allowlisted, the same style
    `fork_allowlist.resolve_agent_fn` already uses) when `allowed_dirs` is
    empty — default deny, an operator must opt in — or `path` resolves
    outside every allowed directory. Returns the resolved, absolute path a
    caller should use in place of the raw, untrusted input, so a later
    `os.path.exists`/`open_sqlite` call can't be re-tricked by a symlink
    swapped in between this check and that use (TOCTOU) for a `path` that
    otherwise pointed inside an allowed directory.
    """
    if not allowed_dirs:
        raise CheckpointPathNotAllowedError(
            "no checkpoint directories are allowlisted; set "
            f"{CHECKPOINT_DIRS_ENV} (or pass --allow-checkpoint-dir to "
            "`tracefork serve`) to opt in"
        )
    real_path = os.path.realpath(path)
    for allowed in allowed_dirs:
        real_allowed = os.path.realpath(allowed)
        if real_path == real_allowed or real_path.startswith(real_allowed + os.sep):
            return real_path
    raise CheckpointPathNotAllowedError(
        f"path {path!r} is not inside an allowlisted checkpoint directory; "
        f"allowlisted: {sorted(allowed_dirs)}"
    )


class CheckpointWriter:
    """Durably appends recorded exchanges to ``path`` one at a time.

    Each ``append_exchange`` call opens the (WAL-mode) SQLite file, takes a
    ``BEGIN IMMEDIATE`` write lock, inserts the one row, and commits before
    returning — so the exchange is on disk before the caller (a transport's
    record branch) proceeds, not just before the recording session ends.

    ``finalize(tape)`` is the clean-exit path: it writes the complete tape
    (draws included) via ``Tape.save`` into the same file — under separate
    table names, so it does not disturb the incremental log already written —
    and marks the checkpoint ``was_finalized``. Call it once, after the
    recording session completes normally.
    """

    def __init__(self, path: str, *, agent_name: str = "", boundary: str = BOUNDARY_V1) -> None:
        self.path = path
        con = open_sqlite(path)
        try:
            con.executescript(_CREATE_CHECKPOINT_SCHEMA)
            con.execute("BEGIN IMMEDIATE")
            con.execute(
                "INSERT OR IGNORE INTO checkpoint_meta (key, value) VALUES ('agent_name', ?)",
                (agent_name,),
            )
            con.execute(
                "INSERT OR IGNORE INTO checkpoint_meta (key, value) VALUES ('boundary', ?)",
                (boundary,),
            )
            con.execute(
                "INSERT OR IGNORE INTO checkpoint_meta (key, value) VALUES ('was_finalized', '0')"
            )
            con.execute("COMMIT")
        finally:
            con.close()

    def append_exchange(self, request_body: bytes, response_body: bytes) -> None:
        """Durably commit one exchange. Safe to pass directly as an
        ``on_exchange`` hook to ``TraceforkTransport``/``AsyncTraceforkTransport``."""
        con = open_sqlite(self.path)
        try:
            con.execute("BEGIN IMMEDIATE")
            con.execute(
                "INSERT INTO checkpoint_exchanges (req, resp) VALUES (?, ?)",
                (request_body, response_body),
            )
            con.execute("COMMIT")
        finally:
            con.close()

    def finalize(self, tape: Tape) -> None:
        """Write the complete ``tape`` (draws + exchanges + tool exchanges) via
        ``Tape.save`` and mark this checkpoint ``was_finalized``. Call once, on
        clean completion of the recording session this writer was backing."""
        tape.save(self.path)
        con = open_sqlite(self.path)
        try:
            con.executescript(_CREATE_CHECKPOINT_SCHEMA)
            con.execute("BEGIN IMMEDIATE")
            con.execute(
                "INSERT OR REPLACE INTO checkpoint_meta (key, value) VALUES ('was_finalized', '1')"
            )
            con.execute("COMMIT")
        finally:
            con.close()


def read_new_exchanges(path: str, since_seq: int = 0) -> list[tuple[int, bytes, bytes]]:
    """Read exchanges committed after ``since_seq``, in commit order.

    The incremental-read counterpart to ``CheckpointWriter.append_exchange``:
    a live-tail consumer polling this file only wants what's NEW since its
    last poll, not the whole prefix ``recover_checkpoint`` reconstructs.
    Read-only — never touches ``checkpoint_meta`` or the writer's tables.
    Same missing-file guard as ``recover_checkpoint``.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"no checkpoint file at {path!r}")
    con = open_sqlite(path)
    try:
        con.executescript(_CREATE_CHECKPOINT_SCHEMA)
        rows = con.execute(
            "SELECT seq, req, resp FROM checkpoint_exchanges WHERE seq > ? ORDER BY seq",
            (since_seq,),
        ).fetchall()
        return [(seq, bytes(req), bytes(resp)) for seq, req, resp in rows]
    finally:
        con.close()


def checkpoint_status(path: str) -> dict[str, Any]:
    """``was_finalized``/``agent_name``/``exchange_count`` for the checkpoint
    at ``path`` — a cheap status probe for a live-tail poller, so it doesn't
    have to reconstruct (or discard) a full ``Tape`` just to check whether the
    recording it's following has finished. Same missing-file guard as
    ``recover_checkpoint``/``read_new_exchanges``.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"no checkpoint file at {path!r}")
    con = open_sqlite(path)
    try:
        con.executescript(_CREATE_CHECKPOINT_SCHEMA)
        meta = dict(con.execute("SELECT key, value FROM checkpoint_meta").fetchall())
        (exchange_count,) = con.execute("SELECT COUNT(*) FROM checkpoint_exchanges").fetchone()
        return {
            "was_finalized": meta.get("was_finalized") == "1",
            "agent_name": meta.get("agent_name", ""),
            "exchange_count": exchange_count,
        }
    finally:
        con.close()


def recover_checkpoint(path: str) -> tuple[Tape, bool]:
    """Recover the tape at ``path`` written by a ``CheckpointWriter``.

    Returns ``(tape, was_finalized)``. If the checkpoint was cleanly
    ``finalize``d, ``tape`` is the complete tape (loaded via ``Tape.load``) and
    ``was_finalized`` is ``True``. Otherwise ``tape`` is reconstructed from
    exactly the exchanges durably committed before the crash (an honest linear
    prefix, in commit order — never a torn or reordered record), with no draws
    (see the module docstring's scope note), and ``was_finalized`` is ``False``.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"no checkpoint file at {path!r}")
    con = open_sqlite(path)
    try:
        con.executescript(_CREATE_CHECKPOINT_SCHEMA)
        meta = dict(con.execute("SELECT key, value FROM checkpoint_meta").fetchall())
        was_finalized = meta.get("was_finalized") == "1"
        if was_finalized:
            return Tape.load(path), True
        tape = Tape(
            agent_name=meta.get("agent_name", ""),
            boundary=meta.get("boundary", BOUNDARY_V1),
        )
        rows = con.execute("SELECT req, resp FROM checkpoint_exchanges ORDER BY seq").fetchall()
        for req, resp in rows:
            tape.append_exchange(bytes(req), bytes(resp))
        return tape, False
    finally:
        con.close()
