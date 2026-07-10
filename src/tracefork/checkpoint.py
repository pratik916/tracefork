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
"""

from __future__ import annotations

import os

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
