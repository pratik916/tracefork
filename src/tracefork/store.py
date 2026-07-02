"""TapeStore — SQLite-backed persistence for tapes and branch metadata.

Schema:
  tapes   (run_id TEXT PK, agent_name TEXT, tape_bytes BLOB, created_at TEXT)
  branches(branch_id TEXT PK, parent_run_id TEXT, divergence_step INT,
           delta_tape_bytes BLOB, mutation_desc TEXT, created_at TEXT)
"""

from __future__ import annotations

import threading
import uuid
from typing import Protocol, runtime_checkable

from .tape import Tape, open_sqlite


@runtime_checkable
class StorageBackend(Protocol):
    """The persistence interface ``TapeStore`` (SQLite) already satisfies.

    Naming this seam lets a filesystem, object-store (S3/GCS), or other
    backend drop in later without touching any caller that only depends on
    this surface (``cli.py``, ``server.py``, ``fork.py``, ``blame.py``).
    ``TapeStore`` stays the default, unchanged implementation — nothing here
    alters its behavior.
    """

    def save_tape(self, tape: Tape, *, run_id: str | None = None, created_at: str = "") -> str: ...

    def load_tape(self, run_id: str) -> Tape: ...

    def list_runs(self) -> list[dict]: ...

    def save_branch(
        self,
        *,
        parent_run_id: str,
        divergence_step: int,
        delta_tape: Tape,
        mutation_desc: str = "",
        created_at: str = "",
    ) -> str: ...

    def load_branch(self, branch_id: str) -> dict: ...

    def list_branches(self, parent_run_id: str) -> list[dict]: ...

    def close(self) -> None: ...


_DDL = """
CREATE TABLE IF NOT EXISTS tapes (
    run_id       TEXT PRIMARY KEY,
    agent_name   TEXT NOT NULL,
    tape_bytes   BLOB NOT NULL,
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS branches (
    branch_id         TEXT PRIMARY KEY,
    parent_run_id     TEXT NOT NULL,
    divergence_step   INTEGER NOT NULL,
    delta_tape_bytes  BLOB NOT NULL,
    mutation_desc     TEXT NOT NULL DEFAULT '',
    created_at        TEXT NOT NULL,
    FOREIGN KEY(parent_run_id) REFERENCES tapes(run_id)
);
"""


class TapeStore:
    """SQLite-backed store for tapes and branches."""

    def __init__(self, db_path: str = "store.db") -> None:
        self._path = db_path
        # WAL + busy_timeout + foreign_keys, autocommit so writers take an explicit
        # write lock (see open_sqlite). The one shared connection is safe across
        # threads (check_same_thread=False); `_write_lock` serializes the blame
        # fork write fan-out so two threads never open a transaction at once.
        self._con = open_sqlite(db_path)
        self._write_lock = threading.Lock()
        self._con.executescript(_DDL)

    # ── tapes ──────────────────────────────────────────────────────────────

    def save_tape(self, tape: Tape, *, run_id: str | None = None, created_at: str = "") -> str:
        rid = run_id or uuid.uuid4().hex[:12]
        blob = tape.to_bytes()
        with self._write_lock:
            self._con.execute("BEGIN IMMEDIATE")
            try:
                self._con.execute(
                    "INSERT OR REPLACE INTO tapes(run_id, agent_name, tape_bytes, created_at) "
                    "VALUES(?,?,?,?)",
                    (rid, tape.agent_name, blob, created_at),
                )
                self._con.execute("COMMIT")
            except BaseException:
                self._con.execute("ROLLBACK")
                raise
        return rid

    def load_tape(self, run_id: str) -> Tape:
        row = self._con.execute("SELECT tape_bytes FROM tapes WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"run_id {run_id!r} not found")
        return Tape.from_bytes(bytes(row[0]))

    def list_runs(self) -> list[dict]:
        rows = self._con.execute(
            "SELECT run_id, agent_name, created_at FROM tapes ORDER BY created_at DESC"
        ).fetchall()
        return [{"run_id": r[0], "agent_name": r[1], "created_at": r[2]} for r in rows]

    # ── branches ────────────────────────────────────────────────────────────

    def save_branch(
        self,
        *,
        parent_run_id: str,
        divergence_step: int,
        delta_tape: Tape,
        mutation_desc: str = "",
        created_at: str = "",
    ) -> str:
        bid = uuid.uuid4().hex[:12]
        blob = delta_tape.to_bytes()
        with self._write_lock:
            self._con.execute("BEGIN IMMEDIATE")
            try:
                self._con.execute(
                    """INSERT INTO branches
                       (branch_id, parent_run_id, divergence_step, delta_tape_bytes,
                        mutation_desc, created_at)
                       VALUES(?,?,?,?,?,?)""",
                    (bid, parent_run_id, divergence_step, blob, mutation_desc, created_at),
                )
                self._con.execute("COMMIT")
            except BaseException:
                self._con.execute("ROLLBACK")
                raise
        return bid

    def load_branch(self, branch_id: str) -> dict:
        row = self._con.execute(
            """SELECT branch_id, parent_run_id, divergence_step,
                      delta_tape_bytes, mutation_desc, created_at
               FROM branches WHERE branch_id=?""",
            (branch_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"branch_id {branch_id!r} not found")
        return {
            "branch_id": row[0],
            "parent_run_id": row[1],
            "divergence_step": row[2],
            "delta_tape": Tape.from_bytes(bytes(row[3])),
            "mutation_desc": row[4],
            "created_at": row[5],
        }

    def list_branches(self, parent_run_id: str) -> list[dict]:
        rows = self._con.execute(
            """SELECT branch_id, divergence_step, mutation_desc, created_at
               FROM branches WHERE parent_run_id=? ORDER BY created_at DESC""",
            (parent_run_id,),
        ).fetchall()
        return [
            {"branch_id": r[0], "divergence_step": r[1], "mutation_desc": r[2], "created_at": r[3]}
            for r in rows
        ]

    def close(self) -> None:
        self._con.close()
