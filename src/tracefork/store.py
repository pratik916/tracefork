"""TapeStore — SQLite-backed persistence for tapes and branch metadata.

Schema:
  tapes   (run_id TEXT PK, agent_name TEXT, tape_bytes BLOB, created_at TEXT)
  branches(branch_id TEXT PK, parent_run_id TEXT, divergence_step INT,
           delta_tape_bytes BLOB, mutation_desc TEXT, created_at TEXT)
  tapes_archived   (tapes + archived_at TEXT) — `prune()`'s soft-archive target
  branches_archived(branches + archived_at TEXT, no FK — see `prune()`)

`prune()` is a soft-archive-only retention pass (git gc / borg prune's
mark-and-sweep-with-soft-archive discipline): a pruned row moves from a live
table to its ``_archived`` twin and stays queryable there forever. There is
no hard-delete anywhere in this module — reclaiming that space is a
deliberately separate, out-of-scope, higher-risk step.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

from .tape import Tape, open_sqlite


class TapeConflictError(RuntimeError):
    """Raised by ``save_tape`` when a ``run_id`` is reused with content whose
    ``Tape.digest()`` differs from what's already stored, and ``overwrite`` was
    not set. Pass ``overwrite=True`` to replace the stored tape explicitly.
    """


@runtime_checkable
class StorageBackend(Protocol):
    """The persistence interface ``TapeStore`` (SQLite) already satisfies.

    Naming this seam lets a filesystem, object-store (S3/GCS), or other
    backend drop in later without touching any caller that only depends on
    this surface (``cli.py``, ``server.py``, ``fork.py``, ``blame.py``).
    ``TapeStore`` stays the default, unchanged implementation — nothing here
    alters its behavior.
    """

    def save_tape(
        self,
        tape: Tape,
        *,
        run_id: str | None = None,
        created_at: str = "",
        overwrite: bool = False,
    ) -> str: ...

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

-- Soft-archive targets for `TapeStore.prune()` — a pruned row is moved here,
-- never hard-deleted (mirrors git gc / borg prune's mark-and-sweep-with-
-- soft-archive discipline). No FOREIGN KEY back to the live tables: an
-- archived row must stay queryable even after its live counterpart, and any
-- of its own live relations, are long gone.
CREATE TABLE IF NOT EXISTS tapes_archived (
    run_id       TEXT PRIMARY KEY,
    agent_name   TEXT NOT NULL,
    tape_bytes   BLOB NOT NULL,
    created_at   TEXT NOT NULL,
    archived_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS branches_archived (
    branch_id         TEXT PRIMARY KEY,
    parent_run_id     TEXT NOT NULL,
    divergence_step   INTEGER NOT NULL,
    delta_tape_bytes  BLOB NOT NULL,
    mutation_desc     TEXT NOT NULL DEFAULT '',
    created_at        TEXT NOT NULL,
    archived_at       TEXT NOT NULL
);
"""


@dataclass
class PruneReport:
    """Outcome of a :meth:`TapeStore.prune` call.

    ``tapes_archived``/``branches_archived`` are the run_ids/branch_ids that
    matched the prune filters — populated identically whether or not
    ``dry_run`` was set, so a caller can preview the exact candidate set
    before committing to it.
    """

    dry_run: bool
    tapes_archived: list[str] = field(default_factory=list)
    branches_archived: list[str] = field(default_factory=list)


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

    def save_tape(
        self,
        tape: Tape,
        *,
        run_id: str | None = None,
        created_at: str = "",
        overwrite: bool = False,
    ) -> str:
        """Persist ``tape`` under ``run_id`` (a fresh id if omitted).

        Install-or-verify-same-content, the same model git uses for its object
        store: reusing a ``run_id`` whose stored content hashes identically
        (via ``Tape.digest()``, never raw bytes — see module docstring on
        ``TAPE_FORMAT_VERSION``) is an idempotent no-op that returns the same
        ``run_id``. Reusing a ``run_id`` with genuinely different content raises
        :class:`TapeConflictError` instead of silently clobbering the prior
        tape, unless ``overwrite=True`` is passed to replace it explicitly.
        """
        rid = run_id or uuid.uuid4().hex[:12]
        blob = tape.to_bytes()
        with self._write_lock:
            self._con.execute("BEGIN IMMEDIATE")
            try:
                row = self._con.execute(
                    "SELECT tape_bytes FROM tapes WHERE run_id=?", (rid,)
                ).fetchone()
                if row is not None and not overwrite:
                    existing_digest = Tape.from_bytes(bytes(row[0])).digest()
                    if existing_digest != tape.digest():
                        raise TapeConflictError(
                            f"run_id {rid!r} already stored with different content "
                            "(digest mismatch); pass overwrite=True to replace it"
                        )
                    self._con.execute("COMMIT")
                    return rid
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

    # ── prune (soft-archive, never hard-delete) ─────────────────────────────

    def prune(
        self,
        *,
        older_than_iso: str | None = None,
        run_ids: list[str] | None = None,
        dry_run: bool = False,
    ) -> PruneReport:
        """Archive tapes (and their branches) — never hard-delete.

        A tape qualifies for pruning if EITHER filter matches it: its
        ``created_at`` sorts before ``older_than_iso`` (plain lexical ISO-8601
        comparison, exclusive of the cutoff), or its ``run_id`` is in
        ``run_ids``. Passing neither filter matches nothing — pruning is
        opt-in per call, never "everything" by accident.

        For each qualifying tape, inside one ``BEGIN IMMEDIATE`` transaction,
        its branches are copied into ``branches_archived`` and deleted from
        the live ``branches`` table FIRST, then the tape row is copied into
        ``tapes_archived`` and deleted from the live ``tapes`` table — the
        order ``branches`` live-table's ``FOREIGN KEY(parent_run_id)
        REFERENCES tapes(run_id)`` requires under ``foreign_keys=ON``.
        Archived rows are never deleted by this method or any other: they
        stay queryable in ``tapes_archived``/``branches_archived``
        indefinitely (reclaiming that space is a distinct, higher-risk,
        deliberately out-of-scope step — see the module docstring's git
        gc / borg prune comparison).

        ``dry_run=True`` computes the exact candidate set and returns it
        without opening a write transaction — zero mutation, safe to call
        speculatively. ``save_tape``/``save_branch``/``load_tape``/
        ``load_branch``/``list_runs``/``list_branches`` are all unaffected by
        this method's existence: a pruned ``run_id`` simply stops appearing
        in ``list_runs()``, and ``load_tape``/``load_branch`` raise the same
        ``KeyError`` they already raise for any unknown id.
        """
        run_id_filter = set(run_ids or [])
        rows = self._con.execute("SELECT run_id, created_at FROM tapes").fetchall()
        candidates = [
            run_id
            for run_id, created_at in rows
            if (older_than_iso is not None and created_at < older_than_iso)
            or run_id in run_id_filter
        ]

        branch_ids: list[str] = []
        for run_id in candidates:
            branch_ids.extend(
                row[0]
                for row in self._con.execute(
                    "SELECT branch_id FROM branches WHERE parent_run_id=?", (run_id,)
                ).fetchall()
            )

        if dry_run or not candidates:
            return PruneReport(
                dry_run=dry_run, tapes_archived=candidates, branches_archived=branch_ids
            )

        archived_at = datetime.now(UTC).isoformat()
        with self._write_lock:
            self._con.execute("BEGIN IMMEDIATE")
            try:
                for run_id in candidates:
                    self._con.execute(
                        """INSERT INTO branches_archived
                           (branch_id, parent_run_id, divergence_step, delta_tape_bytes,
                            mutation_desc, created_at, archived_at)
                           SELECT branch_id, parent_run_id, divergence_step, delta_tape_bytes,
                                  mutation_desc, created_at, ?
                           FROM branches WHERE parent_run_id=?""",
                        (archived_at, run_id),
                    )
                    self._con.execute("DELETE FROM branches WHERE parent_run_id=?", (run_id,))
                    self._con.execute(
                        """INSERT INTO tapes_archived
                           (run_id, agent_name, tape_bytes, created_at, archived_at)
                           SELECT run_id, agent_name, tape_bytes, created_at, ?
                           FROM tapes WHERE run_id=?""",
                        (archived_at, run_id),
                    )
                    self._con.execute("DELETE FROM tapes WHERE run_id=?", (run_id,))
                self._con.execute("COMMIT")
            except BaseException:
                self._con.execute("ROLLBACK")
                raise

        return PruneReport(dry_run=False, tapes_archived=candidates, branches_archived=branch_ids)

    def close(self) -> None:
        self._con.close()
