"""TapeStore — SQLite-backed persistence for tapes, branch metadata, and the
persistent causal graph.

Schema:
  tapes   (run_id TEXT PK, agent_name TEXT, tape_bytes BLOB, created_at TEXT)
  branches(branch_id TEXT PK, parent_run_id TEXT, divergence_step INT,
           delta_tape_bytes BLOB, mutation_desc TEXT, created_at TEXT,
           branch_digest TEXT)
  tapes_archived   (tapes + archived_at TEXT) — `prune()`'s soft-archive target
  branches_archived(branches + archived_at TEXT, no FK — see `prune()`)
  causal_edges(edge_id TEXT PK, run_id TEXT, step_index INT, method TEXT,
               flip_rate REAL, ci_lo/ci_hi REAL, ci_method TEXT,
               p_value/q_value REAL, responsible INT, necessity/sufficiency INT,
               shapley_value REAL, created_at TEXT)

`prune()` is a soft-archive-only retention pass (git gc / borg prune's
mark-and-sweep-with-soft-archive discipline): a pruned row moves from a live
table to its ``_archived`` twin and stays queryable there forever. There is
no hard-delete anywhere in this module — reclaiming that space is a
deliberately separate, out-of-scope, higher-risk step.

``branch_digest`` is ``fork.py``'s content-addressed ``Branch.branch_digest``
(Merkle-DAG identity: parent tape digest + delta tape digest + intervened
steps, folded into one sha256) — Branch/store-level metadata only, never fed
into ``Tape.digest()``. A pre-existing ``store.db`` built before this column
existed is migrated in place via a guarded ``PRAGMA table_info``-gated ``ALTER
TABLE`` in ``TapeStore.__init__`` (never a destructive ``CREATE TABLE``
assumption): a fresh database gets the column straight from ``CREATE TABLE``,
an old one gets it appended with a ``''`` default, no row lost either way.
``find_branch_by_digest`` resolves the branch with a given digest;
``branches_forked_from`` answers the inverse-citation query — which branches
used a given digest's branch as THEIR OWN parent, once that branch's
``delta_tape`` has itself been promoted to a tape via ``save_tape(delta_tape,
run_id=branch_id)`` (same promotion convention ``causal_closure`` already
relies on) — enabling fork-of-fork chains as a plain reachability walk.

``parent_tape_digest``/``divergence_exchange_digest`` are ``fork.py``'s
citable fork-point metadata (``Branch.parent_tape_digest`` — the parent
tape's own ``digest()`` at fork time — and ``Branch.divergence_exchange_digest``
— sha256 of the exact request+response bytes at the first divergence point),
migrated onto the ``branches`` table in the SAME guarded ``ALTER TABLE`` pass
as ``branch_digest`` (see ``_migrate_branch_metadata_columns``). ``load_branch`` is
the re-verification point: it recomputes the CURRENT digest of the tape
referenced by ``parent_run_id`` and compares it against the stored
``parent_tape_digest`` (skipped when that column is ``''`` — a legacy branch,
or one created via ``cli.py``'s fork command, which does not pass it, has
nothing to re-verify against). A mismatch is a hard error
(:class:`ForkPointDriftError`), never a silently logged-and-continued
divergence — the retrospective, read-time complement to ``save_tape``'s
write-time CAS guard.

``intervened_steps_json`` is ``fork.py``'s ``Branch.intervened_steps`` (the
full set of force-set step indices a coalition/rebase fork touched, not just
``divergence_step`` — the coalition's first/lowest step) JSON-serialized as a
list (SQLite has no native tuple/array type); migrated onto ``branches`` in
the SAME guarded ``ALTER TABLE`` pass as the other branch metadata columns
(see ``_migrate_branch_metadata_columns``). ``save_branch`` gains a matching
optional ``intervened_steps`` parameter (default ``()``, every existing
caller unaffected); ``load_branch``/``find_branch_by_digest`` decode it back
to a ``tuple[int, ...]`` (``()`` for a legacy/omitted value) — the prerequisite
``ForkEngine.rebase`` (see ``fork.py``) needs to know a branch's FULL forced
coalition, not merely its first divergence step, when re-forcing those steps
against a new parent tape.

``confinement_tier`` is ``fork.py``'s ``Branch.confinement_tier``
(``compute_confinement_tier``) — an axis orthogonal to a tape's own
``boundary`` tiers (see ``constants.py``): how confined the re-executed
agent was during a fork's tail-record phase, not how the tape itself was
recorded. Branch/store-level metadata only, never fed into ``Tape.digest()``;
migrated onto ``branches`` in the SAME guarded ``ALTER TABLE`` pass as the
other branch metadata columns (see ``_migrate_branch_metadata_columns``).
``save_branch`` gains a matching optional ``confinement_tier`` parameter
(default ``''``, every existing caller unaffected); ``load_branch``/
``find_branch_by_digest``/``list_branches`` return it verbatim (``''`` for a
legacy/omitted value).

``causal_edges`` persists every blame/Shapley result computed by ``blame.py``
instead of discarding it once the CLI/caller has printed or JSON-dumped it —
a causal graph strictly stronger than a bare caused_by DAG, since every edge
carries its Wilson-CI flip-rate (or Shapley value) and BH-FDR ``q_value``.
It deliberately has no ``FOREIGN KEY`` back to ``tapes``: unlike ``branches``,
which `prune()` must archive in FK order, causal edges are independent
metadata `prune()` need not know about (see ``prune()``'s own docstring —
this module makes no attempt to keep the two in sync).

``sessions``/``spawn_edges`` model cross-AGENT orchestration/delegation
lineage — a SEPARATE graph from ``branches`` (the fork/counterfactual DAG)
and from ``Tape.async_batches`` (a single agent's own per-run asyncio
fan-out; unrelated, never conflate the two). A session is rooted at one
``run_id`` (``sessions.root_run_id``); ``spawn_edges`` records each
``parent_run_id -> child_run_id`` delegation within that session, along with
an optional free-text ``spawn_reason``. Modeling delegation as its OWN graph
(rather than collapsing it into ``causal_edges``/``branches`` or a single
trace id) follows 2026 delegated-execution observability practice: an
execution/causal graph and an authority/delegation graph diverge under async
fan-out and re-delegation, which ``fork.py``/``blame.py`` do constantly.
``spawn_edges.parent_run_id``/``child_run_id`` (and ``sessions.root_run_id``)
carry a ``FOREIGN KEY`` back to ``tapes(run_id)`` — a spawn edge or session
may only reference a tape that's actually stored, enforced by SQLite
(``foreign_keys=ON``, see ``open_sqlite``), never a soft/unchecked reference.
:class:`SessionStore` is a NEW, separate ``runtime_checkable`` Protocol
(``TapeStore`` satisfies it) rather than folding these methods into
:class:`StorageBackend` — so no third-party ``StorageBackend`` implementer
is broken by this addition; ``StorageBackend`` itself is completely
unchanged. See :meth:`TapeStore.create_session`, :meth:`TapeStore.
add_spawn_edge`, :meth:`TapeStore.session_tapes` (a BFS over the spawn
graph reachable from a session's root), and :meth:`TapeStore.spawn_children`/
:meth:`TapeStore.spawn_parent`.
"""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from .tape import Tape, open_sqlite

if TYPE_CHECKING:
    from .blame import BlameReport, ShapleyReport


class TapeConflictError(RuntimeError):
    """Raised by ``save_tape`` when a ``run_id`` is reused with content whose
    ``Tape.digest()`` differs from what's already stored, and ``overwrite`` was
    not set. Pass ``overwrite=True`` to replace the stored tape explicitly.
    """


def _encode_intervened_steps(steps: tuple[int, ...]) -> str:
    """JSON-serialize a branch's forced step indices for the
    ``branches.intervened_steps_json`` column (SQLite has no tuple/array
    type)."""
    return json.dumps(list(steps))


def _decode_intervened_steps(raw: str) -> tuple[int, ...]:
    """Inverse of :func:`_encode_intervened_steps`. An empty/falsy value —
    the ``''`` a defensively-blank column would hold, though the guarded
    migration below always defaults to ``'[]'`` — decodes to ``()``, the same
    "nothing recorded" meaning ``branch_digest``/``parent_tape_digest``
    already use for their own ``''`` default."""
    if not raw:
        return ()
    return tuple(json.loads(raw))


class ForkPointDriftError(RuntimeError):
    """Raised by ``load_branch`` when a branch's stored ``parent_tape_digest``
    (the parent tape's ``Tape.digest()`` at fork time) no longer matches the
    parent tape's CURRENT digest — i.e. the fork point this branch cites has
    silently drifted since the fork was made (e.g. the parent tape row was
    overwritten out-of-band). Hard error, never silently logged and
    continued: a branch whose cited ancestry no longer matches what's
    actually stored is not safe to trust. Only raised when the branch
    recorded a non-empty ``parent_tape_digest`` — a legacy branch (default
    ``''``, e.g. one created via ``cli.py``'s fork command, which does not
    pass this field) has nothing to re-verify against and is never checked.
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
        branch_id: str | None = None,
        branch_digest: str = "",
        parent_tape_digest: str = "",
        divergence_exchange_digest: str = "",
        intervened_steps: tuple[int, ...] = (),
        confinement_tier: str = "",
    ) -> str: ...

    def load_branch(self, branch_id: str) -> dict: ...

    def list_branches(self, parent_run_id: str) -> list[dict]: ...

    def save_blame_report(
        self, run_id: str, report: BlameReport, *, created_at: str = ""
    ) -> list[str]: ...

    def save_shapley_report(
        self, run_id: str, report: ShapleyReport, *, created_at: str = ""
    ) -> list[str]: ...

    def causal_edges_for_run(self, run_id: str) -> list[dict]: ...

    def cited_by(self, run_id: str, step_index: int) -> list[str]: ...

    def causal_closure(self, run_id: str) -> list[dict]: ...

    def close(self) -> None: ...


@runtime_checkable
class SessionStore(Protocol):
    """The orchestration-session / spawn-lineage persistence interface
    ``TapeStore`` also satisfies — kept SEPARATE from :class:`StorageBackend`
    (additive-only: naming a new seam here, rather than growing that
    Protocol) so an existing third-party ``StorageBackend`` implementation
    keeps working unmodified. See the module docstring for why spawn
    lineage is its own graph, distinct from ``branches``/``causal_edges``
    and from ``Tape.async_batches``.
    """

    def create_session(
        self, *, root_run_id: str, session_id: str | None = None, created_at: str = ""
    ) -> str: ...

    def get_session(self, session_id: str) -> dict: ...

    def add_spawn_edge(
        self,
        *,
        session_id: str,
        parent_run_id: str,
        child_run_id: str,
        spawn_reason: str = "",
        edge_id: str | None = None,
        created_at: str = "",
    ) -> str: ...

    def session_tapes(self, session_id: str) -> list[str]: ...

    def spawn_children(self, run_id: str) -> list[str]: ...

    def spawn_parent(self, run_id: str) -> str | None: ...


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
    branch_digest     TEXT NOT NULL DEFAULT '',
    parent_tape_digest          TEXT NOT NULL DEFAULT '',
    divergence_exchange_digest  TEXT NOT NULL DEFAULT '',
    intervened_steps_json       TEXT NOT NULL DEFAULT '[]',
    confinement_tier            TEXT NOT NULL DEFAULT '',
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

-- Persistent causal graph: one row per (run_id, step_index, method) blame or
-- Shapley result. No FOREIGN KEY to `tapes` — see module docstring.
CREATE TABLE IF NOT EXISTS causal_edges (
    edge_id       TEXT PRIMARY KEY,
    run_id        TEXT NOT NULL,
    step_index    INTEGER NOT NULL,
    method        TEXT NOT NULL,
    flip_rate     REAL,
    ci_lo         REAL,
    ci_hi         REAL,
    ci_method     TEXT,
    p_value       REAL,
    q_value       REAL,
    responsible   INTEGER,
    necessity     INTEGER,
    sufficiency   INTEGER,
    shapley_value REAL,
    created_at    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_causal_edges_run ON causal_edges(run_id);

-- Orchestration/spawn-lineage graph — cross-AGENT delegation, distinct from
-- `branches` (the fork/counterfactual DAG) and from `Tape.async_batches`
-- (per-agent asyncio fan-out). See module docstring.
CREATE TABLE IF NOT EXISTS sessions (
    session_id   TEXT PRIMARY KEY,
    root_run_id  TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    FOREIGN KEY(root_run_id) REFERENCES tapes(run_id)
);

CREATE TABLE IF NOT EXISTS spawn_edges (
    edge_id        TEXT PRIMARY KEY,
    session_id     TEXT NOT NULL,
    parent_run_id  TEXT NOT NULL,
    child_run_id   TEXT NOT NULL,
    spawn_reason   TEXT NOT NULL DEFAULT '',
    created_at     TEXT NOT NULL,
    FOREIGN KEY(session_id) REFERENCES sessions(session_id),
    FOREIGN KEY(parent_run_id) REFERENCES tapes(run_id),
    FOREIGN KEY(child_run_id) REFERENCES tapes(run_id)
);

CREATE INDEX IF NOT EXISTS idx_spawn_edges_session ON spawn_edges(session_id);
CREATE INDEX IF NOT EXISTS idx_spawn_edges_parent ON spawn_edges(parent_run_id);
CREATE INDEX IF NOT EXISTS idx_spawn_edges_child ON spawn_edges(child_run_id);
"""

_EDGE_COLUMNS = (
    "edge_id, run_id, step_index, method, flip_rate, ci_lo, ci_hi, ci_method, "
    "p_value, q_value, responsible, necessity, sufficiency, shapley_value, created_at"
)

_INSERT_EDGE_SQL = (
    f"INSERT INTO causal_edges({_EDGE_COLUMNS}) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
)


def _edge_row_to_dict(row: tuple) -> dict[str, Any]:
    """Map a raw ``causal_edges`` row (see ``_EDGE_COLUMNS`` order) to a dict,
    restoring the nullable INTEGER boolean columns to ``bool | None``."""
    return {
        "edge_id": row[0],
        "run_id": row[1],
        "step_index": row[2],
        "method": row[3],
        "flip_rate": row[4],
        "ci_lo": row[5],
        "ci_hi": row[6],
        "ci_method": row[7],
        "p_value": row[8],
        "q_value": row[9],
        "responsible": None if row[10] is None else bool(row[10]),
        "necessity": None if row[11] is None else bool(row[11]),
        "sufficiency": None if row[12] is None else bool(row[12]),
        "shapley_value": row[13],
        "created_at": row[14],
    }


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
        self._migrate_branch_metadata_columns()

    def _migrate_branch_metadata_columns(self) -> None:
        """Guarded ``ALTER TABLE`` for a ``store.db`` built before
        ``branch_digest``/``parent_tape_digest``/``divergence_exchange_digest``/
        ``intervened_steps_json``/``confinement_tier`` existed on the
        ``branches`` table.

        ``CREATE TABLE IF NOT EXISTS`` (in ``_DDL``, above) only ever creates
        these columns on a BRAND NEW database — it is a no-op against an
        existing ``branches`` table, so a pre-existing store.db needs
        explicit ``ADD COLUMN``s here. All five columns share ONE
        ``PRAGMA table_info`` read (not five separate migration passes) so
        an old database is altered once, atomically enough for SQLite's
        autocommit DDL, with each ``ADD COLUMN`` independently guarded —
        never destructive: no row is touched, only new columns with a
        default (``''``, or ``'[]'`` for ``intervened_steps_json``) are
        appended. The index is created here too (after the columns are
        guaranteed to exist) rather than inside ``_DDL``, since ``CREATE
        INDEX`` on a column that doesn't exist yet would raise on a
        genuinely old database.
        """
        cols = {row[1] for row in self._con.execute("PRAGMA table_info(branches)").fetchall()}
        if "branch_digest" not in cols:
            self._con.execute(
                "ALTER TABLE branches ADD COLUMN branch_digest TEXT NOT NULL DEFAULT ''"
            )
        if "parent_tape_digest" not in cols:
            self._con.execute(
                "ALTER TABLE branches ADD COLUMN parent_tape_digest TEXT NOT NULL DEFAULT ''"
            )
        if "divergence_exchange_digest" not in cols:
            self._con.execute(
                "ALTER TABLE branches ADD COLUMN divergence_exchange_digest "
                "TEXT NOT NULL DEFAULT ''"
            )
        if "intervened_steps_json" not in cols:
            self._con.execute(
                "ALTER TABLE branches ADD COLUMN intervened_steps_json TEXT NOT NULL DEFAULT '[]'"
            )
        if "confinement_tier" not in cols:
            self._con.execute(
                "ALTER TABLE branches ADD COLUMN confinement_tier TEXT NOT NULL DEFAULT ''"
            )
        self._con.execute(
            "CREATE INDEX IF NOT EXISTS idx_branches_branch_digest ON branches(branch_digest)"
        )

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

    def stored_digest(self, run_id: str) -> str | None:
        """The ``tapes.digest`` column's value for ``run_id``, or ``None`` if
        that column doesn't exist in this schema yet (a future bead's
        addition) or ``run_id`` isn't found. Gates on ``PRAGMA table_info``
        rather than assuming the column, so callers like ``fsck.py`` can
        treat a stronger digest-recompute check as opportunistic — never a
        hard dependency on a column this schema may not have. ``TapeStore``-
        only for now (see ``prune()``'s docstring for the same precedent of
        not extending ``StorageBackend`` with every maintenance/diagnostic
        helper)."""
        cols = {row[1] for row in self._con.execute("PRAGMA table_info(tapes)").fetchall()}
        if "digest" not in cols:
            return None
        row = self._con.execute("SELECT digest FROM tapes WHERE run_id=?", (run_id,)).fetchone()
        return None if row is None else row[0]

    # ── branches ────────────────────────────────────────────────────────────

    def save_branch(
        self,
        *,
        parent_run_id: str,
        divergence_step: int,
        delta_tape: Tape,
        mutation_desc: str = "",
        created_at: str = "",
        branch_id: str | None = None,
        branch_digest: str = "",
        parent_tape_digest: str = "",
        divergence_exchange_digest: str = "",
        intervened_steps: tuple[int, ...] = (),
        confinement_tier: str = "",
    ) -> str:
        """Persist ``delta_tape`` as a new branch of ``parent_run_id`` (a fresh
        ``branch_id`` if omitted).

        ``branch_id`` (optional) lets a caller reuse a specific id instead of
        generating a fresh uuid — e.g. ``bundle.py``'s ``import_bundle``,
        which must preserve a branch's id across stores. When passed, the same
        install-or-verify-same-content CAS guard as :meth:`save_tape` applies:
        reusing a ``branch_id`` whose stored ``delta_tape`` content is
        byte-identical (via ``Tape.digest()``) is an idempotent no-op;
        genuinely different content raises :class:`TapeConflictError`. Every
        existing caller omits ``branch_id`` and keeps today's behavior exactly
        (a fresh uuid, no collision possible, no SELECT before the INSERT).

        ``branch_digest`` (optional, default ``''``) is ``fork.py``'s
        content-addressed ``Branch.branch_digest`` — Branch/store-level
        metadata only, never fed into ``Tape.digest()``. Every existing caller
        that omits it keeps storing ``''`` exactly as before this parameter
        existed. See :meth:`find_branch_by_digest`/:meth:`branches_forked_from`.

        ``parent_tape_digest``/``divergence_exchange_digest`` (optional,
        default ``''``) are ``fork.py``'s citable fork-point metadata —
        ``Branch.parent_tape_digest`` (the parent tape's own ``digest()`` at
        fork time) and ``Branch.divergence_exchange_digest`` (sha256 of the
        exact request+response bytes at the first divergence point). Every
        existing caller that omits them keeps storing ``''`` exactly as
        before these parameters existed. See :meth:`load_branch`, the
        re-verification point for ``parent_tape_digest``.

        ``intervened_steps`` (optional, default ``()``) is ``fork.py``'s
        ``Branch.intervened_steps`` — the full set of force-set step indices,
        not just ``divergence_step`` (the coalition's first/lowest step).
        Every existing caller that omits it keeps storing ``()`` exactly as
        before this parameter existed. See :meth:`load_branch`.

        ``confinement_tier`` (optional, default ``''``) is ``fork.py``'s
        ``Branch.confinement_tier`` — an axis orthogonal to a tape's own
        ``boundary`` (see ``constants.py``), Branch/store-level metadata
        only, never fed into ``Tape.digest()``. Every existing caller that
        omits it keeps storing ``''`` exactly as before this parameter
        existed.
        """
        bid = branch_id or uuid.uuid4().hex[:12]
        blob = delta_tape.to_bytes()
        intervened_steps_json = _encode_intervened_steps(intervened_steps)
        with self._write_lock:
            self._con.execute("BEGIN IMMEDIATE")
            try:
                if branch_id is not None:
                    row = self._con.execute(
                        "SELECT delta_tape_bytes FROM branches WHERE branch_id=?", (bid,)
                    ).fetchone()
                    if row is not None:
                        existing_digest = Tape.from_bytes(bytes(row[0])).digest()
                        if existing_digest != delta_tape.digest():
                            raise TapeConflictError(
                                f"branch_id {bid!r} already stored with different content "
                                "(digest mismatch)"
                            )
                        self._con.execute("COMMIT")
                        return bid
                self._con.execute(
                    """INSERT INTO branches
                       (branch_id, parent_run_id, divergence_step, delta_tape_bytes,
                        mutation_desc, created_at, branch_digest, parent_tape_digest,
                        divergence_exchange_digest, intervened_steps_json, confinement_tier)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        bid,
                        parent_run_id,
                        divergence_step,
                        blob,
                        mutation_desc,
                        created_at,
                        branch_digest,
                        parent_tape_digest,
                        divergence_exchange_digest,
                        intervened_steps_json,
                        confinement_tier,
                    ),
                )
                self._con.execute("COMMIT")
            except BaseException:
                self._con.execute("ROLLBACK")
                raise
        return bid

    def load_branch(self, branch_id: str) -> dict:
        """Load ``branch_id`` and re-verify its cited fork point.

        This is the re-verification point for ``parent_tape_digest``: when a
        branch recorded a non-empty value (i.e. it wasn't produced by a
        legacy caller that omits it), the tape currently stored under
        ``parent_run_id`` is loaded and its CURRENT ``digest()`` compared
        against the stored value. A mismatch — the cited parent tape has
        changed since this branch was forked — raises
        :class:`ForkPointDriftError` naming the drifted ``parent_run_id``; it
        is never silently logged and continued. A branch with an empty
        ``parent_tape_digest`` has nothing to re-verify against and skips the
        check entirely.
        """
        row = self._con.execute(
            """SELECT branch_id, parent_run_id, divergence_step,
                      delta_tape_bytes, mutation_desc, created_at, branch_digest,
                      parent_tape_digest, divergence_exchange_digest, intervened_steps_json,
                      confinement_tier
               FROM branches WHERE branch_id=?""",
            (branch_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"branch_id {branch_id!r} not found")
        parent_run_id = row[1]
        stored_parent_tape_digest = row[7]
        if stored_parent_tape_digest:
            current_parent_digest = self.load_tape(parent_run_id).digest()
            if current_parent_digest != stored_parent_tape_digest:
                raise ForkPointDriftError(
                    f"parent tape {parent_run_id!r} has drifted since branch "
                    f"{branch_id!r} was forked (recorded digest "
                    f"{stored_parent_tape_digest[:12]}, current "
                    f"{current_parent_digest[:12]})"
                )
        return {
            "branch_id": row[0],
            "parent_run_id": row[1],
            "divergence_step": row[2],
            "delta_tape": Tape.from_bytes(bytes(row[3])),
            "mutation_desc": row[4],
            "created_at": row[5],
            "branch_digest": row[6],
            "parent_tape_digest": row[7],
            "divergence_exchange_digest": row[8],
            "intervened_steps": _decode_intervened_steps(row[9]),
            "confinement_tier": row[10],
        }

    def find_branch_by_digest(self, branch_digest: str) -> dict | None:
        """The same shape :meth:`load_branch` returns for the branch whose
        ``branch_digest`` matches, or ``None`` if no branch has that digest
        (an empty ``branch_digest`` never matches — old, pre-migration rows
        all share ``''`` and must not collide with each other or a query for
        ``''``).

        Unlike :meth:`load_branch`, this is a plain lookup — it does NOT
        re-verify ``parent_tape_digest`` against the parent tape's current
        digest (:meth:`load_branch` is the sole re-verification point)."""
        if not branch_digest:
            return None
        row = self._con.execute(
            """SELECT branch_id, parent_run_id, divergence_step,
                      delta_tape_bytes, mutation_desc, created_at, branch_digest,
                      parent_tape_digest, divergence_exchange_digest, intervened_steps_json,
                      confinement_tier
               FROM branches WHERE branch_digest=?""",
            (branch_digest,),
        ).fetchone()
        if row is None:
            return None
        return {
            "branch_id": row[0],
            "parent_run_id": row[1],
            "divergence_step": row[2],
            "delta_tape": Tape.from_bytes(bytes(row[3])),
            "mutation_desc": row[4],
            "created_at": row[5],
            "branch_digest": row[6],
            "parent_tape_digest": row[7],
            "divergence_exchange_digest": row[8],
            "intervened_steps": _decode_intervened_steps(row[9]),
            "confinement_tier": row[10],
        }

    def branches_forked_from(self, branch_digest: str) -> list[str]:
        """Inverse-citation query: branch ids that used the branch identified
        by ``branch_digest`` AS THEIR OWN PARENT — i.e. fork-of-fork chains.

        Only meaningful once that branch's ``delta_tape`` has itself been
        persisted as a tape via ``save_tape(delta_tape, run_id=branch_id)``
        (the same "promote a branch to a re-forkable/re-blamable run" convention
        :meth:`causal_closure` already relies on) — a branch never promoted
        this way is simply never anyone's ``parent_run_id``, so it surfaces no
        results here, not an error. Returns ``[]`` for an unknown or empty
        ``branch_digest``."""
        if not branch_digest:
            return []
        row = self._con.execute(
            "SELECT branch_id FROM branches WHERE branch_digest=?", (branch_digest,)
        ).fetchone()
        if row is None:
            return []
        parent_run_id = row[0]
        rows = self._con.execute(
            "SELECT branch_id FROM branches WHERE parent_run_id=? ORDER BY created_at DESC",
            (parent_run_id,),
        ).fetchall()
        return [r[0] for r in rows]

    def list_branches(self, parent_run_id: str) -> list[dict]:
        """Summary rows for every branch of ``parent_run_id`` — ``branch_id``,
        ``divergence_step``, ``mutation_desc``, ``created_at``,
        ``branch_digest``, and ``confinement_tier`` — with no ``delta_tape``
        decode, unlike :meth:`load_branch`. This is the shape ``report.py``'s
        fork-tree panel (tracefork-bge.15) embeds directly: enough to
        sort/label every branch edge without a per-branch ``load_branch``
        round trip.
        """
        rows = self._con.execute(
            """SELECT branch_id, divergence_step, mutation_desc, created_at, branch_digest,
                      confinement_tier
               FROM branches WHERE parent_run_id=? ORDER BY created_at DESC""",
            (parent_run_id,),
        ).fetchall()
        return [
            {
                "branch_id": r[0],
                "divergence_step": r[1],
                "mutation_desc": r[2],
                "created_at": r[3],
                "branch_digest": r[4],
                "confinement_tier": r[5],
            }
            for r in rows
        ]

    def all_branch_parents(self) -> list[tuple[str, str]]:
        """``(branch_id, parent_run_id)`` for every branch row in the live
        ``branches`` table, regardless of whether ``parent_run_id`` still
        resolves to a live tape. Distinct from ``list_branches(parent_run_id)``,
        which only ever returns branches under a known-live parent — a branch
        whose parent tape row was force-deleted directly (e.g. with
        ``foreign_keys=OFF``, bypassing the FK this schema normally enforces)
        would never surface through that method. Enables ``fsck.py``'s
        orphaned-parent check. ``TapeStore``-only for now, same precedent as
        ``stored_digest``/``prune()``."""
        rows = self._con.execute("SELECT branch_id, parent_run_id FROM branches").fetchall()
        return [(r[0], r[1]) for r in rows]

    # ── raw row access (bundle.py's byte-for-byte export/import) ───────────

    def raw_tape_row(self, run_id: str) -> tuple[str, str, bytes, str] | None:
        """The raw ``tapes`` row for ``run_id`` — ``(run_id, agent_name,
        tape_bytes, created_at)`` — exactly as stored, with zero decode or
        re-encode. Powers ``bundle.py``'s byte-for-byte ``export_bundle``
        (see that module's docstring); returns ``None`` if ``run_id`` isn't
        found. Same "``TapeStore``-only read helper" precedent as
        ``stored_digest``/``all_branch_parents``."""
        row = self._con.execute(
            "SELECT run_id, agent_name, tape_bytes, created_at FROM tapes WHERE run_id=?",
            (run_id,),
        ).fetchone()
        return None if row is None else (row[0], row[1], bytes(row[2]), row[3])

    def raw_branch_rows(self, parent_run_id: str) -> list[tuple[str, str, int, bytes, str, str]]:
        """The raw ``branches`` rows for every branch under ``parent_run_id``
        — ``(branch_id, parent_run_id, divergence_step, delta_tape_bytes,
        mutation_desc, created_at)`` — exactly as stored, with zero decode or
        re-encode. Powers ``bundle.py``'s byte-for-byte ``export_bundle``."""
        rows = self._con.execute(
            """SELECT branch_id, parent_run_id, divergence_step, delta_tape_bytes,
                      mutation_desc, created_at
               FROM branches WHERE parent_run_id=?""",
            (parent_run_id,),
        ).fetchall()
        return [(r[0], r[1], r[2], bytes(r[3]), r[4], r[5]) for r in rows]

    def install_raw_tape_row(self, row: tuple[str, str, bytes, str]) -> None:
        """Write a raw ``tapes`` row (as returned by :meth:`raw_tape_row`)
        verbatim — ``INSERT OR REPLACE``, no digest check, no CAS guard.

        For ``bundle.py``'s ``export_bundle`` writing into a FRESH bundle
        file only, where no prior content can exist to collide with; this is
        deliberately NOT a general-purpose write path (unlike
        :meth:`save_tape`) and must never be pointed at a live store another
        writer is using.
        """
        with self._write_lock:
            self._con.execute("BEGIN IMMEDIATE")
            try:
                self._con.execute(
                    "INSERT OR REPLACE INTO tapes(run_id, agent_name, tape_bytes, created_at) "
                    "VALUES(?,?,?,?)",
                    row,
                )
                self._con.execute("COMMIT")
            except BaseException:
                self._con.execute("ROLLBACK")
                raise

    def install_raw_branch_row(self, row: tuple[str, str, int, bytes, str, str]) -> None:
        """Write a raw ``branches`` row (as returned by :meth:`raw_branch_rows`)
        verbatim — ``INSERT OR REPLACE``, no digest check, no CAS guard. Same
        fresh-bundle-file-only caveat as :meth:`install_raw_tape_row`."""
        with self._write_lock:
            self._con.execute("BEGIN IMMEDIATE")
            try:
                self._con.execute(
                    """INSERT OR REPLACE INTO branches
                       (branch_id, parent_run_id, divergence_step, delta_tape_bytes,
                        mutation_desc, created_at)
                       VALUES(?,?,?,?,?,?)""",
                    row,
                )
                self._con.execute("COMMIT")
            except BaseException:
                self._con.execute("ROLLBACK")
                raise

    # ── causal graph (persistent blame/Shapley edges) ───────────────────────

    def save_blame_report(
        self, run_id: str, report: BlameReport, *, created_at: str = ""
    ) -> list[str]:
        """Persist every ``FlipRateResult`` in ``report`` as a ``causal_edges``
        row (``method="blame"``, ``edge_id=f"{run_id}:{step_index}:blame"``).

        Upsert-by-replace: any blame edges previously saved for ``run_id`` are
        deleted first, so re-calling this after a re-blame REPLACES the row
        set rather than accumulating stale steps alongside fresh ones.
        """
        ci_method = report.ci_method.value
        edge_ids: list[str] = []
        with self._write_lock:
            self._con.execute("BEGIN IMMEDIATE")
            try:
                self._con.execute(
                    "DELETE FROM causal_edges WHERE run_id=? AND method='blame'", (run_id,)
                )
                for r in report.results:
                    edge_id = f"{run_id}:{r.step_index}:blame"
                    edge_ids.append(edge_id)
                    self._con.execute(
                        _INSERT_EDGE_SQL,
                        (
                            edge_id,
                            run_id,
                            r.step_index,
                            "blame",
                            r.flip_rate,
                            r.ci_lo,
                            r.ci_hi,
                            ci_method,
                            r.p_value,
                            r.q_value,
                            int(r.responsible),
                            None,
                            None,
                            None,
                            created_at,
                        ),
                    )
                self._con.execute("COMMIT")
            except BaseException:
                self._con.execute("ROLLBACK")
                raise
        return edge_ids

    def save_shapley_report(
        self, run_id: str, report: ShapleyReport, *, created_at: str = ""
    ) -> list[str]:
        """Persist every ``ShapleyResult`` in ``report`` as a ``causal_edges``
        row (``method="shapley"``); same replace-not-duplicate upsert
        discipline as :meth:`save_blame_report`.
        """
        edge_ids: list[str] = []
        with self._write_lock:
            self._con.execute("BEGIN IMMEDIATE")
            try:
                self._con.execute(
                    "DELETE FROM causal_edges WHERE run_id=? AND method='shapley'", (run_id,)
                )
                for r in report.results:
                    edge_id = f"{run_id}:{r.step_index}:shapley"
                    edge_ids.append(edge_id)
                    self._con.execute(
                        _INSERT_EDGE_SQL,
                        (
                            edge_id,
                            run_id,
                            r.step_index,
                            "shapley",
                            None,
                            r.ci_lo,
                            r.ci_hi,
                            None,
                            None,
                            None,
                            None,
                            int(r.necessity),
                            int(r.sufficiency),
                            r.shapley_value,
                            created_at,
                        ),
                    )
                self._con.execute("COMMIT")
            except BaseException:
                self._con.execute("ROLLBACK")
                raise
        return edge_ids

    def causal_edges_for_run(self, run_id: str) -> list[dict]:
        """All causal edges (blame and Shapley) saved for ``run_id``, ordered
        by ``step_index`` then ``method``."""
        rows = self._con.execute(
            f"SELECT {_EDGE_COLUMNS} FROM causal_edges WHERE run_id=? ORDER BY step_index, method",
            (run_id,),
        ).fetchall()
        return [_edge_row_to_dict(r) for r in rows]

    def cited_by(self, run_id: str, step_index: int) -> list[str]:
        """Branch ids created via :meth:`save_branch` that diverged from
        ``run_id`` at ``step_index`` — derived directly from the existing
        ``branches`` table, no separate citation concept."""
        rows = self._con.execute(
            "SELECT branch_id FROM branches WHERE parent_run_id=? AND divergence_step=? "
            "ORDER BY created_at DESC",
            (run_id, step_index),
        ).fetchall()
        return [r[0] for r in rows]

    def causal_closure(self, run_id: str) -> list[dict]:
        """BFS the fork graph reachable from ``run_id``: each hop follows
        ``branches`` rows whose ``parent_run_id`` is the current frontier run
        and whose ``branch_id`` was itself later persisted as its own tape
        (i.e. promoted to a re-blamable run via
        ``save_tape(delta_tape, run_id=branch_id)``) — unioned with every
        hop's ``responsible=1`` blame edges.

        Returns the union of :meth:`causal_edges_for_run`-shaped dicts
        (``method="blame"``, ``responsible`` true) across every generation
        reachable from ``run_id``, deduplicated by ``edge_id`` and sorted by
        ``(run_id, step_index)``. A branch never promoted to its own tape is a
        dead end — the closure simply doesn't walk into it.
        """
        visited = {run_id}
        frontier = [run_id]
        edges: dict[str, dict] = {}
        while frontier:
            current = frontier.pop(0)
            for edge in self.causal_edges_for_run(current):
                if edge["method"] == "blame" and edge["responsible"]:
                    edges[edge["edge_id"]] = edge
            branch_rows = self._con.execute(
                "SELECT branch_id FROM branches WHERE parent_run_id=?", (current,)
            ).fetchall()
            for (branch_id,) in branch_rows:
                if branch_id in visited:
                    continue
                promoted = self._con.execute(
                    "SELECT 1 FROM tapes WHERE run_id=?", (branch_id,)
                ).fetchone()
                if promoted is not None:
                    visited.add(branch_id)
                    frontier.append(branch_id)
        return sorted(edges.values(), key=lambda e: (e["run_id"], e["step_index"]))

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

    # ── sessions / spawn_edges (orchestration spawn-lineage graph) ──────────

    def create_session(
        self, *, root_run_id: str, session_id: str | None = None, created_at: str = ""
    ) -> str:
        """Create a new orchestration session rooted at ``root_run_id`` (a
        fresh ``session_id`` if omitted).

        ``root_run_id`` must already be a stored tape — enforced by the
        ``sessions.root_run_id`` ``FOREIGN KEY`` to ``tapes(run_id)``, raising
        ``sqlite3.IntegrityError`` for an unknown ``run_id`` rather than
        silently accepting a dangling reference. Mirrors :meth:`save_tape`'s
        ``BEGIN IMMEDIATE`` + ``self._write_lock`` write discipline.
        """
        sid = session_id or uuid.uuid4().hex[:12]
        with self._write_lock:
            self._con.execute("BEGIN IMMEDIATE")
            try:
                self._con.execute(
                    "INSERT INTO sessions(session_id, root_run_id, created_at) VALUES(?,?,?)",
                    (sid, root_run_id, created_at),
                )
                self._con.execute("COMMIT")
            except BaseException:
                self._con.execute("ROLLBACK")
                raise
        return sid

    def get_session(self, session_id: str) -> dict:
        """Load a session's own row (``session_id``/``root_run_id``/
        ``created_at``) — not its spawn graph, see :meth:`session_tapes`.
        Raises ``KeyError`` for an unknown ``session_id``, mirroring
        :meth:`load_tape`/:meth:`load_branch`."""
        row = self._con.execute(
            "SELECT session_id, root_run_id, created_at FROM sessions WHERE session_id=?",
            (session_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"session_id {session_id!r} not found")
        return {"session_id": row[0], "root_run_id": row[1], "created_at": row[2]}

    def add_spawn_edge(
        self,
        *,
        session_id: str,
        parent_run_id: str,
        child_run_id: str,
        spawn_reason: str = "",
        edge_id: str | None = None,
        created_at: str = "",
    ) -> str:
        """Record a ``parent_run_id -> child_run_id`` delegation edge within
        ``session_id`` (a fresh ``edge_id`` if omitted).

        ``session_id``/``parent_run_id``/``child_run_id`` must already exist
        (a live session and two stored tapes) — each enforced by its own
        ``FOREIGN KEY``, raising ``sqlite3.IntegrityError`` on any dangling
        reference rather than silently accepting one. Mirrors
        :meth:`save_branch`'s ``BEGIN IMMEDIATE`` + ``self._write_lock``
        write discipline.
        """
        eid = edge_id or uuid.uuid4().hex[:12]
        with self._write_lock:
            self._con.execute("BEGIN IMMEDIATE")
            try:
                self._con.execute(
                    """INSERT INTO spawn_edges
                       (edge_id, session_id, parent_run_id, child_run_id,
                        spawn_reason, created_at)
                       VALUES(?,?,?,?,?,?)""",
                    (eid, session_id, parent_run_id, child_run_id, spawn_reason, created_at),
                )
                self._con.execute("COMMIT")
            except BaseException:
                self._con.execute("ROLLBACK")
                raise
        return eid

    def session_tapes(self, session_id: str) -> list[str]:
        """BFS the spawn graph reachable from ``session_id``'s
        ``root_run_id``, following ``spawn_edges.parent_run_id ->
        child_run_id`` hops within this session only.

        Returns every reachable ``run_id`` (including the root) in BFS
        discovery order, deduplicated — a run reached via more than one path
        (e.g. a diamond: two parents delegating to the same child) appears
        exactly once. Raises ``KeyError`` (via :meth:`get_session`) for an
        unknown ``session_id``.
        """
        root_run_id = self.get_session(session_id)["root_run_id"]
        order = [root_run_id]
        seen = {root_run_id}
        frontier = [root_run_id]
        while frontier:
            current = frontier.pop(0)
            rows = self._con.execute(
                "SELECT DISTINCT child_run_id FROM spawn_edges "
                "WHERE session_id=? AND parent_run_id=? ORDER BY child_run_id",
                (session_id, current),
            ).fetchall()
            for (child,) in rows:
                if child not in seen:
                    seen.add(child)
                    order.append(child)
                    frontier.append(child)
        return order

    def spawn_children(self, run_id: str) -> list[str]:
        """Direct spawn children of ``run_id`` — every ``child_run_id`` from
        a ``spawn_edges`` row whose ``parent_run_id`` is ``run_id``, across
        ALL sessions, ordered by ``created_at`` then ``child_run_id``. ``[]``
        for a leaf (never spawned anything)."""
        rows = self._con.execute(
            "SELECT child_run_id FROM spawn_edges WHERE parent_run_id=? "
            "ORDER BY created_at, child_run_id",
            (run_id,),
        ).fetchall()
        return [r[0] for r in rows]

    def spawn_parent(self, run_id: str) -> str | None:
        """The spawn parent of ``run_id`` — the ``parent_run_id`` of the
        (oldest, by ``created_at``) ``spawn_edges`` row whose ``child_run_id``
        is ``run_id``, or ``None`` for a session root (never spawned by
        anything)."""
        row = self._con.execute(
            "SELECT parent_run_id FROM spawn_edges WHERE child_run_id=? "
            "ORDER BY created_at, edge_id LIMIT 1",
            (run_id,),
        ).fetchone()
        return None if row is None else row[0]

    def close(self) -> None:
        self._con.close()
