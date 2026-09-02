"""Storage-hardening tests: the versioned to_bytes/from_bytes envelope (round-trip,
golden legacy-blob backward-compat, digest stability, content-addressed dedup) and
the SQLite pragma bundle + write serialization (no `database is locked` under
concurrent writers). All offline and $0."""

import json
import pathlib
import struct
import threading

import pytest

from tracefork.constants import TAPE_FORMAT_VERSION, TAPE_MAGIC
from tracefork.store import ForkPointDriftError, StorageBackend, TapeConflictError, TapeStore
from tracefork.tape import Tape, open_sqlite

# The exact digest of `_golden_tape()`, frozen at the pre-header format. If a change
# to the version envelope ever perturbs this, the header has leaked into the
# content-addressed hash chain — which the must-have forbids.
#
# Updated for DIGEST_CHAIN_VERSION = 2 (fixed-width sha256 framing of draw
# kind/value, see Tape.digest() in tape.py): this is an intentional digest
# change, not header leakage — draws are still part of the content hash, only
# their framing changed to close a delimiter-forgery gap. Verified identical
# across a fresh tape, a v2 round-trip, and the legacy v1 blob fixture.
GOLDEN_DIGEST = "8753f6188737f97012f0b005f126daa6b9919deb54a047025b5e5b0e5dc0dbdc"

_FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "legacy_tape_v1.blob"


def _golden_tape() -> Tape:
    """Byte-for-byte the tape whose legacy blob is committed as the golden fixture."""
    t = Tape(agent_name="golden-agent", boundary="single-process-asyncio-v1")
    t.draws = [("clock", "2026-01-01T00:00:00+00:00"), ("uuid", "abc123")]
    t.append_exchange(b"request-1", b"response-1")
    t.append_exchange(b"request-2", b"response-2")
    t.append_exchange(b"request-1", b"response-1")  # duplicate → exercises dedup
    return t


def _small_tape(tag: bytes = b"x") -> Tape:
    t = Tape(agent_name="w")
    t.append_exchange(b"req-" + tag, b"resp-" + tag)
    return t


def _parse_v2_envelope(blob: bytes) -> tuple[int, dict]:
    assert blob[: len(TAPE_MAGIC)] == TAPE_MAGIC
    (ver,) = struct.unpack_from(">H", blob, len(TAPE_MAGIC))
    off = len(TAPE_MAGIC) + 2
    (hlen,) = struct.unpack_from(">I", blob, off)
    off += 4
    header = json.loads(blob[off : off + hlen])
    return ver, header


# ── format version + round-trip ─────────────────────────────────────────────


def test_to_bytes_writes_magic_and_version():
    blob = _golden_tape().to_bytes()
    assert blob[: len(TAPE_MAGIC)] == TAPE_MAGIC
    (ver,) = struct.unpack_from(">H", blob, len(TAPE_MAGIC))
    assert ver == TAPE_FORMAT_VERSION


def test_v2_roundtrip_preserves_everything():
    t = _golden_tape()
    restored = Tape.from_bytes(t.to_bytes())
    assert restored.draws == t.draws
    assert restored.exchanges == t.exchanges
    assert restored.agent_name == t.agent_name
    assert restored.boundary == t.boundary
    assert restored.digest() == t.digest()


def test_v2_dedups_shared_blobs():
    t = Tape()
    t.append_exchange(b"same-req", b"same-resp")
    t.append_exchange(b"same-req", b"same-resp")
    ver, header = _parse_v2_envelope(t.to_bytes())
    assert ver == TAPE_FORMAT_VERSION
    assert len(header["exchanges"]) == 2
    assert len(header["blob_hashes"]) == 2  # 2 unique blobs, not 4


def test_v2_kills_base64_blowup():
    """Repetitive-but-distinct content: zstd + no base64 must land well under the
    raw payload size (the legacy JSON+base64 path bloated ~1.33x before any zstd)."""
    t = Tape(agent_name="big")
    for i in range(20):
        pad_a, pad_b = "x" * 500, "y" * 500
        req = f'{{"m":{i},"p":"{pad_a}"}}'.encode()
        resp = f'{{"r":{i},"p":"{pad_b}"}}'.encode()
        t.append_exchange(req, resp)
    raw = sum(len(a) + len(b) for a, b in t.exchanges)
    assert len(t.to_bytes()) < raw


def test_unsupported_future_version_raises():
    blob = TAPE_MAGIC + struct.pack(">H", 9999) + b"whatever"
    with pytest.raises(ValueError, match="unsupported tape format version"):
        Tape.from_bytes(blob)


def _encode_as_v4_without_provenance(t: Tape) -> bytes:
    """Hand-construct a genuine pre-v5 (v4) envelope — no `provenance` key in
    the header at all — to prove the v4->v5 upcaster defaults a pre-v5 tape's
    `provenance` to `{}` with an unchanged digest, mirroring how the committed
    v1 golden fixture proves the v1->v2 upcast below."""
    import zstandard as zstd

    from tracefork.tape import sha256_hex

    zctx = zstd.ZstdCompressor(level=3)
    order: list[str] = []
    seen: dict[str, bytes] = {}
    for req, resp in (*t.exchanges, *t.tool_exchanges):
        for blob in (req, resp):
            h = sha256_hex(blob)
            if h not in seen:
                seen[h] = blob
                order.append(h)
    header = {
        "boundary": t.boundary,
        "agent_name": t.agent_name,
        "draws": t.draws,
        "exchanges": [[sha256_hex(r), sha256_hex(s)] for r, s in t.exchanges],
        "tool_exchanges": [[sha256_hex(r), sha256_hex(s)] for r, s in t.tool_exchanges],
        "async_batches": t.async_batches,
        "blob_hashes": order,
        "content_redacted": t.content_redacted,
    }
    header_json = json.dumps(header).encode()
    parts = [TAPE_MAGIC, struct.pack(">H", 4), struct.pack(">I", len(header_json)), header_json]
    for h in order:
        comp = zctx.compress(seen[h])
        parts.append(struct.pack(">I", len(comp)))
        parts.append(comp)
    return b"".join(parts)


def test_v4_tape_without_provenance_upcasts_to_empty_dict_unchanged_digest():
    t = _golden_tape()
    blob = _encode_as_v4_without_provenance(t)
    restored = Tape.from_bytes(blob)
    assert restored.provenance == {}
    assert restored.digest() == t.digest() == GOLDEN_DIGEST


def _encode_as_v6_without_response_meta(t: Tape) -> bytes:
    """Hand-construct a genuine pre-v7 (v6) envelope -- no `response_status`/
    `response_content_type` keys in the header at all -- to prove the
    v6->v7 upcaster defaults a pre-v7 tape's response status/content-type to
    200/"application/json" with an unchanged digest."""
    import zstandard as zstd

    from tracefork.tape import sha256_hex

    zctx = zstd.ZstdCompressor(level=3)
    order: list[str] = []
    seen: dict[str, bytes] = {}
    for req, resp in (*t.exchanges, *t.tool_exchanges):
        for blob in (req, resp):
            h = sha256_hex(blob)
            if h not in seen:
                seen[h] = blob
                order.append(h)
    header = {
        "boundary": t.boundary,
        "agent_name": t.agent_name,
        "draws": t.draws,
        "exchanges": [[sha256_hex(r), sha256_hex(s)] for r, s in t.exchanges],
        "tool_exchanges": [[sha256_hex(r), sha256_hex(s)] for r, s in t.tool_exchanges],
        "async_batches": t.async_batches,
        "provenance": t.provenance,
        "request_urls": t.request_urls,
        "blob_hashes": order,
        "content_redacted": t.content_redacted,
    }
    header_json = json.dumps(header).encode()
    parts = [TAPE_MAGIC, struct.pack(">H", 6), struct.pack(">I", len(header_json)), header_json]
    for h in order:
        comp = zctx.compress(seen[h])
        parts.append(struct.pack(">I", len(comp)))
        parts.append(comp)
    return b"".join(parts)


def test_v6_tape_without_response_meta_upcasts_to_default_unchanged_digest():
    t = _golden_tape()
    blob = _encode_as_v6_without_response_meta(t)
    restored = Tape.from_bytes(blob)
    assert restored.response_status == [200] * len(t.exchanges)
    assert restored.response_content_type == ["application/json"] * len(t.exchanges)
    assert restored.digest() == t.digest() == GOLDEN_DIGEST


def test_v7_roundtrip_preserves_response_status_and_content_type():
    t = Tape(agent_name="v7-agent")
    t.append_exchange(b"req-1", b"resp-1", response_status=429, response_content_type="text/plain")
    t.append_exchange(
        b"req-2",
        b"resp-2",
        response_status=200,
        response_content_type="text/event-stream",
    )
    ver, header = _parse_v2_envelope(t.to_bytes())
    assert ver == 7
    assert header["response_status"] == [429, 200]
    assert header["response_content_type"] == ["text/plain", "text/event-stream"]
    restored = Tape.from_bytes(t.to_bytes())
    assert restored.response_status == [429, 200]
    assert restored.response_content_type == ["text/plain", "text/event-stream"]
    assert restored.exchange_status(0) == 429
    assert restored.exchange_content_type(0) == "text/plain"
    assert restored.exchange_status(1) == 200
    assert restored.exchange_content_type(1) == "text/event-stream"


def test_response_status_and_content_type_never_feed_digest():
    """Two tapes differing ONLY in recorded response status/content-type must
    hash identically -- this metadata is forensic/replay-fidelity only, never
    hash-chained content (mirrors request_urls'/provenance's own guarantee)."""
    a = Tape(agent_name="x")
    a.append_exchange(
        b"req", b"resp", response_status=200, response_content_type="application/json"
    )
    b = Tape(agent_name="x")
    b.append_exchange(b"req", b"resp", response_status=429, response_content_type="text/plain")
    assert a.digest() == b.digest()


def test_response_status_and_content_type_roundtrip_through_sqlite_save_load(tmp_path):
    t = Tape(agent_name="sqlite-agent")
    t.append_exchange(b"req-1", b"resp-1", response_status=503, response_content_type="text/html")
    path = str(tmp_path / "t.tape.sqlite")
    t.save(path)
    restored = Tape.load(path)
    assert restored.response_status == [503]
    assert restored.response_content_type == ["text/html"]
    assert restored.digest() == t.digest()


def test_legacy_sqlite_without_response_meta_loads_default_status_and_content_type(tmp_path):
    """A store.db written before response_status/response_content_type existed
    (no such meta rows) must load with the documented (200, application/json)
    default, not KeyError."""
    t = Tape(agent_name="pre-v7-sqlite")
    t.append_exchange(b"req-1", b"resp-1")
    path = str(tmp_path / "t.tape.sqlite")
    t.save(path)

    con = open_sqlite(path)
    con.execute("DELETE FROM meta WHERE key IN ('response_status', 'response_content_type')")
    con.commit()
    con.close()

    restored = Tape.load(path)
    assert restored.response_status == [200]
    assert restored.response_content_type == ["application/json"]


def test_load_nonexistent_path_raises_without_creating_a_file(tmp_path):
    """tracefork-sis.32: `sqlite3.connect` auto-creates an empty file at
    connect time for a mistyped path whose parent directory exists -- `Tape.load`
    must check existence FIRST so a typo'd path never side-effect-creates a
    junk empty `.sqlite` file before raising."""
    path = tmp_path / "does-not-exist.tape.sqlite"
    assert not path.exists()
    with pytest.raises(FileNotFoundError):
        Tape.load(str(path))
    assert not path.exists()


def test_exchange_status_and_content_type_default_for_hand_built_tape():
    """A `Tape` constructed with `exchanges=` directly (e.g. `tournament.py`'s
    single/pair-exchange cost-estimation probe tapes) never populates
    response_status/response_content_type -- the accessor methods must
    default gracefully rather than raising IndexError."""
    t = Tape(exchanges=[(b"req-1", b"resp-1"), (b"req-2", b"resp-2")])
    assert t.response_status == []
    assert t.response_content_type == []
    assert t.exchange_status(0) == 200
    assert t.exchange_status(1) == 200
    assert t.exchange_content_type(0) == "application/json"
    assert t.exchange_content_type(1) == "application/json"


# ── backward-compat: the committed golden legacy blob ───────────────────────


def test_legacy_blob_is_genuinely_headerless():
    blob = _FIXTURE.read_bytes()
    assert blob[: len(TAPE_MAGIC)] != TAPE_MAGIC
    assert blob[:1] == b"{"  # original JSON encoding


def test_legacy_blob_still_loads_as_v1():
    """CRITICAL backward-compat: an old JSON+base64 blob (no magic header) must
    load through the detect-and-fall-back path, not crash."""
    restored = Tape.from_bytes(_FIXTURE.read_bytes())
    golden = _golden_tape()
    assert restored.exchanges == golden.exchanges
    assert restored.draws == golden.draws
    assert restored.agent_name == "golden-agent"
    assert restored.boundary == golden.boundary


# ── digest stability (header is envelope metadata, not content) ─────────────


def test_digest_is_frozen_for_known_content():
    assert _golden_tape().digest() == GOLDEN_DIGEST


def test_digest_stable_across_format_versions():
    golden = _golden_tape()
    # v2 round-trip and the legacy blob must both yield the exact same digest.
    assert Tape.from_bytes(golden.to_bytes()).digest() == GOLDEN_DIGEST
    assert Tape.from_bytes(_FIXTURE.read_bytes()).digest() == GOLDEN_DIGEST


# ── SQLite hardening: pragmas + concurrency ─────────────────────────────────


def test_open_sqlite_applies_pragma_bundle(tmp_path):
    con = open_sqlite(str(tmp_path / "x.db"))
    try:
        assert con.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert con.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
        assert con.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    finally:
        con.close()


def test_store_connection_is_hardened(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        con = store._con
        assert con.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert con.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
        assert con.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    finally:
        store.close()


def test_concurrent_writers_separate_connections_no_lock(tmp_path):
    """Separate connections to one file (the realistic blame-fork case): WAL +
    busy_timeout must let writers queue instead of raising `database is locked`."""
    db = str(tmp_path / "store.db")
    TapeStore(db).close()  # create schema up front

    errors: list[BaseException] = []

    def worker(w: int) -> None:
        try:
            s = TapeStore(db)
            for j in range(5):
                s.save_tape(_small_tape(f"{w}-{j}".encode()), run_id=f"r{w}_{j}")
            s.close()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    s = TapeStore(db)
    try:
        assert len(s.list_runs()) == 8 * 5
    finally:
        s.close()


# ── StorageBackend protocol conformance ─────────────────────────────────────


def test_tape_store_satisfies_storage_backend_protocol(tmp_path):
    """`TapeStore` (SQLite) must structurally satisfy `StorageBackend` — the
    seam a future filesystem/object-store backend would implement instead."""
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        assert isinstance(store, StorageBackend)
    finally:
        store.close()


def test_save_tape_same_run_id_identical_content_is_idempotent(tmp_path):
    """Reusing a run_id with byte-identical content is a no-op success, not an
    error — install-or-verify-same-content, the git object-store model."""
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        tape = _small_tape(b"same")
        first = store.save_tape(tape, run_id="dup-run")
        second = store.save_tape(_small_tape(b"same"), run_id="dup-run")
        assert first == second == "dup-run"
        assert store.load_tape("dup-run").exchanges == tape.exchanges
    finally:
        store.close()


def test_save_tape_same_run_id_different_content_raises_conflict(tmp_path):
    """Reusing a run_id with DIFFERENT content must raise, not silently clobber
    the previously-stored tape."""
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        first_tape = _small_tape(b"first")
        store.save_tape(first_tape, run_id="conflict-run")
        with pytest.raises(TapeConflictError):
            store.save_tape(_small_tape(b"second"), run_id="conflict-run")
        # No partial clobber: the FIRST content must still be there.
        assert store.load_tape("conflict-run").exchanges == first_tape.exchanges
    finally:
        store.close()


def test_save_tape_overwrite_true_replaces_content(tmp_path):
    """`overwrite=True` is the explicit escape hatch that actually replaces content."""
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        store.save_tape(_small_tape(b"first"), run_id="overwrite-run")
        new_tape = _small_tape(b"second")
        run_id = store.save_tape(new_tape, run_id="overwrite-run", overwrite=True)
        assert run_id == "overwrite-run"
        assert store.load_tape("overwrite-run").exchanges == new_tape.exchanges
    finally:
        store.close()


def test_storage_backend_full_round_trip_through_the_protocol_surface(tmp_path):
    """Exercise every `StorageBackend` method via the protocol-typed surface —
    proof the protocol's shape actually matches what `TapeStore` does."""
    backend: StorageBackend = TapeStore(str(tmp_path / "store.db"))
    try:
        tape = _small_tape(b"proto")
        run_id = backend.save_tape(tape, run_id="proto-run")
        assert backend.load_tape(run_id).exchanges == tape.exchanges
        assert any(r["run_id"] == run_id for r in backend.list_runs())

        branch_id = backend.save_branch(
            parent_run_id=run_id,
            divergence_step=0,
            delta_tape=_small_tape(b"branch"),
            mutation_desc="test branch",
        )
        loaded_branch = backend.load_branch(branch_id)
        assert loaded_branch["parent_run_id"] == run_id
        assert any(b["branch_id"] == branch_id for b in backend.list_branches(run_id))
    finally:
        backend.close()


# ── branch_digest (content-addressed fork DAG) ──────────────────────────────


def test_save_branch_persists_branch_digest_and_load_branch_returns_it(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        run_id = store.save_tape(_small_tape(b"parent"), run_id="parent-run")
        branch_id = store.save_branch(
            parent_run_id=run_id,
            divergence_step=0,
            delta_tape=_small_tape(b"branch"),
            mutation_desc="test",
            branch_digest="deadbeef",
        )
        loaded = store.load_branch(branch_id)
        assert loaded["branch_digest"] == "deadbeef"
    finally:
        store.close()


def test_list_branches_includes_branch_digest(tmp_path):
    """`list_branches` (the no-`delta_tape`-fetch summary `report.py`'s
    fork-tree panel embeds in static mode, see tracefork-bge.15) must carry
    `branch_digest` alongside the pre-existing summary fields, so a branch
    edge can be labeled without a full `load_branch` round trip."""
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        run_id = store.save_tape(_small_tape(b"parent"), run_id="parent-run")
        branch_id = store.save_branch(
            parent_run_id=run_id,
            divergence_step=2,
            delta_tape=_small_tape(b"branch"),
            mutation_desc="test",
            branch_digest="deadbeef",
        )
        summaries = store.list_branches(run_id)
        matching = [b for b in summaries if b["branch_id"] == branch_id]
        assert len(matching) == 1
        assert matching[0]["branch_digest"] == "deadbeef"
        assert matching[0]["divergence_step"] == 2
    finally:
        store.close()


def test_save_branch_default_branch_digest_is_empty_string(tmp_path):
    """Existing callers that omit branch_digest keep working — default ''."""
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        run_id = store.save_tape(_small_tape(b"parent"), run_id="parent-run")
        branch_id = store.save_branch(
            parent_run_id=run_id,
            divergence_step=0,
            delta_tape=_small_tape(b"branch"),
            mutation_desc="test",
        )
        loaded = store.load_branch(branch_id)
        assert loaded["branch_digest"] == ""
    finally:
        store.close()


def test_find_branch_by_digest_resolves_the_right_branch(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        run_id = store.save_tape(_small_tape(b"parent"), run_id="parent-run")
        branch_id = store.save_branch(
            parent_run_id=run_id,
            divergence_step=0,
            delta_tape=_small_tape(b"branch"),
            branch_digest="findme123",
        )
        found = store.find_branch_by_digest("findme123")
        assert found is not None
        assert found["branch_id"] == branch_id
    finally:
        store.close()


def test_find_branch_by_digest_nonexistent_returns_none(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        store.save_tape(_small_tape(b"parent"), run_id="parent-run")
        assert store.find_branch_by_digest("nope-not-there") is None
    finally:
        store.close()


def test_branches_forked_from_finds_fork_of_fork(tmp_path):
    """Fork A, save A's delta_tape as its own tape X, fork X as B --
    branches_forked_from(A.branch_digest) surfaces B (fork-of-fork,
    end-to-end via the inverse-citation query)."""
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        run_id = store.save_tape(_small_tape(b"root"), run_id="root-run")
        branch_a_delta = _small_tape(b"branch-a")
        branch_a_id = store.save_branch(
            parent_run_id=run_id,
            divergence_step=0,
            delta_tape=branch_a_delta,
            branch_digest="digest-a",
        )
        # Promote A's delta_tape to its own tape under run_id == branch_a_id
        # (same convention `causal_closure` already relies on).
        store.save_tape(branch_a_delta, run_id=branch_a_id)

        branch_b_id = store.save_branch(
            parent_run_id=branch_a_id,
            divergence_step=0,
            delta_tape=_small_tape(b"branch-b"),
            branch_digest="digest-b",
        )

        result = store.branches_forked_from("digest-a")
        assert branch_b_id in result
    finally:
        store.close()


def test_branch_digest_migration_adds_column_without_losing_rows(tmp_path):
    """A store.db built with the OLD schema (no `branch_digest` column)
    neither crashes nor loses rows when opened by the new `TapeStore` --
    the column gets added via a guarded `ALTER TABLE`."""
    db_path = str(tmp_path / "old_store.db")

    # Build an old-schema store.db by hand (no branch_digest column at all).
    old_con = open_sqlite(db_path)
    old_con.executescript(
        """
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
    )
    old_tape = _small_tape(b"pre-existing")
    old_con.execute(
        "INSERT INTO tapes(run_id, agent_name, tape_bytes, created_at) VALUES(?,?,?,?)",
        ("old-run", "w", old_tape.to_bytes(), "2020-01-01T00:00:00+00:00"),
    )
    old_con.execute(
        """INSERT INTO branches
           (branch_id, parent_run_id, divergence_step, delta_tape_bytes, mutation_desc, created_at)
           VALUES(?,?,?,?,?,?)""",
        (
            "old-branch",
            "old-run",
            0,
            _small_tape(b"pre-branch").to_bytes(),
            "",
            "2020-01-01T00:00:00+00:00",
        ),
    )
    cols_before = {row[1] for row in old_con.execute("PRAGMA table_info(branches)").fetchall()}
    assert "branch_digest" not in cols_before

    old_con.commit()
    old_con.close()

    # Opening with the new TapeStore must not crash and must not lose rows.
    store = TapeStore(db_path)
    try:
        assert store.load_tape("old-run").exchanges == old_tape.exchanges
        loaded_branch = store.load_branch("old-branch")
        assert loaded_branch["parent_run_id"] == "old-run"
        assert loaded_branch["branch_digest"] == ""  # migrated column defaults to ''

        cols_after = {
            row[1] for row in store._con.execute("PRAGMA table_info(branches)").fetchall()
        }
        assert "branch_digest" in cols_after
    finally:
        store.close()


# ── fork-point verification (parent_tape_digest / divergence_exchange_digest) ──


def test_save_branch_persists_fork_point_digests_and_load_branch_returns_them(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        parent_tape = _small_tape(b"parent")
        run_id = store.save_tape(parent_tape, run_id="parent-run")
        branch_id = store.save_branch(
            parent_run_id=run_id,
            divergence_step=0,
            delta_tape=_small_tape(b"branch"),
            parent_tape_digest=parent_tape.digest(),
            divergence_exchange_digest="exch-digest",
        )
        loaded = store.load_branch(branch_id)
        assert loaded["parent_tape_digest"] == parent_tape.digest()
        assert loaded["divergence_exchange_digest"] == "exch-digest"
    finally:
        store.close()


def test_save_branch_default_fork_point_digests_are_empty_string(tmp_path):
    """Existing callers (e.g. cli.py's fork command) that omit the new
    digests keep working exactly as before -- default ''."""
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        run_id = store.save_tape(_small_tape(b"parent"), run_id="parent-run")
        branch_id = store.save_branch(
            parent_run_id=run_id,
            divergence_step=0,
            delta_tape=_small_tape(b"branch"),
        )
        loaded = store.load_branch(branch_id)
        assert loaded["parent_tape_digest"] == ""
        assert loaded["divergence_exchange_digest"] == ""
    finally:
        store.close()


def test_load_branch_no_drift_succeeds_baseline_regression(tmp_path):
    """save_branch then immediate load_branch with no tape mutation succeeds
    -- no ForkPointDriftError (baseline regression)."""
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        parent_tape = _small_tape(b"parent")
        run_id = store.save_tape(parent_tape, run_id="parent-run")
        branch_id = store.save_branch(
            parent_run_id=run_id,
            divergence_step=0,
            delta_tape=_small_tape(b"branch"),
            parent_tape_digest=parent_tape.digest(),
        )
        loaded = store.load_branch(branch_id)  # must not raise
        assert loaded["branch_id"] == branch_id
    finally:
        store.close()


# ── intervened_steps (rebase's full-coalition prerequisite) ─────────────────


def test_save_branch_persists_intervened_steps_and_load_branch_returns_them(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        run_id = store.save_tape(_small_tape(b"parent"), run_id="parent-run")
        branch_id = store.save_branch(
            parent_run_id=run_id,
            divergence_step=0,
            delta_tape=_small_tape(b"branch"),
            intervened_steps=(0, 2, 5),
        )
        loaded = store.load_branch(branch_id)
        assert loaded["intervened_steps"] == (0, 2, 5)
    finally:
        store.close()


def test_save_branch_default_intervened_steps_is_empty_tuple(tmp_path):
    """Existing callers that omit intervened_steps keep working -- default ()."""
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        run_id = store.save_tape(_small_tape(b"parent"), run_id="parent-run")
        branch_id = store.save_branch(
            parent_run_id=run_id,
            divergence_step=0,
            delta_tape=_small_tape(b"branch"),
        )
        loaded = store.load_branch(branch_id)
        assert loaded["intervened_steps"] == ()
    finally:
        store.close()


def test_find_branch_by_digest_also_returns_intervened_steps(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        run_id = store.save_tape(_small_tape(b"parent"), run_id="parent-run")
        store.save_branch(
            parent_run_id=run_id,
            divergence_step=0,
            delta_tape=_small_tape(b"branch"),
            branch_digest="findme-steps",
            intervened_steps=(1, 3),
        )
        found = store.find_branch_by_digest("findme-steps")
        assert found is not None
        assert found["intervened_steps"] == (1, 3)
    finally:
        store.close()


def test_intervened_steps_migration_adds_column_without_losing_rows(tmp_path):
    """A store.db built with the OLD schema (no `intervened_steps_json`
    column) neither crashes nor loses rows -- the column is added via a
    guarded `ALTER TABLE`, same discipline as `branch_digest`."""
    db_path = str(tmp_path / "old_store.db")

    old_con = open_sqlite(db_path)
    old_con.executescript(
        """
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
            FOREIGN KEY(parent_run_id) REFERENCES tapes(run_id)
        );
        """
    )
    old_tape = _small_tape(b"pre-existing")
    old_con.execute(
        "INSERT INTO tapes(run_id, agent_name, tape_bytes, created_at) VALUES(?,?,?,?)",
        ("old-run", "w", old_tape.to_bytes(), "2020-01-01T00:00:00+00:00"),
    )
    old_con.execute(
        """INSERT INTO branches
           (branch_id, parent_run_id, divergence_step, delta_tape_bytes, mutation_desc, created_at)
           VALUES(?,?,?,?,?,?)""",
        (
            "old-branch",
            "old-run",
            0,
            _small_tape(b"pre-branch").to_bytes(),
            "",
            "2020-01-01T00:00:00+00:00",
        ),
    )
    cols_before = {row[1] for row in old_con.execute("PRAGMA table_info(branches)").fetchall()}
    assert "intervened_steps_json" not in cols_before
    old_con.commit()
    old_con.close()

    store = TapeStore(db_path)
    try:
        assert store.load_tape("old-run").exchanges == old_tape.exchanges
        loaded_branch = store.load_branch("old-branch")
        assert loaded_branch["intervened_steps"] == ()

        cols_after = {
            row[1] for row in store._con.execute("PRAGMA table_info(branches)").fetchall()
        }
        assert "intervened_steps_json" in cols_after
    finally:
        store.close()


def test_load_branch_raises_fork_point_drift_error_on_parent_mutation(tmp_path):
    """A raw-SQL mutation of the cited parent tape's content after the fork
    was made must be caught -- re-verified, not trusted -- at the next
    load_branch, naming the mismatched parent_run_id."""
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        parent_tape = _small_tape(b"parent")
        run_id = store.save_tape(parent_tape, run_id="parent-run")
        branch_id = store.save_branch(
            parent_run_id=run_id,
            divergence_step=0,
            delta_tape=_small_tape(b"branch"),
            parent_tape_digest=parent_tape.digest(),
        )

        # Simulate silent drift: raw-SQL swap the parent tape's stored bytes.
        mutated_tape = _small_tape(b"mutated-parent")
        store._con.execute(
            "UPDATE tapes SET tape_bytes=? WHERE run_id=?", (mutated_tape.to_bytes(), run_id)
        )
        store._con.commit()

        with pytest.raises(ForkPointDriftError, match=run_id):
            store.load_branch(branch_id)
    finally:
        store.close()


def test_load_branch_skips_reverification_when_parent_tape_digest_is_empty(tmp_path):
    """Legacy/pre-migration branches (parent_tape_digest == '', e.g. produced
    by cli.py's fork command, which does not pass it) have nothing to
    re-verify against -- load_branch must not raise even if the parent has
    since changed."""
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        run_id = store.save_tape(_small_tape(b"parent"), run_id="parent-run")
        branch_id = store.save_branch(
            parent_run_id=run_id,
            divergence_step=0,
            delta_tape=_small_tape(b"branch"),
        )
        mutated_tape = _small_tape(b"mutated-parent")
        store._con.execute(
            "UPDATE tapes SET tape_bytes=? WHERE run_id=?", (mutated_tape.to_bytes(), run_id)
        )
        store._con.commit()

        loaded = store.load_branch(branch_id)  # must not raise
        assert loaded["parent_tape_digest"] == ""
    finally:
        store.close()


def test_fork_point_columns_migration_adds_all_three_in_one_pass_without_losing_rows(tmp_path):
    """A store.db built with the OLD schema (no branch_digest,
    parent_tape_digest, or divergence_exchange_digest columns at all) neither
    crashes nor loses rows -- all three are added via one PRAGMA-guarded
    ADD COLUMN pass."""
    db_path = str(tmp_path / "old_store2.db")

    old_con = open_sqlite(db_path)
    old_con.executescript(
        """
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
    )
    old_tape = _small_tape(b"pre-existing2")
    old_con.execute(
        "INSERT INTO tapes(run_id, agent_name, tape_bytes, created_at) VALUES(?,?,?,?)",
        ("old-run2", "w", old_tape.to_bytes(), "2020-01-01T00:00:00+00:00"),
    )
    old_con.execute(
        """INSERT INTO branches
           (branch_id, parent_run_id, divergence_step, delta_tape_bytes, mutation_desc, created_at)
           VALUES(?,?,?,?,?,?)""",
        (
            "old-branch2",
            "old-run2",
            0,
            _small_tape(b"pre-branch2").to_bytes(),
            "",
            "2020-01-01T00:00:00+00:00",
        ),
    )
    cols_before = {row[1] for row in old_con.execute("PRAGMA table_info(branches)").fetchall()}
    assert "branch_digest" not in cols_before
    assert "parent_tape_digest" not in cols_before
    assert "divergence_exchange_digest" not in cols_before

    old_con.commit()
    old_con.close()

    store = TapeStore(db_path)
    try:
        assert store.load_tape("old-run2").exchanges == old_tape.exchanges
        loaded_branch = store.load_branch("old-branch2")
        assert loaded_branch["parent_run_id"] == "old-run2"
        assert loaded_branch["branch_digest"] == ""
        assert loaded_branch["parent_tape_digest"] == ""
        assert loaded_branch["divergence_exchange_digest"] == ""

        cols_after = {
            row[1] for row in store._con.execute("PRAGMA table_info(branches)").fetchall()
        }
        assert "branch_digest" in cols_after
        assert "parent_tape_digest" in cols_after
        assert "divergence_exchange_digest" in cols_after
    finally:
        store.close()


# ── prune (soft-archive, never hard-delete) ─────────────────────────────────


def test_prune_older_than_cutoff_archives_only_matching_tapes(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        store.save_tape(
            _small_tape(b"old"), run_id="old-run", created_at="2020-01-01T00:00:00+00:00"
        )
        store.save_tape(
            _small_tape(b"new"), run_id="new-run", created_at="2030-01-01T00:00:00+00:00"
        )

        report = store.prune(older_than_iso="2025-01-01T00:00:00+00:00")

        assert report.dry_run is False
        assert report.tapes_archived == ["old-run"]
        assert report.branches_archived == []

        assert [r["run_id"] for r in store.list_runs()] == ["new-run"]
        with pytest.raises(KeyError):
            store.load_tape("old-run")

        archived = store._con.execute("SELECT run_id FROM tapes_archived").fetchall()
        assert [a[0] for a in archived] == ["old-run"]
    finally:
        store.close()


def test_prune_archives_branches_with_their_tape_atomically_no_fk_violation(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        run_id = store.save_tape(
            _small_tape(b"parent"), run_id="parent-run", created_at="2020-01-01T00:00:00+00:00"
        )
        branch_id = store.save_branch(
            parent_run_id=run_id,
            divergence_step=0,
            delta_tape=_small_tape(b"branch"),
            mutation_desc="test branch",
            created_at="2020-01-01T00:00:00+00:00",
        )

        report = store.prune(older_than_iso="2025-01-01T00:00:00+00:00")

        assert report.tapes_archived == ["parent-run"]
        assert report.branches_archived == [branch_id]

        archived_tapes = {
            r[0] for r in store._con.execute("SELECT run_id FROM tapes_archived").fetchall()
        }
        archived_branches = store._con.execute(
            "SELECT branch_id, parent_run_id FROM branches_archived"
        ).fetchall()
        assert run_id in archived_tapes
        assert archived_branches == [(branch_id, run_id)]  # no orphaned archived branch row

        # Live tables fully cleaned up, no FK violation was raised getting here.
        remaining = store._con.execute(
            "SELECT COUNT(*) FROM branches WHERE parent_run_id=?", (run_id,)
        ).fetchone()[0]
        assert remaining == 0
        with pytest.raises(KeyError):
            store.load_tape(run_id)
        with pytest.raises(KeyError):
            store.load_branch(branch_id)
    finally:
        store.close()


def test_prune_dry_run_computes_candidates_with_zero_mutation(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        tape = _small_tape(b"old")
        store.save_tape(tape, run_id="old-run", created_at="2020-01-01T00:00:00+00:00")

        report = store.prune(older_than_iso="2025-01-01T00:00:00+00:00", dry_run=True)

        assert report.dry_run is True
        assert report.tapes_archived == ["old-run"]

        # Zero mutation: the live tape still reloads, no archived row exists.
        assert store.load_tape("old-run").exchanges == tape.exchanges
        n_archived = store._con.execute("SELECT COUNT(*) FROM tapes_archived").fetchone()[0]
        assert n_archived == 0
    finally:
        store.close()


def test_prune_by_explicit_run_ids_ignores_created_at(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        store.save_tape(_small_tape(b"a"), run_id="run-a", created_at="2030-01-01T00:00:00+00:00")
        store.save_tape(_small_tape(b"b"), run_id="run-b", created_at="2030-01-01T00:00:00+00:00")

        report = store.prune(run_ids=["run-a"])

        assert report.tapes_archived == ["run-a"]
        assert [r["run_id"] for r in store.list_runs()] == ["run-b"]
    finally:
        store.close()


def test_prune_with_no_filters_is_a_safe_noop(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        store.save_tape(_small_tape(b"a"), run_id="run-a")
        report = store.prune()
        assert report.dry_run is False
        assert report.tapes_archived == []
        assert report.branches_archived == []
        assert len(store.list_runs()) == 1
    finally:
        store.close()


def test_vacuum_reclaims_disk_space_pruned_rows_left_behind(tmp_path):
    """tracefork-sis.59: `prune()`'s `DELETE FROM tapes/branches` frees pages
    *inside* the sqlite file but never shrinks the file itself -- `vacuum()`
    is the deliberately-separate, higher-risk step that actually reclaims
    that space, without touching (let alone hard-deleting) the archived rows
    prune() copied out first."""
    import os

    db_path = tmp_path / "store.db"
    store = TapeStore(str(db_path))
    try:
        run_ids = []
        for i in range(8):
            t = Tape(agent_name="w")
            t.append_exchange(os.urandom(80_000), os.urandom(80_000))
            rid = store.save_tape(t, run_id=f"big-run-{i}", created_at="2020-01-01T00:00:00+00:00")
            run_ids.append(rid)

        report = store.prune(run_ids=run_ids)
        assert sorted(report.tapes_archived) == sorted(run_ids)

        # Checkpoint (WAL) so the on-disk main file actually reflects the
        # post-prune state before measuring -- otherwise recently-committed
        # bytes may still be sitting in the -wal file, not the main one.
        store._con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        size_before_vacuum = os.path.getsize(db_path)

        size_before_report, size_after_report = store.vacuum()

        size_after_vacuum = os.path.getsize(db_path)

        assert size_after_vacuum < size_before_vacuum
        assert size_after_report < size_before_report

        # Archived rows survive untouched -- vacuum reclaims freed PAGES, it
        # never deletes the soft-archived data prune() already moved aside.
        archived = {
            r[0] for r in store._con.execute("SELECT run_id FROM tapes_archived").fetchall()
        }
        assert set(run_ids) <= archived
        for rid in run_ids:
            row = store._con.execute(
                "SELECT tape_bytes FROM tapes_archived WHERE run_id=?", (rid,)
            ).fetchone()
            assert row is not None and len(row[0]) > 0

        # Live tables are uncorrupted: still queryable, no rows lost.
        assert store.list_runs() == []
    finally:
        store.close()


def test_concurrent_writers_shared_store_serialized(tmp_path):
    """One shared connection across threads: the write lock must serialize the
    fan-out so two threads never open a transaction on it at once."""
    store = TapeStore(str(tmp_path / "store.db"))
    errors: list[BaseException] = []

    def worker(w: int) -> None:
        try:
            for j in range(5):
                store.save_tape(_small_tape(f"{w}-{j}".encode()), run_id=f"r{w}_{j}")
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    try:
        assert not errors, errors
        assert len(store.list_runs()) == 8 * 5
    finally:
        store.close()


# ── branches(parent_run_id) index ───────────────────────────────────────────
#
# `list_branches(parent_run_id)` -- WHERE parent_run_id=? ORDER BY
# created_at DESC -- is the store's hottest read filter (backs `report.py`'s
# fork-tree panel and `cli.py branch`'s DAG queries). A composite index on
# (parent_run_id, created_at DESC) lets SQLite satisfy the filter AND the
# ORDER BY straight off the index, with neither a full table scan nor a
# separate sort pass.

_LIST_BRANCHES_QUERY = (
    "SELECT branch_id, divergence_step, mutation_desc, created_at, branch_digest, "
    "confinement_tier FROM branches WHERE parent_run_id=? ORDER BY created_at DESC"
)


def test_list_branches_query_plan_avoids_scan_and_temp_sort(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        plan = store._con.execute(
            f"EXPLAIN QUERY PLAN {_LIST_BRANCHES_QUERY}", ("some-parent",)
        ).fetchall()
        plan_text = " | ".join(row[3] for row in plan)
        assert "SCAN branches" not in plan_text, plan_text
        assert "TEMP B-TREE" not in plan_text, plan_text
        assert "idx_branches_parent_run_id" in plan_text, plan_text
    finally:
        store.close()


def test_branches_parent_run_id_index_exists_on_fresh_and_migrated_db(tmp_path):
    """The index must exist both for a brand-new database (created straight
    from `_DDL`) and for a `store.db` that predates this index (simulated by
    dropping it and re-opening a `TapeStore` against the same file, which
    re-runs `_migrate_branch_metadata_columns`/`_DDL`)."""
    path = str(tmp_path / "store.db")
    store = TapeStore(path)
    try:
        names = {
            row[0]
            for row in store._con.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='branches'"
            ).fetchall()
        }
        assert "idx_branches_parent_run_id" in names
        store._con.execute("DROP INDEX idx_branches_parent_run_id")
    finally:
        store.close()

    reopened = TapeStore(path)
    try:
        names = {
            row[0]
            for row in reopened._con.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='branches'"
            ).fetchall()
        }
        assert "idx_branches_parent_run_id" in names
    finally:
        reopened.close()


def test_index_on_branches_parent_run_id_measures_speedup(tmp_path):
    """The bead's own claim: the composite index measurably speeds up
    `list_branches`'s filter+order query relative to a full scan + temp
    b-tree sort, at a dataset large enough for the difference to be stable.
    Same connection, same data, index dropped/restored in place -- isolates
    the index itself as the only variable."""
    import time

    store = TapeStore(str(tmp_path / "store.db"))
    try:
        target_parent = "needle-parent"
        store.save_tape(_small_tape(b"needle-tape"), run_id=target_parent)
        n_other = 3000
        for i in range(n_other):
            other_parent = f"other-parent-{i}"
            store.save_tape(_small_tape(f"o{i}".encode()), run_id=other_parent)
            store.save_branch(
                parent_run_id=other_parent,
                divergence_step=0,
                delta_tape=_small_tape(f"ob{i}".encode()),
                mutation_desc="filler",
            )
        for i in range(5):
            store.save_branch(
                parent_run_id=target_parent,
                divergence_step=i,
                delta_tape=_small_tape(f"nb{i}".encode()),
                mutation_desc="needle",
            )

        def timed(reps: int) -> float:
            start = time.perf_counter()
            for _ in range(reps):
                store._con.execute(_LIST_BRANCHES_QUERY, (target_parent,)).fetchall()
            return time.perf_counter() - start

        reps = 300
        with_index = timed(reps)

        store._con.execute("DROP INDEX idx_branches_parent_run_id")
        try:
            without_index = timed(reps)
        finally:
            store._con.execute(
                "CREATE INDEX idx_branches_parent_run_id "
                "ON branches(parent_run_id, created_at DESC)"
            )

        assert without_index > with_index * 1.5, (
            f"expected the index to measurably speed up list_branches's query "
            f"(indexed={with_index:.4f}s over {reps} reps, "
            f"unindexed={without_index:.4f}s over {reps} reps)"
        )
    finally:
        store.close()
