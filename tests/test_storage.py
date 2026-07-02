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
from tracefork.store import StorageBackend, TapeStore
from tracefork.tape import Tape, open_sqlite

# The exact digest of `_golden_tape()`, frozen at the pre-header format. If a change
# to the version envelope ever perturbs this, the header has leaked into the
# content-addressed hash chain — which the must-have forbids.
GOLDEN_DIGEST = "93302637a9c2d12f983c3d2bae5150d63e902f427e8555b42b666c9e7fcb8c4e"

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
