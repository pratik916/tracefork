"""bundle.py tests: lossless tape+branch export/import (portable bundle,
tracefork-bge.16). A bundle is a second, smaller store.db — export copies
`tapes`/`branches` BLOB columns byte-for-byte (never Tape.from_bytes/to_bytes
round-tripped) and import goes through the CAS-guarded save_tape/save_branch
write path (never raw INSERT), so a genuine content collision on import is
caught, not silently clobbered. All offline/$0."""

from __future__ import annotations

import sqlite3

import pytest

from tracefork.bundle import export_bundle, import_bundle
from tracefork.replay import ReplayVerifier
from tracefork.store import TapeConflictError, TapeStore
from tracefork.tape import Tape
from tracefork.validate import _record_clean_tape, synthetic_agent


def _tape(tag: str = "x") -> Tape:
    t = Tape(agent_name=f"agent-{tag}")
    t.append_exchange(f"req-{tag}".encode(), f"resp-{tag}".encode())
    return t


def _seeded_store_with_branches(tmp_path) -> tuple[TapeStore, str, list[str]]:
    db = tmp_path / "store.db"
    store = TapeStore(str(db))
    run_id = store.save_tape(_tape("run"), run_id="run-a")
    branch_ids = [
        store.save_branch(
            parent_run_id=run_id,
            divergence_step=0,
            delta_tape=_tape("branch-1"),
            mutation_desc="mutation one",
        ),
        store.save_branch(
            parent_run_id=run_id,
            divergence_step=0,
            delta_tape=_tape("branch-2"),
            mutation_desc="mutation two",
        ),
    ]
    return store, run_id, branch_ids


def _raw_blob(db_path, table: str, id_col: str, id_val: str, blob_col: str) -> bytes:
    con = sqlite3.connect(str(db_path))
    try:
        row = con.execute(f"SELECT {blob_col} FROM {table} WHERE {id_col}=?", (id_val,)).fetchone()
        return bytes(row[0])
    finally:
        con.close()


def test_export_bundle_produces_a_store_a_plain_tapestore_can_open(tmp_path):
    store, run_id, branch_ids = _seeded_store_with_branches(tmp_path)
    try:
        bundle_path = tmp_path / "bundle.db"
        result = export_bundle(store, run_id, str(bundle_path))
        assert result.run_id == run_id
        assert sorted(result.branch_ids) == sorted(branch_ids)
    finally:
        store.close()

    assert bundle_path.exists()
    bundle = TapeStore(str(bundle_path))
    try:
        assert [r["run_id"] for r in bundle.list_runs()] == [run_id]
        bundle_branch_ids = {b["branch_id"] for b in bundle.list_branches(run_id)}
        assert bundle_branch_ids == set(branch_ids)
    finally:
        bundle.close()


def test_export_bundle_list_runs_and_branches_match_originals_digest_equal(tmp_path):
    store, run_id, branch_ids = _seeded_store_with_branches(tmp_path)
    try:
        original_tape_digest = store.load_tape(run_id).digest()
        original_branch_digests = {
            bid: store.load_branch(bid)["delta_tape"].digest() for bid in branch_ids
        }
        bundle_path = tmp_path / "bundle.db"
        export_bundle(store, run_id, str(bundle_path))
    finally:
        store.close()

    bundle = TapeStore(str(bundle_path))
    try:
        assert bundle.load_tape(run_id).digest() == original_tape_digest
        for bid, digest in original_branch_digests.items():
            assert bundle.load_branch(bid)["delta_tape"].digest() == digest
    finally:
        bundle.close()


def test_export_bundle_copies_raw_blob_bytes_verbatim_no_reencode(tmp_path):
    """The bundle's stored tape_bytes/delta_tape_bytes must be the exact same
    bytes as the source store's — a raw-byte compare, not just a digest
    compare, proving no decode/re-encode round trip happened on export."""
    store, run_id, branch_ids = _seeded_store_with_branches(tmp_path)
    source_db = tmp_path / "store.db"
    store.close()

    bundle_path = tmp_path / "bundle.db"
    store = TapeStore(str(source_db))
    try:
        export_bundle(store, run_id, str(bundle_path))
    finally:
        store.close()

    source_tape_bytes = _raw_blob(source_db, "tapes", "run_id", run_id, "tape_bytes")
    bundle_tape_bytes = _raw_blob(bundle_path, "tapes", "run_id", run_id, "tape_bytes")
    assert source_tape_bytes == bundle_tape_bytes

    for bid in branch_ids:
        source_bytes = _raw_blob(source_db, "branches", "branch_id", bid, "delta_tape_bytes")
        bundle_bytes = _raw_blob(bundle_path, "branches", "branch_id", bid, "delta_tape_bytes")
        assert source_bytes == bundle_bytes


def test_export_bundle_unknown_run_id_raises_key_error(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        with pytest.raises(KeyError):
            export_bundle(store, "does-not-exist", str(tmp_path / "bundle.db"))
    finally:
        store.close()


def test_import_bundle_into_fresh_store_recreates_identical_ids_byte_identical(tmp_path):
    store, run_id, branch_ids = _seeded_store_with_branches(tmp_path)
    bundle_path = tmp_path / "bundle.db"
    export_bundle(store, run_id, str(bundle_path))
    store.close()

    target_path = tmp_path / "target.db"
    target = TapeStore(str(target_path))
    try:
        result = import_bundle(target, str(bundle_path))
        assert result.run_ids == [run_id]
        assert sorted(result.branch_ids) == sorted(branch_ids)
    finally:
        target.close()

    bundle_tape_bytes = _raw_blob(bundle_path, "tapes", "run_id", run_id, "tape_bytes")
    target_tape_bytes = _raw_blob(target_path, "tapes", "run_id", run_id, "tape_bytes")
    assert bundle_tape_bytes == target_tape_bytes

    for bid in branch_ids:
        bundle_bytes = _raw_blob(bundle_path, "branches", "branch_id", bid, "delta_tape_bytes")
        target_bytes = _raw_blob(target_path, "branches", "branch_id", bid, "delta_tape_bytes")
        assert bundle_bytes == target_bytes


def test_import_bundle_same_run_id_identical_content_is_idempotent(tmp_path):
    store, run_id, branch_ids = _seeded_store_with_branches(tmp_path)
    bundle_path = tmp_path / "bundle.db"
    export_bundle(store, run_id, str(bundle_path))

    # Import into the SAME store that already has this exact run_id/branches.
    result = import_bundle(store, str(bundle_path))
    store.close()

    assert result.run_ids == [run_id]
    assert sorted(result.branch_ids) == sorted(branch_ids)


def test_import_bundle_same_run_id_different_content_raises_conflict(tmp_path):
    store, run_id, _branch_ids = _seeded_store_with_branches(tmp_path)
    bundle_path = tmp_path / "bundle.db"
    export_bundle(store, run_id, str(bundle_path))
    store.close()

    # A second store that already has run_id but with DIFFERENT content.
    target_path = tmp_path / "target.db"
    target = TapeStore(str(target_path))
    try:
        target.save_tape(_tape("conflicting-content"), run_id=run_id)
        with pytest.raises(TapeConflictError):
            import_bundle(target, str(bundle_path))
    finally:
        target.close()


def test_bundle_round_trip_then_replay_verifier_is_bit_exact(tmp_path):
    """export then import into a brand-new store, then ReplayVerifier against
    the imported tape -> bit_exact=True."""
    store = TapeStore(str(tmp_path / "store.db"))
    run_id = store.save_tape(_record_clean_tape(), run_id="clean-run")

    bundle_path = tmp_path / "bundle.db"
    export_bundle(store, run_id, str(bundle_path))
    store.close()

    target = TapeStore(str(tmp_path / "target.db"))
    try:
        import_bundle(target, str(bundle_path))
        imported_tape = target.load_tape(run_id)
    finally:
        target.close()

    result = ReplayVerifier(imported_tape, synthetic_agent).verify()
    assert result.bit_exact is True
