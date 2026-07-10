"""Orchestration session model tests (tracefork-bge.12): `sessions`/`spawn_edges`
persist the cross-agent spawn-lineage/delegation graph as its OWN schema,
distinct from `Tape.async_batches` (per-agent asyncio fan-out — unrelated).
`create_session`/`add_spawn_edge`/`session_tapes`/`spawn_children`/
`spawn_parent` mirror `save_tape`/`save_branch`'s BEGIN IMMEDIATE +
`self._write_lock` write discipline; `SessionStore` is a NEW, separate
runtime_checkable Protocol so `StorageBackend` itself stays unchanged. All
offline/$0."""

from __future__ import annotations

import sqlite3

import pytest
from typer.testing import CliRunner

from tracefork.cli import app
from tracefork.store import SessionStore, StorageBackend, TapeStore
from tracefork.tape import Tape

runner = CliRunner()


def _small_tape(tag: bytes = b"x") -> Tape:
    t = Tape(agent_name="w")
    t.append_exchange(b"req-" + tag, b"resp-" + tag)
    return t


# ── create_session / add_spawn_edge round-trip ──────────────────────────────


def test_create_session_and_add_spawn_edge_round_trip(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        store.save_tape(_small_tape(b"root"), run_id="root")
        store.save_tape(_small_tape(b"child"), run_id="child")

        session_id = store.create_session(root_run_id="root")
        session = store.get_session(session_id)
        assert session == {
            "session_id": session_id,
            "root_run_id": "root",
            "created_at": "",
        }

        edge_id = store.add_spawn_edge(
            session_id=session_id,
            parent_run_id="root",
            child_run_id="child",
            spawn_reason="delegate",
        )
        assert isinstance(edge_id, str) and edge_id

        assert store.spawn_children("root") == ["child"]
        assert store.spawn_parent("child") == "root"
    finally:
        store.close()


# ── session_tapes BFS over a diamond spawn graph ────────────────────────────


def test_session_tapes_bfs_over_diamond_returns_all_reachable(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        for rid in ("root", "a", "b", "c"):
            store.save_tape(_small_tape(rid.encode()), run_id=rid)

        session_id = store.create_session(root_run_id="root")
        # Diamond: root -> a, root -> b, a -> c, b -> c.
        store.add_spawn_edge(session_id=session_id, parent_run_id="root", child_run_id="a")
        store.add_spawn_edge(session_id=session_id, parent_run_id="root", child_run_id="b")
        store.add_spawn_edge(session_id=session_id, parent_run_id="a", child_run_id="c")
        store.add_spawn_edge(session_id=session_id, parent_run_id="b", child_run_id="c")

        tapes = store.session_tapes(session_id)
        assert set(tapes) == {"root", "a", "b", "c"}
        assert len(tapes) == 4  # "c" reached via two paths, still deduplicated
    finally:
        store.close()


# ── spawn_parent / spawn_children on root and leaf tapes ────────────────────


def test_spawn_parent_and_children_correct_on_root_and_leaf(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        for rid in ("root", "mid", "leaf"):
            store.save_tape(_small_tape(rid.encode()), run_id=rid)

        session_id = store.create_session(root_run_id="root")
        store.add_spawn_edge(session_id=session_id, parent_run_id="root", child_run_id="mid")
        store.add_spawn_edge(session_id=session_id, parent_run_id="mid", child_run_id="leaf")

        # Root: has a child, no parent.
        assert store.spawn_children("root") == ["mid"]
        assert store.spawn_parent("root") is None

        # Leaf: has a parent, no children.
        assert store.spawn_children("leaf") == []
        assert store.spawn_parent("leaf") == "mid"
    finally:
        store.close()


# ── FK violation on an unknown child_run_id ─────────────────────────────────


def test_add_spawn_edge_fk_violation_on_unknown_child_run_id(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        store.save_tape(_small_tape(b"root"), run_id="root")
        session_id = store.create_session(root_run_id="root")

        with pytest.raises(sqlite3.IntegrityError):
            store.add_spawn_edge(
                session_id=session_id, parent_run_id="root", child_run_id="does-not-exist"
            )
    finally:
        store.close()


def test_create_session_fk_violation_on_unknown_root_run_id(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        with pytest.raises(sqlite3.IntegrityError):
            store.create_session(root_run_id="does-not-exist")
    finally:
        store.close()


# ── StorageBackend stays unchanged; SessionStore is a NEW, separate protocol ─


def test_storage_backend_unchanged_session_store_is_separate_protocol(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        assert isinstance(store, StorageBackend)
        assert isinstance(store, SessionStore)
        assert not hasattr(StorageBackend, "create_session")
        assert not hasattr(StorageBackend, "add_spawn_edge")
    finally:
        store.close()


# ── CLI: session create / spawn / show ───────────────────────────────────────


def test_cli_session_create_spawn_show_exit_codes(tmp_path):
    db = tmp_path / "store.db"
    store = TapeStore(str(db))
    try:
        store.save_tape(_small_tape(b"root"), run_id="root")
        store.save_tape(_small_tape(b"child"), run_id="child")
    finally:
        store.close()

    create_result = runner.invoke(app, ["session", "create", "root", "--store", str(db)])
    assert create_result.exit_code == 0, create_result.output
    session_id = None
    for line in create_result.output.splitlines():
        if "session_id" in line:
            session_id = line.split()[-1]
            break
    assert session_id

    spawn_result = runner.invoke(
        app,
        [
            "session",
            "spawn",
            session_id,
            "root",
            "child",
            "--reason",
            "delegated subtask",
            "--store",
            str(db),
        ],
    )
    assert spawn_result.exit_code == 0, spawn_result.output

    show_result = runner.invoke(app, ["session", "show", session_id, "--store", str(db)])
    assert show_result.exit_code == 0, show_result.output
    assert session_id in show_result.output
    assert "root" in show_result.output
    assert "child" in show_result.output


def test_cli_session_show_unknown_session_exits_nonzero(tmp_path):
    db = tmp_path / "store.db"
    TapeStore(str(db)).close()

    result = runner.invoke(app, ["session", "show", "no-such-session", "--store", str(db)])
    assert result.exit_code != 0


# ── server.py: GET /api/session/{session_id} ────────────────────────────────


def test_server_get_session_returns_json_and_404s_on_unknown_session(tmp_path):
    """The additive `/api/session/{id}` surface: 200 with root_run_id +
    reachable tapes on a known session, 404 (via the same KeyError -> 404
    pattern as `get_run`/`get_branch`) on an unknown one."""
    from fastapi.testclient import TestClient

    from tracefork.server import app as fastapi_app
    from tracefork.server import init_store

    db = tmp_path / "store.db"
    store = TapeStore(str(db))
    store.save_tape(_small_tape(b"root"), run_id="root")
    store.save_tape(_small_tape(b"child"), run_id="child")
    session_id = store.create_session(root_run_id="root")
    store.add_spawn_edge(session_id=session_id, parent_run_id="root", child_run_id="child")
    store.close()

    init_store(str(db))
    client = TestClient(fastapi_app)

    ok = client.get(f"/api/session/{session_id}")
    assert ok.status_code == 200
    body = ok.json()
    assert body["session_id"] == session_id
    assert body["root_run_id"] == "root"
    assert set(body["tapes"]) == {"root", "child"}

    missing = client.get("/api/session/does-not-exist")
    assert missing.status_code == 404
