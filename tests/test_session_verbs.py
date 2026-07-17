"""Tests for `tracefork-bge.66`'s `session` CLI verb family
(record/replay/fork/blame/serve).

`src/tracefork/cli.py` is owned by the orchestrator in this workflow, so the
actual `@session_app.command()` Typer wrappers are NOT wired in by this
bead's own edits -- they ship as this bead's `cli_command` result for the
orchestrator to paste in. Every piece of REAL logic this bead adds lives in
`tracefork.session_ops` (a small, directly-testable module with no Typer
dependency) plus reuse of tracefork-bge.65's already-shipped
`tracefork.session_replay.session_divergence_rollup`, and is exercised here
directly, fully offline/$0, with no skip needed.

The CLI-wiring smoke tests at the bottom assert the exact end-to-end
behavior the bead spec names (exit codes, printed text) and are
`skipif`-gated on whether `cli.py`'s `session_app` actually registers each
command yet -- the same pattern already used by `test_session_cost.py`'s
`_COST_WIRED` guard. They activate automatically once the orchestrator
pastes this bead's `cli_command` code into `cli.py`.
"""

from __future__ import annotations

import sqlite3

import pytest
from typer.testing import CliRunner

from tests.fakes import make_text_response
from tracefork.cli import app, session_app
from tracefork.session_ops import (
    SpawnEdgeSpec,
    build_uniform_agent_manifest,
    ensure_run_in_session,
    parse_spawn_spec,
    record_session,
    session_deep_link_path,
)
from tracefork.session_replay import session_divergence_rollup
from tracefork.store import TapeStore
from tracefork.validate import _record_clean_tape, synthetic_agent

runner = CliRunner()


# ── parse_spawn_spec ─────────────────────────────────────────────────────────


def test_parse_spawn_spec_parses_parent_child_reason():
    spec = parse_spawn_spec("root:child:delegated subtask")
    assert spec == SpawnEdgeSpec("root", "child", "delegated subtask")


def test_parse_spawn_spec_defaults_reason_to_empty_string():
    spec = parse_spawn_spec("root:child")
    assert spec == SpawnEdgeSpec("root", "child", "")


def test_parse_spawn_spec_reason_may_itself_contain_a_colon():
    spec = parse_spawn_spec("root:child:reason: with a colon")
    assert spec.spawn_reason == "reason: with a colon"


@pytest.mark.parametrize(
    "bad_spec",
    ["no-colon-here", ":missing-parent", "missing-child:", ""],
)
def test_parse_spawn_spec_rejects_malformed_input(bad_spec):
    with pytest.raises(ValueError, match="PARENT:CHILD"):
        parse_spawn_spec(bad_spec)


# ── record_session ───────────────────────────────────────────────────────────


def test_record_session_creates_session_and_registers_every_spawn_edge(tmp_path):
    db = TapeStore(str(tmp_path / "store.db"))
    try:
        db.save_tape(_record_clean_tape(), run_id="root")
        db.save_tape(_record_clean_tape(), run_id="child-a")
        db.save_tape(_record_clean_tape(), run_id="child-b")

        session_id, edges = record_session(
            db, "root", ["root:child-a:delegated A", "child-a:child-b"]
        )

        assert edges == [
            SpawnEdgeSpec("root", "child-a", "delegated A"),
            SpawnEdgeSpec("child-a", "child-b", ""),
        ]
        # session_tapes' BFS proves every --spawn edge actually landed and is
        # reachable from the session's root, not merely inserted.
        assert db.session_tapes(session_id) == ["root", "child-a", "child-b"]
    finally:
        db.close()


def test_record_session_with_no_spawn_specs_still_creates_a_root_only_session(tmp_path):
    db = TapeStore(str(tmp_path / "store.db"))
    try:
        db.save_tape(_record_clean_tape(), run_id="root")
        session_id, edges = record_session(db, "root", [])
        assert edges == []
        assert db.session_tapes(session_id) == ["root"]
    finally:
        db.close()


def test_record_session_malformed_spawn_spec_raises_before_any_write(tmp_path):
    db = TapeStore(str(tmp_path / "store.db"))
    try:
        db.save_tape(_record_clean_tape(), run_id="root")
        with pytest.raises(ValueError, match="PARENT:CHILD"):
            record_session(db, "root", ["not-a-valid-spec"])
        # No session was created -- the malformed spec is caught before any write.
        assert db._con.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
    finally:
        db.close()


def test_record_session_propagates_integrity_error_on_dangling_child_run_id(tmp_path):
    db = TapeStore(str(tmp_path / "store.db"))
    try:
        db.save_tape(_record_clean_tape(), run_id="root")
        with pytest.raises(sqlite3.IntegrityError):
            record_session(db, "root", ["root:no-such-child"])
    finally:
        db.close()


# ── replay: build_uniform_agent_manifest + session_divergence_rollup reuse ──


def test_replay_rollup_passes_on_a_clean_two_tape_session(tmp_path):
    db = TapeStore(str(tmp_path / "store.db"))
    try:
        db.save_tape(_record_clean_tape(), run_id="root")
        db.save_tape(_record_clean_tape(), run_id="child")
        session_id = db.create_session(root_run_id="root")
        db.add_spawn_edge(session_id=session_id, parent_run_id="root", child_run_id="child")

        run_ids = db.session_tapes(session_id)
        manifest = build_uniform_agent_manifest(run_ids, synthetic_agent)
        assert manifest == {"root": synthetic_agent, "child": synthetic_agent}

        result = session_divergence_rollup(db, session_id, manifest)
        assert result.diverged_run_id is None
        assert result.checked_run_ids == ["root", "child"]
    finally:
        db.close()


def test_replay_rollup_unknown_session_id_raises_key_error(tmp_path):
    db = TapeStore(str(tmp_path / "store.db"))
    try:
        with pytest.raises(KeyError):
            session_divergence_rollup(db, "no-such-session", {})
    finally:
        db.close()


# ── ensure_run_in_session ─────────────────────────────────────────────────


def test_ensure_run_in_session_returns_run_ids_when_run_is_a_member(tmp_path):
    db = TapeStore(str(tmp_path / "store.db"))
    try:
        db.save_tape(_record_clean_tape(), run_id="root")
        db.save_tape(_record_clean_tape(), run_id="child")
        session_id = db.create_session(root_run_id="root")
        db.add_spawn_edge(session_id=session_id, parent_run_id="root", child_run_id="child")

        assert ensure_run_in_session(db, session_id, "child") == ["root", "child"]
    finally:
        db.close()


def test_ensure_run_in_session_rejects_run_id_outside_the_session(tmp_path):
    db = TapeStore(str(tmp_path / "store.db"))
    try:
        db.save_tape(_record_clean_tape(), run_id="root")
        db.save_tape(_record_clean_tape(), run_id="stray")
        session_id = db.create_session(root_run_id="root")

        with pytest.raises(ValueError, match="not reachable"):
            ensure_run_in_session(db, session_id, "stray")
    finally:
        db.close()


def test_ensure_run_in_session_unknown_session_id_raises_key_error(tmp_path):
    db = TapeStore(str(tmp_path / "store.db"))
    try:
        db.save_tape(_record_clean_tape(), run_id="root")
        with pytest.raises(KeyError):
            ensure_run_in_session(db, "no-such-session", "root")
    finally:
        db.close()


# ── session_deep_link_path ───────────────────────────────────────────────


def test_session_deep_link_path_format():
    assert session_deep_link_path("abc123") == "/api/session/abc123"


# ── CLI wiring smoke tests (skip-gated until cli.py is patched) ─────────────

_REGISTERED = {c.name for c in session_app.registered_commands}
_RECORD_WIRED = "record" in _REGISTERED
_REPLAY_WIRED = "replay" in _REGISTERED
_FORK_WIRED = "fork" in _REGISTERED
_BLAME_WIRED = "blame" in _REGISTERED
_SERVE_WIRED = "serve" in _REGISTERED

_WIRING_REASON = (
    "`session {0}` not yet wired into cli.py session_app (see cli_command in bead result)"
)


def _seed_two_tape_session(db_path):
    store = TapeStore(str(db_path))
    root_id = store.save_tape(_record_clean_tape(), run_id="root")
    child_id = store.save_tape(_record_clean_tape(), run_id="child")
    session_id = store.create_session(root_run_id=root_id)
    store.add_spawn_edge(session_id=session_id, parent_run_id=root_id, child_run_id=child_id)
    store.close()
    return session_id, root_id, child_id


@pytest.mark.skipif(not _RECORD_WIRED, reason=_WIRING_REASON.format("record"))
def test_cli_session_record_batch_creates_and_registers_spawn_edges(tmp_path):
    db = tmp_path / "store.db"
    store = TapeStore(str(db))
    store.save_tape(_record_clean_tape(), run_id="root")
    store.save_tape(_record_clean_tape(), run_id="child")
    store.close()

    result = runner.invoke(
        app,
        ["session", "record", "root", "--spawn", "root:child:delegated", "--store", str(db)],
    )
    assert result.exit_code == 0, result.output

    session_id = None
    for line in result.output.splitlines():
        if "session_id" in line:
            session_id = line.split()[-1]
            break
    assert session_id

    store = TapeStore(str(db))
    try:
        assert store.session_tapes(session_id) == ["root", "child"]
    finally:
        store.close()


@pytest.mark.skipif(not _REPLAY_WIRED, reason=_WIRING_REASON.format("replay"))
def test_cli_session_replay_clean_two_tape_session_exits_zero(tmp_path):
    db = tmp_path / "store.db"
    session_id, _root_id, _child_id = _seed_two_tape_session(db)

    result = runner.invoke(
        app,
        [
            "session",
            "replay",
            session_id,
            "--agent",
            "tracefork.validate:synthetic_agent",
            "--store",
            str(db),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "every tape replayed bit-exact" in result.output


@pytest.mark.skipif(not _REPLAY_WIRED, reason=_WIRING_REASON.format("replay"))
def test_cli_session_replay_unknown_session_id_exits_nonzero(tmp_path):
    db = tmp_path / "store.db"
    TapeStore(str(db)).close()

    result = runner.invoke(
        app,
        [
            "session",
            "replay",
            "no-such-session",
            "--agent",
            "tracefork.validate:synthetic_agent",
            "--store",
            str(db),
        ],
    )
    assert result.exit_code != 0


@pytest.mark.skipif(not _FORK_WIRED, reason=_WIRING_REASON.format("fork"))
def test_cli_session_fork_at_last_step_is_offline_and_exits_zero(tmp_path):
    db = tmp_path / "store.db"
    session_id, root_id, _child_id = _seed_two_tape_session(db)

    resp_path = tmp_path / "mutated.bytes"
    resp_path.write_bytes(make_text_response("FAIL — cancelled"))

    result = runner.invoke(
        app,
        [
            "session",
            "fork",
            session_id,
            root_id,
            "--step",
            "1",
            "--response",
            str(resp_path),
            "--agent",
            "tracefork.validate:synthetic_agent",
            "--store",
            str(db),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Fork created" in result.output


@pytest.mark.skipif(not _FORK_WIRED, reason=_WIRING_REASON.format("fork"))
def test_cli_session_fork_run_id_outside_session_exits_nonzero_before_any_fork(tmp_path):
    db = tmp_path / "store.db"
    session_id, _root_id, _child_id = _seed_two_tape_session(db)
    store = TapeStore(str(db))
    store.save_tape(_record_clean_tape(), run_id="stray")
    store.close()

    resp_path = tmp_path / "mutated.bytes"
    resp_path.write_bytes(make_text_response("irrelevant"))

    result = runner.invoke(
        app,
        [
            "session",
            "fork",
            session_id,
            "stray",
            "--step",
            "1",
            "--response",
            str(resp_path),
            "--agent",
            "tracefork.validate:synthetic_agent",
            "--store",
            str(db),
        ],
    )
    assert result.exit_code != 0
    assert "not reachable" in result.output

    store = TapeStore(str(db))
    try:
        assert store.list_branches("stray") == []
    finally:
        store.close()


@pytest.mark.skipif(not _BLAME_WIRED, reason=_WIRING_REASON.format("blame"))
def test_cli_session_blame_budget_gate_blocks_overspend_before_any_network_call(tmp_path):
    db = tmp_path / "store.db"
    session_id, root_id, _child_id = _seed_two_tape_session(db)

    result = runner.invoke(
        app,
        [
            "session",
            "blame",
            session_id,
            root_id,
            "--agent",
            "tracefork.validate:synthetic_agent",
            "--store",
            str(db),
            "--budget",
            "0",
        ],
    )
    assert result.exit_code == 1, result.output
    assert "exceeds budget" in result.output


@pytest.mark.skipif(not _BLAME_WIRED, reason=_WIRING_REASON.format("blame"))
def test_cli_session_blame_run_id_outside_session_exits_nonzero(tmp_path):
    db = tmp_path / "store.db"
    session_id, _root_id, _child_id = _seed_two_tape_session(db)
    store = TapeStore(str(db))
    store.save_tape(_record_clean_tape(), run_id="stray")
    store.close()

    result = runner.invoke(
        app,
        [
            "session",
            "blame",
            session_id,
            "stray",
            "--agent",
            "tracefork.validate:synthetic_agent",
            "--store",
            str(db),
        ],
    )
    assert result.exit_code != 0
    assert "not reachable" in result.output


@pytest.mark.skipif(not _SERVE_WIRED, reason=_WIRING_REASON.format("serve"))
def test_cli_session_serve_wires_host_and_port_and_prints_deep_link(tmp_path, monkeypatch):
    import uvicorn as uvicorn_module

    db = tmp_path / "store.db"
    session_id, _root_id, _child_id = _seed_two_tape_session(db)

    captured: dict = {}

    def fake_run(app_obj, *, host, port, workers, log_level):
        captured["host"] = host
        captured["port"] = port

    monkeypatch.setattr(uvicorn_module, "run", fake_run)

    result = runner.invoke(
        app, ["session", "serve", session_id, "--store", str(db), "--port", "9922"]
    )
    assert result.exit_code == 0, result.output
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 9922
    assert f"/api/session/{session_id}" in result.output


@pytest.mark.skipif(not _SERVE_WIRED, reason=_WIRING_REASON.format("serve"))
def test_cli_session_serve_unknown_session_id_exits_nonzero_and_never_calls_uvicorn(
    tmp_path, monkeypatch
):
    import uvicorn as uvicorn_module

    db = tmp_path / "store.db"
    TapeStore(str(db)).close()

    called = {"run": False}

    def fake_run(*args, **kwargs):
        called["run"] = True

    monkeypatch.setattr(uvicorn_module, "run", fake_run)

    result = runner.invoke(app, ["session", "serve", "no-such-session", "--store", str(db)])
    assert result.exit_code != 0
    assert called["run"] is False
