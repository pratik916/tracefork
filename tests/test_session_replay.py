"""Session-wide replay divergence rollup tests (tracefork-bge.65): pure
composition over the already-shipped `store.session_tapes()` BFS and the
existing `replay.ReplayVerifier` — no new engine logic. All offline/$0."""

from __future__ import annotations

import json

import anthropic
import httpx
from typer.testing import CliRunner

from tests.fakes import ScriptedFakeLLM, make_text_response
from tracefork.cli import app
from tracefork.fixtures import single_turn_agent, two_turn_agent
from tracefork.replay import DriftCause, DriftDoctor
from tracefork.session_replay import resolve_agent_manifest, session_divergence_rollup
from tracefork.store import TapeStore
from tracefork.tape import Tape
from tracefork.transport import TraceforkTransport

runner = CliRunner()


def _record(agent_fn, responses: list[bytes]) -> Tape:
    """Record a tape by running `agent_fn` against a `ScriptedFakeLLM` seeded
    with `responses` — mirrors `test_replay.py`'s `_record_tape` helper."""
    fake = ScriptedFakeLLM(responses)
    tape = Tape()
    transport = TraceforkTransport("record", tape, fake)
    client = anthropic.Anthropic(
        api_key="sk-ant-fake",
        http_client=httpx.Client(transport=transport),
        max_retries=0,
    )
    agent_fn(client)
    return tape


def _make_agent(content: str):
    """A one-exchange agent whose request body is parameterized by `content`
    — two agents built from different `content` values diverge from one
    another's recorded tape (different request bytes), the same "code
    changed" shape as `test_replay.py`'s `test_drift_doctor_classifies_code_change`."""

    def agent(client: anthropic.Anthropic) -> str:
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=100,
            messages=[{"role": "user", "content": content}],
        )
        return resp.content[0].text

    return agent


TEXT_RESP = make_text_response("Done.")


# ── clean session: every mapped tape replays bit-exact ──────────────────────


def test_rollup_returns_none_when_every_mapped_tape_replays_clean(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        root_agent = _make_agent("root task")
        child_agent = _make_agent("child task")
        store.save_tape(_record(root_agent, [TEXT_RESP]), run_id="root")
        store.save_tape(_record(child_agent, [TEXT_RESP]), run_id="child")
        session_id = store.create_session(root_run_id="root")
        store.add_spawn_edge(session_id=session_id, parent_run_id="root", child_run_id="child")

        result = session_divergence_rollup(
            store, session_id, {"root": root_agent, "child": child_agent}
        )

        assert result.diverged_run_id is None
        assert result.divergence is None
        assert result.checked_run_ids == ["root", "child"]
        assert result.skipped_run_ids == []
    finally:
        store.close()


# ── diverging CHILD is reported even though the root replays fine ──────────


def test_rollup_reports_diverging_child_not_root(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        root_agent = _make_agent("root task")
        child_agent_recorded = _make_agent("child task")
        child_agent_changed = _make_agent("child task — changed")
        store.save_tape(_record(root_agent, [TEXT_RESP]), run_id="root")
        store.save_tape(_record(child_agent_recorded, [TEXT_RESP]), run_id="child")
        session_id = store.create_session(root_run_id="root")
        store.add_spawn_edge(session_id=session_id, parent_run_id="root", child_run_id="child")

        result = session_divergence_rollup(
            store, session_id, {"root": root_agent, "child": child_agent_changed}
        )

        assert result.diverged_run_id == "child"
        assert result.checked_run_ids == ["root", "child"]
        assert result.skipped_run_ids == []
        assert result.divergence is not None
        assert DriftDoctor.classify(result.divergence) == DriftCause.CODE_CHANGE
    finally:
        store.close()


# ── a run_id absent from agent_fns is skipped, not crashed on or mistaken ───


def test_run_id_missing_from_agent_fns_is_skipped_not_crashed_or_counted(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        root_agent = _make_agent("root task")
        child_agent = _make_agent("child task")
        store.save_tape(_record(root_agent, [TEXT_RESP]), run_id="root")
        store.save_tape(_record(child_agent, [TEXT_RESP]), run_id="child")
        session_id = store.create_session(root_run_id="root")
        store.add_spawn_edge(session_id=session_id, parent_run_id="root", child_run_id="child")

        result = session_divergence_rollup(store, session_id, {"root": root_agent})

        assert result.diverged_run_id is None
        assert result.divergence is None
        assert result.checked_run_ids == ["root"]
        assert result.skipped_run_ids == ["child"]
    finally:
        store.close()


# ── diamond session: report the FIRST BFS-order divergence, not any match ──


def test_diamond_session_reports_first_bfs_order_divergence_not_any_match(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        recorded_agents = {rid: _make_agent(f"{rid} task") for rid in ("root", "a", "b", "c")}
        for rid, agent in recorded_agents.items():
            store.save_tape(_record(agent, [TEXT_RESP]), run_id=rid)

        session_id = store.create_session(root_run_id="root")
        # Diamond: root -> a, root -> b, a -> c, b -> c.
        store.add_spawn_edge(session_id=session_id, parent_run_id="root", child_run_id="a")
        store.add_spawn_edge(session_id=session_id, parent_run_id="root", child_run_id="b")
        store.add_spawn_edge(session_id=session_id, parent_run_id="a", child_run_id="c")
        store.add_spawn_edge(session_id=session_id, parent_run_id="b", child_run_id="c")
        assert store.session_tapes(session_id) == ["root", "a", "b", "c"]

        # Both "a" and "c" would diverge if replayed — the rollup must stop
        # at the FIRST BFS-order divergence ("a") and never even reach "c".
        agent_fns = dict(recorded_agents)
        agent_fns["a"] = _make_agent("a task — changed")
        agent_fns["c"] = _make_agent("c task — changed")

        result = session_divergence_rollup(store, session_id, agent_fns)

        assert result.diverged_run_id == "a"
        assert result.checked_run_ids == ["root", "a"]
        assert result.skipped_run_ids == []
        assert "b" not in result.checked_run_ids
        assert "c" not in result.checked_run_ids
    finally:
        store.close()


# ── LangChain-shaped planner/worker session (offline, no langchain) ────────


def test_langchain_shaped_planner_worker_session_round_trips_clean(tmp_path):
    """Substantiates the 'validated against one adapter first' scope note:
    a LangChain-style root-planner-delegates-to-worker session, built purely
    with `ScriptedFakeLLM` tapes (the same offline pattern
    `test_adapters_langchain.py` establishes) — no real `langchain` install
    needed."""
    store = TapeStore(str(tmp_path / "store.db"))
    try:

        def planner(client: anthropic.Anthropic) -> str:
            resp = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=100,
                messages=[{"role": "user", "content": "plan: delegate research to worker"}],
            )
            return resp.content[0].text

        def worker(client: anthropic.Anthropic) -> str:
            resp = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=100,
                messages=[{"role": "user", "content": "worker: perform research subtask"}],
            )
            return resp.content[0].text

        store.save_tape(_record(planner, [TEXT_RESP]), run_id="planner-run")
        store.save_tape(_record(worker, [TEXT_RESP]), run_id="worker-run")
        session_id = store.create_session(root_run_id="planner-run")
        store.add_spawn_edge(
            session_id=session_id,
            parent_run_id="planner-run",
            child_run_id="worker-run",
            spawn_reason="delegate research subtask",
        )

        result = session_divergence_rollup(
            store, session_id, {"planner-run": planner, "worker-run": worker}
        )

        assert result.diverged_run_id is None
        assert result.checked_run_ids == ["planner-run", "worker-run"]
        assert result.skipped_run_ids == []
    finally:
        store.close()


# ── resolve_agent_manifest: importlib "module:fn" resolution ────────────────


def test_resolve_agent_manifest_resolves_module_fn_entries():
    resolved = resolve_agent_manifest(
        {
            "root": "tracefork.fixtures:single_turn_agent",
            "child": "tracefork.fixtures:two_turn_agent",
        }
    )
    assert resolved == {"root": single_turn_agent, "child": two_turn_agent}


# ── CLI: tracefork session divergence ───────────────────────────────────────


def _seeded_session_store(tmp_path):
    """A session with a `single_turn_agent`-recorded root spawning a
    `two_turn_agent`-recorded child — real, importable fixture agents (the
    same ones `test_cli_smoke.py` uses) so `--agents-manifest` entries can
    name real `module:fn` paths."""
    db = tmp_path / "store.db"
    store = TapeStore(str(db))
    store.save_tape(_record(single_turn_agent, [TEXT_RESP]), run_id="root")
    store.save_tape(
        _record(two_turn_agent, [make_text_response("blue"), make_text_response("sky blue")]),
        run_id="child",
    )
    session_id = store.create_session(root_run_id="root")
    store.add_spawn_edge(session_id=session_id, parent_run_id="root", child_run_id="child")
    store.close()
    return db, session_id


def test_cli_session_divergence_clean_session_exits_zero(tmp_path):
    db, session_id = _seeded_session_store(tmp_path)
    manifest_path = tmp_path / "agents.json"
    manifest_path.write_text(
        json.dumps(
            {
                "root": "tracefork.fixtures:single_turn_agent",
                "child": "tracefork.fixtures:two_turn_agent",
            }
        )
    )

    result = runner.invoke(
        app,
        [
            "session",
            "divergence",
            session_id,
            "--agents-manifest",
            str(manifest_path),
            "--store",
            str(db),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "no divergence" in result.output


def test_cli_session_divergence_diverging_session_exits_one(tmp_path):
    db, session_id = _seeded_session_store(tmp_path)
    manifest_path = tmp_path / "agents.json"
    # "child" was recorded with `two_turn_agent`; mapping it to the
    # differently-shaped `synthetic_agent` forces a real divergence.
    manifest_path.write_text(
        json.dumps(
            {
                "root": "tracefork.fixtures:single_turn_agent",
                "child": "tracefork.validate:synthetic_agent",
            }
        )
    )

    result = runner.invoke(
        app,
        [
            "session",
            "divergence",
            session_id,
            "--agents-manifest",
            str(manifest_path),
            "--store",
            str(db),
        ],
    )
    assert result.exit_code == 1, result.output
    assert "child" in result.output


def test_cli_session_divergence_unknown_session_id_exits_nonzero(tmp_path):
    db = tmp_path / "store.db"
    TapeStore(str(db)).close()
    manifest_path = tmp_path / "agents.json"
    manifest_path.write_text(json.dumps({}))

    result = runner.invoke(
        app,
        [
            "session",
            "divergence",
            "no-such-session",
            "--agents-manifest",
            str(manifest_path),
            "--store",
            str(db),
        ],
    )
    assert result.exit_code != 0
