"""Offline unit tests for tracefork-bge.33's report_session.py:
`_session_to_data`/`generate_session_report` against a real 2-tape
parent->child spawn-edge session fixture."""

from __future__ import annotations

import json

import anthropic
import httpx

from tests.fakes import ScriptedFakeLLM, make_text_response
from tracefork.fixtures import single_turn_agent
from tracefork.report_session import _session_to_data, generate_session_report
from tracefork.store import TapeStore
from tracefork.tape import Tape
from tracefork.transport import TraceforkTransport

TEXT_RESP = make_text_response("4")


def _record(agent_fn, responses: list[bytes]) -> Tape:
    fake = ScriptedFakeLLM(responses)
    tape = Tape()
    transport = TraceforkTransport("record", tape, fake)
    client = anthropic.Anthropic(
        api_key="sk-ant-fake", http_client=httpx.Client(transport=transport), max_retries=0
    )
    agent_fn(client)
    return tape


def _seeded_session(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    store.save_tape(_record(single_turn_agent, [TEXT_RESP]), run_id="root")
    store.save_tape(_record(single_turn_agent, [TEXT_RESP]), run_id="child")
    session_id = store.create_session(root_run_id="root")
    store.add_spawn_edge(session_id=session_id, parent_run_id="root", child_run_id="child")
    return store, session_id


def _extract_session_data(content: str) -> dict:
    marker = "window.__TRACEFORK_SESSION_DATA__ = "
    start = content.find(marker) + len(marker)
    end = content.find(";\n", start)
    return json.loads(content[start:end])


def test_lanes_appear_in_session_tapes_bfs_order(tmp_path):
    store, session_id = _seeded_session(tmp_path)
    try:
        data = _session_to_data(store, session_id)
        run_ids = [lane["run_id"] for lane in data["lanes"]]
        assert run_ids == store.session_tapes(session_id)
        assert run_ids == ["root", "child"]
    finally:
        store.close()


def test_lane_spawn_lineage_matches_store(tmp_path):
    store, session_id = _seeded_session(tmp_path)
    try:
        data = _session_to_data(store, session_id)
        by_run_id = {lane["run_id"]: lane for lane in data["lanes"]}
        assert by_run_id["root"]["spawn_parent"] == store.spawn_parent("root")
        assert by_run_id["root"]["spawn_children"] == store.spawn_children("root")
        assert by_run_id["child"]["spawn_parent"] == store.spawn_parent("child")
        assert by_run_id["child"]["spawn_children"] == store.spawn_children("child")
        assert by_run_id["root"]["spawn_parent"] is None
        assert by_run_id["child"]["spawn_parent"] == "root"
    finally:
        store.close()


def test_run_id_absent_from_agent_map_yields_falsy_replay(tmp_path):
    store, session_id = _seeded_session(tmp_path)
    try:
        data = _session_to_data(store, session_id, agent_map={})
        for lane in data["lanes"]:
            assert lane["replay"] == {}
    finally:
        store.close()


def test_run_id_present_in_agent_map_yields_real_replay_receipt(tmp_path):
    store, session_id = _seeded_session(tmp_path)
    try:
        data = _session_to_data(
            store, session_id, agent_map={"root": single_turn_agent, "child": single_turn_agent}
        )
        by_run_id = {lane["run_id"]: lane for lane in data["lanes"]}
        assert by_run_id["root"]["replay"]["bit_exact"] is True
        assert by_run_id["child"]["replay"]["bit_exact"] is True
    finally:
        store.close()


def test_generate_session_report_escapes_script_breakout(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        evil = make_text_response("</script><img src=x onerror=alert(1)>")
        tape = _record(single_turn_agent, [evil])
        store.save_tape(tape, run_id="root")
        session_id = store.create_session(root_run_id="root")

        out = tmp_path / "board.html"
        generate_session_report(store, session_id, out)
        content = out.read_text()
        marker = "window.__TRACEFORK_SESSION_DATA__ = "
        start = content.find(marker)
        end = content.find(";\n", start)
        injected = content[start:end]
        assert "</script" not in injected
        data = _extract_session_data(content)
        preview = data["lanes"][0]["exchanges"][0]["preview"]
        assert preview == "</script><img src=x onerror=alert(1)>"
    finally:
        store.close()
