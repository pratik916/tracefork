"""CLI smoke test for tracefork-bge.33's `tracefork session board` command:
session create -> session spawn -> session board end-to-end via Typer's
CliRunner against a temp store.db."""

from __future__ import annotations

import json

import anthropic
import httpx
from typer.testing import CliRunner

from tests.fakes import ScriptedFakeLLM, make_text_response
from tracefork.cli import app
from tracefork.store import TapeStore
from tracefork.tape import Tape
from tracefork.transport import TraceforkTransport

runner = CliRunner()


def _record_tape() -> Tape:
    fake = ScriptedFakeLLM([make_text_response("hi")])
    tape = Tape()
    transport = TraceforkTransport("record", tape, fake)
    client = anthropic.Anthropic(
        api_key="sk-ant-fake", http_client=httpx.Client(transport=transport), max_retries=0
    )
    client.messages.create(
        model="claude-sonnet-4-6", max_tokens=100, messages=[{"role": "user", "content": "hello"}]
    )
    return tape


def _session_id_from_output(output: str) -> str:
    return output.split("session_id")[1].split()[0].strip()


def test_session_board_cli_end_to_end(tmp_path):
    db = tmp_path / "store.db"
    store = TapeStore(str(db))
    store.save_tape(_record_tape(), run_id="root")
    store.save_tape(_record_tape(), run_id="child")
    store.close()

    create_result = runner.invoke(app, ["session", "create", "root", "--store", str(db)])
    assert create_result.exit_code == 0, create_result.output
    session_id = _session_id_from_output(create_result.output)

    spawn_result = runner.invoke(
        app, ["session", "spawn", session_id, "root", "child", "--store", str(db)]
    )
    assert spawn_result.exit_code == 0, spawn_result.output

    out = tmp_path / "board.html"
    board_result = runner.invoke(
        app, ["session", "board", session_id, "--store", str(db), "-o", str(out)]
    )
    assert board_result.exit_code == 0, board_result.output
    assert out.exists()
    content = out.read_text()
    assert '"root"' in content
    assert '"child"' in content
    assert "__TRACEFORK_SESSION_DATA__" in content


def test_session_board_cli_with_agent_map_embeds_real_replay_receipt(tmp_path):
    from tracefork.fixtures import single_turn_agent

    db = tmp_path / "store.db"
    store = TapeStore(str(db))
    fake = ScriptedFakeLLM([make_text_response("4")])
    tape = Tape()
    transport = TraceforkTransport("record", tape, fake)
    client = anthropic.Anthropic(
        api_key="sk-ant-fake", http_client=httpx.Client(transport=transport), max_retries=0
    )
    single_turn_agent(client)
    store.save_tape(tape, run_id="root")
    store.close()

    session_result = runner.invoke(app, ["session", "create", "root", "--store", str(db)])
    assert session_result.exit_code == 0, session_result.output
    session_id = _session_id_from_output(session_result.output)

    agent_map_path = tmp_path / "agent_map.json"
    agent_map_path.write_text(json.dumps({"root": "tracefork.fixtures:single_turn_agent"}))

    out = tmp_path / "board.html"
    result = runner.invoke(
        app,
        [
            "session",
            "board",
            session_id,
            "--store",
            str(db),
            "--agent-map",
            str(agent_map_path),
            "-o",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    content = out.read_text()
    assert '"bit_exact": true' in content
