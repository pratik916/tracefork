"""Offline tests for tracefork-bge.54: the report accepts a Shapley
necessity/sufficiency JSON and renders a per-step quadrant badge inline in
the Timeline panel (no new panel)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import anthropic
import httpx
from typer.testing import CliRunner

from tests.fakes import ScriptedFakeLLM, make_text_response
from tracefork.cli import app
from tracefork.report import _tape_to_data, generate_report
from tracefork.store import TapeStore
from tracefork.tape import Tape
from tracefork.transport import TraceforkTransport

runner = CliRunner()

TEXT_RESP = make_text_response("Hello world")


def _make_tape() -> Tape:
    fake = ScriptedFakeLLM([TEXT_RESP])
    tape = Tape(agent_name="test_agent")
    transport = TraceforkTransport("record", tape, fake)
    client = anthropic.Anthropic(
        api_key="sk-ant-fake", http_client=httpx.Client(transport=transport), max_retries=0
    )
    client.messages.create(
        model="claude-sonnet-4-6", max_tokens=100, messages=[{"role": "user", "content": "Hello"}]
    )
    return tape


def _extract_data(content: str) -> dict:
    marker = "window.__TRACEFORK_DATA__ = "
    start = content.find(marker) + len(marker)
    end = content.find(";\n", start)
    return json.loads(content[start:end])


def test_tape_to_data_includes_shapley_when_provided():
    tape = _make_tape()
    shapley = {
        0: {
            "necessity": True,
            "necessity_score": 0.81,
            "sufficiency": False,
            "sufficiency_score": 0.12,
            "shapley_value": 0.81,
            "interpretation": "decisive",
        }
    }
    data = _tape_to_data(tape, shapley=shapley)
    assert data["shapley"] == shapley


def test_tape_to_data_defaults_shapley_to_empty_dict():
    tape = _make_tape()
    data = _tape_to_data(tape)
    assert data["shapley"] == {}


def test_cli_report_shapley_report_flag_embeds_quadrant_data(tmp_path):
    db = tmp_path / "store.db"
    store = TapeStore(str(db))
    tape = _make_tape()
    run_id = store.save_tape(tape, run_id="run1")
    store.close()

    shapley_path = tmp_path / f"shapley_{run_id}.json"
    shapley_path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "step_index": 0,
                        "necessity": True,
                        "necessity_score": 0.9,
                        "sufficiency": False,
                        "sufficiency_score": 0.05,
                        "shapley_value": 0.9,
                        "interpretation": "decisive",
                    }
                ]
            }
        )
    )

    out = tmp_path / "report.html"
    result = runner.invoke(
        app,
        [
            "report",
            run_id,
            "--store",
            str(db),
            "--shapley-report",
            str(shapley_path),
            "-o",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    data = _extract_data(out.read_text())
    assert data["shapley"]["0"]["necessity"] is True
    assert data["shapley"]["0"]["necessity_score"] == 0.9
    assert data["shapley"]["0"]["sufficiency"] is False
    assert data["shapley"]["0"]["sufficiency_score"] == 0.05


def test_cli_report_without_shapley_report_flag_defaults_empty(tmp_path):
    db = tmp_path / "store.db"
    store = TapeStore(str(db))
    tape = _make_tape()
    run_id = store.save_tape(tape, run_id="run1")
    store.close()

    out = tmp_path / "report.html"
    result = runner.invoke(app, ["report", run_id, "--store", str(db), "-o", str(out)])
    assert result.exit_code == 0, result.output
    data = _extract_data(out.read_text())
    assert data["shapley"] == {}


def test_report_html_ships_shapley_badge_wiring():
    tape = _make_tape()
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "report.html"
        generate_report(tape, out)
        content = out.read_text()
        assert "shapleyQuadrantHtml" in content
        assert "shapley-badge" in content
