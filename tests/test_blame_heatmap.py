"""Offline tests for tracefork-bge.35: the fork-tree panel's causal heatmap
overlay (necessity/sufficiency/responsible-set) built on already-persisted
`causal_edges` — no new blame math, purely additive threading + CSS/JS
wiring in web/report.html and one additive line in server.py's get_run."""

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

_SAMPLE_EDGES = [
    {
        "edge_id": "run1:0:blame",
        "run_id": "run1",
        "step_index": 0,
        "method": "blame",
        "flip_rate": 0.9,
        "ci_lo": 0.7,
        "ci_hi": 0.98,
        "ci_method": "wilson",
        "p_value": 0.01,
        "q_value": 0.02,
        "responsible": True,
        "necessity": None,
        "sufficiency": None,
        "shapley_value": None,
        "created_at": "",
    },
    {
        "edge_id": "run1:0:shapley",
        "run_id": "run1",
        "step_index": 0,
        "method": "shapley",
        "flip_rate": None,
        "ci_lo": None,
        "ci_hi": None,
        "ci_method": None,
        "p_value": None,
        "q_value": None,
        "responsible": None,
        "necessity": True,
        "sufficiency": False,
        "shapley_value": 0.81,
        "created_at": "",
    },
]


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


def test_tape_to_data_defaults_causal_edges_to_empty_list():
    tape = _make_tape()
    assert _tape_to_data(tape)["causal_edges"] == []


def test_tape_to_data_includes_populated_causal_edges_list():
    tape = _make_tape()
    data = _tape_to_data(tape, causal_edges=_SAMPLE_EDGES)
    assert data["causal_edges"] == _SAMPLE_EDGES


def test_report_embeds_causal_edges_and_heatmap_css_classes():
    tape = _make_tape()
    branches = [
        {
            "branch_id": "b1",
            "divergence_step": 0,
            "mutation_desc": "mutated response",
            "created_at": "2026-01-01T00:00:00",
            "branch_digest": "abc123",
        }
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "report.html"
        generate_report(tape, out, branches=branches, causal_edges=_SAMPLE_EDGES)
        content = out.read_text()
        data = _extract_data(content)
        assert data["causal_edges"] == _SAMPLE_EDGES
        assert "forktree-node-responsible" in content
        assert "forktree-node-necessary" in content
        assert "forktree-node-sufficient" in content


def test_cli_report_threads_causal_edges_for_run_id_path(tmp_path):
    from tracefork.blame import BlameReport, CIMethod, FlipRateResult

    db = tmp_path / "store.db"
    store = TapeStore(str(db))
    tape = _make_tape()
    run_id = store.save_tape(tape, run_id="run1")
    report = BlameReport(
        results=[
            FlipRateResult(
                step_index=0,
                flip_rate=0.9,
                ci_lo=0.7,
                ci_hi=0.98,
                flips=9,
                trials=10,
                p_value=0.01,
                q_value=0.02,
                responsible=True,
            )
        ],
        k=10,
        total_forks=10,
        ci_method=CIMethod.WILSON,
    )
    store.save_blame_report(run_id, report)
    expected = store.causal_edges_for_run(run_id)
    store.close()

    out = tmp_path / "report.html"
    result = runner.invoke(app, ["report", run_id, "--store", str(db), "-o", str(out)])
    assert result.exit_code == 0, result.output
    data = _extract_data(out.read_text())
    assert data["causal_edges"] == expected
    assert len(data["causal_edges"]) == 1


def test_cli_report_tape_path_leaves_causal_edges_empty(tmp_path):
    tape = _make_tape()
    tape_path = tmp_path / "run.tape.sqlite"
    tape.save(str(tape_path))

    out = tmp_path / "report.html"
    result = runner.invoke(app, ["report", "--tape", str(tape_path), "-o", str(out)])
    assert result.exit_code == 0, result.output
    data = _extract_data(out.read_text())
    assert data["causal_edges"] == []


def test_server_get_run_includes_causal_edges(tmp_path):
    from fastapi.testclient import TestClient

    from tracefork.blame import BlameReport, CIMethod, FlipRateResult
    from tracefork.server import app as fastapi_app
    from tracefork.server import init_store

    db = tmp_path / "store.db"
    store = TapeStore(str(db))
    tape = _make_tape()
    run_id = store.save_tape(tape, run_id="run1")
    report = BlameReport(
        results=[
            FlipRateResult(
                step_index=0,
                flip_rate=0.9,
                ci_lo=0.7,
                ci_hi=0.98,
                flips=9,
                trials=10,
                p_value=0.01,
                q_value=0.02,
                responsible=True,
            )
        ],
        k=10,
        total_forks=10,
        ci_method=CIMethod.WILSON,
    )
    store.save_blame_report(run_id, report)
    expected = store.causal_edges_for_run(run_id)
    store.close()

    init_store(str(db))
    client = TestClient(fastapi_app)
    resp = client.get(f"/api/run/{run_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["causal_edges"] == expected
    assert "branches" in body
