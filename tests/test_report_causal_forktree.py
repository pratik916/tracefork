"""Offline tests for tracefork-bge.37: the default `report <run_id>` path
bakes in persisted causal edges (`tracefork blame`'s saved blame/Shapley
results) and every branch's full delta-tape detail, so the fork-tree panel's
causal highlighting and click-to-inspect work with zero live server."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import anthropic
import httpx
from typer.testing import CliRunner

from tests.fakes import ScriptedFakeLLM, make_text_response
from tracefork.blame import BlameReport, CIMethod, FlipRateResult
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


_SAMPLE_EDGE = {
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
}


def test_tape_to_data_defaults_causal_edges_and_branch_details_to_empty():
    tape = _make_tape()
    data = _tape_to_data(tape)
    assert data["causal_edges"] == []
    assert data["branch_details"] == {}


def test_tape_to_data_includes_causal_edges_and_branch_details():
    tape = _make_tape()
    causal_edges = [_SAMPLE_EDGE]
    branch_details = {
        "b1": {
            **_tape_to_data(tape),
            "divergence_step": 0,
            "mutation_desc": "swapped response",
            "branch_digest": "d1",
            "parent_run_id": "run1",
        }
    }
    data = _tape_to_data(tape, causal_edges=causal_edges, branch_details=branch_details)
    assert data["causal_edges"] == causal_edges
    assert data["branch_details"] == branch_details


def test_generate_report_embeds_causal_edges_and_branch_details():
    tape = _make_tape()
    causal_edges = [_SAMPLE_EDGE]
    branch_details = {
        "b1": {
            **_tape_to_data(tape),
            "divergence_step": 0,
            "mutation_desc": "swapped response",
            "branch_digest": "d1",
            "parent_run_id": "run1",
        }
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "report.html"
        generate_report(tape, out, causal_edges=causal_edges, branch_details=branch_details)
        content = out.read_text()
        data = _extract_data(content)
        assert data["causal_edges"] == causal_edges
        assert data["branch_details"]["b1"]["mutation_desc"] == "swapped response"
        # Static-render wiring shipped in the template (mirrors the existing
        # "renderForkTree in content" proxy-assertion style).
        assert "branch_details" in content
        assert "forktree-node-responsible" in content


def test_cli_report_with_run_id_auto_embeds_persisted_causal_edges_and_branch_details(tmp_path):
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
    store.save_branch(
        parent_run_id=run_id,
        divergence_step=0,
        delta_tape=tape,
        mutation_desc="swap",
        branch_digest="digest1",
    )
    store.close()

    out = tmp_path / "report.html"
    result = runner.invoke(app, ["report", run_id, "--store", str(db), "-o", str(out)])
    assert result.exit_code == 0, result.output
    data = _extract_data(out.read_text())
    assert len(data["causal_edges"]) == 1
    assert data["causal_edges"][0]["responsible"] is True
    assert len(data["branch_details"]) == 1
    (branch_detail,) = data["branch_details"].values()
    assert len(branch_detail["exchanges"]) == 1
    assert branch_detail["mutation_desc"] == "swap"


def test_cli_report_branch_details_survives_fork_point_drift(tmp_path):
    db = tmp_path / "store.db"
    store = TapeStore(str(db))
    tape = _make_tape()
    run_id = store.save_tape(tape, run_id="run1")
    branch_id = store.save_branch(
        parent_run_id=run_id,
        divergence_step=0,
        delta_tape=tape,
        mutation_desc="swap",
        parent_tape_digest="not-the-real-digest",
    )
    store.close()

    out = tmp_path / "report.html"
    result = runner.invoke(app, ["report", run_id, "--store", str(db), "-o", str(out)])
    assert result.exit_code == 0, result.output
    data = _extract_data(out.read_text())
    assert data["branch_details"][branch_id]["error"] == "fork_point_drift"


def test_cli_report_tape_path_leaves_causal_edges_and_branch_details_empty(tmp_path):
    tape = _make_tape()
    tape_path = tmp_path / "run.tape.sqlite"
    tape.save(str(tape_path))

    out = tmp_path / "report.html"
    result = runner.invoke(app, ["report", "--tape", str(tape_path), "-o", str(out)])
    assert result.exit_code == 0, result.output
    data = _extract_data(out.read_text())
    assert data["causal_edges"] == []
    assert data["branch_details"] == {}
