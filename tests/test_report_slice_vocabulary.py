"""Offline tests for tracefork-bge.70: Shepherd's Slice vocabulary
(selected / external-anchor / support-evidence) threaded onto data that
already exists end-to-end -- `blame.py`'s `responsible` flag (selected vs.
support evidence) and `store.causal_closure` (external anchors: responsible
blame edges reachable via fork-promotion lineage, possibly from other
run_ids)."""

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


def _responsible_report() -> BlameReport:
    return BlameReport(
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


def test_tape_to_data_defaults_causal_closure_to_empty_and_run_id_to_none():
    tape = _make_tape()
    data = _tape_to_data(tape)
    assert data["causal_closure"] == []
    assert data["run_id"] is None


def test_tape_to_data_includes_causal_closure_and_run_id_verbatim():
    tape = _make_tape()
    closure = [
        {
            "edge_id": "other-run:2:blame",
            "run_id": "other-run",
            "step_index": 2,
            "method": "blame",
            "flip_rate": 0.95,
            "ci_lo": 0.8,
            "ci_hi": 0.99,
            "ci_method": "wilson",
            "p_value": 0.001,
            "q_value": 0.01,
            "responsible": True,
            "necessity": None,
            "sufficiency": None,
            "shapley_value": None,
            "created_at": "",
        }
    ]
    data = _tape_to_data(tape, causal_closure=closure, run_id="run-a")
    assert data["causal_closure"] == closure
    assert data["run_id"] == "run-a"


def test_generate_report_embeds_causal_closure_and_run_id():
    tape = _make_tape()
    closure = [
        {
            "edge_id": "other-run:2:blame",
            "run_id": "other-run",
            "step_index": 2,
            "method": "blame",
            "responsible": True,
        }
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "report.html"
        generate_report(tape, out, causal_closure=closure, run_id="run-a")
        content = out.read_text()
        data = _extract_data(content)
        assert data["causal_closure"] == closure
        assert data["run_id"] == "run-a"
        assert "renderExternalAnchors" in content
        assert "evidence-badge" in content


def test_cli_report_with_run_id_embeds_causal_closure_from_fork_promoted_branch(tmp_path):
    """A branch promoted to its own tape (`save_tape(delta_tape,
    run_id=branch_id)`) with a saved responsible blame edge is an external
    anchor of the root run: `causal_closure(root)` returns it with
    `run_id == branch_id != root`."""
    db = tmp_path / "store.db"
    store = TapeStore(str(db))
    root_tape = _make_tape()
    root_id = store.save_tape(root_tape, run_id="root-run")
    branch_tape = _make_tape()
    branch_id = store.save_branch(
        parent_run_id=root_id, divergence_step=0, delta_tape=branch_tape, mutation_desc="m"
    )
    # Promote the branch to its own re-blamable tape (the same promotion
    # convention causal_closure/branches_forked_from document).
    store.save_tape(branch_tape, run_id=branch_id)
    store.save_blame_report(branch_id, _responsible_report())
    expected = store.causal_closure(root_id)
    store.close()

    assert len(expected) == 1
    assert expected[0]["run_id"] == branch_id

    out = tmp_path / "report.html"
    result = runner.invoke(app, ["report", root_id, "--store", str(db), "-o", str(out)])
    assert result.exit_code == 0, result.output
    data = _extract_data(out.read_text())
    assert data["causal_closure"] == expected
    assert data["run_id"] == root_id


def test_cli_report_tape_path_leaves_causal_closure_empty_and_run_id_none(tmp_path):
    tape = _make_tape()
    tape_path = tmp_path / "run.tape.sqlite"
    tape.save(str(tape_path))

    out = tmp_path / "report.html"
    result = runner.invoke(app, ["report", "--tape", str(tape_path), "-o", str(out)])
    assert result.exit_code == 0, result.output
    data = _extract_data(out.read_text())
    assert data["causal_closure"] == []
    assert data["run_id"] is None


def test_server_get_run_includes_causal_closure_key(tmp_path):
    from fastapi.testclient import TestClient

    from tracefork.server import app as fastapi_app
    from tracefork.server import init_store

    db = tmp_path / "store.db"
    store = TapeStore(str(db))
    tape = _make_tape()
    run_id = store.save_tape(tape, run_id="run1")
    store.close()

    init_store(str(db))
    client = TestClient(fastapi_app)
    resp = client.get(f"/api/run/{run_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert "causal_closure" in body
    assert body["causal_closure"] == []
