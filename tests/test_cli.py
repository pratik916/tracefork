"""CLI tests — exercise the Typer surface offline, especially the budget money-guard
(`blame`'s pre-flight cost gate is the only thing standing between a typo and a real bill)
and the now-enforced negative control in `validate`."""
import json

from typer.testing import CliRunner

from tracefork.cli import app
from tracefork.store import TapeStore
from tracefork.validate import _record_clean_tape

runner = CliRunner()


def test_blame_budget_gate_blocks_overspend(tmp_path):
    """`blame` must refuse to spend when the estimate exceeds --budget, and it must do
    so *before* any network call — the gate fires on the pre-flight estimate."""
    db = tmp_path / "store.db"
    store = TapeStore(str(db))
    run_id = store.save_tape(_record_clean_tape(), run_id="testrun")
    store.close()

    result = runner.invoke(app, [
        "blame", run_id,
        "--agent", "tracefork.validate:synthetic_agent",
        "--store", str(db),
        "--budget", "0",
    ])
    assert result.exit_code == 1, result.output
    assert "exceeds budget" in result.output


def test_blame_rejects_unsafe_run_id(tmp_path):
    """run_id is validated up front — this is what keeps the `blame_<run_id>.json`
    output path from being a traversal sink."""
    db = tmp_path / "store.db"
    TapeStore(str(db)).close()

    result = runner.invoke(app, [
        "blame", "../etc/passwd",
        "--agent", "tracefork.validate:synthetic_agent",
        "--store", str(db),
    ])
    assert result.exit_code != 0


def test_validate_runs_and_enforces_control(tmp_path):
    """`validate` runs fully offline; the negative control is enforced, not cosmetic."""
    out = tmp_path / "vr.json"
    result = runner.invoke(app, [
        "validate", "--k", "1", "--n-runs", "1", "--output", str(out),
    ])
    assert result.exit_code == 0, result.output
    assert "negative control" in result.output

    data = json.loads(out.read_text())
    assert data["negative_control_max_flip"] < 0.30
    assert data["overall_top1_precision"] >= 0.7
