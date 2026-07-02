"""CLI tests — exercise the Typer surface offline, especially the budget money-guard
(`blame`'s pre-flight cost gate is the only thing standing between a typo and a real bill)
and the now-enforced negative control in `validate`."""

import json
import shutil
from pathlib import Path

from typer.testing import CliRunner

from tracefork.cli import app
from tracefork.store import TapeStore
from tracefork.tape import Tape
from tracefork.validate import _record_clean_tape

runner = CliRunner()

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "experiments" / "replay_fixtures"


def test_blame_budget_gate_blocks_overspend(tmp_path):
    """`blame` must refuse to spend when the estimate exceeds --budget, and it must do
    so *before* any network call — the gate fires on the pre-flight estimate."""
    db = tmp_path / "store.db"
    store = TapeStore(str(db))
    run_id = store.save_tape(_record_clean_tape(), run_id="testrun")
    store.close()

    result = runner.invoke(
        app,
        [
            "blame",
            run_id,
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


def test_blame_rejects_unsafe_run_id(tmp_path):
    """run_id is validated up front — this is what keeps the `blame_<run_id>.json`
    output path from being a traversal sink."""
    db = tmp_path / "store.db"
    TapeStore(str(db)).close()

    result = runner.invoke(
        app,
        [
            "blame",
            "../etc/passwd",
            "--agent",
            "tracefork.validate:synthetic_agent",
            "--store",
            str(db),
        ],
    )
    assert result.exit_code != 0


def test_validate_runs_and_enforces_control(tmp_path):
    """`validate` runs fully offline; the negative control is enforced, not cosmetic."""
    out = tmp_path / "vr.json"
    result = runner.invoke(
        app,
        [
            "validate",
            "--k",
            "1",
            "--n-runs",
            "1",
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "negative control" in result.output

    data = json.loads(out.read_text())
    assert data["negative_control_max_flip"] < 0.30
    assert data["overall_top1_precision"] >= 0.7


def test_replay_check_passes_on_committed_fixture_corpus():
    result = runner.invoke(app, ["replay", "--check", str(FIXTURES_DIR)])
    assert result.exit_code == 0, result.output
    assert "fixtures passed" in result.output


def test_replay_check_fails_on_tampered_corpus(tmp_path):
    tamper_dir = tmp_path / "fixtures"
    shutil.copytree(FIXTURES_DIR, tamper_dir)

    manifest = json.loads((tamper_dir / "manifest.json").read_text())
    entry = manifest[0]
    tape_path = tamper_dir / entry["tape"]
    tape = Tape.load(str(tape_path))
    req, resp = tape.exchanges[0]
    tape.exchanges[0] = (req, resp + b" ")
    tape.save(str(tape_path))

    result = runner.invoke(app, ["replay", "--check", str(tamper_dir)])
    assert result.exit_code == 1, result.output
    assert "FAIL" in result.output


def test_replay_check_missing_manifest_exits_1(tmp_path):
    empty_dir = tmp_path / "empty_fixtures"
    empty_dir.mkdir()
    result = runner.invoke(app, ["replay", "--check", str(empty_dir)])
    assert result.exit_code == 1, result.output


def test_replay_without_check_still_requires_agent_and_tape():
    result = runner.invoke(app, ["replay"])
    assert result.exit_code == 1
    assert "Provide a tape path and --agent" in result.output
