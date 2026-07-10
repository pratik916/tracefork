"""Tests for `tracefork.receipt`: the shareable trust-receipt / badge
builders, and the `tracefork receipt` CLI command.

All offline, $0 — the fixture tape is recorded against a scripted fake
transport (never a real API), replay re-runs the same synthetic agent
against the recorded tape ($0), and validate/bench evidence is passed in as
plain dicts shaped exactly like `validation_report.json`/`bench_report.json`.
"""

from __future__ import annotations

import json

from typer.testing import CliRunner

from tracefork.cli import app
from tracefork.receipt import build_shield_json, build_trust_receipt
from tracefork.replay import ReplayVerifier
from tracefork.validate import _record_clean_tape, synthetic_agent

runner = CliRunner()

_VALIDATE_REPORT = {
    "top1_precision_by_class": {"wrong_tool_args": 1.0},
    "overall_top1_precision": 1.0,
    "negative_control_max_flip": 0.0,
    "n_runs_per_class": 5,
    "k": 3,
    "reproduce_cmd": "tracefork validate --k 3 --n-runs 5",
}

_LOW_PRECISION_VALIDATE_REPORT = {
    **_VALIDATE_REPORT,
    "overall_top1_precision": 0.2,
}

_BENCH_REPORT = {
    "k": 3,
    "m_samples": 2,
    "accuracy": 0.91,
    "n_resolved": 10,
    "n_cases": 11,
    "ci_lo": 0.6,
    "ci_hi": 0.98,
    "who_and_when_anchor": 0.142,
    "cases": [],
}


def _clean_tape():
    return _record_clean_tape()


def _bit_exact_replay_result():
    tape = _clean_tape()
    return ReplayVerifier(tape, synthetic_agent).verify(), tape


# ── build_trust_receipt ────────────────────────────────────────────────


def test_build_trust_receipt_full_payload_has_all_expected_keys():
    result, tape = _bit_exact_replay_result()

    receipt = build_trust_receipt(
        tape,
        replay=result,
        validate_report=_VALIDATE_REPORT,
        bench_report=_BENCH_REPORT,
        generated_at="2026-07-10T00:00:00+00:00",
    )

    assert receipt["schemaVersion"]
    assert receipt["tape_fingerprint"] == tape.digest()[:16]
    assert receipt["boundary"] == tape.boundary
    assert receipt["content_redacted"] == tape.content_redacted
    assert receipt["generated_at"] == "2026-07-10T00:00:00+00:00"

    assert receipt["replay"]["available"] is True
    assert receipt["replay"]["bit_exact"] is True
    assert receipt["replay"]["matched"] == result.matched
    assert receipt["replay"]["total"] == result.total

    assert receipt["validate"]["available"] is True
    assert receipt["validate"]["overall_top1_precision"] == 1.0

    assert receipt["bench"]["available"] is True
    assert receipt["bench"]["accuracy"] == 0.91


def test_build_trust_receipt_no_evidence_has_explicit_absent_markers():
    tape = _clean_tape()

    receipt = build_trust_receipt(tape)

    # Explicit absent markers, never omitted keys.
    assert receipt["replay"] == {"available": False}
    assert receipt["validate"] == {"available": False}
    assert receipt["bench"] == {"available": False}
    assert "replay" in receipt
    assert "validate" in receipt
    assert "bench" in receipt
    assert receipt["tape_fingerprint"] == tape.digest()[:16]


def test_build_trust_receipt_defaults_generated_at_to_something_nonempty():
    tape = _clean_tape()
    receipt = build_trust_receipt(tape)
    assert receipt["generated_at"]


# ── build_shield_json ───────────────────────────────────────────────────


def test_build_shield_json_green_only_when_bit_exact_and_high_precision():
    result, tape = _bit_exact_replay_result()
    receipt = build_trust_receipt(
        tape, replay=result, validate_report=_VALIDATE_REPORT, generated_at="t"
    )

    shield = build_shield_json(receipt)

    assert shield["schemaVersion"] == 1
    assert shield["color"] == "brightgreen"
    assert receipt["tape_fingerprint"][:8] in shield["message"]


def test_build_shield_json_yellow_when_precision_low():
    result, tape = _bit_exact_replay_result()
    receipt = build_trust_receipt(
        tape,
        replay=result,
        validate_report=_LOW_PRECISION_VALIDATE_REPORT,
        generated_at="t",
    )

    shield = build_shield_json(receipt)

    assert shield["color"] == "yellow"


def test_build_shield_json_yellow_when_no_evidence_at_all():
    tape = _clean_tape()
    receipt = build_trust_receipt(tape, generated_at="t")

    shield = build_shield_json(receipt)

    assert shield["color"] == "yellow"


def test_build_shield_json_red_when_replay_diverged():
    tape = _clean_tape()

    def _drifting_agent(client):
        # Diverges immediately: a request the tape never recorded.
        client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=10,
            messages=[{"role": "user", "content": "not what was recorded"}],
        )

    result = ReplayVerifier(tape, _drifting_agent).verify()
    receipt = build_trust_receipt(
        tape, replay=result, validate_report=_VALIDATE_REPORT, generated_at="t"
    )

    assert receipt["replay"]["bit_exact"] is False

    shield = build_shield_json(receipt)

    assert shield["color"] == "red"


def test_build_shield_json_never_green_for_content_redacted_tape():
    result, tape = _bit_exact_replay_result()
    tape.content_redacted = True
    receipt = build_trust_receipt(
        tape, replay=result, validate_report=_VALIDATE_REPORT, generated_at="t"
    )

    shield = build_shield_json(receipt)

    assert shield["color"] != "brightgreen"


# ── CLI ───────────────────────────────────────────────────────────────


def test_cli_receipt_writes_receipt_json_with_correct_digest_prefix(tmp_path):
    tape_path = tmp_path / "run.tape.sqlite"
    tape = _clean_tape()
    tape.save(str(tape_path))
    out = tmp_path / "receipt.json"

    result = runner.invoke(
        app,
        ["receipt", "--tape", str(tape_path), "-o", str(out)],
    )

    assert result.exit_code == 0, result.output
    data = json.loads(out.read_text())
    assert data["tape_fingerprint"] == tape.digest()[:16]
    assert data["replay"] == {"available": False}


def test_cli_receipt_with_agent_and_disk_reports_embeds_full_evidence(tmp_path):
    tape_path = tmp_path / "run.tape.sqlite"
    tape = _clean_tape()
    tape.save(str(tape_path))

    (tmp_path / "validation_report.json").write_text(json.dumps(_VALIDATE_REPORT))
    (tmp_path / "bench_report.json").write_text(json.dumps(_BENCH_REPORT))

    out = tmp_path / "receipt.json"
    shield_out = tmp_path / "shield.json"

    result = runner.invoke(
        app,
        [
            "receipt",
            "--tape",
            str(tape_path),
            "--agent",
            "tracefork.validate:synthetic_agent",
            "--validation-report",
            str(tmp_path / "validation_report.json"),
            "--bench-report",
            str(tmp_path / "bench_report.json"),
            "-o",
            str(out),
            "--shield-output",
            str(shield_out),
        ],
    )

    assert result.exit_code == 0, result.output
    data = json.loads(out.read_text())
    assert data["replay"]["bit_exact"] is True
    assert data["validate"]["available"] is True
    assert data["bench"]["available"] is True

    shield = json.loads(shield_out.read_text())
    assert shield["color"] == "brightgreen"


def test_cli_receipt_missing_tape_and_run_id_is_nonzero_exit(tmp_path):
    result = runner.invoke(app, ["receipt", "--store", str(tmp_path / "store.db")])
    assert result.exit_code == 1


def test_cli_receipt_missing_disk_reports_are_explicit_absent(tmp_path):
    tape_path = tmp_path / "run.tape.sqlite"
    _clean_tape().save(str(tape_path))
    out = tmp_path / "receipt.json"

    result = runner.invoke(
        app,
        [
            "receipt",
            "--tape",
            str(tape_path),
            "--validation-report",
            str(tmp_path / "does_not_exist.json"),
            "-o",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.output
    data = json.loads(out.read_text())
    assert data["validate"] == {"available": False}
