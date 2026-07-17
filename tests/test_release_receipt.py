"""Tests for `tracefork.release_receipt`: the per-release trust-receipt
composer/signer, its junit/coverage parsers, and the `tracefork
release-receipt` CLI command.

All offline, $0 — the replay-corpus check replays the real committed
`experiments/replay_fixtures` corpus ($0), the calibration sweep uses a
small, fast `n_repeats`/grid (pure math over `random.Random(seed)`, never a
network call), and junit/coverage inputs are hand-written fixture files.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from typer.testing import CliRunner

from tracefork.blame import CIMethod
from tracefork.ci_calibration import run_calibration
from tracefork.release_receipt import (
    build_release_receipt,
    calibration_report_to_dict,
    corpus_check_result_to_dict,
    parse_coverage_summary,
    parse_junit_test_summary,
    sign_release_receipt,
    verify_release_receipt_signature,
)
from tracefork.replay import run_fixture_corpus_check

runner = CliRunner()

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "experiments" / "replay_fixtures"

_TEST_SUMMARY = {"tests": 890, "failures": 0, "errors": 0, "skipped": 0, "time": 12.34}
_COVERAGE_SUMMARY = {
    "percent_covered": 91.5,
    "num_statements": 4000,
    "covered_lines": 3660,
    "missing_lines": 340,
}
_VALIDATE_REPORT = {"overall_top1_precision": 1.0}
_BENCH_REPORT = {"accuracy": 0.91}


def _small_calibration_report():
    return run_calibration(
        true_ps=(0.5,),
        n_trials_grid=(10,),
        methods=(CIMethod.WILSON,),
        n_repeats=50,
        seed=0,
    )


def _fresh_replay_corpus_result():
    return run_fixture_corpus_check(FIXTURES_DIR)


# ── build_release_receipt ────────────────────────────────────────────────


def test_build_release_receipt_full_payload_has_all_expected_keys_and_stable_digest():
    replay_corpus = _fresh_replay_corpus_result()
    calibration = _small_calibration_report()

    kwargs = {
        "version": "v1.2.3",
        "test_summary": _TEST_SUMMARY,
        "coverage_summary": _COVERAGE_SUMMARY,
        "validate_report": _VALIDATE_REPORT,
        "bench_report": _BENCH_REPORT,
        "replay_corpus": replay_corpus,
        "calibration": calibration,
        "generated_at": "2026-07-17T00:00:00+00:00",
    }

    receipt_a = build_release_receipt(**kwargs)
    receipt_b = build_release_receipt(**kwargs)

    assert receipt_a["schemaVersion"]
    assert receipt_a["version"] == "v1.2.3"
    assert receipt_a["test"]["available"] is True
    assert receipt_a["test"]["tests"] == 890
    assert receipt_a["coverage"]["available"] is True
    assert receipt_a["coverage"]["percent_covered"] == 91.5
    assert receipt_a["validate"]["available"] is True
    assert receipt_a["bench"]["available"] is True
    assert receipt_a["replay_corpus"]["available"] is True
    assert receipt_a["calibration"]["available"] is True
    assert receipt_a["generated_at"] == "2026-07-17T00:00:00+00:00"

    digest = receipt_a["receipt_digest"]
    assert isinstance(digest, str)
    assert len(digest) == 64
    int(digest, 16)  # valid hex

    # Byte-stable across two builds with identical inputs.
    assert receipt_b["receipt_digest"] == digest

    # Changes when any one input field changes.
    changed = build_release_receipt(**{**kwargs, "version": "v1.2.4"})
    assert changed["receipt_digest"] != digest


def test_build_release_receipt_no_evidence_has_explicit_absent_markers():
    receipt = build_release_receipt(version="v1.0.0")

    assert receipt["test"] == {"available": False}
    assert receipt["coverage"] == {"available": False}
    assert receipt["validate"] == {"available": False}
    assert receipt["bench"] == {"available": False}
    assert receipt["replay_corpus"] == {"available": False}
    assert receipt["calibration"] == {"available": False}
    # Never omitted keys.
    for key in ("test", "coverage", "validate", "bench", "replay_corpus", "calibration"):
        assert key in receipt


def test_build_release_receipt_embeds_replay_corpus_and_calibration_dataclasses():
    replay_corpus = _fresh_replay_corpus_result()
    calibration = _small_calibration_report()

    receipt = build_release_receipt(
        version="v1.0.0",
        replay_corpus=replay_corpus,
        calibration=calibration,
    )

    assert receipt["replay_corpus"]["all_passed"] == replay_corpus.all_passed
    assert receipt["calibration"]["all_within_tolerance"] == calibration.all_within_tolerance()
    assert len(receipt["replay_corpus"]["fixtures"]) == len(replay_corpus.fixtures)
    assert len(receipt["calibration"]["results"]) == len(calibration.results)


# ── shaping helpers ───────────────────────────────────────────────────────


def test_corpus_check_result_to_dict_mirrors_dataclass():
    result = _fresh_replay_corpus_result()
    data = corpus_check_result_to_dict(result)
    assert data["all_passed"] == result.all_passed
    assert [f["name"] for f in data["fixtures"]] == [f.name for f in result.fixtures]


def test_calibration_report_to_dict_mirrors_dataclass():
    report = _small_calibration_report()
    data = calibration_report_to_dict(report)
    assert data["seed"] == report.seed
    assert data["all_within_tolerance"] == report.all_within_tolerance()
    assert data["results"][0]["method"] == report.results[0].method.value


# ── junit / coverage parsers ─────────────────────────────────────────────


def test_parse_junit_test_summary_reads_testsuite_counts(tmp_path):
    bare = tmp_path / "bare.xml"
    bare.write_text(
        '<testsuite tests="5" failures="1" errors="0" skipped="2" time="1.25"></testsuite>'
    )

    wrapped = tmp_path / "wrapped.xml"
    wrapped.write_text(
        "<testsuites>"
        '<testsuite tests="5" failures="1" errors="0" skipped="2" time="1.25">'
        "</testsuite>"
        "</testsuites>"
    )

    expected = {"tests": 5, "failures": 1, "errors": 0, "skipped": 2, "time": 1.25}

    bare_summary = parse_junit_test_summary(bare)
    wrapped_summary = parse_junit_test_summary(wrapped)

    assert bare_summary == expected
    assert wrapped_summary == expected
    assert isinstance(bare_summary["tests"], int)
    assert isinstance(bare_summary["time"], float)


def test_parse_coverage_summary_reads_totals(tmp_path):
    coverage_json = tmp_path / "coverage.json"
    totals = {
        "covered_lines": 3660,
        "num_statements": 4000,
        "percent_covered": 91.5,
        "percent_covered_display": "92",
        "missing_lines": 340,
        "excluded_lines": 0,
    }
    coverage_json.write_text(json.dumps({"meta": {}, "files": {}, "totals": totals}))

    summary = parse_coverage_summary(coverage_json)

    assert summary["percent_covered"] == totals["percent_covered"]
    assert summary["num_statements"] == totals["num_statements"]
    assert summary["covered_lines"] == totals["covered_lines"]
    assert summary["missing_lines"] == totals["missing_lines"]


# ── signing ───────────────────────────────────────────────────────────────


def test_sign_release_receipt_no_key_is_explicit_absent_marker():
    receipt = build_release_receipt(version="v1.0.0")
    signed = sign_release_receipt(receipt, signing_key=None)
    assert signed["signature"] == {"available": False}


def test_sign_release_receipt_verify_roundtrip_and_rejects_tamper_or_wrong_key():
    receipt = build_release_receipt(version="v1.0.0")
    signed = sign_release_receipt(receipt, signing_key=b"top-secret-key")

    assert signed["signature"]["available"] is True
    assert signed["signature"]["algorithm"] == "HMAC-SHA256"
    assert isinstance(signed["signature"]["value"], str)
    int(signed["signature"]["value"], 16)  # valid hex

    assert verify_release_receipt_signature(signed, signing_key=b"top-secret-key") is True

    tampered = copy.deepcopy(signed)
    tampered["receipt_digest"] = "0" * 64
    assert verify_release_receipt_signature(tampered, signing_key=b"top-secret-key") is False

    assert verify_release_receipt_signature(signed, signing_key=b"wrong-key") is False


def test_verify_release_receipt_signature_false_when_unsigned():
    receipt = build_release_receipt(version="v1.0.0")
    unsigned = sign_release_receipt(receipt, signing_key=None)
    assert verify_release_receipt_signature(unsigned, signing_key=b"any-key") is False


# ── CLI: `tracefork release-receipt` ─────────────────────────────────────
#
# NOTE: as of this bead, `release-receipt` is not yet registered on
# `tracefork.cli.app` (cli.py is an orchestrator-owned file this bead does
# not edit — see the bead's `cli_command` deliverable). These two tests
# exercise the command end-to-end exactly as the CLI is specified to behave
# once wired, and are expected to start passing unchanged the moment that
# wiring lands.
#
# The exit code is asserted as a function of the receipt's OWN recorded
# `replay_corpus["all_passed"]` / `calibration["all_within_tolerance"]`
# fields rather than hardcoded to 0: `ci_calibration.run_calibration()`'s
# full default grid has a handful of small-`n_trials`, near-boundary-`true_p`
# cells that fall outside tolerance (a real, pre-existing, documented
# property of the unmodified Wilson/Jeffreys/Agresti-Coull backends at very
# small n — see `ci_calibration.py`'s module docstring), so whether the
# happy path here is a 0 or a 1 depends on that harness's current numbers,
# not on anything this test should hardcode.


def test_cli_release_receipt_writes_json_with_absent_markers_when_no_disk_reports(
    tmp_path,
):
    from tracefork.cli import app

    output_dir = tmp_path / "release_receipts"

    result = runner.invoke(
        app,
        [
            "release-receipt",
            "v9.9.9",
            "--junit-xml",
            str(tmp_path / "junit.xml"),
            "--coverage-json",
            str(tmp_path / "coverage.json"),
            "--validation-report",
            str(tmp_path / "validation_report.json"),
            "--bench-report",
            str(tmp_path / "bench_report.json"),
            "--replay-fixtures",
            str(FIXTURES_DIR),
            "--output-dir",
            str(output_dir),
        ],
    )

    data = json.loads((output_dir / "v9.9.9.json").read_text())
    assert data["test"] == {"available": False}
    assert data["coverage"] == {"available": False}
    assert data["validate"] == {"available": False}
    assert data["bench"] == {"available": False}
    assert data["replay_corpus"]["available"] is True
    assert data["calibration"]["available"] is True
    assert len(data["receipt_digest"]) == 64

    expected_ok = (
        data["replay_corpus"]["all_passed"] and data["calibration"]["all_within_tolerance"]
    )
    assert result.exit_code == (0 if expected_ok else 1), result.output


def test_cli_release_receipt_signs_when_env_var_set(tmp_path, monkeypatch):
    from tracefork.cli import app

    monkeypatch.setenv("TRACEFORK_RELEASE_SIGNING_KEY", "test-signing-key")
    output_dir = tmp_path / "release_receipts"

    result = runner.invoke(
        app,
        [
            "release-receipt",
            "v9.9.9",
            "--junit-xml",
            str(tmp_path / "junit.xml"),
            "--coverage-json",
            str(tmp_path / "coverage.json"),
            "--validation-report",
            str(tmp_path / "validation_report.json"),
            "--bench-report",
            str(tmp_path / "bench_report.json"),
            "--replay-fixtures",
            str(FIXTURES_DIR),
            "--output-dir",
            str(output_dir),
        ],
    )

    data = json.loads((output_dir / "v9.9.9.json").read_text())
    assert data["signature"]["available"] is True
    assert data["signature"]["algorithm"] == "HMAC-SHA256"
    int(data["signature"]["value"], 16)  # well-formed hex

    expected_ok = (
        data["replay_corpus"]["all_passed"] and data["calibration"]["all_within_tolerance"]
    )
    assert result.exit_code == (0 if expected_ok else 1), result.output
