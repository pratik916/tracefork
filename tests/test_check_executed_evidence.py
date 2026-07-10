"""Tests for scripts/check_executed_evidence.py, the executed-evidence CI
sentinel (tracefork-bge.25) — all offline, $0, pure local file parsing.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts.check_executed_evidence import (
    main,
    parse_required_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_MANIFEST = REPO_ROOT / "tests" / "required_test_ids.txt"


def _write_junit(tmp_path: Path, testcases_xml: str) -> Path:
    xml = (
        '<?xml version="1.0" encoding="utf-8"?>'
        f'<testsuites><testsuite name="pytest">{testcases_xml}</testsuite></testsuites>'
    )
    path = tmp_path / "junit.xml"
    path.write_text(xml, encoding="utf-8")
    return path


def _nodeid_from_required_id(required_id: str) -> str:
    """`tests.test_tape::test_digest_is_deterministic` -> the real pytest
    node id `tests/test_tape.py::test_digest_is_deterministic`."""
    classname, name = required_id.split("::", 1)
    return f"{classname.replace('.', '/')}.py::{name}"


def test_checker_exits_0_when_all_required_ids_present_and_passed(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("tests.test_tape::test_digest_is_deterministic\n")
    junit = _write_junit(
        tmp_path,
        '<testcase classname="tests.test_tape" name="test_digest_is_deterministic" time="0.01" />',
    )

    exit_code = main(["--junit-xml", str(junit), "--manifest", str(manifest)])

    assert exit_code == 0


def test_checker_exits_1_and_names_the_id_when_a_required_id_is_absent(
    tmp_path: Path, capsys
) -> None:
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("tests.test_tape::test_this_id_does_not_exist\n")
    junit = _write_junit(
        tmp_path,
        '<testcase classname="tests.test_tape" name="test_digest_is_deterministic" time="0.01" />',
    )

    exit_code = main(["--junit-xml", str(junit), "--manifest", str(manifest)])

    assert exit_code == 1
    err = capsys.readouterr().err
    assert "test_this_id_does_not_exist" in err
    assert "MISSING" in err


def test_checker_exits_1_and_names_the_id_when_present_but_skipped(tmp_path: Path, capsys) -> None:
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("tests.test_tape::test_digest_is_deterministic\n")
    junit = _write_junit(
        tmp_path,
        '<testcase classname="tests.test_tape" name="test_digest_is_deterministic" time="0.0">'
        '<skipped message="skipped for reasons" /></testcase>',
    )

    exit_code = main(["--junit-xml", str(junit), "--manifest", str(manifest)])

    assert exit_code == 1
    err = capsys.readouterr().err
    assert "test_digest_is_deterministic" in err
    assert "SKIPPED" in err


def test_checker_exits_1_when_junit_xml_itself_is_missing(tmp_path: Path, capsys) -> None:
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("tests.test_tape::test_digest_is_deterministic\n")
    missing_junit = tmp_path / "does_not_exist.xml"

    exit_code = main(["--junit-xml", str(missing_junit), "--manifest", str(manifest)])

    assert exit_code == 1
    err = capsys.readouterr().err
    assert "not found" in err


def test_checker_exits_1_when_junit_xml_is_malformed(tmp_path: Path, capsys) -> None:
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("tests.test_tape::test_digest_is_deterministic\n")
    junit = tmp_path / "junit.xml"
    junit.write_text("<testsuites><not-closed>", encoding="utf-8")

    exit_code = main(["--junit-xml", str(junit), "--manifest", str(manifest)])

    assert exit_code == 1
    err = capsys.readouterr().err
    assert "malformed" in err


def test_real_e2e_pytest_run_against_the_seeded_manifest_passes_today(tmp_path: Path) -> None:
    """Runs pytest for exactly the tests named in the real, committed
    manifest (not a synthetic fixture) and asserts the checker accepts the
    resulting junit.xml -- proving the seeded manifest itself is not already
    rotten (a stale/renamed required id would fail this test today)."""
    required_ids = parse_required_manifest(REQUIRED_MANIFEST)
    node_ids = [_nodeid_from_required_id(required_id) for required_id in required_ids]
    junit_path = tmp_path / "junit.xml"

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", f"--junit-xml={junit_path}", *node_ids],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    exit_code = main(["--junit-xml", str(junit_path), "--manifest", str(REQUIRED_MANIFEST)])

    assert exit_code == 0
