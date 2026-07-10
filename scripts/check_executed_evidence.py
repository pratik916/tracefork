#!/usr/bin/env python3
"""scripts/check_executed_evidence.py — the executed-evidence CI sentinel
(tracefork-bge.25).

A bare pytest exit code proves nothing about a narrowed ``-k`` selection or a
silently-skipped test: pytest exits 0 whether it ran 700 tests or 3, and a
``@pytest.mark.skip`` on a safety-critical test is invisible in the exit
code too. This script closes that gap: it parses the JUnit XML report pytest
already writes (via ``--junit-xml``) and cross-checks it against a
required-test-id manifest (``tests/required_test_ids.txt``) — "these
specific N tests, by id, definitely ran and definitely passed" replaces "the
exit code was 0".

Both ``scripts/e2e.sh`` and ``.github/workflows/ci.yml`` run pytest with
``--junit-xml=junit.xml`` and then this script, so neither surface is weaker
than the other.

Offline/$0 — pure local file parsing, no network, no subprocess spawned by
this module itself. A missing or malformed ``junit.xml``, or a required id
that is absent or present-but-skipped, is a hard failure (exit 1), never a
silent no-op pass — the same negative-control discipline the rest of
tracefork applies to replay/tape verification applies here too.

Maintenance obligation: renaming or removing a test named in the manifest
must update the manifest in the same change, or this sentinel correctly
starts failing.

Uses stdlib ``xml.etree.ElementTree`` deliberately (no new dependency, per
this bead's approach) rather than ``defusedxml``: the XML parsed here is
``junit.xml``, freshly written by pytest itself in the very same CI/e2e run
that then invokes this script — never an externally-sourced or
attacker-controlled document — so the XXE/billion-laughs trust boundary
`defusedxml` guards against does not apply.

    uv run python scripts/check_executed_evidence.py \\
        --junit-xml junit.xml --manifest tests/required_test_ids.txt
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

DEFAULT_JUNIT_XML = Path("junit.xml")
DEFAULT_MANIFEST = Path("tests/required_test_ids.txt")


class ExecutedEvidenceError(RuntimeError):
    """Raised when the manifest or JUnit report can't be read as evidence."""


@dataclass(frozen=True)
class TestOutcome:
    """The recorded outcome of one JUnit ``<testcase>`` element."""

    node_id: str
    status: str
    """One of ``"passed"``, ``"failed"``, or ``"skipped"``."""


def parse_required_manifest(path: Path) -> list[str]:
    """Read a required-test-id manifest into an ordered list of ids.

    Each non-blank, non-comment (``#``-prefixed) line is one id in
    ``classname::name`` form, matching pytest's JUnit XML ``classname`` +
    ``name`` attributes exactly. Raises `ExecutedEvidenceError` if `path`
    does not exist — a missing manifest is a hard failure, not an empty
    (vacuously-passing) requirement set.
    """
    if not path.is_file():
        raise ExecutedEvidenceError(f"required-test-id manifest not found: {path}")
    ids: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        ids.append(line)
    return ids


def parse_junit_outcomes(path: Path) -> dict[str, TestOutcome]:
    """Parse a JUnit XML report into a ``classname::name`` -> `TestOutcome` map.

    Raises `ExecutedEvidenceError` if `path` is missing or the XML fails to
    parse — a missing/malformed report is a hard failure, never a silent
    no-op pass-through.
    """
    if not path.is_file():
        raise ExecutedEvidenceError(f"junit xml report not found: {path}")
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise ExecutedEvidenceError(f"junit xml report is malformed: {path} ({exc})") from exc

    outcomes: dict[str, TestOutcome] = {}
    for testcase in root.iter("testcase"):
        classname = testcase.get("classname")
        name = testcase.get("name")
        if not classname or not name:
            continue
        node_id = f"{classname}::{name}"
        if testcase.find("skipped") is not None:
            status = "skipped"
        elif testcase.find("failure") is not None or testcase.find("error") is not None:
            status = "failed"
        else:
            status = "passed"
        outcomes[node_id] = TestOutcome(node_id=node_id, status=status)
    return outcomes


def check_executed_evidence(required_ids: list[str], outcomes: dict[str, TestOutcome]) -> list[str]:
    """Cross-check `required_ids` against `outcomes`.

    Returns one human-readable diagnostic line per required id that is
    missing from the report or present with a non-``"passed"`` status. An
    empty list means every required id genuinely ran and passed.
    """
    problems: list[str] = []
    for node_id in required_ids:
        outcome = outcomes.get(node_id)
        if outcome is None:
            problems.append(f"MISSING: required test id not found in junit report: {node_id}")
        elif outcome.status != "passed":
            problems.append(f"{outcome.status.upper()}: required test id did not pass: {node_id}")
    return problems


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code (0 or 1)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--junit-xml",
        type=Path,
        default=DEFAULT_JUNIT_XML,
        help=f"path to the JUnit XML report (default: {DEFAULT_JUNIT_XML})",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"path to the required-test-id manifest (default: {DEFAULT_MANIFEST})",
    )
    args = parser.parse_args(argv)

    try:
        required_ids = parse_required_manifest(args.manifest)
        outcomes = parse_junit_outcomes(args.junit_xml)
    except ExecutedEvidenceError as exc:
        print(f"executed-evidence sentinel: {exc}", file=sys.stderr)
        return 1

    problems = check_executed_evidence(required_ids, outcomes)
    if problems:
        print("executed-evidence sentinel: FAILED", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    print(f"executed-evidence sentinel: OK ({len(required_ids)} required test ids passed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
