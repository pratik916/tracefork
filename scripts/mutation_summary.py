#!/usr/bin/env python3
"""scripts/mutation_summary.py — offline `mutmut junitxml` summarizer
(tracefork-bge.49).

`mutmut` (see the optional `mutation` extra + `[tool.mutmut]` in
`pyproject.toml`) mutates `transport.py`/`tape.py`/`matcher.py`/`blame.py` —
the four modules this bead's mutation-testing pass proof-bears — and reports
one `<testcase>` per mutant via `mutmut junitxml`. A killed mutant (the test
suite genuinely caught the injected bug) has no `<failure>` child; a mutant
that survived, timed out, or came back "suspicious" (an inconsistent/flaky
run) does carry one, tagged via the failure's `type` attribute. This script
turns that report into a small `MutationSummary` — killed/survived/timeout/
suspicious counts plus a `killed / total` mutation score — the same
"parse the JUnit report a prior step already wrote" pattern
`scripts/check_executed_evidence.py` (tracefork-bge.25) established.

Deliberately informational, never a merge gate: `scripts/mutation.sh` and
`.github/workflows/mutation.yml` (nightly cron + `workflow_dispatch` only,
never `push`/`pull_request`) invoke this script's CLI with no `--fail-under`,
so it always exits 0 regardless of how many mutants survived — the report is
for human review, not for blocking a PR on a coverage-detection technique
still being calibrated. `--fail-under` exists for a future, deliberate
opt-in tightening; nothing in this repo's wiring passes it today.

Offline/$0 — pure local file parsing, no network, no subprocess spawned by
this module itself. Uses stdlib `xml.etree.ElementTree` deliberately (no new
dependency), for the same trust-boundary reason `check_executed_evidence.py`
does: the XML parsed here is `mutmut-junit.xml`, freshly written by `mutmut`
itself in the very same run that then invokes this script — never an
externally-sourced or attacker-controlled document.

    uv run python scripts/mutation_summary.py --junit-xml mutmut-junit.xml
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

DEFAULT_JUNIT_XML = Path("mutmut-junit.xml")

_FAILURE_STATUSES = ("survived", "timeout", "suspicious")


class MutationSummaryError(RuntimeError):
    """Raised when the mutmut JUnit report can't be read as evidence."""


@dataclass(frozen=True)
class MutationSummary:
    """Aggregate counts parsed from a `mutmut junitxml` report.

    `score` is the classic mutation-score ratio, `killed / total` (`0.0` for
    an empty report — never a `ZeroDivisionError`): `1.0` means every mutant
    introduced into the scoped modules was caught by the test suite.
    """

    total: int
    killed: int
    survived: int
    timeout: int
    suspicious: int
    score: float


def _status_of(testcase: ET.Element) -> str:
    """Return one of `"killed"`/`"survived"`/`"timeout"`/`"suspicious"` for
    one `mutmut junitxml` `<testcase>` element.

    A mutant mutmut could not kill carries a `<failure>` child whose `type`
    attribute names which of the three it is; a killed mutant (the test
    suite genuinely caught it) has no `<failure>` child at all. An
    unrecognized or missing `type` on a present `<failure>` still counts as
    a failure — conservatively bucketed as `"survived"`, the worse-case
    default — rather than being silently dropped from the total.
    """
    failure = testcase.find("failure")
    if failure is None:
        return "killed"
    status = (failure.get("type") or "").strip().lower()
    return status if status in _FAILURE_STATUSES else "survived"


def parse_mutmut_junit(path: Path) -> list[str]:
    """Parse a `mutmut junitxml` report into a list of per-mutant statuses.

    Raises `MutationSummaryError` if `path` is missing or the XML fails to
    parse — a missing/malformed report is a hard failure here too, never a
    silent empty-summary pass-through.
    """
    if not path.is_file():
        raise MutationSummaryError(f"mutmut junit xml report not found: {path}")
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise MutationSummaryError(f"mutmut junit xml report is malformed: {path} ({exc})") from exc
    return [_status_of(testcase) for testcase in root.iter("testcase")]


def summarize_mutmut_junit(path: Path) -> MutationSummary:
    """Summarize the `mutmut junitxml` report at `path` into a `MutationSummary`.

    Raises `MutationSummaryError` under the same conditions as
    `parse_mutmut_junit`.
    """
    statuses = parse_mutmut_junit(path)
    total = len(statuses)
    killed = statuses.count("killed")
    survived = statuses.count("survived")
    timeout = statuses.count("timeout")
    suspicious = statuses.count("suspicious")
    score = killed / total if total else 0.0
    return MutationSummary(
        total=total,
        killed=killed,
        survived=survived,
        timeout=timeout,
        suspicious=suspicious,
        score=score,
    )


def format_summary(summary: MutationSummary) -> str:
    """Render a `MutationSummary` as one human-readable line."""
    return (
        f"mutation summary: {summary.killed}/{summary.total} killed "
        f"(survived={summary.survived}, timeout={summary.timeout}, "
        f"suspicious={summary.suspicious}, score={summary.score:.2%})"
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code (0 or 1).

    Informational by default: exits 0 even when mutants survived. Only
    exits 1 when `--fail-under` is explicitly passed AND the observed score
    falls short of it — nothing in this repo's nightly/local wiring passes
    `--fail-under`, so the default invocation can never fail the job.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--junit-xml",
        type=Path,
        default=DEFAULT_JUNIT_XML,
        help=f"path to the mutmut junitxml report (default: {DEFAULT_JUNIT_XML})",
    )
    parser.add_argument(
        "--fail-under",
        type=float,
        default=None,
        help="optional mutation-score floor (0.0-1.0); exit 1 if the observed score is "
        "lower. Omitted by default everywhere in this repo's CI/nightly wiring, keeping "
        "this script purely informational.",
    )
    args = parser.parse_args(argv)

    try:
        summary = summarize_mutmut_junit(args.junit_xml)
    except MutationSummaryError as exc:
        print(f"mutation summary: {exc}", file=sys.stderr)
        return 1

    print(format_summary(summary))

    if args.fail_under is not None and summary.score < args.fail_under:
        print(
            f"mutation summary: FAILED (score {summary.score:.2%} < "
            f"--fail-under {args.fail_under:.2%})",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
