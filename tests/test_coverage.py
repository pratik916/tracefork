"""coverage.py — determinism-coverage report tests.

Two surfaces, both read-only/static: `tape_draw_coverage` (tallies an
already-loaded `Tape`'s draw kinds, concurrency-batch presence, and recorded
`BoundaryGuard` activity) and `scan_source_for_nondeterminism_calls` (a
best-effort `ast.parse`-only lint over agent source *text*, never imported or
executed, for call sites shaped like the direct nondeterminism/boundary-
crossing operations `BoundaryGuard` does — or, documented, does not —
intercept).
"""

from __future__ import annotations

from tracefork.coverage import (
    coverage_report,
    scan_source_for_nondeterminism_calls,
    tape_draw_coverage,
)
from tracefork.tape import Tape

# ── tape_draw_coverage ──────────────────────────────────────────────────


def test_draw_counts_tally_by_kind_only_for_kinds_present():
    tape = Tape(draws=[("clock", "a"), ("clock", "b"), ("uuid", "c")])
    draw_counts, _concurrency, _guard = tape_draw_coverage(tape)
    assert draw_counts == {"clock": 2, "uuid": 1}


def test_draw_counts_empty_for_tape_with_no_draws():
    tape = Tape()
    draw_counts, _concurrency, _guard = tape_draw_coverage(tape)
    assert draw_counts == {}


def test_concurrency_recorded_true_for_non_empty_async_batches():
    tape = Tape(async_batches=[[0, 1]])
    _draws, concurrency_recorded, _guard = tape_draw_coverage(tape)
    assert concurrency_recorded is True


def test_concurrency_recorded_false_for_empty_async_batches():
    tape = Tape(async_batches=[])
    _draws, concurrency_recorded, _guard = tape_draw_coverage(tape)
    assert concurrency_recorded is False


def test_boundary_guard_active_reads_provenance_true():
    tape = Tape(provenance={"boundary_guard": "true"})
    _draws, _concurrency, guard_active = tape_draw_coverage(tape)
    assert guard_active is True


def test_boundary_guard_active_false_when_absent_or_false():
    assert tape_draw_coverage(Tape())[2] is False
    assert tape_draw_coverage(Tape(provenance={"boundary_guard": "false"}))[2] is False


# ── scan_source_for_nondeterminism_calls ────────────────────────────────


def test_scan_finds_two_guardable_calls():
    source = (
        "import random\nimport threading\n\nrandom.random()\nthreading.Thread(target=x).start()\n"
    )
    findings = scan_source_for_nondeterminism_calls(source, boundary_guard_active=True)
    guardable = [f for f in findings if f.guardable]
    assert len(guardable) == 2
    calls = {f.call for f in guardable}
    assert calls == {"random.random", "threading.Thread.start"}
    assert all(f.caught for f in guardable)


def test_scan_finds_one_non_guardable_informational_finding():
    source = "import datetime\ndatetime.datetime.now()\n"
    findings = scan_source_for_nondeterminism_calls(source, boundary_guard_active=True)
    assert len(findings) == 1
    f = findings[0]
    assert f.guardable is False
    assert f.call == "datetime.datetime.now()"
    # Never claimed as caught, even though a guard was active on this tape --
    # this is BoundaryGuard's own documented non-coverage, not a violation.
    assert f.caught is False


def test_non_guardable_finding_stays_uncaught_regardless_of_guard_state():
    source = "import time\ntime.time()\n"
    for guard_active in (True, False):
        findings = scan_source_for_nondeterminism_calls(source, boundary_guard_active=guard_active)
        assert len(findings) == 1
        assert findings[0].guardable is False
        assert findings[0].caught is False


def test_guardable_finding_caught_flag_tracks_guard_state():
    source = "import random\nrandom.random()\n"
    caught_findings = scan_source_for_nondeterminism_calls(source, boundary_guard_active=True)
    uncaught_findings = scan_source_for_nondeterminism_calls(source, boundary_guard_active=False)
    assert caught_findings[0].caught is True
    assert uncaught_findings[0].caught is False


def test_scan_no_matching_calls_returns_empty_no_false_positives():
    source = "def f(x):\n    return x + 1\n\nf(2)\n"
    findings = scan_source_for_nondeterminism_calls(source, boundary_guard_active=True)
    assert findings == []


def test_scan_never_imports_or_executes_source():
    # A module-level side effect that would prove import/exec if it fired.
    source = "raise RuntimeError('scanned source must never execute')\n"
    findings = scan_source_for_nondeterminism_calls(source, boundary_guard_active=False)
    assert findings == []


# ── coverage_report (combined) ──────────────────────────────────────────


def test_coverage_report_without_agent_source_has_no_findings():
    tape = Tape(draws=[("clock", "a")], async_batches=[[0, 1]])
    report = coverage_report(tape)
    assert report.draw_counts == {"clock": 1}
    assert report.concurrency_recorded is True
    assert report.findings == []


def test_coverage_report_with_agent_source_scans_and_threads_guard_state():
    tape = Tape(draws=[("uuid", "x")], provenance={"boundary_guard": "true"})
    report = coverage_report(tape, agent_source="import random\nrandom.random()\n")
    assert report.boundary_guard_active is True
    assert len(report.findings) == 1
    assert report.findings[0].caught is True
