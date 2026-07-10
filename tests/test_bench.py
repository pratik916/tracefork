"""`tracefork bench` / `tracefork.bench` tests — all offline, zero API keys."""

import json

from typer.testing import CliRunner

from tracefork.bench import KNOWN_LIMITATION_CASES, run_bench
from tracefork.cli import app

runner = CliRunner()


def test_run_bench_scores_eleven_cases():
    report = run_bench(k=3, m_samples=2)
    assert report.n_cases == 11
    assert len(report.cases) == 11


def test_run_bench_matches_ground_truth_except_the_documented_limitation():
    """10/11 cases resolve cleanly; the one that doesn't is the documented,
    named limitation (`gate_half_of_conjunction`) of a strictly SEQUENTIAL
    tape -- the two concurrent-tape cases (`tracefork-bge.10`) both resolve,
    never a surprise miss."""
    report = run_bench(k=3, m_samples=2)
    unresolved = [c.name for c in report.cases if not c.resolved]
    assert unresolved == ["gate_half_of_conjunction"]
    assert report.n_resolved == 10
    assert report.accuracy == 10 / 11
    assert report.unexpected_failures() == []


def test_known_limitation_case_carries_an_explanatory_note():
    report = run_bench(k=3, m_samples=2)
    limitation = next(c for c in report.cases if c.name in KNOWN_LIMITATION_CASES)
    assert limitation.resolved is False
    assert "LIMITATION" in limitation.note


def test_bench_report_ci_brackets_the_point_estimate():
    report = run_bench(k=3, m_samples=2)
    assert 0.0 <= report.ci_lo <= report.accuracy <= report.ci_hi <= 1.0


def test_bench_cites_the_who_and_when_anchor_without_claiming_it_was_run():
    report = run_bench(k=3, m_samples=2)
    assert report.who_and_when_anchor == 0.142


# ── CLI ──────────────────────────────────────────────────────────────────


def test_bench_cli_runs_offline_and_exits_zero(tmp_path):
    out = tmp_path / "bench_report.json"
    result = runner.invoke(app, ["bench", "--k", "2", "--m-samples", "1", "--output", str(out)])
    assert result.exit_code == 0, result.output
    assert "competing-fault discrimination" in result.output
    assert "LIMITATION" in result.output

    data = json.loads(out.read_text())
    assert data["n_cases"] == 11
    assert data["n_resolved"] == 10
    resolved_by_name = {c["name"]: c["resolved"] for c in data["cases"]}
    assert resolved_by_name["gate_half_of_conjunction"] is False
    assert all(v for name, v in resolved_by_name.items() if name != "gate_half_of_conjunction")
