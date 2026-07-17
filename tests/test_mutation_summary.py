"""Tests for scripts/mutation_summary.py, the offline `mutmut junitxml`
summarizer (tracefork-bge.49) — all offline, $0, pure local file parsing.
Never invokes real mutmut.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from scripts.mutation_summary import (
    MutationSummaryError,
    main,
    summarize_mutmut_junit,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_junit(tmp_path: Path, testcases_xml: str, name: str = "mutmut-junit.xml") -> Path:
    xml = (
        '<?xml version="1.0" encoding="utf-8"?>'
        f'<testsuites><testsuite name="mutmut">{testcases_xml}</testsuite></testsuites>'
    )
    path = tmp_path / name
    path.write_text(xml, encoding="utf-8")
    return path


def _testcase(name: str, failure_type: str | None = None) -> str:
    if failure_type is None:
        return f'<testcase classname="mutmut" name="{name}" time="0.0" />'
    return (
        f'<testcase classname="mutmut" name="{name}" time="0.0">'
        f'<failure type="{failure_type}" message="{failure_type}"></failure>'
        "</testcase>"
    )


def test_summarize_counts_killed_and_survived_and_computes_score(tmp_path: Path) -> None:
    # N=5 testcases, M=2 carrying a <failure> child (both tagged "survived"),
    # so killed = N - M = 3 and score = killed / N = 0.6.
    junit = _write_junit(
        tmp_path,
        _testcase("mutant_1")
        + _testcase("mutant_2", "survived")
        + _testcase("mutant_3")
        + _testcase("mutant_4", "survived")
        + _testcase("mutant_5"),
    )

    summary = summarize_mutmut_junit(junit)

    assert summary.total == 5
    assert summary.survived == 2
    assert summary.killed == 3
    assert summary.score == pytest.approx(0.6)


def test_summarize_buckets_timeout_and_suspicious_separately(tmp_path: Path) -> None:
    junit = _write_junit(
        tmp_path,
        _testcase("mutant_1")
        + _testcase("mutant_2", "timeout")
        + _testcase("mutant_3", "suspicious")
        + _testcase("mutant_4", "survived"),
    )

    summary = summarize_mutmut_junit(junit)

    assert summary.total == 4
    assert summary.killed == 1
    assert summary.timeout == 1
    assert summary.suspicious == 1
    assert summary.survived == 1
    assert summary.score == pytest.approx(0.25)


def test_summarize_defaults_an_unrecognized_failure_type_to_survived(tmp_path: Path) -> None:
    junit = _write_junit(tmp_path, _testcase("mutant_1", "totally_unknown_status"))

    summary = summarize_mutmut_junit(junit)

    assert summary.total == 1
    assert summary.killed == 0
    assert summary.survived == 1
    assert summary.timeout == 0
    assert summary.suspicious == 0


def test_summarize_all_killed_gives_score_1(tmp_path: Path) -> None:
    junit = _write_junit(tmp_path, _testcase("mutant_1") + _testcase("mutant_2"))

    summary = summarize_mutmut_junit(junit)

    assert summary.total == 2
    assert summary.killed == 2
    assert summary.survived == 0
    assert summary.score == pytest.approx(1.0)


def test_summarize_empty_report_gives_zero_total_and_zero_score_no_division_error(
    tmp_path: Path,
) -> None:
    junit = _write_junit(tmp_path, "")

    summary = summarize_mutmut_junit(junit)

    assert summary.total == 0
    assert summary.killed == 0
    assert summary.score == 0.0


def test_summarize_raises_when_junit_xml_is_missing(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.xml"

    with pytest.raises(MutationSummaryError, match="not found"):
        summarize_mutmut_junit(missing)


def test_summarize_raises_when_junit_xml_is_malformed(tmp_path: Path) -> None:
    junit = tmp_path / "mutmut-junit.xml"
    junit.write_text("<testsuites><not-closed>", encoding="utf-8")

    with pytest.raises(MutationSummaryError, match="malformed"):
        summarize_mutmut_junit(junit)


def test_cli_main_prints_summary_and_exits_0_by_default_even_with_survivors(
    tmp_path: Path, capsys
) -> None:
    junit = _write_junit(tmp_path, _testcase("mutant_1", "survived") + _testcase("mutant_2"))

    exit_code = main(["--junit-xml", str(junit)])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "1/2 killed" in out


def test_cli_main_exits_1_when_junit_xml_is_missing(tmp_path: Path, capsys) -> None:
    missing = tmp_path / "does_not_exist.xml"

    exit_code = main(["--junit-xml", str(missing)])

    assert exit_code == 1
    err = capsys.readouterr().err
    assert "not found" in err


def test_cli_main_exits_0_when_fail_under_passed_and_met(tmp_path: Path) -> None:
    junit = _write_junit(tmp_path, _testcase("mutant_1") + _testcase("mutant_2"))

    exit_code = main(["--junit-xml", str(junit), "--fail-under", "0.5"])

    assert exit_code == 0


def test_cli_main_exits_1_when_fail_under_passed_and_unmet(tmp_path: Path, capsys) -> None:
    junit = _write_junit(tmp_path, _testcase("mutant_1", "survived") + _testcase("mutant_2"))

    exit_code = main(["--junit-xml", str(junit), "--fail-under", "0.9"])

    assert exit_code == 1
    err = capsys.readouterr().err
    assert "FAILED" in err


def test_pyproject_mutmut_paths_to_mutate_matches_the_four_named_modules() -> None:
    """Maintenance tripwire (mirrors required_test_ids.txt's discipline): once
    pyproject.toml carries [tool.mutmut], its paths_to_mutate must resolve to
    exactly the four proof-bearing modules this bead scopes mutation testing
    to, and all four must still exist on disk -- a future rename/removal of
    transport.py/tape.py/matcher.py/blame.py must update the config in the
    same change or this test fails.

    Skips (rather than fails) while [tool.mutmut] itself is not yet present:
    tracefork-bge.49's pyproject.toml diff is a forbidden-file edit deferred
    to the orchestrator (see scripts/mutation_summary.py's module docstring
    and this bead's handoff notes) -- once applied, this test starts
    enforcing the tripwire for real.
    """
    pyproject_path = REPO_ROOT / "pyproject.toml"
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    mutmut_config = data.get("tool", {}).get("mutmut")
    if mutmut_config is None:
        pytest.skip(
            "pyproject.toml has no [tool.mutmut] table yet -- deferred to the "
            "orchestrator for tracefork-bge.49; see mutation_summary.py's docstring."
        )

    paths = {(REPO_ROOT / p).resolve() for p in mutmut_config["paths_to_mutate"]}
    expected = {
        (REPO_ROOT / "src/tracefork/transport.py").resolve(),
        (REPO_ROOT / "src/tracefork/tape.py").resolve(),
        (REPO_ROOT / "src/tracefork/matcher.py").resolve(),
        (REPO_ROOT / "src/tracefork/blame.py").resolve(),
    }
    assert paths == expected
    for path in expected:
        assert path.is_file(), f"missing scoped module: {path}"
