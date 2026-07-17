"""Tests for `tracefork.narrative`: deterministic markdown templating over
hand-built `blame.py` dataclasses.

Every input here is constructed directly -- no `BlameEngine.rank()`/
`shapley_rank()`, no fork, no network, no synthetic transport. These
functions are pure string formatting over already-computed fields, so the
tests only need to prove the formatting is exact and repeatable.
"""

from __future__ import annotations

from tracefork.blame import BlameReport, CIMethod, FlipRateResult, ShapleyReport, ShapleyResult
from tracefork.narrative import (
    explain_blame_report,
    explain_flip_result,
    explain_shapley_report,
    explain_shapley_result,
)


def _flip_result(**overrides: object) -> FlipRateResult:
    defaults: dict[str, object] = {
        "step_index": 2,
        "flip_rate": 0.7,
        "ci_lo": 0.45,
        "ci_hi": 0.88,
        "flips": 7,
        "trials": 10,
        "interpretation": "decisive — this step caused it",
        "valid_trials": 10,
        "undefined": 0,
        "divergences": 0,
        "divergence_rate": 0.0,
        "trustworthy": True,
        "p_value": 0.01,
        "q_value": 0.012,
        "responsible": True,
    }
    defaults.update(overrides)
    return FlipRateResult(**defaults)  # type: ignore[arg-type]


def _shapley_result(**overrides: object) -> ShapleyResult:
    defaults: dict[str, object] = {
        "step_index": 1,
        "shapley_value": 0.6,
        "ci_lo": 0.4,
        "ci_hi": 0.8,
        "n_samples": 5,
        "coalition_flip_rate": 0.8,
        "base_flip_rate": 0.2,
        "interpretation": "decisive — this step caused it",
        "necessity": True,
        "necessity_score": 0.6,
        "sufficiency": True,
        "sufficiency_score": 0.75,
    }
    defaults.update(overrides)
    return ShapleyResult(**defaults)  # type: ignore[arg-type]


# ── explain_flip_result ──────────────────────────────────────────────────────


def test_explain_flip_result_fixed_format() -> None:
    r = _flip_result()
    text = explain_flip_result(r)
    assert text == (
        "Step 2: flip rate 70% (CI [45%, 88%], q=0.012) — "
        "decisive — this step caused it; responsible, trustworthy."
    )


def test_explain_flip_result_byte_identical_on_repeat() -> None:
    r = _flip_result()
    assert explain_flip_result(r) == explain_flip_result(r)


def test_explain_flip_result_untrustworthy_not_responsible() -> None:
    r = _flip_result(
        step_index=0,
        flip_rate=0.1,
        ci_lo=0.0,
        ci_hi=0.3,
        q_value=0.9,
        interpretation="diffuse — not the cause",
        trustworthy=False,
        responsible=False,
    )
    text = explain_flip_result(r)
    assert text == (
        "Step 0: flip rate 10% (CI [0%, 30%], q=0.9) — "
        "diffuse — not the cause; not responsible, untrustworthy."
    )


# ── explain_shapley_result ───────────────────────────────────────────────────


def test_explain_shapley_result_necessity_and_sufficiency_wording() -> None:
    r = _shapley_result()
    text = explain_shapley_result(r)
    assert text == (
        "Step 1: Shapley value 60% (CI [40%, 80%]) — decisive — this step caused it; "
        "necessary (60%), sufficient (75%)."
    )


def test_explain_shapley_result_not_necessary_not_sufficient() -> None:
    r = _shapley_result(
        step_index=3,
        shapley_value=0.05,
        ci_lo=-0.05,
        ci_hi=0.15,
        interpretation="diffuse — not the cause",
        necessity=False,
        necessity_score=0.05,
        sufficiency=False,
        sufficiency_score=0.0,
    )
    text = explain_shapley_result(r)
    assert text == (
        "Step 3: Shapley value 5% (CI [-5%, 15%]) — diffuse — not the cause; "
        "not necessary (5%), not sufficient (0%)."
    )


def test_explain_shapley_result_wording_driven_only_by_necessity_sufficiency_fields() -> None:
    # coalition_flip_rate/base_flip_rate differ but necessity/sufficiency
    # fields are identical -> the necessity/sufficiency clause is identical.
    a = _shapley_result(coalition_flip_rate=0.9, base_flip_rate=0.1)
    b = _shapley_result(coalition_flip_rate=0.3, base_flip_rate=0.25)
    clause_a = explain_shapley_result(a).split("; ", 1)[1]
    clause_b = explain_shapley_result(b).split("; ", 1)[1]
    assert clause_a == clause_b


# ── explain_blame_report ─────────────────────────────────────────────────────


def test_explain_blame_report_one_bullet_per_step_and_responsible_summary() -> None:
    r0 = _flip_result(step_index=0, flip_rate=0.1, q_value=0.9, responsible=False)
    r1 = _flip_result(step_index=1, flip_rate=0.8, q_value=0.02, responsible=True)
    report = BlameReport(
        results=[r0, r1],
        k=10,
        total_forks=20,
        ci_method=CIMethod.WILSON,
        confidence=0.95,
        fdr_q=0.10,
        responsible_set=[1],
    )
    text = explain_blame_report(report)
    lines = text.splitlines()
    assert lines[0] == "# Blame report"
    assert lines[2] == "k=10, 20 total forks, wilson 95% CI, FDR q≤0.1."
    assert f"- {explain_flip_result(r0)}" in lines
    assert f"- {explain_flip_result(r1)}" in lines
    assert lines[-1] == "**Responsible set** (FDR q≤0.1): step 1 (q=0.02)."


def test_explain_blame_report_responsible_subset_sorted_like_responsible_method() -> None:
    # Two responsible steps: report.responsible_set is ascending by step
    # index, but the narrative orders them like BlameReport.responsible()
    # does -- ascending q-value, then descending flip-rate.
    r_low_q = _flip_result(step_index=5, flip_rate=0.5, q_value=0.01, responsible=True)
    r_high_q = _flip_result(step_index=2, flip_rate=0.9, q_value=0.08, responsible=True)
    report = BlameReport(
        results=[r_low_q, r_high_q],
        k=10,
        total_forks=20,
        responsible_set=[2, 5],
    )
    text = explain_blame_report(report)
    summary = text.splitlines()[-1]
    assert summary == "**Responsible set** (FDR q≤0.1): step 5 (q=0.01), step 2 (q=0.08)."
    assert report.responsible_set == [2, 5]  # unchanged, ascending-index order


def test_explain_blame_report_empty_results_and_empty_responsible_set() -> None:
    report = BlameReport(results=[], k=10, total_forks=0, responsible_set=[])
    text = explain_blame_report(report)
    lines = text.splitlines()
    assert lines[0] == "# Blame report"
    assert lines[-1] == "**Responsible set**: no step cleared the significance bar."
    assert not any(line.startswith("- ") for line in lines)


# ── explain_shapley_report ───────────────────────────────────────────────────


def test_explain_shapley_report_one_bullet_per_step_and_necessary_summary() -> None:
    s0 = _shapley_result(step_index=0, shapley_value=0.1, necessity=False)
    s1 = _shapley_result(step_index=1, shapley_value=0.7, necessity=True)
    report = ShapleyReport(results=[s0, s1], n_permutation_samples=50, k=10, total_forks=20)
    text = explain_shapley_report(report)
    lines = text.splitlines()
    assert lines[0] == "# Shapley report"
    assert lines[2] == "50 permutation samples, k=10, 20 total forks."
    assert f"- {explain_shapley_result(s0)}" in lines
    assert f"- {explain_shapley_result(s1)}" in lines
    assert lines[-1] == "**Necessary steps**: step 1."


def test_explain_shapley_report_no_necessary_steps() -> None:
    s0 = _shapley_result(step_index=0, necessity=False)
    report = ShapleyReport(results=[s0], n_permutation_samples=50, k=10, total_forks=20)
    text = explain_shapley_report(report)
    assert text.splitlines()[-1] == "**Necessary steps**: no step cleared the necessity bar."


def test_explain_shapley_report_empty_results() -> None:
    report = ShapleyReport(results=[], n_permutation_samples=50, k=10, total_forks=0)
    text = explain_shapley_report(report)
    lines = text.splitlines()
    assert lines[0] == "# Shapley report"
    assert lines[-1] == "**Necessary steps**: no step cleared the necessity bar."
    assert not any(line.startswith("- ") for line in lines)
