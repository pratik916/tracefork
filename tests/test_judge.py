"""LLM-judge oracle, gold-set calibration, and Rogan-Gladen debiasing tests
(`judge.py`) — all offline, zero API spend. `judge_fn` is always a
deterministic synthetic callable; no real Anthropic (or any) API call ever
happens in this file.
"""

from __future__ import annotations

import math

import pytest

from tracefork.blame import FlipRateResult, get_oracle, registered_oracles, z_from_confidence
from tracefork.judge import (
    CalibrationResult,
    GoldExample,
    JudgeExample,
    LLMJudgeOracle,
    calibrate_oracle,
    debias_flip_rate,
    rogan_gladen_correct,
)

# ── LLMJudgeOracle: registry ─────────────────────────────────────────────────


def test_llm_judge_oracle_registered_by_import():
    assert "llm_judge" in registered_oracles()
    assert get_oracle("llm_judge") is LLMJudgeOracle


# ── LLMJudgeOracle: grading + position-swap + abstain ────────────────────────


def _consistent_judge_fn(prompt: str) -> str:
    """A synthetic judge that reads the SAME verdict regardless of where the
    candidate output sits in the prompt — no position bias."""
    if "GOOD_OUTPUT" in prompt:
        return '{"verdict": "PASS", "confidence": 0.95}'
    return '{"verdict": "FAIL", "confidence": 0.95}'


def test_llm_judge_oracle_grades_pass():
    oracle = LLMJudgeOracle(rubric="Did the agent solve the task?", judge_fn=_consistent_judge_fn)
    assert oracle.grade("GOOD_OUTPUT: task solved") is True


def test_llm_judge_oracle_grades_fail():
    oracle = LLMJudgeOracle(rubric="Did the agent solve the task?", judge_fn=_consistent_judge_fn)
    assert oracle.grade("BAD_OUTPUT: task not solved") is False


def test_llm_judge_oracle_includes_rubric_and_examples_in_prompt():
    seen_prompts: list[str] = []

    def capturing_judge_fn(prompt: str) -> str:
        seen_prompts.append(prompt)
        return '{"verdict": "PASS", "confidence": 0.9}'

    oracle = LLMJudgeOracle(
        rubric="THE_RUBRIC_TEXT",
        judge_fn=capturing_judge_fn,
        examples=(
            JudgeExample(output="ex-pass", verdict=True, rationale="clean solve"),
            JudgeExample(output="ex-fail", verdict=False),
        ),
    )
    oracle.grade("candidate output text")
    assert len(seen_prompts) == 2  # position-swap: two calls
    for prompt in seen_prompts:
        assert "THE_RUBRIC_TEXT" in prompt
        assert "ex-pass" in prompt and "ex-fail" in prompt
        assert "clean solve" in prompt
        assert "candidate output text" in prompt


def test_llm_judge_oracle_position_swap_orders_differ():
    """The two graded prompts must actually differ in section order — position
    swap is only a real bias check if the candidate output moves."""
    seen_prompts: list[str] = []

    def capturing_judge_fn(prompt: str) -> str:
        seen_prompts.append(prompt)
        return '{"verdict": "PASS", "confidence": 0.9}'

    oracle = LLMJudgeOracle(rubric="R", judge_fn=capturing_judge_fn)
    oracle.grade("CANDIDATE")
    assert len(seen_prompts) == 2
    idx_output_0 = seen_prompts[0].index("CANDIDATE")
    idx_rubric_0 = seen_prompts[0].index("RUBRIC:")
    idx_output_1 = seen_prompts[1].index("CANDIDATE")
    idx_rubric_1 = seen_prompts[1].index("RUBRIC:")
    # In one ordering the output comes before the rubric; in the other, after.
    assert (idx_output_0 < idx_rubric_0) != (idx_output_1 < idx_rubric_1)


def test_llm_judge_oracle_abstains_on_position_disagreement():
    """A judge whose verdict flips depending on where the candidate sits in
    the prompt is exhibiting position bias — grade() must abstain (None)
    rather than silently trust one ordering."""

    def position_biased_judge_fn(prompt: str) -> str:
        # Output appears first in the "output_first" ordering.
        if prompt.startswith("OUTPUT TO GRADE"):
            return '{"verdict": "FAIL", "confidence": 0.9}'
        return '{"verdict": "PASS", "confidence": 0.9}'

    oracle = LLMJudgeOracle(rubric="R", judge_fn=position_biased_judge_fn)
    assert oracle.grade("anything") is None


def test_llm_judge_oracle_abstains_on_low_confidence():
    def low_confidence_judge_fn(prompt: str) -> str:
        return '{"verdict": "PASS", "confidence": 0.3}'

    oracle = LLMJudgeOracle(rubric="R", judge_fn=low_confidence_judge_fn, confidence_threshold=0.6)
    assert oracle.grade("anything") is None


def test_llm_judge_oracle_high_confidence_passes_threshold():
    def high_confidence_judge_fn(prompt: str) -> str:
        return '{"verdict": "PASS", "confidence": 0.85}'

    oracle = LLMJudgeOracle(rubric="R", judge_fn=high_confidence_judge_fn, confidence_threshold=0.6)
    assert oracle.grade("anything") is True


def test_llm_judge_oracle_short_circuits_on_unparseable_first_call():
    """An unparseable first response abstains without wasting a second (real,
    budget-capped) judge call."""
    calls = []

    def gibberish_judge_fn(prompt: str) -> str:
        calls.append(prompt)
        return "not a verdict at all"

    oracle = LLMJudgeOracle(rubric="R", judge_fn=gibberish_judge_fn)
    assert oracle.grade("anything") is None
    assert len(calls) == 1


def test_llm_judge_oracle_fallback_keyword_parsing_without_json():
    def plain_text_judge_fn(prompt: str) -> str:
        return "My verdict is PASS, I am confident."

    oracle = LLMJudgeOracle(rubric="R", judge_fn=plain_text_judge_fn)
    assert oracle.grade("anything") is True


def test_llm_judge_oracle_position_swap_disabled_uses_single_call():
    calls = []

    def judge_fn(prompt: str) -> str:
        calls.append(prompt)
        return '{"verdict": "FAIL", "confidence": 0.9}'

    oracle = LLMJudgeOracle(rubric="R", judge_fn=judge_fn, position_swap=False)
    assert oracle.grade("anything") is False
    assert len(calls) == 1


# ── LLMJudgeOracle: cross-family / self-judge guard ──────────────────────────


def test_llm_judge_oracle_rejects_self_judge_by_default():
    with pytest.raises(ValueError, match="self-preference"):
        LLMJudgeOracle(
            rubric="R",
            judge_fn=_consistent_judge_fn,
            judge_model="claude-sonnet-4-6",
            graded_model="claude-sonnet-4-6",
        )


def test_llm_judge_oracle_allows_self_judge_when_overridden():
    oracle = LLMJudgeOracle(
        rubric="R",
        judge_fn=_consistent_judge_fn,
        judge_model="claude-sonnet-4-6",
        graded_model="claude-sonnet-4-6",
        allow_self_judge=True,
    )
    assert oracle.grade("GOOD_OUTPUT") is True


def test_llm_judge_oracle_cross_family_configured_without_error():
    oracle = LLMJudgeOracle(
        rubric="R",
        judge_fn=_consistent_judge_fn,
        judge_model="claude-opus-4-8",
        graded_model="claude-sonnet-4-6",
    )
    assert oracle.grade("GOOD_OUTPUT") is True


# ── gold-set calibration: FPR/FNR/kappa ──────────────────────────────────────


class _ScriptedOracle:
    """A test-double Oracle: fixed text -> verdict mapping (no LLM call)."""

    def __init__(self, predictions: dict[str, bool | None]) -> None:
        self._predictions = predictions

    def grade(self, output: str) -> bool | None:
        return self._predictions[output]


def test_calibrate_oracle_kappa_at_alert_boundary_is_not_alerted():
    # tp=4, tn=4, fp=1, fn=1 -> po=0.8, pe=0.5, kappa=(0.8-0.5)/0.5=0.6 exactly.
    predictions = {
        "tp0": True,
        "tp1": True,
        "tp2": True,
        "tp3": True,
        "fn0": False,
        "fp0": True,
        "tn0": False,
        "tn1": False,
        "tn2": False,
        "tn3": False,
    }
    gold = [
        GoldExample("tp0", True),
        GoldExample("tp1", True),
        GoldExample("tp2", True),
        GoldExample("tp3", True),
        GoldExample("fn0", True),
        GoldExample("fp0", False),
        GoldExample("tn0", False),
        GoldExample("tn1", False),
        GoldExample("tn2", False),
        GoldExample("tn3", False),
    ]
    result = calibrate_oracle(_ScriptedOracle(predictions), gold)
    assert result.tp == 4
    assert result.tn == 4
    assert result.fp == 1
    assert result.fn == 1
    assert result.n_pos == 5
    assert result.n_neg == 5
    assert result.fpr == pytest.approx(0.2)
    assert result.fnr == pytest.approx(0.2)
    assert result.kappa == pytest.approx(0.6, abs=1e-9)
    assert result.kappa_alert is False


def test_calibrate_oracle_low_kappa_triggers_alert():
    # tp=3, tn=3, fp=2, fn=2 -> po=0.6, pe=0.5, kappa=(0.6-0.5)/0.5=0.2 < 0.6.
    predictions = {
        "tp0": True,
        "tp1": True,
        "tp2": True,
        "fn0": False,
        "fn1": False,
        "fp0": True,
        "fp1": True,
        "tn0": False,
        "tn1": False,
        "tn2": False,
    }
    gold = [
        GoldExample("tp0", True),
        GoldExample("tp1", True),
        GoldExample("tp2", True),
        GoldExample("fn0", True),
        GoldExample("fn1", True),
        GoldExample("fp0", False),
        GoldExample("fp1", False),
        GoldExample("tn0", False),
        GoldExample("tn1", False),
        GoldExample("tn2", False),
    ]
    result = calibrate_oracle(_ScriptedOracle(predictions), gold)
    assert result.kappa == pytest.approx(0.2, abs=1e-9)
    assert result.kappa_alert is True


def test_calibrate_oracle_excludes_abstentions_from_confusion_matrix():
    predictions = {"a": True, "b": False, "c": None, "d": None}
    gold = [
        GoldExample("a", True),  # tp
        GoldExample("b", False),  # tn
        GoldExample("c", True),  # abstain
        GoldExample("d", False),  # abstain
    ]
    result = calibrate_oracle(_ScriptedOracle(predictions), gold)
    assert result.n == 4
    assert result.n_abstain == 2
    assert result.tp == 1
    assert result.tn == 1
    assert result.fp == 0
    assert result.fn == 0
    assert result.n_pos == 1
    assert result.n_neg == 1


def test_calibrate_oracle_perfect_judge_has_kappa_one_and_zero_error():
    predictions = {"a": True, "b": True, "c": False, "d": False}
    gold = [
        GoldExample("a", True),
        GoldExample("b", True),
        GoldExample("c", False),
        GoldExample("d", False),
    ]
    result = calibrate_oracle(_ScriptedOracle(predictions), gold)
    assert result.kappa == pytest.approx(1.0)
    assert result.fpr == 0.0
    assert result.fnr == 0.0
    assert result.kappa_alert is False


# ── Rogan-Gladen prevalence correction ───────────────────────────────────────


def test_rogan_gladen_correct_worked_example():
    # fpr=0.1, fnr=0.2 -> denom=1-0.1-0.2=0.7; true=(q-0.1)/0.7
    assert rogan_gladen_correct(0.5, fpr=0.1, fnr=0.2) == pytest.approx(4 / 7, rel=1e-9)
    assert rogan_gladen_correct(0.1, fpr=0.1, fnr=0.2) == pytest.approx(0.0, abs=1e-9)


def test_rogan_gladen_correct_clamps_to_unit_interval():
    # observed=0.0 with this fpr/fnr algebraically corrects below 0 -> clamped.
    assert rogan_gladen_correct(0.0, fpr=0.1, fnr=0.2) == 0.0
    # observed=1.0 algebraically corrects above 1 -> clamped.
    assert rogan_gladen_correct(1.0, fpr=0.1, fnr=0.2) == 1.0


def test_rogan_gladen_correct_chance_level_judge_returns_observed_unchanged():
    # fpr + fnr == 1 -> denominator ~0 -> no information to correct with.
    assert rogan_gladen_correct(0.37, fpr=0.5, fnr=0.5) == pytest.approx(0.37)


def test_rogan_gladen_correct_perfect_judge_is_identity():
    assert rogan_gladen_correct(0.42, fpr=0.0, fnr=0.0) == pytest.approx(0.42)


# ── debias_flip_rate: CI widening ────────────────────────────────────────────


def test_debias_flip_rate_matches_closed_form_and_widens_ci():
    q, fpr, fnr = 0.5, 0.1, 0.2
    valid_trials = 100
    calibration = CalibrationResult(
        n=1000,
        n_abstain=0,
        n_pos=500,
        n_neg=500,
        tp=400,
        tn=450,
        fp=50,
        fn=100,
        fpr=fpr,
        fnr=fnr,
        kappa=0.65,
        kappa_alert=False,
    )
    result = FlipRateResult(
        step_index=3,
        flip_rate=q,
        ci_lo=0.35,
        ci_hi=0.65,
        flips=50,
        trials=valid_trials,
        valid_trials=valid_trials,
    )

    out = debias_flip_rate(result, calibration, confidence=0.95)

    expected_corrected = (q - fpr) / (1.0 - fpr - fnr)
    assert out.step_index == 3
    assert out.corrected_flip_rate == pytest.approx(expected_corrected, rel=1e-9)
    assert out.corrected_flip_rate == pytest.approx(4 / 7, rel=1e-9)
    assert out.raw_flip_rate == q
    assert out.raw_ci_lo == result.ci_lo
    assert out.raw_ci_hi == result.ci_hi
    assert out.kappa_alert is False

    # Independent closed-form delta-method check (re-derived here, not
    # imported from judge.py, so this is a genuine cross-check).
    denom = 1.0 - fpr - fnr
    numerator = q - fpr
    var_q = q * (1.0 - q) / valid_trials
    var_fpr = fpr * (1.0 - fpr) / calibration.n_neg
    var_fnr = fnr * (1.0 - fnr) / calibration.n_pos
    d_dq = 1.0 / denom
    d_dfpr = (numerator - denom) / denom**2
    d_dfnr = numerator / denom**2
    var_true = d_dq**2 * var_q + d_dfpr**2 * var_fpr + d_dfnr**2 * var_fnr
    z = z_from_confidence(0.95)
    margin = z * math.sqrt(var_true)
    assert out.ci_lo == pytest.approx(expected_corrected - margin, rel=1e-9)
    assert out.ci_hi == pytest.approx(expected_corrected + margin, rel=1e-9)

    # Widening: judge noise (finite gold-set FPR/FNR) must widen the interval
    # beyond what k-sampling noise on its own would give.
    sampling_only_margin = z * math.sqrt(var_q)
    assert (out.ci_hi - out.ci_lo) > 2 * sampling_only_margin


def test_debias_flip_rate_ci_narrows_as_gold_set_grows():
    """More gold-set evidence for the SAME FPR/FNR must narrow the corrected
    CI back toward the sampling-only width, converging to the same point
    estimate — this is the "judge noise" term shrinking, isolated from the
    sampling-noise term."""
    q, fpr, fnr = 0.4, 0.05, 0.1
    result = FlipRateResult(
        step_index=0, flip_rate=q, ci_lo=0.2, ci_hi=0.6, flips=40, trials=100, valid_trials=100
    )
    small_gold = CalibrationResult(
        n=40,
        n_abstain=0,
        n_pos=20,
        n_neg=20,
        tp=18,
        tn=18,
        fp=1,
        fn=1,
        fpr=fpr,
        fnr=fnr,
        kappa=0.8,
        kappa_alert=False,
    )
    large_gold = CalibrationResult(
        n=4000,
        n_abstain=0,
        n_pos=2000,
        n_neg=2000,
        tp=1800,
        tn=1800,
        fp=100,
        fn=100,
        fpr=fpr,
        fnr=fnr,
        kappa=0.8,
        kappa_alert=False,
    )

    small = debias_flip_rate(result, small_gold)
    large = debias_flip_rate(result, large_gold)

    assert (small.ci_hi - small.ci_lo) > (large.ci_hi - large.ci_lo)
    assert small.corrected_flip_rate == pytest.approx(large.corrected_flip_rate, rel=1e-9)


def test_debias_flip_rate_propagates_kappa_alert_flag():
    result = FlipRateResult(
        step_index=1, flip_rate=0.3, ci_lo=0.1, ci_hi=0.5, flips=3, trials=10, valid_trials=10
    )
    calibration = CalibrationResult(
        n=20,
        n_abstain=0,
        n_pos=10,
        n_neg=10,
        tp=6,
        tn=6,
        fp=4,
        fn=4,
        fpr=0.4,
        fnr=0.4,
        kappa=0.2,
        kappa_alert=True,
    )
    out = debias_flip_rate(result, calibration)
    assert out.kappa_alert is True
