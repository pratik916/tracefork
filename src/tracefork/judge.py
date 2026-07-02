"""LLM-judge oracle, gold-set calibration, and Rogan-Gladen debiasing.

``StringMatchOracle`` (in ``blame.py``) can only grade outcomes a regex can
see. Open-ended outcomes ("did the agent actually solve the task?") need a
judge — but a judge is a NOISY instrument: it has its own false-positive and
false-negative rate against ground truth, and today's blame confidence
intervals reflect only k-sampling noise while silently treating the oracle as
ground truth. This module adds three additive, OPT-IN pieces on top of the
``Oracle`` protocol defined in ``blame.py``:

1. ``LLMJudgeOracle`` — a binary rubric-based judge with few-shot examples, a
   configurable ("cross-family") judge model with a self-judge guard,
   position-swap averaging (grade twice with the candidate output in a
   different place in the prompt; disagreement or low average confidence
   abstains rather than guessing), and ``None`` abstention. It is testable
   OFFLINE: the caller injects ``judge_fn`` (prompt -> raw response text), so
   tests use a deterministic synthetic judge — no real API in tests. Wiring
   ``judge_fn`` to a live Anthropic call is the caller's job on the
   budget-capped ``blame`` path.
2. ``calibrate_oracle`` — runs any ``Oracle`` (typically an ``LLMJudgeOracle``)
   over a labeled gold set and measures its false-positive rate, false-negative
   rate, and Cohen's kappa against ground truth, flagging ``kappa_alert`` when
   kappa < 0.6 (poor-agreement threshold, Landis & Koch 1977).
3. ``rogan_gladen_correct`` / ``debias_flip_rate`` — the Rogan-Gladen (1978)
   prevalence-correction estimator, applied to a step's observed flip-rate
   using the judge's calibrated FPR/FNR, plus a delta-method propagation of
   BOTH the k-sampling variance and the finite-gold-set FPR/FNR variance into
   a widened confidence interval around the corrected estimate.

``StringMatchOracle`` remains the default oracle everywhere in tracefork; none
of this module is imported or exercised by the default $0 offline path. Pure
Python/math throughout (reuses ``blame.z_from_confidence``) — no scipy, no new
hard dependency.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .blame import FlipRateResult, Oracle, register_oracle, z_from_confidence

# ── few-shot examples ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class JudgeExample:
    """One few-shot example shown to the judge: a candidate output, its
    ground-truth verdict, and an optional one-line rationale."""

    output: str
    verdict: bool
    rationale: str = ""


# ── LLM-judge oracle ─────────────────────────────────────────────────────────

_JSON_VERDICT_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)
_PASS_RE = re.compile(r"\bPASS\b", re.IGNORECASE)
_FAIL_RE = re.compile(r"\bFAIL\b", re.IGNORECASE)


def _parse_verdict(raw: str) -> tuple[bool | None, float]:
    """Parse a judge's raw response into ``(verdict, confidence)``.

    Prefers a JSON object with ``verdict``/``confidence`` keys (the format the
    prompt asks for); falls back to a bare PASS/FAIL keyword scan with an
    assumed confidence of 1.0 (no confidence signal available). Returns
    ``(None, 0.0)`` if neither is found — an unparseable response abstains.
    """
    for match in _JSON_VERDICT_RE.finditer(raw):
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        verdict_raw = obj.get("verdict")
        if isinstance(verdict_raw, str):
            v = verdict_raw.strip().upper()
            if v == "PASS":
                conf = obj.get("confidence", 1.0)
                return True, float(conf) if isinstance(conf, (int, float)) else 1.0
            if v == "FAIL":
                conf = obj.get("confidence", 1.0)
                return False, float(conf) if isinstance(conf, (int, float)) else 1.0
        if isinstance(verdict_raw, bool):
            conf = obj.get("confidence", 1.0)
            return verdict_raw, float(conf) if isinstance(conf, (int, float)) else 1.0

    has_pass = _PASS_RE.search(raw) is not None
    has_fail = _FAIL_RE.search(raw) is not None
    if has_pass and not has_fail:
        return True, 1.0
    if has_fail and not has_pass:
        return False, 1.0
    return None, 0.0


class LLMJudgeOracle:
    """``Oracle`` backed by an LLM judge: binary rubric grading with few-shot
    examples, a cross-family judge model, position-swap averaging, and
    abstention (``None``) on disagreement or low confidence.

    Testable OFFLINE: ``judge_fn`` is injected (``prompt -> raw response
    text``), so tests pass a deterministic synthetic judge — this class never
    makes a network call itself. A live caller wires ``judge_fn`` to e.g.
    ``lambda prompt: client.messages.create(...).content[0].text``.
    """

    def __init__(
        self,
        *,
        rubric: str,
        judge_fn: Callable[[str], str],
        examples: Sequence[JudgeExample] = (),
        judge_model: str | None = None,
        graded_model: str | None = None,
        allow_self_judge: bool = False,
        confidence_threshold: float = 0.6,
        position_swap: bool = True,
    ) -> None:
        if (
            not allow_self_judge
            and judge_model is not None
            and graded_model is not None
            and judge_model == graded_model
        ):
            raise ValueError(
                f"judge_model == graded_model ({judge_model!r}): a judge should "
                "not grade its own family's output (self-preference bias); pass "
                "a different judge_model or allow_self_judge=True to override."
            )
        self._rubric = rubric
        self._judge_fn = judge_fn
        self._examples = tuple(examples)
        self._judge_model = judge_model
        self._graded_model = graded_model
        self._confidence_threshold = confidence_threshold
        self._position_swap = position_swap

    def grade(self, output: str) -> bool | None:
        verdict_a, conf_a = self._grade_once(output, output_first=False)
        if not self._position_swap:
            if verdict_a is None or conf_a < self._confidence_threshold:
                return None
            return verdict_a

        if verdict_a is None:
            # Unparseable first response: abstain without spending a second
            # (real, budget-capped) judge call on a doomed comparison.
            return None
        verdict_b, conf_b = self._grade_once(output, output_first=True)
        if verdict_b is None or verdict_a != verdict_b:
            # Unparseable second response, or a position-bias signal (the
            # judge's verdict changed depending on where the candidate output
            # sits in the prompt) — abstain rather than trust one ordering.
            return None
        avg_conf = (conf_a + conf_b) / 2.0
        if avg_conf < self._confidence_threshold:
            return None
        return verdict_a

    def _grade_once(self, output: str, *, output_first: bool) -> tuple[bool | None, float]:
        prompt = self._build_prompt(output, output_first=output_first)
        raw = self._judge_fn(prompt)
        return _parse_verdict(raw)

    def _build_prompt(self, output: str, *, output_first: bool) -> str:
        examples_block = "\n\n".join(
            f"Example output:\n{ex.output}\nVerdict: {'PASS' if ex.verdict else 'FAIL'}"
            + (f"\nRationale: {ex.rationale}" if ex.rationale else "")
            for ex in self._examples
        )
        output_block = (
            f"OUTPUT TO GRADE:\n{output}\n\n"
            'Respond with a JSON object: {"verdict": "PASS"|"FAIL", "confidence": 0.0-1.0}'
        )
        rubric_block = f"RUBRIC:\n{self._rubric}"
        sections = (
            [output_block, rubric_block, examples_block]
            if output_first
            else [rubric_block, examples_block, output_block]
        )
        return "\n\n".join(s for s in sections if s)


register_oracle("llm_judge", LLMJudgeOracle)


# ── gold-set calibration ─────────────────────────────────────────────────────

#: Cohen's kappa below this is "poor agreement, at best" (Landis & Koch 1977);
#: `calibrate_oracle` flags it via `CalibrationResult.kappa_alert`.
KAPPA_ALERT_THRESHOLD = 0.6


@dataclass(frozen=True)
class GoldExample:
    """One labeled gold-set example: candidate text plus ground-truth label."""

    text: str
    label: bool


@dataclass
class CalibrationResult:
    """A judge's measured error rates against a labeled gold set."""

    n: int
    n_abstain: int
    n_pos: int  # gold positives among non-abstained trials (tp + fn)
    n_neg: int  # gold negatives among non-abstained trials (tn + fp)
    tp: int
    tn: int
    fp: int
    fn: int
    fpr: float  # P(judge=True | truth=False) — false positive rate
    fnr: float  # P(judge=False | truth=True) — false negative rate
    kappa: float
    kappa_alert: bool = False


def _cohens_kappa(tp: int, tn: int, fp: int, fn: int) -> float:
    """Cohen's kappa for a 2x2 judge-vs-truth confusion matrix."""
    n = tp + tn + fp + fn
    if n == 0:
        return 0.0
    po = (tp + tn) / n
    p_pred_pos = (tp + fp) / n
    p_actual_pos = (tp + fn) / n
    p_pred_neg = (tn + fn) / n
    p_actual_neg = (tn + fp) / n
    pe = p_pred_pos * p_actual_pos + p_pred_neg * p_actual_neg
    if pe >= 1.0:
        return 1.0 if po >= 1.0 else 0.0
    return (po - pe) / (1.0 - pe)


def calibrate_oracle(oracle: Oracle, gold: Sequence[GoldExample]) -> CalibrationResult:
    """Measure ``oracle``'s FPR/FNR/Cohen's kappa against a labeled gold set.

    Any ``Oracle`` (``grade(text: str) -> bool | None``) works — typically an
    ``LLMJudgeOracle``, but this is generic, including ``StringMatchOracle``.
    Abstentions (``grade`` returning ``None``) are excluded from the confusion
    matrix (there is no verdict to score) and tallied separately in
    ``n_abstain`` — a high abstain rate is itself a calibration signal even
    though it doesn't enter FPR/FNR/kappa. Pure math, offline.
    """
    tp = tn = fp = fn = 0
    n_abstain = 0
    for ex in gold:
        pred = oracle.grade(ex.text)
        if pred is None:
            n_abstain += 1
            continue
        if ex.label and pred:
            tp += 1
        elif ex.label and not pred:
            fn += 1
        elif not ex.label and pred:
            fp += 1
        else:
            tn += 1

    n_pos = tp + fn
    n_neg = tn + fp
    fpr = fp / n_neg if n_neg > 0 else 0.0
    fnr = fn / n_pos if n_pos > 0 else 0.0
    kappa = _cohens_kappa(tp, tn, fp, fn)
    return CalibrationResult(
        n=len(gold),
        n_abstain=n_abstain,
        n_pos=n_pos,
        n_neg=n_neg,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        fpr=fpr,
        fnr=fnr,
        kappa=kappa,
        kappa_alert=kappa < KAPPA_ALERT_THRESHOLD,
    )


# ── Rogan-Gladen debiasing + CI widening ─────────────────────────────────────


def rogan_gladen_correct(observed: float, *, fpr: float, fnr: float) -> float:
    """Rogan-Gladen (1978) prevalence correction: debias an observed flip-rate
    for a judge's known false-positive/false-negative rate.

    ``true = (observed - fpr) / (1 - fpr - fnr)``, clamped to ``[0, 1]``. This
    is the standard estimator (``AP = true·Se + (1-true)·FPR`` solved for
    ``true``, with ``Se = 1 - fnr`` the judge's sensitivity) — sanity-checked
    by its two fixed points: a perfect judge (``fpr = fnr = 0``) must be the
    identity (``true == observed``, and indeed ``(observed-0)/(1-0-0) ==
    observed``), and a judge that is always wrong (``fpr = fnr = 1``) must
    exactly invert (``(observed-1)/(1-1-1) == (observed-1)/-1 ==
    1-observed``). When the judge performs at chance (``fpr + fnr ≈ 1``,
    denominator ≈ 0) there is no information to correct with — the observed
    rate (clamped) is returned unchanged rather than dividing by ~0.
    """
    denom = 1.0 - fpr - fnr
    if abs(denom) < 1e-9:
        return max(0.0, min(1.0, observed))
    corrected = (observed - fpr) / denom
    return max(0.0, min(1.0, corrected))


@dataclass
class DebiasedFlipRate:
    """A step's Rogan-Gladen-corrected flip-rate, with a CI widened to reflect
    both k-sampling noise and judge-calibration (FPR/FNR) noise."""

    step_index: int
    raw_flip_rate: float
    raw_ci_lo: float
    raw_ci_hi: float
    corrected_flip_rate: float
    ci_lo: float
    ci_hi: float
    kappa_alert: bool = False


def debias_flip_rate(
    result: FlipRateResult,
    calibration: CalibrationResult,
    *,
    confidence: float = 0.95,
) -> DebiasedFlipRate:
    """Correct one step's flip-rate for judge noise (Rogan-Gladen) and widen
    its confidence interval to reflect BOTH sampling noise (the k-trial
    binomial variance behind ``result``'s own CI) and judge-calibration noise
    (the FPR/FNR estimated on ``calibration``'s finite gold set), via a
    first-order (delta-method) propagation of error through the Rogan-Gladen
    formula. Pure math, offline; does not mutate ``result``.

    Given ``true = (q - fpr) / (1 - fpr - fnr)`` with ``q`` the observed
    flip-rate:

        d(true)/dq   = 1 / D
        d(true)/dfpr = (N - D) / D²
        d(true)/dfnr = N / D²

    where ``N = q - fpr`` and ``D = 1 - fpr - fnr``, and
    ``Var(true) ≈ Σ (∂true/∂x)² Var(x)`` over the three (assumed independent)
    inputs — each with its own binomial variance (``q`` over ``result``'s
    valid trials, ``fpr``/``fnr`` over the gold set's negative/positive
    counts). Any additional judge-calibration variance only widens the
    interval relative to sampling noise alone.
    """
    fpr, fnr = calibration.fpr, calibration.fnr
    observed = result.flip_rate
    corrected = rogan_gladen_correct(observed, fpr=fpr, fnr=fnr)
    z = z_from_confidence(confidence)

    n = result.valid_trials
    var_q = (observed * (1.0 - observed) / n) if n > 0 else 0.0
    var_fpr = (fpr * (1.0 - fpr) / calibration.n_neg) if calibration.n_neg > 0 else 0.0
    var_fnr = (fnr * (1.0 - fnr) / calibration.n_pos) if calibration.n_pos > 0 else 0.0

    denom = 1.0 - fpr - fnr
    if abs(denom) < 1e-9:
        # No correction applied (chance-level judge) — CI reflects sampling
        # noise only, same shape as an uncorrected Wald interval on `q`.
        se = math.sqrt(var_q)
        lo, hi = max(0.0, observed - z * se), min(1.0, observed + z * se)
    else:
        numerator = observed - fpr
        d_dq = 1.0 / denom
        d_dfpr = (numerator - denom) / (denom * denom)
        d_dfnr = numerator / (denom * denom)
        var_true = (d_dq**2) * var_q + (d_dfpr**2) * var_fpr + (d_dfnr**2) * var_fnr
        se_true = math.sqrt(max(0.0, var_true))
        lo = max(0.0, corrected - z * se_true)
        hi = min(1.0, corrected + z * se_true)

    return DebiasedFlipRate(
        step_index=result.step_index,
        raw_flip_rate=observed,
        raw_ci_lo=result.ci_lo,
        raw_ci_hi=result.ci_hi,
        corrected_flip_rate=corrected,
        ci_lo=lo,
        ci_hi=hi,
        kappa_alert=calibration.kappa_alert,
    )
