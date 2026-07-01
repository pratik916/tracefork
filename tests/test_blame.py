"""Blame engine tests — all offline, zero API spend."""

import anthropic
import httpx
import pytest

from tests.fakes import ScriptedFakeLLM, make_text_response
from tracefork.blame import (
    BlameEngine,
    BudgetGovernor,
    CIMethod,
    StringMatchOracle,
    benjamini_hochberg,
    binom_sf_ge,
    proportion_ci,
    wilson_ci,
    z_from_confidence,
)
from tracefork.tape import Tape
from tracefork.transport import TraceforkTransport

# ── Wilson CI ────────────────────────────────────────────────────────────────


def test_wilson_ci_all_flips():
    lo, hi = wilson_ci(10, 10)
    assert lo > 0.6
    assert hi <= 1.0


def test_wilson_ci_no_flips():
    lo, hi = wilson_ci(0, 10)
    assert lo == 0.0
    assert hi < 0.4


def test_wilson_ci_half():
    lo, hi = wilson_ci(5, 10)
    assert 0.2 < lo < 0.5
    assert 0.5 < hi < 0.8


def test_wilson_ci_single_trial():
    lo, hi = wilson_ci(1, 1)
    assert lo >= 0.0
    assert hi <= 1.0


# ── StringMatchOracle ────────────────────────────────────────────────────────


def test_oracle_success():
    oracle = StringMatchOracle(success_re=r"SUCCESS", failure_re=r"FAIL")
    assert oracle.grade("the agent said SUCCESS and nothing else") is True


def test_oracle_failure():
    oracle = StringMatchOracle(success_re=r"SUCCESS", failure_re=r"FAIL")
    assert oracle.grade("FAIL — something went wrong") is False


def test_oracle_no_match_returns_none():
    oracle = StringMatchOracle(success_re=r"SUCCESS", failure_re=r"FAIL")
    assert oracle.grade("ambiguous output") is None


# ── BlameEngine ───────────────────────────────────────────────────────────────

SUCCESS_RESP = make_text_response("SUCCESS — booking confirmed")
FAIL_RESP = make_text_response("FAIL — no flights available")
NEUTRAL_RESP = make_text_response("Checking availability")


def _booking_agent(client: anthropic.Anthropic) -> str:
    """Two-turn agent; turn2's history embeds turn1's reply text, so a mutation
    at turn1 changes what turn2 asks (and thus the counterfactual tail)."""
    r1 = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=100,
        messages=[{"role": "user", "content": "book a flight"}],
    )
    first = r1.content[0].text
    r2 = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=100,
        messages=[
            {"role": "user", "content": "book a flight"},
            {"role": "assistant", "content": first},
            {"role": "user", "content": "confirm"},
        ],
    )
    return r2.content[0].text


def _record_booking(resp1: bytes, resp2: bytes) -> Tape:
    fake = ScriptedFakeLLM([resp1, resp2])
    tape = Tape()
    transport = TraceforkTransport("record", tape, fake)
    client = anthropic.Anthropic(
        api_key="sk-ant-fake",
        http_client=httpx.Client(transport=transport),
        max_retries=0,
    )
    _booking_agent(client)
    return tape


def test_blame_engine_ranks_causal_step_highest():
    """The decisive final step (step 1) should have the highest flip-rate."""
    # Parent run: turn1=NEUTRAL, turn2=SUCCESS → outcome SUCCESS.
    tape = _record_booking(NEUTRAL_RESP, SUCCESS_RESP)
    oracle = StringMatchOracle(success_re=r"SUCCESS", failure_re=r"FAIL")

    # Perturb every step with FAIL; the tail (only reached when the perturbed
    # step is NOT final) returns SUCCESS, so only perturbing the final step
    # flips the graded outcome.
    def perturb_factory(step_idx: int) -> tuple[bytes, object]:
        return FAIL_RESP, ScriptedFakeLLM([SUCCESS_RESP])

    report = BlameEngine.rank(
        tape,
        _booking_agent,
        oracle,
        perturb_factory=perturb_factory,
        k=3,
        budget_usd=100.0,
    )

    assert report is not None
    assert report.parent_outcome is True
    assert len(report.results) == 2  # 2 exchanges → 2 candidates
    top = max(report.results, key=lambda r: r.flip_rate)
    assert top.step_index == 1
    assert top.flip_rate == 1.0
    step0 = next(r for r in report.results if r.step_index == 0)
    assert step0.flip_rate == 0.0


def test_blame_engine_returns_wilson_ci():
    tape = _record_booking(NEUTRAL_RESP, SUCCESS_RESP)
    oracle = StringMatchOracle(success_re=r"SUCCESS", failure_re=r"FAIL")

    def perturb_factory(step_idx: int) -> tuple[bytes, object]:
        return FAIL_RESP, ScriptedFakeLLM([SUCCESS_RESP])

    report = BlameEngine.rank(
        tape,
        _booking_agent,
        oracle,
        perturb_factory=perturb_factory,
        k=3,
        budget_usd=100.0,
    )
    for r in report.results:
        assert 0.0 <= r.ci_lo <= r.flip_rate <= r.ci_hi <= 1.0


def test_blame_engine_total_forks_counts_all_trials():
    tape = _record_booking(NEUTRAL_RESP, SUCCESS_RESP)
    oracle = StringMatchOracle(success_re=r"SUCCESS", failure_re=r"FAIL")

    def perturb_factory(step_idx: int) -> tuple[bytes, object]:
        return FAIL_RESP, ScriptedFakeLLM([SUCCESS_RESP])

    report = BlameEngine.rank(
        tape,
        _booking_agent,
        oracle,
        perturb_factory=perturb_factory,
        k=4,
        budget_usd=100.0,
    )
    assert report.total_forks == 2 * 4  # n_candidates * k


def test_budget_governor_estimates():
    tape = _record_booking(NEUTRAL_RESP, SUCCESS_RESP)
    est = BudgetGovernor.estimate(tape, k=10, cost_per_fork_usd=0.01)
    assert est.n_candidates == 2
    assert est.n_forks == 20
    assert abs(est.est_usd - 0.20) < 0.01


# ── proportion CI backends ────────────────────────────────────────────────────

ALL_METHODS = [
    CIMethod.WILSON,
    CIMethod.JEFFREYS,
    CIMethod.CLOPPER_PEARSON,
    CIMethod.AGRESTI_COULL,
]


@pytest.mark.parametrize("method", ALL_METHODS)
def test_ci_zero_flip_boundary(method):
    """0 successes: lower bound pinned at 0, upper strictly inside (0, 1)."""
    lo, hi = proportion_ci(0, 10, method=method)
    assert lo == 0.0
    assert 0.0 < hi < 1.0


@pytest.mark.parametrize("method", ALL_METHODS)
def test_ci_all_flip_boundary(method):
    """n successes: upper bound pinned at 1, lower strictly inside (0, 1)."""
    lo, hi = proportion_ci(10, 10, method=method)
    assert hi == 1.0
    assert 0.0 < lo < 1.0


@pytest.mark.parametrize("method", ALL_METHODS)
def test_ci_brackets_point_estimate(method):
    lo, hi = proportion_ci(5, 10, method=method)
    assert 0.0 <= lo <= 0.5 <= hi <= 1.0


@pytest.mark.parametrize("method", ALL_METHODS)
def test_ci_empty_sample_is_maximally_uncertain(method):
    assert proportion_ci(0, 0, method=method) == (0.0, 1.0)


def test_ci_clopper_pearson_matches_known_values():
    # Textbook exact (95%): 5/10 → (0.1871, 0.8129); 0/10 upper → 0.3085.
    lo, hi = proportion_ci(5, 10, method=CIMethod.CLOPPER_PEARSON)
    assert abs(lo - 0.187086) < 1e-4
    assert abs(hi - 0.812914) < 1e-4
    _, hi0 = proportion_ci(0, 10, method=CIMethod.CLOPPER_PEARSON)
    assert abs(hi0 - 0.308497) < 1e-4


def test_ci_confidence_level_widens_interval():
    lo95, hi95 = proportion_ci(5, 20, method=CIMethod.WILSON, confidence=0.95)
    lo99, hi99 = proportion_ci(5, 20, method=CIMethod.WILSON, confidence=0.99)
    assert lo99 < lo95 and hi99 > hi95


def test_ci_explicit_z_override():
    # z=1.96 reproduces the legacy wilson_ci default exactly.
    assert proportion_ci(3, 10, method=CIMethod.WILSON, z=1.96) == wilson_ci(3, 10, 1.96)


def test_z_from_confidence():
    assert abs(z_from_confidence(0.95) - 1.959964) < 1e-4
    with pytest.raises(ValueError):
        z_from_confidence(1.5)


def test_proportion_ci_rejects_out_of_range():
    with pytest.raises(ValueError):
        proportion_ci(11, 10, method=CIMethod.WILSON)


# ── binomial tail + Benjamini-Hochberg ────────────────────────────────────────


def test_binom_sf_ge_boundaries():
    assert binom_sf_ge(0, 5, 0.05) == 1.0  # P(X>=0) == 1
    assert binom_sf_ge(6, 5, 0.05) == 0.0  # impossible
    assert abs(binom_sf_ge(5, 5, 0.05) - 0.05**5) < 1e-12
    assert abs(binom_sf_ge(3, 5, 0.5) - 0.5) < 1e-12  # symmetric


def test_bh_selects_only_significant_step():
    # A decisive step (tiny p) among inert steps (p=1) is the responsible set.
    selected, qvals = benjamini_hochberg([3e-7, 1.0, 1.0, 1.0], q=0.10)
    assert selected == {0}
    assert qvals[0] <= 0.10
    assert all(qvals[i] > 0.10 for i in (1, 2, 3))


def test_bh_empty_and_all_null():
    assert benjamini_hochberg([], 0.1) == (set(), [])
    selected, _ = benjamini_hochberg([1.0, 1.0, 1.0], 0.1)
    assert selected == set()


def test_bh_qvalues_are_monotone_in_pvalue_order():
    pvals = [0.001, 0.002, 0.5, 0.9]
    _, qvals = benjamini_hochberg(pvals, 0.05)
    ordered = [qvals[i] for i in sorted(range(len(pvals)), key=lambda i: pvals[i])]
    assert ordered == sorted(ordered)


# ── three-valued (UNDEFINED) accounting ───────────────────────────────────────


class _RaisingTransport(httpx.BaseTransport):
    """A tail transport that errors on use — turns a fork into an errored trial."""

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        raise RuntimeError("tail transport unavailable")


def test_undefined_ambiguous_grade_not_counted_as_no_flip():
    """A tail whose outcome the oracle cannot grade is UNDEFINED, not a non-flip:
    it leaves the flip-rate denominator (valid_trials), and the divergence/UNDEFINED
    rate is surfaced as a trust flag."""
    tape = _record_booking(NEUTRAL_RESP, SUCCESS_RESP)
    oracle = StringMatchOracle(success_re=r"SUCCESS", failure_re=r"FAIL")

    # Perturb the FINAL step with an ungradeable response every trial.
    def perturb_factory(step_idx: int) -> tuple[bytes, object]:
        if step_idx == 1:
            return NEUTRAL_RESP, ScriptedFakeLLM([SUCCESS_RESP])
        return SUCCESS_RESP, ScriptedFakeLLM([SUCCESS_RESP])

    report = BlameEngine.rank(
        tape, _booking_agent, oracle, perturb_factory=perturb_factory, k=4, budget_usd=100.0
    )
    step1 = next(r for r in report.results if r.step_index == 1)
    assert step1.valid_trials == 0
    assert step1.undefined == 4
    assert step1.flip_rate == 0.0  # 0 / 0 → 0, NOT counted as 4 non-flips
    assert step1.divergence_rate == 1.0
    assert step1.trustworthy is False
    assert report.total_forks == 2 * 4  # every attempt still counted


def test_flip_rate_is_over_valid_trials_only():
    """Mixed FLIP + UNDEFINED trials: flip-rate uses valid trials as denominator."""
    tape = _record_booking(NEUTRAL_RESP, SUCCESS_RESP)
    oracle = StringMatchOracle(success_re=r"SUCCESS", failure_re=r"FAIL")

    calls = {"n": 0}

    def perturb_factory(step_idx: int) -> tuple[bytes, object]:
        if step_idx == 1:
            i = calls["n"]
            calls["n"] += 1
            # even trials flip (FAIL), odd trials ungradeable (UNDEFINED)
            resp = FAIL_RESP if i % 2 == 0 else NEUTRAL_RESP
            return resp, ScriptedFakeLLM([SUCCESS_RESP])
        return SUCCESS_RESP, ScriptedFakeLLM([SUCCESS_RESP])

    report = BlameEngine.rank(
        tape, _booking_agent, oracle, perturb_factory=perturb_factory, k=4, budget_usd=100.0
    )
    step1 = next(r for r in report.results if r.step_index == 1)
    assert step1.flips == 2
    assert step1.valid_trials == 2
    assert step1.undefined == 2
    assert step1.flip_rate == 1.0  # 2/2, not 2/4


def test_errored_fork_is_undefined_and_counted():
    """A fork whose tail transport raises is recorded as UNDEFINED (not swallowed
    into a non-flip); the trial is still counted in total_forks."""
    tape = _record_booking(NEUTRAL_RESP, SUCCESS_RESP)
    oracle = StringMatchOracle(success_re=r"SUCCESS", failure_re=r"FAIL")

    # Perturb the NON-final step so the agent must make a tail call — which errors.
    def perturb_factory(step_idx: int) -> tuple[bytes, object]:
        if step_idx == 0:
            return FAIL_RESP, _RaisingTransport()
        return SUCCESS_RESP, ScriptedFakeLLM([SUCCESS_RESP])

    report = BlameEngine.rank(
        tape, _booking_agent, oracle, perturb_factory=perturb_factory, k=3, budget_usd=100.0
    )
    step0 = next(r for r in report.results if r.step_index == 0)
    assert step0.undefined == 3
    assert step0.valid_trials == 0
    assert report.total_forks == 2 * 3


# ── FDR responsible set at the engine level ────────────────────────────────────


def test_responsible_set_fingers_causal_step():
    tape = _record_booking(NEUTRAL_RESP, SUCCESS_RESP)
    oracle = StringMatchOracle(success_re=r"SUCCESS", failure_re=r"FAIL")

    def perturb_factory(step_idx: int) -> tuple[bytes, object]:
        return FAIL_RESP, ScriptedFakeLLM([SUCCESS_RESP])

    report = BlameEngine.rank(
        tape, _booking_agent, oracle, perturb_factory=perturb_factory, k=5, budget_usd=100.0
    )
    assert report.responsible_set == [1]
    step1 = next(r for r in report.results if r.step_index == 1)
    step0 = next(r for r in report.results if r.step_index == 0)
    assert step1.responsible is True and step1.q_value <= report.fdr_q
    assert step0.responsible is False
    assert [r.step_index for r in report.responsible()] == [1]
    # top() stays a back-compat argmax accessor.
    assert report.top().step_index == 1
