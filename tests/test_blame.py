"""Blame engine tests — all offline, zero API spend."""
import anthropic
import httpx

from tracefork.tape import Tape
from tracefork.transport import TraceforkTransport
from tracefork.blame import (
    BlameEngine, StringMatchOracle, FlipRateResult,
    BudgetGovernor, BlameEstimate, wilson_ci,
)
from tests.fakes import ScriptedFakeLLM, make_text_response


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
FAIL_RESP    = make_text_response("FAIL — no flights available")
NEUTRAL_RESP = make_text_response("Checking availability")


def _booking_agent(client: anthropic.Anthropic) -> str:
    """Two-turn agent; turn2's history embeds turn1's reply text, so a mutation
    at turn1 changes what turn2 asks (and thus the counterfactual tail)."""
    r1 = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=100,
        messages=[{"role": "user", "content": "book a flight"}],
    )
    first = r1.content[0].text
    r2 = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=100,
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
        tape, _booking_agent, oracle,
        perturb_factory=perturb_factory, k=3, budget_usd=100.0,
    )

    assert report is not None
    assert report.parent_outcome is True
    assert len(report.results) == 2          # 2 exchanges → 2 candidates
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
        tape, _booking_agent, oracle,
        perturb_factory=perturb_factory, k=3, budget_usd=100.0,
    )
    for r in report.results:
        assert 0.0 <= r.ci_lo <= r.flip_rate <= r.ci_hi <= 1.0


def test_blame_engine_total_forks_counts_all_trials():
    tape = _record_booking(NEUTRAL_RESP, SUCCESS_RESP)
    oracle = StringMatchOracle(success_re=r"SUCCESS", failure_re=r"FAIL")

    def perturb_factory(step_idx: int) -> tuple[bytes, object]:
        return FAIL_RESP, ScriptedFakeLLM([SUCCESS_RESP])

    report = BlameEngine.rank(
        tape, _booking_agent, oracle,
        perturb_factory=perturb_factory, k=4, budget_usd=100.0,
    )
    assert report.total_forks == 2 * 4       # n_candidates * k


def test_budget_governor_estimates():
    tape = _record_booking(NEUTRAL_RESP, SUCCESS_RESP)
    est = BudgetGovernor.estimate(tape, k=10, cost_per_fork_usd=0.01)
    assert est.n_candidates == 2
    assert est.n_forks == 20
    assert abs(est.est_usd - 0.20) < 0.01
