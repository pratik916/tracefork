"""Tests for `BudgetGovernor.confinement_risk`, the `confinement_risk` field on
`BlameReport`/`ShapleyReport`, and the `confinement=` passthrough from
`BlameEngine.rank()`/`shapley_rank()` into `ForkEngine.fork()`/`fork_coalition()`.
All offline, zero API spend.
"""

import anthropic
import httpx

from tests.fakes import ScriptedFakeLLM, make_text_response
from tracefork.blame import BlameEngine, BudgetGovernor, ConfinementRisk, StringMatchOracle
from tracefork.boundary_guard import ConfinementSpec
from tracefork.tape import Tape
from tracefork.transport import TraceforkTransport

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


# ── BudgetGovernor.confinement_risk ─────────────────────────────────────────


def test_confinement_risk_unconfined_matches_estimate_n_forks():
    """With no `confinement=`, `confined` is False and `projected_trials`
    equals `estimate(...).n_forks` exactly; the note mentions UNCONFINED."""
    tape = _record_booking(NEUTRAL_RESP, SUCCESS_RESP)

    risk = BudgetGovernor.confinement_risk(tape, k=4)
    est = BudgetGovernor.estimate(tape, k=4)

    assert risk.confined is False
    assert risk.projected_trials == est.n_forks
    assert "UNCONFINED" in risk.note or "unconfined" in risk.note.lower()


def test_confinement_risk_confined_true_when_spec_passed():
    """Passing a `ConfinementSpec` sets `confined=True`."""
    tape = _record_booking(NEUTRAL_RESP, SUCCESS_RESP)

    risk = BudgetGovernor.confinement_risk(tape, k=4, confinement=ConfinementSpec())

    assert risk.confined is True
    assert risk.projected_trials == BudgetGovernor.estimate(tape, k=4).n_forks


def test_confinement_risk_never_raises_regardless_of_projected_cost():
    """Pure disclosure: an absurdly high k (huge projected_trials/cost) still
    never raises — only `estimate()`'s own cost gate (in `rank()`) can."""
    tape = _record_booking(NEUTRAL_RESP, SUCCESS_RESP)

    risk = BudgetGovernor.confinement_risk(tape, k=100_000)

    assert risk.projected_trials == 100_000 * len(tape.exchanges)
    assert isinstance(risk, ConfinementRisk)


# ── BlameEngine.rank() / shapley_rank() attach confinement_risk ────────────


def test_rank_without_confinement_attaches_unconfined_risk_matching_total_forks():
    """Without `confinement=`, `rank()`'s report carries a `confinement_risk`
    with `confined=False` and `projected_trials == report.total_forks` exactly
    (no coalition multiplier) — and this never raises purely from disclosure;
    only the pre-existing cost gate (budget_usd) can raise, and it's set high
    here so it doesn't."""
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
        budget_usd=1_000_000.0,
    )

    assert report.confinement_risk is not None
    assert report.confinement_risk.confined is False
    assert report.confinement_risk.projected_trials == report.total_forks


def test_shapley_rank_attaches_non_none_confinement_risk():
    """`shapley_rank()` also attaches a non-None `confinement_risk`."""
    tape = _record_booking(NEUTRAL_RESP, SUCCESS_RESP)
    oracle = StringMatchOracle(success_re=r"SUCCESS", failure_re=r"FAIL")

    def perturb_factory(step_idx: int) -> tuple[bytes, object]:
        return FAIL_RESP, ScriptedFakeLLM([SUCCESS_RESP])

    report = BlameEngine.shapley_rank(
        tape,
        _booking_agent,
        oracle,
        perturb_factory=perturb_factory,
        k=2,
        m_samples=2,
        budget_usd=1_000_000.0,
    )

    assert report.confinement_risk is not None
    assert report.confinement_risk.confined is False


# ── confinement= passthrough actually reaches ForkEngine.fork() ────────────


def test_rank_confinement_blocks_every_trial_write(tmp_path):
    """Passing `confinement=ConfinementSpec()` (empty writable_roots/
    allowed_hosts) into `rank()` with an agent that performs a disallowed
    file write blocks EVERY trial's write (the `ConfinementViolationError` is
    caught by `_run_trial`'s broad `except Exception` -> counted UNDEFINED,
    so `undefined == trials` for every step), and the target file is never
    created — proving the `confinement=` passthrough reaches
    `ForkEngine.fork()` for real, not just cosmetically."""
    leak_file = tmp_path / "leak.txt"

    def _write_attempting_agent(client: anthropic.Anthropic) -> str:
        r1 = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=100,
            messages=[{"role": "user", "content": "book a flight"}],
        )
        first = r1.content[0].text
        # Attempted side effect inside the re-executed agent's own window —
        # ConfinementSpec() declares NO writable roots, so this must always
        # be rejected while confinement is active, regardless of which step
        # is being forked.
        with open(leak_file, "w") as f:
            f.write("leak")
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

    tape = _record_booking(NEUTRAL_RESP, SUCCESS_RESP)
    oracle = StringMatchOracle(success_re=r"SUCCESS", failure_re=r"FAIL")

    def perturb_factory(step_idx: int) -> tuple[bytes, object]:
        return FAIL_RESP, ScriptedFakeLLM([])

    report = BlameEngine.rank(
        tape,
        _write_attempting_agent,
        oracle,
        perturb_factory=perturb_factory,
        k=3,
        budget_usd=1_000_000.0,
        confinement=ConfinementSpec(),
    )

    assert len(report.results) == 2  # 2 exchanges → 2 candidate steps
    for r in report.results:
        assert r.undefined == r.trials
        assert r.valid_trials == 0
    assert not leak_file.exists()
    assert report.confinement_risk is not None
    assert report.confinement_risk.confined is True
