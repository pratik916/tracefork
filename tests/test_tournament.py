"""Tournament engine tests — all offline, zero API spend.

`TournamentEngine.run()` compares N candidate continuations at ONE fixed step
(a different axis from `blame.py`'s per-step-across-runs comparison): each
variant is forked `k` times at the same `step_index` with its own scripted
response/tail, graded by an `Oracle`, and ranked by success rate with a
reused Wilson CI and a reused Benjamini-Hochberg significance test of the
top variant against every runner-up.
"""

from __future__ import annotations

import anthropic
import httpx
import pytest

from tests.fakes import ScriptedFakeLLM, make_text_response
from tracefork.blame import BudgetExceededError, StringMatchOracle
from tracefork.tape import Tape
from tracefork.tournament import TournamentEngine, Variant
from tracefork.transport import TraceforkTransport

SUCCESS_RESP = make_text_response("SUCCESS — booking confirmed")
FAIL_RESP = make_text_response("FAIL — no flights available")
NEUTRAL_RESP = make_text_response("Checking availability")


def _final_answer_agent(client: anthropic.Anthropic) -> str:
    """Two-turn agent; the SECOND turn is the fixed step a tournament compares
    candidate continuations at (the tape's last exchange, so a fork at that
    step has an empty tail — no post-fork network call, ever)."""
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


def _record_tape() -> Tape:
    fake = ScriptedFakeLLM([NEUTRAL_RESP, NEUTRAL_RESP])
    tape = Tape()
    transport = TraceforkTransport("record", tape, fake)
    client = anthropic.Anthropic(
        api_key="sk-ant-fake",
        http_client=httpx.Client(transport=transport),
        max_retries=0,
    )
    _final_answer_agent(client)
    return tape


def test_tournament_picks_the_clearly_better_variant_with_excluding_ci():
    """3 variants at the same (last) step, one clearly better -> run() picks
    it with a Wilson CI excluding the runner-up's mean."""
    tape = _record_tape()
    oracle = StringMatchOracle(success_re=r"SUCCESS", failure_re=r"FAIL")

    variants = [
        Variant(name="always-fail", response=FAIL_RESP),
        Variant(name="mostly-fail", response=FAIL_RESP),
        Variant(name="always-success", response=SUCCESS_RESP),
    ]

    report = TournamentEngine.run(
        tape,
        step_index=1,
        variants=variants,
        agent_fn=_final_answer_agent,
        oracle=oracle,
        k=8,
        budget_usd=100.0,
    )

    assert report.step_index == 1
    top = report.results[0]
    runner_up = report.results[1]
    assert top.name == "always-success"
    assert top.score == 1.0
    assert runner_up.score == 0.0
    assert top.ci_lo > runner_up.score
    assert report.winner() is not None
    assert report.winner().name == "always-success"


def test_budget_governor_estimate_raises_before_any_trial_runs():
    """A BudgetGovernor-style estimate raises BudgetExceededError before any
    fork trial when N*k exceeds budget_usd — checked via a flat
    cost_per_fork_usd so the boundary is exactly N*k dollars."""
    tape = _record_tape()
    oracle = StringMatchOracle(success_re=r"SUCCESS", failure_re=r"FAIL")

    calls = 0
    real_response = FAIL_RESP

    class _CountingTransport(httpx.BaseTransport):
        def handle_request(self, request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(200, content=real_response, request=request)

    variants = [
        Variant(name="a", response=FAIL_RESP, tail_transport=_CountingTransport()),
        Variant(name="b", response=SUCCESS_RESP, tail_transport=_CountingTransport()),
    ]
    n_variants, k = len(variants), 5  # N*k = 10 dollars at cost_per_fork_usd=1.0

    with pytest.raises(BudgetExceededError, match="exceeds budget"):
        TournamentEngine.run(
            tape,
            step_index=0,  # tail exists (step 0 of a 2-exchange tape) -> billable
            variants=variants,
            agent_fn=_final_answer_agent,
            oracle=oracle,
            k=k,
            budget_usd=(n_variants * k) - 1.0,
            cost_per_fork_usd=1.0,
        )
    assert calls == 0


def _three_turn_agent(client: anthropic.Anthropic) -> str:
    """Three-turn agent; forking the MIDDLE step (index 1) leaves a one-call
    tail (index 2) whose response is what gets graded."""
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
            {"role": "user", "content": "any preference?"},
        ],
    )
    second = r2.content[0].text
    r3 = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=100,
        messages=[
            {"role": "user", "content": "book a flight"},
            {"role": "assistant", "content": first},
            {"role": "user", "content": "any preference?"},
            {"role": "assistant", "content": second},
            {"role": "user", "content": "confirm"},
        ],
    )
    return r3.content[0].text


def _record_three_turn_tape() -> Tape:
    fake = ScriptedFakeLLM([NEUTRAL_RESP, NEUTRAL_RESP, NEUTRAL_RESP])
    tape = Tape()
    transport = TraceforkTransport("record", tape, fake)
    client = anthropic.Anthropic(
        api_key="sk-ant-fake",
        http_client=httpx.Client(transport=transport),
        max_retries=0,
    )
    _three_turn_agent(client)
    return tape


def test_indistinguishable_variants_do_not_get_a_spurious_winner():
    """Two variants whose tails are scripted to the SAME alternating
    success/failure pattern (identical underlying probability) must not have
    a winner declared at the configured fdr_q/confidence."""
    tape = _record_three_turn_tape()
    oracle = StringMatchOracle(success_re=r"SUCCESS", failure_re=r"FAIL")

    k = 6
    pattern = [SUCCESS_RESP, FAIL_RESP, SUCCESS_RESP, FAIL_RESP, SUCCESS_RESP, FAIL_RESP]
    variants = [
        Variant(name="v1", response=NEUTRAL_RESP, tail_transport=ScriptedFakeLLM(list(pattern))),
        Variant(name="v2", response=NEUTRAL_RESP, tail_transport=ScriptedFakeLLM(list(pattern))),
    ]

    report = TournamentEngine.run(
        tape,
        step_index=1,
        variants=variants,
        agent_fn=_three_turn_agent,
        oracle=oracle,
        k=k,
        budget_usd=100.0,
        fdr_q=0.10,
    )

    assert report.winner() is None
    assert report.results[0].score == report.results[1].score == 0.5


def test_tournament_cli_command_against_fixture_store_exits_zero(tmp_path):
    """New `tournament` CLI command against a fixture store, real exit code 0."""
    from typer.testing import CliRunner

    from tracefork.cli import app
    from tracefork.store import TapeStore

    db_path = tmp_path / "store.db"
    db = TapeStore(str(db_path))
    tape = _record_tape()
    run_id = db.save_tape(tape, run_id="tourney-run")

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "tournament",
            run_id,
            "--agent",
            "tests.test_tournament:_final_answer_agent",
            "--store",
            str(db_path),
            "--step",
            "1",
            "--candidate",
            "success:SUCCESS all done",
            "--candidate",
            "fail:FAIL nothing worked",
            "--success-re",
            "SUCCESS",
            "--failure-re",
            "FAIL",
            "--k",
            "3",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "success" in result.output
