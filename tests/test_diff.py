"""Point-to-point / fork-branch diff tests — offline, no API keys.

`diff.py` is a pure sequence-of-steps orchestration layer on top of
`divergence.py`'s existing structural-diff primitive (`diff_json`/
`diff_request_bytes`/`MISSING`) — these tests exercise the two entry points:

* `branch_diff` — a branch's `delta_tape` vs its parent, from the divergence
  step onward (both the live `fork.Branch` form and the store-reloaded
  plain-`Tape` + `divergence_step` form).
* `tape_diff` — two independent tapes compared at one step index, no
  parent/child relationship assumed.
"""

import anthropic
import httpx

from tests.fakes import ScriptedFakeLLM, make_text_response
from tracefork.diff import MISSING, RangeDiff, StepDiff, branch_diff, tape_diff
from tracefork.divergence import FieldDiff
from tracefork.fork import BranchSpec, ForkEngine
from tracefork.tape import Tape
from tracefork.transport import TraceforkTransport

RESP_A = make_text_response("Response A")
RESP_B = make_text_response("Response B — mutated")
RESP_C = make_text_response("Response C — final turn")


def _conversation_agent(client: anthropic.Anthropic) -> str:
    """Two-turn agent; turn2's history embeds turn1's reply text."""
    r1 = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=100,
        messages=[{"role": "user", "content": "turn1"}],
    )
    first = r1.content[0].text
    r2 = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=100,
        messages=[
            {"role": "user", "content": "turn1"},
            {"role": "assistant", "content": first},
            {"role": "user", "content": "turn2"},
        ],
    )
    return r2.content[0].text


def _build_two_turn_tape() -> Tape:
    """Parent run: turn1 → RESP_A, turn2 → RESP_C (2 exchanges)."""
    fake = ScriptedFakeLLM([RESP_A, RESP_C])
    tape = Tape()
    transport = TraceforkTransport("record", tape, fake)
    client = anthropic.Anthropic(
        api_key="sk-ant-fake",
        http_client=httpx.Client(transport=transport),
        max_retries=0,
    )
    _conversation_agent(client)
    return tape


# ── branch_diff ──────────────────────────────────────────────────────────────


def test_branch_diff_reports_no_changes_when_delta_matches_parent_exactly():
    """Forking at the last step and re-serving the SAME recorded response is a
    no-op fork — the delta should read identical from the divergence step on."""
    parent_tape = _build_two_turn_tape()
    spec = BranchSpec(divergence_step=1, mutated_response=RESP_C)  # not actually mutated
    branch = ForkEngine.fork(
        parent_tape, spec, _conversation_agent, post_fork_transport=ScriptedFakeLLM([])
    )

    result = branch_diff(parent_tape, branch)

    assert isinstance(result, RangeDiff)
    assert result.identical is True
    assert result.changed_steps == ()


def test_branch_diff_reports_changed_response_at_mutation_step_and_changed_request_in_tail():
    """Forking step 0 with a different response changes both the mutation
    step's response AND turn2's downstream request (it embeds turn1's reply)."""
    parent_tape = _build_two_turn_tape()
    spec = BranchSpec(divergence_step=0, mutated_response=RESP_B)
    branch = ForkEngine.fork(
        parent_tape, spec, _conversation_agent, post_fork_transport=ScriptedFakeLLM([RESP_C])
    )

    result = branch_diff(parent_tape, branch)

    assert result.identical is False
    assert result.changed_steps == (0, 1)

    step0 = result.steps[0]
    assert step0.step_index == 0
    assert step0.request_diffs == ()  # same request replayed against parent (mutation step)
    assert step0.changed is True  # response differs (RESP_C recorded -> RESP_B mutated)

    step1 = result.steps[1]
    assert step1.step_index == 1
    assert step1.changed is True
    assert any("Response B" in str(d.live) for d in step1.request_diffs)


def test_branch_diff_delta_tape_shorter_than_parent_tail_uses_missing_sentinel():
    """A fork whose agent stops early (or whose delta_tape was truncated)
    leaves parent tail steps unmatched on the branch side — MISSING, not a
    crash."""
    parent_tape = _build_two_turn_tape()  # 2 exchanges: steps 0, 1

    # A delta_tape that only covers step 0 (the mutation), never reaching the
    # parent's step 1 tail at all — simulates an agent that stopped short.
    delta_tape = Tape(boundary=parent_tape.boundary, agent_name=parent_tape.agent_name)
    req0, _ = parent_tape.exchange(0)
    delta_tape.append_exchange(req0, RESP_B)

    result = branch_diff(parent_tape, delta_tape, divergence_step=0)

    assert result.changed_steps == (0, 1)
    step1 = result.steps[1]
    assert step1.step_index == 1
    assert any(d.live == MISSING for d in step1.request_diffs)
    assert any(d.live == MISSING for d in step1.response_diffs)


def test_branch_diff_wraps_a_live_fork_engine_result_directly():
    """`branch_diff` must accept the `Branch` dataclass ForkEngine.fork()
    returns directly — not require the caller to unpack delta_tape/step
    first (that's the store-reloaded path, exercised separately)."""
    parent_tape = _build_two_turn_tape()
    spec = BranchSpec(divergence_step=1, mutated_response=RESP_B)
    branch = ForkEngine.fork(
        parent_tape, spec, _conversation_agent, post_fork_transport=ScriptedFakeLLM([])
    )

    result = branch_diff(parent_tape, branch)  # `branch` is a live fork.Branch, not a Tape

    assert isinstance(result, RangeDiff)
    assert result.steps[0].step_index == 1


def test_branch_diff_from_step_before_divergence_step_raises():
    parent_tape = _build_two_turn_tape()
    spec = BranchSpec(divergence_step=1, mutated_response=RESP_B)
    branch = ForkEngine.fork(
        parent_tape, spec, _conversation_agent, post_fork_transport=ScriptedFakeLLM([])
    )
    try:
        branch_diff(parent_tape, branch, from_step=0)
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "divergence_step" in str(exc)


def test_branch_diff_plain_tape_requires_divergence_step():
    parent_tape = _build_two_turn_tape()
    delta_tape = Tape()
    delta_tape.append_exchange(*parent_tape.exchange(1))
    try:
        branch_diff(parent_tape, delta_tape)
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "divergence_step" in str(exc)


# ── tape_diff ─────────────────────────────────────────────────────────────────


def test_tape_diff_compares_two_independent_tapes_at_a_step():
    """Two independently-recorded tapes (no parent/child relationship) diffed
    at a single step index."""
    tape_a = Tape()
    tape_a.append_exchange(b'{"model":"claude-sonnet-4-6","max_tokens":100}', b'{"text":"hi"}')
    tape_b = Tape()
    tape_b.append_exchange(b'{"model":"claude-sonnet-4-6","max_tokens":200}', b'{"text":"hi"}')

    result = tape_diff(tape_a, tape_b, 0)

    assert isinstance(result, StepDiff)
    assert result.changed is True
    assert result.request_diffs == (FieldDiff("$.max_tokens", 100, 200),)
    assert result.response_diffs == ()


def test_tape_diff_identical_tapes_are_empty():
    tape_a = Tape()
    tape_a.append_exchange(b'{"a":1}', b'{"b":2}')
    tape_b = Tape()
    tape_b.append_exchange(b'{"a":1}', b'{"b":2}')

    result = tape_diff(tape_a, tape_b, 0)
    assert result.changed is False


def test_tape_diff_out_of_range_step_on_one_side_uses_missing_sentinel():
    tape_a = Tape()
    tape_a.append_exchange(b'{"a":1}', b'{"b":2}')
    tape_b = Tape()  # no exchanges at all

    result = tape_diff(tape_a, tape_b, 0)

    assert result.changed is True
    assert any(d.live == MISSING for d in result.request_diffs)
    assert any(d.live == MISSING for d in result.response_diffs)
