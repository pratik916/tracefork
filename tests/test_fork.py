"""Fork engine tests — all offline, no API keys.

The fork model re-runs the *same* agent that produced the parent tape. The
fake agent's second request depends on the first response's text, so mutating
an early step changes the downstream request bytes — exactly the counterfactual
behaviour a fork must capture.
"""
import anthropic
import httpx

from tracefork.tape import Tape
from tracefork.transport import TraceforkTransport
from tracefork.fork import ForkEngine, BranchSpec
from tests.fakes import ScriptedFakeLLM, make_text_response


RESP_A = make_text_response("Response A")
RESP_B = make_text_response("Response B — mutated")
RESP_C = make_text_response("Response C — final turn")


def _conversation_agent(client: anthropic.Anthropic) -> str:
    """Two-turn agent; turn2's history embeds turn1's reply text."""
    r1 = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=100,
        messages=[{"role": "user", "content": "turn1"}],
    )
    first = r1.content[0].text
    r2 = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=100,
        messages=[{"role": "user", "content": "turn1"},
                  {"role": "assistant", "content": first},
                  {"role": "user", "content": "turn2"}],
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


def test_fork_at_last_step_delta_is_mutation_only():
    """Fork at the final step: agent stops, so delta holds only the mutation."""
    parent_tape = _build_two_turn_tape()
    assert len(parent_tape.exchanges) == 2

    spec = BranchSpec(divergence_step=1, mutated_response=RESP_B)
    fake_post = ScriptedFakeLLM([])  # tail never reached

    branch = ForkEngine.fork(parent_tape, spec, _conversation_agent, post_fork_transport=fake_post)

    assert branch.divergence_step == 1
    assert len(branch.delta_tape.exchanges) == 1          # just the mutation
    assert branch.delta_tape.exchanges[0][1] == RESP_B    # mutated response stored
    assert branch.tail_recorded == 0


def test_fork_prefix_replay_is_zero_cost():
    """The prefix is served from the parent tape — the inner transport is untouched."""
    parent_tape = _build_two_turn_tape()
    spec = BranchSpec(divergence_step=1, mutated_response=RESP_B)
    fake_post = ScriptedFakeLLM([])

    branch = ForkEngine.fork(parent_tape, spec, _conversation_agent, post_fork_transport=fake_post)

    # step 0 replayed from parent for $0; the fake (real-API stand-in) saw nothing
    assert branch.prefix_replayed == 1
    assert len(fake_post.requests_received) == 0


def test_fork_at_step_zero_records_counterfactual_tail():
    """Fork at step 0: the mutated response changes turn2's request, recorded fresh."""
    parent_tape = _build_two_turn_tape()
    spec = BranchSpec(divergence_step=0, mutated_response=RESP_B)
    fake_post = ScriptedFakeLLM([RESP_C])  # serves the single counterfactual tail request

    branch = ForkEngine.fork(parent_tape, spec, _conversation_agent, post_fork_transport=fake_post)

    assert branch.divergence_step == 0
    assert branch.prefix_replayed == 0
    assert branch.tail_recorded == 1
    # delta: mutation at step 0 + one recorded tail exchange
    assert len(branch.delta_tape.exchanges) == 2
    assert branch.delta_tape.exchanges[0][1] == RESP_B
    # the tail request embedded the MUTATED reply, not the parent's "Response A"
    assert b"Response B" in branch.delta_tape.exchanges[1][0]
    assert len(fake_post.requests_received) == 1


def test_fork_out_of_range_step_raises():
    parent_tape = _build_two_turn_tape()
    spec = BranchSpec(divergence_step=5, mutated_response=RESP_B)
    try:
        ForkEngine.fork(parent_tape, spec, _conversation_agent, post_fork_transport=ScriptedFakeLLM([]))
        assert False, "expected ValueError"
    except ValueError as e:
        assert "out of range" in str(e)


def test_branch_spec_mutation_desc():
    spec = BranchSpec(divergence_step=1, mutated_response=RESP_B, mutation_desc="flip seats to 0")
    assert spec.mutation_desc == "flip seats to 0"
