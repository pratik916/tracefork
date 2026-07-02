"""Fork engine tests — all offline, no API keys.

The fork model re-runs the *same* agent that produced the parent tape. The
fake agent's second request depends on the first response's text, so mutating
an early step changes the downstream request bytes — exactly the counterfactual
behaviour a fork must capture.
"""

import anthropic
import httpx

from tests.fakes import ScriptedFakeLLM, make_text_response
from tracefork.fork import BranchSpec, CoalitionSpec, ForkEngine, StepIntervention
from tracefork.nondet import find_divergence
from tracefork.tape import Tape
from tracefork.transport import TraceforkTransport

RESP_A = make_text_response("Response A")
RESP_B = make_text_response("Response B — mutated")
RESP_C = make_text_response("Response C — final turn")
RESP_D = make_text_response("Response D — mutated turn2")
RESP_E = make_text_response("Response E — final turn3")


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


def test_fork_at_last_step_delta_is_mutation_only():
    """Fork at the final step: agent stops, so delta holds only the mutation."""
    parent_tape = _build_two_turn_tape()
    assert len(parent_tape.exchanges) == 2

    spec = BranchSpec(divergence_step=1, mutated_response=RESP_B)
    fake_post = ScriptedFakeLLM([])  # tail never reached

    branch = ForkEngine.fork(parent_tape, spec, _conversation_agent, post_fork_transport=fake_post)

    assert branch.divergence_step == 1
    assert len(branch.delta_tape.exchanges) == 1  # just the mutation
    assert branch.delta_tape.exchanges[0][1] == RESP_B  # mutated response stored
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
        ForkEngine.fork(
            parent_tape, spec, _conversation_agent, post_fork_transport=ScriptedFakeLLM([])
        )
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "out of range" in str(e)


def test_branch_spec_mutation_desc():
    spec = BranchSpec(divergence_step=1, mutated_response=RESP_B, mutation_desc="flip seats to 0")
    assert spec.mutation_desc == "flip seats to 0"


# ── coalition forks (joint, multi-step interventions) ───────────────────────


def _three_turn_agent(client: anthropic.Anthropic) -> str:
    """Three-turn agent; each turn's history embeds the previous reply text."""
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
    second = r2.content[0].text
    r3 = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=100,
        messages=[
            {"role": "user", "content": "turn1"},
            {"role": "assistant", "content": first},
            {"role": "user", "content": "turn2"},
            {"role": "assistant", "content": second},
            {"role": "user", "content": "turn3"},
        ],
    )
    return r3.content[0].text


def _build_three_turn_tape() -> Tape:
    """Parent run: turn1 → RESP_A, turn2 → RESP_C, turn3 → RESP_E (3 exchanges)."""
    fake = ScriptedFakeLLM([RESP_A, RESP_C, RESP_E])
    tape = Tape()
    transport = TraceforkTransport("record", tape, fake)
    client = anthropic.Anthropic(
        api_key="sk-ant-fake",
        http_client=httpx.Client(transport=transport),
        max_retries=0,
    )
    _three_turn_agent(client)
    return tape


def test_coalition_spec_sorts_and_validates():
    spec = CoalitionSpec(
        interventions=(StepIntervention(2, RESP_C), StepIntervention(0, RESP_A)),
    )
    assert spec.steps == (0, 2)
    assert spec.first_step == 0


def test_coalition_spec_rejects_empty():
    try:
        CoalitionSpec(interventions=())
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "at least one" in str(e)


def test_coalition_spec_rejects_duplicate_steps():
    try:
        CoalitionSpec(interventions=(StepIntervention(0, RESP_A), StepIntervention(0, RESP_B)))
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "duplicate" in str(e)


def test_coalition_spec_single_matches_branch_spec_semantics():
    spec = CoalitionSpec.single(1, RESP_B, mutation_desc="d")
    assert spec.steps == (1,)
    assert spec.first_step == 1
    assert spec.mutation_desc == "d"


def test_fork_coalition_single_step_matches_classic_fork():
    """A one-element coalition should behave exactly like `ForkEngine.fork`."""
    parent_tape = _build_two_turn_tape()
    classic = ForkEngine.fork(
        parent_tape,
        BranchSpec(divergence_step=0, mutated_response=RESP_B),
        _conversation_agent,
        post_fork_transport=ScriptedFakeLLM([RESP_C]),
    )
    coalition = ForkEngine.fork_coalition(
        parent_tape,
        CoalitionSpec.single(0, RESP_B),
        _conversation_agent,
        post_fork_transport=ScriptedFakeLLM([RESP_C]),
    )
    assert coalition.divergence_step == classic.divergence_step
    assert coalition.prefix_replayed == classic.prefix_replayed
    assert coalition.tail_recorded == classic.tail_recorded
    assert coalition.intervened_steps == (0,)
    coalition_resps = [e[1] for e in coalition.delta_tape.exchanges]
    classic_resps = [e[1] for e in classic.delta_tape.exchanges]
    assert coalition_resps == classic_resps


def test_fork_coalition_forces_two_steps_jointly():
    """Coalition {0, 1}: both forced; only step2 is a genuine recorded tail."""
    parent_tape = _build_three_turn_tape()
    assert len(parent_tape.exchanges) == 3

    spec = CoalitionSpec(
        interventions=(StepIntervention(0, RESP_B), StepIntervention(1, RESP_D)),
    )
    fake_post = ScriptedFakeLLM([RESP_E])  # only the tail request (turn3) reaches it

    branch = ForkEngine.fork_coalition(
        parent_tape, spec, _three_turn_agent, post_fork_transport=fake_post
    )

    assert branch.divergence_step == 0
    assert branch.intervened_steps == (0, 1)
    assert branch.prefix_replayed == 0
    assert branch.tail_recorded == 1
    # delta: two forced exchanges + one recorded tail
    assert len(branch.delta_tape.exchanges) == 3
    assert branch.delta_tape.exchanges[0][1] == RESP_B
    assert branch.delta_tape.exchanges[1][1] == RESP_D
    # the tail request embeds BOTH mutated replies, not the parent's originals
    tail_req = branch.delta_tape.exchanges[2][0]
    assert b"Response B" in tail_req
    assert b"Response D" in tail_req
    assert len(fake_post.requests_received) == 1


def test_fork_coalition_prefix_replayed_before_first_intervention():
    """Coalition {1, 2} on a 3-turn tape: step0 is replayed from parent for $0."""
    parent_tape = _build_three_turn_tape()
    spec = CoalitionSpec(
        interventions=(StepIntervention(1, RESP_D), StepIntervention(2, RESP_E)),
    )
    fake_post = ScriptedFakeLLM([])  # tail never reached — coalition covers the rest

    branch = ForkEngine.fork_coalition(
        parent_tape, spec, _three_turn_agent, post_fork_transport=fake_post
    )

    assert branch.divergence_step == 1
    assert branch.intervened_steps == (1, 2)
    assert branch.prefix_replayed == 1  # step0 replayed unmodified from parent
    assert branch.tail_recorded == 0
    assert len(branch.delta_tape.exchanges) == 2
    assert branch.delta_tape.exchanges[0][1] == RESP_D
    assert branch.delta_tape.exchanges[1][1] == RESP_E
    assert len(fake_post.requests_received) == 0


def test_fork_coalition_prefix_divergence_raises():
    """A mismatched prefix (agent no longer deterministic before the coalition)
    still raises DivergenceError, same as classic ForkEngine.fork."""
    parent_tape = _build_three_turn_tape()
    spec = CoalitionSpec(interventions=(StepIntervention(1, RESP_D),))

    def _bad_prefix_agent(client: anthropic.Anthropic) -> str:
        client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=100,
            messages=[{"role": "user", "content": "NOT turn1"}],  # diverges immediately
        )
        return ""

    try:
        ForkEngine.fork_coalition(
            parent_tape, spec, _bad_prefix_agent, post_fork_transport=ScriptedFakeLLM([])
        )
        raise AssertionError("expected DivergenceError")
    except Exception as exc:
        # the SDK wraps transport exceptions in APIConnectionError; unwrap it.
        divergence = find_divergence(exc)
        assert divergence is not None


def test_fork_coalition_out_of_range_step_raises():
    parent_tape = _build_two_turn_tape()
    spec = CoalitionSpec(interventions=(StepIntervention(5, RESP_B),))
    try:
        ForkEngine.fork_coalition(
            parent_tape, spec, _conversation_agent, post_fork_transport=ScriptedFakeLLM([])
        )
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "out of range" in str(e)
