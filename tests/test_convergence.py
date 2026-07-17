"""Reconvergence detection tests — offline, no API keys.

`convergence.py` is a pure fingerprint-comparison layer over `fork.py`'s
existing `compute_divergence_exchange_digest` primitive; these tests exercise
`find_reconvergence`'s hand-built-tape contract (matched/reconverged/stable
semantics, the same-divergence-step requirement, and the shorter-tape
truncation) plus one end-to-end test through the real `ForkEngine.fork()`
path.
"""

import anthropic
import httpx
import pytest

from tests.fakes import ScriptedFakeLLM, make_text_response
from tracefork.convergence import ConvergenceResult, StepFingerprint, find_reconvergence
from tracefork.fork import BranchSpec, ForkEngine
from tracefork.tape import Tape
from tracefork.transport import TraceforkTransport


def _tape_with_exchanges(pairs: list[tuple[bytes, bytes]]) -> Tape:
    tape = Tape()
    for req, resp in pairs:
        tape.append_exchange(req, resp)
    return tape


# ── hand-built delta_tape cases ─────────────────────────────────────────────


def test_find_reconvergence_matches_from_first_convergent_step_onward():
    """Differ at offset 0, byte-identical at every offset after -> stable
    reconvergence starting one step past the divergence step."""
    divergence_step = 5
    delta_a = _tape_with_exchanges(
        [
            (b"req0", b"resp0-a"),
            (b"req1", b"resp1"),
            (b"req2", b"resp2"),
        ]
    )
    delta_b = _tape_with_exchanges(
        [
            (b"req0", b"resp0-b"),
            (b"req1", b"resp1"),
            (b"req2", b"resp2"),
        ]
    )

    result = find_reconvergence(delta_a, divergence_step, delta_b, divergence_step)

    assert isinstance(result, ConvergenceResult)
    assert all(isinstance(s, StepFingerprint) for s in result.steps)
    assert result.reconverged is True
    assert result.first_convergent_step == divergence_step + 1
    assert result.stable is True
    assert result.matched_steps == (divergence_step + 1, divergence_step + 2)


def test_find_reconvergence_no_match_anywhere():
    """Every offset differs -> no reconvergence at all."""
    delta_a = _tape_with_exchanges([(b"req0", b"resp0-a"), (b"req1", b"resp1-a")])
    delta_b = _tape_with_exchanges([(b"req0", b"resp0-b"), (b"req1", b"resp1-b")])

    result = find_reconvergence(delta_a, 2, delta_b, 2)

    assert result.reconverged is False
    assert result.first_convergent_step is None
    assert result.stable is False
    assert result.matched_steps == ()


def test_find_reconvergence_coincidental_match_is_not_stable():
    """A match at exactly one offset, followed by a diverging offset, proves
    `reconverged` vs. `stable` are genuinely different signals."""
    delta_a = _tape_with_exchanges(
        [
            (b"req0", b"resp0-a"),
            (b"req1", b"resp1"),  # coincidental match
            (b"req2", b"resp2-a"),  # reverts to diverging
        ]
    )
    delta_b = _tape_with_exchanges(
        [
            (b"req0", b"resp0-b"),
            (b"req1", b"resp1"),
            (b"req2", b"resp2-b"),
        ]
    )

    result = find_reconvergence(delta_a, 0, delta_b, 0)

    assert result.reconverged is True
    assert result.first_convergent_step == 1
    assert result.stable is False
    assert result.matched_steps == (1,)


def test_find_reconvergence_requires_same_divergence_step():
    """Comparing two branches forked at different divergence steps has no
    well-defined shared step-index alignment -> ValueError."""
    delta_a = _tape_with_exchanges([(b"req0", b"resp0")])
    delta_b = _tape_with_exchanges([(b"req0", b"resp0")])

    with pytest.raises(ValueError):
        find_reconvergence(delta_a, 0, delta_b, 1)


def test_find_reconvergence_truncates_to_shorter_tape():
    """A delta_tape shorter than the other silently truncates the comparison
    to the shorter tail -- no IndexError."""
    delta_a = _tape_with_exchanges(
        [
            (b"req0", b"resp0"),
            (b"req1", b"resp1"),
            (b"req2", b"resp2"),
        ]
    )
    delta_b = _tape_with_exchanges([(b"req0", b"resp0")])

    result = find_reconvergence(delta_a, 0, delta_b, 0)

    assert len(result.steps) == 1
    assert result.stable is True


# ── ForkEngine integration ───────────────────────────────────────────────────

RESP_A = make_text_response("Response A")
RESP_B = make_text_response("Response B — mutated")
RESP_D = make_text_response("Response D — differently mutated")
RESP_E = make_text_response("Response E — shared tail")


def _independent_turns_agent(client: anthropic.Anthropic) -> str:
    """Two-turn agent whose turn2 request does NOT embed turn1's response
    text (unlike test_fork.py's `_conversation_agent`) -- so mutating turn1's
    response changes only the mutation exchange itself, never the downstream
    request, letting two differently-mutated siblings still reconverge once a
    shared tail response is served."""
    client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=100,
        messages=[{"role": "user", "content": "turn1"}],
    )
    r2 = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=100,
        messages=[{"role": "user", "content": "turn2"}],
    )
    return r2.content[0].text


def _build_two_turn_tape() -> Tape:
    """Parent run: turn1 -> RESP_A, turn2 -> RESP_E (2 exchanges)."""
    fake = ScriptedFakeLLM([RESP_A, RESP_E])
    tape = Tape()
    transport = TraceforkTransport("record", tape, fake)
    client = anthropic.Anthropic(
        api_key="sk-ant-fake",
        http_client=httpx.Client(transport=transport),
        max_retries=0,
    )
    _independent_turns_agent(client)
    return tape


def test_find_reconvergence_on_real_forks_from_forkengine():
    """Fork the SAME parent tape twice at the SAME divergence step with two
    DIFFERENT mutated responses; since the agent's downstream request doesn't
    embed the mutated text, both tails reconverge on the same served
    response -- proving the real fork path (not just hand-built tapes)
    produces a provable reconvergence."""
    parent_tape = _build_two_turn_tape()

    branch_a = ForkEngine.fork(
        parent_tape,
        BranchSpec(divergence_step=0, mutated_response=RESP_B),
        _independent_turns_agent,
        post_fork_transport=ScriptedFakeLLM([RESP_E]),
    )
    branch_b = ForkEngine.fork(
        parent_tape,
        BranchSpec(divergence_step=0, mutated_response=RESP_D),
        _independent_turns_agent,
        post_fork_transport=ScriptedFakeLLM([RESP_E]),
    )

    result = find_reconvergence(
        branch_a.delta_tape,
        branch_a.divergence_step,
        branch_b.delta_tape,
        branch_b.divergence_step,
    )

    assert result.stable is True
