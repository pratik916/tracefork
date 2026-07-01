"""Replay + DriftDoctor tests — all offline, no API keys."""

import anthropic
import httpx

from tests.fakes import ScriptedFakeLLM, make_text_response, make_tool_use_response
from tracefork.replay import DriftCause, DriftDoctor, ReplayVerifier
from tracefork.tape import Tape
from tracefork.transport import TraceforkTransport

TEXT_RESP = make_text_response("Done.")
TOOL_RESP = make_tool_use_response("book_flight", {"destination": "Tokyo", "seats": 1})


def _record_tape(responses: list[bytes]) -> Tape:
    """Record a tape using ScriptedFakeLLM; return tape."""
    fake = ScriptedFakeLLM(responses)
    tape = Tape()
    transport = TraceforkTransport("record", tape, fake)
    client = anthropic.Anthropic(
        api_key="sk-ant-fake",
        http_client=httpx.Client(transport=transport),
        max_retries=0,
    )
    client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=100,
        messages=[{"role": "user", "content": "Hello"}],
    )
    return tape


def _agent_fn(client: anthropic.Anthropic) -> str:
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=100,
        messages=[{"role": "user", "content": "Hello"}],
    )
    return resp.content[0].text


def test_verifier_passes_on_exact_replay():
    tape = _record_tape([TEXT_RESP])
    result = ReplayVerifier(tape, _agent_fn).verify()
    assert result.bit_exact is True
    assert result.matched == 1
    assert result.total == 1
    assert result.fingerprints_match is True
    assert result.divergence is None


def test_verifier_fails_on_code_change():
    """If the agent builds a different request, replay should diverge."""
    tape = _record_tape([TEXT_RESP])

    def different_agent(client: anthropic.Anthropic) -> str:
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=100,
            messages=[{"role": "user", "content": "Completely different prompt"}],
        )
        return resp.content[0].text

    result = ReplayVerifier(tape, different_agent).verify()
    assert result.bit_exact is False
    assert result.divergence is not None


def test_verifier_matched_count():
    """With two exchanges recorded, both must match for bit_exact=True."""
    fake_rec = ScriptedFakeLLM([TOOL_RESP, TEXT_RESP])
    tape = Tape()
    rec_transport = TraceforkTransport("record", tape, fake_rec)
    client = anthropic.Anthropic(
        api_key="sk-ant-fake",
        http_client=httpx.Client(transport=rec_transport),
        max_retries=0,
    )

    def two_turn_agent(c: anthropic.Anthropic) -> None:
        c.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=100,
            messages=[{"role": "user", "content": "turn1"}],
        )
        c.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=100,
            messages=[
                {"role": "user", "content": "turn1"},
                {"role": "assistant", "content": "..."},
                {"role": "user", "content": "turn2"},
            ],
        )

    two_turn_agent(client)

    result = ReplayVerifier(tape, two_turn_agent).verify()
    assert result.matched == 2
    assert result.bit_exact is True


def test_drift_doctor_classifies_code_change():
    tape = _record_tape([TEXT_RESP])

    def changed_agent(client: anthropic.Anthropic) -> str:
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=100,
            messages=[{"role": "user", "content": "different"}],
        )
        return resp.content[0].text

    result = ReplayVerifier(tape, changed_agent).verify()
    assert result.divergence is not None
    cause = DriftDoctor.classify(result.divergence)
    assert cause == DriftCause.CODE_CHANGE


def test_fingerprints_match_on_exact_replay():
    tape = _record_tape([TEXT_RESP])
    result = ReplayVerifier(tape, _agent_fn).verify()
    assert result.fingerprints_match is True
    assert result.recorded_fingerprint == result.replayed_fingerprint
