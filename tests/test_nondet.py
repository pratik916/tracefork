"""nondet.py — direct class tests for the random channel, plus an end-to-end
record/replay/drift receipt through TraceforkTransport (mirrors
tests/test_spike0.py's pattern, for the production package)."""

from __future__ import annotations

import random

import anthropic
import httpx
import pytest

from tests.fakes import ScriptedFakeLLM, make_text_response
from tracefork.nondet import (
    DivergenceError,
    DriftingNondet,
    RecordingNondet,
    ReplayNondet,
    find_divergence,
)
from tracefork.tape import Tape
from tracefork.transport import TraceforkTransport

TEXT_RESP = make_text_response("Done.")


def _toy_agent(client: anthropic.Anthropic, nondet) -> str:
    """Embeds a random draw (as its exact hex representation) into the request
    so a divergence in the draw becomes a divergence in the request body."""
    v = nondet.random_float()
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=100,
        messages=[{"role": "user", "content": f"roll: {v.hex()}"}],
    )
    return resp.content[0].text


def _client(transport: httpx.BaseTransport) -> anthropic.Anthropic:
    return anthropic.Anthropic(
        api_key="sk-ant-fake",
        http_client=httpx.Client(transport=transport),
        max_retries=0,
    )


# ── Direct class tests ──────────────────────────────────────────────────────


def test_recording_nondet_random_float_is_logged_exactly():
    random.seed(20260702)
    nd = RecordingNondet()
    v = nd.random_float()
    assert isinstance(v, float)
    assert nd.draws == [("random", v.hex())]


def test_random_float_record_replay_round_trip_is_exact():
    """No float dust: replay must return the *exact* recorded float, not an
    approximately-equal one — float.hex() round-trips losslessly."""
    random.seed(20260702)
    nd = RecordingNondet()
    v1 = nd.random_float()
    v2 = nd.random_float()

    replay = ReplayNondet(nd.draws)
    r1 = replay.random_float()
    r2 = replay.random_float()

    assert r1 == v1
    assert r2 == v2
    # bit-for-bit, not just numerically close
    assert r1.hex() == v1.hex()
    assert r2.hex() == v2.hex()
    assert replay.fully_consumed()


def test_interleaved_clock_uuid_random_round_trip_in_order():
    """The three draw kinds share one ordered log; replay must serve each
    kind back in the order it was recorded, regardless of interleaving."""
    random.seed(1)
    nd = RecordingNondet()
    clock1 = nd.now_iso()
    rand1 = nd.random_float()
    uuid1 = nd.new_uuid_hex()
    rand2 = nd.random_float()

    assert [k for k, _ in nd.draws] == ["clock", "random", "uuid", "random"]

    replay = ReplayNondet(nd.draws)
    assert replay.now_iso() == clock1
    assert replay.random_float() == rand1
    assert replay.new_uuid_hex() == uuid1
    assert replay.random_float() == rand2
    assert replay.fully_consumed()


def test_replay_random_float_rejects_kind_mismatch():
    replay = ReplayNondet([("uuid", "deadbeef")])
    with pytest.raises(DivergenceError, match="random"):
        replay.random_float()


def test_replay_random_float_exhausted_tape_raises():
    replay = ReplayNondet([])
    with pytest.raises(DivergenceError, match="exhausted"):
        replay.random_float()


def test_drifting_nondet_random_float_draws_fresh_value():
    """DriftingNondet inherits RecordingNondet's random_float — it must draw a
    genuinely fresh value, not replay a fixed/recorded one."""
    random.seed(2)
    recorded = RecordingNondet().random_float()
    drifted = DriftingNondet().random_float()
    # A fresh 53-bit float draw colliding with the recorded one is
    # astronomically unlikely; equality would mean drift isn't actually fresh.
    assert drifted != recorded


# ── End-to-end record → replay → drift receipt (mirrors test_spike0.py) ─────


def test_random_channel_record_replay_end_to_end_bit_exact():
    tape = Tape()
    rec_nondet = RecordingNondet()
    rec_transport = TraceforkTransport("record", tape, ScriptedFakeLLM([TEXT_RESP]))
    _toy_agent(_client(rec_transport), rec_nondet)
    tape.draws = rec_nondet.draws

    rep_transport = TraceforkTransport("replay", tape)
    out = _toy_agent(_client(rep_transport), ReplayNondet(tape.draws))
    assert out == "Done."
    assert rep_transport.fully_consumed()


def test_random_channel_negative_control_drift_is_detected():
    """Replaying with fresh (drifting) random draws must diverge — the same
    negative-control shape the spike uses for clock/uuid, extended to random.

    Going through the real `anthropic.Anthropic` client (unlike the spike's
    fake), the SDK wraps the transport's `DivergenceError` in
    `APIConnectionError` — `find_divergence` recovers it, exactly as
    `replay.py`'s `ReplayVerifier` does.
    """
    tape = Tape()
    rec_nondet = RecordingNondet()
    rec_transport = TraceforkTransport("record", tape, ScriptedFakeLLM([TEXT_RESP]))
    _toy_agent(_client(rec_transport), rec_nondet)
    tape.draws = rec_nondet.draws

    with pytest.raises(anthropic.APIConnectionError) as exc_info:
        _toy_agent(_client(TraceforkTransport("replay", tape)), DriftingNondet())

    divergence = find_divergence(exc_info.value)
    assert divergence is not None
    assert isinstance(divergence, DivergenceError)
