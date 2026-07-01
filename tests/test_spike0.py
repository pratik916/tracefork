"""Spike 0 receipt, as offline pytest assertions ($0, no key, no network).

This is the R1 receipt in miniature: bit-exact replay within the declared
determinism boundary, plus a negative control proving drift is detected.
"""

from __future__ import annotations

import json

import httpx
import pytest

from tracefork_spike.agent import make_client, run_agent
from tracefork_spike.fake_llm import FakeAnthropicTransport
from tracefork_spike.nondet import DivergenceError, DriftingNondet, RecordingNondet, ReplayNondet
from tracefork_spike.spike import record_replay_verify
from tracefork_spike.tape import Tape
from tracefork_spike.transport import TraceforkTransport


def test_spike_passes_end_to_end(tmp_path):
    result = record_replay_verify(str(tmp_path / "run.tape.sqlite"))
    assert result["passed"], result
    assert all(result["checks"].values()), result["checks"]


def test_two_exchanges_and_two_draws():
    result = record_replay_verify()
    # one tool-use turn + one final turn = 2 exchanges; clock + id = 2 draws.
    assert result["exchanges"] == 2
    assert result["draws"] == 2
    assert result["request_hashes_matched"] == 2


def test_record_and_replay_fingerprints_match():
    result = record_replay_verify()
    assert result["record_fingerprint"] == result["replay_fingerprint"]
    assert len(result["record_fingerprint"]) == 64  # sha256 hex


def test_replay_makes_no_network_call_inner_is_none():
    """A replay transport has no inner transport; if replay tried to hit the
    network it would AttributeError. Reaching a clean result proves zero calls."""
    rec_tape = Tape()
    nd = RecordingNondet()
    run_agent(make_client(TraceforkTransport("record", rec_tape, FakeAnthropicTransport())), nd)
    rec_tape.draws = nd.draws

    rep = TraceforkTransport("replay", rec_tape)  # inner=None
    out = run_agent(make_client(rep), ReplayNondet(rec_tape.draws))
    assert out["final_text"].startswith("Done")
    assert rep.fully_consumed()


def test_negative_control_drift_is_detected():
    """Replaying with fresh (drifting) nondeterminism must diverge — proves the
    verifier catches drift rather than always passing."""
    rec_tape = Tape()
    nd = RecordingNondet()
    run_agent(make_client(TraceforkTransport("record", rec_tape, FakeAnthropicTransport())), nd)
    rec_tape.draws = nd.draws

    with pytest.raises(DivergenceError):
        run_agent(make_client(TraceforkTransport("replay", rec_tape)), DriftingNondet())


def test_tampered_tape_is_detected():
    """Corrupt a recorded request body; replay must refuse with DivergenceError."""
    rec_tape = Tape()
    nd = RecordingNondet()
    run_agent(make_client(TraceforkTransport("record", rec_tape, FakeAnthropicTransport())), nd)
    rec_tape.draws = nd.draws

    # tamper the first recorded request body
    req, resp = rec_tape.exchanges[0]
    rec_tape.exchanges[0] = (req + b" ", resp)

    with pytest.raises(DivergenceError):
        run_agent(make_client(TraceforkTransport("replay", rec_tape)), ReplayNondet(rec_tape.draws))


def test_tape_survives_save_load_roundtrip(tmp_path):
    rec_tape = Tape()
    nd = RecordingNondet()
    run_agent(make_client(TraceforkTransport("record", rec_tape, FakeAnthropicTransport())), nd)
    rec_tape.draws = nd.draws

    path = str(tmp_path / "rt.tape.sqlite")
    rec_tape.save(path)
    loaded = Tape.load(path)
    assert loaded.digest() == rec_tape.digest()
    assert loaded.exchanges == rec_tape.exchanges
    assert loaded.draws == rec_tape.draws


def test_fake_endpoint_emits_real_anthropic_wire_format():
    """Sanity: the fake speaks Anthropic wire format well enough that the real SDK
    parses it into a tool_use Message."""
    transport = FakeAnthropicTransport()
    req = httpx.Request(
        "POST",
        "https://api.anthropic.com/v1/messages",
        content=json.dumps(
            {"model": "claude-opus-4-8", "messages": [{"role": "user", "content": "hi"}]}
        ).encode(),
    )
    resp = transport.handle_request(req)
    body = json.loads(resp.read())
    assert body["type"] == "message"
    assert body["stop_reason"] == "tool_use"
    assert any(b["type"] == "tool_use" and b["name"] == "book_flight" for b in body["content"])
