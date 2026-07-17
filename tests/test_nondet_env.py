"""nondet.py — direct class tests for the env channel (`get_env`), plus an
end-to-end record/replay receipt through TraceforkTransport (mirrors
tests/test_nondet.py's random-channel tests for the production package), and
coverage.py's tally of the new "env" draw kind."""

from __future__ import annotations

import random

import anthropic
import httpx
import pytest

from tests.fakes import ScriptedFakeLLM, make_text_response
from tracefork.coverage import tape_draw_coverage
from tracefork.nondet import (
    DivergenceError,
    DriftingNondet,
    RecordingNondet,
    ReplayNondet,
)
from tracefork.tape import Tape
from tracefork.transport import TraceforkTransport

TEXT_RESP = make_text_response("Done.")


def _toy_agent(client: anthropic.Anthropic, nondet) -> str:
    """Embeds an env draw into the request so a divergence in the draw
    becomes a divergence in the request body."""
    v = nondet.get_env("TF_TOY_ENV", "default")
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=100,
        messages=[{"role": "user", "content": f"env: {v}"}],
    )
    return resp.content[0].text


def _client(transport: httpx.BaseTransport) -> anthropic.Anthropic:
    return anthropic.Anthropic(
        api_key="sk-ant-fake",
        http_client=httpx.Client(transport=transport),
        max_retries=0,
    )


# ── Direct class tests ──────────────────────────────────────────────────────


def test_recording_nondet_get_env_reads_real_env_and_logs_exactly_one_draw(monkeypatch):
    monkeypatch.setenv("TF_ENV_TEST", "some_value")
    nd = RecordingNondet()
    v = nd.get_env("TF_ENV_TEST")
    assert v == "some_value"
    assert nd.draws == [("env", "1\0TF_ENV_TEST\0some_value")]


def test_get_env_record_replay_round_trip_distinguishes_unset_from_empty_string(
    monkeypatch,
):
    """The unset (None) case must round-trip distinctly from an empty-string
    value — both must survive record→replay exactly, even when the process
    environment differs at replay time (proves replay serves the tape, not
    the live environment)."""
    monkeypatch.delenv("TF_ENV_UNSET", raising=False)
    monkeypatch.setenv("TF_ENV_EMPTY", "")
    monkeypatch.setenv("TF_ENV_SET", "recorded")

    nd = RecordingNondet()
    unset = nd.get_env("TF_ENV_UNSET")
    empty = nd.get_env("TF_ENV_EMPTY")
    present = nd.get_env("TF_ENV_SET")
    assert unset is None
    assert empty == ""
    assert present == "recorded"

    # Simulate a differing process environment at replay time.
    monkeypatch.setenv("TF_ENV_UNSET", "now-set-should-be-ignored")
    monkeypatch.setenv("TF_ENV_EMPTY", "now-nonempty-should-be-ignored")
    monkeypatch.setenv("TF_ENV_SET", "drifted-should-be-ignored")

    replay = ReplayNondet(nd.draws)
    assert replay.get_env("TF_ENV_UNSET") is None
    assert replay.get_env("TF_ENV_EMPTY") == ""
    assert replay.get_env("TF_ENV_SET") == "recorded"
    assert replay.fully_consumed()


def test_get_env_uses_default_when_var_unset(monkeypatch):
    monkeypatch.delenv("TF_ENV_DEFAULTED", raising=False)
    nd = RecordingNondet()
    v = nd.get_env("TF_ENV_DEFAULTED", default="fallback")
    assert v == "fallback"

    replay = ReplayNondet(nd.draws)
    assert replay.get_env("TF_ENV_DEFAULTED", default="fallback") == "fallback"


def test_interleaved_clock_uuid_random_env_round_trip_in_order(monkeypatch):
    """All four draw kinds share one ordered log; replay must serve each
    kind back in the order it was recorded, regardless of interleaving."""
    monkeypatch.setenv("TF_ENV_INTERLEAVE", "interleaved_value")
    random.seed(1)
    nd = RecordingNondet()
    clock1 = nd.now_iso()
    rand1 = nd.random_float()
    uuid1 = nd.new_uuid_hex()
    env1 = nd.get_env("TF_ENV_INTERLEAVE")
    rand2 = nd.random_float()

    assert [k for k, _ in nd.draws] == ["clock", "random", "uuid", "env", "random"]

    replay = ReplayNondet(nd.draws)
    assert replay.now_iso() == clock1
    assert replay.random_float() == rand1
    assert replay.new_uuid_hex() == uuid1
    assert replay.get_env("TF_ENV_INTERLEAVE") == env1
    assert replay.random_float() == rand2
    assert replay.fully_consumed()


def test_replay_get_env_rejects_kind_mismatch():
    replay = ReplayNondet([("uuid", "deadbeef")])
    with pytest.raises(DivergenceError, match="env"):
        replay.get_env("ANY_VAR")


def test_replay_get_env_exhausted_tape_raises():
    replay = ReplayNondet([])
    with pytest.raises(DivergenceError, match="exhausted"):
        replay.get_env("ANY_VAR")


def test_replay_get_env_rejects_name_mismatch(monkeypatch):
    """Only get_env takes an argument, so replay must additionally assert the
    requested name matches the recorded one -- a stronger check than
    clock/uuid/random need."""
    monkeypatch.setenv("MY_VAR", "value")
    nd = RecordingNondet()
    nd.get_env("MY_VAR")

    replay = ReplayNondet(nd.draws)
    with pytest.raises(DivergenceError, match="OTHER_VAR"):
        replay.get_env("OTHER_VAR")


def test_drifting_nondet_get_env_reads_fresh_value_at_replay(monkeypatch):
    """DriftingNondet inherits RecordingNondet's get_env -- it must read a
    genuinely fresh env value, not replay a fixed/recorded one."""
    monkeypatch.setenv("TF_ENV_DRIFT", "recorded_value")
    recorded = RecordingNondet().get_env("TF_ENV_DRIFT")

    monkeypatch.setenv("TF_ENV_DRIFT", "drifted_value")
    drifted = DriftingNondet().get_env("TF_ENV_DRIFT")

    assert drifted != recorded
    assert drifted == "drifted_value"


# ── End-to-end record → replay receipt (mirrors test_nondet.py) ────────────


def test_env_channel_record_replay_end_to_end_bit_exact_despite_differing_env(
    monkeypatch,
):
    monkeypatch.setenv("TF_TOY_ENV", "record-time-value")
    tape = Tape()
    rec_nondet = RecordingNondet()
    rec_transport = TraceforkTransport("record", tape, ScriptedFakeLLM([TEXT_RESP]))
    _toy_agent(_client(rec_transport), rec_nondet)
    tape.draws = rec_nondet.draws

    # Simulate the process environment having changed by replay time -- the
    # replay must serve the tape's recorded value, not this live one.
    monkeypatch.setenv("TF_TOY_ENV", "replay-time-value-should-be-ignored")

    rep_transport = TraceforkTransport("replay", tape)
    out = _toy_agent(_client(rep_transport), ReplayNondet(tape.draws))
    assert out == "Done."
    assert rep_transport.fully_consumed()


# ── coverage.py: tape_draw_coverage tallies the new "env" kind ──────────────


def test_tape_draw_coverage_reports_nonzero_env_count():
    tape = Tape(draws=[("env", "1\0X\0y"), ("env", "0\0Z\0")])
    draw_counts, _concurrency, _guard = tape_draw_coverage(tape)
    assert draw_counts == {"env": 2}


def test_tape_draw_coverage_pre_existing_tape_has_no_zero_filled_env_entry():
    tape = Tape(draws=[("clock", "a"), ("uuid", "b"), ("random", "c")])
    draw_counts, _concurrency, _guard = tape_draw_coverage(tape)
    assert draw_counts == {"clock": 1, "uuid": 1, "random": 1}
    assert "env" not in draw_counts
