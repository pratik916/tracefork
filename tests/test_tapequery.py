"""Offline tests for `tapequery.py`'s `state_at`/`slice` read-only views.

Built via the same record-mode fixture pattern `test_diff.py` already uses:
`ScriptedFakeLLM` + `TraceforkTransport("record", ...)` — no real API key,
no network.
"""

import base64
import json

import anthropic
import httpx

from tests.fakes import ScriptedFakeLLM, make_text_response
from tracefork.tape import Tape
from tracefork.tapequery import ExchangeView, TapeState, state_at
from tracefork.tapequery import slice as tq_slice
from tracefork.transport import TraceforkTransport

RESP_1 = make_text_response("Response 1")
RESP_2 = make_text_response("Response 2")
RESP_3 = make_text_response("Response 3")


def _three_turn_agent(client: anthropic.Anthropic) -> None:
    """Three independent single-turn calls — just enough steps to exercise a
    fold (`state_at`) and a range (`slice`) over more than two exchanges."""
    client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=100,
        messages=[{"role": "user", "content": "turn1"}],
    )
    client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=100,
        messages=[{"role": "user", "content": "turn2"}],
    )
    client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=100,
        messages=[{"role": "user", "content": "turn3"}],
    )


def _build_three_exchange_tape() -> Tape:
    fake = ScriptedFakeLLM([RESP_1, RESP_2, RESP_3])
    tape = Tape()
    transport = TraceforkTransport("record", tape, fake)
    client = anthropic.Anthropic(
        api_key="sk-ant-fake",
        http_client=httpx.Client(transport=transport),
        max_retries=0,
    )
    _three_turn_agent(client)
    return tape


# ── state_at ─────────────────────────────────────────────────────────────────


def test_state_at_returns_exchanges_0_to_n_inclusive():
    tape = _build_three_exchange_tape()

    result = state_at(tape, 1)

    assert isinstance(result, TapeState)
    assert result.step_index == 1
    assert len(result.exchanges) == 2
    for view in result.exchanges:
        assert isinstance(view, ExchangeView)
    assert result.exchanges[0].step_index == 0
    assert result.exchanges[1].step_index == 1
    # Request bodies round-trip through JSON decode.
    assert result.exchanges[0].request["messages"][0]["content"] == "turn1"
    assert result.exchanges[1].request["messages"][0]["content"] == "turn2"


def test_state_at_negative_n_raises_value_error():
    tape = _build_three_exchange_tape()
    try:
        state_at(tape, -1)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_state_at_out_of_range_n_raises_index_error():
    tape = _build_three_exchange_tape()
    try:
        state_at(tape, len(tape.exchanges))
        raise AssertionError("expected IndexError")
    except IndexError:
        pass


# ── slice ────────────────────────────────────────────────────────────────────


def test_slice_returns_matching_exchange_views():
    tape = _build_three_exchange_tape()

    result = tq_slice(tape, 0, 2)

    assert len(result) == 2
    req0, resp0 = tape.exchange(0)
    req1, resp1 = tape.exchange(1)
    assert result[0].step_index == 0
    assert result[0].request == json.loads(req0)
    assert result[0].response == json.loads(resp0)
    assert result[1].step_index == 1
    assert result[1].request == json.loads(req1)
    assert result[1].response == json.loads(resp1)


def test_slice_clamps_upper_bound_past_tape_length():
    tape = _build_three_exchange_tape()

    result = tq_slice(tape, 1, 100)

    assert len(result) == len(tape.exchanges) - 1
    assert [v.step_index for v in result] == [1, 2]


def test_slice_start_past_end_of_short_tape_returns_empty_tuple():
    tape = _build_three_exchange_tape()

    result = tq_slice(tape, 5, 100)

    assert result == ()


def test_slice_start_greater_than_end_raises_value_error():
    tape = _build_three_exchange_tape()
    try:
        tq_slice(tape, 2, 1)
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "start" in str(exc)


def test_slice_on_empty_tape_returns_empty_tuple():
    result = tq_slice(Tape(), 0, 5)
    assert result == ()


# ── raw / non-JSON bodies ────────────────────────────────────────────────────


def test_non_json_exchange_body_decodes_via_raw_b64_fallback():
    """A raw (non-JSON) exchange body decodes via the same `{'_raw_b64': ...}`
    fallback `divergence.py` already uses — proving no new decode logic was
    invented for this module."""
    tape = Tape()
    raw_request = b"not-json-bytes"
    tape.append_exchange(raw_request, b'{"text": "ok"}')

    result = state_at(tape, 0)

    view = result.exchanges[0]
    assert view.request == {"_raw_b64": base64.b64encode(raw_request).decode("ascii")}
    assert view.response == {"text": "ok"}
