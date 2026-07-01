"""Provider abstraction seam tests — registry, AnthropicAdapter round-trips, and
proof that routing wire/blame/faults through the adapter is byte/behaviour-identical.

Offline, zero API keys.
"""

import json

import pytest

from tracefork import blame as blame_mod
from tracefork import faults as faults_mod
from tracefork.constants import SONNET
from tracefork.faults import FAULT_MARKER, FaultClass, FaultInjector, _text_message
from tracefork.providers import (
    ContentPart,
    NormalizedResponse,
    ProviderAdapter,
    default_adapter,
    get_adapter,
    register_adapter,
    registered_providers,
)
from tracefork.providers.anthropic import AnthropicAdapter
from tracefork.tape import Tape, sha256_hex
from tracefork.wire import make_text_response, make_tool_use_response

ADAPTER = get_adapter("anthropic")


# ── registry ──────────────────────────────────────────────────────────────


def test_anthropic_is_registered_by_default():
    assert "anthropic" in registered_providers()
    assert isinstance(get_adapter(), AnthropicAdapter)
    assert get_adapter() is default_adapter()


def test_get_adapter_defaults_to_anthropic():
    assert get_adapter().name == "anthropic"


def test_unknown_adapter_raises_with_helpful_message():
    with pytest.raises(KeyError) as exc:
        get_adapter("gpt-9")
    assert "gpt-9" in str(exc.value)
    assert "anthropic" in str(exc.value)


def test_register_and_retrieve_custom_adapter():
    class DummyAdapter(AnthropicAdapter):
        name = "dummy"

    register_adapter(DummyAdapter())
    try:
        assert get_adapter("dummy").name == "dummy"
        assert "dummy" in registered_providers()
        # explicit name override wins over adapter.name
        register_adapter(DummyAdapter(), name="dummy2")
        assert get_adapter("dummy2").name == "dummy"
    finally:
        from tracefork.providers.base import _REGISTRY

        _REGISTRY.pop("dummy", None)
        _REGISTRY.pop("dummy2", None)


def test_adapter_satisfies_protocol():
    assert isinstance(ADAPTER, ProviderAdapter)


# ── build round-trips (byte-identical to the pre-seam wire builders) ────────


def test_build_text_response_is_byte_identical_to_wire():
    direct = ADAPTER.build_text_response("hello world")
    via_wire = make_text_response("hello world")
    assert direct == via_wire
    # exact envelope shape preserved
    d = json.loads(direct)
    assert d["type"] == "message"
    assert d["role"] == "assistant"
    assert d["model"] == SONNET
    assert d["content"] == [{"type": "text", "text": "hello world"}]
    assert d["stop_reason"] == "end_turn"
    assert d["usage"] == {"input_tokens": 100, "output_tokens": 20}
    assert d["id"] == "msg_" + sha256_hex(("hello world" + SONNET).encode())[:20]


def test_build_tool_use_response_is_byte_identical_to_wire():
    direct = ADAPTER.build_tool_use_response("book", {"city": "Tokyo"}, preamble="ok")
    via_wire = make_tool_use_response("book", {"city": "Tokyo"}, preamble="ok")
    assert direct == via_wire
    d = json.loads(direct)
    assert d["stop_reason"] == "tool_use"
    assert d["content"][0] == {"type": "text", "text": "ok"}
    assert d["content"][1]["type"] == "tool_use"
    assert d["content"][1]["name"] == "book"
    assert d["content"][1]["input"] == {"city": "Tokyo"}


def test_build_text_response_message_id_override():
    b = ADAPTER.build_text_response("x", message_id="msg_fault", input_tokens=10, output_tokens=10)
    d = json.loads(b)
    assert d["id"] == "msg_fault"
    assert d["usage"] == {"input_tokens": 10, "output_tokens": 10}


# ── parse_response ──────────────────────────────────────────────────────────


def test_parse_response_text():
    norm = ADAPTER.parse_response(make_text_response("hi", input_tokens=7, output_tokens=3))
    assert norm.model == SONNET
    assert norm.input_tokens == 7
    assert norm.output_tokens == 3
    assert norm.finish_reason == "end_turn"
    assert norm.first_text() == "hi"
    assert norm.text_parts() == ["hi"]


def test_parse_response_tool_use():
    norm = ADAPTER.parse_response(make_tool_use_response("book", {"city": "Tokyo"}))
    assert norm.finish_reason == "tool_use"
    part = norm.content[0]
    assert part.type == "tool_use"
    assert part.tool_name == "book"
    assert part.tool_input == {"city": "Tokyo"}
    assert norm.first_text() == ""


def test_parse_response_missing_usage_yields_none_tokens():
    raw = json.dumps({"type": "message", "content": [{"type": "text", "text": "hi"}]}).encode()
    norm = ADAPTER.parse_response(raw)
    assert norm.input_tokens is None
    assert norm.output_tokens is None
    assert norm.first_text() == "hi"


def test_parse_response_non_dict_json_is_empty():
    norm = ADAPTER.parse_response(b"[1, 2, 3]")
    assert norm == NormalizedResponse()
    assert norm.first_text() == ""


def test_parse_response_non_json_raises():
    with pytest.raises(json.JSONDecodeError):
        ADAPTER.parse_response(b"not json at all")


def test_parse_response_unknown_block_type_preserved():
    raw = json.dumps({"content": [{"type": "thinking", "thinking": "..."}]}).encode()
    norm = ADAPTER.parse_response(raw)
    assert norm.content[0].type == "thinking"
    assert norm.first_text() == ""


# ── detect_model ────────────────────────────────────────────────────────────


def test_detect_model_present():
    req = json.dumps({"model": "claude-opus-4-8", "messages": []}).encode()
    assert ADAPTER.detect_model(req) == "claude-opus-4-8"


def test_detect_model_absent_returns_none():
    assert ADAPTER.detect_model(json.dumps({"messages": []}).encode()) is None


def test_detect_model_non_json_returns_none():
    assert ADAPTER.detect_model(b"garbage") is None


# ── parse_sse ───────────────────────────────────────────────────────────────


def test_parse_sse_extracts_first_object():
    sse = b'event: x\ndata: {"type": "message_start", "n": 1}\ndata: [DONE]\n'
    assert ADAPTER.parse_sse(sse) == {"type": "message_start", "n": 1}


def test_parse_sse_only_done_returns_none():
    assert ADAPTER.parse_sse(b"data: [DONE]\n") is None


def test_parse_sse_no_data_returns_none():
    assert ADAPTER.parse_sse(b"just some text\n") is None


def test_parse_sse_bad_json_returns_none():
    assert ADAPTER.parse_sse(b"data: {not json\n") is None


# ── tool_use_inputs ─────────────────────────────────────────────────────────


def test_tool_use_inputs_returns_mutable_dicts():
    resp = make_tool_use_response("book", {"city": "Tokyo"})
    d, inputs = ADAPTER.tool_use_inputs(resp)
    assert d is not None
    assert inputs == [{"city": "Tokyo"}]
    inputs[0]["city"] = "Paris"
    assert json.loads(json.dumps(d))["content"][-1]["input"]["city"] == "Paris"


def test_tool_use_inputs_no_tool_use():
    d, inputs = ADAPTER.tool_use_inputs(make_text_response("hi"))
    assert d is not None
    assert inputs == []


def test_tool_use_inputs_non_json():
    d, inputs = ADAPTER.tool_use_inputs(b"nope")
    assert d is None
    assert inputs == []


# ── canonicalize_request / mutate_response ──────────────────────────────────


def test_canonicalize_request_is_sha256_of_bytes():
    req = b'{"model": "claude-sonnet-4-6"}'
    assert ADAPTER.canonicalize_request(req) == sha256_hex(req)


def test_mutate_response_round_trips_text():
    norm = NormalizedResponse(content=(ContentPart(type="text", text="hi"),), model=SONNET)
    out = ADAPTER.mutate_response(norm)
    reparsed = ADAPTER.parse_response(out)
    assert reparsed.first_text() == "hi"
    assert json.loads(out)["stop_reason"] == "end_turn"


def test_mutate_response_round_trips_tool_use():
    norm = NormalizedResponse(
        content=(ContentPart(type="tool_use", tool_name="book", tool_input={"city": "Tokyo"}),),
    )
    out = ADAPTER.mutate_response(norm)
    reparsed = ADAPTER.parse_response(out)
    assert reparsed.content[0].tool_name == "book"
    assert reparsed.content[0].tool_input == {"city": "Tokyo"}
    assert json.loads(out)["stop_reason"] == "tool_use"


# ── proof: routing blame/faults through the adapter is unchanged ────────────


def _tape_with(*resps: bytes) -> Tape:
    tape = Tape()
    req = json.dumps({"model": SONNET, "messages": []}).encode()
    for r in resps:
        tape.append_exchange(req, r)
    return tape


def test_blame_outcome_text_matches_expected():
    assert blame_mod._outcome_text(make_text_response("hello")) == "hello"
    # non-JSON falls back to decoded raw bytes
    assert blame_mod._outcome_text(b"raw bytes") == "raw bytes"
    # JSON list (non-dict) yields empty string, not the raw list text
    assert blame_mod._outcome_text(b"[1,2]") == ""


def test_blame_detect_model_reads_request_model():
    tape = _tape_with(make_text_response("hi"))
    assert blame_mod._detect_model(tape) == SONNET


def test_blame_avg_tokens_reads_recorded_usage():
    tape = _tape_with(make_text_response("hi", input_tokens=40, output_tokens=8))
    avg_in, avg_out = blame_mod._avg_tokens(tape)
    # both come straight from the response's recorded ``usage``
    assert avg_in == 40
    assert avg_out == 8


def test_blame_avg_tokens_byte_fallback_when_usage_absent():
    raw = json.dumps({"type": "message", "content": [{"type": "text", "text": "hi"}]}).encode()
    tape = _tape_with(raw)
    avg_in, avg_out = blame_mod._avg_tokens(tape)
    # no usage → ~4-bytes-per-token estimate from the raw request/response bytes
    assert avg_in > 0
    assert avg_out > 0


def test_faults_text_message_is_stable_and_marked():
    b = _text_message(f"boom {FAULT_MARKER}")
    d = json.loads(b)
    assert d["id"] == "msg_fault"
    assert d["model"] == SONNET
    assert d["usage"] == {"input_tokens": 10, "output_tokens": 10}
    assert FAULT_MARKER in d["content"][0]["text"]


def test_faults_injectors_stay_valid_json_with_marker_through_adapter():
    tape = _tape_with(make_tool_use_response("check", {"seats": 3, "destination": "Tokyo"}))
    for fc in FaultClass:
        mutated = FaultInjector.inject(tape, 0, fc)
        json.loads(mutated)  # valid JSON
        assert faults_mod.FAULT_MARKER_BYTES in mutated


def test_corrupt_tool_output_preserves_tool_block_and_flips_field():
    resp = make_tool_use_response("check", {"seats": 3})
    mutated = FaultInjector.corrupt_tool_output(resp, field="seats", new_value=0)
    d = json.loads(mutated)
    tool = [b for b in d["content"] if b.get("type") == "tool_use"][0]
    assert tool["input"]["seats"] == 0
    assert tool["input"]["_tracefork_fault"] == FAULT_MARKER
