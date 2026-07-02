"""Gemini adapter tests — generateContent parse/build/SSE round-trips.

Offline, zero API keys: synthetic wire bytes only (no `google-genai` SDK import).
"""

import json

import pytest

from tracefork.providers import NormalizedResponse, ProviderAdapter, get_adapter
from tracefork.providers.gemini import DEFAULT_GEMINI_MODEL, GeminiAdapter
from tracefork.tape import sha256_hex

ADAPTER = get_adapter("gemini")


# ── registry / protocol ──────────────────────────────────────────────────────


def test_gemini_registered_and_satisfies_protocol():
    assert isinstance(ADAPTER, GeminiAdapter)
    assert ADAPTER.name == "gemini"
    assert isinstance(ADAPTER, ProviderAdapter)


# ── build + parse round-trips ────────────────────────────────────────────────


def test_build_text_response_shape_and_parse():
    raw = ADAPTER.build_text_response("hello world", input_tokens=7, output_tokens=3)
    d = json.loads(raw)
    assert d["modelVersion"] == DEFAULT_GEMINI_MODEL
    cand = d["candidates"][0]
    assert cand["content"] == {"role": "model", "parts": [{"text": "hello world"}]}
    assert cand["finishReason"] == "STOP"
    assert d["usageMetadata"] == {
        "promptTokenCount": 7,
        "candidatesTokenCount": 3,
        "totalTokenCount": 10,
    }

    norm = ADAPTER.parse_response(raw)
    assert norm.model == DEFAULT_GEMINI_MODEL
    assert norm.first_text() == "hello world"
    assert norm.input_tokens == 7
    assert norm.output_tokens == 3
    assert norm.finish_reason == "STOP"


def test_build_tool_use_response_shape_and_parse():
    raw = ADAPTER.build_tool_use_response("book", {"city": "Tokyo"}, preamble="ok")
    parts = json.loads(raw)["candidates"][0]["content"]["parts"]
    assert parts[0] == {"text": "ok"}
    # Gemini function args are a nested OBJECT (not a string)
    assert parts[1] == {"functionCall": {"name": "book", "args": {"city": "Tokyo"}}}

    norm = ADAPTER.parse_response(raw)
    tool = [p for p in norm.content if p.type == "tool_use"][0]
    assert tool.tool_name == "book"
    assert tool.tool_input == {"city": "Tokyo"}
    assert [p for p in norm.content if p.type == "text"][0].text == "ok"


def test_message_id_override():
    raw = ADAPTER.build_text_response("x", message_id="resp-fixed")
    assert json.loads(raw)["responseId"] == "resp-fixed"
    assert ADAPTER.parse_response(raw).message_id == "resp-fixed"


def test_parse_response_missing_usage_yields_none_tokens():
    raw = json.dumps(
        {"candidates": [{"content": {"parts": [{"text": "hi"}]}, "finishReason": "STOP"}]}
    ).encode()
    norm = ADAPTER.parse_response(raw)
    assert norm.input_tokens is None
    assert norm.output_tokens is None
    assert norm.first_text() == "hi"


def test_parse_response_non_dict_json_is_empty():
    assert ADAPTER.parse_response(b"[1, 2, 3]") == NormalizedResponse()


def test_parse_response_non_json_raises():
    with pytest.raises(json.JSONDecodeError):
        ADAPTER.parse_response(b"not json")


# ── detect_model (Gemini model usually lives in the URL, not the body) ────────


def test_detect_model_from_body_field():
    req = json.dumps({"model": "gemini-1.5-pro"}).encode()
    assert ADAPTER.detect_model(req) == "gemini-1.5-pro"


def test_detect_model_strips_models_prefix():
    req = json.dumps({"model": "models/gemini-2.0-flash"}).encode()
    assert ADAPTER.detect_model(req) == "gemini-2.0-flash"


def test_detect_model_absent_returns_none():
    assert ADAPTER.detect_model(json.dumps({"contents": []}).encode()) is None


def test_detect_model_non_json_returns_none():
    assert ADAPTER.detect_model(b"garbage") is None


# ── SSE streaming ─────────────────────────────────────────────────────────────


def test_parse_sse_extracts_first_chunk():
    sse = (
        b'data: {"candidates":[{"content":{"parts":[{"text":"He"}]},"index":0}]}\n'
        b'data: {"candidates":[{"content":{"parts":[{"text":"llo"}]},"index":0}]}\n'
    )
    parsed = ADAPTER.parse_sse(sse)
    assert parsed is not None
    assert parsed["candidates"][0]["content"]["parts"][0]["text"] == "He"


def test_parse_sse_no_data_returns_none():
    assert ADAPTER.parse_sse(b"random line\n") is None


def test_parse_sse_bad_json_returns_none():
    assert ADAPTER.parse_sse(b"data: {not json\n") is None


# ── tool_use_inputs / canonicalize / mutate round-trip ───────────────────────


def test_tool_use_inputs_returns_live_args_dicts():
    raw = ADAPTER.build_tool_use_response("book", {"city": "Tokyo"})
    d, inputs = ADAPTER.tool_use_inputs(raw)
    assert d is not None
    assert inputs == [{"city": "Tokyo"}]
    # live-linked: mutating the returned dict reflects when d is re-serialized
    inputs[0]["city"] = "Paris"
    reparsed = json.loads(json.dumps(d))
    fc = reparsed["candidates"][0]["content"]["parts"][-1]["functionCall"]
    assert fc["args"]["city"] == "Paris"


def test_tool_use_inputs_non_json():
    d, inputs = ADAPTER.tool_use_inputs(b"nope")
    assert d is None
    assert inputs == []


def test_canonicalize_request_is_sha256_of_bytes():
    req = b'{"contents": []}'
    assert ADAPTER.canonicalize_request(req) == sha256_hex(req)


def test_mutate_response_round_trips_text():
    norm = ADAPTER.parse_response(ADAPTER.build_text_response("hi"))
    out = ADAPTER.mutate_response(norm)
    assert ADAPTER.parse_response(out).first_text() == "hi"
    assert json.loads(out)["candidates"][0]["finishReason"] == "STOP"


def test_mutate_response_round_trips_tool_use():
    norm = ADAPTER.parse_response(ADAPTER.build_tool_use_response("book", {"city": "Tokyo"}))
    out = ADAPTER.mutate_response(norm)
    reparsed = ADAPTER.parse_response(out)
    tool = [p for p in reparsed.content if p.type == "tool_use"][0]
    assert tool.tool_name == "book"
    assert tool.tool_input == {"city": "Tokyo"}
