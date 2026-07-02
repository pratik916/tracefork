"""OpenAI adapter tests — Chat Completions parse/build/SSE round-trips.

Offline, zero API keys: synthetic wire bytes only (no `openai` SDK import).
"""

import json

import pytest

from tracefork.providers import NormalizedResponse, ProviderAdapter, get_adapter
from tracefork.providers.openai import DEFAULT_OPENAI_MODEL, OpenAIAdapter
from tracefork.tape import sha256_hex

ADAPTER = get_adapter("openai")


# ── registry / protocol ──────────────────────────────────────────────────────


def test_openai_registered_and_satisfies_protocol():
    assert isinstance(ADAPTER, OpenAIAdapter)
    assert ADAPTER.name == "openai"
    assert isinstance(ADAPTER, ProviderAdapter)


# ── build + parse round-trips ────────────────────────────────────────────────


def test_build_text_response_shape_and_parse():
    raw = ADAPTER.build_text_response("hello world", input_tokens=7, output_tokens=3)
    d = json.loads(raw)
    assert d["object"] == "chat.completion"
    assert d["model"] == DEFAULT_OPENAI_MODEL
    assert d["choices"][0]["message"] == {"role": "assistant", "content": "hello world"}
    assert d["choices"][0]["finish_reason"] == "stop"
    assert d["usage"] == {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}

    norm = ADAPTER.parse_response(raw)
    assert norm.model == DEFAULT_OPENAI_MODEL
    assert norm.first_text() == "hello world"
    assert norm.input_tokens == 7
    assert norm.output_tokens == 3
    assert norm.finish_reason == "stop"


def test_build_tool_use_response_shape_and_parse():
    raw = ADAPTER.build_tool_use_response("book", {"city": "Tokyo"}, preamble="ok")
    d = json.loads(raw)
    msg = d["choices"][0]["message"]
    assert msg["content"] == "ok"
    call = msg["tool_calls"][0]
    assert call["type"] == "function"
    assert call["function"]["name"] == "book"
    # OpenAI arguments are a JSON *string*
    assert call["function"]["arguments"] == json.dumps({"city": "Tokyo"})
    assert d["choices"][0]["finish_reason"] == "tool_calls"

    norm = ADAPTER.parse_response(raw)
    assert norm.finish_reason == "tool_calls"
    text_parts = [p for p in norm.content if p.type == "text"]
    tool_parts = [p for p in norm.content if p.type == "tool_use"]
    assert text_parts[0].text == "ok"
    assert tool_parts[0].tool_name == "book"
    assert tool_parts[0].tool_input == {"city": "Tokyo"}


def test_build_tool_use_no_preamble_uses_null_content():
    raw = ADAPTER.build_tool_use_response("book", {"city": "Tokyo"})
    assert json.loads(raw)["choices"][0]["message"]["content"] is None


def test_message_id_override():
    raw = ADAPTER.build_text_response("x", message_id="chatcmpl-fixed")
    assert json.loads(raw)["id"] == "chatcmpl-fixed"
    assert ADAPTER.parse_response(raw).message_id == "chatcmpl-fixed"


def test_parse_response_missing_usage_yields_none_tokens():
    raw = json.dumps(
        {"model": "gpt-4o", "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}]}
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


def test_parse_response_structured_content_list():
    raw = json.dumps(
        {
            "model": "gpt-4o",
            "choices": [
                {
                    "message": {"content": [{"type": "text", "text": "chunked"}]},
                    "finish_reason": "stop",
                }
            ],
        }
    ).encode()
    assert ADAPTER.parse_response(raw).first_text() == "chunked"


# ── detect_model ─────────────────────────────────────────────────────────────


def test_detect_model_from_request_body():
    req = json.dumps({"model": "gpt-4o-mini", "messages": []}).encode()
    assert ADAPTER.detect_model(req) == "gpt-4o-mini"


def test_detect_model_absent_returns_none():
    assert ADAPTER.detect_model(json.dumps({"messages": []}).encode()) is None


def test_detect_model_non_json_returns_none():
    assert ADAPTER.detect_model(b"garbage") is None


# ── SSE streaming ─────────────────────────────────────────────────────────────


def test_parse_sse_extracts_first_chunk():
    sse = (
        b'data: {"id":"chatcmpl-1","object":"chat.completion.chunk",'
        b'"choices":[{"index":0,"delta":{"role":"assistant","content":"He"},'
        b'"finish_reason":null}]}\n'
        b'data: {"choices":[{"delta":{"content":"llo"},"finish_reason":null}]}\n'
        b"data: [DONE]\n"
    )
    parsed = ADAPTER.parse_sse(sse)
    assert parsed is not None
    assert parsed["id"] == "chatcmpl-1"
    assert parsed["choices"][0]["delta"]["content"] == "He"


def test_parse_sse_only_done_returns_none():
    assert ADAPTER.parse_sse(b"data: [DONE]\n") is None


def test_parse_sse_no_data_returns_none():
    assert ADAPTER.parse_sse(b"just some text\n") is None


def test_parse_sse_bad_json_returns_none():
    assert ADAPTER.parse_sse(b"data: {not json\n") is None


# ── tool_use_inputs / canonicalize / mutate round-trip ───────────────────────


def test_tool_use_inputs_decodes_arguments_string():
    raw = ADAPTER.build_tool_use_response("book", {"city": "Tokyo"})
    d, inputs = ADAPTER.tool_use_inputs(raw)
    assert d is not None
    assert inputs == [{"city": "Tokyo"}]


def test_tool_use_inputs_non_json():
    d, inputs = ADAPTER.tool_use_inputs(b"nope")
    assert d is None
    assert inputs == []


def test_canonicalize_request_is_sha256_of_bytes():
    req = b'{"model": "gpt-4o"}'
    assert ADAPTER.canonicalize_request(req) == sha256_hex(req)


def test_mutate_response_round_trips_text():
    norm = ADAPTER.parse_response(ADAPTER.build_text_response("hi"))
    out = ADAPTER.mutate_response(norm)
    assert ADAPTER.parse_response(out).first_text() == "hi"
    assert json.loads(out)["choices"][0]["finish_reason"] == "stop"


def test_mutate_response_round_trips_tool_use():
    norm = ADAPTER.parse_response(ADAPTER.build_tool_use_response("book", {"city": "Tokyo"}))
    out = ADAPTER.mutate_response(norm)
    reparsed = ADAPTER.parse_response(out)
    tool = [p for p in reparsed.content if p.type == "tool_use"][0]
    assert tool.tool_name == "book"
    assert tool.tool_input == {"city": "Tokyo"}
    assert json.loads(out)["choices"][0]["finish_reason"] == "tool_calls"
