"""BedrockAdapter tests — InvokeModel (Anthropic Messages shape) delegation,
Converse-shape best-effort reads, and fault re-serialization. Offline, $0.
"""

import json

from tracefork.providers import get_adapter
from tracefork.providers.base import ContentPart, NormalizedResponse
from tracefork.providers.bedrock import (
    ANTHROPIC_VERSION,
    BedrockAdapter,
    build_invoke_model_request,
)


def test_registered_under_bedrock_name():
    adapter = get_adapter("bedrock")
    assert isinstance(adapter, BedrockAdapter)
    assert adapter.name == "bedrock"


def test_detect_model_always_none():
    # Bedrock's InvokeModel body has no "model" field at all -- the model id
    # is a URL path segment, never body content.
    adapter = BedrockAdapter()
    req = build_invoke_model_request([{"role": "user", "content": "hi"}])
    assert adapter.detect_model(req) is None
    assert adapter.detect_model(b"not json at all") is None


def test_build_invoke_model_request_shape():
    body_bytes = build_invoke_model_request(
        [{"role": "user", "content": "hi"}], max_tokens=256, system="be terse"
    )
    body = json.loads(body_bytes)
    assert body["anthropic_version"] == ANTHROPIC_VERSION
    assert body["max_tokens"] == 256
    assert body["system"] == "be terse"
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    assert "model" not in body  # model id lives in the URL, never the body


def test_parse_response_invoke_model_shape_delegates_to_anthropic_adapter():
    adapter = BedrockAdapter()
    resp = adapter.build_text_response("hello from bedrock", model="anthropic.claude-sonnet-4-6")
    normalized = adapter.parse_response(resp)
    assert normalized.first_text() == "hello from bedrock"
    assert normalized.model == "anthropic.claude-sonnet-4-6"
    assert normalized.finish_reason == "end_turn"

    # And it's genuinely the Anthropic Messages shape: the anthropic adapter
    # parses it identically.
    anthropic_view = get_adapter("anthropic").parse_response(resp)
    assert anthropic_view.first_text() == normalized.first_text()


def test_parse_response_tool_use_invoke_model_shape():
    adapter = BedrockAdapter()
    resp = adapter.build_tool_use_response("get_weather", {"city": "Paris"}, preamble="checking...")
    normalized = adapter.parse_response(resp)
    assert normalized.first_text() == "checking..."
    tool_parts = [p for p in normalized.content if p.type == "tool_use"]
    assert len(tool_parts) == 1
    assert tool_parts[0].tool_name == "get_weather"
    assert tool_parts[0].tool_input == {"city": "Paris"}
    assert normalized.finish_reason == "tool_use"


def test_parse_response_converse_shape_text():
    adapter = BedrockAdapter()
    body = json.dumps(
        {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": "hello from converse"}],
                }
            },
            "stopReason": "end_turn",
            "usage": {"inputTokens": 42, "outputTokens": 7, "totalTokens": 49},
        }
    ).encode()
    normalized = adapter.parse_response(body)
    assert normalized.first_text() == "hello from converse"
    assert normalized.input_tokens == 42
    assert normalized.output_tokens == 7
    assert normalized.finish_reason == "end_turn"
    assert normalized.model is None


def test_parse_response_converse_shape_tool_use():
    adapter = BedrockAdapter()
    body = json.dumps(
        {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {"toolUse": {"toolUseId": "tu_1", "name": "lookup", "input": {"q": "x"}}}
                    ],
                }
            },
            "stopReason": "tool_use",
            "usage": {"inputTokens": 10, "outputTokens": 3},
        }
    ).encode()
    normalized = adapter.parse_response(body)
    tool_parts = [p for p in normalized.content if p.type == "tool_use"]
    assert len(tool_parts) == 1
    assert tool_parts[0].tool_name == "lookup"
    assert tool_parts[0].tool_id == "tu_1"
    assert tool_parts[0].tool_input == {"q": "x"}
    assert normalized.finish_reason == "tool_use"


def test_tool_use_inputs_returns_mutable_inputs_for_invoke_model_shape():
    adapter = BedrockAdapter()
    resp = adapter.build_tool_use_response("search", {"query": "cats"})
    parsed, inputs = adapter.tool_use_inputs(resp)
    assert parsed is not None
    assert inputs == [{"query": "cats"}]
    # Mutating in place and re-dumping is the lossless fault-injection path
    # (mirrors AnthropicAdapter.tool_use_inputs' contract).
    inputs[0]["query"] = "dogs"
    mutated_bytes = json.dumps(parsed).encode()
    assert json.loads(mutated_bytes)["content"][0]["input"]["query"] == "dogs"


def test_tool_use_inputs_converse_shape_returns_empty():
    adapter = BedrockAdapter()
    body = json.dumps(
        {"output": {"message": {"content": [{"text": "no tools here"}]}}, "stopReason": "end_turn"}
    ).encode()
    parsed, inputs = adapter.tool_use_inputs(body)
    assert parsed is None
    assert inputs == []


def test_parse_sse_is_always_none():
    # Bedrock streaming is AWS event-stream binary framing, not SSE.
    adapter = BedrockAdapter()
    assert adapter.parse_sse(b"data: {}\n\n") is None
    assert adapter.parse_sse(b"") is None


def test_mutate_response_reserializes_fault_to_invoke_model_shape():
    adapter = BedrockAdapter()
    normalized = NormalizedResponse(
        model="anthropic.claude-sonnet-4-6",
        content=(ContentPart(type="text", text="faulted content"),),
        input_tokens=100,
        output_tokens=20,
        finish_reason="end_turn",
        message_id="msg_fault",
    )
    mutated_bytes = adapter.mutate_response(normalized)
    reparsed = adapter.parse_response(mutated_bytes)
    assert reparsed.first_text() == "faulted content"
    assert reparsed.message_id == "msg_fault"
    # It's genuinely the InvokeModel/Anthropic Messages wire shape.
    data = json.loads(mutated_bytes)
    assert data["type"] == "message"
    assert data["content"][0]["text"] == "faulted content"
