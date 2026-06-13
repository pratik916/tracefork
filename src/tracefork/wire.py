"""Anthropic wire-format response builders.

Real Anthropic Messages-API JSON, used in three places:
  - the offline test fakes (`tests/fakes.py` re-exports these),
  - the blame engine's perturbation responses,
  - the fault-injection validation suite.

Keeping them in the package (not in tests/) means production code never
imports from the test tree.
"""
from __future__ import annotations

import json

from .tape import sha256_hex


def make_text_response(
    text: str,
    *,
    model: str = "claude-sonnet-4-6",
    input_tokens: int = 100,
    output_tokens: int = 20,
) -> bytes:
    """Return Anthropic wire-format JSON bytes for a final text response."""
    rid = "msg_" + sha256_hex((text + model).encode())[:20]
    return json.dumps({
        "id": rid,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }).encode()


def make_tool_use_response(
    tool_name: str,
    tool_input: dict,
    *,
    model: str = "claude-sonnet-4-6",
    preamble: str = "",
    input_tokens: int = 100,
    output_tokens: int = 30,
) -> bytes:
    """Return Anthropic wire-format JSON bytes for a tool_use response."""
    content: list[dict] = []
    if preamble:
        content.append({"type": "text", "text": preamble})
    toolu_id = "toolu_" + sha256_hex((tool_name + json.dumps(tool_input)).encode())[:18]
    content.append({
        "type": "tool_use",
        "id": toolu_id,
        "name": tool_name,
        "input": tool_input,
    })
    rid = "msg_" + sha256_hex((tool_name + model).encode())[:20]
    return json.dumps({
        "id": rid,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }).encode()
