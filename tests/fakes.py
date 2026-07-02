"""Test re-exports of the offline Anthropic stand-ins.

The real implementations are production components in the package:
  - wire-format builders  → `tracefork.wire`
  - synthetic transports  → `tracefork.synthetic`

Tests import them from here for convenience; nothing in the package imports
from the test tree.
"""

from __future__ import annotations

from tracefork.synthetic import (
    AsyncScriptedFakeLLM,
    AsyncStreamingFakeLLM,
    FaultAwareFakeLLM,
    ScriptedFakeLLM,
)
from tracefork.wire import make_text_response, make_tool_use_response

__all__ = [
    "make_text_response",
    "make_tool_use_response",
    "ScriptedFakeLLM",
    "AsyncScriptedFakeLLM",
    "AsyncStreamingFakeLLM",
    "FaultAwareFakeLLM",
]
