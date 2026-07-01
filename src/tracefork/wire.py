"""Anthropic wire-format response builders (thin wrappers over the adapter).

The real building logic lives in the Anthropic provider adapter
(``tracefork.providers.anthropic``); these functions are kept as the stable,
Anthropic-defaulted entry points used in three places:
  - the offline test fakes (`tests/fakes.py` re-exports these),
  - the blame engine's perturbation responses,
  - the fault-injection validation suite.

Keeping them in the package (not in tests/) means production code never imports
from the test tree. Output bytes are unchanged from the pre-seam builders, so
record/replay bit-exactness is unaffected.
"""

from __future__ import annotations

from .constants import SONNET
from .providers import get_adapter


def make_text_response(
    text: str,
    *,
    model: str = SONNET,
    input_tokens: int = 100,
    output_tokens: int = 20,
) -> bytes:
    """Return Anthropic wire-format JSON bytes for a final text response."""
    return get_adapter("anthropic").build_text_response(
        text, model=model, input_tokens=input_tokens, output_tokens=output_tokens
    )


def make_tool_use_response(
    tool_name: str,
    tool_input: dict,
    *,
    model: str = SONNET,
    preamble: str = "",
    input_tokens: int = 100,
    output_tokens: int = 30,
) -> bytes:
    """Return Anthropic wire-format JSON bytes for a tool_use response."""
    return get_adapter("anthropic").build_tool_use_response(
        tool_name,
        tool_input,
        model=model,
        preamble=preamble,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
