"""AWS Bedrock adapter — Claude-on-Bedrock's ``InvokeModel`` wire format behind
the seam.

Bedrock's InvokeModel API for Anthropic Claude models returns the model's
*native* response directly in the HTTP body: for Claude that is byte-for-byte
the Anthropic Messages API response shape (``content[]`` blocks, top-level
``model``/``stop_reason``, ``usage.input_tokens``/``output_tokens``) — AWS's
own docs example reads the response with ``response_body.get("content")``,
exactly like a direct Anthropic Messages response. The *request* body differs
from the direct API in exactly two ways: it adds a top-level
``"anthropic_version": "bedrock-2023-05-31"`` field, and it OMITS the
top-level ``"model"`` field — the model id is a URL path segment
(``/model/{modelId}/invoke``), not body content (mirroring
``providers/gemini.py``'s "model id lives in the URL" limitation — see
``detect_model`` below).

This adapter therefore delegates its InvokeModel-shape read/build logic
straight to the registered ``"anthropic"`` adapter rather than duplicating it,
and adds only: the request-shape delta (``anthropic_version``, via
``build_invoke_model_request``) and a best-effort reader for the Converse
API's distinct envelope (``output.message``/``stopReason``/
``usage.inputTokens``). Converse *building* is intentionally NOT
implemented — fault re-serialization always targets the InvokeModel
(Anthropic Messages) shape, which is a legitimate Bedrock wire form on its
own; see the bead's scope note in ``bedrock_transport.py``.

SCOPE — streaming: ``InvokeModelWithResponseStream``/``ConverseStream`` use
AWS's binary ``application/vnd.amazon.eventstream`` framing (see
``eventstream.py``), not SSE. ``parse_sse`` is therefore always a no-op here
(kept only for ``ProviderAdapter`` protocol conformance) — a real streaming
response is read via ``eventstream.py``'s codec directly, never through this
method. See ``bedrock_transport.py``'s module docstring for the full
streaming-through-botocore limitation.
"""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import unquote

from ..tape import sha256_hex
from .anthropic import AnthropicAdapter
from .base import (
    ContentPart,
    NormalizedResponse,
    ProviderCapabilities,
    register_adapter,
    register_capabilities,
)

#: Bedrock's fixed InvokeModel request-body version tag (not a Claude model id).
ANTHROPIC_VERSION = "bedrock-2023-05-31"

_INNER = AnthropicAdapter()

# Matches the model segment of a Bedrock InvokeModel URL path, e.g.
# ".../model/anthropic.claude-haiku-4-5-20251001-v1%3A0/invoke" -> the
# (still URL-encoded) "anthropic.claude-haiku-4-5-20251001-v1%3A0".
_URL_MODEL_RE = re.compile(r"/model/([^/]+)/")


class BedrockAdapter:
    """Normalizes/builds Bedrock InvokeModel wire bytes (Claude Messages shape)."""

    name = "bedrock"

    # ── request side ──────────────────────────────────────────────────────────

    def canonicalize_request(self, request_bytes: bytes) -> str:
        # As with the other adapters, this is a hashable handle for the
        # divergence-contract seam (see providers/base.py's docstring). The
        # actual replay-time identity comparison for Bedrock lives in
        # bedrock_transport.py via matcher.bedrock_matcher(), which
        # additionally strips SigV4 signing headers -- out of scope for a
        # body-only hash.
        return sha256_hex(request_bytes)

    def detect_model(self, request_bytes: bytes, request_url: str | None = None) -> str | None:
        """Bedrock's InvokeModel body has no ``model`` field at all — the
        model id is a URL path segment (``/model/{modelId}/invoke``), never
        body content — so this parses it out of ``request_url`` (the tape's
        recorded ``request_urls[i]``, see ``tape.py``) when given, and
        ``urllib.parse.unquote``s the captured segment (Bedrock model ids like
        ``anthropic.claude-haiku-4-5-20251001-v1:0`` are URL-encoded as
        ``...v1%3A0``). Mirrors ``GeminiAdapter.detect_model``'s URL-based
        limitation. ``None`` when ``request_url`` is absent or doesn't match —
        the pre-existing, still-supported no-URL contract."""
        if not request_url:
            return None
        match = _URL_MODEL_RE.search(request_url)
        return unquote(match.group(1)) if match else None

    # ── response side (read) ──────────────────────────────────────────────────

    def parse_response(self, response_bytes: bytes) -> NormalizedResponse:
        """Parse an InvokeModel response (the Anthropic Messages shape) or,
        when recognizable, a Converse API response (``output.message``/
        ``stopReason``/``usage.inputTokens``). Raises on non-JSON input, same
        contract as the sibling adapters."""
        data = json.loads(response_bytes)
        if isinstance(data, dict) and _is_converse_shape(data):
            return _parse_converse(data)
        return _INNER.parse_response(response_bytes)

    def parse_sse(self, response_bytes: bytes) -> dict[str, Any] | None:
        # Bedrock streaming is AWS event-stream binary framing, not SSE -- see
        # module docstring. Never reached on the real streaming path.
        return None

    def tool_use_inputs(
        self, response_bytes: bytes
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        try:
            data = json.loads(response_bytes)
        except Exception:
            return None, []
        if isinstance(data, dict) and _is_converse_shape(data):
            return None, []  # Converse in-place tool-input mutation: out of scope
        return _INNER.tool_use_inputs(response_bytes)

    # ── response side (build) ───────────────────────────────────────────────

    def build_text_response(
        self,
        text: str,
        *,
        model: str | None = None,
        input_tokens: int = 100,
        output_tokens: int = 20,
        message_id: str | None = None,
    ) -> bytes:
        return _INNER.build_text_response(
            text,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            message_id=message_id,
        )

    def build_tool_use_response(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        *,
        model: str | None = None,
        preamble: str = "",
        input_tokens: int = 100,
        output_tokens: int = 30,
        message_id: str | None = None,
    ) -> bytes:
        return _INNER.build_tool_use_response(
            tool_name,
            tool_input,
            model=model,
            preamble=preamble,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            message_id=message_id,
        )

    def mutate_response(self, normalized: NormalizedResponse) -> bytes:
        """Serialize back to the InvokeModel (Anthropic Messages) wire shape —
        the fault re-serialization target for both InvokeModel- and
        Converse-sourced normalized views (Converse *building* is out of
        scope; re-serializing a Converse-sourced view to the InvokeModel shape
        is still a legitimate Bedrock wire form)."""
        return _INNER.mutate_response(normalized)


def build_invoke_model_request(
    messages: list[dict[str, Any]],
    *,
    max_tokens: int = 1024,
    system: str | None = None,
    **extra: Any,
) -> bytes:
    """Build a Bedrock InvokeModel request body: the Anthropic Messages fields
    plus the required ``anthropic_version``, and deliberately WITHOUT a
    top-level ``model`` field (the model id is a URL path segment on Bedrock,
    never a body field — see ``BedrockAdapter.detect_model``)."""
    body: dict[str, Any] = {
        "anthropic_version": ANTHROPIC_VERSION,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if system is not None:
        body["system"] = system
    body.update(extra)
    return json.dumps(body).encode()


def _is_converse_shape(data: dict[str, Any]) -> bool:
    """Converse responses are ``{"output": {"message": {...}}, "stopReason":
    ..., "usage": {...}}`` — distinct from InvokeModel's top-level
    ``content``/``stop_reason``. Checking for ``"output"`` without
    ``"content"`` is enough to disambiguate the two Bedrock response shapes
    from Claude models."""
    return "output" in data and "content" not in data


def _parse_converse(data: dict[str, Any]) -> NormalizedResponse:
    message = (data.get("output") or {}).get("message") or {}
    parts: list[ContentPart] = []
    for block in message.get("content") or []:
        if not isinstance(block, dict):
            continue
        if "text" in block:
            parts.append(ContentPart(type="text", text=block.get("text", "")))
        elif "toolUse" in block:
            tool_use = block.get("toolUse") or {}
            parts.append(
                ContentPart(
                    type="tool_use",
                    tool_name=tool_use.get("name"),
                    tool_id=tool_use.get("toolUseId"),
                    tool_input=tool_use.get("input") or {},
                )
            )
        else:
            parts.append(ContentPart(type="unknown"))
    usage = data.get("usage") or {}
    return NormalizedResponse(
        model=None,  # Converse responses don't echo a model id either
        content=tuple(parts),
        input_tokens=usage.get("inputTokens"),
        output_tokens=usage.get("outputTokens"),
        finish_reason=data.get("stopReason"),
        message_id=None,
    )


register_adapter(BedrockAdapter())
# model_detectable=False: the body never carries a model field (it's a URL
# path segment); detect_model resolves it from an optional request_url
# instead of the body `_CAPABILITIES` describes. parse_response uniquely
# recognizes a second envelope (Converse) in addition to its native
# InvokeModel/Anthropic-Messages shape.
register_capabilities(
    ProviderCapabilities(name="bedrock", model_detectable=False, converse_response=True)
)
