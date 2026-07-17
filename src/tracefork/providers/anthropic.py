"""Anthropic adapter — the Anthropic Messages-API wire format behind the seam.

This is the *only* place that knows Anthropic's JSON shape (``content[]`` blocks
with ``type`` in ``{text, tool_use}``, top-level ``model``/``stop_reason``,
``usage.input_tokens``/``output_tokens``, ``data: `` SSE framing). The wire
builders, blame, faults, and the report route through it so nothing else assumes
one provider's schema. Byte output is intentionally identical to the pre-seam
``wire.py`` builders (record/replay bit-exactness is unaffected).
"""

from __future__ import annotations

import json
from typing import Any

from ..constants import SONNET
from ..tape import sha256_hex
from .base import (
    ContentPart,
    NormalizedResponse,
    ProviderCapabilities,
    register_adapter,
    register_capabilities,
)


class AnthropicAdapter:
    """Normalizes/builds Anthropic Messages-API wire bytes."""

    name = "anthropic"

    # ── request side ──────────────────────────────────────────────────────────

    def canonicalize_request(self, request_bytes: bytes) -> str:
        # Anthropic request bytes are already canonical: the sha256 IS the identity
        # transport.py asserts on at replay. This seam exists for the divergence-
        # contract work; it does not change transport's byte-for-byte matching.
        return sha256_hex(request_bytes)

    def detect_model(self, request_bytes: bytes, request_url: str | None = None) -> str | None:
        # request_url is unused: the body already carries a real "model" field.
        try:
            return json.loads(request_bytes).get("model")
        except Exception:
            return None

    # ── response side (read) ──────────────────────────────────────────────────

    def parse_response(self, response_bytes: bytes) -> NormalizedResponse:
        data = json.loads(response_bytes)  # raises on non-JSON — caller decides fallback
        if not isinstance(data, dict):
            return NormalizedResponse()
        parts: list[ContentPart] = []
        for block in data.get("content", []):
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                parts.append(ContentPart(type="text", text=block.get("text", "")))
            elif btype == "tool_use":
                parts.append(
                    ContentPart(
                        type="tool_use",
                        tool_name=block.get("name"),
                        tool_id=block.get("id"),
                        tool_input=block.get("input") or {},
                    )
                )
            else:
                parts.append(ContentPart(type=btype or "unknown"))
        usage = data.get("usage") or {}
        return NormalizedResponse(
            model=data.get("model"),
            content=tuple(parts),
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            finish_reason=data.get("stop_reason"),
            message_id=data.get("id"),
        )

    def parse_sse(self, response_bytes: bytes) -> dict[str, Any] | None:
        text = response_bytes.decode(errors="replace")
        data_lines = [
            line[6:]
            for line in text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        if not data_lines:
            return None
        try:
            parsed = json.loads(data_lines[0])
        except Exception:
            return None
        return parsed if isinstance(parsed, dict) else None

    def tool_use_inputs(
        self, response_bytes: bytes
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        try:
            data = json.loads(response_bytes)
        except Exception:
            return None, []
        if not isinstance(data, dict):
            return None, []
        inputs: list[dict[str, Any]] = []
        for block in data.get("content", []):
            if isinstance(block, dict) and block.get("type") == "tool_use":
                inputs.append(block.setdefault("input", {}))
        return data, inputs

    # ── response side (build) ─────────────────────────────────────────────────

    def _envelope(
        self,
        *,
        message_id: str,
        model: str,
        content: list[dict[str, Any]],
        stop_reason: str,
        input_tokens: int,
        output_tokens: int,
    ) -> bytes:
        return json.dumps(
            {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": content,
                "stop_reason": stop_reason,
                "stop_sequence": None,
                "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
            }
        ).encode()

    def build_text_response(
        self,
        text: str,
        *,
        model: str | None = None,
        input_tokens: int = 100,
        output_tokens: int = 20,
        message_id: str | None = None,
    ) -> bytes:
        model = model or SONNET
        rid = message_id or "msg_" + sha256_hex((text + model).encode())[:20]
        return self._envelope(
            message_id=rid,
            model=model,
            content=[{"type": "text", "text": text}],
            stop_reason="end_turn",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
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
        model = model or SONNET
        content: list[dict[str, Any]] = []
        if preamble:
            content.append({"type": "text", "text": preamble})
        toolu_id = "toolu_" + sha256_hex((tool_name + json.dumps(tool_input)).encode())[:18]
        content.append({"type": "tool_use", "id": toolu_id, "name": tool_name, "input": tool_input})
        rid = message_id or "msg_" + sha256_hex((tool_name + model).encode())[:20]
        return self._envelope(
            message_id=rid,
            model=model,
            content=content,
            stop_reason="tool_use",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def mutate_response(self, normalized: NormalizedResponse) -> bytes:
        """Rebuild Anthropic wire bytes from a normalized view.

        A neutral counterfactual-builder seam (used by future multi-provider blame
        coalitions); faults use the lossless ``tool_use_inputs`` path instead.
        """
        content: list[dict[str, Any]] = []
        for part in normalized.content:
            if part.type == "text":
                content.append({"type": "text", "text": part.text or ""})
            elif part.type == "tool_use":
                block: dict[str, Any] = {"type": "tool_use"}
                if part.tool_id is not None:
                    block["id"] = part.tool_id
                block["name"] = part.tool_name
                block["input"] = dict(part.tool_input or {})
                content.append(block)
        model = normalized.model or SONNET
        rid = (
            normalized.message_id
            or "msg_" + sha256_hex(json.dumps(content, sort_keys=True).encode())[:20]
        )
        has_tool = any(p.type == "tool_use" for p in normalized.content)
        stop_reason = normalized.finish_reason or ("tool_use" if has_tool else "end_turn")
        return self._envelope(
            message_id=rid,
            model=model,
            content=content,
            stop_reason=stop_reason,
            input_tokens=normalized.input_tokens or 0,
            output_tokens=normalized.output_tokens or 0,
        )


register_adapter(AnthropicAdapter())
register_capabilities(
    ProviderCapabilities(name="anthropic", model_detectable=True, converse_response=False)
)
