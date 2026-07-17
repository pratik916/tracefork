"""OpenAI adapter — the OpenAI Chat Completions wire format behind the seam.

The only place that knows OpenAI's JSON shape: ``choices[].message.content`` (+
``tool_calls[].function.{name,arguments}``, where ``arguments`` is a JSON
*string*), top-level ``model``/``id``, ``choices[].finish_reason``, and
``usage.prompt_tokens``/``completion_tokens``. Streaming is ``data: {...}``
chunk deltas terminated by ``data: [DONE]``.

Like the Anthropic adapter, this never touches the byte contract owned by
``transport.py``/``tape.py``; it derives a neutral ``NormalizedResponse`` and
builds fresh (counterfactual/fault) response bytes. The OpenAI SDK is **not** a
hard dependency — adapters parse raw wire JSON, so offline tests feed synthetic
bytes exactly like the Anthropic fakes.
"""

from __future__ import annotations

import json
from typing import Any

from ..tape import sha256_hex
from .base import (
    ContentPart,
    NormalizedResponse,
    ProviderCapabilities,
    register_adapter,
    register_capabilities,
)

DEFAULT_OPENAI_MODEL = "gpt-4o"


class OpenAIAdapter:
    """Normalizes/builds OpenAI Chat Completions wire bytes."""

    name = "openai"

    # ── request side ──────────────────────────────────────────────────────────

    def canonicalize_request(self, request_bytes: bytes) -> str:
        # As with Anthropic, the raw request bytes ARE the replay identity that
        # transport.py asserts on; this seam only exposes a hashable handle.
        return sha256_hex(request_bytes)

    def detect_model(self, request_bytes: bytes) -> str | None:
        try:
            model = json.loads(request_bytes).get("model")
        except Exception:
            return None
        return model if isinstance(model, str) else None

    # ── response side (read) ──────────────────────────────────────────────────

    def parse_response(self, response_bytes: bytes) -> NormalizedResponse:
        data = json.loads(response_bytes)  # raises on non-JSON — caller decides fallback
        if not isinstance(data, dict):
            return NormalizedResponse()
        parts: list[ContentPart] = []
        finish_reason: str | None = None
        choices = data.get("choices") or []
        if choices and isinstance(choices[0], dict):
            choice = choices[0]
            finish_reason = choice.get("finish_reason")
            message = choice.get("message") or {}
            content = message.get("content")
            if isinstance(content, str) and content:
                parts.append(ContentPart(type="text", text=content))
            elif isinstance(content, list):  # structured content parts
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(ContentPart(type="text", text=block.get("text", "")))
            for call in message.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                fn = call.get("function") or {}
                parts.append(
                    ContentPart(
                        type="tool_use",
                        tool_name=fn.get("name"),
                        tool_id=call.get("id"),
                        tool_input=_decode_arguments(fn.get("arguments")),
                    )
                )
        usage = data.get("usage") or {}
        return NormalizedResponse(
            model=data.get("model"),
            content=tuple(parts),
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            finish_reason=finish_reason,
            message_id=data.get("id"),
        )

    def parse_sse(self, response_bytes: bytes) -> dict[str, Any] | None:
        for payload in _sse_data_payloads(response_bytes):
            try:
                parsed = json.loads(payload)
            except Exception:
                return None
            return parsed if isinstance(parsed, dict) else None
        return None

    def tool_use_inputs(
        self, response_bytes: bytes
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        """Read-only view of tool-call inputs.

        Unlike Anthropic (where a tool input is a nested object mutated in place
        then ``json.dumps``-ed), OpenAI serializes arguments as a JSON *string*,
        so faults round-trip through ``parse_response`` -> mutate parts ->
        ``mutate_response`` rather than this accessor. Returned dicts are decoded
        copies of each ``function.arguments`` string.
        """
        try:
            data = json.loads(response_bytes)
        except Exception:
            return None, []
        if not isinstance(data, dict):
            return None, []
        inputs: list[dict[str, Any]] = []
        for choice in data.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message") or {}
            for call in message.get("tool_calls") or []:
                if isinstance(call, dict):
                    fn = call.get("function") or {}
                    inputs.append(_decode_arguments(fn.get("arguments")))
        return data, inputs

    # ── response side (build) ─────────────────────────────────────────────────

    def _envelope(
        self,
        *,
        message_id: str,
        model: str,
        message: dict[str, Any],
        finish_reason: str,
        input_tokens: int,
        output_tokens: int,
    ) -> bytes:
        return json.dumps(
            {
                "id": message_id,
                "object": "chat.completion",
                "created": 0,
                "model": model,
                "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
                "usage": {
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                },
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
        model = model or DEFAULT_OPENAI_MODEL
        rid = message_id or "chatcmpl-" + sha256_hex((text + model).encode())[:20]
        return self._envelope(
            message_id=rid,
            model=model,
            message={"role": "assistant", "content": text},
            finish_reason="stop",
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
        model = model or DEFAULT_OPENAI_MODEL
        call_id = "call_" + sha256_hex((tool_name + json.dumps(tool_input)).encode())[:18]
        message: dict[str, Any] = {
            "role": "assistant",
            "content": preamble or None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": tool_name, "arguments": json.dumps(tool_input)},
                }
            ],
        }
        rid = message_id or "chatcmpl-" + sha256_hex((tool_name + model).encode())[:20]
        return self._envelope(
            message_id=rid,
            model=model,
            message=message,
            finish_reason="tool_calls",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def mutate_response(self, normalized: NormalizedResponse) -> bytes:
        """Serialize a normalized view back to OpenAI wire bytes."""
        text = "".join(p.text or "" for p in normalized.content if p.type == "text")
        tool_calls: list[dict[str, Any]] = []
        for part in normalized.content:
            if part.type != "tool_use":
                continue
            args = dict(part.tool_input or {})
            cid = (
                part.tool_id
                or "call_" + sha256_hex((str(part.tool_name) + json.dumps(args)).encode())[:18]
            )
            tool_calls.append(
                {
                    "id": cid,
                    "type": "function",
                    "function": {"name": part.tool_name, "arguments": json.dumps(args)},
                }
            )
        message: dict[str, Any] = {"role": "assistant", "content": text or None}
        if tool_calls:
            message["tool_calls"] = tool_calls
        finish_reason = normalized.finish_reason or ("tool_calls" if tool_calls else "stop")
        rid = (
            normalized.message_id
            or "chatcmpl-" + sha256_hex(json.dumps(message, sort_keys=True).encode())[:20]
        )
        return self._envelope(
            message_id=rid,
            model=normalized.model or DEFAULT_OPENAI_MODEL,
            message=message,
            finish_reason=finish_reason,
            input_tokens=normalized.input_tokens or 0,
            output_tokens=normalized.output_tokens or 0,
        )


def _decode_arguments(arguments: Any) -> dict[str, Any]:
    """OpenAI tool arguments are a JSON string; decode to a dict (``{}`` on error)."""
    if isinstance(arguments, dict):
        return dict(arguments)
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _sse_data_payloads(response_bytes: bytes):
    """Yield ``data:`` payloads from an SSE stream, skipping the ``[DONE]`` marker."""
    text = response_bytes.decode(errors="replace")
    for line in text.splitlines():
        if line.startswith("data:"):
            payload = line[len("data:") :].strip()
            if payload and payload != "[DONE]":
                yield payload


register_adapter(OpenAIAdapter())
register_capabilities(
    ProviderCapabilities(name="openai", model_detectable=True, converse_response=False)
)
