"""Gemini adapter — Google Gemini ``generateContent`` wire format behind the seam.

The only place that knows Gemini's JSON shape:
``candidates[].content.parts[]`` (``{"text": ...}`` for text,
``{"functionCall": {"name", "args"}}`` for tool calls, where ``args`` is a JSON
*object*), ``candidates[].finishReason``, token usage under
``usageMetadata.{promptTokenCount, candidatesTokenCount}``, and the model id in
``modelVersion`` (response) / a ``model`` field or the request URL. Streaming is
``streamGenerateContent?alt=sse`` — ``data: {...}`` chunks.

Gemini puts the model in the request URL path
(``/v1beta/models/<model>:generateContent``), not the body, so ``detect_model``
is best-effort from the body's ``model`` field (``None`` when absent — pricing
then falls back to the snapshot default). As elsewhere, the byte contract is
untouched and the ``google-genai`` SDK is not a hard dependency.
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

DEFAULT_GEMINI_MODEL = "gemini-1.5-pro"


class GeminiAdapter:
    """Normalizes/builds Gemini ``generateContent`` wire bytes."""

    name = "gemini"

    # ── request side ──────────────────────────────────────────────────────────

    def canonicalize_request(self, request_bytes: bytes) -> str:
        return sha256_hex(request_bytes)

    def detect_model(self, request_bytes: bytes) -> str | None:
        try:
            body = json.loads(request_bytes)
        except Exception:
            return None
        if not isinstance(body, dict):
            return None
        model = body.get("model")
        if not isinstance(model, str):
            return None
        # Bodies that carry a model often prefix it ("models/gemini-1.5-pro").
        return model.split("/", 1)[1] if model.startswith("models/") else model

    # ── response side (read) ──────────────────────────────────────────────────

    def parse_response(self, response_bytes: bytes) -> NormalizedResponse:
        data = json.loads(response_bytes)  # raises on non-JSON — caller decides fallback
        if not isinstance(data, dict):
            return NormalizedResponse()
        parts: list[ContentPart] = []
        finish_reason: str | None = None
        candidates = data.get("candidates") or []
        if candidates and isinstance(candidates[0], dict):
            candidate = candidates[0]
            finish_reason = candidate.get("finishReason")
            content = candidate.get("content") or {}
            for part in content.get("parts") or []:
                if not isinstance(part, dict):
                    continue
                if "functionCall" in part:
                    fc = part.get("functionCall") or {}
                    parts.append(
                        ContentPart(
                            type="tool_use",
                            tool_name=fc.get("name"),
                            tool_input=dict(fc.get("args") or {}),
                        )
                    )
                elif "text" in part:
                    parts.append(ContentPart(type="text", text=part.get("text", "")))
        usage = data.get("usageMetadata") or {}
        return NormalizedResponse(
            model=data.get("modelVersion") or data.get("model"),
            content=tuple(parts),
            input_tokens=usage.get("promptTokenCount"),
            output_tokens=usage.get("candidatesTokenCount"),
            finish_reason=finish_reason,
            message_id=data.get("responseId"),
        )

    def parse_sse(self, response_bytes: bytes) -> dict[str, Any] | None:
        text = response_bytes.decode(errors="replace")
        for line in text.splitlines():
            if line.startswith("data:"):
                payload = line[len("data:") :].strip()
                if not payload or payload == "[DONE]":
                    continue
                try:
                    parsed = json.loads(payload)
                except Exception:
                    return None
                return parsed if isinstance(parsed, dict) else None
        return None

    def tool_use_inputs(
        self, response_bytes: bytes
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        """Return ``(parsed_response, [functionCall.args dicts])`` for in-place edits.

        Gemini ``args`` is already a nested object, so — like Anthropic — mutating
        the returned dicts and re-``json.dumps``-ing the response is lossless.
        """
        try:
            data = json.loads(response_bytes)
        except Exception:
            return None, []
        if not isinstance(data, dict):
            return None, []
        inputs: list[dict[str, Any]] = []
        for candidate in data.get("candidates") or []:
            if not isinstance(candidate, dict):
                continue
            content = candidate.get("content") or {}
            for part in content.get("parts") or []:
                if isinstance(part, dict) and "functionCall" in part:
                    fc = part["functionCall"]
                    if isinstance(fc, dict):
                        inputs.append(fc.setdefault("args", {}))
        return data, inputs

    # ── response side (build) ─────────────────────────────────────────────────

    def _envelope(
        self,
        *,
        response_id: str,
        model: str,
        parts: list[dict[str, Any]],
        finish_reason: str,
        input_tokens: int,
        output_tokens: int,
    ) -> bytes:
        return json.dumps(
            {
                "candidates": [
                    {
                        "content": {"role": "model", "parts": parts},
                        "finishReason": finish_reason,
                        "index": 0,
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": input_tokens,
                    "candidatesTokenCount": output_tokens,
                    "totalTokenCount": input_tokens + output_tokens,
                },
                "modelVersion": model,
                "responseId": response_id,
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
        model = model or DEFAULT_GEMINI_MODEL
        rid = message_id or "resp_" + sha256_hex((text + model).encode())[:20]
        return self._envelope(
            response_id=rid,
            model=model,
            parts=[{"text": text}],
            finish_reason="STOP",
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
        model = model or DEFAULT_GEMINI_MODEL
        parts: list[dict[str, Any]] = []
        if preamble:
            parts.append({"text": preamble})
        parts.append({"functionCall": {"name": tool_name, "args": tool_input}})
        rid = message_id or "resp_" + sha256_hex((tool_name + model).encode())[:20]
        return self._envelope(
            response_id=rid,
            model=model,
            parts=parts,
            finish_reason="STOP",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def mutate_response(self, normalized: NormalizedResponse) -> bytes:
        """Serialize a normalized view back to Gemini wire bytes."""
        parts: list[dict[str, Any]] = []
        for part in normalized.content:
            if part.type == "text":
                parts.append({"text": part.text or ""})
            elif part.type == "tool_use":
                parts.append(
                    {"functionCall": {"name": part.tool_name, "args": dict(part.tool_input or {})}}
                )
        model = normalized.model or DEFAULT_GEMINI_MODEL
        rid = (
            normalized.message_id
            or "resp_" + sha256_hex(json.dumps(parts, sort_keys=True).encode())[:20]
        )
        finish_reason = normalized.finish_reason or "STOP"
        return self._envelope(
            response_id=rid,
            model=model,
            parts=parts,
            finish_reason=finish_reason,
            input_tokens=normalized.input_tokens or 0,
            output_tokens=normalized.output_tokens or 0,
        )


register_adapter(GeminiAdapter())
# Model lives in the request URL path (see module docstring), not the body,
# so detect_model above is best-effort/usually None.
register_capabilities(
    ProviderCapabilities(name="gemini", model_detectable=False, converse_response=False)
)
