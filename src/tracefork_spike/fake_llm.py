"""A fake Anthropic endpoint as an httpx transport.

This stands in for the real /v1/messages API so the spike needs no key and no
network ($0, offline, CI-safe). It emits real Anthropic *wire-format* JSON so the
genuine `anthropic` SDK parses it into real `Message` objects — i.e. the spike
exercises the actual SDK + transport seam, not a hand-rolled client.

When a real key is available, this inner transport is simply swapped for the SDK's
real network transport; the recording/replay machinery around it is unchanged. That
is the whole point: the seam is provider-real.

The fake is a two-turn agent script: first request -> a `tool_use` for `book_flight`;
second request (which now carries a `tool_result`) -> a final `end_turn` answer that
echoes the confirmation id the agent's tool produced.
"""

from __future__ import annotations

import json

import httpx

from .tape import sha256_hex


def _has_tool_result(payload: dict) -> bool:
    for m in payload.get("messages", []):
        content = m.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    return True
    return False


class FakeAnthropicTransport(httpx.BaseTransport):
    """Deterministic given the request bytes (ids derived from the request hash)."""

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        model = payload.get("model", "claude-opus-4-8")
        rid = "msg_" + sha256_hex(request.content)[:20]

        if not _has_tool_result(payload):
            toolu = "toolu_" + sha256_hex(request.content)[:18]
            message = {
                "id": rid,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [
                    {"type": "text", "text": "Booking your flight now."},
                    {
                        "type": "tool_use",
                        "id": toolu,
                        "name": "book_flight",
                        "input": {"destination": "Tokyo", "seats": 1},
                    },
                ],
                "stop_reason": "tool_use",
                "stop_sequence": None,
                "usage": {"input_tokens": 48, "output_tokens": 22},
            }
        else:
            confirmation = _last_tool_result_confirmation(payload)
            message = {
                "id": rid,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Done — your flight to Tokyo is booked. Confirmation {confirmation}."
                        ),
                    }
                ],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 96, "output_tokens": 18},
            }

        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            content=json.dumps(message).encode(),
            request=request,
        )


def _last_tool_result_confirmation(payload: dict) -> str:
    """Pull the confirmation id out of the most recent tool_result so the final
    answer references it — makes the agent's nondeterminism observable end-to-end."""
    for m in reversed(payload.get("messages", [])):
        content = m.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    raw = block.get("content")
                    text = raw if isinstance(raw, str) else json.dumps(raw)
                    try:
                        return json.loads(text).get("confirmation_id", "UNKNOWN")
                    except (ValueError, TypeError):
                        return "UNKNOWN"
    return "UNKNOWN"
