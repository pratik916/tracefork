"""A tiny tool-using agent built on the real Anthropic SDK.

This is the "agent under recording". It is deliberately nondeterministic: the
`book_flight` tool stamps a wall-clock `booked_at` and a fresh `confirmation_id` on
every run. Those values flow into the *next* request body, so an honest replay can
only be byte-exact if that nondeterminism was captured and virtualized. The agent
reads time/ids exclusively through the injected `NondetSource`.
"""

from __future__ import annotations

import json

import anthropic
import httpx

from .nondet import NondetSource, find_divergence

MODEL = "claude-opus-4-8"

TOOLS = [
    {
        "name": "book_flight",
        "description": "Book a flight to a destination.",
        "input_schema": {
            "type": "object",
            "properties": {
                "destination": {"type": "string"},
                "seats": {"type": "integer"},
            },
            "required": ["destination", "seats"],
        },
    }
]


def make_client(transport: httpx.BaseTransport) -> anthropic.Anthropic:
    return anthropic.Anthropic(
        api_key="sk-ant-offline-fake",
        http_client=httpx.Client(transport=transport),
        max_retries=0,
    )


def _execute_tool(name: str, tool_input: dict, nondet: NondetSource) -> dict:
    if name == "book_flight":
        return {
            "confirmation_id": nondet.new_id("CONF"),  # virtualized nondeterminism
            "booked_at": nondet.now_iso(),             # virtualized nondeterminism
            "destination": tool_input["destination"],
            "seats": tool_input["seats"],
        }
    raise ValueError(f"unknown tool {name!r}")


def run_agent(client: anthropic.Anthropic, nondet: NondetSource) -> dict:
    """Run the agent loop to completion; return its observable trajectory."""
    messages: list[dict] = [{"role": "user", "content": "Book me a flight to Tokyo."}]
    turns: list[dict] = []

    while True:
        try:
            resp = client.messages.create(
                model=MODEL, max_tokens=1024, tools=TOOLS, messages=messages
            )
        except anthropic.APIConnectionError as e:
            # The SDK masks transport-layer exceptions as connection errors; recover
            # a replay DivergenceError so callers see the real cause.
            div = find_divergence(e)
            if div is not None:
                raise div from None
            raise
        turns.append({"id": resp.id, "stop_reason": resp.stop_reason})
        messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason != "tool_use":
            final_text = next((b.text for b in resp.content if b.type == "text"), "")
            return {"final_text": final_text, "turns": turns}

        results = []
        for block in resp.content:
            if block.type == "tool_use":
                out = _execute_tool(block.name, dict(block.input), nondet)
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(out),
                    }
                )
        messages.append({"role": "user", "content": results})
