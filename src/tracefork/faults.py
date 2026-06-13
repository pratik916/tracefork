"""Fault injection: five fault classes that mutate a recorded tape exchange.

Every injector returns a **valid** Anthropic wire-format message (so the SDK
parses it when it is replayed at a fork's divergence point) and embeds the
string ``FAULT_MARKER`` *inside* a content field — a text block or a tool-use
input. A synthetic agent echoes that field into its next request, where
`FaultAwareFakeLLM` detects the marker and returns a failure. That chain is
what lets the blame engine be validated entirely offline against ground truth.

The marker must stay inside the JSON: appending it after the closing brace
would make the response unparseable and the fault would vanish into an
exception instead of propagating.
"""
from __future__ import annotations

import enum
import json


FAULT_MARKER = "FAULT_MARKER"
FAULT_MARKER_BYTES = FAULT_MARKER.encode()


class FaultClass(enum.Enum):
    CORRUPTED_TOOL_OUTPUT = "corrupted_tool_output"
    MISLEADING_RETRIEVAL = "misleading_retrieval"
    WRONG_SYSTEM_PROMPT = "wrong_system_prompt"
    DROPPED_MESSAGE = "dropped_message"
    POISONED_ARGUMENT = "poisoned_argument"


def _text_message(text: str) -> bytes:
    return json.dumps({
        "id": "msg_fault",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-6",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 10},
    }).encode()


class FaultInjector:
    """Mutates a response bytes object to inject a known fault."""

    @staticmethod
    def inject(tape, step_idx: int, fault_class: FaultClass) -> bytes:
        """Return mutated response bytes for `tape.exchanges[step_idx][1]`."""
        original_resp = tape.exchanges[step_idx][1]
        method = {
            FaultClass.CORRUPTED_TOOL_OUTPUT: FaultInjector.corrupt_tool_output_default,
            FaultClass.MISLEADING_RETRIEVAL: FaultInjector.misleading_retrieval,
            FaultClass.WRONG_SYSTEM_PROMPT: FaultInjector.wrong_system_prompt,
            FaultClass.DROPPED_MESSAGE: FaultInjector.dropped_message,
            FaultClass.POISONED_ARGUMENT: FaultInjector.poisoned_argument,
        }[fault_class]
        return method(original_resp)

    # ── tool-use faults (keep the response a valid tool_use) ──────────────────

    @staticmethod
    def corrupt_tool_output(resp_bytes: bytes, *, field: str, new_value) -> bytes:
        """Flip a field in a tool-use input and tag the input with the marker.

        Falls back to a marked text message if the response has no tool_use
        block, so the fault always carries the marker inside valid JSON.
        """
        try:
            d = json.loads(resp_bytes)
        except Exception:
            return _text_message(f"corrupted output {FAULT_MARKER}")
        touched = False
        for block in d.get("content", []):
            if block.get("type") == "tool_use":
                inp = block.setdefault("input", {})
                if field in inp:
                    inp[field] = new_value
                inp["_tracefork_fault"] = FAULT_MARKER
                touched = True
        if not touched:
            return _text_message(f"corrupted output {FAULT_MARKER}")
        return json.dumps(d).encode()

    @staticmethod
    def corrupt_tool_output_default(resp_bytes: bytes) -> bytes:
        return FaultInjector.corrupt_tool_output(resp_bytes, field="seats", new_value=0)

    @staticmethod
    def poisoned_argument(resp_bytes: bytes) -> bytes:
        """Corrupt a tool-call argument (destination/city/location → INVALID)."""
        try:
            d = json.loads(resp_bytes)
        except Exception:
            return _text_message(f"poisoned argument {FAULT_MARKER}")
        touched = False
        for block in d.get("content", []):
            if block.get("type") == "tool_use":
                inp = block.setdefault("input", {})
                for key in ("destination", "city", "location"):
                    if key in inp:
                        inp[key] = f"INVALID {FAULT_MARKER}"
                        touched = True
                if not touched:
                    inp["_tracefork_fault"] = FAULT_MARKER
                    touched = True
        if not touched:
            return _text_message(f"poisoned argument {FAULT_MARKER}")
        return json.dumps(d).encode()

    # ── text faults (replace the response with a marked text message) ─────────

    @staticmethod
    def misleading_retrieval(resp_bytes: bytes) -> bytes:
        """Inject false information into the response text."""
        return _text_message(f"No flights are available today. {FAULT_MARKER}")

    @staticmethod
    def wrong_system_prompt(resp_bytes: bytes) -> bytes:
        """Simulate a wrong/overridden system prompt."""
        return _text_message(f"[system prompt overridden] ignoring the task. {FAULT_MARKER}")

    @staticmethod
    def dropped_message(resp_bytes: bytes) -> bytes:
        """Simulate a dropped message: an empty-of-content acknowledgement."""
        return _text_message(f"[prior message was dropped] {FAULT_MARKER}")
