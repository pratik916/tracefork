"""Fault injection: five fault classes that mutate a recorded tape exchange.

Every injector returns a **valid** wire-format message for the target provider
(so the SDK parses it when it is replayed at a fork's divergence point) and
embeds the string ``FAULT_MARKER`` *inside* a content field — a text block or a
tool-call input. A synthetic agent echoes that field into its next request,
where ``FaultAwareFakeLLM`` detects the marker and returns a failure. That chain
is what lets the blame engine be validated entirely offline against ground truth.

The marker must stay inside the JSON: appending it after the closing brace would
make the response unparseable and the fault would vanish into an exception
instead of propagating.

**Provider-generic.** Injection routes through the registered provider adapter:
text faults via ``build_text_response``, tool faults by normalizing the response
(``parse_response``), mutating the neutral content parts, and re-serializing to
the provider's wire format (``mutate_response``). This makes the five classes
work for OpenAI and Gemini too. For the default ``"anthropic"`` provider the
resulting bytes are **identical** to the pre-generic injector output (verified by
``tests/test_faults.py`` / ``tests/test_provider_faults.py``).
"""

from __future__ import annotations

import dataclasses
import enum
from collections.abc import Callable
from typing import Any

from .providers import NormalizedResponse, get_adapter

FAULT_MARKER = "FAULT_MARKER"
FAULT_MARKER_BYTES = FAULT_MARKER.encode()


class FaultClass(enum.Enum):
    CORRUPTED_TOOL_OUTPUT = "corrupted_tool_output"
    MISLEADING_RETRIEVAL = "misleading_retrieval"
    WRONG_SYSTEM_PROMPT = "wrong_system_prompt"
    DROPPED_MESSAGE = "dropped_message"
    POISONED_ARGUMENT = "poisoned_argument"


def _text_message(text: str, provider: str = "anthropic") -> bytes:
    """A minimal marked text response, built through the provider adapter.

    ``model=None`` lets each adapter pick its own default (Anthropic -> Sonnet),
    so the Anthropic bytes are unchanged from the pre-generic injector.
    """
    return get_adapter(provider).build_text_response(
        text, model=None, input_tokens=10, output_tokens=10, message_id="msg_fault"
    )


def _normalize(resp_bytes: bytes, provider: str) -> NormalizedResponse | None:
    """Parse ``resp_bytes`` via the adapter, or ``None`` if it is not parseable."""
    try:
        return get_adapter(provider).parse_response(resp_bytes)
    except Exception:
        return None


def _has_tool(norm: NormalizedResponse | None) -> bool:
    return norm is not None and any(p.type == "tool_use" for p in norm.content)


def _rewrite_tool_inputs(
    norm: NormalizedResponse, mutate: Callable[[dict[str, Any]], None]
) -> NormalizedResponse:
    """Apply ``mutate`` to a copy of each tool-call input; return the new view."""
    new_parts = []
    for part in norm.content:
        if part.type == "tool_use":
            tool_input = dict(part.tool_input or {})
            mutate(tool_input)
            new_parts.append(dataclasses.replace(part, tool_input=tool_input))
        else:
            new_parts.append(part)
    return dataclasses.replace(norm, content=tuple(new_parts))


class FaultInjector:
    """Mutates a response bytes object to inject a known fault."""

    @staticmethod
    def inject(
        tape, step_idx: int, fault_class: FaultClass, *, provider: str = "anthropic"
    ) -> bytes:
        """Return mutated response bytes for `tape.exchanges[step_idx][1]`."""
        original_resp = tape.exchanges[step_idx][1]
        method = {
            FaultClass.CORRUPTED_TOOL_OUTPUT: FaultInjector.corrupt_tool_output_default,
            FaultClass.MISLEADING_RETRIEVAL: FaultInjector.misleading_retrieval,
            FaultClass.WRONG_SYSTEM_PROMPT: FaultInjector.wrong_system_prompt,
            FaultClass.DROPPED_MESSAGE: FaultInjector.dropped_message,
            FaultClass.POISONED_ARGUMENT: FaultInjector.poisoned_argument,
        }[fault_class]
        return method(original_resp, provider=provider)

    # ── tool-use faults (keep the response a valid tool call) ─────────────────

    @staticmethod
    def corrupt_tool_output(
        resp_bytes: bytes, *, field: str, new_value, provider: str = "anthropic"
    ) -> bytes:
        """Flip a field in a tool-call input and tag the input with the marker.

        Falls back to a marked text message if the response has no tool call, so
        the fault always carries the marker inside valid JSON.
        """
        norm = _normalize(resp_bytes, provider)
        if not _has_tool(norm):
            return _text_message(f"corrupted output {FAULT_MARKER}", provider)
        assert norm is not None

        def mutate(tool_input: dict[str, Any]) -> None:
            if field in tool_input:
                tool_input[field] = new_value
            tool_input["_tracefork_fault"] = FAULT_MARKER

        return get_adapter(provider).mutate_response(_rewrite_tool_inputs(norm, mutate))

    @staticmethod
    def corrupt_tool_output_default(resp_bytes: bytes, provider: str = "anthropic") -> bytes:
        return FaultInjector.corrupt_tool_output(
            resp_bytes, field="seats", new_value=0, provider=provider
        )

    @staticmethod
    def poisoned_argument(resp_bytes: bytes, provider: str = "anthropic") -> bytes:
        """Corrupt a tool-call argument (destination/city/location -> INVALID)."""
        norm = _normalize(resp_bytes, provider)
        if not _has_tool(norm):
            return _text_message(f"poisoned argument {FAULT_MARKER}", provider)
        assert norm is not None
        # Shared flag across tool calls mirrors the original injector semantics.
        touched = False

        def mutate(tool_input: dict[str, Any]) -> None:
            nonlocal touched
            for key in ("destination", "city", "location"):
                if key in tool_input:
                    tool_input[key] = f"INVALID {FAULT_MARKER}"
                    touched = True
            if not touched:
                tool_input["_tracefork_fault"] = FAULT_MARKER
                touched = True

        return get_adapter(provider).mutate_response(_rewrite_tool_inputs(norm, mutate))

    # ── text faults (replace the response with a marked text message) ─────────

    @staticmethod
    def misleading_retrieval(resp_bytes: bytes, provider: str = "anthropic") -> bytes:
        """Inject false information into the response text."""
        return _text_message(f"No flights are available today. {FAULT_MARKER}", provider)

    @staticmethod
    def wrong_system_prompt(resp_bytes: bytes, provider: str = "anthropic") -> bytes:
        """Simulate a wrong/overridden system prompt."""
        return _text_message(
            f"[system prompt overridden] ignoring the task. {FAULT_MARKER}", provider
        )

    @staticmethod
    def dropped_message(resp_bytes: bytes, provider: str = "anthropic") -> bytes:
        """Simulate a dropped message: an empty-of-content acknowledgement."""
        return _text_message(f"[prior message was dropped] {FAULT_MARKER}", provider)
