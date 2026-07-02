"""Provider-generic fault injection tests.

The five fault classes must work for OpenAI and Gemini (valid wire JSON with the
marker embedded inside a content field / tool input), while Anthropic output
stays byte-identical to the pre-generic injector.

Offline, zero API keys.
"""

import json

import pytest

from tracefork.constants import SONNET
from tracefork.faults import FAULT_MARKER, FAULT_MARKER_BYTES, FaultClass, FaultInjector
from tracefork.providers import get_adapter
from tracefork.tape import Tape


def _tool_tape(provider: str) -> Tape:
    adapter = get_adapter(provider)
    resp = adapter.build_tool_use_response("check", {"seats": 3, "destination": "Tokyo"})
    tape = Tape()
    tape.append_exchange(b'{"model": "x"}', resp)
    return tape


def _tool_parts(provider: str, mutated: bytes) -> list:
    norm = get_adapter(provider).parse_response(mutated)
    return [p for p in norm.content if p.type == "tool_use"]


# ── all five classes, all three providers: valid JSON + marker inside ────────


@pytest.mark.parametrize("provider", ["anthropic", "openai", "gemini"])
def test_all_fault_classes_valid_json_with_marker(provider):
    tape = _tool_tape(provider)
    for fc in FaultClass:
        mutated = FaultInjector.inject(tape, 0, fc, provider=provider)
        json.loads(mutated)  # raises if not valid JSON
        assert FAULT_MARKER_BYTES in mutated, f"{provider}/{fc} dropped the marker"
        # marker survives a parse -> the adapter can still read the response
        norm = get_adapter(provider).parse_response(mutated)
        assert norm is not None


@pytest.mark.parametrize("provider", ["anthropic", "openai", "gemini"])
def test_corrupt_tool_output_flips_field_and_marks_input(provider):
    tape = _tool_tape(provider)
    mutated = FaultInjector.corrupt_tool_output(
        tape.exchanges[0][1], field="seats", new_value=0, provider=provider
    )
    tool = _tool_parts(provider, mutated)
    assert tool, f"{provider}: tool call lost"
    assert tool[0].tool_input["seats"] == 0
    assert tool[0].tool_input["_tracefork_fault"] == FAULT_MARKER


@pytest.mark.parametrize("provider", ["anthropic", "openai", "gemini"])
def test_poisoned_argument_marks_destination(provider):
    tape = _tool_tape(provider)
    mutated = FaultInjector.poisoned_argument(tape.exchanges[0][1], provider=provider)
    tool = _tool_parts(provider, mutated)
    assert tool[0].tool_input["destination"] == f"INVALID {FAULT_MARKER}"


@pytest.mark.parametrize("provider", ["anthropic", "openai", "gemini"])
def test_text_faults_carry_marker_in_text(provider):
    tape = _tool_tape(provider)
    for fc in (
        FaultClass.MISLEADING_RETRIEVAL,
        FaultClass.WRONG_SYSTEM_PROMPT,
        FaultClass.DROPPED_MESSAGE,
    ):
        mutated = FaultInjector.inject(tape, 0, fc, provider=provider)
        norm = get_adapter(provider).parse_response(mutated)
        assert FAULT_MARKER in norm.first_text()


@pytest.mark.parametrize("provider", ["openai", "gemini"])
def test_tool_fault_on_textless_response_falls_back_to_text(provider):
    adapter = get_adapter(provider)
    text_resp = adapter.build_text_response("no tools here")
    mutated = FaultInjector.corrupt_tool_output(
        text_resp, field="seats", new_value=0, provider=provider
    )
    assert FAULT_MARKER in adapter.parse_response(mutated).first_text()


# ── Anthropic byte-identity regression (golden bytes) ────────────────────────

_GOLDEN_CORRUPTED = (
    b'{"id": "msg_45fc3f6da3693be73130", "type": "message", "role": "assistant", '
    b'"model": "claude-sonnet-4-6", "content": [{"type": "tool_use", '
    b'"id": "toolu_36f37eb1952f3cd1ae", "name": "check", "input": '
    b'{"seats": 0, "destination": "Tokyo", "_tracefork_fault": "FAULT_MARKER"}}], '
    b'"stop_reason": "tool_use", "stop_sequence": null, '
    b'"usage": {"input_tokens": 100, "output_tokens": 30}}'
)
_GOLDEN_POISONED = (
    b'{"id": "msg_45fc3f6da3693be73130", "type": "message", "role": "assistant", '
    b'"model": "claude-sonnet-4-6", "content": [{"type": "tool_use", '
    b'"id": "toolu_36f37eb1952f3cd1ae", "name": "check", "input": '
    b'{"seats": 3, "destination": "INVALID FAULT_MARKER"}}], '
    b'"stop_reason": "tool_use", "stop_sequence": null, '
    b'"usage": {"input_tokens": 100, "output_tokens": 30}}'
)


def test_anthropic_corrupted_bytes_are_unchanged():
    tape = _tool_tape("anthropic")
    assert FaultInjector.inject(tape, 0, FaultClass.CORRUPTED_TOOL_OUTPUT) == _GOLDEN_CORRUPTED


def test_anthropic_poisoned_bytes_are_unchanged():
    tape = _tool_tape("anthropic")
    assert FaultInjector.inject(tape, 0, FaultClass.POISONED_ARGUMENT) == _GOLDEN_POISONED


def test_anthropic_text_fault_bytes_match_direct_build():
    tape = _tool_tape("anthropic")
    mutated = FaultInjector.inject(tape, 0, FaultClass.MISLEADING_RETRIEVAL)
    expected = get_adapter("anthropic").build_text_response(
        f"No flights are available today. {FAULT_MARKER}",
        model=SONNET,
        input_tokens=10,
        output_tokens=10,
        message_id="msg_fault",
    )
    assert mutated == expected
