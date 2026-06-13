"""Fault injection + self-validation tests — all offline, zero API keys."""
import json

import anthropic
import httpx

from tracefork.tape import Tape
from tracefork.transport import TraceforkTransport
from tracefork.faults import FaultClass, FaultInjector, FAULT_MARKER_BYTES
from tracefork.validate import ValidationRunner, run_all_fault_classes
from tests.fakes import (
    ScriptedFakeLLM, FaultAwareFakeLLM,
    make_text_response, make_tool_use_response,
)


SUCCESS_TEXT = "SUCCESS — booking confirmed"
FAIL_TEXT    = "FAIL — no flights available"
SUCCESS_RESP = make_text_response(SUCCESS_TEXT)
FAIL_RESP    = make_text_response(FAIL_TEXT)
TOOL_RESP    = make_tool_use_response("check_availability", {"seats": 3, "destination": "Tokyo"})


def _record_tool_use_tape() -> Tape:
    """A 2-exchange tape: turn1=tool_use, turn2=success."""
    fake = ScriptedFakeLLM([TOOL_RESP, SUCCESS_RESP])
    tape = Tape()
    transport = TraceforkTransport("record", tape, fake)
    client = anthropic.Anthropic(
        api_key="sk-ant-fake",
        http_client=httpx.Client(transport=transport),
        max_retries=0,
    )
    client.messages.create(
        model="claude-sonnet-4-6", max_tokens=100,
        messages=[{"role": "user", "content": "book a flight to Tokyo"}],
    )
    client.messages.create(
        model="claude-sonnet-4-6", max_tokens=100,
        messages=[
            {"role": "user", "content": "book a flight to Tokyo"},
            {"role": "assistant", "content": "checking…"},
            {"role": "user", "content": "confirm"},
        ],
    )
    return tape


# ── FaultInjector ─────────────────────────────────────────────────────────

def test_fault_class_enum_has_five_members():
    assert len(list(FaultClass)) == 5


def test_corrupt_tool_output_flips_numeric():
    tape = _record_tool_use_tape()
    original = tape.exchanges[0][1]
    mutated = FaultInjector.corrupt_tool_output(original, field="seats", new_value=0)
    d = json.loads(mutated)  # must remain valid JSON
    seats = [b["input"]["seats"] for b in d["content"] if b.get("type") == "tool_use"]
    assert seats == [0]


def test_all_injected_faults_are_valid_json_with_marker():
    """Regression: every fault must stay parseable AND carry the marker inside.

    Appending the marker after the JSON (the obvious bug) would crash the SDK
    on replay and silently disable the fault — so we assert both properties.
    """
    tape = _record_tool_use_tape()
    for fc in FaultClass:
        mutated = FaultInjector.inject(tape, 0, fc)
        json.loads(mutated)  # raises if the response is not valid JSON
        assert FAULT_MARKER_BYTES in mutated, f"{fc} dropped the fault marker"


# ── FaultAwareFakeLLM (the CI-layer fault → failure mechanism) ─────────────

def _client(transport: TraceforkTransport) -> anthropic.Anthropic:
    return anthropic.Anthropic(
        api_key="sk-ant-fake",
        http_client=httpx.Client(transport=transport),
        max_retries=0,
    )


def test_fault_aware_fake_returns_failure_on_marker():
    marker = b"FAULT_MARKER"
    t1 = TraceforkTransport("record", Tape(), FaultAwareFakeLLM(
        normal_responses=[SUCCESS_RESP], fault_responses=[FAIL_RESP], fault_marker=marker))
    r = _client(t1).messages.create(
        model="claude-sonnet-4-6", max_tokens=100,
        messages=[{"role": "user", "content": "book normally"}],
    )
    assert r.content[0].text == SUCCESS_TEXT

    t2 = TraceforkTransport("record", Tape(), FaultAwareFakeLLM(
        normal_responses=[SUCCESS_RESP], fault_responses=[FAIL_RESP], fault_marker=marker))
    r2 = _client(t2).messages.create(
        model="claude-sonnet-4-6", max_tokens=100,
        messages=[{"role": "user", "content": "FAULT_MARKER inject fault here"}],
    )
    assert r2.content[0].text == FAIL_TEXT


# ── self-validation: blame fingers the injected step ───────────────────────

def test_validation_runner_fingers_fault_step():
    """With deterministic fakes the injected step is ranked #1 every time, and
    the no-fault control stays flat."""
    report = ValidationRunner(FaultClass.CORRUPTED_TOOL_OUTPUT, k=3, n_runs=5).run()
    assert report.fault_class == FaultClass.CORRUPTED_TOOL_OUTPUT
    assert report.n_runs == 5
    assert report.top1_precision == 1.0
    assert report.negative_control_max_flip == 0.0


def test_all_fault_classes_validate():
    """All five fault classes clear the 0.7 precision bar with a flat control."""
    results = run_all_fault_classes(k=2, n_runs=3)
    assert len(results) == 5
    for fc, data in results.items():
        assert data["top1_precision"] >= 0.7, f"{fc}: precision {data['top1_precision']}"
        assert data["negative_control_max_flip"] < 0.3, f"{fc}: control too high"
