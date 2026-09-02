"""Tests for the correlational (non-interventional) attribution baseline.

Everything here drives `correlational_baseline.py` through an injected,
deterministic `attribution_fn` -- exactly like `judge.py`'s own offline
testing story for `LLMJudgeOracle` -- so this file makes zero real API
calls and needs no `ANTHROPIC_API_KEY`. `tests/conftest.py`'s autouse
socket guard would fail the suite loudly if anything here reached past
loopback anyway.
"""

from __future__ import annotations

import json

import pytest

from tracefork.correlational_baseline import (
    FAULT_STEP,
    CorrelationalBaselineReport,
    _parse_step,
    correlational_attribute,
    live_attribution_fn,
    render_transcript,
    run_all_fault_classes_correlational,
    run_correlational_baseline,
)
from tracefork.faults import FAULT_MARKER, FaultClass

# ── render_transcript ────────────────────────────────────────────────────────


def test_render_transcript_redacts_fault_marker():
    exchanges = [
        (b'{"role": "user"}', f"corrupted output {FAULT_MARKER}".encode()),
        (b'{"role": "user", "prior": "text"}', b"FAIL - cancelled"),
    ]
    transcript = render_transcript(exchanges)
    assert FAULT_MARKER not in transcript
    assert "corrupted output" in transcript
    assert "FAIL - cancelled" in transcript
    assert "--- step 0 ---" in transcript
    assert "--- step 1 ---" in transcript


def test_render_transcript_numbers_steps_in_order():
    exchanges = [(b"req0", b"resp0"), (b"req1", b"resp1"), (b"req2", b"resp2")]
    transcript = render_transcript(exchanges)
    assert transcript.index("--- step 0 ---") < transcript.index("--- step 1 ---")
    assert transcript.index("--- step 1 ---") < transcript.index("--- step 2 ---")


# ── _parse_step ───────────────────────────────────────────────────────────────


def test_parse_step_prefers_json_step_key():
    raw = 'Some preamble.\n{"step": 1, "rationale": "the tool call is bad"}\nmore text'
    assert _parse_step(raw, n_steps=3) == 1


def test_parse_step_falls_back_to_bare_integer():
    raw = "I believe step 0 is responsible for this."
    assert _parse_step(raw, n_steps=2) == 0


def test_parse_step_abstains_on_unparseable_response():
    assert _parse_step("I'm not sure what happened here.", n_steps=2) is None


def test_parse_step_abstains_on_out_of_range_json_step():
    # step 7 can never be a valid index into a 2-step transcript.
    raw = '{"step": 7, "rationale": "out of range"}'
    assert _parse_step(raw, n_steps=2) is None


def test_parse_step_abstains_on_out_of_range_bare_integer():
    assert _parse_step("step 99 caused it", n_steps=2) is None


def test_parse_step_rejects_json_bool_as_step():
    # bool is an int subclass in Python -- {"step": true} must not parse as
    # step index 1.
    raw = '{"step": true}'
    assert _parse_step(raw, n_steps=2) is None


# ── correlational_attribute ──────────────────────────────────────────────────


def test_correlational_attribute_returns_parsed_step():
    exchanges = [(b"req0", b"resp0"), (b"req1", b"resp1")]

    def fake_judge(prompt: str) -> str:
        assert "step 0" in prompt or "REQUEST: req0" in prompt
        return '{"step": 0, "rationale": "looks wrong here"}'

    assert correlational_attribute(exchanges, attribution_fn=fake_judge) == 0


def test_correlational_attribute_passes_full_transcript_to_judge():
    exchanges = [(b"req-alpha", b"resp-alpha"), (b"req-beta", b"resp-beta")]
    seen_prompts = []

    def recording_judge(prompt: str) -> str:
        seen_prompts.append(prompt)
        return '{"step": 1}'

    correlational_attribute(exchanges, attribution_fn=recording_judge)
    assert len(seen_prompts) == 1
    prompt = seen_prompts[0]
    assert "req-alpha" in prompt
    assert "req-beta" in prompt
    assert "resp-alpha" in prompt
    assert "resp-beta" in prompt


def test_correlational_attribute_abstains_when_judge_is_confused():
    exchanges = [(b"req0", b"resp0"), (b"req1", b"resp1")]
    assert correlational_attribute(exchanges, attribution_fn=lambda p: "no idea") is None


# ── run_correlational_baseline (the fixture, driven end to end offline) ──────


@pytest.mark.parametrize(
    "fault_class",
    [
        FaultClass.CORRUPTED_TOOL_OUTPUT,
        FaultClass.MISLEADING_RETRIEVAL,
        FaultClass.WRONG_SYSTEM_PROMPT,
        FaultClass.DROPPED_MESSAGE,
        FaultClass.POISONED_ARGUMENT,
    ],
)
def test_run_correlational_baseline_perfect_judge_scores_top1(fault_class):
    """An oracle-perfect judge (always guesses FAULT_STEP) must score 1.0 --
    proves the fixture wiring (record -> inject -> fork -> render -> parse)
    is self-consistent for every fault class, independent of how good a real
    LLM judge would actually be."""

    def always_correct(_prompt: str) -> str:
        return json.dumps({"step": FAULT_STEP, "rationale": "always guess the root"})

    report = run_correlational_baseline(fault_class, always_correct, n_runs=3)
    assert isinstance(report, CorrelationalBaselineReport)
    assert report.fault_class == fault_class
    assert report.n_runs == 3
    assert report.top1_correct == 3
    assert report.top1_precision == 1.0
    assert report.n_abstain == 0


def test_run_correlational_baseline_always_wrong_scores_zero():
    def always_wrong(_prompt: str) -> str:
        return json.dumps({"step": 1, "rationale": "guessing the last step"})

    report = run_correlational_baseline(FaultClass.MISLEADING_RETRIEVAL, always_wrong, n_runs=4)
    assert report.top1_correct == 0
    assert report.top1_precision == 0.0
    assert report.n_abstain == 0


def test_run_correlational_baseline_counts_abstentions_separately_from_misses():
    def confused(_prompt: str) -> str:
        return "unparseable nonsense with no digits at all"

    report = run_correlational_baseline(FaultClass.DROPPED_MESSAGE, confused, n_runs=2)
    assert report.top1_correct == 0
    assert report.top1_precision == 0.0
    assert report.n_abstain == 2


def test_run_correlational_baseline_zero_runs_is_zero_precision_not_a_crash():
    report = run_correlational_baseline(
        FaultClass.WRONG_SYSTEM_PROMPT, lambda p: '{"step": 0}', n_runs=0
    )
    assert report.n_runs == 0
    assert report.top1_correct == 0
    assert report.top1_precision == 0.0


def test_run_correlational_baseline_marker_is_never_visible_to_judge():
    """The fault marker must never leak into the judge's prompt -- otherwise
    a judge could trivially grep for it instead of reasoning about the
    fault's actual content (see this module's Fairness note)."""
    seen_prompts = []

    def recording_judge(prompt: str) -> str:
        seen_prompts.append(prompt)
        return '{"step": 0}'

    run_correlational_baseline(FaultClass.POISONED_ARGUMENT, recording_judge, n_runs=1)
    assert len(seen_prompts) == 1
    assert FAULT_MARKER not in seen_prompts[0]


def test_run_correlational_baseline_transcript_carries_every_step():
    """`observed_exchanges` must be the FULL two-exchange run (prefix +
    `Branch.delta_tape`), not just the tail -- a judge that never saw step 0
    could never correctly attribute the (always step-0) fault to it."""
    seen_prompts = []

    def recording_judge(prompt: str) -> str:
        seen_prompts.append(prompt)
        return '{"step": 0}'

    run_correlational_baseline(FaultClass.CORRUPTED_TOOL_OUTPUT, recording_judge, n_runs=1)
    assert len(seen_prompts) == 1
    assert "--- step 0 ---" in seen_prompts[0]
    assert "--- step 1 ---" in seen_prompts[0]


# ── run_all_fault_classes_correlational ──────────────────────────────────────


def test_run_all_fault_classes_correlational_shape():
    def always_correct(_prompt: str) -> str:
        return '{"step": 0}'

    results = run_all_fault_classes_correlational(always_correct, n_runs=2)
    assert set(results.keys()) == {fc.value for fc in FaultClass}
    for entry in results.values():
        assert entry == {
            "top1_precision": 1.0,
            "top1_correct": 2,
            "n_runs": 2,
            "n_abstain": 0,
        }


# ── live_attribution_fn (wiring only, via an injected fake client) ──────────


class _FakeTextBlock:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _FakeMessage:
    def __init__(self, text: str) -> None:
        self.content = [_FakeTextBlock(text)]


class _FakeMessagesAPI:
    def __init__(self, reply_text: str) -> None:
        self._reply_text = reply_text
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeMessage(self._reply_text)


class _FakeAnthropicClient:
    def __init__(self, reply_text: str) -> None:
        self.messages = _FakeMessagesAPI(reply_text)


def test_live_attribution_fn_wires_prompt_and_parses_text_content():
    fake_client = _FakeAnthropicClient(reply_text='{"step": 0, "rationale": "ok"}')
    judge_fn = live_attribution_fn(client=fake_client, model="claude-opus-5")

    result = judge_fn("hello judge")

    assert result == '{"step": 0, "rationale": "ok"}'
    assert len(fake_client.messages.calls) == 1
    call = fake_client.messages.calls[0]
    assert call["model"] == "claude-opus-5"
    assert call["messages"] == [{"role": "user", "content": "hello judge"}]
    assert call["max_tokens"] == 256


def test_live_attribution_fn_ignores_non_text_content_blocks():
    class _ThinkingBlock:
        type = "thinking"
        thinking = "internal reasoning, not the answer"

    class _MixedMessage:
        def __init__(self) -> None:
            self.content = [_ThinkingBlock(), _FakeTextBlock('{"step": 1}')]

    class _MixedMessagesAPI:
        def create(self, **kwargs):
            return _MixedMessage()

    class _MixedClient:
        def __init__(self) -> None:
            self.messages = _MixedMessagesAPI()

    judge_fn = live_attribution_fn(client=_MixedClient())
    assert judge_fn("prompt") == '{"step": 1}'


def test_live_attribution_fn_end_to_end_through_correlational_attribute():
    """The injected fake client's output flows all the way through
    `correlational_attribute`'s parsing -- proving `live_attribution_fn`'s
    return value is a drop-in `attribution_fn`, offline."""
    fake_client = _FakeAnthropicClient(reply_text='{"step": 1, "rationale": "final step"}')
    judge_fn = live_attribution_fn(client=fake_client)

    exchanges = [(b"req0", b"resp0"), (b"req1", b"resp1")]
    assert correlational_attribute(exchanges, attribution_fn=judge_fn) == 1
