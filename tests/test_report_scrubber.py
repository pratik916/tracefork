"""Offline, string-assertion tests for tracefork-bge.51's timeline scrubber
(play/step/slider/click-to-jump), matching test_report.py's existing
convention -- no JS runtime/headless browser exists in this suite."""

from __future__ import annotations

import tempfile
from pathlib import Path

import anthropic
import httpx

from tests.fakes import ScriptedFakeLLM, make_text_response
from tracefork.report import generate_report
from tracefork.tape import Tape
from tracefork.transport import TraceforkTransport


def _make_tape(n_exchanges: int) -> Tape:
    responses = [make_text_response(f"reply {i}") for i in range(n_exchanges)]
    fake = ScriptedFakeLLM(responses)
    tape = Tape(agent_name="test_agent")
    transport = TraceforkTransport("record", tape, fake)
    client = anthropic.Anthropic(
        api_key="sk-ant-fake", http_client=httpx.Client(transport=transport), max_retries=0
    )
    messages: list[dict] = []
    for i in range(n_exchanges):
        messages.append({"role": "user", "content": f"question {i}"})
        resp = client.messages.create(model="claude-sonnet-4-6", max_tokens=100, messages=messages)
        messages.append({"role": "assistant", "content": resp.content[0].text})
    return tape


def test_scrubber_markup_present_for_multi_exchange_tape():
    tape = _make_tape(3)
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "report.html"
        generate_report(tape, out)
        content = out.read_text()
        assert '<input type="range" id="scrubber-slider"' in content
        assert 'id="scrubber-play-btn"' in content
        assert "renderScrubber" in content
        assert "tickPosition" in content


def test_scrubber_tick_count_matches_exchange_count():
    """The JS builds one tick per exchange at render time (data-driven, not a
    fixed count baked into the template) -- assert the rendering function
    that produces them (renderScrubber/tickPosition) iterates data.exchanges,
    proven structurally since no JS runtime executes in this suite."""
    tape = _make_tape(4)
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "report.html"
        generate_report(tape, out)
        content = out.read_text()
        assert "data.exchanges.map((ex, i)" in content
        assert 'class="scrubber-tick"' in content


def test_scrubber_single_exchange_tape_renders_without_div_by_zero():
    """A 1-exchange tape must still render a (disabled-but-present) slider --
    no `i/(n-1)` ZeroDivisionError path in tickPosition's fallback."""
    tape = _make_tape(1)
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "report.html"
        generate_report(tape, out)
        content = out.read_text()
        assert '<input type="range" id="scrubber-slider"' in content
        assert "n <= 1 ? 0 : i / (n - 1)" in content


def test_scrubber_wired_into_boot_sequence():
    tape = _make_tape(2)
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "report.html"
        generate_report(tape, out)
        content = out.read_text()
        assert "renderScrubber(DATA)" in content
        assert "updateScrubberPosition(i)" in content
