"""Offline, string-assertion test for the Timeline panel's selected-step
highlight (v1.0.0 readiness item 10), matching test_report_scrubber.py's
existing convention -- no JS runtime/headless browser exists in this suite.

Bug: `selectExchange` toggled `.active` via a BARE, unscoped
`document.querySelector('[data-i="N"]')`. The scrubber renders one
`.scrubber-tick[data-i="N"]` per exchange into `#timeline-scrubber`, which
sits BEFORE `#timeline-content` (whose `.exchange-item[data-i="N"]` rows are
what `.exchange-item.active`'s CSS actually highlights) in document order.
`querySelector` returns the FIRST match in document order, so the unscoped
selector always toggled `.active` on the scrubber tick -- which already gets
its own `.active` state from `updateScrubberPosition`'s
`querySelectorAll('.scrubber-tick')` loop -- and NEVER on the exchange-item
row, so the Timeline list's selected-step highlight could never render.
"""

from __future__ import annotations

import re
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


def _report_content() -> str:
    tape = _make_tape(3)
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "report.html"
        generate_report(tape, out)
        return out.read_text()


def test_select_exchange_scopes_active_toggle_to_timeline_content():
    content = _report_content()
    # both the add and remove sites must be scoped to #timeline-content,
    # never a bare `[data-i="..."]` that could match the scrubber tick first
    assert 'document.querySelector(`#timeline-content [data-i="${_selected}"]`)' in content
    assert 'document.querySelector(`#timeline-content [data-i="${i}"]`)' in content


def test_select_exchange_never_uses_an_unscoped_data_i_selector():
    content = _report_content()
    start = content.index("function selectExchange(i)")
    select_fn = content[start : start + 700]
    # no querySelector call in this function may target `[data-i=...]`
    # without a preceding ancestor-scoping selector
    for match in re.finditer(r"querySelector\(`([^`]*)`\)", select_fn):
        selector = match.group(1)
        if "data-i=" in selector:
            assert selector.strip().startswith("#timeline-content"), (
                f"unscoped data-i selector would match the scrubber tick first: {selector!r}"
            )
