"""Offline, string-assertion tests for tracefork-bge.51's timeline scrubber
(play/step/slider/click-to-jump), matching test_report.py's existing
convention -- no JS runtime/headless browser exists in this suite."""

from __future__ import annotations

import tempfile
from pathlib import Path

import anthropic
import httpx2

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
        api_key="sk-ant-fake", http_client=httpx2.Client(transport=transport), max_retries=0
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


# ── keyboard-first stepping layer (v1.0.0 readiness item 29) ────────────────
# TF.keys.register was built (a complete keyboard manager) but never called
# once -- the workbench shipped with zero keyboard shortcuts despite every
# action already having a plain JS function. This section proves it's wired
# up, the autoplay interval respects prefers-reduced-motion, and no TF
# member is left defined-and-never-called (the item's own explicit
# acceptance bar).


def _report_content_default() -> str:
    tape = _make_tape(3)
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "report.html"
        generate_report(tape, out)
        return out.read_text()


def test_tf_keys_register_is_called_exactly_once():
    content = _report_content_default()
    assert content.count("TF.keys.register(") == 1


def test_key_bindings_cover_the_documented_keymap():
    content = _report_content_default()
    start = content.index("const KEY_BINDINGS = [")
    bindings_block = content[start : start + 2200]
    # j/k or arrows to step, Home/End, Space to play/pause, d to jump to
    # divergence, 1/2/3 for tabs, [ and ] for rails, ? for the overlay --
    # exactly the keymap the item's own proposed change names.
    for expected in (
        "'j', 'ArrowDown'",
        "'k', 'ArrowUp'",
        "keys: ['Home']",
        "keys: ['End']",
        "keys: [' ']",
        "keys: ['d']",
        "keys: ['1']",
        "keys: ['2']",
        "keys: ['3']",
        "keys: ['[']",
        "keys: [']']",
        "keys: ['?']",
    ):
        assert expected in bindings_block, f"missing keymap entry: {expected!r}"


def test_space_binding_prevents_default_to_avoid_double_toggle():
    """Space's native behaviour is scroll-the-page, and (when the play
    button itself has focus) ALSO a native click -- the handler must call
    preventDefault, or Space would either scroll or double-toggle play."""
    content = _report_content_default()
    start = content.index("keys: [' ']")
    handler_region = content[start : start + 400]
    assert "e.preventDefault();" in handler_region
    assert "playPause();" in handler_region


def test_typing_j_inside_the_fork_textarea_does_not_step():
    """TF.keys' shared listener already ignores INPUT/TEXTAREA/SELECT/
    contenteditable focus -- this is the ONE keydown listener the new
    bindings are registered through (see test_tf_keys_register_is_called_
    exactly_once), so the guard applies to every new binding automatically."""
    content = _report_content_default()
    start = content.index("_install() {")
    install_body = content[start : start + 400]
    assert "t.tagName === 'TEXTAREA'" in install_body


def test_reduced_motion_guard_present_on_report_scrubber_autoplay():
    content = _report_content_default()
    assert "function _prefersReducedMotion()" in content
    assert "'(prefers-reduced-motion: reduce)'" in content
    start = content.index("function playPause(forceStop)")
    body = content[start : start + 600]
    assert "if (_prefersReducedMotion()) return;" in body
    # the guard must come BEFORE the interval is ever created
    assert body.index("if (_prefersReducedMotion()) return;") < body.index("setInterval")


def test_shortcuts_overlay_built_from_the_single_key_bindings_list():
    """One source of truth: the overlay's visible content is built by
    mapping over KEY_BINDINGS, never a second hand-maintained list."""
    content = _report_content_default()
    assert "function toggleShortcutsOverlay(show)" in content
    assert "KEY_BINDINGS.map((b) =>" in content


def test_shortcuts_overlay_uses_tf_h_hyperscript_builder():
    content = _report_content_default()
    assert content.count("TF.h(") >= 5


def test_tf_esc_and_escattr_dead_aliases_were_removed():
    """Regression pin: these were defined and never called (aliases for the
    file's own top-level escape()/escapeAttr(), never referenced as
    TF.esc(...)/TF.escAttr(...) anywhere) -- deleted per the item's own
    'delete any TF member that remains uncalled' instruction rather than
    shipping unused surface."""
    content = _report_content_default()
    assert "esc: escape," not in content
    assert "escAttr: escapeAttr," not in content
    # the real functions themselves are untouched and still used directly
    assert "function escape(s)" in content
    assert "function escapeAttr(s)" in content


def test_no_tf_member_is_defined_and_never_called():
    content = _report_content_default()
    # TF.h, TF.store.get/set, TF.keys.register -- every remaining member has
    # at least one call site.
    assert content.count("TF.h(") >= 1
    assert content.count("TF.store.get(") >= 1
    assert content.count("TF.store.set(") >= 1
    assert content.count("TF.keys.register(") >= 1


# ── session_report.html's subset (files line: "web/report.html,
#    web/session_report.html") ───────────────────────────────────────────────


def test_session_report_html_has_a_keyboard_subset_wired_at_boot():
    content = (Path(__file__).resolve().parent.parent / "web" / "session_report.html").read_text()
    assert "function _boardKeydown(e)" in content
    assert "document.addEventListener('keydown', _boardKeydown);" in content
    for expected in ("stepBoard(1)", "stepBoard(-1)", "setIndex(0)", "playPause();"):
        assert expected in content


def test_session_report_html_reduced_motion_guard_present():
    content = (Path(__file__).resolve().parent.parent / "web" / "session_report.html").read_text()
    assert "function _prefersReducedMotion()" in content
    start = content.index("function playPause(forceStop)")
    body = content[start : start + 600]
    assert "if (_prefersReducedMotion()) return;" in body
    assert body.index("if (_prefersReducedMotion()) return;") < body.index("setInterval")
