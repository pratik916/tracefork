"""effects.py -- tool-effect extraction + cross-branch conflict tests.

Offline, no fork/store dependency: tapes are built directly with
`wire.make_tool_use_response` (LLM `tool_use` blocks) and
`tools.make_tool_call_frame`/`make_result_frame` (JSON-RPC tool frames).
"""

from __future__ import annotations

from tracefork.effects import (
    EFFECT_EXTRACTOR_REGISTRY,
    ConflictReport,
    Effect,
    diff_effects,
    extract_effects,
    register_effect_extractor,
)
from tracefork.tape import Tape
from tracefork.tools import make_result_frame, make_tool_call_frame
from tracefork.wire import make_tool_use_response

# ── extract_effects: llm_tool_use source ────────────────────────────────────


def test_extract_effects_finds_one_per_tool_use_block():
    tape = Tape()
    tape.append_exchange(b"{}", make_tool_use_response("read_file", {"path": "a.txt"}))
    tape.append_exchange(b"{}", make_tool_use_response("read_file", {"path": "b.txt"}))

    effects = extract_effects(tape)

    assert len(effects) == 2
    assert all(isinstance(e, Effect) for e in effects)
    assert all(e.source == "llm_tool_use" for e in effects)
    assert [e.index for e in effects] == [0, 1]
    assert effects[0].tool_name == "read_file"
    assert effects[0].resource == "a.txt"
    assert effects[1].resource == "b.txt"


def test_extract_effects_ignores_text_only_exchanges():
    tape = Tape()
    tape.append_exchange(b"{}", b'{"type": "message", "content": [{"type": "text", "text": "hi"}]}')

    assert extract_effects(tape) == ()


# ── extract_effects: tool_frame source ──────────────────────────────────────


def test_extract_effects_finds_one_per_tool_frame():
    tape = Tape()
    tape.append_tool_exchange(
        make_tool_call_frame(1, "fetch_url", {"url": "http://x/1"}),
        make_result_frame(1, {"ok": True}),
    )
    tape.append_tool_exchange(
        make_tool_call_frame(2, "fetch_url", {"url": "http://x/2"}),
        make_result_frame(2, {"ok": True}),
    )

    effects = extract_effects(tape)

    assert len(effects) == 2
    assert all(e.source == "tool_frame" for e in effects)
    assert [e.index for e in effects] == [0, 1]
    assert effects[0].tool_name == "fetch_url"
    assert effects[0].resource == "http://x/1"
    assert effects[1].resource == "http://x/2"


def test_extract_effects_combines_both_sources():
    tape = Tape()
    tape.append_exchange(b"{}", make_tool_use_response("read_file", {"path": "a.txt"}))
    tape.append_tool_exchange(
        make_tool_call_frame(1, "fetch_url", {"url": "http://x/1"}),
        make_result_frame(1, {"ok": True}),
    )

    effects = extract_effects(tape)

    assert {e.source for e in effects} == {"llm_tool_use", "tool_frame"}
    assert len(effects) == 2


# ── default resource key-probe ───────────────────────────────────────────────


def test_default_extractor_pulls_path_key():
    tape = Tape()
    tape.append_exchange(
        b"{}", make_tool_use_response("write", {"path": "out.txt", "content": "hi"})
    )

    effect = extract_effects(tape)[0]

    assert effect.resource == "out.txt"
    assert effect.resource_is_fallback is False


def test_default_extractor_pulls_url_key():
    tape = Tape()
    tape.append_exchange(b"{}", make_tool_use_response("web", {"url": "http://example"}))

    effect = extract_effects(tape)[0]

    assert effect.resource == "http://example"
    assert effect.resource_is_fallback is False


# ── register_effect_extractor ────────────────────────────────────────────────


def test_register_effect_extractor_overrides_resolution():
    def extractor(arguments):
        return f"db:{arguments['table']}"

    register_effect_extractor("query_db", extractor)
    try:
        tape = Tape()
        tape.append_exchange(
            b"{}", make_tool_use_response("query_db", {"table": "users", "path": "ignored"})
        )

        effect = extract_effects(tape)[0]

        assert effect.resource == "db:users"
        assert effect.resource_is_fallback is False
    finally:
        EFFECT_EXTRACTOR_REGISTRY.pop("query_db", None)


def test_registered_extractor_returning_none_falls_back_to_default_key_probe():
    def extractor(arguments):
        return None

    register_effect_extractor("maybe_tool", extractor)
    try:
        tape = Tape()
        tape.append_exchange(b"{}", make_tool_use_response("maybe_tool", {"path": "fallback.txt"}))

        effect = extract_effects(tape)[0]

        assert effect.resource == "fallback.txt"
        assert effect.resource_is_fallback is False
    finally:
        EFFECT_EXTRACTOR_REGISTRY.pop("maybe_tool", None)


# ── canonical-JSON fallback ───────────────────────────────────────────────────


def test_unregistered_tool_no_matched_key_uses_canonical_json_fallback():
    tape = Tape()
    tape.append_exchange(b"{}", make_tool_use_response("compute", {"a": 1, "b": 2}))
    # Same arguments, different key order -- must canonicalize equal.
    tape.append_exchange(b"{}", make_tool_use_response("compute", {"b": 2, "a": 1}))
    # Different arguments -- must canonicalize different.
    tape.append_exchange(b"{}", make_tool_use_response("compute", {"a": 9, "b": 2}))

    effects = extract_effects(tape)

    assert all(e.resource_is_fallback for e in effects)
    assert effects[0].resource == effects[1].resource
    assert effects[0].resource != effects[2].resource


# ── diff_effects / ConflictReport ────────────────────────────────────────────


def test_diff_effects_has_conflict_true_when_same_tool_and_resource():
    tape_a = Tape()
    tape_a.append_exchange(b"{}", make_tool_use_response("read_file", {"path": "shared.txt"}))
    tape_b = Tape()
    tape_b.append_exchange(b"{}", make_tool_use_response("read_file", {"path": "shared.txt"}))

    report = diff_effects(tape_a, tape_b)

    assert isinstance(report, ConflictReport)
    assert report.has_conflict is True
    assert len(report.overlaps) == 1
    overlap = report.overlaps[0]
    assert overlap.tool_name == "read_file"
    assert overlap.resource == "shared.txt"
    assert overlap.effect_a.source == "llm_tool_use"
    assert overlap.effect_b.source == "llm_tool_use"


def test_diff_effects_has_conflict_false_when_resources_differ():
    tape_a = Tape()
    tape_a.append_exchange(b"{}", make_tool_use_response("read_file", {"path": "a.txt"}))
    tape_b = Tape()
    tape_b.append_exchange(b"{}", make_tool_use_response("read_file", {"path": "b.txt"}))

    report = diff_effects(tape_a, tape_b)

    assert report.has_conflict is False
    assert report.overlaps == ()
    assert len(report.effects_a) == 1
    assert len(report.effects_b) == 1


def test_diff_effects_no_conflict_when_same_resource_different_tool():
    tape_a = Tape()
    tape_a.append_exchange(b"{}", make_tool_use_response("read_file", {"path": "shared.txt"}))
    tape_b = Tape()
    tape_b.append_exchange(b"{}", make_tool_use_response("write_file", {"path": "shared.txt"}))

    report = diff_effects(tape_a, tape_b)

    assert report.has_conflict is False


def test_diff_effects_across_tool_frame_source():
    tape_a = Tape()
    tape_a.append_tool_exchange(
        make_tool_call_frame(1, "fetch_url", {"url": "http://x/shared"}),
        make_result_frame(1, {"ok": True}),
    )
    tape_b = Tape()
    tape_b.append_tool_exchange(
        make_tool_call_frame(1, "fetch_url", {"url": "http://x/shared"}),
        make_result_frame(1, {"ok": True}),
    )

    report = diff_effects(tape_a, tape_b)

    assert report.has_conflict is True
    assert report.overlaps[0].effect_a.source == "tool_frame"
