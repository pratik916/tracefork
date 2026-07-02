"""Structured divergence diagnostics tests — offline, no API keys.

Two guarantees are load-bearing here (mirrors `test_matcher.py`'s framing):

1. `diff_json`/`diff_request_bytes` correctly identify the changed field(s)
   between a recorded and a live request body.
2. `diagnose()` — built on top of `RequestMatcher.stored_request` — excludes
   any field a `CanonicalizingMatcher` normalizes away (rotating auth,
   idempotency keys, ...) from the diff, so a tolerated volatile-field
   normalization is never reported as a real divergence, while a genuine
   content change alongside it still is.
"""

import httpx

from tracefork.divergence import (
    MISSING,
    DivergenceDiagnostic,
    FieldDiff,
    diagnose,
    diagnostic_to_dict,
    diff_json,
    diff_request_bytes,
)
from tracefork.matcher import IDENTITY_MATCHER, redacting_matcher
from tracefork.tape import Tape

# ── diff_json / diff_request_bytes ──────────────────────────────────────────


def test_diff_json_identifies_the_changed_field():
    recorded = {"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": "hi"}]}
    live = {"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": "bye"}]}
    diffs = diff_json(recorded, live)
    assert diffs == [FieldDiff("$.messages[0].content", "hi", "bye")]


def test_diff_json_empty_when_semantically_equal():
    recorded = {"a": 1, "b": {"c": 2}}
    live = {"a": 1, "b": {"c": 2}}
    assert diff_json(recorded, live) == []


def test_diff_json_reports_multiple_changed_fields():
    recorded = {"max_tokens": 100, "model": "claude-sonnet-4-6"}
    live = {"max_tokens": 200, "model": "claude-opus-4-8"}
    diffs = diff_json(recorded, live)
    paths = {d.path for d in diffs}
    assert paths == {"$.max_tokens", "$.model"}


def test_diff_json_marks_missing_key_with_sentinel():
    recorded = {"a": 1}
    live = {"a": 1, "b": 2}
    diffs = diff_json(recorded, live)
    assert diffs == [FieldDiff("$.b", MISSING, 2)]


def test_diff_json_handles_lists_positionally():
    recorded = {"tags": ["x", "y"]}
    live = {"tags": ["x", "z"]}
    diffs = diff_json(recorded, live)
    assert diffs == [FieldDiff("$.tags[1]", "y", "z")]


def test_diff_request_bytes_non_semantic_whitespace_is_not_a_diff():
    """Two JSON bodies that differ only in whitespace/key order parse to the
    same structure — json.loads already normalizes that away."""
    recorded = b'{"a": 1, "b": 2}'
    live = b'{"b":2,"a":1}'
    assert diff_request_bytes(recorded, live) == []


def test_diff_request_bytes_non_json_falls_back_to_base64():
    recorded = b"\x00\x01binary"
    live = b"\x00\x01different"
    diffs = diff_request_bytes(recorded, live)
    assert len(diffs) == 1
    assert diffs[0].path == "$._raw_b64"


# ── diagnose() ───────────────────────────────────────────────────────────────


def _req(body: bytes, *, url: str = "https://api.anthropic.com/v1/messages") -> httpx.Request:
    return httpx.Request("POST", url, content=body)


def test_diagnose_identity_matcher_real_divergence():
    tape = Tape()
    recorded_body = b'{"model":"claude-sonnet-4-6","max_tokens":100}'
    tape.append_exchange(recorded_body, b"{}")
    live = _req(b'{"model":"claude-sonnet-4-6","max_tokens":200}')

    diag = diagnose(tape, 0, live, matcher=IDENTITY_MATCHER)

    assert diag is not None
    assert diag.is_real_divergence is True
    assert diag.field_diffs == (FieldDiff("$.max_tokens", 100, 200),)
    assert diag.matcher_name == "identity"
    assert diag.recorded_fingerprint != diag.live_fingerprint


def test_diagnose_identity_matcher_non_semantic_difference():
    """Byte-different but JSON-semantically-equal bodies (the only way the
    identity matcher can diverge with an empty diff) must NOT read as a real
    divergence."""
    tape = Tape()
    tape.append_exchange(b'{"a": 1, "b": 2}', b"{}")
    live = _req(b'{"b":2,"a":1}')

    diag = diagnose(tape, 0, live, matcher=IDENTITY_MATCHER)

    assert diag is not None
    assert diag.field_diffs == ()
    assert diag.is_real_divergence is False
    assert "non-semantic" in diag.message


def test_diagnose_returns_none_when_step_out_of_range():
    tape = Tape()
    tape.append_exchange(b'{"a":1}', b"{}")
    live = _req(b'{"a":1}')
    assert diagnose(tape, 5, live, matcher=IDENTITY_MATCHER) is None
    assert diagnose(tape, -1, live, matcher=IDENTITY_MATCHER) is None


def test_diagnose_canonicalizing_matcher_excludes_volatile_fields():
    """A rotated idempotency_key (volatile, stripped by `redacting_matcher()`)
    must not show up in the diff even when a genuine field ALSO changed
    alongside it."""
    matcher = redacting_matcher()
    recorded_req = _req(b'{"idempotency_key":"abc","max_tokens":100}')
    tape = Tape()
    tape.append_exchange(matcher.stored_request(recorded_req), b"{}")

    live = _req(b'{"idempotency_key":"xyz","max_tokens":200}')
    diag = diagnose(tape, 0, live, matcher=matcher)

    assert diag is not None
    paths = {d.path for d in diag.field_diffs}
    assert "$.body.max_tokens" in paths
    assert not any("idempotency_key" in p for p in paths)
    assert diag.is_real_divergence is True
    assert "idempotency_key" in diag.normalized_fields


def test_diagnose_canonicalizing_matcher_pure_volatile_rotation_is_not_real():
    """If the ONLY difference is a volatile field, the canonical forms are
    identical (they'd never actually diverge at the transport layer either —
    stored_fingerprint would match). Calling diagnose() directly on that pair
    still proves the exclusion: zero diffs, not a real divergence."""
    matcher = redacting_matcher()
    recorded_req = _req(b'{"idempotency_key":"abc","max_tokens":100}')
    tape = Tape()
    tape.append_exchange(matcher.stored_request(recorded_req), b"{}")

    live = _req(b'{"idempotency_key":"xyz","max_tokens":100}')
    diag = diagnose(tape, 0, live, matcher=matcher)

    assert diag is not None
    assert diag.field_diffs == ()
    assert diag.is_real_divergence is False
    assert diag.recorded_fingerprint == diag.live_fingerprint


def test_diagnose_defaults_to_identity_matcher_when_none_given():
    tape = Tape()
    tape.append_exchange(b'{"a":1}', b"{}")
    live = _req(b'{"a":2}')
    diag = diagnose(tape, 0, live)
    assert diag is not None
    assert diag.matcher_name == "identity"


def test_diagnostic_to_dict_round_trips_through_json():
    import json

    diag = DivergenceDiagnostic(
        step_index=0,
        recorded_fingerprint="abc",
        live_fingerprint="def",
        matcher_name="identity",
        normalized_fields=(),
        field_diffs=(FieldDiff("$.max_tokens", 100, 200),),
        is_real_divergence=True,
        message="1 field(s) differ from the recorded request",
    )
    data = diagnostic_to_dict(diag)
    encoded = json.dumps(data)
    decoded = json.loads(encoded)
    assert decoded["field_diffs"] == [{"path": "$.max_tokens", "recorded": 100, "live": 200}]
    assert decoded["is_real_divergence"] is True
    assert decoded["normalized_fields"] == []
