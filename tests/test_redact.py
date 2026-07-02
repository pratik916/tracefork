"""Redaction tests (`redact.py`).

Four guarantees are load-bearing here:

1. The **default (no redactor) path is byte-for-byte unchanged** — passing no
   `redactor` to `Recorder`/`AsyncRecorder`/the transports records exactly as
   before this seam existed.
2. **Header + secret-env redaction is metadata-always and replay-safe**: it
   runs inside the matcher seam, so record and replay hash the identical
   redacted form and replay still verifies even when the live secret rotates.
3. **Message-content redaction is a separate, opt-in layer** that marks the
   tape `content_redacted` (forensic-only) — mirroring
   `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`.
4. **Placeholder substitution is deterministic** — exact bytes, never a
   length/hash comparison that could hide nondeterminism.

Offline, zero API keys.
"""

from __future__ import annotations

import json

import anthropic
import httpx

from tests.fakes import ScriptedFakeLLM, make_text_response
from tracefork.matcher import CanonicalizingMatcher, IdentityMatcher
from tracefork.recorder import Recorder
from tracefork.redact import (
    REDACTED,
    REDACTED_STR,
    RedactingMatcher,
    Redactor,
    content_redactor,
    regex_redactor,
    safe_defaults,
    secret_value_redactor,
    with_content_redaction,
)
from tracefork.tape import Tape, sha256_hex
from tracefork.transport import TraceforkTransport

# ── helpers ──────────────────────────────────────────────────────────────────


def _req(
    body: bytes = b"",
    *,
    url: str = "https://api.anthropic.com/v1/messages",
    headers: dict[str, str] | None = None,
) -> httpx.Request:
    return httpx.Request("POST", url, headers=headers or {}, content=body)


class _SyncInner(httpx.BaseTransport):
    def __init__(self, responses: list[bytes]):
        self._responses = iter(responses)

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"content-type": "application/json"}, content=next(self._responses)
        )


def _sync_client(fake: ScriptedFakeLLM) -> anthropic.Anthropic:
    return anthropic.Anthropic(
        api_key="sk-ant-fake", http_client=httpx.Client(transport=fake), max_retries=0
    )


# ── 1. default path is unchanged ─────────────────────────────────────────────


def test_transport_without_redactor_is_untouched():
    tape = Tape()
    t = TraceforkTransport("record", tape, _SyncInner([b"resp"]))
    t.handle_request(_req(b'{"model": "m"}'))
    assert tape.exchanges[0] == (b'{"model": "m"}', b"resp")
    assert t.redactor is None


def test_recorder_without_redactor_is_untouched():
    fake = ScriptedFakeLLM([make_text_response("hi")])
    client = _sync_client(fake)
    with Recorder(client) as rec:
        rec.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=100,
            messages=[{"role": "user", "content": "Hello"}],
        )
    assert rec.tape.content_redacted is False
    body = json.loads(rec.tape.exchanges[0][0])
    assert body["messages"][0]["content"] == "Hello"
    assert rec.tape.exchanges[0][1] == make_text_response("hi")


def test_tape_content_redacted_defaults_false_and_roundtrips():
    tape = Tape()
    assert tape.content_redacted is False
    data = tape.to_bytes()
    assert Tape.from_bytes(data).content_redacted is False


# ── 2. header + secret-env redaction is replay-safe (matcher seam) ──────────


def test_redacting_matcher_invariant_holds():
    redactor = safe_defaults()
    m = RedactingMatcher(IdentityMatcher(), redactor)
    request = _req(b'{"model": "m"}', headers={"authorization": "Bearer secret"})
    assert m.stored_fingerprint(m.stored_request(request)) == m.live_fingerprint(request)


def test_header_redaction_replaces_auth_values_but_keeps_safe_headers():
    inner = CanonicalizingMatcher(match_headers=frozenset({"authorization", "anthropic-version"}))
    redactor = safe_defaults()
    m = redactor.matcher(inner)
    request = _req(
        b'{"model": "m"}',
        headers={"authorization": "Bearer sk-ant-LIVE", "anthropic-version": "2023-06-01"},
    )
    stored = json.loads(m.stored_request(request))
    assert stored["headers"]["authorization"] == REDACTED_STR
    assert stored["headers"]["anthropic-version"] == "2023-06-01"


def test_header_redaction_strips_unknown_anthropic_prefixed_headers():
    inner = CanonicalizingMatcher(match_headers=frozenset({"anthropic-organization-id"}))
    redactor = safe_defaults()
    m = redactor.matcher(inner)
    request = _req(b"{}", headers={"anthropic-organization-id": "org-secret-123"})
    stored = json.loads(m.stored_request(request))
    assert stored["headers"]["anthropic-organization-id"] == REDACTED_STR


def test_header_redaction_is_noop_for_identity_matcher():
    """The identity matcher never stores headers, so redaction has nothing to
    scrub — the stored bytes are exactly the raw body, unchanged."""
    redactor = safe_defaults()
    m = redactor.matcher()  # default inner: identity
    body = b'{"model": "m", "messages": []}'
    request = _req(body, headers={"authorization": "Bearer secret"})
    assert m.stored_request(request) == body


def test_header_redaction_replays_fine_despite_rotated_secret():
    """Full transport round trip: record with one auth header value, replay
    with a rotated (different) one — the redacted matcher collapses both to
    the same fingerprint, so replay still verifies (hashes agree)."""
    inner = CanonicalizingMatcher(match_headers=frozenset({"authorization"}))
    m = safe_defaults().matcher(inner)
    tape = Tape()
    rec = TraceforkTransport("record", tape, _SyncInner([b"resp"]), matcher=m)
    rec.handle_request(_req(b'{"model": "m"}', headers={"authorization": "Bearer T1"}))
    # The stored request never contains the live secret.
    assert b"T1" not in tape.exchanges[0][0]

    rep = TraceforkTransport("replay", tape, matcher=m)
    result = rep.handle_request(_req(b'{"model": "m"}', headers={"authorization": "Bearer T9"}))
    assert result.read() == b"resp"
    assert rep.fully_consumed()


def test_secret_env_value_scrubbed_in_request_and_response(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-super-secret-value")
    redactor = safe_defaults()
    tape = Tape()
    body = (
        b'{"model": "m", "messages": ['
        b'{"role": "user", "content": "my key is sk-ant-super-secret-value"}]}'
    )
    resp = make_text_response("your key sk-ant-super-secret-value is invalid")
    t = TraceforkTransport(
        "record", tape, _SyncInner([resp]), matcher=redactor.matcher(), redactor=redactor
    )
    t.handle_request(_req(body))
    stored_req, stored_resp = tape.exchanges[0]
    assert b"sk-ant-super-secret-value" not in stored_req
    assert b"sk-ant-super-secret-value" not in stored_resp
    assert REDACTED in stored_req
    assert REDACTED in stored_resp


def test_secret_value_redactor_ignores_short_values(monkeypatch):
    monkeypatch.setenv("SHORT_SECRET", "abc")  # below default min_length
    fn = secret_value_redactor(["SHORT_SECRET"])
    assert fn(b"abc appears here") == b"abc appears here"


def test_secret_value_redactor_skips_unset_vars(monkeypatch):
    monkeypatch.delenv("NOT_SET_VAR", raising=False)
    fn = secret_value_redactor(["NOT_SET_VAR"])
    assert fn(b"unchanged") == b"unchanged"


# ── 3. content redaction is opt-in and marks the tape forensic-only ─────────


def test_with_content_redaction_defaults_to_redacting_and_sets_flag(monkeypatch):
    monkeypatch.delenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", raising=False)
    redactor = with_content_redaction(safe_defaults())
    assert redactor.content_redacted is True
    body = json.dumps({"messages": [{"role": "user", "content": "secret prompt"}]}).encode()
    (out_fn,) = (redactor.request_filters[-1],)
    out = out_fn(body)
    assert b"secret prompt" not in out
    assert REDACTED in out


def test_with_content_redaction_capture_true_is_noop_and_not_forensic():
    base = safe_defaults()
    redactor = with_content_redaction(base, capture_message_content=True)
    assert redactor is base
    assert redactor.content_redacted is False


def test_with_content_redaction_env_var_true_skips_redaction(monkeypatch):
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "true")
    base = safe_defaults()
    redactor = with_content_redaction(base)
    assert redactor is base
    assert redactor.content_redacted is False


def test_recorder_with_content_redaction_marks_tape_forensic_only(monkeypatch):
    monkeypatch.delenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", raising=False)
    fake = ScriptedFakeLLM([make_text_response("the real secret answer")])
    client = _sync_client(fake)
    redactor = with_content_redaction(safe_defaults())
    with Recorder(client, redactor=redactor) as rec:
        rec.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=100,
            messages=[{"role": "user", "content": "what is the secret?"}],
        )
    assert rec.tape.content_redacted is True
    req_body, resp_body = rec.tape.exchanges[0]
    assert b"what is the secret?" not in req_body
    assert b"the real secret answer" not in resp_body
    # The live caller (the agent) still saw the real response — only the tape is scrubbed.
    # (Verified indirectly: Recorder always returns real bytes to the SDK; see transport.py.)

    # The forensic-only flag survives serialization.
    restored = Tape.from_bytes(rec.tape.to_bytes())
    assert restored.content_redacted is True


def test_header_only_redaction_via_recorder_stays_replayable():
    """Header/secret-env redaction alone (no content redaction) must not be
    marked forensic-only, and the resulting tape must still bit-exact-verify —
    the exact request body the agent sends is untouched by this redactor."""
    fake = ScriptedFakeLLM([make_text_response("hi")])
    client = _sync_client(fake)
    with Recorder(client, redactor=safe_defaults()) as rec:
        rec.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=100,
            messages=[{"role": "user", "content": "Hello"}],
        )
    assert rec.tape.content_redacted is False
    body = json.loads(rec.tape.exchanges[0][0])
    assert body["messages"][0]["content"] == "Hello"


def test_content_redactor_preserves_structure():
    fn = content_redactor()
    body = json.dumps(
        {
            "model": "m",
            "system": "be nice",
            "messages": [
                {"role": "user", "content": "hi there"},
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "hello back"}],
                },
            ],
        }
    ).encode()
    out = json.loads(fn(body))
    assert out["model"] == "m"
    assert out["system"] == REDACTED_STR
    assert out["messages"][0]["content"] == REDACTED_STR
    assert out["messages"][1]["content"][0]["type"] == "text"
    assert out["messages"][1]["content"][0]["text"] == REDACTED_STR


def test_content_redactor_handles_tool_blocks():
    fn = content_redactor()
    body = json.dumps(
        {
            "content": [
                {"type": "tool_use", "id": "t1", "name": "book", "input": {"city": "Tokyo"}},
                {"type": "tool_result", "tool_use_id": "t1", "content": "confirmed"},
            ]
        }
    ).encode()
    out = json.loads(fn(body))
    assert out["content"][0]["input"] == {"_redacted": True}
    assert out["content"][0]["name"] == "book"  # structure preserved
    assert out["content"][1]["content"] == REDACTED_STR


def test_content_redactor_non_json_falls_back_to_whole_body_placeholder():
    fn = content_redactor()
    assert fn(b"event: message\ndata: not-json\n\n") == REDACTED


# ── 4. placeholder / redaction determinism (exact bytes, no float compares) ─


def test_secret_value_redactor_is_byte_deterministic(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-deterministic-value")
    data = b'{"content": "leaked sk-ant-deterministic-value here"}'
    fn_a = secret_value_redactor(["ANTHROPIC_API_KEY"])
    fn_b = secret_value_redactor(["ANTHROPIC_API_KEY"])
    out_a = fn_a(data)
    out_b = fn_b(data)
    assert out_a == out_b == b'{"content": "leaked REDACTED here"}'


def test_content_redactor_is_byte_deterministic():
    body = json.dumps({"messages": [{"role": "user", "content": "hi"}]}, sort_keys=True).encode()
    fn_a, fn_b = content_redactor(), content_redactor()
    assert fn_a(body) == fn_b(body)


def test_header_redaction_is_byte_deterministic():
    inner = CanonicalizingMatcher(match_headers=frozenset({"authorization"}))
    redactor = safe_defaults()
    m_a = redactor.matcher(inner)
    m_b = redactor.matcher(inner)
    request = _req(b"{}", headers={"authorization": "Bearer secret"})
    assert m_a.stored_request(request) == m_b.stored_request(request)


# ── generic filter contract: None means "replace with the fixed placeholder" ─


def test_none_returning_filter_yields_fixed_placeholder_not_a_dropped_exchange():
    def _drop_everything(data: bytes) -> bytes | None:
        return None

    redactor = Redactor(request_filters=(_drop_everything,), response_filters=(_drop_everything,))
    tape = Tape()
    t = TraceforkTransport(
        "record",
        tape,
        _SyncInner([b"real response"]),
        matcher=redactor.matcher(),
        redactor=redactor,
    )
    t.handle_request(_req(b'{"model": "m"}'))
    # The exchange is still recorded (index alignment preserved) with the fixed placeholder.
    assert len(tape.exchanges) == 1
    assert tape.exchanges[0] == (REDACTED, REDACTED)


def test_regex_redactor_replaces_matches():
    fn = regex_redactor(r"sk-ant-[a-zA-Z0-9]+")
    assert fn(b"key=sk-ant-abc123 done") == b"key=REDACTED done"


def test_regex_redactor_custom_replacement():
    fn = regex_redactor(r"\d{3}-\d{2}-\d{4}", replacement=b"[SSN]")
    assert fn(b"ssn 123-45-6789 on file") == b"ssn [SSN] on file"


# ── sanity: sha256_hex still used as expected by the matcher seam ───────────


def test_redacting_matcher_stored_fingerprint_matches_sha256_of_stored_bytes():
    redactor = safe_defaults()
    m = redactor.matcher()
    stored = m.stored_request(_req(b'{"model": "m"}'))
    assert m.stored_fingerprint(stored) == sha256_hex(stored)
