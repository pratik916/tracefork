"""RequestMatcher / canonicalization tests.

Two guarantees are load-bearing here:

1. The **default identity path is byte-for-byte unchanged** — the tape stores the
   raw request body and the fingerprint is exactly ``sha256(request.content)``, so
   every existing tape and hash is preserved.
2. Opt-in canonicalizing matchers **collapse volatile-field differences** (Gemini
   ``?key=``, Bedrock ``x-amz-date``, rotating auth, idempotency keys) so
   semantically-identical requests hash equal — and record+replay agree under the
   same matcher — while genuinely different requests still diverge.

Offline, zero API keys. Equality is asserted on exact bytes/strings, never floats.
"""

import httpx
import pytest

from tracefork.matcher import (
    IDENTITY_MATCHER,
    AdapterMatcher,
    CanonicalizingMatcher,
    IdentityMatcher,
    bedrock_matcher,
    gemini_matcher,
    redacting_matcher,
)
from tracefork.nondet import DivergenceError
from tracefork.providers import get_adapter
from tracefork.tape import Tape, sha256_hex
from tracefork.transport import AsyncTraceforkTransport, TraceforkTransport

# ── helpers ──────────────────────────────────────────────────────────────────


def _req(
    body: bytes = b"",
    *,
    url: str = "https://api.anthropic.com/v1/messages",
    headers: dict[str, str] | None = None,
    method: str = "POST",
) -> httpx.Request:
    return httpx.Request(method, url, headers=headers or {}, content=body)


class _SyncInner(httpx.BaseTransport):
    def __init__(self, responses: list[bytes]):
        self._responses = iter(responses)

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"content-type": "application/json"}, content=next(self._responses)
        )


class _AsyncInner(httpx.AsyncBaseTransport):
    def __init__(self, responses: list[bytes]):
        self._responses = iter(responses)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"content-type": "application/json"}, content=next(self._responses)
        )


ALL_MATCHERS = [
    IdentityMatcher(),
    AdapterMatcher(get_adapter("anthropic")),
    gemini_matcher(),
    bedrock_matcher(),
    redacting_matcher(),
    CanonicalizingMatcher(),
]


# ── the invariant every matcher must uphold ──────────────────────────────────


@pytest.mark.parametrize("matcher", ALL_MATCHERS, ids=lambda m: m.name)
def test_matcher_invariant_stored_fp_equals_live_fp(matcher):
    body = b'{"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": "hi"}]}'
    request = _req(body, headers={"authorization": "Bearer secret", "x-amz-date": "20260101"})
    assert matcher.stored_fingerprint(matcher.stored_request(request)) == matcher.live_fingerprint(
        request
    )


# ── 1. identity default is unchanged ─────────────────────────────────────────


def test_identity_is_raw_sha256_of_body():
    body = b'{"model": "claude-sonnet-4-6"}'
    m = IdentityMatcher()
    req = _req(body, url="https://host/p?key=SECRET", headers={"authorization": "Bearer x"})
    # Stored bytes are the raw body verbatim; fingerprint ignores URL + headers.
    assert m.stored_request(req) == body
    assert m.live_fingerprint(req) == sha256_hex(body)
    assert m.stored_fingerprint(body) == sha256_hex(body)


def test_transport_default_matcher_is_identity_singleton():
    assert TraceforkTransport("replay", Tape()).matcher is IDENTITY_MATCHER
    assert AsyncTraceforkTransport("replay", Tape()).matcher is IDENTITY_MATCHER


def test_transport_default_records_raw_body_bytes_unchanged():
    tape = Tape()
    body = b'{"model": "x", "messages": []}'
    t = TraceforkTransport("record", tape, _SyncInner([b"resp"]))
    t.handle_request(_req(body, url="https://host/p?key=SECRET"))
    # Default path stores the raw request body exactly as before the seam existed.
    assert tape.exchanges[0][0] == body
    assert sha256_hex(tape.exchanges[0][0]) == sha256_hex(body)


def test_transport_default_replay_matches_and_diverges_as_before():
    tape = Tape()
    tape.append_exchange(b"expected", b"resp")
    ok = TraceforkTransport("replay", tape)
    assert ok.handle_request(_req(b"expected")).read() == b"resp"
    bad = TraceforkTransport("replay", Tape())
    bad.tape.append_exchange(b"expected", b"resp")
    with pytest.raises(DivergenceError, match="diverged"):
        bad.handle_request(_req(b"different"))


# ── 2. canonicalizers collapse volatile-field differences ────────────────────


def test_gemini_key_query_param_collapses():
    m = gemini_matcher()
    body = b'{"contents": [{"parts": [{"text": "hi"}]}]}'
    base = "https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent"
    a = _req(body, url=f"{base}?key=SECRET_A")
    b = _req(body, url=f"{base}?key=SECRET_B")
    assert m.stored_request(a) == m.stored_request(b)
    assert m.live_fingerprint(a) == m.live_fingerprint(b)


def test_gemini_different_path_does_not_collapse():
    m = gemini_matcher()
    body = b'{"contents": []}'
    a = _req(body, url="https://g/v1/models/gemini-pro:generateContent?key=K")
    b = _req(body, url="https://g/v1/models/gemini-flash:generateContent?key=K")
    assert m.live_fingerprint(a) != m.live_fingerprint(b)


def test_bedrock_x_amz_date_header_collapses_but_target_matters():
    m = bedrock_matcher()
    body = b'{"anthropic_version": "bedrock-2023-05-31"}'
    url = "https://bedrock-runtime.us-east-1.amazonaws.com/model/foo/invoke"
    a = _req(body, url=url, headers={"x-amz-date": "20260101T000000Z", "x-amz-target": "Invoke"})
    b = _req(body, url=url, headers={"x-amz-date": "20260702T121212Z", "x-amz-target": "Invoke"})
    # x-amz-date is volatile signing material -> collapses.
    assert m.live_fingerprint(a) == m.live_fingerprint(b)
    # x-amz-target is a real, non-volatile part of the identity -> distinguishes.
    c = _req(body, url=url, headers={"x-amz-date": "20260101T000000Z", "x-amz-target": "Stream"})
    assert m.live_fingerprint(a) != m.live_fingerprint(c)


def test_redacting_collapses_auth_and_idempotency():
    m = redacting_matcher()
    body_a = b'{"model": "m", "idempotency_key": "aaa", "request_id": "r1"}'
    body_b = b'{"model": "m", "idempotency_key": "zzz", "request_id": "r2"}'
    a = _req(body_a, url="https://h/v1?api_key=K1", headers={"x-api-key": "sk-A"})
    b = _req(body_b, url="https://h/v1?api_key=K2", headers={"x-api-key": "sk-B"})
    assert m.live_fingerprint(a) == m.live_fingerprint(b)


def test_redacting_still_distinguishes_real_body_change():
    m = redacting_matcher()
    a = _req(b'{"model": "sonnet", "idempotency_key": "x"}')
    b = _req(b'{"model": "opus", "idempotency_key": "x"}')
    assert m.live_fingerprint(a) != m.live_fingerprint(b)


def test_canonicalizing_nested_key_order_is_normalized():
    m = CanonicalizingMatcher()
    a = _req(b'{"a": 1, "b": {"y": 2, "x": 3}}')
    b = _req(b'{"b": {"x": 3, "y": 2}, "a": 1}')
    assert m.stored_request(a) == m.stored_request(b)


def test_canonicalizing_non_json_body_is_deterministic_and_distinct():
    m = CanonicalizingMatcher()
    a1 = _req(b"\x00\x01not-json")
    a2 = _req(b"\x00\x01not-json")
    b = _req(b"\x00\x01other")
    assert m.stored_request(a1) == m.stored_request(a2)
    assert m.live_fingerprint(a1) != m.live_fingerprint(b)


def test_adapter_matcher_anthropic_equals_identity():
    m = AdapterMatcher(get_adapter("anthropic"))
    body = b'{"model": "claude-sonnet-4-6"}'
    req = _req(body, url="https://h/p?key=SECRET", headers={"x-api-key": "sk-A"})
    # Anthropic adapter canonicalize_request is sha256(body) -> identical to identity.
    assert m.stored_request(req) == body
    assert m.live_fingerprint(req) == sha256_hex(body)


# ── 3. record + replay agreement under a canonicalizer ───────────────────────


def test_record_replay_agreement_under_canonicalizer_with_rotated_volatiles():
    m = redacting_matcher()
    tape = Tape()
    # Record with one set of volatile values.
    rec = TraceforkTransport("record", tape, _SyncInner([b"r0", b"r1"]), matcher=m)
    rec.handle_request(
        _req(
            b'{"model": "m", "idempotency_key": "A0"}',
            url="https://h/v1?api_key=SECRET1",
            headers={"authorization": "Bearer T1"},
        )
    )
    rec.handle_request(
        _req(
            b'{"model": "m2", "idempotency_key": "B0"}',
            url="https://h/v1?api_key=SECRET1",
            headers={"authorization": "Bearer T1"},
        )
    )
    # Replay with the SAME matcher but rotated key / token / idempotency values.
    rep = TraceforkTransport("replay", tape, matcher=m)
    assert (
        rep.handle_request(
            _req(
                b'{"model": "m", "idempotency_key": "A9"}',
                url="https://h/v1?api_key=ROTATED2",
                headers={"authorization": "Bearer T9"},
            )
        ).read()
        == b"r0"
    )
    assert (
        rep.handle_request(
            _req(
                b'{"model": "m2", "idempotency_key": "B9"}',
                url="https://h/v1?api_key=ROTATED2",
                headers={"authorization": "Bearer T9"},
            )
        ).read()
        == b"r1"
    )
    assert rep.fully_consumed()


def test_record_replay_diverges_on_real_change_under_canonicalizer():
    m = redacting_matcher()
    tape = Tape()
    rec = TraceforkTransport("record", tape, _SyncInner([b"r0"]), matcher=m)
    rec.handle_request(
        _req(b'{"model": "m", "idempotency_key": "A"}', url="https://h/v1?api_key=K")
    )
    rep = TraceforkTransport("replay", tape, matcher=m)
    # A genuinely different (non-volatile) body must still be caught.
    with pytest.raises(DivergenceError, match="diverged"):
        rep.handle_request(
            _req(b'{"model": "DIFFERENT", "idempotency_key": "A"}', url="https://h/v1?api_key=K")
        )


@pytest.mark.asyncio
async def test_async_record_replay_agreement_under_canonicalizer():
    m = gemini_matcher()
    tape = Tape()
    base = "https://g/v1/models/gemini-pro:generateContent"
    body = b'{"contents": [{"parts": [{"text": "hi"}]}]}'
    rec = AsyncTraceforkTransport("record", tape, _AsyncInner([b"resp"]), matcher=m)
    await rec.handle_async_request(_req(body, url=f"{base}?key=SECRET1"))
    rep = AsyncTraceforkTransport("replay", tape, matcher=m)
    r = await rep.handle_async_request(_req(body, url=f"{base}?key=ROTATED"))
    assert await r.aread() == b"resp"
    assert rep.fully_consumed()
