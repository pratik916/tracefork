"""tracefork-bge.45 — request-URL capture (tape v6) + URL-based model detection.

Covers, offline/$0, in one isolated file:
  * `Tape.append_exchange`/`request_urls` stay parallel-indexed to `exchanges`.
  * `to_bytes`/`from_bytes` round-trips `request_urls` (v6); a hand-constructed
    pre-v6 envelope upcasts to `[""] * len(exchanges)`.
  * `digest()` is byte-identical regardless of `request_url` — the single
    most load-bearing invariant for this bead (explicit test, not implied).
  * `GeminiAdapter`/`BedrockAdapter.detect_model` fall back to parsing the
    model id out of an optional `request_url` when the body has none, with
    the pre-existing `request_url=None` behavior fully preserved.
  * `transport.py`/`bedrock_transport.py` capture the real request URL into
    `tape.request_urls` at their real capture seams.
  * `blame._detect_model`/`interop._normalize_exchange` resolve a Bedrock/
    Gemini model id via the URL where the body alone could not, while an
    existing Anthropic-shaped tape still resolves unchanged.
"""

from __future__ import annotations

import json
import struct

import httpx
import zstandard as zstd

from tracefork import blame as blame_mod
from tracefork.bedrock_transport import BedrockTransport
from tracefork.constants import SONNET, TAPE_MAGIC
from tracefork.interop import _normalize_exchange
from tracefork.providers import get_adapter
from tracefork.providers.bedrock import BedrockAdapter, build_invoke_model_request
from tracefork.providers.gemini import GeminiAdapter
from tracefork.synthetic import (
    FakeAWSPreparedRequest,
    FakeEventEmitter,
    ScriptedBedrockSender,
    first_non_none_response,
)
from tracefork.tape import Tape, sha256_hex
from tracefork.transport import TraceforkTransport

# ── Tape.append_exchange / request_urls parity ──────────────────────────────


def test_append_exchange_without_request_url_stores_empty_string():
    tape = Tape()
    tape.append_exchange(b"req", b"resp")
    assert tape.request_urls == [""]
    assert len(tape.request_urls) == len(tape.exchanges)


def test_append_exchange_with_request_url_stored_verbatim_at_index():
    tape = Tape()
    tape.append_exchange(b"req-1", b"resp-1")  # no URL -> ""
    tape.append_exchange(b"req-2", b"resp-2", request_url="https://example.test/v1/messages")
    assert len(tape.request_urls) == len(tape.exchanges) == 2
    assert tape.request_urls[0] == ""
    assert tape.request_urls[1] == "https://example.test/v1/messages"


# ── v6 to_bytes/from_bytes round-trip + pre-v6 upcast ───────────────────────


def test_request_urls_roundtrips_through_to_bytes_from_bytes():
    tape = Tape(agent_name="url-agent")
    tape.append_exchange(b"req-1", b"resp-1", request_url="https://a.test/1")
    tape.append_exchange(b"req-2", b"resp-2", request_url="https://a.test/2")
    restored = Tape.from_bytes(tape.to_bytes())
    assert restored.request_urls == tape.request_urls
    assert restored.digest() == tape.digest()


def _encode_as_v5_without_request_urls(t: Tape) -> bytes:
    """Hand-construct a genuine pre-v6 (v5) envelope — no `request_urls` key
    in the header at all — to prove the v5->v6 upcaster defaults a pre-v6
    tape's `request_urls` to `[""] * len(exchanges)` with an unchanged digest,
    mirroring `test_storage.py`'s `_encode_as_v4_without_provenance` proof for
    the previous bump."""
    zctx = zstd.ZstdCompressor(level=3)
    order: list[str] = []
    seen: dict[str, bytes] = {}
    for req, resp in (*t.exchanges, *t.tool_exchanges):
        for b in (req, resp):
            h = sha256_hex(b)
            if h not in seen:
                seen[h] = b
                order.append(h)
    header = {
        "boundary": t.boundary,
        "agent_name": t.agent_name,
        "draws": t.draws,
        "exchanges": [[sha256_hex(r), sha256_hex(s)] for r, s in t.exchanges],
        "tool_exchanges": [[sha256_hex(r), sha256_hex(s)] for r, s in t.tool_exchanges],
        "async_batches": t.async_batches,
        "provenance": t.provenance,
        "blob_hashes": order,
        "content_redacted": t.content_redacted,
    }
    header_json = json.dumps(header).encode()
    parts = [TAPE_MAGIC, struct.pack(">H", 5), struct.pack(">I", len(header_json)), header_json]
    for h in order:
        comp = zctx.compress(seen[h])
        parts.append(struct.pack(">I", len(comp)))
        parts.append(comp)
    return b"".join(parts)


def test_pre_v6_envelope_upcasts_request_urls_to_empty_strings():
    t = Tape(agent_name="pre-v6")
    t.append_exchange(b"req-1", b"resp-1")
    t.append_exchange(b"req-2", b"resp-2")
    blob = _encode_as_v5_without_request_urls(t)
    restored = Tape.from_bytes(blob)
    assert restored.request_urls == [""] * len(restored.exchanges)
    assert restored.digest() == t.digest()


# ── digest() byte-stability under differing request_url (THE invariant) ────


def test_digest_identical_regardless_of_request_url():
    t1 = Tape()
    t1.append_exchange(b"req", b"resp", request_url="https://a.test/x")
    t2 = Tape()
    t2.append_exchange(b"req", b"resp", request_url="https://b.test/y")
    t3 = Tape()
    t3.append_exchange(b"req", b"resp")  # no URL at all
    assert t1.request_urls != t2.request_urls
    assert t1.digest() == t2.digest() == t3.digest()


# ── GeminiAdapter.detect_model URL fallback ─────────────────────────────────


def test_gemini_detect_model_falls_back_to_url_when_body_has_no_model():
    adapter = GeminiAdapter()
    body = json.dumps({"contents": []}).encode()
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent"
    assert adapter.detect_model(body, request_url=url) == "gemini-1.5-pro"


def test_gemini_detect_model_body_field_wins_over_url():
    adapter = GeminiAdapter()
    body = json.dumps({"model": "gemini-2.0-flash"}).encode()
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent"
    assert adapter.detect_model(body, request_url=url) == "gemini-2.0-flash"


def test_gemini_detect_model_none_url_unchanged():
    adapter = GeminiAdapter()
    body = json.dumps({"contents": []}).encode()
    assert adapter.detect_model(body, request_url=None) is None
    assert adapter.detect_model(body) is None


# ── BedrockAdapter.detect_model URL fallback ────────────────────────────────

BEDROCK_URL = (
    "https://bedrock-runtime.us-east-1.amazonaws.com/model/"
    "anthropic.claude-haiku-4-5-20251001-v1%3A0/invoke"
)


def test_bedrock_detect_model_parses_and_unquotes_url():
    adapter = BedrockAdapter()
    body = build_invoke_model_request([{"role": "user", "content": "hi"}])
    assert (
        adapter.detect_model(body, request_url=BEDROCK_URL)
        == "anthropic.claude-haiku-4-5-20251001-v1:0"
    )


def test_bedrock_detect_model_none_url_unchanged():
    # Matches the pre-existing test_bedrock_provider.py::test_detect_model_always_none
    # expectation: no request_url, still None.
    adapter = BedrockAdapter()
    body = build_invoke_model_request([{"role": "user", "content": "hi"}])
    assert adapter.detect_model(body) is None
    assert adapter.detect_model(body, request_url=None) is None


# ── transport.py: real captured URL lands in tape.request_urls ─────────────


class _SyncInner(httpx.BaseTransport):
    def __init__(self, responses: list[bytes]) -> None:
        self._responses = iter(responses)

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"content-type": "application/json"}, content=next(self._responses)
        )


def test_transport_record_mode_captures_request_url():
    tape = Tape()
    inner = _SyncInner([b"resp-1"])
    t = TraceforkTransport("record", tape, inner)
    url = "https://api.anthropic.com/v1/messages"
    request = httpx.Request("POST", url, content=b"req-1")
    t.handle_request(request)
    assert tape.request_urls[-1] == str(request.url)
    assert tape.request_urls[-1] == url


# ── bedrock_transport.py: real captured URL lands in tape.request_urls ─────

INVOKE_URL = (
    "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-sonnet-4-6/invoke"
)
EVENT_NAME = "before-send.bedrock-runtime.InvokeModel"

RESP_BODY = (
    b'{"id": "msg_1", "type": "message", "role": "assistant", '
    b'"model": "anthropic.claude-sonnet-4-6", "content": [{"type": "text", "text": "hello"}], '
    b'"stop_reason": "end_turn", "usage": {"input_tokens": 10, "output_tokens": 5}}'
)


def _prepared(body: bytes) -> FakeAWSPreparedRequest:
    return FakeAWSPreparedRequest(
        method="POST",
        url=INVOKE_URL,
        headers={"content-type": "application/json"},
        body=body,
    )


def test_bedrock_transport_record_captures_request_url():
    tape = Tape()
    sender = ScriptedBedrockSender([RESP_BODY])
    recorder = BedrockTransport("record", tape, sender=sender)
    emitter = FakeEventEmitter()
    recorder.register(emitter)

    result = first_non_none_response(
        emitter.emit(EVENT_NAME, request=_prepared(build_invoke_model_request([])))
    )
    assert result is not None
    assert tape.request_urls[-1] == INVOKE_URL


# ── blame._detect_model: multi-adapter URL fallback ─────────────────────────


def test_blame_detect_model_anthropic_tape_unchanged():
    tape = Tape()
    req = json.dumps({"model": SONNET, "messages": []}).encode()
    resp = get_adapter("anthropic").build_text_response("hi", model=SONNET)
    tape.append_exchange(req, resp, request_url="https://api.anthropic.com/v1/messages")
    assert blame_mod._detect_model(tape) == SONNET


def test_blame_detect_model_resolves_bedrock_model_via_url():
    tape = Tape()
    req = build_invoke_model_request([{"role": "user", "content": "hi"}])
    resp = get_adapter("bedrock").build_text_response(
        "hi", model="anthropic.claude-haiku-4-5-20251001-v1:0"
    )
    tape.append_exchange(req, resp, request_url=BEDROCK_URL)
    assert blame_mod._detect_model(tape) == "anthropic.claude-haiku-4-5-20251001-v1:0"


def test_blame_detect_model_resolves_gemini_model_via_url():
    tape = Tape()
    req = json.dumps({"contents": []}).encode()  # Gemini body carries no model
    resp = get_adapter("gemini").build_text_response("hi", model="gemini-1.5-pro")
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent"
    tape.append_exchange(req, resp, request_url=url)
    assert blame_mod._detect_model(tape) == "gemini-1.5-pro"


def test_blame_detect_model_falls_back_to_sonnet_with_no_match_anywhere():
    tape = Tape()
    tape.append_exchange(b"not json", b"not json either")  # no URL, no model anywhere
    assert blame_mod._detect_model(tape) == SONNET


# ── interop._normalize_exchange: URL threaded to detect_model ──────────────


def test_normalize_exchange_bedrock_resolves_model_via_url():
    req = build_invoke_model_request([{"role": "user", "content": "hi"}])
    resp = get_adapter("bedrock").build_text_response(
        "hi", model="anthropic.claude-haiku-4-5-20251001-v1:0"
    )
    request_model, _normalized = _normalize_exchange("bedrock", req, resp, BEDROCK_URL)
    assert request_model == "anthropic.claude-haiku-4-5-20251001-v1:0"
    # Body alone (no request_url) can't recover it.
    request_model_no_url, _ = _normalize_exchange("bedrock", req, resp)
    assert request_model_no_url is None


def test_normalize_exchange_gemini_resolves_model_via_url():
    req = json.dumps({"contents": []}).encode()
    resp = get_adapter("gemini").build_text_response("hi", model="gemini-1.5-pro")
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent"
    request_model, _normalized = _normalize_exchange("gemini", req, resp, url)
    assert request_model == "gemini-1.5-pro"
    request_model_no_url, _ = _normalize_exchange("gemini", req, resp)
    assert request_model_no_url is None
