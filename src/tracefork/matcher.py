"""Pluggable request-matcher seam behind the divergence contract.

``transport.py`` proves bit-exactness by comparing a *fingerprint* of every
recorded request against the same fingerprint of the request the agent rebuilds
at replay. Raw ``sha256(request.content)`` is perfect for providers whose request
bytes are already deterministic (Anthropic, OpenAI), but it false-positives on
providers that fold volatile material into the request: Gemini's ``?key=`` in the
URL, Bedrock's ``x-amz-date`` signing header, rotating bearer / ``x-api-key``
auth, per-call idempotency keys. A ``RequestMatcher`` is the VCR ``match_on``-style
seam that lets those be normalized *before* hashing.

**The default is and must stay identity.** ``IdentityMatcher`` hashes the raw
request body exactly as the pre-seam transport did, so existing tapes,
``validate --check`` and every current test produce byte-identical hashes and
behavior. Canonicalization is strictly opt-in per provider/config; it is never on
by default.

The contract each matcher must uphold is a single invariant::

    stored_fingerprint(stored_request(R)) == live_fingerprint(R)

i.e. the fingerprint the recorder persists for request ``R`` equals the
fingerprint recomputed from the replayed request. Record and replay MUST use the
same matcher instance/config, or the two sides of that equation drift apart.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import httpx

from .tape import sha256_hex

if TYPE_CHECKING:
    from .providers.base import ProviderAdapter


@runtime_checkable
class RequestMatcher(Protocol):
    """Turns a request into the bytes stored on the tape + a hashable identity.

    ``transport.py`` calls ``stored_request`` on the record side (what to persist)
    and compares ``stored_fingerprint(recorded_bytes)`` against
    ``live_fingerprint(replay_request)`` on the replay side. Implementations must
    keep the invariant ``stored_fingerprint(stored_request(R)) == live_fingerprint(R)``.
    """

    def stored_request(self, request: httpx.Request) -> bytes:
        """The request bytes to persist in the tape for this exchange."""
        ...

    def live_fingerprint(self, request: httpx.Request) -> str:
        """A hashable identity of a live (record- or replay-time) request."""
        ...

    def stored_fingerprint(self, stored: bytes) -> str:
        """A hashable identity of a previously-stored request blob."""
        ...


class IdentityMatcher:
    """Default matcher: raw ``sha256(request.content)`` — bit-exact, no change.

    Stores the raw request body verbatim and fingerprints it with the exact hash
    ``transport.py`` used before this seam existed. This is the only matcher on
    the default path; keeping it identity is what preserves every existing tape.
    """

    name = "identity"

    def stored_request(self, request: httpx.Request) -> bytes:
        return request.content

    def live_fingerprint(self, request: httpx.Request) -> str:
        return sha256_hex(request.content)

    def stored_fingerprint(self, stored: bytes) -> str:
        return sha256_hex(stored)


class AdapterMatcher:
    """Opt-in matcher delegating body canonicalization to a ``ProviderAdapter``.

    Wires the adapter's ``canonicalize_request(bytes) -> str`` into the transport
    hash call site without changing what is stored (the raw body is persisted, so
    ``detect_model`` / report keep working). For the Anthropic adapter this is
    exactly identity behavior — its ``canonicalize_request`` is ``sha256`` of the
    body — but a future provider whose adapter normalizes its own body shape gets
    picked up here for free.
    """

    def __init__(self, adapter: ProviderAdapter) -> None:
        self._adapter = adapter
        self.name = f"adapter:{adapter.name}"

    def stored_request(self, request: httpx.Request) -> bytes:
        return request.content

    def live_fingerprint(self, request: httpx.Request) -> str:
        return self._adapter.canonicalize_request(request.content)

    def stored_fingerprint(self, stored: bytes) -> str:
        return self._adapter.canonicalize_request(stored)


def _canonical_body(body: bytes, volatile_fields: frozenset[str]) -> Any:
    """A deterministic, JSON-serializable view of a request body.

    JSON objects drop ``volatile_fields`` (top-level) and are re-emitted with
    sorted keys downstream; non-JSON bodies are represented losslessly by their
    base64 so equal bytes always canonicalize equal and unequal bytes never
    collide.
    """
    if not body:
        return None
    try:
        obj = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return {"_raw_b64": base64.b64encode(body).decode("ascii")}
    if isinstance(obj, dict) and volatile_fields:
        return {k: v for k, v in obj.items() if k not in volatile_fields}
    return obj


@dataclass(frozen=True)
class CanonicalizingMatcher:
    """Opt-in matcher that strips volatile fields before hashing.

    Normalizes a request to a canonical form so semantically-identical requests
    that differ only in volatile material hash equal:

    * ``volatile_query_keys`` — URL query keys dropped (Gemini ``?key=``);
    * ``volatile_headers`` — header names dropped from the identity (``x-amz-date``,
      ``authorization`` / ``x-api-key`` bearer rotation);
    * ``volatile_body_fields`` — top-level JSON body fields dropped (idempotency keys);
    * ``match_headers`` — allowlist of header names that *do* participate in the
      identity (minus any that are volatile); empty means headers are out of scope;
    * ``match_url`` — whether the canonical URL (path + non-volatile, sorted query)
      participates.

    The canonical form *is* what gets stored, so the recorded and replayed sides
    normalize identically and rotated volatile fields never spuriously diverge.
    """

    volatile_query_keys: frozenset[str] = field(default_factory=frozenset)
    volatile_headers: frozenset[str] = field(default_factory=frozenset)
    volatile_body_fields: frozenset[str] = field(default_factory=frozenset)
    match_headers: frozenset[str] = field(default_factory=frozenset)
    match_url: bool = True
    name: str = "canonical"

    def _canonical_url(self, request: httpx.Request) -> dict[str, Any]:
        url = request.url
        drop = {k.lower() for k in self.volatile_query_keys}
        kept = sorted((k, v) for k, v in url.params.multi_items() if k.lower() not in drop)
        return {
            "scheme": url.scheme,
            "host": url.host,
            "path": url.path,
            "query": kept,
        }

    def _canonical_headers(self, request: httpx.Request) -> dict[str, str]:
        volatile = {h.lower() for h in self.volatile_headers}
        out: dict[str, str] = {}
        for name in self.match_headers:
            key = name.lower()
            if key in volatile:
                continue
            value = request.headers.get(key)
            if value is not None:
                out[key] = value
        return out

    def _canonical(self, request: httpx.Request) -> bytes:
        obj: dict[str, Any] = {
            "method": request.method.upper(),
            "body": _canonical_body(request.content, self.volatile_body_fields),
        }
        if self.match_url:
            obj["url"] = self._canonical_url(request)
        if self.match_headers:
            obj["headers"] = self._canonical_headers(request)
        return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()

    def stored_request(self, request: httpx.Request) -> bytes:
        return self._canonical(request)

    def live_fingerprint(self, request: httpx.Request) -> str:
        return sha256_hex(self._canonical(request))

    def stored_fingerprint(self, stored: bytes) -> str:
        return sha256_hex(stored)


# ── the single default instance the transport falls back to ──────────────────

IDENTITY_MATCHER = IdentityMatcher()


# ── provider presets (opt-in) ────────────────────────────────────────────────


def gemini_matcher() -> CanonicalizingMatcher:
    """Collapse Gemini's ``?key=`` URL secret and its ``x-goog-api-key`` header."""
    return CanonicalizingMatcher(
        volatile_query_keys=frozenset({"key"}),
        volatile_headers=frozenset({"authorization", "x-goog-api-key"}),
        name="gemini",
    )


def bedrock_matcher() -> CanonicalizingMatcher:
    """Collapse Bedrock SigV4 signing material (``x-amz-date`` etc.) and auth."""
    signing = frozenset(
        {
            "authorization",
            "x-amz-date",
            "x-amz-security-token",
            "x-amz-content-sha256",
        }
    )
    return CanonicalizingMatcher(
        volatile_headers=signing,
        match_headers=signing | frozenset({"x-amz-target"}),
        name="bedrock",
    )


def redacting_matcher() -> CanonicalizingMatcher:
    """Provider-neutral matcher that drops common rotating auth / idempotency material."""
    return CanonicalizingMatcher(
        volatile_query_keys=frozenset({"key", "api_key"}),
        volatile_headers=frozenset({"authorization", "x-api-key"}),
        volatile_body_fields=frozenset({"idempotency_key", "request_id"}),
        name="redacting",
    )
