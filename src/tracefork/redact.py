"""Opt-in secret/PII redaction on record.

``transport.py`` tees raw request+response bytes into the tape; recording real
production traffic means those bytes can carry API keys, bearer tokens, and
user PII. Redaction here is:

* **opt-in** — nothing in this module runs unless a `Redactor` is built and
  passed to `Recorder`/`AsyncRecorder` (default: `redactor=None`, unchanged
  behavior);
* **metadata-always** — once a `Redactor` is in use, header/secret-env-value
  scrubbing always runs (there is no knob to keep a live secret on a tape);
* **content-opt-in** — message CONTENT redaction is a separate, explicit
  layer: `safe_defaults()` alone never touches content or sets the forensic
  flag; only reaching for `with_content_redaction()` turns it on. Once opted
  in, its default posture mirrors the OpenTelemetry GenAI semantic-convention
  env var `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` (default:
  content is *not* captured, i.e. it is redacted) — set that env var, or pass
  `capture_message_content=True`, to keep full content instead;
* **bit-exactness-aware.** Header/secret-env redaction runs *inside* the
  matcher seam (`RedactingMatcher`, wired via `Redactor.matcher()`): the exact
  same deterministic transform is applied to the bytes that get stored *and*
  to the bytes re-derived at replay time for fingerprinting, so
  ``stored_fingerprint(stored_request(R)) == live_fingerprint(R)`` continues
  to hold and replay verifies cleanly — see `matcher.py`. Redacting
  agent-visible RESPONSE CONTENT, by contrast, changes what the agent
  *reads back* on replay (and weakens the request-side divergence check, since
  genuinely different redacted prompts can collapse to the same placeholder),
  so any tape built with content redaction on is marked
  ``Tape.content_redacted = True`` — forensic-only, not a guaranteed
  bit-exact-replayable artifact. See the README's Redaction section.

Redactor callback contract: ``(bytes) -> bytes | None``. A callback may return
the bytes unchanged, a mutated copy, or ``None``. ``None`` means "this value
should not survive onto the tape"; it is replaced with the fixed ``REDACTED``
placeholder rather than removed, so a redacted exchange is always a
well-formed, same-shaped record — dropping the exchange itself would shift
every later index and break index-addressed replay/fork/blame.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import httpx

from .matcher import IDENTITY_MATCHER, RequestMatcher
from .tape import sha256_hex

RedactorFn = Callable[[bytes], bytes | None]

REDACTED = b"REDACTED"
REDACTED_STR = "REDACTED"

# Safe-default secret env vars scrubbed wherever their literal value appears in
# recorded bytes (request or response). Extend via `secret_env_vars=` on
# `safe_defaults()`.
DEFAULT_SECRET_ENV_VARS: frozenset[str] = frozenset({"ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"})

# Header names (case-insensitive) always redacted once a `Redactor` is active.
# Only takes effect when the wrapped matcher's canonical form actually captures
# headers (e.g. a `CanonicalizingMatcher` configured with `match_headers=`) —
# the identity matcher never stores headers at all, so this is a no-op there.
DEFAULT_REDACTED_HEADERS: frozenset[str] = frozenset({"authorization", "x-api-key"})

# Known-safe informational anthropic-* headers (not secrets) exempted from the
# blanket "anthropic-*" auth-header rule below.
_ANTHROPIC_SAFE_HEADERS: frozenset[str] = frozenset({"anthropic-version", "anthropic-beta"})

_OTEL_CAPTURE_MESSAGE_CONTENT_ENV = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
_TRUTHY = {"1", "true", "yes", "on"}


# ── generic filter factories (user-supplied regex/callable) ─────────────────


def regex_redactor(pattern: str | re.Pattern[bytes], replacement: bytes = REDACTED) -> RedactorFn:
    """Build a redactor that replaces every regex match with `replacement`."""
    compiled = pattern if isinstance(pattern, re.Pattern) else re.compile(pattern.encode())

    def _redact(data: bytes) -> bytes:
        return compiled.sub(replacement, data)

    return _redact


def secret_value_redactor(env_var_names: Iterable[str], *, min_length: int = 8) -> RedactorFn:
    """Scrub literal occurrences of the named env vars' *current* values.

    Values are read once, at construction time — not per call — so the same
    redactor keeps scrubbing the identical literal bytes deterministically
    across record and replay (a value that rotated between the two would
    simply no longer match, which is fine: this redactor's job is hiding a
    known secret, not detecting drift). Values shorter than `min_length` are
    skipped — short strings are too likely to false-positive against ordinary
    content. Longest-first so overlapping secrets scrub cleanly.
    """
    secrets = sorted(
        {
            value.encode()
            for name in env_var_names
            if (value := os.environ.get(name)) and len(value) >= min_length
        },
        key=len,
        reverse=True,
    )

    def _redact(data: bytes) -> bytes:
        for secret in secrets:
            data = data.replace(secret, REDACTED)
        return data

    return _redact


# ── message-content redaction (opt-in, forensic-only) ────────────────────────


def _redact_content_blocks(blocks: object) -> None:
    """Recursively blank natural-language text in an Anthropic content shape
    (a list of typed blocks) in place. Preserves block `type` and other
    structural keys so the JSON shape stays valid and inspectable."""
    if not isinstance(blocks, list):
        return
    for block in blocks:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype in ("text", "thinking") and "text" in block:
            block["text"] = REDACTED_STR
        elif btype == "tool_use" and "input" in block:
            block["input"] = {"_redacted": True}
        elif btype == "tool_result" and "content" in block:
            if isinstance(block["content"], str):
                block["content"] = REDACTED_STR
            else:
                _redact_content_blocks(block["content"])


def _redact_message_content(obj: dict) -> None:
    """Blank `system` / `messages[].content` (request shape) and top-level
    `content` (response shape) in an Anthropic Messages-API JSON body, in
    place."""
    system = obj.get("system")
    if isinstance(system, str):
        obj["system"] = REDACTED_STR
    elif isinstance(system, list):
        _redact_content_blocks(system)
    for message in obj.get("messages") or []:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            message["content"] = REDACTED_STR
        elif isinstance(content, list):
            _redact_content_blocks(content)
    top_content = obj.get("content")
    if isinstance(top_content, list):
        _redact_content_blocks(top_content)


def content_redactor() -> RedactorFn:
    """Redact Anthropic message CONTENT — request `system`/`messages[].content`
    and response `content[]` — in place, keeping the surrounding JSON shape
    intact. Falls back to the whole-body `REDACTED` placeholder for bodies
    that aren't a single JSON object (e.g. streaming SSE framing): content
    redaction is already forensic-only, so losing fine structure there is an
    accepted, documented limitation rather than a fragile partial parse.
    """

    def _redact(data: bytes) -> bytes:
        try:
            obj = json.loads(data)
        except (ValueError, UnicodeDecodeError):
            return REDACTED
        if not isinstance(obj, dict):
            return REDACTED
        _redact_message_content(obj)
        return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()

    return _redact


# ── header redaction (metadata-always, replay-safe via the matcher seam) ────


def _is_redacted_header(name: str, extra: frozenset[str]) -> bool:
    lower = name.lower()
    if lower in extra or lower in DEFAULT_REDACTED_HEADERS:
        return True
    return lower.startswith("anthropic-") and lower not in _ANTHROPIC_SAFE_HEADERS


def _redact_canonical_headers(data: bytes, extra_headers: frozenset[str]) -> bytes:
    """If `data` is a `CanonicalizingMatcher`-style canonical JSON blob with a
    top-level `headers` object, replace the value of any sensitive header
    (auth headers, plus any `anthropic-*` header not on the known-safe
    informational allowlist) with the fixed placeholder. A no-op for any other
    JSON shape — in particular the identity matcher's raw request body, which
    never has a `headers` key, so the default (no `match_headers`) path is
    untouched.
    """
    try:
        obj = json.loads(data)
    except (ValueError, UnicodeDecodeError):
        return data
    if not isinstance(obj, dict):
        return data
    headers = obj.get("headers")
    if not isinstance(headers, dict):
        return data
    changed = False
    for key in list(headers):
        if isinstance(headers[key], str) and _is_redacted_header(key, extra_headers):
            headers[key] = REDACTED_STR
            changed = True
    if not changed:
        return data
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


class RedactingMatcher:
    """`RequestMatcher` wrapper that redacts a request *before* it is stored
    or fingerprinted, so record and replay hash the identical redacted form —
    the seam that keeps redaction bit-exact-replayable (module docstring).

    Wraps an inner matcher (default: the identity matcher, or any
    `CanonicalizingMatcher` preset) and layers the owning `Redactor`'s header
    redaction and `request_filters` on top of whatever bytes the inner matcher
    decided to store. Because `live_fingerprint` recomputes that exact same
    transform from the live request, `stored_fingerprint(stored_request(R)) ==
    live_fingerprint(R)` holds for any deterministic filter pipeline.
    """

    def __init__(self, inner: RequestMatcher, redactor: Redactor, name: str = "redacted") -> None:
        self._inner = inner
        self._redactor = redactor
        self.name = name

    def _transform(self, request: httpx.Request) -> bytes:
        data = self._inner.stored_request(request)
        data = _redact_canonical_headers(data, self._redactor.redact_headers)
        for fn in self._redactor.request_filters:
            result = fn(data)
            data = REDACTED if result is None else result
        return data

    def stored_request(self, request: httpx.Request) -> bytes:
        return self._transform(request)

    def live_fingerprint(self, request: httpx.Request) -> str:
        return sha256_hex(self._transform(request))

    def stored_fingerprint(self, stored: bytes) -> str:
        return sha256_hex(stored)


@dataclass(frozen=True)
class Redactor:
    """Ordered, opt-in redaction pipeline passed to `Recorder`/`AsyncRecorder`.

    * `request_filters` / `response_filters` — ordered `RedactorFn` callbacks
      applied to the bytes about to be stored (request side: after the
      wrapped matcher's own `stored_request()`; response side: the raw
      response body). Build your own with `regex_redactor()` /
      `secret_value_redactor()`, or use a plain function.
    * `redact_headers` — extra header names (beyond the metadata-always
      defaults in `DEFAULT_REDACTED_HEADERS` and the `anthropic-*` auth-header
      rule) to scrub wherever the wrapped matcher's canonical form captures
      headers.
    * `content_redacted` — set by `with_content_redaction()`; copied onto
      `Tape.content_redacted` by `Recorder`/`AsyncRecorder` to mark the tape
      forensic-only.

    Construct via `safe_defaults()` (metadata only, fully replayable) and
    optionally layer `with_content_redaction()` on top (opt-in, forensic-only).
    """

    request_filters: tuple[RedactorFn, ...] = ()
    response_filters: tuple[RedactorFn, ...] = ()
    redact_headers: frozenset[str] = frozenset()
    content_redacted: bool = False

    def matcher(self, inner: RequestMatcher | None = None) -> RequestMatcher:
        """Wrap `inner` (default: identity) so its stored/fingerprinted bytes
        are redacted identically on both sides of the record/replay boundary."""
        return RedactingMatcher(inner or IDENTITY_MATCHER, self)

    def apply_response(self, resp_body: bytes) -> bytes:
        """Run `response_filters` over a response body about to be stored."""
        data = resp_body
        for fn in self.response_filters:
            result = fn(data)
            data = REDACTED if result is None else result
        return data


def safe_defaults(*, secret_env_vars: Iterable[str] = DEFAULT_SECRET_ENV_VARS) -> Redactor:
    """Metadata-only safe defaults: auth headers + known secret env values.

    Never touches message content. A tape recorded with only this `Redactor`
    stays fully bit-exact-replayable — header/secret-env redaction runs
    through the matcher seam, so record and replay still hash identically.
    """
    scrub = secret_value_redactor(secret_env_vars)
    return Redactor(
        request_filters=(scrub,),
        response_filters=(scrub,),
        redact_headers=DEFAULT_REDACTED_HEADERS,
    )


def _resolve_capture_message_content(explicit: bool | None) -> bool:
    if explicit is not None:
        return explicit
    env = os.environ.get(_OTEL_CAPTURE_MESSAGE_CONTENT_ENV)
    if env is None:
        # Mirrors OTEL's own default: capture is off, so content gets redacted.
        # Reaching for `with_content_redaction()` at all is the opt-in step —
        # `safe_defaults()` alone never redacts content or sets this flag.
        return False
    return env.strip().lower() in _TRUTHY


def with_content_redaction(
    base: Redactor | None = None,
    *,
    capture_message_content: bool | None = None,
) -> Redactor:
    """Layer opt-in message-CONTENT redaction on top of `base` (default:
    `safe_defaults()`).

    Mirrors the OpenTelemetry GenAI convention
    `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`: pass
    `capture_message_content=True` (or set that env var truthy) to keep full
    content — this call then returns `base` unchanged, and the tape is *not*
    marked forensic-only. Otherwise (the default), message content is
    replaced with `REDACTED` placeholders and the returned `Redactor` sets
    `content_redacted=True`; `Recorder`/`AsyncRecorder` copy that onto
    `Tape.content_redacted` — forensic-only, not a guaranteed bit-exact
    replay artifact (see the README's Redaction section).
    """
    base = base if base is not None else safe_defaults()
    if _resolve_capture_message_content(capture_message_content):
        return base
    scrub_content = content_redactor()
    return Redactor(
        request_filters=(*base.request_filters, scrub_content),
        response_filters=(*base.response_filters, scrub_content),
        redact_headers=base.redact_headers,
        content_redacted=True,
    )
