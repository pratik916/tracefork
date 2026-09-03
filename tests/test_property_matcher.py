"""Property-based (Hypothesis) proof of `matcher.py`'s core contract.

`test_matcher.py` pins the identity path (and each canonicalizing preset)
against a handful of fixed request bodies; this module generalizes the
invariant every `RequestMatcher` must uphold —

    stored_fingerprint(stored_request(R)) == live_fingerprint(R)

— over arbitrary JSON request bodies (and, for the canonicalizing family,
arbitrary headers too), so the fingerprint-equality contract (see
`matcher.py`'s module docstring) is proven generally, not just spot-checked.

Deterministic and offline: `derandomize=True` seeds every example from the
test itself (no example-database file needed — see `test_property_tape.py`'s
`_SETTINGS` comment), and `max_examples` is bounded so this stays well within
CI's time budget. Pure in-process JSON/sha256 work, no network, $0.
"""

from __future__ import annotations

import json

import httpx2
from hypothesis import given, settings
from hypothesis import strategies as st

from tracefork.matcher import (
    IDENTITY_MATCHER,
    RequestMatcher,
    anthropic_header_matcher,
    bedrock_matcher,
    gemini_matcher,
    redacting_matcher,
)

_SETTINGS = settings(max_examples=75, derandomize=True, deadline=None)

# Arbitrary JSON values, bounded so generated request bodies stay small.
_json_leaf = (
    st.none()
    | st.booleans()
    | st.integers(min_value=-(10**9), max_value=10**9)
    | st.floats(allow_nan=False, allow_infinity=False, width=32)
    | st.text(max_size=24)
)
_json_value = st.recursive(
    _json_leaf,
    lambda children: (
        st.lists(children, max_size=4) | st.dictionaries(st.text(max_size=12), children, max_size=4)
    ),
    max_leaves=15,
)

# Header names each canonicalizing preset actually branches on (volatile,
# matched, or both), plus a couple of always-out-of-scope names — bounded so
# Hypothesis explores the interesting cases instead of random noise.
_header_name = st.sampled_from(
    [
        "authorization",
        "x-api-key",
        "x-goog-api-key",
        "x-amz-date",
        "x-amz-target",
        "anthropic-version",
        "anthropic-beta",
        "user-agent",
    ]
)
# Printable-ASCII, no control chars -- httpx2.Request rejects raw header values
# containing them (e.g. CR/LF), so this keeps every generated example valid.
_header_value = st.text(
    alphabet=st.characters(min_codepoint=0x20, max_codepoint=0x7E), min_size=1, max_size=16
)
_headers = st.dictionaries(_header_name, _header_value, max_size=5)

_CANONICALIZING_MATCHERS: list[RequestMatcher] = [
    gemini_matcher(),
    bedrock_matcher(),
    redacting_matcher(),
    anthropic_header_matcher(),
]


def _request(body_obj: object, headers: dict[str, str]) -> httpx2.Request:
    body = json.dumps(body_obj).encode()
    return httpx2.Request(
        "POST",
        "https://api.anthropic.com/v1/messages?key=SECRET",
        headers={**headers, "content-type": "application/json"},
        content=body,
    )


@_SETTINGS
@given(_json_value)
def test_identity_matcher_fingerprint_roundtrip_over_arbitrary_json_bodies(
    body_obj: object,
) -> None:
    """`IdentityMatcher.stored_fingerprint(stored_request(R)) ==
    live_fingerprint(R)` for any JSON-serializable request body — the general
    form of the fixed-input identity checks in `test_matcher.py`."""
    body = json.dumps(body_obj).encode()
    request = httpx2.Request(
        "POST",
        "https://api.anthropic.com/v1/messages",
        headers={"content-type": "application/json"},
        content=body,
    )
    stored = IDENTITY_MATCHER.stored_request(request)
    assert IDENTITY_MATCHER.stored_fingerprint(stored) == IDENTITY_MATCHER.live_fingerprint(request)


@_SETTINGS
@given(_json_value, _headers)
def test_canonicalizing_matchers_fingerprint_roundtrip_over_arbitrary_bodies_and_headers(
    body_obj: object, headers: dict[str, str]
) -> None:
    """The same round-trip invariant, generalized to every `CanonicalizingMatcher`
    preset (`gemini_matcher()`/`bedrock_matcher()`/`redacting_matcher()`/
    `anthropic_header_matcher()`) over arbitrary bodies AND arbitrary header
    combinations — including the new header-aware Anthropic variant
    (tracefork-sis.57), so its `match_headers`/`volatile_headers` interaction
    is proven generally, not just on the fixed examples in `test_matcher.py`."""
    request = _request(body_obj, headers)
    for matcher in _CANONICALIZING_MATCHERS:
        stored = matcher.stored_request(request)
        assert matcher.stored_fingerprint(stored) == matcher.live_fingerprint(request)
