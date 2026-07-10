"""Property-based (Hypothesis) proof of `matcher.py`'s core contract.

`test_matcher.py` pins the identity path against a handful of fixed request
bodies; this module generalizes ``IdentityMatcher``'s invariant —

    stored_fingerprint(stored_request(R)) == live_fingerprint(R)

— over arbitrary JSON request bodies, so the fingerprint-equality contract
`RequestMatcher` implementations must uphold (see `matcher.py`'s module
docstring) is proven generally for the default matcher, not just spot-checked.

Deterministic and offline: `derandomize=True` seeds every example from the
test itself (no example-database file needed — see `test_property_tape.py`'s
`_SETTINGS` comment), and `max_examples` is bounded so this stays well within
CI's time budget. Pure in-process JSON/sha256 work, no network, $0.
"""

from __future__ import annotations

import json

import httpx
from hypothesis import given, settings
from hypothesis import strategies as st

from tracefork.matcher import IDENTITY_MATCHER

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


@_SETTINGS
@given(_json_value)
def test_identity_matcher_fingerprint_roundtrip_over_arbitrary_json_bodies(
    body_obj: object,
) -> None:
    """`IdentityMatcher.stored_fingerprint(stored_request(R)) ==
    live_fingerprint(R)` for any JSON-serializable request body — the general
    form of the fixed-input identity checks in `test_matcher.py`."""
    body = json.dumps(body_obj).encode()
    request = httpx.Request(
        "POST",
        "https://api.anthropic.com/v1/messages",
        headers={"content-type": "application/json"},
        content=body,
    )
    stored = IDENTITY_MATCHER.stored_request(request)
    assert IDENTITY_MATCHER.stored_fingerprint(stored) == IDENTITY_MATCHER.live_fingerprint(request)
