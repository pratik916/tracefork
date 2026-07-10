"""Example third-party ``RequestMatcher`` plugin package.

This is a standalone, installable package that demonstrates the public
plugin extension point documented in ``docs/plugin-api.md``: it ships a
``tracefork.matchers`` entry point (see this package's ``pyproject.toml``)
and, once explicitly allowlisted by an operator or caller (see the security
model in that doc), is loadable via
``tracefork.plugins.Registry.load_entry_points()`` without tracefork ever
importing this package directly.

The example implements the same "canonicalize before hashing" pattern
tracefork's own built-in ``CanonicalizingMatcher`` uses for Gemini/Bedrock
(see ``tracefork.matcher``): it folds a request's headers into its identity
but drops one volatile, per-call header — ``x-request-nonce`` — that a real
agent might regenerate on every call. Without that drop, replaying the exact
same logical request would still fail to match because the nonce differs
byte-for-byte between record and replay.
"""

from __future__ import annotations

import hashlib

import httpx

#: Header dropped from the canonical form — a stand-in for whatever
#: per-call volatile material a real third-party matcher would need to
#: normalize away (idempotency keys, rotating signatures, etc.).
_VOLATILE_HEADER = "x-request-nonce"


def _canonical(request: httpx.Request) -> bytes:
    """Canonical bytes for ``request``: sorted headers (minus the volatile
    nonce) followed by the raw body. This *is* what gets persisted on the
    tape, so the recorded and replayed sides always hash identically."""
    headers = sorted(
        (name.lower(), value)
        for name, value in request.headers.items()
        if name.lower() != _VOLATILE_HEADER
    )
    header_blob = "&".join(f"{name}={value}" for name, value in headers).encode()
    return header_blob + b"\0" + request.content


class NonceStrippingMatcher:
    """Ignore ``x-request-nonce`` when computing a request's identity.

    Satisfies ``tracefork.matcher.RequestMatcher``'s protocol and its
    round-trip invariant::

        stored_fingerprint(stored_request(R)) == live_fingerprint(R)

    for every request ``R``, regardless of what value ``x-request-nonce``
    holds — because ``stored_request`` persists the *canonical* form (the
    same bytes ``live_fingerprint`` would hash), not the raw request.
    """

    name = "example_nonce_stripping"

    def stored_request(self, request: httpx.Request) -> bytes:
        return _canonical(request)

    def live_fingerprint(self, request: httpx.Request) -> str:
        return hashlib.sha256(_canonical(request)).hexdigest()

    def stored_fingerprint(self, stored: bytes) -> str:
        return hashlib.sha256(stored).hexdigest()
