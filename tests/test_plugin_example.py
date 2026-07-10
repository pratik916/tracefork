"""Tests for `examples/plugin_example/` — the standalone third-party plugin
package documented in `docs/plugin-api.md`. Proves two things at once:

1. The example's `NonceStrippingMatcher` genuinely satisfies `RequestMatcher`'s
   round-trip invariant `stored_fingerprint(stored_request(R)) ==
   live_fingerprint(R)`, standing on its own two feet as a real matcher, not
   just a stub.
2. `Registry.load_entry_points()` — the exact mechanism a real third-party
   package uses — loads it once explicitly allowlisted, and is a no-op
   without one, reusing `test_plugins.py`'s existing
   `_fake_entry_point`/`_patch_entry_points` pattern.

Offline, zero API keys, zero network — the example package's directory is
added to `sys.path` for the duration of these tests only, exactly as if it
had been `pip install -e`d.
"""

from __future__ import annotations

import sys
from importlib import metadata
from pathlib import Path

import httpx
import pytest

from tracefork.matcher import MATCHER_REGISTRY

EXAMPLE_ROOT = Path(__file__).parent.parent / "examples" / "plugin_example"


@pytest.fixture(autouse=True)
def _example_on_path():
    sys.path.insert(0, str(EXAMPLE_ROOT))
    try:
        yield
    finally:
        sys.path.remove(str(EXAMPLE_ROOT))
        sys.modules.pop("plugin_example", None)
        sys.modules.pop("plugin_example.matcher", None)


def _request(**headers: str) -> httpx.Request:
    return httpx.Request(
        "POST", "https://api.example.com/v1/chat", content=b'{"x":1}', headers=headers
    )


def test_example_matcher_satisfies_fingerprint_round_trip():
    from plugin_example.matcher import NonceStrippingMatcher

    matcher = NonceStrippingMatcher()
    live = _request(**{"x-request-nonce": "abc123", "x-session": "s1"})
    stored = matcher.stored_request(live)
    assert matcher.stored_fingerprint(stored) == matcher.live_fingerprint(live)


def test_example_matcher_ignores_nonce_value():
    from plugin_example.matcher import NonceStrippingMatcher

    matcher = NonceStrippingMatcher()
    first = _request(**{"x-request-nonce": "aaa", "x-session": "s1"})
    second = _request(**{"x-request-nonce": "zzz", "x-session": "s1"})
    assert matcher.live_fingerprint(first) == matcher.live_fingerprint(second)


def test_example_matcher_still_distinguishes_real_differences():
    from plugin_example.matcher import NonceStrippingMatcher

    matcher = NonceStrippingMatcher()
    first = _request(**{"x-request-nonce": "aaa", "x-session": "s1"})
    second = _request(**{"x-request-nonce": "aaa", "x-session": "s2"})
    assert matcher.live_fingerprint(first) != matcher.live_fingerprint(second)


def _fake_matcher_entry_point(name: str) -> metadata.EntryPoint:
    # Points at the example package's real, importable class — `.load()`
    # succeeds via pure import machinery, no installed distribution needed.
    return metadata.EntryPoint(
        name=name,
        value="plugin_example.matcher:NonceStrippingMatcher",
        group=MATCHER_REGISTRY.group,
    )


def test_registry_loads_example_plugin_with_explicit_allow(monkeypatch):
    ep = _fake_matcher_entry_point("example_nonce_stripping")
    monkeypatch.setattr(
        metadata,
        "entry_points",
        lambda group=None: [ep] if group == MATCHER_REGISTRY.group else [],
    )
    try:
        loaded = MATCHER_REGISTRY.load_entry_points(allow={"example_nonce_stripping"})
        assert loaded == ["example_nonce_stripping"]
        from plugin_example.matcher import NonceStrippingMatcher

        assert isinstance(MATCHER_REGISTRY["example_nonce_stripping"], NonceStrippingMatcher)
        assert "example_nonce_stripping" in MATCHER_REGISTRY.loaded_entry_points
    finally:
        MATCHER_REGISTRY.pop("example_nonce_stripping", None)
        MATCHER_REGISTRY.loaded_entry_points.discard("example_nonce_stripping")


def test_registry_loading_example_plugin_is_noop_without_allowlist(monkeypatch):
    ep = _fake_matcher_entry_point("example_nonce_stripping_2")
    monkeypatch.setattr(
        metadata,
        "entry_points",
        lambda group=None: [ep] if group == MATCHER_REGISTRY.group else [],
    )
    monkeypatch.delenv("TRACEFORK_ALLOW_PLUGINS", raising=False)
    assert MATCHER_REGISTRY.load_entry_points() == []
    assert "example_nonce_stripping_2" not in MATCHER_REGISTRY
