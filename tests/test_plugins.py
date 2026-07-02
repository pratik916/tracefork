"""Plugin registry tests (`plugins.py`) — generic `Registry` mechanics, the
per-protocol registries it backs (providers/oracles/matchers/serializers), and
the opt-in entry-point allowlist gate.

The central security guarantee under test: `Registry.load_entry_points()` must
be a no-op unless the caller (or operator, via `TRACEFORK_ALLOW_PLUGINS`)
explicitly allowlists a name. A package merely being importable must never be
enough to inject a provider/oracle/matcher/serializer into tracefork.

Offline, zero API keys.
"""

from __future__ import annotations

from importlib import metadata

import pytest

from tracefork.blame import StringMatchOracle, get_oracle, register_oracle, registered_oracles
from tracefork.matcher import (
    IDENTITY_MATCHER,
    IdentityMatcher,
    get_matcher,
    register_matcher,
    registered_matchers,
)
from tracefork.plugins import ALLOW_PLUGINS_ENV, Registry
from tracefork.providers import get_adapter, registered_providers
from tracefork.providers.anthropic import AnthropicAdapter
from tracefork.tape import BinaryTapeSerializer, Tape, get_serializer, registered_serializers

# ── generic Registry: register / get / list ─────────────────────────────────


def test_register_get_and_names():
    reg: Registry[str] = Registry("test.group", kind="widget")
    reg.register("b", "B")
    reg.register("a", "A")
    assert reg.get_or_raise("a") == "A"
    assert reg.get_or_raise("b") == "B"
    assert reg.names() == ["a", "b"]


def test_get_or_raise_unknown_lists_registered():
    reg: Registry[str] = Registry("test.group", kind="widget")
    reg.register("known", "K")
    with pytest.raises(KeyError) as exc:
        reg.get_or_raise("missing")
    msg = str(exc.value)
    assert "missing" in msg
    assert "known" in msg
    assert "widget" in msg


def test_register_overwrites_existing_name():
    reg: Registry[str] = Registry("test.group", kind="widget")
    reg.register("a", "first")
    reg.register("a", "second")
    assert reg.get_or_raise("a") == "second"


def test_registry_is_a_real_dict_subclass():
    """`sorted(registry)` / `in` / `.pop()` must keep working — this is what
    lets `providers/base.py`'s `_REGISTRY.pop("dummy", None)` idiom (used by
    `tests/test_providers.py`) survive the fold into `Registry` unchanged."""
    reg: Registry[str] = Registry("test.group", kind="widget")
    reg.register("z", "Z")
    reg.register("a", "A")
    assert isinstance(reg, dict)
    assert sorted(reg) == ["a", "z"]
    assert "a" in reg
    reg.pop("a", None)
    assert "a" not in reg


# ── entry-point allowlist gating (the security guarantee) ───────────────────


def _fake_entry_point(name: str, group: str) -> metadata.EntryPoint:
    # Points at a real, already-importable, zero-arg-constructible class so
    # `.load()` succeeds without needing an actual installed distribution —
    # `EntryPoint.load()` is pure import machinery over `self.value`.
    return metadata.EntryPoint(name=name, value="tracefork.matcher:IdentityMatcher", group=group)


def _patch_entry_points(monkeypatch, group: str, eps: list[metadata.EntryPoint]) -> None:
    target = group
    monkeypatch.setattr(metadata, "entry_points", lambda group=None: eps if group == target else [])


def test_load_entry_points_is_noop_without_any_allowlist(monkeypatch):
    reg: Registry[object] = Registry("tracefork.test.plugins", kind="widget")
    ep = _fake_entry_point("evil", reg.group)
    _patch_entry_points(monkeypatch, reg.group, [ep])
    monkeypatch.delenv(ALLOW_PLUGINS_ENV, raising=False)
    assert reg.load_entry_points() == []
    assert "evil" not in reg


def test_load_entry_points_with_explicit_allow_set(monkeypatch):
    reg: Registry[object] = Registry("tracefork.test.plugins", kind="widget")
    ep = _fake_entry_point("good", reg.group)
    _patch_entry_points(monkeypatch, reg.group, [ep])
    loaded = reg.load_entry_points(allow={"good"})
    assert loaded == ["good"]
    assert isinstance(reg["good"], IdentityMatcher)
    assert "good" in reg.loaded_entry_points


def test_load_entry_points_ignores_names_not_in_allowlist(monkeypatch):
    reg: Registry[object] = Registry("tracefork.test.plugins", kind="widget")
    ep = _fake_entry_point("not_allowed", reg.group)
    _patch_entry_points(monkeypatch, reg.group, [ep])
    assert reg.load_entry_points(allow={"something_else"}) == []
    assert "not_allowed" not in reg


def test_load_entry_points_allow_all_flag(monkeypatch):
    reg: Registry[object] = Registry("tracefork.test.plugins", kind="widget")
    ep = _fake_entry_point("anything", reg.group)
    _patch_entry_points(monkeypatch, reg.group, [ep])
    assert reg.load_entry_points(allow_all=True) == ["anything"]


def test_load_entry_points_env_var_names_specific_entry(monkeypatch):
    reg: Registry[object] = Registry("tracefork.test.plugins", kind="widget")
    ep = _fake_entry_point("envloaded", reg.group)
    _patch_entry_points(monkeypatch, reg.group, [ep])
    monkeypatch.setenv(ALLOW_PLUGINS_ENV, "envloaded")
    assert reg.load_entry_points() == ["envloaded"]


def test_load_entry_points_env_var_other_names_still_blocked(monkeypatch):
    reg: Registry[object] = Registry("tracefork.test.plugins", kind="widget")
    ep = _fake_entry_point("not_named", reg.group)
    _patch_entry_points(monkeypatch, reg.group, [ep])
    monkeypatch.setenv(ALLOW_PLUGINS_ENV, "some_other_name")
    assert reg.load_entry_points() == []


def test_load_entry_points_env_var_star_allows_everything(monkeypatch):
    reg: Registry[object] = Registry("tracefork.test.plugins", kind="widget")
    ep = _fake_entry_point("anything2", reg.group)
    _patch_entry_points(monkeypatch, reg.group, [ep])
    monkeypatch.setenv(ALLOW_PLUGINS_ENV, "*")
    assert reg.load_entry_points() == ["anything2"]


def test_load_entry_points_only_loads_matching_group(monkeypatch):
    reg: Registry[object] = Registry("tracefork.test.plugins.only", kind="widget")
    other_group_ep = _fake_entry_point("x", "some.other.group")
    monkeypatch.setattr(
        metadata,
        "entry_points",
        lambda group=None: [other_group_ep] if group == "some.other.group" else [],
    )
    assert reg.load_entry_points(allow_all=True) == []


# ── per-protocol registries: built-ins present, register/get round trip ─────


def test_provider_registry_has_anthropic_builtin():
    assert "anthropic" in registered_providers()
    assert isinstance(get_adapter("anthropic"), AnthropicAdapter)


def test_matcher_registry_has_builtins_and_round_trips_custom():
    for name in ("identity", "gemini", "bedrock", "redacting"):
        assert name in registered_matchers()
    assert get_matcher("identity") is IDENTITY_MATCHER
    custom = IdentityMatcher()
    try:
        register_matcher("custom_test_matcher", custom)
        assert get_matcher("custom_test_matcher") is custom
        assert "custom_test_matcher" in registered_matchers()
    finally:
        from tracefork.matcher import MATCHER_REGISTRY

        MATCHER_REGISTRY.pop("custom_test_matcher", None)


def test_oracle_registry_has_string_match_builtin_and_round_trips_custom():
    assert "string_match" in registered_oracles()
    cls = get_oracle("string_match")
    assert cls is StringMatchOracle
    instance = cls(success_re="OK", failure_re="ERR")
    assert instance.grade("OK is here") is True
    assert instance.grade("ERR here") is False
    assert instance.grade("neither") is None

    class DummyOracle:
        def grade(self, output: str) -> bool | None:
            return None

    try:
        register_oracle("dummy_test_oracle", DummyOracle)
        assert get_oracle("dummy_test_oracle") is DummyOracle
    finally:
        from tracefork.blame import ORACLE_REGISTRY

        ORACLE_REGISTRY.pop("dummy_test_oracle", None)


def test_serializer_registry_has_binary_builtin():
    assert "binary" in registered_serializers()
    assert isinstance(get_serializer("binary"), BinaryTapeSerializer)


def test_binary_serializer_dumps_loads_round_trips():
    tape = Tape(agent_name="x")
    tape.append_exchange(b"req", b"resp")
    ser = get_serializer("binary")
    restored = ser.loads(ser.dumps(tape))
    assert restored.exchanges == tape.exchanges
    assert restored.agent_name == tape.agent_name
    assert restored.digest() == tape.digest()


def test_get_or_raise_unknown_provider_still_helpful():
    with pytest.raises(KeyError) as exc:
        get_adapter("nonexistent-provider")
    assert "nonexistent-provider" in str(exc.value)
