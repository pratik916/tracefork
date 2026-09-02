"""Pricing fallback fail-safety + current-generation model coverage.

`tests/test_pricing.py` (owned by another lane) pins the PRE-existing
"unknown model -> Sonnet" fallback behavior by name
(`test_unknown_model_falls_back_to_sonnet` and friends) and an exact
`pricing_version()` string; this file is additive rather than editing that
one, per this lane's file-ownership boundary (see planning/HANDOFF.md for
the exact rename/update those three tests + the version pin need once this
lane's fail-safe-fallback change lands). Offline, zero API keys, no network.
"""

from __future__ import annotations

from tracefork import pricing

_MILLION = 1_000_000


def test_fable_5_is_the_most_expensive_known_anthropic_rate():
    """The fallback must be an upper bound: no other known-Anthropic rate in
    the snapshot may exceed it, or an unknown model could still be
    under-priced relative to something we DO know about."""
    fallback_input, fallback_output = pricing.get_rates("claude-fable-5", "anthropic")
    for model in pricing.registered_models("anthropic"):
        in_rate, out_rate = pricing.get_rates(model, "anthropic")
        assert in_rate <= fallback_input, f"{model} input rate exceeds the fallback"
        assert out_rate <= fallback_output, f"{model} output rate exceeds the fallback"


def test_unknown_model_falls_back_to_the_most_expensive_rate_not_sonnet():
    """The headline fix: an unrecognized model id must never resolve to a
    rate CHEAPER than a real current-generation model — the old Sonnet
    default did exactly that for e.g. an Opus-5 tape."""
    sonnet_rates = pricing.get_rates("claude-sonnet-4-6", "anthropic")
    opus5_rates = pricing.get_rates("claude-opus-5", "anthropic")
    fallback_rates = pricing.get_rates("some-future-model-this-snapshot-does-not-know")

    assert fallback_rates != sonnet_rates
    assert fallback_rates[0] >= opus5_rates[0]
    assert fallback_rates[1] >= opus5_rates[1]


def test_is_fallback_model_is_true_only_when_the_lookup_actually_fell_back():
    assert pricing.is_fallback_model("totally-made-up-model") is True
    assert pricing.is_fallback_model(None) is True
    assert pricing.is_fallback_model("gpt-4o", "anthropic") is True  # wrong-provider miss
    assert pricing.is_fallback_model("claude-sonnet-4-6", "anthropic") is False
    assert pricing.is_fallback_model("claude-opus-5", "anthropic") is False
    # The fallback model itself, looked up directly, is correctly NOT
    # reported as a fallback resolution -- it's a real, matched entry.
    assert pricing.is_fallback_model("claude-fable-5", "anthropic") is False


def test_current_generation_models_are_in_the_snapshot():
    """`tracefork.validate` (2026-08-25 evidence) shows the pre-fix snapshot
    only knew claude-sonnet-4-6/haiku-4-5/opus-4-8 — every model released
    since was silently mispriced via the Sonnet default. Regression guard."""
    for model, expected in [
        ("claude-opus-5", (5.0 / _MILLION, 25.0 / _MILLION)),
        ("claude-sonnet-5", (2.0 / _MILLION, 10.0 / _MILLION)),
        ("claude-fable-5", (10.0 / _MILLION, 50.0 / _MILLION)),
        ("claude-mythos-5", (10.0 / _MILLION, 50.0 / _MILLION)),
        ("claude-opus-4-7", (5.0 / _MILLION, 25.0 / _MILLION)),
        ("claude-opus-4-6", (5.0 / _MILLION, 25.0 / _MILLION)),
    ]:
        assert pricing.get_rates(model, "anthropic") == expected, model
        assert pricing.is_fallback_model(model, "anthropic") is False, model


def test_pricing_version_was_bumped_alongside_the_snapshot_change():
    assert pricing.pricing_version() != "2026-06b"
    from tracefork.constants import PRICING_VERSION

    assert pricing.pricing_version() == PRICING_VERSION
