"""Pricing registry tests — provider-generic lookups backed by the pinned,
bundled offline snapshot, with Anthropic rates byte-identical to constants.

Offline, zero API keys, no network.
"""

import pytest

from tracefork import pricing
from tracefork.blame import BudgetGovernor
from tracefork.constants import (
    HAIKU,
    HAIKU_INPUT_PER_TOKEN,
    HAIKU_OUTPUT_PER_TOKEN,
    OPUS,
    OPUS_INPUT_PER_TOKEN,
    OPUS_OUTPUT_PER_TOKEN,
    SONNET,
    SONNET_INPUT_PER_TOKEN,
    SONNET_OUTPUT_PER_TOKEN,
)
from tracefork.tape import Tape
from tracefork.wire import make_text_response

_MILLION = 1_000_000


# ── Anthropic rates MUST stay byte-identical to constants ────────────────────


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        (SONNET, (SONNET_INPUT_PER_TOKEN, SONNET_OUTPUT_PER_TOKEN)),
        (HAIKU, (HAIKU_INPUT_PER_TOKEN, HAIKU_OUTPUT_PER_TOKEN)),
        (OPUS, (OPUS_INPUT_PER_TOKEN, OPUS_OUTPUT_PER_TOKEN)),
    ],
)
def test_anthropic_rates_are_byte_identical_to_constants(model, expected):
    assert pricing.get_rates(model) == expected


def test_anthropic_rates_scoped_by_provider_match():
    assert pricing.get_rates(SONNET, "anthropic") == (
        SONNET_INPUT_PER_TOKEN,
        SONNET_OUTPUT_PER_TOKEN,
    )


# ── OpenAI + Gemini lookups ──────────────────────────────────────────────────


def test_openai_rates_lookup():
    assert pricing.get_rates("gpt-4o", "openai") == (2.5 / _MILLION, 10.0 / _MILLION)
    assert pricing.get_rates("gpt-4o-mini", "openai") == (0.15 / _MILLION, 0.6 / _MILLION)


def test_gemini_rates_lookup():
    assert pricing.get_rates("gemini-1.5-pro", "gemini") == (1.25 / _MILLION, 5.0 / _MILLION)
    assert pricing.get_rates("gemini-1.5-flash", "gemini") == (0.075 / _MILLION, 0.3 / _MILLION)


# ── Bedrock lookups (Claude-on-Bedrock InvokeModel model ids) ────────────────
#
# Bedrock's on-demand global-endpoint pricing for Claude models matches the
# Anthropic direct API list price dollar-for-dollar (Anthropic sets the
# price) -- see pricing.json's top-level "note". Both the bare and the
# `global.`-prefixed model id (AWS's documented default form) resolve to the
# same rates.


def test_bedrock_rates_lookup_matches_anthropic_direct_list_price():
    assert pricing.get_rates("anthropic.claude-sonnet-4-6", "bedrock") == (
        SONNET_INPUT_PER_TOKEN,
        SONNET_OUTPUT_PER_TOKEN,
    )
    assert pricing.get_rates("global.anthropic.claude-sonnet-4-6", "bedrock") == (
        SONNET_INPUT_PER_TOKEN,
        SONNET_OUTPUT_PER_TOKEN,
    )
    assert pricing.get_rates("anthropic.claude-haiku-4-5-20251001-v1:0", "bedrock") == (
        HAIKU_INPUT_PER_TOKEN,
        HAIKU_OUTPUT_PER_TOKEN,
    )
    assert pricing.get_rates("anthropic.claude-opus-4-8", "bedrock") == (
        OPUS_INPUT_PER_TOKEN,
        OPUS_OUTPUT_PER_TOKEN,
    )


def test_bedrock_rates_scoped_lookup_does_not_leak_into_anthropic():
    # A Bedrock-prefixed id is not a first-party Anthropic model id -> scoped
    # to "anthropic" it's a miss -> falls back to Sonnet, same as any unknown
    # model (see test_known_model_wrong_provider_falls_back_to_sonnet).
    assert pricing.get_rates("anthropic.claude-sonnet-4-6", "anthropic") == (
        SONNET_INPUT_PER_TOKEN,
        SONNET_OUTPUT_PER_TOKEN,
    )


def test_lookup_without_provider_finds_model_across_providers():
    assert pricing.get_rates("gpt-4o") == (2.5 / _MILLION, 10.0 / _MILLION)
    assert pricing.get_rates("gemini-1.5-pro") == (1.25 / _MILLION, 5.0 / _MILLION)


def test_per_million_view_returns_stored_values():
    assert pricing.get_rates_per_million("gpt-4o", "openai") == (2.5, 10.0)
    assert pricing.get_rates_per_million(SONNET) == (3.0, 15.0)


# ── unknown-model fallback (preserves pre-registry SONNET default) ───────────


def test_unknown_model_falls_back_to_sonnet():
    assert pricing.get_rates("totally-made-up-model") == (
        SONNET_INPUT_PER_TOKEN,
        SONNET_OUTPUT_PER_TOKEN,
    )


def test_none_model_falls_back_to_sonnet():
    assert pricing.get_rates(None) == (SONNET_INPUT_PER_TOKEN, SONNET_OUTPUT_PER_TOKEN)


def test_known_model_wrong_provider_falls_back_to_sonnet():
    # gpt-4o is not an anthropic model → provider-scoped miss → fallback.
    assert pricing.get_rates("gpt-4o", "anthropic") == (
        SONNET_INPUT_PER_TOKEN,
        SONNET_OUTPUT_PER_TOKEN,
    )


# ── snapshot metadata ─────────────────────────────────────────────────────────


def test_pricing_version_present():
    assert pricing.pricing_version() == "2026-06b"


def test_registered_providers_and_models():
    assert set(pricing.registered_providers()) == {"anthropic", "openai", "gemini", "bedrock"}
    assert SONNET in pricing.registered_models("anthropic")
    assert "gpt-4o" in pricing.registered_models("openai")
    assert "gemini-1.5-pro" in pricing.registered_models("gemini")
    assert "anthropic.claude-sonnet-4-6" in pricing.registered_models("bedrock")
    # unscoped view is the union
    allm = pricing.registered_models()
    assert {SONNET, "gpt-4o", "gemini-1.5-pro", "anthropic.claude-sonnet-4-6"} <= set(allm)


# ── BudgetGovernor budget behaviour is unchanged for Anthropic ───────────────


def test_budget_governor_anthropic_estimate_uses_sonnet_rates():
    tape = Tape()
    req = b'{"model": "claude-sonnet-4-6", "messages": []}'
    tape.append_exchange(req, make_text_response("hi", input_tokens=100, output_tokens=20))
    tape.append_exchange(req, make_text_response("bye", input_tokens=100, output_tokens=20))
    est = BudgetGovernor.estimate(tape, k=1)
    # Two exchanges: billed tail calls = (n-1-0)+(n-1-1) = 1+0 = 1, times k=1.
    expected = 1 * (100 * SONNET_INPUT_PER_TOKEN + 20 * SONNET_OUTPUT_PER_TOKEN)
    assert est.est_usd == expected
