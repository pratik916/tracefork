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
    # to "anthropic" it's a miss -> falls back to the fallback entry, same
    # as any unknown model (see
    # test_known_model_wrong_provider_falls_back_to_the_most_expensive_known_rate).
    assert pricing.get_rates("anthropic.claude-sonnet-4-6", "anthropic") == (
        pricing.get_rates("claude-fable-5", "anthropic")
    )


def test_lookup_without_provider_finds_model_across_providers():
    assert pricing.get_rates("gpt-4o") == (2.5 / _MILLION, 10.0 / _MILLION)
    assert pricing.get_rates("gemini-1.5-pro") == (1.25 / _MILLION, 5.0 / _MILLION)


def test_per_million_view_returns_stored_values():
    assert pricing.get_rates_per_million("gpt-4o", "openai") == (2.5, 10.0)
    assert pricing.get_rates_per_million(SONNET) == (3.0, 15.0)


# ── unknown-model fallback (preserves pre-registry SONNET default) ───────────


def test_unknown_model_falls_back_to_the_most_expensive_known_rate():
    # Fail-safe fallback (item 27): an unrecognized model id resolves to
    # the MOST EXPENSIVE known Anthropic rate, not Sonnet -- see
    # pricing.json's "fallback" key and pricing.is_fallback_model().
    assert pricing.get_rates("totally-made-up-model") == pricing.get_rates(
        "claude-fable-5", "anthropic"
    )
    assert pricing.is_fallback_model("totally-made-up-model") is True


def test_none_model_falls_back_to_the_most_expensive_known_rate():
    assert pricing.get_rates(None) == pricing.get_rates("claude-fable-5", "anthropic")


def test_known_model_wrong_provider_falls_back_to_the_most_expensive_known_rate():
    # gpt-4o is not an anthropic model → provider-scoped miss → fallback.
    assert pricing.get_rates("gpt-4o", "anthropic") == pricing.get_rates(
        "claude-fable-5", "anthropic"
    )


# ── snapshot metadata ─────────────────────────────────────────────────────────


def test_pricing_version_present():
    assert pricing.pricing_version() == "2026-08c"


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


# ── prompt-cache economics (tracefork-sis.66) ────────────────────────────────


def test_get_cache_rates_scales_input_rate_by_the_standard_multipliers():
    in_rate, _out_rate = pricing.get_rates(SONNET)
    cache_read_rate, cache_write_rate = pricing.get_cache_rates(SONNET)
    assert cache_read_rate == pytest.approx(in_rate * 0.1)
    assert cache_write_rate == pytest.approx(in_rate * 1.25)
    # A cache read must always be cheaper than a fresh input token, and a
    # cache write always more expensive -- the whole point of the feature.
    assert cache_read_rate < in_rate
    assert cache_write_rate > in_rate


def test_get_cache_rates_falls_back_like_get_rates_for_unknown_model():
    assert pricing.get_cache_rates("totally-unknown-model") == pricing.get_cache_rates(
        "claude-fable-5", "anthropic"
    )


def test_get_cache_rates_scoped_by_provider_and_model_matches_get_rates_scaling():
    for model in (HAIKU, OPUS, SONNET):
        in_rate, _out = pricing.get_rates(model)
        cache_read_rate, cache_write_rate = pricing.get_cache_rates(model)
        assert cache_read_rate == pytest.approx(in_rate * pricing.CACHE_READ_MULTIPLIER)
        assert cache_write_rate == pytest.approx(in_rate * pricing.CACHE_WRITE_MULTIPLIER_5M)


def _text_response_with_cache(
    text: str,
    *,
    model: str = SONNET,
    input_tokens: int,
    output_tokens: int,
    cache_read_input_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
) -> bytes:
    """Build a normal Anthropic text response then patch in the two cache
    usage fields the shared builder doesn't currently accept (see
    pricing.parse_cache_tokens's docstring -- providers/ is outside this
    wave's file ownership)."""
    import json as _json

    from tracefork.providers import get_adapter

    raw = get_adapter("anthropic").build_text_response(
        text, model=model, input_tokens=input_tokens, output_tokens=output_tokens
    )
    data = _json.loads(raw)
    data["usage"]["cache_read_input_tokens"] = cache_read_input_tokens
    data["usage"]["cache_creation_input_tokens"] = cache_creation_input_tokens
    return _json.dumps(data).encode()


def test_parse_cache_tokens_reads_both_fields():
    resp = _text_response_with_cache(
        "hi",
        input_tokens=10,
        output_tokens=5,
        cache_read_input_tokens=1000,
        cache_creation_input_tokens=200,
    )
    assert pricing.parse_cache_tokens(resp) == (1000, 200)


def test_parse_cache_tokens_defaults_to_zero_when_absent():
    resp = make_text_response("hi", input_tokens=10, output_tokens=5)  # no cache fields
    assert pricing.parse_cache_tokens(resp) == (0, 0)


def test_parse_cache_tokens_defaults_to_zero_on_garbage_bytes():
    assert pricing.parse_cache_tokens(b"not json at all") == (0, 0)
    assert pricing.parse_cache_tokens(b"") == (0, 0)


def test_budget_governor_estimate_accounts_for_cache_tokens_at_cached_rate():
    """The bead's own literal acceptance criterion: BudgetGovernor's cost
    estimate must price cache_read_input_tokens/cache_creation_input_tokens
    at the cheaper/pricier cached rates, not silently ignore them."""
    tape = Tape()
    req = b'{"model": "claude-sonnet-4-6", "messages": []}'
    tape.append_exchange(
        req,
        _text_response_with_cache(
            "hi",
            input_tokens=100,
            output_tokens=20,
            cache_read_input_tokens=5000,
            cache_creation_input_tokens=1000,
        ),
    )
    tape.append_exchange(
        req,
        _text_response_with_cache(
            "bye",
            input_tokens=100,
            output_tokens=20,
            cache_read_input_tokens=5000,
            cache_creation_input_tokens=1000,
        ),
    )
    est = BudgetGovernor.estimate(tape, k=1)
    cache_read_rate, cache_write_rate = pricing.get_cache_rates(SONNET)
    # Two exchanges: billed tail calls = (n-1-0)+(n-1-1) = 1+0 = 1, times k=1.
    expected = 1 * (
        100 * SONNET_INPUT_PER_TOKEN
        + 20 * SONNET_OUTPUT_PER_TOKEN
        + 5000 * cache_read_rate
        + 1000 * cache_write_rate
    )
    assert est.est_usd == pytest.approx(expected)
    # Sanity: ignoring cache tokens entirely would under-report real spend.
    without_cache = 1 * (100 * SONNET_INPUT_PER_TOKEN + 20 * SONNET_OUTPUT_PER_TOKEN)
    assert est.est_usd > without_cache


def test_budget_governor_estimate_unchanged_when_no_cache_tokens_present():
    """Regression guard: a tape with no cache activity at all must estimate
    EXACTLY as before this fix -- zero cache tokens contribute zero cost."""
    tape = Tape()
    req = b'{"model": "claude-sonnet-4-6", "messages": []}'
    tape.append_exchange(req, make_text_response("hi", input_tokens=100, output_tokens=20))
    tape.append_exchange(req, make_text_response("bye", input_tokens=100, output_tokens=20))
    est = BudgetGovernor.estimate(tape, k=1)
    expected = 1 * (100 * SONNET_INPUT_PER_TOKEN + 20 * SONNET_OUTPUT_PER_TOKEN)
    assert est.est_usd == pytest.approx(expected)
