"""Provider-generic pricing registry backed by a pinned, bundled JSON snapshot.

Replaces the flat ``constants.PRICING_TABLE`` with a ``(provider, model) -> rates``
lookup so ``BudgetGovernor``/blame can price OpenAI and Gemini tapes, not just
Anthropic. The rate table is a **pinned snapshot shipped inside the package**
(``tracefork/data/pricing.json``) and loaded **offline** — there is no network
fetch, ever, so the whole test suite / validate / demo stay $0.

**Anthropic rates are byte-identical to the pre-registry values.** The snapshot
stores list price in USD per 1M tokens; dividing by ``1_000_000`` here reproduces
``constants.SONNET_INPUT_PER_TOKEN`` and friends bit-for-bit, so the budget
estimate ``BudgetGovernor`` computes is unchanged for every model already
anchored there. Unknown models fall back to the snapshot's declared
``fallback`` entry -- deliberately the MOST EXPENSIVE known Anthropic rate
(not the old mid-tier Sonnet default), so a model this snapshot hasn't
caught up to yet is priced as an upper bound rather than silently
undercounted; ``is_fallback_model()`` lets a caller detect and disclose
that a rate came from the fallback rather than a real match.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from typing import Any

__all__ = [
    "pricing_version",
    "get_rates",
    "CACHE_READ_MULTIPLIER",
    "CACHE_WRITE_MULTIPLIER_5M",
    "get_cache_rates",
    "parse_cache_tokens",
    "is_fallback_model",
    "get_rates_per_million",
    "registered_models",
    "registered_providers",
]


#: List price in the snapshot is quoted per this many tokens.
_PER_MILLION = 1_000_000

#: Package-relative location of the pinned snapshot (shipped in the wheel).
_DATA_PACKAGE = "tracefork"
_DATA_FILE = ("data", "pricing.json")


@lru_cache(maxsize=1)
def _snapshot() -> dict[str, Any]:
    """Load and cache the bundled pricing snapshot (offline, no network)."""
    resource = resources.files(_DATA_PACKAGE).joinpath(*_DATA_FILE)
    data = json.loads(resource.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "providers" not in data:
        raise ValueError("malformed pricing snapshot: missing 'providers'")
    return data


def pricing_version() -> str:
    """Version tag of the bundled snapshot (bump when rates change)."""
    return str(_snapshot().get("version", "unknown"))


def _fallback_entry(snap: dict[str, Any]) -> dict[str, Any]:
    fb = snap["fallback"]
    entry: dict[str, Any] = snap["providers"][fb["provider"]][fb["model"]]
    return entry


def _lookup_entry_ex(model: str | None, provider: str | None) -> tuple[dict[str, Any], bool]:
    """Find the raw per-1M rate entry for ``(provider, model)``, plus whether
    resolving it required falling back.

    An explicit ``provider`` scopes the search *strictly* to that provider (a
    miss falls back, it does not leak into another provider's table); with no
    provider, the first provider that lists ``model`` wins. Unknown models
    resolve to the snapshot's declared ``fallback`` entry (deliberately the
    most expensive known Anthropic rate -- see ``is_fallback_model``).

    The boolean is tracked explicitly here rather than derived by comparing
    the returned entry's identity against ``_fallback_entry(snap)`` at the
    call site: when ``model`` names the fallback model itself, a direct
    match and a genuine fallback resolve to the exact same dict object, so
    identity alone can't tell them apart.
    """
    snap = _snapshot()
    providers: dict[str, dict[str, Any]] = snap["providers"]
    if provider is not None:
        if model is not None:
            entry = providers.get(provider, {}).get(model)
            if entry is not None:
                return entry, False
        return _fallback_entry(snap), True
    if model is not None:
        for models in providers.values():
            if model in models:
                return models[model], False
    return _fallback_entry(snap), True


def _lookup_entry(model: str | None, provider: str | None) -> dict[str, Any]:
    """Find the raw per-1M rate entry for ``(provider, model)``. See
    ``_lookup_entry_ex`` for the fallback-tracking variant."""
    entry, _ = _lookup_entry_ex(model, provider)
    return entry


def get_rates(model: str | None, provider: str | None = None) -> tuple[float, float]:
    """Return ``(input_per_token, output_per_token)`` USD rates for a model.

    Falls back to the snapshot's declared fallback entry for unknown models
    (see ``is_fallback_model`` -- the fallback is deliberately the MOST
    EXPENSIVE known Anthropic rate, not a mid-tier guess, so an unrecognized
    model id never silently under-prices a spend estimate). Anthropic rates
    for every model actually IN the snapshot are bit-identical to
    ``constants.*_PER_TOKEN`` for the three legacy-anchored models.
    """
    entry = _lookup_entry(model, provider)
    return (entry["input"] / _PER_MILLION, entry["output"] / _PER_MILLION)


# ── Prompt-cache economics ───────────────────────────────────────────────────
#
# Anthropic's cache read/write multipliers are uniform ACROSS every model this
# snapshot prices (verified against the claude-api skill's authoritative
# rate table at implementation time, per this repo's standing "never a
# remembered table" rule for pricing data) -- a cache READ costs ~0.1x the
# model's base input rate, and a cache WRITE costs 1.25x for the default
# 5-minute TTL (2x for an opt-in 1-hour TTL). ``get_cache_rates`` prices the
# 5-minute-TTL write, since a recorded tape doesn't currently distinguish
# which TTL produced a given ``cache_creation_input_tokens`` count -- this is
# a documented, conservative (under-estimating, never over-estimating) gap
# for the rarer 1-hour-TTL case, not a silently-assumed one. (Claude Fable
# 5.1/Mythos 5.1's cheaper 0.025x cache-read rate is a documented
# model-specific exception; those model ids aren't in this snapshot's
# provider set yet, so it isn't modeled here.)
CACHE_READ_MULTIPLIER = 0.1
CACHE_WRITE_MULTIPLIER_5M = 1.25


def get_cache_rates(model: str | None, provider: str | None = None) -> tuple[float, float]:
    """Return ``(cache_read_per_token, cache_write_per_token)`` USD rates for
    a model -- ``get_rates``'s input rate scaled by the standard prompt-cache
    multipliers (see the module note above). Falls back exactly like
    ``get_rates`` for an unknown model (the fallback's input rate, scaled).
    """
    input_rate, _output_rate = get_rates(model, provider)
    return (input_rate * CACHE_READ_MULTIPLIER, input_rate * CACHE_WRITE_MULTIPLIER_5M)


def parse_cache_tokens(resp: bytes) -> tuple[int, int]:
    """Best-effort ``(cache_read_input_tokens, cache_creation_input_tokens)``
    parsed directly from a raw Anthropic-wire-format response body's
    ``usage`` object. Returns ``(0, 0)`` on any parse failure, or when either
    field is absent -- a tape recorded before prompt caching was in play, a
    non-Anthropic response body, or a streaming/opaque marker (the same
    tolerance ``blame._avg_tokens``'s own fallback already has for
    ``input_tokens``/``output_tokens``).

    Lives here rather than on ``providers.base.NormalizedResponse`` /
    ``ProviderAdapter.parse_response`` (the more natural long-term home once
    that surface grows these two fields) because ``providers/`` was outside
    this wave's file ownership -- see ``planning/HANDOFF.md``.
    """
    try:
        usage = json.loads(resp).get("usage", {})
        cache_read = int(usage.get("cache_read_input_tokens") or 0)
        cache_creation = int(usage.get("cache_creation_input_tokens") or 0)
        return (max(0, cache_read), max(0, cache_creation))
    except Exception:
        return (0, 0)


def is_fallback_model(model: str | None, provider: str | None = None) -> bool:
    """Would ``get_rates(model, provider)`` resolve via the fallback entry?

    Additive disclosure signal, separate from ``get_rates`` itself so every
    existing caller's return shape is unchanged. A caller that wants to warn
    "priced as fallback (upper bound, may overstate real cost)" -- the
    natural next step for ``BudgetGovernor.estimate``'s pre-flight banner --
    can check this without re-deriving the lookup logic. Returns ``True``
    whenever the resolved entry IS the snapshot's declared fallback, whether
    that's because ``model`` is unknown, ``model`` is ``None``, or ``model``
    is real but absent from the given ``provider``'s table.
    """
    _, was_fallback = _lookup_entry_ex(model, provider)
    return was_fallback


def get_rates_per_million(model: str | None, provider: str | None = None) -> tuple[float, float]:
    """Return ``(input, output)`` list price in USD per 1M tokens (as stored)."""
    entry = _lookup_entry(model, provider)
    return (float(entry["input"]), float(entry["output"]))


def registered_models(provider: str | None = None) -> list[str]:
    """Sorted model ids known to the snapshot (optionally scoped to a provider)."""
    snap = _snapshot()
    providers: dict[str, dict[str, Any]] = snap["providers"]
    if provider is not None:
        return sorted(providers.get(provider, {}))
    seen: set[str] = set()
    for models in providers.values():
        seen.update(models)
    return sorted(seen)


def registered_providers() -> list[str]:
    """Sorted provider names present in the pricing snapshot."""
    return sorted(_snapshot()["providers"])
