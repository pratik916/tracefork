"""Provider-generic pricing registry backed by a pinned, bundled JSON snapshot.

Replaces the flat ``constants.PRICING_TABLE`` with a ``(provider, model) -> rates``
lookup so ``BudgetGovernor``/blame can price OpenAI and Gemini tapes, not just
Anthropic. The rate table is a **pinned snapshot shipped inside the package**
(``tracefork/data/pricing.json``) and loaded **offline** — there is no network
fetch, ever, so the whole test suite / validate / demo stay $0.

**Anthropic rates are byte-identical to the pre-registry values.** The snapshot
stores list price in USD per 1M tokens; dividing by ``1_000_000`` here reproduces
``constants.SONNET_INPUT_PER_TOKEN`` and friends bit-for-bit, so the budget
estimate ``BudgetGovernor`` computes is unchanged. Unknown models fall back to
the snapshot's declared default (Sonnet), preserving the old
``PRICING_TABLE.get(model, PRICING_TABLE[SONNET])`` behavior.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from typing import Any

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


def _lookup_entry(model: str | None, provider: str | None) -> dict[str, Any]:
    """Find the raw per-1M rate entry for ``(provider, model)``.

    An explicit ``provider`` scopes the search *strictly* to that provider (a
    miss falls back, it does not leak into another provider's table); with no
    provider, the first provider that lists ``model`` wins. Unknown models
    resolve to the snapshot's fallback (Sonnet), matching the pre-registry
    ``PRICING_TABLE.get(model, PRICING_TABLE[SONNET])`` default.
    """
    snap = _snapshot()
    providers: dict[str, dict[str, Any]] = snap["providers"]
    if provider is not None:
        if model is not None:
            entry = providers.get(provider, {}).get(model)
            if entry is not None:
                return entry
        return _fallback_entry(snap)
    if model is not None:
        for models in providers.values():
            if model in models:
                return models[model]
    return _fallback_entry(snap)


def get_rates(model: str | None, provider: str | None = None) -> tuple[float, float]:
    """Return ``(input_per_token, output_per_token)`` USD rates for a model.

    Falls back to the snapshot's default model (Sonnet) for unknown models,
    preserving the pre-registry ``BudgetGovernor`` behavior. Anthropic rates are
    bit-identical to ``constants.*_PER_TOKEN``.
    """
    entry = _lookup_entry(model, provider)
    return (entry["input"] / _PER_MILLION, entry["output"] / _PER_MILLION)


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
