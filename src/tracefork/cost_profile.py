"""Per-run cost/profile aggregation for the report's cost-profile panel
(tracefork-bge.52).

Built entirely on the existing `providers.get_adapter`/`pricing.get_rates`
seams `blame.BudgetGovernor.estimate` already uses — no new pricing logic.
Walks `tape.exchanges` (the same `(req_bytes, resp_bytes)` pairs
`report._tape_to_data`/`blame._avg_tokens`/`blame._detect_model` already
iterate), normalizes each response via the adapter (falling back to
`detect_model`/`constants.SONNET` and a len(bytes)//4 token estimate on parse
failure — mirroring `blame._avg_tokens`'s exact fallback), and groups the
priced exchanges into per-model (`ModelCost`) and per-tool (`ToolCost`) rows.

Per-tool cost attribution is an honest over-count for multi-tool exchanges:
Anthropic bills per-exchange, not per-tool-call, so an exchange invoking more
than one tool attributes its FULL modeled cost to each tool name it called —
no token-level sub-split is invented.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import pricing
from .constants import SONNET
from .providers import get_adapter
from .providers.base import NormalizedResponse, ProviderAdapter
from .tape import Tape


@dataclass(frozen=True)
class ModelCost:
    model: str
    n_exchanges: int
    input_tokens: int
    output_tokens: int
    total_cost_usd: float


@dataclass(frozen=True)
class ToolCost:
    tool_name: str
    n_calls: int
    total_cost_usd: float


@dataclass(frozen=True)
class CostProfile:
    by_model: tuple[ModelCost, ...] = ()
    by_tool: tuple[ToolCost, ...] = ()
    total_cost_usd: float = 0.0


def _normalize_exchange(adapter: ProviderAdapter, req: bytes, resp: bytes) -> NormalizedResponse:
    """Best-effort `NormalizedResponse` for one exchange.

    Mirrors `blame._avg_tokens`'s exact fallback: try the adapter's
    `parse_response`, filling any missing model/token fields via
    `detect_model`/`constants.SONNET` and a ~4-bytes-per-token estimate; a
    hard parse failure (e.g. a streaming/opaque marker) falls all the way
    back to the same estimate from raw bytes.
    """
    try:
        norm = adapter.parse_response(resp)
        if norm.input_tokens is not None and norm.output_tokens is not None and norm.model:
            return norm
        in_tok = norm.input_tokens if norm.input_tokens is not None else max(1, len(req) // 4)
        out_tok = norm.output_tokens if norm.output_tokens is not None else max(1, len(resp) // 4)
        return NormalizedResponse(
            model=norm.model or adapter.detect_model(req) or SONNET,
            content=norm.content,
            input_tokens=in_tok,
            output_tokens=out_tok,
            finish_reason=norm.finish_reason,
            message_id=norm.message_id,
        )
    except Exception:
        return NormalizedResponse(
            model=adapter.detect_model(req) or SONNET,
            content=(),
            input_tokens=max(1, len(req) // 4),
            output_tokens=max(1, len(resp) // 4),
        )


def compute_cost_profile(tape: Tape, *, provider: str = "anthropic") -> CostProfile:
    """Aggregate `tape`'s exchanges into a per-model/per-tool cost profile."""
    if not tape.exchanges:
        return CostProfile()

    adapter = get_adapter(provider)
    model_totals: dict[str, dict[str, float]] = {}
    tool_totals: dict[str, dict[str, float]] = {}
    total_cost = 0.0

    for req, resp in tape.exchanges:
        norm = _normalize_exchange(adapter, req, resp)
        model = norm.model or SONNET
        in_rate, out_rate = pricing.get_rates(model, provider)
        in_tok = norm.input_tokens or 0
        out_tok = norm.output_tokens or 0
        cost = in_tok * in_rate + out_tok * out_rate
        total_cost += cost

        mt = model_totals.setdefault(model, {"n": 0.0, "in": 0.0, "out": 0.0, "cost": 0.0})
        mt["n"] += 1
        mt["in"] += in_tok
        mt["out"] += out_tok
        mt["cost"] += cost

        tool_names = {p.tool_name for p in norm.content if p.type == "tool_use" and p.tool_name}
        for name in tool_names:
            tt = tool_totals.setdefault(name, {"n": 0.0, "cost": 0.0})
            tt["n"] += 1
            tt["cost"] += cost

    by_model = tuple(
        ModelCost(
            model=m,
            n_exchanges=int(v["n"]),
            input_tokens=int(v["in"]),
            output_tokens=int(v["out"]),
            total_cost_usd=v["cost"],
        )
        for m, v in sorted(model_totals.items())
    )
    by_tool = tuple(
        ToolCost(tool_name=t, n_calls=int(v["n"]), total_cost_usd=v["cost"])
        for t, v in sorted(tool_totals.items())
    )
    return CostProfile(by_model=by_model, by_tool=by_tool, total_cost_usd=total_cost)


def cost_profile_to_dict(profile: CostProfile) -> dict[str, Any]:
    """JSON-safe view of a `CostProfile`, mirroring
    `replay.verification_result_to_dict`'s dataclass -> dict conversion."""
    return {
        "by_model": [
            {
                "model": m.model,
                "n_exchanges": m.n_exchanges,
                "input_tokens": m.input_tokens,
                "output_tokens": m.output_tokens,
                "total_cost_usd": m.total_cost_usd,
            }
            for m in profile.by_model
        ],
        "by_tool": [
            {
                "tool_name": t.tool_name,
                "n_calls": t.n_calls,
                "total_cost_usd": t.total_cost_usd,
            }
            for t in profile.by_tool
        ],
        "total_cost_usd": profile.total_cost_usd,
    }
