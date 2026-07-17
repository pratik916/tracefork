"""Fork-endpoint allowlist and single-fork cost estimator (tracefork-bge.36).

Nothing is fork-able through the click-to-fork server endpoints
(`server.py`'s `POST /api/run/{id}/fork[/estimate]`) unless the operator
names it explicitly — the same opt-in-only security posture `plugins.py`'s
entry-point registry already establishes for provider adapters/oracles/
matchers: merely running the server must never be enough, by itself, to let
a request re-execute arbitrary agent code.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable

from . import pricing
from .constants import SONNET
from .providers import get_adapter
from .tape import Tape

#: Comma-separated `name=module:fn` pairs, e.g. "my_agent=pkg.mod:run_agent".
#: Unset (the default) means: nothing allowlisted, every fork endpoint 403s.
FORK_AGENTS_ENV = "TRACEFORK_FORK_AGENTS"


class AgentNotAllowlistedError(RuntimeError):
    """Raised when an `agent_name` isn't in the fork endpoint's allowlist."""


def parse_allowlist_env(raw: str | None = None) -> dict[str, str]:
    """Parse `"name=module:fn,name2=module:fn2"` into `{name: "module:fn"}`.

    Reads `TRACEFORK_FORK_AGENTS` when `raw` is `None` (mirrors
    `plugins.py`'s `_env_allowlist` env-var pattern); an unset/empty value
    parses to `{}` — opt-in only, never a default-open allowlist.
    """
    text = os.environ.get(FORK_AGENTS_ENV, "") if raw is None else raw
    allowlist: dict[str, str] = {}
    for entry in text.split(","):
        entry = entry.strip()
        if not entry:
            continue
        name, sep, path = entry.partition("=")
        if not sep or not path.strip():
            continue
        allowlist[name.strip()] = path.strip()
    return allowlist


def resolve_agent_fn(agent_name: str, allowlist: dict[str, str]) -> Callable:
    """Resolve `agent_name`'s `"module:fn"` import path from `allowlist`.

    Raises `AgentNotAllowlistedError` (naming what IS allowlisted, the same
    style `Registry.get_or_raise` already uses) when `agent_name` isn't a
    key in `allowlist` — never a bare `KeyError`/`ImportError` leaking an
    internal detail to an HTTP caller.
    """
    path = allowlist.get(agent_name)
    if path is None:
        raise AgentNotAllowlistedError(
            f"agent {agent_name!r} is not allowlisted for forking; allowlisted: {sorted(allowlist)}"
        )
    module_path, _, fn_name = path.rpartition(":")
    module = importlib.import_module(module_path)
    return getattr(module, fn_name)


def estimate_single_fork_usd(tape: Tape, step: int, model: str | None = None) -> float:
    """Price ONE targeted fork at `step` (k=1): the `n - 1 - step` tail calls
    a single click-to-fork actually bills.

    Reuses only the already-public `providers.get_adapter`/`pricing.get_rates`
    seams — deliberately NOT `blame.BudgetGovernor.estimate`, which prices a
    full multi-step blame sweep (every step, `k` trials each) and would badly
    overstate a single click-to-fork's cost.
    """
    n = len(tape.exchanges)
    remaining = max(0, n - 1 - step)
    if remaining == 0 or n == 0:
        return 0.0

    adapter = get_adapter("anthropic")
    resolved_model = model
    if resolved_model is None:
        for req, _resp in tape.exchanges:
            detected = adapter.detect_model(req)
            if detected:
                resolved_model = detected
                break
    resolved_model = resolved_model or SONNET
    in_rate, out_rate = pricing.get_rates(resolved_model)

    # Average (input, output) tokens per exchange from recorded usage, the
    # same ~4-bytes-per-token fallback `blame._avg_tokens` uses on parse
    # failure — kept local (not imported) to leave blame.py untouched.
    ins: list[float] = []
    outs: list[float] = []
    for req, resp in tape.exchanges:
        try:
            norm = adapter.parse_response(resp)
            in_tok, out_tok = norm.input_tokens, norm.output_tokens
        except Exception:
            in_tok = out_tok = None
        ins.append(in_tok or max(1, len(req) // 4))
        outs.append(out_tok or max(1, len(resp) // 4))
    avg_in = sum(ins) / len(ins) if ins else 0.0
    avg_out = sum(outs) / len(outs) if outs else 0.0

    return remaining * (avg_in * in_rate + avg_out * out_rate)
