"""Pluggable tool-effect extraction + cross-branch conflict detection.

Read-only report layer over an already-loaded `Tape` (same shape as
`coverage.py`): no engine/digest changes. It normalizes two independent
"tool call" representations already recorded on any tape into one `Effect`
shape:

1. Anthropic `tool_use` content blocks inside `Tape.exchanges` response bytes
   (`{"type": "tool_use", "name": ..., "input": {...}}`, see
   `providers/anthropic.py:build_tool_use_response` and
   `redact.py:_redact_content_blocks`).
2. JSON-RPC `tools/call` request frames inside `Tape.tool_exchanges`
   (`{"method": "tools/call", "params": {"name": ..., "arguments": {...}}}`,
   see `tools.py`).

A per-tool-name pluggable registry (`EFFECT_EXTRACTOR_REGISTRY`,
`register_effect_extractor()`) maps a tool's arguments to a comparable
resource string. This is deliberately a lightweight `dict` registry, NOT
`plugins.py`'s heavier entry-point `Registry` (the one `blame.py`'s
`ORACLE_REGISTRY` uses) -- this seam never needs cross-package
distribution, only same-process registration. An unregistered tool falls
back to a small default key-probe (`path`/`file_path`/`filename`/`url`/
`uri`/`key`/`resource`/`id`), and if none of those match, to a canonical
`json.dumps(arguments, sort_keys=True)` string so byte-identical calls still
compare equal without inventing a false resource identity.

`extract_effects(tape)` needs no step-range slicing: `fork.py` already
guarantees a `Branch.delta_tape` "holds only the exchanges from the
divergence step onward", so calling it directly on a store-reloaded
branch's `delta_tape` (`store.py`'s `TapeStore.load_branch()["delta_tape"]`)
already scopes to exactly that branch's post-divergence writes --
`extract_effects`/`diff_effects` are decoupled from `TapeStore`/`Branch`
themselves, exactly like `diff.py`'s `branch_diff`/`tape_diff` (which take
plain `Tape` objects).

`diff_effects(tape_a, tape_b)` computes both sides' effects and flags every
`(tool_name, resource)` pair present on both as an `EffectOverlap` --
read-only, no merge/apply logic. Scoped to the reviewer-sanity signal this
is meant to give: "did these two branches' post-divergence tool activity
touch the same thing", not an automatic merge/rebase decision.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .tape import Tape

#: `Effect.source` for a call found as an Anthropic `tool_use` content block
#: inside `Tape.exchanges` response bytes.
SOURCE_LLM_TOOL_USE = "llm_tool_use"
#: `Effect.source` for a call found as a JSON-RPC `tools/call` request frame
#: inside `Tape.tool_exchanges`.
SOURCE_TOOL_FRAME = "tool_frame"

#: Default key-probe order for an unregistered tool's resource extraction --
#: the first of these keys present in `arguments` (in this order) resolves
#: the resource. Deliberately small and generic; a tool with richer/nested
#: identity should register its own extractor instead of stretching this list.
_DEFAULT_RESOURCE_KEYS: tuple[str, ...] = (
    "path",
    "file_path",
    "filename",
    "url",
    "uri",
    "key",
    "resource",
    "id",
)


@dataclass(frozen=True)
class Effect:
    """One normalized tool-call effect: a tool touching a resource.

    `index` is the call's position within its OWN source's ordered log
    (encounter order of `tool_use` blocks across `Tape.exchanges`, or
    `Tape.tool_exchanges` index respectively) -- not a shared/global index
    across both sources. `resource_is_fallback` is `True` when neither a
    registered extractor nor the default key-probe matched, so `resource`
    is the canonical-JSON fallback of the raw arguments (see `_resource_for`).
    """

    source: str
    index: int
    tool_name: str
    resource: str
    resource_is_fallback: bool = False


#: A tool's resolved (JSON-decoded) arguments -> a resource string, or `None`
#: to defer to the default key-probe / canonical-JSON fallback.
EffectExtractor = Callable[[dict[str, Any]], "str | None"]

#: Pluggable per-tool-name resource-extractor registry. A plain `dict` --
#: see the module docstring for why this is intentionally NOT `plugins.py`'s
#: `Registry`.
EFFECT_EXTRACTOR_REGISTRY: dict[str, EffectExtractor] = {}


def register_effect_extractor(tool_name: str, extractor: EffectExtractor) -> None:
    """Register `extractor` to resolve `tool_name`'s arguments to a resource
    string, overriding the default key-probe/canonical-JSON fallback for that
    tool name. `extractor` may itself return `None` to defer to the fallback
    for a particular call (e.g. missing an expected field)."""
    EFFECT_EXTRACTOR_REGISTRY[tool_name] = extractor


def _default_resource(arguments: dict[str, Any]) -> str | None:
    """The first `_DEFAULT_RESOURCE_KEYS` entry present in `arguments`,
    stringified (verbatim if already a string, else canonical JSON)."""
    for key in _DEFAULT_RESOURCE_KEYS:
        if key in arguments:
            value = arguments[key]
            return value if isinstance(value, str) else json.dumps(value, sort_keys=True)
    return None


def _resource_for(tool_name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
    """Resolve `arguments` to `(resource, resource_is_fallback)` for
    `tool_name`: a registered extractor wins if it returns non-`None`;
    otherwise the default key-probe; otherwise the canonical-JSON fallback
    (`resource_is_fallback=True`)."""
    extractor = EFFECT_EXTRACTOR_REGISTRY.get(tool_name)
    if extractor is not None:
        resolved = extractor(arguments)
        if resolved is not None:
            return resolved, False
    default = _default_resource(arguments)
    if default is not None:
        return default, False
    return json.dumps(arguments, sort_keys=True), True


def _effects_from_tool_use_blocks(tape: Tape) -> list[Effect]:
    """Every Anthropic `tool_use` content block across `tape.exchanges`'
    response bytes, in encounter order."""
    effects: list[Effect] = []
    index = 0
    for _request, response in tape.exchanges:
        try:
            obj = json.loads(response)
        except (ValueError, UnicodeDecodeError):
            continue
        if not isinstance(obj, dict):
            continue
        content = obj.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            tool_name = block.get("name")
            if not isinstance(tool_name, str):
                continue
            arguments = block.get("input")
            if not isinstance(arguments, dict):
                arguments = {}
            resource, is_fallback = _resource_for(tool_name, arguments)
            effects.append(
                Effect(
                    source=SOURCE_LLM_TOOL_USE,
                    index=index,
                    tool_name=tool_name,
                    resource=resource,
                    resource_is_fallback=is_fallback,
                )
            )
            index += 1
    return effects


def _effects_from_tool_frames(tape: Tape) -> list[Effect]:
    """Every JSON-RPC `tools/call` request frame across `tape.tool_exchanges`,
    in tape order."""
    effects: list[Effect] = []
    for index, (request, _response) in enumerate(tape.tool_exchanges):
        try:
            obj = json.loads(request)
        except (ValueError, UnicodeDecodeError):
            continue
        if not isinstance(obj, dict) or obj.get("method") != "tools/call":
            continue
        params = obj.get("params")
        if not isinstance(params, dict):
            continue
        tool_name = params.get("name")
        if not isinstance(tool_name, str):
            continue
        arguments = params.get("arguments")
        if not isinstance(arguments, dict):
            arguments = {}
        resource, is_fallback = _resource_for(tool_name, arguments)
        effects.append(
            Effect(
                source=SOURCE_TOOL_FRAME,
                index=index,
                tool_name=tool_name,
                resource=resource,
                resource_is_fallback=is_fallback,
            )
        )
    return effects


def extract_effects(tape: Tape) -> tuple[Effect, ...]:
    """Normalize both tool-call representations recorded on `tape` into one
    `Effect` shape -- see the module docstring for the two source kinds.

    Read-only: only reads `tape.exchanges`/`tape.tool_exchanges` -- never
    touches `digest()`/`to_bytes()`/`from_bytes()`. No step-range slicing:
    pass an already-scoped tape (e.g. a branch's `delta_tape`) to limit the
    result to that scope.
    """
    return tuple(_effects_from_tool_use_blocks(tape) + _effects_from_tool_frames(tape))


@dataclass(frozen=True)
class EffectOverlap:
    """One `(tool_name, resource)` pair touched by BOTH tapes in a
    `diff_effects` comparison, citing one matching `Effect` from each side."""

    tool_name: str
    resource: str
    effect_a: Effect
    effect_b: Effect


@dataclass(frozen=True)
class ConflictReport:
    """Read-only tool-effect overlap report between two tapes."""

    effects_a: tuple[Effect, ...]
    effects_b: tuple[Effect, ...]
    overlaps: tuple[EffectOverlap, ...] = field(default_factory=tuple)

    @property
    def has_conflict(self) -> bool:
        """`True` iff at least one `(tool_name, resource)` pair is touched by
        both sides."""
        return len(self.overlaps) > 0


def diff_effects(tape_a: Tape, tape_b: Tape) -> ConflictReport:
    """Compute both tapes' effects (via `extract_effects`) and flag every
    `(tool_name, resource)` pair present on both as an `EffectOverlap`.

    Read-only reviewer-sanity signal -- no merge/apply logic. Typically
    called on two branches' post-divergence `delta_tape`s (see the module
    docstring), but takes plain `Tape` objects so it works on any pair.
    """
    effects_a = extract_effects(tape_a)
    effects_b = extract_effects(tape_b)
    first_b_by_key: dict[tuple[str, str], Effect] = {}
    for effect in effects_b:
        key = (effect.tool_name, effect.resource)
        first_b_by_key.setdefault(key, effect)

    overlaps: list[EffectOverlap] = []
    seen_keys: set[tuple[str, str]] = set()
    for effect in effects_a:
        key = (effect.tool_name, effect.resource)
        if key in seen_keys:
            continue
        match = first_b_by_key.get(key)
        if match is not None:
            overlaps.append(
                EffectOverlap(
                    tool_name=effect.tool_name,
                    resource=effect.resource,
                    effect_a=effect,
                    effect_b=match,
                )
            )
            seen_keys.add(key)
    return ConflictReport(effects_a=effects_a, effects_b=effects_b, overlaps=tuple(overlaps))
