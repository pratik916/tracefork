"""Optional Pydantic AI framework adapter.

Two seams, both reusing tracefork's *existing* byte capture -- never a second one:

* **``bind``** routes a Pydantic AI ``Agent``'s (or an already-resolved
  ``Model`` instance's) underlying LLM client through a ``TraceforkTransport``
  -- the same ``client.copy(http_client=...)`` move ``recorder.py`` and every
  other openai/anthropic-shaped adapter here already use. Pydantic AI's model
  classes (``OpenAIModel``/``OpenAIChatModel``, ``AnthropicModel``, ...) each
  wrap a ``Provider`` (``OpenAIProvider``, ``AnthropicProvider``, ...) that
  holds the real ``openai``/``anthropic`` SDK client the provider's own httpx
  transport already flows through -- exactly the "provider layer is already
  httpx-based" seam that makes this adapter cheap to add. Neither the
  ``Model``'s nor the ``Provider``'s internal attribute name for that client
  is documented as a frozen public API, so ``bind`` walks a short list of
  candidate holder paths (the target itself, an ``Agent``-shaped ``.model``,
  a ``Model``-shaped ``.provider``/``._provider``, and both combined) and a
  short list of candidate client attribute names (``client``, ``_client``,
  ``openai_client``, ``_openai_client``, ``anthropic_client``,
  ``_anthropic_client``) -- the same defensive-search style
  ``adapters/openai_agents.py``'s ``_inject`` and ``adapters/adk.py``'s
  candidate-path walk already use, for exactly the same "not a frozen public
  API" reason. Pydantic AI's model/provider clients are async-first, so
  ``bind`` only searches for (and injects) the async client; record mode's
  best-effort inner-transport lookup mirrors that same async-only scope.
* **``on_step``** turns one framework-neutral, dict-or-object-shaped run event
  into a ``Step`` (a tape annotation), using the same key-lookup conventions
  ``adapters/base.py``'s ``StepDAG.from_run_tree`` already documents (``id``/
  ``step_id``/``run_id``, ``kind``/``run_type``/``type``, ``name``, ``model``,
  ``parent_id``/``parent_run_id``). Pydantic AI does not (as of this writing)
  expose a single documented public callback/tracing API shaped like
  LangChain's callbacks or the OpenAI Agents SDK's ``TracingProcessor`` -- its
  own graph-iteration nodes (``agent.iter()``'s ``ModelRequestNode``/
  ``CallToolsNode``/... from ``pydantic-graph``) and its OpenTelemetry
  instrumentation are two different, evolving surfaces -- so ``on_step``
  accepts the SAME framework-neutral node shape every other seam in this
  package already speaks, rather than guessing at either surface's exact
  attribute names. A caller wiring a live Pydantic AI run into the step-DAG
  maps its own node/span shape into that neutral shape before calling
  ``on_step`` -- observer-only, never a second capture path (the design
  invariant in ``adapters/base.py``).

``pydantic-ai`` is OPTIONAL (the ``pydantic-ai`` extra). Nothing here imports
it at module load: the availability guard is the only place a real import
happens, so ``import tracefork`` and the whole offline test suite run with it
NOT installed. ``bind``'s candidate-path/candidate-attr injection is duck-
typed and fully offline-testable with a synthetic client double (mirroring
``tests/test_adapters_openai_agents.py``'s ``_FakeAsyncOpenAIClient``).
Honesty over coverage: this is a synthetic-interface validation of the
binding logic, not a live-framework integration test.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..nondet import NondetSource
from ..tape import Tape
from .base import (
    BaseFrameworkAdapter,
    BindResult,
    Step,
    build_http_clients,
    register_framework_adapter,
)

PYDANTIC_AI_IMPORT_HINT = (
    "Pydantic AI support needs the optional 'pydantic-ai' extra: "
    "pip install 'tracefork[pydantic-ai]'"
)


# ── availability guard (mirrors adapters/openai_agents.py) ──────────────────


def pydantic_ai_available() -> bool:
    """Whether the optional ``pydantic_ai`` package is importable."""
    try:
        import pydantic_ai  # noqa: F401
    except ImportError:
        return False
    return True


def require_pydantic_ai() -> None:
    """Raise a helpful ``ImportError`` if ``pydantic_ai`` is missing.

    Attempts the import itself (rather than delegating to
    ``pydantic_ai_available()``) and chains the real cause via ``from exc``,
    so an installed-but-broken ``pydantic_ai`` package surfaces its own error
    instead of being masked as "not installed".
    """
    try:
        import pydantic_ai  # noqa: F401
    except ImportError as exc:
        raise ImportError(PYDANTIC_AI_IMPORT_HINT) from exc


# ── defensive extractors (work on dict-or-object payloads) ──────────────────


def _get(obj: Any, *keys: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        for key in keys:
            if key in obj:
                return obj[key]
        return default
    for key in keys:
        if hasattr(obj, key):
            return getattr(obj, key)
    return default


def _set_attr(obj: Any, name: str, value: Any) -> None:
    """Assign ``obj.name = value``, falling back through pydantic's guardrails."""
    try:
        setattr(obj, name, value)
    except Exception:  # pragma: no cover - pydantic-version dependent
        object.__setattr__(obj, name, value)


# ── bind: candidate holder-path + candidate client-attribute search ─────────

# Attribute paths (from the bound `target`) that might hold the object whose
# OWN attribute holds the real client -- an Agent exposes `.model`, a Model
# exposes `.provider`/`._provider`, and an Agent's `.model` may itself only
# resolve its provider under one of those. Not a frozen public API -- see the
# module docstring.
_HOLDER_PATHS: tuple[tuple[str, ...], ...] = (
    (),
    ("model",),
    ("provider",),
    ("_provider",),
    ("model", "provider"),
    ("model", "_provider"),
)

# Candidate attribute names an httpx-based provider/model wrapper might store
# its underlying openai/anthropic SDK client under.
_CLIENT_ATTRS = (
    "client",
    "_client",
    "openai_client",
    "_openai_client",
    "anthropic_client",
    "_anthropic_client",
)


def _resolve_holder(target: Any, path: tuple[str, ...]) -> Any:
    obj = target
    for attr in path:
        obj = getattr(obj, attr, None)
        if obj is None:
            return None
    return obj


def _find_client_attr(target: Any) -> tuple[Any, tuple[str, ...], str] | None:
    """Find the first (holder, path, attr_name) holding a client-shaped value.

    "Client-shaped" means it exposes ``.copy`` (the ``openai``/``anthropic``
    SDK client contract every ``.copy(http_client=...)`` injection in this
    codebase relies on).
    """
    for path in _HOLDER_PATHS:
        holder = _resolve_holder(target, path)
        if holder is None:
            continue
        for name in _CLIENT_ATTRS:
            current = getattr(holder, name, None)
            if current is not None and hasattr(current, "copy"):
                return holder, path, name
    return None


def _inject(target: Any, async_client: Any) -> list[str]:
    """Best-effort search-and-replace of the target's underlying async SDK client.

    Stops at the first client-shaped attribute found -- returns its dotted
    path (e.g. ``"model.provider.client"``) as a one-element list, or ``[]``
    when nothing was found.
    """
    found = _find_client_attr(target)
    if found is None:
        return []
    holder, path, name = found
    _set_attr(holder, name, getattr(holder, name).copy(http_client=async_client))
    return [".".join((*path, name)) if path else name]


def _underlying_transports(target: Any) -> tuple[Any, Any]:  # pragma: no cover - needs real package
    """Best-effort inner async httpx transport, for record mode only.

    Pydantic AI's model/provider clients are async-first (see the module
    docstring); this returns ``(None, inner_async)`` -- record mode never
    needs a sync inner transport for this adapter.
    """
    found = _find_client_attr(target)
    if found is None:
        return None, None
    holder, _path, name = found
    current = getattr(holder, name)
    inner_async = getattr(getattr(current, "_client", None), "_transport", None)
    return None, inner_async


# ── on_step: one framework-neutral event -> Step ────────────────────────────


def _event_to_step(event: Any) -> Step:
    raw_id = _get(event, "id", "step_id", "run_id", default=None)
    step_id = str(raw_id) if raw_id is not None else str(id(event))
    parent_id = _get(event, "parent_id", "parent_run_id", default=None)
    kind = str(_get(event, "kind", "run_type", "type", default="") or "")
    name = str(_get(event, "name", default="") or "")
    model = _get(event, "model", default=None)
    return Step(
        step_id=step_id,
        parent_id=str(parent_id) if parent_id is not None else None,
        kind=kind,
        name=name,
        model=str(model) if model else None,
        inputs=_get(event, "inputs", "input", default=None),
        outputs=_get(event, "outputs", "output", default=None),
    )


# ── the adapter ───────────────────────────────────────────────────────────────


class PydanticAIAdapter(BaseFrameworkAdapter):
    """Bind a Pydantic AI agent/model to tracefork's transport + annotate its run."""

    name = "pydantic_ai"

    def bind(
        self,
        target: Any,
        tape: Tape,
        mode: str = "replay",
        *,
        nondet: NondetSource | None = None,
        patch_uuid: bool = True,
        matcher: Any = None,
        redactor: Any = None,
        **kwargs: Any,
    ) -> BindResult:
        """Route ``target`` (an ``Agent`` or an already-resolved ``Model``) through tracefork.

        ``replay`` mode needs no inner transport and no live client; ``record``
        mode reuses the target's current underlying async httpx transport as
        the inner so live calls still reach the network (that path needs the
        real package and is not offline-testable). On replay, a
        ``ReplayNondet``-backed uuid patch (``patch_uuid=True``) makes
        framework-generated ids match the tape.
        """
        inner_async = None
        if mode == "record":  # pragma: no cover - needs real package
            _inner, inner_async = _underlying_transports(target)

        sync_client, async_client, sync_t, async_t = build_http_clients(
            tape, mode, async_inner=inner_async, matcher=matcher, redactor=redactor
        )
        injected = _inject(target, async_client)

        active_nondet = nondet
        if mode == "replay":
            if active_nondet is None:
                active_nondet = self._replay_nondet(tape)
            if patch_uuid:
                self._install_uuid_patch(active_nondet)

        notes = (
            ""
            if injected
            else (
                "no known LLM-client attribute found on target "
                f"({type(target).__name__}); nothing was injected"
            )
        )
        return BindResult(
            mode=mode,
            http_client=sync_client,
            http_async_client=async_client,
            transport=sync_t,
            async_transport=async_t,
            nondet=active_nondet,
            injected_fields=tuple(injected),
            notes=notes,
        )

    def on_step(self, event: Mapping[str, Any]) -> Step:
        """Map one framework-neutral run event/node to a ``Step`` (see module docstring)."""
        return self.record_step(_event_to_step(event))


# Register the built-in adapter at import time (never via the entry-point path).
register_framework_adapter(PydanticAIAdapter())


__all__ = [
    "PYDANTIC_AI_IMPORT_HINT",
    "PydanticAIAdapter",
    "pydantic_ai_available",
    "require_pydantic_ai",
]
