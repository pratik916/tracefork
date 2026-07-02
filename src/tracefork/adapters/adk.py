"""Optional Google ADK (Agent Development Kit) adapter.

Two seams, both reusing tracefork's *existing* byte capture — never a second one:

* **``bind``** routes an ADK agent's underlying ``google.genai.Client`` through a
  ``TraceforkTransport``. ADK's model calls never touch httpx directly — they go
  through the ``google-genai`` SDK, whose ``Client`` wraps a private
  ``BaseApiClient`` that stores its transport as two plain instance attributes,
  ``_httpx_client`` (sync) and ``_async_httpx_client`` (async) — see
  ``google.genai._api_client.BaseApiClient.__init__`` (verified against the SDK's
  own source; not exposed as `httpx.Client`/`AsyncClient` fields in any public
  type). Those two attributes are not documented as a frozen public API (ADK's
  own ``Gemini`` docstring instead documents *subclassing* ``Gemini`` and
  overriding its ``api_client`` cached property to reach ``google.genai.Client``
  options ADK doesn't expose as fields), so ``bind`` walks a short list of
  candidate attribute paths — the target itself, a ``genai.Client``
  (``._api_client``), an ADK ``Gemini`` model wrapper (``.api_client._api_client``),
  or an ``LlmAgent``-shaped object whose ``.model`` already holds a resolved
  ``Gemini`` instance (``.model.api_client._api_client``) — and swaps in
  tracefork's own sync/async httpx clients wherever it finds an object exposing
  ``_httpx_client``/``_async_httpx_client``. This is the same "transport
  injection at the httpx boundary" move every other adapter makes (mirroring
  ``client.copy(http_client=…)`` for the openai-shaped adapters), just without a
  ``.copy()`` convenience method — ``google.genai``'s ``BaseApiClient`` takes the
  swap as a plain attribute assignment instead. The wire bytes that cross this
  seam are the same Gemini ``generateContent`` JSON ``providers/gemini.py``
  already parses.

  Caveat (ADK/google-genai design, not a tracefork limitation): ADK's ``Gemini``
  stores its client under a ``functools.cached_property`` named ``api_client``
  that *lazily constructs* a real ``google.genai.Client`` on first access, and
  that construction requires some credential-shaped value to be present (a real
  or placeholder API key, or Vertex AI project/location) even though no network
  call happens until a request is actually sent — so a caller must ensure their
  ``LlmAgent``/``Gemini`` can construct a client at all (e.g. a placeholder
  ``GOOGLE_API_KEY`` env var is sufficient for pure replay) before calling
  ``bind``, or hand ``bind`` the already-constructed ``google.genai.Client``
  directly.

* **``on_step`` / ``make_plugin``** turn ADK's documented ``BasePlugin`` hooks
  (``before_agent_callback``/``after_agent_callback``,
  ``before_model_callback``/``after_model_callback``,
  ``before_tool_callback``/``after_tool_callback`` — see
  ``google.adk.plugins.base_plugin.BasePlugin``) into neutral ``Step``s.
  ``BasePlugin`` is the cleaner documented seam versus ADK's per-agent
  ``before_model_callback=``/... constructor kwargs: a plugin is registered
  *once* on the ``Runner`` (``Runner(..., plugins=[plugin])``) and observes
  every agent/model/tool boundary across the whole run, rather than needing to
  be threaded through every individual ``LlmAgent``. The plugin is
  OBSERVER-ONLY here — every callback returns ``None`` (ADK's contract for
  "just observe"; a non-``None`` return short-circuits execution, which this
  adapter never does) — so it feeds the step-DAG, never a second capture path
  (the design invariant in ``adapters/base.py``).

Honesty note: ADK's ``CallbackContext``/``ToolContext`` (verified against
``google.adk.agents.readonly_context.ReadonlyContext``) expose ``invocation_id``
(one id per top-level run) and ``agent_name`` (one name per agent *type*, not
per call) — neither is a unique id per model/tool invocation the way
LangChain's callback ``run_id`` is. So ``make_plugin``'s real-event id
derivation is best-effort: a LIFO stack keyed by ``(kind, invocation_id,
agent_name)`` pairs each ``before_*``/``after_*`` call, which is correct for the
common case (an agent's model/tool calls inside one invocation are sequential,
not concurrently interleaved) but not a verified unique-id contract — the same
class of best-effort ``TraceforkCrewEventCore``/``TraceforkInterventionCore``
already document for CrewAI/AutoGen. The framework-neutral core
(``TraceforkAdkCore``) and its ``on_step`` dict dispatch are fully
offline-tested against explicit ``id``/``parent_id`` events — the DAG-building
logic itself is proven; only the real-event-to-dict mapping inside the guarded
``make_plugin`` factory is unverified without the real package installed
(import-guarded, ``pytest.importorskip`` in the test suite). Honesty over
coverage: this is a synthetic-interface validation of the binding logic, not a
live-framework integration test.

``google-adk`` is OPTIONAL (the ``adk`` extra, which pulls in ``google-genai``
transitively). Nothing here imports it at module load: the availability guard
and ``make_plugin`` are the only places a real import happens, so
``import tracefork`` and the whole offline test suite run with it NOT
installed. ``bind``'s candidate-path injection never imports ``google.adk`` or
``google.genai`` either — it is duck-typed, exactly like the CrewAI/AutoGen/
OpenAI-Agents adapters' ``bind``.
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
    StepDAG,
    build_http_clients,
    register_framework_adapter,
)

ADK_IMPORT_HINT = "Google ADK support needs the optional 'adk' extra: pip install 'tracefork[adk]'"


# ── availability guard (mirrors adapters/langchain.py) ──────────────────────


def adk_available() -> bool:
    """Whether the optional ``google.adk`` package is importable."""
    try:
        import google.adk  # noqa: F401
    except ImportError:
        return False
    return True


def require_adk() -> None:
    """Raise a helpful ``ImportError`` if ``google.adk`` is missing.

    Attempts the import itself (rather than delegating to ``adk_available()``)
    and chains the real cause via ``from exc``, so an installed-but-broken
    ``google.adk`` surfaces its own error instead of being masked as "not
    installed".
    """
    try:
        import google.adk  # noqa: F401
    except ImportError as exc:
        raise ImportError(ADK_IMPORT_HINT) from exc


# ── defensive extractors (work on dict-or-object payloads) ─────────────────────


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


# ── bind: candidate-path injection into the google-genai BaseApiClient ─────────

# Attribute-hop paths tried, in order, to find an object exposing
# ``_httpx_client``/``_async_httpx_client`` (a google-genai ``BaseApiClient``):
# the target itself, a ``genai.Client``, an ADK ``Gemini`` model wrapper, or an
# ``LlmAgent``-shaped object whose ``.model`` already holds a resolved ``Gemini``.
# Not a frozen public API on any of these — see the module docstring.
_HOLDER_PATHS: tuple[tuple[str, ...], ...] = (
    (),
    ("_api_client",),
    ("api_client", "_api_client"),
    ("model", "api_client", "_api_client"),
)


def _walk(target: Any, path: tuple[str, ...]) -> Any:
    obj = target
    for attr in path:
        obj = getattr(obj, attr, None)
        if obj is None:
            return None
    return obj


def _find_holder(target: Any) -> Any:
    for path in _HOLDER_PATHS:
        holder = _walk(target, path)
        if holder is not None and hasattr(holder, "_httpx_client"):
            return holder
    return None


def _inject(target: Any, sync_client: Any, async_client: Any) -> list[str]:
    holder = _find_holder(target)
    if holder is None:
        return []
    injected: list[str] = []
    if hasattr(holder, "_httpx_client"):
        _set_attr(holder, "_httpx_client", sync_client)
        injected.append("_httpx_client")
    if hasattr(holder, "_async_httpx_client"):
        _set_attr(holder, "_async_httpx_client", async_client)
        injected.append("_async_httpx_client")
    return injected


def _underlying_transports(target: Any) -> tuple[Any, Any]:  # pragma: no cover - needs real SDK
    """Best-effort (sync, async) inner httpx transports, for record mode only."""
    holder = _find_holder(target)
    if holder is None:
        return None, None
    inner = getattr(getattr(holder, "_httpx_client", None), "_transport", None)
    inner_async = getattr(getattr(holder, "_async_httpx_client", None), "_transport", None)
    return inner, inner_async


# ── framework-independent event core (fully offline-testable) ──────────────────


class TraceforkAdkCore:
    """Accumulate a ``StepDAG`` from ADK ``BasePlugin`` callback events, framework-free.

    ``start``/``end`` mirror a before/after callback pair (agent, model, or tool
    boundary); nothing here imports ``google.adk``, so a test drives these
    directly with explicit step ids.
    """

    def __init__(self, dag: StepDAG | None = None) -> None:
        self.dag = dag if dag is not None else StepDAG()

    def start(
        self,
        kind: str,
        step_id: Any,
        parent_id: Any = None,
        *,
        name: str = "",
        model: str | None = None,
        inputs: Any = None,
        metadata: Any = None,
    ) -> Step:
        step = Step(
            step_id=str(step_id),
            parent_id=str(parent_id) if parent_id is not None else None,
            kind=kind,
            name=name,
            model=model,
            inputs=inputs,
            metadata=dict(metadata) if isinstance(metadata, Mapping) else {},
        )
        return self.dag.add(step)

    def end(self, step_id: Any, outputs: Any = None) -> Step | None:
        step = self.dag.by_id(str(step_id))
        if step is not None:
            step.outputs = outputs
        return step


# Event-type name -> Step kind, for the "before" half of each callback pair.
_START_KINDS = {
    "before_agent_callback": "agent",
    "before_model_callback": "llm",
    "before_tool_callback": "tool",
}
_END_EVENTS = frozenset({"after_agent_callback", "after_model_callback", "after_tool_callback"})


class _EventCore(TraceforkAdkCore):
    """Dispatch a single dict-shaped ``BasePlugin`` callback event onto the event core."""

    def dispatch(self, event: Mapping[str, Any]) -> Step:
        etype = str(event.get("event") or event.get("type") or "")
        step_id = event.get("id")
        if etype in _START_KINDS:
            if step_id is None:
                step_id = id(event)
            return self.start(
                _START_KINDS[etype],
                step_id,
                event.get("parent_id"),
                name=str(event.get("name", "") or ""),
                model=event.get("model"),
                inputs=event.get("inputs"),
                metadata=event.get("metadata"),
            )
        if etype in _END_EVENTS:
            if step_id is not None:
                step = self.end(step_id, event.get("outputs"))
                if step is not None:
                    return step
            return Step(step_id=str(step_id if step_id is not None else id(event)), kind=etype)
        # Unknown event: record a neutral step so nothing is silently dropped.
        resolved_id = step_id if step_id is not None else id(event)
        return self.dag.add(
            Step(step_id=str(resolved_id), kind=etype or "unknown", name=str(event.get("name", "")))
        )


def make_plugin(core: TraceforkAdkCore | None = None) -> Any:
    """Build a real ``BasePlugin`` that forwards ADK callback events to a ``TraceforkAdkCore``.

    Guarded: raises ``ImportError`` (install hint) if ``google.adk`` is missing.
    The returned plugin carries ``.core``/``.dag`` for inspection; pass it to
    ``Runner(..., plugins=[plugin])`` yourself. Every callback is
    OBSERVER-ONLY (always returns ``None``) — see the module docstring for the
    best-effort id-pairing this uses (ADK gives no unique per-call id).
    """
    require_adk()
    from google.adk.plugins.base_plugin import BasePlugin

    the_core = core if core is not None else TraceforkAdkCore()

    class TraceforkAdkPlugin(BasePlugin):  # pragma: no cover - needs google-adk
        """Thin ADK adapter: every callback forwards to the neutral core, observer-only."""

        def __init__(self) -> None:
            super().__init__(name="tracefork")
            self.core = the_core
            self.dag = the_core.dag
            self._open: dict[tuple[str, str, str], list[str]] = {}

        def _ids(self, ctx: Any) -> tuple[str, str]:
            invocation_id = str(_get(ctx, "invocation_id", default="") or "")
            agent_name = str(_get(ctx, "agent_name", default="") or "")
            return invocation_id, agent_name

        def _push(self, kind: str, invocation_id: str, agent_name: str, **start_kwargs: Any) -> str:
            key = (kind, invocation_id, agent_name)
            open_ids = self._open.setdefault(key, [])
            step_id = f"{invocation_id}:{agent_name}:{kind}:{len(open_ids)}"
            open_ids.append(step_id)
            self.core.start(kind, step_id, parent_id=invocation_id or None, **start_kwargs)
            return step_id

        def _pop(self, kind: str, invocation_id: str, agent_name: str) -> str | None:
            open_ids = self._open.get((kind, invocation_id, agent_name))
            return open_ids.pop() if open_ids else None

        async def before_agent_callback(self, *, agent: Any, callback_context: Any) -> None:
            invocation_id, agent_name = self._ids(callback_context)
            self._push("agent", invocation_id, agent_name, name=agent_name)
            return None

        async def after_agent_callback(self, *, agent: Any, callback_context: Any) -> None:
            invocation_id, agent_name = self._ids(callback_context)
            step_id = self._pop("agent", invocation_id, agent_name)
            if step_id is not None:
                self.core.end(step_id)
            return None

        async def before_model_callback(self, *, callback_context: Any, llm_request: Any) -> None:
            invocation_id, agent_name = self._ids(callback_context)
            model = _get(llm_request, "model", default=None)
            inputs = _get(llm_request, "contents", default=None)
            self._push("llm", invocation_id, agent_name, model=model, inputs=inputs)
            return None

        async def after_model_callback(self, *, callback_context: Any, llm_response: Any) -> None:
            invocation_id, agent_name = self._ids(callback_context)
            step_id = self._pop("llm", invocation_id, agent_name)
            if step_id is not None:
                self.core.end(step_id, outputs=_get(llm_response, "content", default=None))
            return None

        async def before_tool_callback(
            self, *, tool: Any, tool_args: Any, tool_context: Any
        ) -> None:
            invocation_id, agent_name = self._ids(tool_context)
            name = str(_get(tool, "name", default="") or "")
            self._push("tool", invocation_id, agent_name, name=name, inputs=tool_args)
            return None

        async def after_tool_callback(
            self, *, tool: Any, tool_args: Any, tool_context: Any, result: Any
        ) -> None:
            invocation_id, agent_name = self._ids(tool_context)
            step_id = self._pop("tool", invocation_id, agent_name)
            if step_id is not None:
                self.core.end(step_id, outputs=result)
            return None

    return TraceforkAdkPlugin()


# ── the adapter ────────────────────────────────────────────────────────────────


class AdkAdapter(BaseFrameworkAdapter):
    """Bind an ADK agent's google-genai client to tracefork's transport + annotate its run."""

    name = "adk"

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
        """Route ``target`` (a genai ``BaseApiClient``/``Client``/ADK ``Gemini``/
        ``LlmAgent``-shaped object) through tracefork.

        ``replay`` mode needs no inner transport and no live client; ``record``
        mode reuses the target's current underlying httpx transports as the
        inner so live calls still reach the network (that path needs the real
        SDKs and a real API key, and is not offline-testable). On replay, a
        ``ReplayNondet``-backed uuid patch (``patch_uuid=True``) makes
        framework-generated ids match the tape.
        """
        inner = inner_async = None
        if mode == "record":  # pragma: no cover - needs real SDK
            inner, inner_async = _underlying_transports(target)

        sync_client, async_client, sync_t, async_t = build_http_clients(
            tape, mode, inner=inner, async_inner=inner_async, matcher=matcher, redactor=redactor
        )
        injected = _inject(target, sync_client, async_client)

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
                "no google-genai BaseApiClient (_httpx_client/_async_httpx_client) found on "
                f"target ({type(target).__name__}) via the known candidate paths; "
                "nothing was injected"
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
        """Map one ADK ``BasePlugin`` callback event (as a dict) to a neutral ``Step``.

        ``event["event"]`` is the callback name (``"before_model_callback"``,
        ``"after_tool_callback"``, ...); the rest are its payload fields.
        """
        core = _EventCore(self.dag)
        return core.dispatch(event)


# Register the built-in adapter at import time (never via the entry-point path).
register_framework_adapter(AdkAdapter())


__all__ = [
    "ADK_IMPORT_HINT",
    "AdkAdapter",
    "TraceforkAdkCore",
    "adk_available",
    "make_plugin",
    "require_adk",
]
