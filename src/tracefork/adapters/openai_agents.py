"""Optional OpenAI Agents SDK adapter.

Two seams, both reusing tracefork's *existing* byte capture — never a second one:

* **``bind``** routes an Agents SDK model wrapper's underlying ``openai`` client
  through a ``TraceforkTransport``, the same ``client.copy(http_client=…)`` move
  ``recorder.py`` and ``adapters/langchain.py`` already use. The SDK's
  ``OpenAIChatCompletionsModel`` constructor takes the client as an
  ``openai_client`` keyword argument (see the SDK's ``models`` docs), but does
  not document the *attribute* name it stores it under, and that internal shape
  is not a frozen public API — so ``bind`` searches a short list of common
  candidate attribute names (``_client``, ``client``, ``openai_client``,
  ``_openai_client``) rather than hard-coding one, exactly the defensive style
  ``LangChainAdapter._inject_openai`` already uses for ``ChatOpenAI``.
  ``bind_default_client`` is the other, fully-documented injection path: the
  SDK's own top-level ``agents.set_default_openai_client(client)`` redirects
  *every* model call process-wide to a supplied ``AsyncOpenAI`` instance — no
  attribute guessing needed, but it requires the real ``agents`` package, so
  it is import-guarded and only exercised when installed.
* **``on_step`` / ``make_tracing_processor``** turn the SDK's tracing events
  (``TracingProcessor.on_trace_start`` / ``on_trace_end`` / ``on_span_start`` /
  ``on_span_end``) into neutral ``Step``s. Tracing is OBSERVER-ONLY here — it
  feeds the step-DAG, never a second capture path (the design invariant in
  ``adapters/base.py``). ``Span``/``Trace`` field names (``span_id``,
  ``parent_id``, ``trace_id``, ``span_data.type``/``.model``/``.output``) are
  read defensively (attribute-or-mapping, matching common names) for the same
  API-stability reason as the client attribute search above.

``openai-agents`` is OPTIONAL (the ``openai-agents`` extra). Nothing here
imports it at module load: the availability guard and the two guarded
factories (``bind_default_client`` / ``make_tracing_processor``) are the only
places a real import happens, so ``import tracefork`` and the whole offline
test suite run with it NOT installed. The framework-neutral core
(``TraceforkTracingCore``) and ``bind``'s attribute-search injection are fully
offline-testable with a synthetic client double (mirroring
``tests/test_adapters_langchain.py``'s ``_FakeChatOpenAI``) and synthetic
dict-shaped tracing events; the thin real ``TracingProcessor``/
``InterventionHandler``-style subclasses are only reachable — and only
validated — with the real SDK present (import-guarded, ``pytest.importorskip``
in the test suite). Honesty over coverage: this is a synthetic-interface
validation of the binding logic, not a live-framework integration test.
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

OPENAI_AGENTS_IMPORT_HINT = (
    "OpenAI Agents SDK support needs the optional 'openai-agents' extra: "
    "pip install 'tracefork[openai-agents]'"
)

# LLM-flavoured span_data.type values the OpenAI Agents SDK emits for model
# calls (as opposed to "function"/"agent"/"handoff"/... structural spans).
_LLM_SPAN_TYPES = frozenset({"generation", "response"})


# ── availability guard (mirrors adapters/langchain.py) ──────────────────────


def openai_agents_available() -> bool:
    """Whether the optional ``agents`` (OpenAI Agents SDK) package is importable."""
    try:
        import agents  # noqa: F401
    except ImportError:
        return False
    return True


def require_openai_agents() -> None:
    """Raise a helpful ``ImportError`` if the ``agents`` package is missing.

    Attempts the import itself (rather than delegating to
    ``openai_agents_available()``) and chains the real cause via ``from exc``,
    so an installed-but-broken ``agents`` package surfaces its own error
    instead of being masked as "not installed".
    """
    try:
        import agents  # noqa: F401
    except ImportError as exc:
        raise ImportError(OPENAI_AGENTS_IMPORT_HINT) from exc


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


# ── framework-independent tracing core (fully offline-testable) ────────────────


class TraceforkTracingCore:
    """Accumulate a ``StepDAG`` from OpenAI Agents SDK tracing events, framework-free.

    Method names mirror the SDK's ``TracingProcessor`` interface
    (``on_trace_start``/``on_trace_end``/``on_span_start``/``on_span_end``) so the
    thin real subclass (``make_tracing_processor``) forwards straight through;
    nothing here imports the SDK, so a test drives these directly with plain
    dicts or objects exposing the SDK's documented ``Span``/``Trace`` field names.
    """

    def __init__(self, dag: StepDAG | None = None) -> None:
        self.dag = dag if dag is not None else StepDAG()

    def on_trace_start(self, trace: Any) -> Step:
        trace_id = _get(trace, "trace_id", "id", default=None)
        step_id = str(trace_id) if trace_id is not None else str(id(trace))
        step = Step(step_id=step_id, kind="trace", name=str(_get(trace, "name", default="") or ""))
        return self.dag.add(step)

    def on_trace_end(self, trace: Any) -> Step | None:
        trace_id = _get(trace, "trace_id", "id", default=None)
        return self.dag.by_id(str(trace_id)) if trace_id is not None else None

    def on_span_start(self, span: Any) -> Step:
        span_id = _get(span, "span_id", "id", default=None)
        step_id = str(span_id) if span_id is not None else str(id(span))
        parent_id = _get(span, "parent_id", default=None)
        if parent_id is None:
            parent_id = _get(span, "trace_id", default=None)
        span_data = _get(span, "span_data", default=None)
        raw_type = str(_get(span_data, "type", default="") or "")
        kind = "llm" if raw_type in _LLM_SPAN_TYPES else (raw_type or "span")
        name = str(_get(span_data, "name", default="") or _get(span, "name", default="") or "")
        model = _get(span_data, "model", default=None)
        step = Step(
            step_id=step_id,
            parent_id=str(parent_id) if parent_id is not None else None,
            kind=kind,
            name=name,
            model=str(model) if model else None,
            inputs=_get(span_data, "input", default=None),
        )
        return self.dag.add(step)

    def on_span_end(self, span: Any) -> Step | None:
        span_id = _get(span, "span_id", "id", default=None)
        step = self.dag.by_id(str(span_id)) if span_id is not None else None
        if step is not None:
            span_data = _get(span, "span_data", default=None)
            output = _get(span_data, "output", default=None)
            if output is None:
                output = _get(span_data, "response", default=None)
            step.outputs = output
        return step


class _EventCore(TraceforkTracingCore):
    """Dispatch a single dict-shaped tracing event onto the tracing core."""

    def dispatch(self, event: Mapping[str, Any]) -> Step:
        etype = str(event.get("event") or event.get("type") or "")
        if etype == "on_trace_start":
            return self.on_trace_start(event)
        if etype == "on_trace_end":
            trace_id = _get(event, "trace_id", "id", default="")
            return self.on_trace_end(event) or Step(step_id=str(trace_id))
        if etype == "on_span_start":
            return self.on_span_start(event)
        if etype == "on_span_end":
            span_id = _get(event, "span_id", "id", default="")
            return self.on_span_end(event) or Step(step_id=str(span_id))
        # Unknown event: record a neutral step so nothing is silently dropped.
        step_id = _get(event, "id", default=str(id(event)))
        return self.dag.add(Step(step_id=str(step_id), kind=etype or "unknown"))


def make_tracing_processor(
    core: TraceforkTracingCore | None = None, *, install: bool = False
) -> Any:
    """Build a real ``TracingProcessor`` that forwards to a ``TraceforkTracingCore``.

    Guarded: raises ``ImportError`` (install hint) if ``agents`` is missing. The
    returned processor carries ``.core``/``.dag`` for inspection; pass it to
    ``agents.set_trace_processors([processor])`` yourself, or pass
    ``install=True`` to have this call it for you.
    """
    require_openai_agents()
    import agents
    from agents.tracing.processor_interface import TracingProcessor

    the_core = core if core is not None else TraceforkTracingCore()

    class TraceforkTracingProcessor(TracingProcessor):  # pragma: no cover - needs openai-agents
        """Thin Agents SDK adapter: every callback forwards to the neutral core."""

        def __init__(self) -> None:
            self.core = the_core
            self.dag = the_core.dag

        def on_trace_start(self, trace: Any) -> None:
            self.core.on_trace_start(trace)

        def on_trace_end(self, trace: Any) -> None:
            self.core.on_trace_end(trace)

        def on_span_start(self, span: Any) -> None:
            self.core.on_span_start(span)

        def on_span_end(self, span: Any) -> None:
            self.core.on_span_end(span)

        def shutdown(self) -> None:
            pass

        def force_flush(self) -> None:
            pass

    processor = TraceforkTracingProcessor()
    if install:
        agents.set_trace_processors([processor])
    return processor


def bind_default_client(
    tape: Tape,
    mode: str = "replay",
    *,
    inner: Any = None,
    async_inner: Any = None,
    matcher: Any = None,
    redactor: Any = None,
) -> BindResult:
    """Build tracefork-backed clients and install them SDK-wide.

    Guarded: raises ``ImportError`` (install hint) if ``agents`` is missing.
    Calls the SDK's own documented ``agents.set_default_openai_client(client)``,
    which redirects every model call process-wide — the cleanest injection path
    when nothing has constructed an ``OpenAIChatCompletionsModel`` yet. For an
    already-constructed model instance, use ``OpenAIAgentsAdapter.bind`` instead
    (attribute-search injection, fully offline-testable).
    """
    require_openai_agents()
    import agents

    sync_client, async_client, sync_t, async_t = build_http_clients(
        tape, mode, inner=inner, async_inner=async_inner, matcher=matcher, redactor=redactor
    )
    agents.set_default_openai_client(async_client)
    return BindResult(
        mode=mode,
        http_client=sync_client,
        http_async_client=async_client,
        transport=sync_t,
        async_transport=async_t,
        injected_fields=("default_openai_client",),
        notes="",
    )


# ── the adapter ────────────────────────────────────────────────────────────────

# Candidate attribute names an Agents SDK model wrapper (e.g.
# ``OpenAIChatCompletionsModel``) might store its underlying openai client
# under. Not a frozen public API — see the module docstring.
_ASYNC_CLIENT_ATTRS = ("_client", "client", "openai_client", "_openai_client")
_SYNC_CLIENT_ATTRS = ("_sync_client", "sync_client")


def _inject(target: Any, sync_client: Any, async_client: Any) -> list[str]:
    injected: list[str] = []
    for name in _ASYNC_CLIENT_ATTRS:
        current = getattr(target, name, None)
        if current is not None and hasattr(current, "copy"):
            _set_attr(target, name, current.copy(http_client=async_client))
            injected.append(name)
            break
    for name in _SYNC_CLIENT_ATTRS:
        current = getattr(target, name, None)
        if current is not None and hasattr(current, "copy"):
            _set_attr(target, name, current.copy(http_client=sync_client))
            injected.append(name)
            break
    return injected


def _underlying_transports(target: Any) -> tuple[Any, Any]:  # pragma: no cover - needs real SDK
    """Best-effort (sync, async) inner httpx transports, for record mode only."""
    inner = async_inner = None
    for name in _SYNC_CLIENT_ATTRS:
        current = getattr(target, name, None)
        if current is not None and hasattr(current, "copy"):
            inner = getattr(getattr(current, "_client", None), "_transport", None)
            break
    for name in _ASYNC_CLIENT_ATTRS:
        current = getattr(target, name, None)
        if current is not None and hasattr(current, "copy"):
            async_inner = getattr(getattr(current, "_client", None), "_transport", None)
            break
    return inner, async_inner


class OpenAIAgentsAdapter(BaseFrameworkAdapter):
    """Bind an OpenAI Agents SDK model wrapper to tracefork's transport + annotate its run."""

    name = "openai_agents"

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
        """Route ``target`` (a model wrapper holding an ``openai`` client) through tracefork.

        ``replay`` mode needs no inner transport and no live client; ``record``
        mode reuses the target's current underlying httpx transport as the inner
        so live calls still reach the network (that path needs the real SDK and
        is not offline-testable). On replay, a ``ReplayNondet``-backed uuid patch
        (``patch_uuid=True``) makes framework-generated ids match the tape.
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
                "no known OpenAI-client attribute found on target "
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
        """Map one tracing event (as a dict) to a neutral ``Step``.

        ``event["event"]`` is the callback name (``"on_trace_start"``,
        ``"on_span_end"``, ...); the rest are the ``Trace``/``Span`` fields, read
        defensively (see ``TraceforkTracingCore``).
        """
        core = _EventCore(self.dag)
        return core.dispatch(event)


# Register the built-in adapter at import time (never via the entry-point path).
register_framework_adapter(OpenAIAgentsAdapter())


__all__ = [
    "OPENAI_AGENTS_IMPORT_HINT",
    "OpenAIAgentsAdapter",
    "TraceforkTracingCore",
    "bind_default_client",
    "make_tracing_processor",
    "openai_agents_available",
    "require_openai_agents",
]
