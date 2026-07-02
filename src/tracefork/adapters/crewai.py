"""Optional CrewAI adapter.

Two seams, both reusing tracefork's *existing* byte capture — never a second one:

* **``bind``** targets CrewAI's actual httpx chokepoint. CrewAI never touches
  httpx directly — every model call goes through LiteLLM
  (``litellm.completion``/``litellm.acompletion``), which exposes a documented
  global customization point for exactly this: the module-level
  ``litellm.client_session`` (sync, ``httpx.Client``) and
  ``litellm.aclient_session`` (async, ``httpx.AsyncClient``) attributes (see
  LiteLLM's "Custom HTTP Handler" docs). ``bind`` sets both to httpx clients
  wrapping the *existing* ``TraceforkTransport``. ``target`` is the ``litellm``
  module itself (or a duck-typed stand-in exposing ``completion``/
  ``acompletion`` — used only to confirm the target looks like litellm before
  mutating it — plus assignable ``client_session``/``aclient_session``
  attributes); ``bind`` never imports ``litellm`` itself.
* **``on_step`` / ``make_event_listener``** turn ``crewai_event_bus`` events
  (crew/agent/task/tool/LLM-call boundaries) into neutral ``Step``s. The event
  bus is OBSERVER-ONLY here — it feeds the step-DAG, never a second capture
  path (the design invariant in ``adapters/base.py``).

Honesty note: CrewAI's event payloads (``TaskStartedEvent``,
``AgentExecutionStartedEvent``, ...) do not carry an explicit
run_id/parent_run_id pair the way LangChain's callbacks do, so
``make_event_listener``'s real-event id/parent extraction is best-effort (falls
back to python ``id()`` identity keyed off the event's ``task``/``agent``
sub-object when present, and leaves steps unparented — a flat structure, not a
verified tree). The framework-neutral core (``TraceforkCrewEventCore``) and its
``on_step`` dict dispatch are fully offline-tested against explicit
``id``/``parent_id`` events — the DAG-building logic itself is proven; only the
real-event-to-dict mapping inside the guarded factory is unverified without the
real package installed (import-guarded, ``pytest.importorskip`` in the test
suite). Honesty over coverage: this is a synthetic-interface validation of the
binding logic, not a live-framework integration test.

``crewai`` is OPTIONAL (the ``crewai`` extra, which pulls in ``litellm``
transitively). Nothing here imports it at module load: the availability guard
and ``make_event_listener`` are the only places a real import happens, so
``import tracefork`` and the whole offline test suite run with it NOT
installed.
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

CREWAI_IMPORT_HINT = (
    "CrewAI support needs the optional 'crewai' extra: pip install 'tracefork[crewai]'"
)

# The two litellm top-level callables bind() checks for before mutating a
# target, to confirm it actually looks like the litellm module.
_LITELLM_SIGNATURE_ATTRS = ("completion", "acompletion")


# ── availability guard (mirrors adapters/langchain.py) ──────────────────────


def crewai_available() -> bool:
    """Whether the optional ``crewai`` package is importable."""
    try:
        import crewai  # noqa: F401
    except ImportError:
        return False
    return True


def require_crewai() -> None:
    """Raise a helpful ``ImportError`` if ``crewai`` is missing."""
    if not crewai_available():
        raise ImportError(CREWAI_IMPORT_HINT)


# ── framework-independent event core (fully offline-testable) ──────────────────


class TraceforkCrewEventCore:
    """Accumulate a ``StepDAG`` from CrewAI event-bus events, framework-free.

    ``start``/``end`` mirror a start/end event pair (crew kickoff, agent
    execution, task, tool usage, LLM call); nothing here imports CrewAI, so a
    test drives these directly with explicit step ids.
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


# Event-type name -> (Step kind, is this a start or end event).
_START_KINDS = {
    "crew_kickoff_started": "crew",
    "agent_execution_started": "agent",
    "task_started": "task",
    "tool_usage_started": "tool",
    "llm_call_started": "llm",
}
_END_EVENTS = frozenset(
    {
        "crew_kickoff_completed",
        "agent_execution_completed",
        "task_completed",
        "tool_usage_finished",
        "llm_call_completed",
    }
)


class _EventCore(TraceforkCrewEventCore):
    """Dispatch a single dict-shaped event-bus event onto the event core."""

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


def make_event_listener(core: TraceforkCrewEventCore | None = None) -> Any:
    """Build a real ``BaseEventListener`` that forwards ``crewai_event_bus`` events.

    Guarded: raises ``ImportError`` (install hint) if ``crewai`` is missing. The
    returned listener carries ``.core``/``.dag`` for inspection. Registers a
    representative set of crew/agent/task/tool/LLM-call boundary events
    (``crewai.events``); id/parent extraction is best-effort — see the module
    docstring.
    """
    require_crewai()
    from crewai.events import (
        AgentExecutionCompletedEvent,
        AgentExecutionStartedEvent,
        BaseEventListener,
        CrewKickoffCompletedEvent,
        CrewKickoffStartedEvent,
        LLMCallCompletedEvent,
        LLMCallStartedEvent,
        TaskCompletedEvent,
        TaskStartedEvent,
        ToolUsageFinishedEvent,
        ToolUsageStartedEvent,
    )

    the_core = core if core is not None else TraceforkCrewEventCore()

    def _id(event: Any) -> str:
        for attr in ("task", "agent"):
            obj = getattr(event, attr, None)
            ident = getattr(obj, "id", None)
            if ident is not None:
                return f"{attr}:{ident}"
        return str(id(event))

    class TraceforkEventListener(BaseEventListener):  # pragma: no cover - needs crewai
        """Thin CrewAI adapter: every handler forwards to the neutral core."""

        def __init__(self) -> None:
            super().__init__()
            self.core = the_core
            self.dag = the_core.dag

        def setup_listeners(self, event_bus: Any) -> None:
            @event_bus.on(CrewKickoffStartedEvent)
            def _crew_started(source: Any, event: Any) -> None:
                self.core.start("crew", _id(event), name=str(getattr(event, "crew_name", "") or ""))

            @event_bus.on(CrewKickoffCompletedEvent)
            def _crew_completed(source: Any, event: Any) -> None:
                self.core.end(_id(event), getattr(event, "output", None))

            @event_bus.on(AgentExecutionStartedEvent)
            def _agent_started(source: Any, event: Any) -> None:
                agent = getattr(event, "agent", None)
                self.core.start("agent", _id(event), name=str(getattr(agent, "role", "") or ""))

            @event_bus.on(AgentExecutionCompletedEvent)
            def _agent_completed(source: Any, event: Any) -> None:
                self.core.end(_id(event), getattr(event, "output", None))

            @event_bus.on(TaskStartedEvent)
            def _task_started(source: Any, event: Any) -> None:
                task = getattr(event, "task", None)
                description = str(getattr(task, "description", "") or "")
                self.core.start("task", _id(event), name=description)

            @event_bus.on(TaskCompletedEvent)
            def _task_completed(source: Any, event: Any) -> None:
                self.core.end(_id(event), getattr(event, "output", None))

            @event_bus.on(ToolUsageStartedEvent)
            def _tool_started(source: Any, event: Any) -> None:
                self.core.start("tool", _id(event), name=str(getattr(event, "tool_name", "") or ""))

            @event_bus.on(ToolUsageFinishedEvent)
            def _tool_finished(source: Any, event: Any) -> None:
                self.core.end(_id(event), getattr(event, "output", None))

            @event_bus.on(LLMCallStartedEvent)
            def _llm_started(source: Any, event: Any) -> None:
                self.core.start("llm", _id(event), model=getattr(event, "model", None))

            @event_bus.on(LLMCallCompletedEvent)
            def _llm_completed(source: Any, event: Any) -> None:
                self.core.end(_id(event), getattr(event, "response", None))

    return TraceforkEventListener()


# ── the adapter ────────────────────────────────────────────────────────────────


class CrewAIAdapter(BaseFrameworkAdapter):
    """Bind CrewAI's LiteLLM httpx chokepoint to tracefork + annotate its run."""

    name = "crewai"

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
        """Route ``target`` (the ``litellm`` module, or a duck-typed stand-in) through tracefork.

        ``replay`` mode needs no inner transport; ``record`` mode reuses
        ``target``'s current ``client_session``/``aclient_session`` (if any) as
        the inner so live calls still reach the network — that path needs a
        real litellm install + live key and is not offline-testable.
        """
        looks_like_litellm = all(hasattr(target, attr) for attr in _LITELLM_SIGNATURE_ATTRS)

        inner = inner_async = None
        if mode == "record":  # pragma: no cover - needs real litellm + live key
            inner = getattr(getattr(target, "client_session", None), "_transport", None)
            inner_async = getattr(getattr(target, "aclient_session", None), "_transport", None)

        sync_client, async_client, sync_t, async_t = build_http_clients(
            tape, mode, inner=inner, async_inner=inner_async, matcher=matcher, redactor=redactor
        )

        injected: list[str] = []
        if looks_like_litellm:
            target.client_session = sync_client
            target.aclient_session = async_client
            injected = ["client_session", "aclient_session"]

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
                "target does not look like the litellm module (missing "
                "completion/acompletion); nothing was injected"
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
        """Map one CrewAI event-bus event (as a dict) to a neutral ``Step``.

        ``event["event"]`` is the event type name (``"task_started"``,
        ``"llm_call_completed"``, ...); the rest are its payload fields.
        """
        core = _EventCore(self.dag)
        return core.dispatch(event)


# Register the built-in adapter at import time (never via the entry-point path).
register_framework_adapter(CrewAIAdapter())


__all__ = [
    "CREWAI_IMPORT_HINT",
    "CrewAIAdapter",
    "TraceforkCrewEventCore",
    "crewai_available",
    "make_event_listener",
    "require_crewai",
]
