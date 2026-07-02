"""Optional AutoGen (``autogen-core`` / ``autogen-ext``) adapter.

Two seams, both reusing tracefork's *existing* byte capture — never a second one:

* **``bind``** routes an AutoGen model client's underlying ``openai`` client
  through a ``TraceforkTransport``, the same ``client.copy(http_client=…)`` move
  ``recorder.py`` and ``adapters/langchain.py`` already use. AutoGen's
  ``autogen_ext.models.openai.OpenAIChatCompletionClient`` wraps an
  ``AsyncOpenAI``/``AsyncAzureOpenAI`` instance (its base class's constructor
  parameter is literally named ``_client``), but the *stored* attribute name is
  not documented as a frozen public API — so ``bind`` searches a short list of
  common candidate attribute names (``_client``, ``client``, ``openai_client``),
  exactly the defensive style ``LangChainAdapter._inject_openai`` already uses.
* **``on_step`` / ``make_intervention_handler``** turn AutoGen's
  ``InterventionHandler`` message-level callbacks (``on_send``/``on_publish``/
  ``on_response`` — called by ``SingleThreadedAgentRuntime`` on every
  ``send_message``/``publish_message``/response) into neutral ``Step``s. This is
  the "message-level fork/blame seam" the design calls for: it is OBSERVER-ONLY
  here — the handler always returns the message unmodified (never
  ``DropMessage``), so it feeds the step-DAG rather than becoming a second
  capture path (the design invariant in ``adapters/base.py``); the actual
  bit-exact byte record stays at the httpx transport via ``bind``.

Honesty note: AutoGen's message/``AgentId`` objects don't carry an explicit
run_id/parent_run_id pair the way LangChain's callbacks do, so
``make_intervention_handler``'s real-event id derivation is best-effort (python
``id()`` identity on the message object, name from ``AgentId.type``/``.key``
when present). The framework-neutral core (``TraceforkInterventionCore``) and
its ``on_step`` dict dispatch are fully offline-tested against explicit
``id``/``parent_id`` events — the DAG-building logic itself is proven; only the
real-message-to-dict mapping inside the guarded factory is unverified without
the real package installed (import-guarded, ``pytest.importorskip`` in the test
suite). Honesty over coverage: this is a synthetic-interface validation of the
binding logic, not a live-framework integration test.

``autogen-core``/``autogen-ext`` are OPTIONAL (the ``autogen`` extra). Nothing
here imports them at module load: the availability guard and
``make_intervention_handler`` are the only places a real import happens, so
``import tracefork`` and the whole offline test suite run with neither
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

AUTOGEN_IMPORT_HINT = (
    "AutoGen support needs the optional 'autogen' extra: pip install 'tracefork[autogen]'"
)


# ── availability guard (mirrors adapters/langchain.py) ──────────────────────


def autogen_available() -> bool:
    """Whether the optional ``autogen_core`` package is importable."""
    try:
        import autogen_core  # noqa: F401
    except ImportError:
        return False
    return True


def require_autogen() -> None:
    """Raise a helpful ``ImportError`` if ``autogen_core`` is missing."""
    if not autogen_available():
        raise ImportError(AUTOGEN_IMPORT_HINT)


def _set_attr(obj: Any, name: str, value: Any) -> None:
    """Assign ``obj.name = value``, falling back through pydantic's guardrails."""
    try:
        setattr(obj, name, value)
    except Exception:  # pragma: no cover - pydantic-version dependent
        object.__setattr__(obj, name, value)


# ── framework-independent intervention core (fully offline-testable) ───────────


class TraceforkInterventionCore:
    """Accumulate a ``StepDAG`` from AutoGen intervention events, framework-free.

    ``record`` mirrors one ``on_send``/``on_publish``/``on_response`` call;
    nothing here imports ``autogen_core``, so a test drives this directly with
    explicit step ids.
    """

    def __init__(self, dag: StepDAG | None = None) -> None:
        self.dag = dag if dag is not None else StepDAG()

    def record(
        self,
        kind: str,
        step_id: Any,
        parent_id: Any = None,
        *,
        name: str = "",
        inputs: Any = None,
        outputs: Any = None,
    ) -> Step:
        step = Step(
            step_id=str(step_id),
            parent_id=str(parent_id) if parent_id is not None else None,
            kind=kind,
            name=name,
            inputs=inputs,
            outputs=outputs,
        )
        return self.dag.add(step)


_KIND_BY_EVENT = {"on_send": "send", "on_publish": "publish", "on_response": "response"}


class _EventCore(TraceforkInterventionCore):
    """Dispatch a single dict-shaped intervention event onto the core."""

    def dispatch(self, event: Mapping[str, Any]) -> Step:
        etype = str(event.get("event") or event.get("type") or "")
        step_id = event.get("id")
        if step_id is None:
            step_id = id(event)
        kind = _KIND_BY_EVENT.get(etype, etype or "unknown")
        return self.record(
            kind,
            step_id,
            event.get("parent_id"),
            name=str(event.get("name", "") or ""),
            inputs=event.get("message") if etype in ("on_send", "on_publish") else None,
            outputs=event.get("message") if etype == "on_response" else None,
        )


def make_intervention_handler(core: TraceforkInterventionCore | None = None) -> Any:
    """Build a real ``InterventionHandler`` that forwards to a ``TraceforkInterventionCore``.

    Guarded: raises ``ImportError`` (install hint) if ``autogen_core`` is
    missing. Pass-through only: every method returns ``message`` unchanged
    (never ``DropMessage``) — see the module docstring. The returned handler
    carries ``.core``/``.dag`` for inspection and is passed via
    ``SingleThreadedAgentRuntime(intervention_handlers=[handler])``.
    """
    require_autogen()
    from autogen_core import InterventionHandler

    the_core = core if core is not None else TraceforkInterventionCore()

    def _agent_id(value: Any) -> str:
        agent_type = getattr(value, "type", None)
        key = getattr(value, "key", None)
        return f"{agent_type}/{key}" if agent_type is not None else str(value)

    class TraceforkInterventionHandler(InterventionHandler):  # pragma: no cover - needs autogen
        """Thin AutoGen adapter: every callback forwards to the neutral core, pass-through."""

        def __init__(self) -> None:
            self.core = the_core
            self.dag = the_core.dag

        async def on_send(self, message: Any, *, message_context: Any, recipient: Any) -> Any:
            step_id = f"send:{id(message)}"
            self.core.record("send", step_id, name=_agent_id(recipient), inputs=message)
            return message

        async def on_publish(self, message: Any, *, message_context: Any) -> Any:
            self.core.record("publish", f"publish:{id(message)}", inputs=message)
            return message

        async def on_response(self, message: Any, *, sender: Any, recipient: Any = None) -> Any:
            step_id = f"response:{id(message)}"
            self.core.record("response", step_id, name=_agent_id(sender), outputs=message)
            return message

    return TraceforkInterventionHandler()


# ── the adapter ────────────────────────────────────────────────────────────────

# Candidate attribute names an AutoGen model client (e.g.
# ``OpenAIChatCompletionClient``) might store its underlying openai client
# under. Not a frozen public API — see the module docstring.
_ASYNC_CLIENT_ATTRS = ("_client", "client", "openai_client")
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


class AutoGenAdapter(BaseFrameworkAdapter):
    """Bind an AutoGen model client to tracefork's transport + annotate its run."""

    name = "autogen"

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
        """Route ``target`` (a model client holding an ``openai`` client) through tracefork.

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
        """Map one intervention event (as a dict) to a neutral ``Step``.

        ``event["event"]`` is the callback name (``"on_send"``, ``"on_publish"``,
        ``"on_response"``); the rest are its payload fields.
        """
        core = _EventCore(self.dag)
        return core.dispatch(event)


# Register the built-in adapter at import time (never via the entry-point path).
register_framework_adapter(AutoGenAdapter())


__all__ = [
    "AUTOGEN_IMPORT_HINT",
    "AutoGenAdapter",
    "TraceforkInterventionCore",
    "autogen_available",
    "make_intervention_handler",
    "require_autogen",
]
