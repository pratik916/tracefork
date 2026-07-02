"""Optional LangChain + LangGraph adapter.

Two seams, both reusing tracefork's *existing* byte capture — never a second one:

* **``bind``** routes a LangChain chat model's underlying LLM client through a
  ``TraceforkTransport``. ``ChatOpenAI`` exposes the raw ``openai`` clients on
  ``root_client`` / ``root_async_client``, swapped via the openai SDK's
  ``.copy(http_client=...)`` (the ``recorder.py`` move). ``ChatAnthropic`` has NO
  ``http_client`` field and builds ``_client`` / ``_async_client`` lazily as
  cached properties, so on replay ``bind`` seeds fresh ``anthropic`` clients
  (wrapping the tracefork transport) into the instance via ``object.__setattr__``
  before first access. On replay the client serves recorded bytes for $0, and
  ``bind`` optionally installs a ``ReplayNondet``-backed uuid patch so
  framework-generated ids match the tape.
* **``on_step``** turns a LangChain ``BaseCallbackHandler`` event into a neutral
  ``Step``. Callbacks are OBSERVER-ONLY (a structure/annotation layer feeding the
  step-DAG) — they never capture bytes.

The marquee is **tape-backed LangGraph time-travel**: pair a bound (replay) chat
model with ``make_tape_backed_checkpointer(tape)`` and LangGraph's own
checkpoint time-travel resumes graph state while the model replays its I/O from
the tape — bit-exact and $0.

``langchain-*`` / ``langgraph`` are OPTIONAL (the ``frameworks`` extra). Nothing
here imports them at module load: the availability guards and the two guarded
factories (``make_callback_handler`` / ``make_tape_backed_checkpointer``) are the
only places a real framework import happens, so ``import tracefork`` and the
whole offline test suite run with none of them installed. The framework-neutral
cores (``TraceforkCallbackCore`` / ``TapeBackedCheckpointStore``) are exercised
offline with synthetic events; the thin BaseCallbackHandler / BaseCheckpointSaver
subclasses that wrap them are only reachable with the real framework present.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
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

FRAMEWORKS_IMPORT_HINT = (
    "LangChain/LangGraph support needs the optional 'frameworks' extra: "
    "pip install 'tracefork[frameworks]'"
)


# ── availability guards (mirror mcp_client.py / observability.py) ───────────────


def langchain_available() -> bool:
    """Whether ``langchain-core`` is importable."""
    try:
        import langchain_core  # noqa: F401
    except ImportError:
        return False
    return True


def require_langchain() -> None:
    """Raise a helpful ``ImportError`` if ``langchain-core`` is missing.

    Attempts the import itself (rather than delegating to
    ``langchain_available()``) and chains the real cause via ``from exc``, so an
    installed-but-broken ``langchain-core`` surfaces its own error instead of
    being masked as "not installed".
    """
    try:
        import langchain_core  # noqa: F401
    except ImportError as exc:
        raise ImportError(FRAMEWORKS_IMPORT_HINT) from exc


def langgraph_available() -> bool:
    """Whether ``langgraph`` is importable."""
    try:
        import langgraph  # noqa: F401
    except ImportError:
        return False
    return True


def require_langgraph() -> None:
    """Raise a helpful ``ImportError`` if ``langgraph`` is missing.

    Attempts the import itself (rather than delegating to
    ``langgraph_available()``) and chains the real cause via ``from exc``, so an
    installed-but-broken ``langgraph`` surfaces its own error instead of being
    masked as "not installed".
    """
    try:
        import langgraph  # noqa: F401
    except ImportError as exc:
        raise ImportError(FRAMEWORKS_IMPORT_HINT) from exc


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


def _class_name(serialized: Any) -> str:
    """Best-effort class name from a LangChain ``serialized`` payload."""
    name = _get(serialized, "name", default=None)
    if name:
        return str(name)
    ident = _get(serialized, "id", default=None)
    if isinstance(ident, (list, tuple)) and ident:
        return str(ident[-1])
    return ""


def _model_name(serialized: Any, metadata: Any) -> str | None:
    """Model id from callback metadata (``ls_model_name``) or serialized kwargs."""
    meta_model = _get(metadata, "ls_model_name", "model", "model_name", default=None)
    if meta_model:
        return str(meta_model)
    kwargs = _get(serialized, "kwargs", default=None)
    kw_model = _get(kwargs, "model", "model_name", default=None)
    return str(kw_model) if kw_model else None


def _llm_result_text_and_model(response: Any) -> tuple[str, str | None]:
    """Pull first-generation text + model id from a LangChain ``LLMResult``-shape."""
    generations = _get(response, "generations", default=None) or []
    text = ""
    for gen_list in generations:
        for gen in gen_list or []:
            text = _get(gen, "text", default="") or ""
            if not text:
                message = _get(gen, "message", default=None)
                text = _get(message, "content", default="") or ""
            if text:
                break
        if text:
            break
    llm_output = _get(response, "llm_output", default=None) or {}
    model = _get(llm_output, "model_name", "model", default=None)
    return str(text), (str(model) if model else None)


# ── framework-independent callback core (fully offline-testable) ────────────────


class TraceforkCallbackCore:
    """Accumulate a ``StepDAG`` from LangChain callback events, framework-free.

    The method names match ``langchain_core.callbacks.BaseCallbackHandler`` so the
    thin real subclass (``make_callback_handler``) forwards straight through; but
    nothing here imports LangChain, so a test drives these directly with synthetic
    ``run_id`` / ``parent_run_id`` values and dict payloads.
    """

    def __init__(self, dag: StepDAG | None = None) -> None:
        self.dag = dag if dag is not None else StepDAG()

    def _start(
        self,
        kind: str,
        run_id: Any,
        parent_run_id: Any,
        name: str,
        *,
        model: str | None = None,
        inputs: Any = None,
        metadata: Any = None,
    ) -> Step:
        step = Step(
            step_id=str(run_id),
            parent_id=str(parent_run_id) if parent_run_id is not None else None,
            kind=kind,
            name=name,
            model=model,
            inputs=inputs,
            metadata=dict(metadata) if isinstance(metadata, Mapping) else {},
        )
        return self.dag.add(step)

    def _end(self, run_id: Any, outputs: Any) -> Step | None:
        step = self.dag.by_id(str(run_id))
        if step is not None:
            step.outputs = outputs
        return step

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, **kwargs) -> Step:
        return self._start(
            "chain",
            run_id,
            parent_run_id,
            _class_name(serialized),
            inputs=inputs,
            metadata=kwargs.get("metadata"),
        )

    def on_chain_end(self, outputs, *, run_id, **kwargs) -> Step | None:
        return self._end(run_id, outputs)

    def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id=None, **kwargs) -> Step:
        return self._start(
            "llm",
            run_id,
            parent_run_id,
            _class_name(serialized),
            model=_model_name(serialized, kwargs.get("metadata")),
            inputs=prompts,
            metadata=kwargs.get("metadata"),
        )

    def on_chat_model_start(
        self, serialized, messages, *, run_id, parent_run_id=None, **kwargs
    ) -> Step:
        return self._start(
            "chat_model",
            run_id,
            parent_run_id,
            _class_name(serialized),
            model=_model_name(serialized, kwargs.get("metadata")),
            inputs=messages,
            metadata=kwargs.get("metadata"),
        )

    def on_llm_end(self, response, *, run_id, **kwargs) -> Step | None:
        step = self.dag.by_id(str(run_id))
        if step is not None:
            text, model = _llm_result_text_and_model(response)
            step.outputs = text
            if model and not step.model:
                step.model = model
        return step

    def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, **kwargs) -> Step:
        return self._start(
            "tool",
            run_id,
            parent_run_id,
            _class_name(serialized),
            inputs=input_str,
            metadata=kwargs.get("metadata"),
        )

    def on_tool_end(self, output, *, run_id, **kwargs) -> Step | None:
        return self._end(run_id, output)


def make_callback_handler(core: TraceforkCallbackCore | None = None) -> Any:
    """Build a real ``BaseCallbackHandler`` that forwards to a ``TraceforkCallbackCore``.

    Guarded: raises ``ImportError`` (install hint) if ``langchain-core`` is
    missing. The returned handler carries ``.core`` and ``.dag`` for inspection
    and can be passed via ``config={"callbacks": [handler]}`` to any LangChain /
    LangGraph invocation.
    """
    require_langchain()
    from langchain_core.callbacks.base import BaseCallbackHandler

    the_core = core if core is not None else TraceforkCallbackCore()

    class TraceforkCallbackHandler(BaseCallbackHandler):  # pragma: no cover - needs langchain
        """Thin LangChain adapter: every ``on_*`` forwards to the neutral core."""

        raise_error = False

        def __init__(self) -> None:
            super().__init__()
            self.core = the_core
            self.dag = the_core.dag

        def on_chain_start(self, *args, **kwargs):
            return self.core.on_chain_start(*args, **kwargs)

        def on_chain_end(self, *args, **kwargs):
            return self.core.on_chain_end(*args, **kwargs)

        def on_llm_start(self, *args, **kwargs):
            return self.core.on_llm_start(*args, **kwargs)

        def on_chat_model_start(self, *args, **kwargs):
            return self.core.on_chat_model_start(*args, **kwargs)

        def on_llm_end(self, *args, **kwargs):
            return self.core.on_llm_end(*args, **kwargs)

        def on_tool_start(self, *args, **kwargs):
            return self.core.on_tool_start(*args, **kwargs)

        def on_tool_end(self, *args, **kwargs):
            return self.core.on_tool_end(*args, **kwargs)

    return TraceforkCallbackHandler()


# ── tape-backed LangGraph checkpointer ─────────────────────────────────────────


@dataclass
class CheckpointRecord:
    """One stored graph checkpoint, linked (optionally) to a tape step."""

    thread_id: str
    checkpoint_id: str
    data: Any
    parent_id: str | None = None
    tape_index: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


# Alias so the store's ``list`` method (named to mirror LangGraph's checkpointer)
# doesn't shadow builtin ``list`` inside its own return annotations.
CheckpointRecordList = list[CheckpointRecord]


class TapeBackedCheckpointStore:
    """Framework-independent checkpoint store — the reusable, fully-tested core.

    Keyed by ``(thread_id, checkpoint_id)`` and carried alongside the tape so a
    LangGraph run's *state* time-travel lines up with the tape's *byte* record:
    resuming from a stored checkpoint re-invokes a tape-bound chat model, which
    replays its I/O for $0. Newest-first ``list`` mirrors LangGraph's
    ``get_state_history`` ordering (the time-travel surface).
    """

    def __init__(self, tape: Tape) -> None:
        self.tape = tape
        self._by_thread: dict[str, list[CheckpointRecord]] = {}

    def put(
        self,
        thread_id: str,
        checkpoint_id: str,
        data: Any,
        *,
        parent_id: str | None = None,
        tape_index: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> CheckpointRecord:
        record = CheckpointRecord(
            thread_id=thread_id,
            checkpoint_id=checkpoint_id,
            data=data,
            parent_id=parent_id,
            tape_index=tape_index,
            metadata=dict(metadata or {}),
        )
        thread = self._by_thread.setdefault(thread_id, [])
        for i, existing in enumerate(thread):
            if existing.checkpoint_id == checkpoint_id:
                thread[i] = record
                return record
        thread.append(record)
        return record

    def get(self, thread_id: str, checkpoint_id: str | None = None) -> CheckpointRecord | None:
        thread = self._by_thread.get(thread_id, [])
        if not thread:
            return None
        if checkpoint_id is None:
            return thread[-1]  # latest
        for record in thread:
            if record.checkpoint_id == checkpoint_id:
                return record
        return None

    def list(
        self,
        thread_id: str,
        *,
        before: str | None = None,
        limit: int | None = None,
    ) -> CheckpointRecordList:
        """Checkpoints newest-first (LangGraph time-travel order).

        ``before`` excludes everything from that checkpoint id onward (older
        history for time-travel); ``limit`` caps the count.
        """
        thread = list(self._by_thread.get(thread_id, []))
        if before is not None:
            cut = next((i for i, r in enumerate(thread) if r.checkpoint_id == before), len(thread))
            thread = thread[:cut]
        ordered = list(reversed(thread))
        if limit is not None:
            ordered = ordered[:limit]
        return ordered

    def history(self, thread_id: str) -> CheckpointRecordList:
        """Full newest-first history for a thread (time-travel candidates)."""
        return self.list(thread_id)


def make_tape_backed_checkpointer(
    tape: Tape, store: TapeBackedCheckpointStore | None = None
) -> Any:
    """Build a LangGraph ``BaseCheckpointSaver`` backed by the tape.

    Guarded: raises ``ImportError`` (install hint) if ``langgraph`` is missing.
    Delegates all persistence to a framework-independent
    ``TapeBackedCheckpointStore`` (the tested core); this wrapper only adapts the
    LangGraph ``BaseCheckpointSaver`` surface (``get_tuple`` / ``put`` /
    ``put_writes`` / ``list``) onto it, using the saver's own ``serde`` for
    checkpoint bytes and the ``configurable.thread_id`` / ``checkpoint_id`` config
    contract. See the module docstring for the bit-exact/$0 time-travel story.
    """
    require_langgraph()
    from langgraph.checkpoint.base import BaseCheckpointSaver, CheckpointTuple

    the_store = store if store is not None else TapeBackedCheckpointStore(tape)

    def _cfg(config: Any, key: str) -> Any:
        return (config or {}).get("configurable", {}).get(key)

    class TapeBackedCheckpointer(BaseCheckpointSaver):  # pragma: no cover - needs langgraph
        """LangGraph checkpoint saver whose storage is a tape-linked store."""

        def __init__(self) -> None:
            super().__init__()
            self.store = the_store

        def get_tuple(self, config):
            thread_id = _cfg(config, "thread_id")
            record = self.store.get(thread_id, _cfg(config, "checkpoint_id"))
            if record is None:
                return None
            checkpoint, metadata = record.data
            parent_config = None
            if record.parent_id is not None:
                parent_config = {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_id": record.parent_id,
                    }
                }
            return CheckpointTuple(
                config={
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_id": record.checkpoint_id,
                    }
                },
                checkpoint=checkpoint,
                metadata=metadata,
                parent_config=parent_config,
            )

        def list(self, config, *, filter=None, before=None, limit=None):
            thread_id = _cfg(config, "thread_id")
            before_id = _cfg(before, "checkpoint_id") if before else None
            for record in self.store.list(thread_id, before=before_id, limit=limit):
                checkpoint, metadata = record.data
                yield CheckpointTuple(
                    config={
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_id": record.checkpoint_id,
                        }
                    },
                    checkpoint=checkpoint,
                    metadata=metadata,
                )

        def put(self, config, checkpoint, metadata, new_versions):
            thread_id = _cfg(config, "thread_id")
            checkpoint_id = checkpoint["id"]
            self.store.put(
                thread_id,
                checkpoint_id,
                (checkpoint, metadata),
                parent_id=_cfg(config, "checkpoint_id"),
                tape_index=len(self.store.tape.exchanges),
            )
            return {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_id": checkpoint_id,
                }
            }

        def put_writes(self, config, writes, task_id, task_path=""):
            # Intermediate channel writes are not needed for tape-backed replay
            # (the tape, not pending writes, is the source of LLM I/O truth).
            return None

    return TapeBackedCheckpointer()


# ── the adapter ────────────────────────────────────────────────────────────────


def _set_attr(obj: Any, name: str, value: Any) -> None:
    """Assign ``obj.name = value``, falling back through pydantic's guardrails."""
    try:
        setattr(obj, name, value)
    except Exception:  # pragma: no cover - pydantic-version dependent
        object.__setattr__(obj, name, value)


class LangChainAdapter(BaseFrameworkAdapter):
    """Bind a LangChain chat model to tracefork's transport + annotate its run."""

    name = "langchain"

    def _inject_openai(self, model: Any, sync_client: Any, async_client: Any) -> list[str]:
        """``ChatOpenAI`` keeps the raw ``openai`` clients on ``root_client`` /
        ``root_async_client``; the openai SDK's ``.copy(http_client=...)`` swaps
        the transport while preserving base_url/key. Re-point ``client`` /
        ``async_client`` (the ``chat.completions`` resources langchain actually
        calls) at the copies."""
        injected: list[str] = []
        root = getattr(model, "root_client", None)
        if root is not None and hasattr(root, "copy"):
            new_root = root.copy(http_client=sync_client)
            _set_attr(model, "root_client", new_root)
            _set_attr(model, "client", new_root.chat.completions)
            injected += ["root_client", "client"]
        aroot = getattr(model, "root_async_client", None)
        if aroot is not None and hasattr(aroot, "copy"):
            new_aroot = aroot.copy(http_client=async_client)
            _set_attr(model, "root_async_client", new_aroot)
            _set_attr(model, "async_client", new_aroot.chat.completions)
            injected += ["root_async_client", "async_client"]
        return injected

    def _inject_anthropic(
        self, model: Any, sync_client: Any, async_client: Any, mode: str
    ) -> list[str]:
        """``ChatAnthropic`` has NO ``http_client`` field; its ``_client`` /
        ``_async_client`` are ``cached_property``s built lazily from an api key.

        Replay builds fresh ``anthropic`` clients wrapping the tracefork transport
        (no live key needed — the transport serves recorded bytes) and seeds them
        into the instance dict via ``object.__setattr__`` so the cached_property
        returns ours. Record copies the *existing* client (preserving the user's
        base_url/key) and only swaps the transport — that path needs a live key
        and is not offline-testable. ``anthropic`` is a hard tracefork dependency,
        so this import always succeeds.
        """
        import anthropic

        if mode == "replay":
            new_sync: Any = anthropic.Anthropic(
                api_key="sk-ant-tracefork-replay", http_client=sync_client, max_retries=0
            )
            new_async: Any = anthropic.AsyncAnthropic(
                api_key="sk-ant-tracefork-replay", http_client=async_client, max_retries=0
            )
        else:  # pragma: no cover - record needs a live key + SDK
            new_sync = model._client.copy(http_client=sync_client)
            new_async = model._async_client.copy(http_client=async_client)
        object.__setattr__(model, "_client", new_sync)
        object.__setattr__(model, "_async_client", new_async)
        return ["_client", "_async_client"]

    @staticmethod
    def _family(model: Any) -> str:
        """Detect the provider family WITHOUT touching ``_client`` (which would
        trigger ``ChatAnthropic``'s cached_property and demand an api key)."""
        module = (type(model).__module__ or "").lower()
        cls = type(model).__name__.lower()
        if "openai" in module or hasattr(model, "root_client"):
            return "openai"
        if "anthropic" in module or "anthropic" in cls:
            return "anthropic"
        return "unknown"

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
        """Route ``target`` (a ``ChatOpenAI`` / ``ChatAnthropic``) through tracefork.

        ``replay`` mode needs no inner transport and no live client; ``record``
        mode reuses the model's current underlying httpx transport as the inner
        so live calls still reach the network (that path needs the real SDK and
        is not offline-testable). On replay, a ``ReplayNondet``-backed uuid patch
        (``patch_uuid=True``) makes framework-generated ids match the tape.
        """
        family = self._family(target)
        inner = inner_async = None
        if mode == "record":
            inner, inner_async = _underlying_transports(target, family)

        sync_client, async_client, sync_t, async_t = build_http_clients(
            tape, mode, inner=inner, async_inner=inner_async, matcher=matcher, redactor=redactor
        )

        if family == "openai":
            injected = self._inject_openai(target, sync_client, async_client)
        elif family == "anthropic":
            injected = self._inject_anthropic(target, sync_client, async_client, mode)
        else:
            injected = []

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
        """Map one LangChain callback event (as a dict) to a neutral ``Step``.

        ``event["event"]`` is the callback name (``"on_chat_model_start"``,
        ``"on_llm_end"``, ...); the rest are its payload fields. Start events add
        a step; end events update the matching step's ``outputs``. Returns the
        affected ``Step`` (a fresh empty ``Step`` if an end event has no match).
        """
        core = _EventCore(self.dag)
        return core.dispatch(event)


class _EventCore(TraceforkCallbackCore):
    """Dispatch a single dict-shaped callback event onto the callback core."""

    def dispatch(self, event: Mapping[str, Any]) -> Step:
        etype = str(event.get("event") or event.get("type") or "")
        run_id = event.get("run_id")
        parent = event.get("parent_run_id")
        serialized = event.get("serialized", {})
        meta = {"metadata": event.get("metadata")}
        if etype == "on_chain_start":
            return self.on_chain_start(
                serialized, event.get("inputs"), run_id=run_id, parent_run_id=parent, **meta
            )
        if etype == "on_chain_end":
            return self.on_chain_end(event.get("outputs"), run_id=run_id) or Step(str(run_id))
        if etype == "on_llm_start":
            return self.on_llm_start(
                serialized, event.get("prompts"), run_id=run_id, parent_run_id=parent, **meta
            )
        if etype == "on_chat_model_start":
            return self.on_chat_model_start(
                serialized, event.get("messages"), run_id=run_id, parent_run_id=parent, **meta
            )
        if etype == "on_llm_end":
            return self.on_llm_end(event.get("response"), run_id=run_id) or Step(str(run_id))
        if etype == "on_tool_start":
            return self.on_tool_start(
                serialized, event.get("input_str"), run_id=run_id, parent_run_id=parent, **meta
            )
        if etype == "on_tool_end":
            return self.on_tool_end(event.get("output"), run_id=run_id) or Step(str(run_id))
        # Unknown event: record a neutral step so nothing is silently dropped.
        return self._start(etype or "unknown", run_id, parent, str(event.get("name", "")))


def _underlying_transports(
    target: Any, family: str
) -> tuple[Any, Any]:  # pragma: no cover - needs real SDK
    """Best-effort (sync, async) inner httpx transports of a chat model's client.

    Used only in record mode, which requires the live SDK (and, for anthropic, an
    api key — reading ``_client`` builds the cached client). Returns ``(None,
    None)`` when the shape is unrecognized, so record then errors clearly instead
    of silently dropping the live call.
    """
    if family == "openai":
        sync_c = getattr(target, "root_client", None)
        async_c = getattr(target, "root_async_client", None)
    elif family == "anthropic":
        sync_c = getattr(target, "_client", None)
        async_c = getattr(target, "_async_client", None)
    else:
        return None, None
    inner = getattr(getattr(sync_c, "_client", None), "_transport", None)
    async_inner = getattr(getattr(async_c, "_client", None), "_transport", None)
    return inner, async_inner


# Register the built-in adapter at import time (never via the entry-point path).
register_framework_adapter(LangChainAdapter())


__all__ = [
    "CheckpointRecord",
    "FRAMEWORKS_IMPORT_HINT",
    "LangChainAdapter",
    "TapeBackedCheckpointStore",
    "TraceforkCallbackCore",
    "langchain_available",
    "langgraph_available",
    "make_callback_handler",
    "make_tape_backed_checkpointer",
    "require_langchain",
    "require_langgraph",
]
