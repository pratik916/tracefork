"""Framework adapter seam: route a framework's LLM client through tracefork,
and overlay a framework-neutral **step-DAG** (annotation only).

Design invariant (see the repo ``CLAUDE.md`` and the feature research): the
byte seam stays at the httpx transport (``transport.py``). A framework's
tracing/callback API is **observer-only** here — it feeds a structure/annotation
layer (``Step`` / ``StepDAG``), never a second capture path. An adapter's real
job is ``bind()``: point the framework's underlying LLM client at tracefork's
*existing* ``TraceforkTransport`` (+ optional ``NondetSource``), reusing the
record/replay seam rather than re-capturing at the framework layer. That is what
keeps a framework run **bit-exact and $0** on replay — the same contract every
other seam in tracefork inherits from ``transport.py``.

Nothing in this module imports any third-party framework; it is pure, offline,
and fully testable with synthetic callback events and a fake LLM client. The
concrete LangChain/LangGraph adapter (``adapters/langchain.py``) guards its
framework imports and registers itself here.
"""

from __future__ import annotations

import uuid as _uuid_module
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import httpx

from ..matcher import RequestMatcher
from ..nondet import NondetSource, ReplayNondet
from ..plugins import ADAPTER_GROUP, Registry
from ..redact import Redactor
from ..tape import Tape
from ..transport import AsyncTraceforkTransport, TraceforkTransport

# LLM-flavoured step kinds — the two the blame/report seam treats as model calls.
LLM_STEP_KINDS = ("llm", "chat_model")


@dataclass
class Step:
    """One framework-neutral node in a run's step-DAG (a tape *annotation*).

    A ``Step`` is structure only — it never carries the bit-exact request/response
    bytes (those live in the tape's ``exchanges``). ``tape_index`` links an LLM
    step to the tape exchange it produced, so the report/blame seam can line up a
    framework's chain/tool structure with the raw byte record.
    """

    step_id: str
    parent_id: str | None = None
    kind: str = ""
    name: str = ""
    model: str | None = None
    tape_index: int | None = None
    inputs: Any = None
    outputs: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def is_llm(self) -> bool:
        """Whether this step is a model call (``"llm"`` or ``"chat_model"``)."""
        return self.kind in LLM_STEP_KINDS


@dataclass
class StepDAG:
    """An ordered, parent-linked overlay of a framework run.

    Steps are held in insertion order (a run's natural, deterministic order);
    parent/child structure is recovered from each step's ``parent_id``. This is
    the single normalized shape every framework adapter targets, so downstream
    consumers (report, blame annotation) never learn one framework's run-tree
    schema.
    """

    steps: list[Step] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self) -> Any:
        return iter(self.steps)

    def add(self, step: Step) -> Step:
        """Append ``step`` (overwriting any existing step with the same id)."""
        for i, existing in enumerate(self.steps):
            if existing.step_id == step.step_id:
                self.steps[i] = step
                return step
        self.steps.append(step)
        return step

    def by_id(self, step_id: str) -> Step | None:
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None

    def children(self, step_id: str) -> list[Step]:
        return [s for s in self.steps if s.parent_id == step_id]

    def roots(self) -> list[Step]:
        """Steps whose parent is absent from the DAG (top-level nodes)."""
        ids = {s.step_id for s in self.steps}
        return [s for s in self.steps if s.parent_id is None or s.parent_id not in ids]

    def of_kind(self, kind: str) -> list[Step]:
        return [s for s in self.steps if s.kind == kind]

    def llm_steps(self) -> list[Step]:
        """Every model-call step, in order — the ones that map to tape exchanges."""
        return [s for s in self.steps if s.is_llm()]

    def assign_tape_indices(self) -> None:
        """Number the LLM steps 0..n in DAG order, mirroring tape exchange order.

        The byte seam records one exchange per model call in the order the agent
        made them; a single-process framework run makes them in DAG order, so
        this is the annotation-to-tape linkage the report/blame seam reads.
        """
        for i, step in enumerate(self.llm_steps()):
            step.tape_index = i

    @classmethod
    def from_steps(cls, steps: Iterable[Step]) -> StepDAG:
        dag = cls()
        for step in steps:
            dag.add(step)
        return dag

    @classmethod
    def from_run_tree(
        cls,
        tree: Any,
        *,
        parent_id: str | None = None,
        _dag: StepDAG | None = None,
    ) -> StepDAG:
        """Flatten a framework run tree (nested nodes) into one ordered step-DAG.

        Accepts a single root node, or an iterable of root nodes, where each node
        is either a mapping or an object exposing the LangChain/LangSmith run-tree
        surface: an id (``id`` / ``step_id`` / ``run_id``), a kind
        (``kind`` / ``run_type``), a ``name``, an optional ``model``, and nested
        children under ``children`` / ``child_runs``. Depth-first, preserving each
        node's child order, so the flattening is deterministic.
        """
        dag = _dag if _dag is not None else cls()
        if tree is None:
            return dag
        if _is_node_sequence(tree):
            for node in tree:
                cls.from_run_tree(node, parent_id=parent_id, _dag=dag)
            return dag
        step = _node_to_step(tree, parent_id)
        dag.add(step)
        for child in _node_children(tree):
            cls.from_run_tree(child, parent_id=step.step_id, _dag=dag)
        return dag


def _is_node_sequence(tree: Any) -> bool:
    return isinstance(tree, (list, tuple))


def _node_get(node: Any, *keys: str, default: Any = None) -> Any:
    """Read the first present attribute/key from ``node`` (mapping or object)."""
    if isinstance(node, Mapping):
        for key in keys:
            if key in node:
                return node[key]
        return default
    for key in keys:
        if hasattr(node, key):
            return getattr(node, key)
    return default


def _node_children(node: Any) -> list[Any]:
    children = _node_get(node, "children", "child_runs", default=None)
    return list(children) if children else []


def _node_to_step(node: Any, parent_id: str | None) -> Step:
    raw_id = _node_get(node, "id", "step_id", "run_id", default=None)
    step_id = str(raw_id) if raw_id is not None else _uuid_module.uuid4().hex
    explicit_parent = _node_get(node, "parent_id", "parent_run_id", default=None)
    kind = _node_get(node, "kind", "run_type", default="") or ""
    return Step(
        step_id=step_id,
        parent_id=str(explicit_parent) if explicit_parent is not None else parent_id,
        kind=str(kind),
        name=str(_node_get(node, "name", default="") or ""),
        model=_node_get(node, "model", default=None),
        metadata=_node_get(node, "metadata", "extra", default=None) or {},
    )


@dataclass
class BindResult:
    """What ``FrameworkAdapter.bind`` injected — returned for inspection/teardown.

    ``http_client`` / ``http_async_client`` are the httpx clients (wrapping a
    ``TraceforkTransport``) that were routed into the framework's LLM client;
    ``nondet`` is the ``NondetSource`` the caller should hand the framework for
    its own clock/id draws (replay mode installs a ``ReplayNondet`` from the
    tape). ``injected_fields`` names what ``bind`` mutated, and ``notes`` records
    anything the adapter could not fully wire (honest, not silent).
    """

    mode: str
    http_client: httpx.Client | None = None
    http_async_client: httpx.AsyncClient | None = None
    transport: TraceforkTransport | None = None
    async_transport: AsyncTraceforkTransport | None = None
    nondet: NondetSource | None = None
    injected_fields: tuple[str, ...] = ()
    notes: str = ""


def build_http_clients(
    tape: Tape,
    mode: str,
    *,
    inner: httpx.BaseTransport | None = None,
    async_inner: httpx.AsyncBaseTransport | None = None,
    matcher: RequestMatcher | None = None,
    redactor: Redactor | None = None,
) -> tuple[httpx.Client, httpx.AsyncClient, TraceforkTransport, AsyncTraceforkTransport]:
    """Build the sync+async httpx clients an adapter injects into a framework.

    This is the one place adapters reuse the existing capture seam: each client
    wraps a ``TraceforkTransport`` in exactly the mode/matcher/redactor contract
    ``transport.py`` already owns. ``replay`` mode needs no inner transport (an
    unrecorded request is a hard error, per ``transport.py``); ``record`` mode
    requires the framework client's current transport as ``inner`` so live calls
    still reach the network.
    """
    sync_transport = TraceforkTransport(mode, tape, inner, matcher=matcher, redactor=redactor)
    async_transport = AsyncTraceforkTransport(
        mode, tape, async_inner, matcher=matcher, redactor=redactor
    )
    return (
        httpx.Client(transport=sync_transport),
        httpx.AsyncClient(transport=async_transport),
        sync_transport,
        async_transport,
    )


class UuidPatch:
    """Globally serve tape-recorded uuids on replay (mirrors ``recorder.py``).

    Frameworks call ``uuid.uuid4()`` internally (run ids, message ids) and never
    read tracefork's ``NondetSource``; on replay those ids must match the tape or
    request bytes diverge. This patches ``uuid.uuid4`` for the bind window the
    same way ``Recorder`` does for record — a no-op until ``install`` and always
    reversible via ``restore``.
    """

    def __init__(self, nondet: NondetSource) -> None:
        self._nondet = nondet
        self._orig: Callable[[], _uuid_module.UUID] | None = None
        self._active = False

    def install(self) -> None:
        if self._active:
            return
        self._orig = _uuid_module.uuid4
        nondet = self._nondet

        def _patched_uuid4() -> _uuid_module.UUID:
            return _uuid_module.UUID(nondet.new_uuid_hex())

        _uuid_module.uuid4 = _patched_uuid4
        self._active = True

    def restore(self) -> None:
        if not self._active:
            return
        _uuid_module.uuid4 = self._orig  # type: ignore[assignment]
        self._active = False


@runtime_checkable
class FrameworkAdapter(Protocol):
    """Bind a framework's LLM client to tracefork, and annotate its run.

    Three responsibilities, matching the design (byte seam at httpx, callbacks as
    annotation):

    * ``bind`` — route the framework's underlying LLM client through
      ``TraceforkTransport`` (+ a ``NondetSource``), returning a ``BindResult``.
    * ``on_step`` — turn one framework callback/tracing event into a neutral
      ``Step`` (a tape annotation), for assembling a ``StepDAG``.
    * ``teardown`` — undo anything global ``bind`` installed (e.g. a uuid patch).
    """

    name: str

    def bind(self, target: Any, tape: Tape, mode: str = "replay", **kwargs: Any) -> BindResult: ...

    def on_step(self, event: Mapping[str, Any]) -> Step: ...

    def teardown(self) -> None: ...


class BaseFrameworkAdapter:
    """Shared plumbing for concrete adapters (uuid patch lifecycle + a DAG).

    Subclasses implement ``bind`` (framework-specific injection) and ``on_step``
    (framework-specific event mapping); everything else — the ``StepDAG`` an
    adapter accumulates, the optional replay uuid patch, and ``teardown`` — is
    handled here so each concrete adapter stays small.
    """

    name = "base"

    def __init__(self) -> None:
        self.dag = StepDAG()
        self._uuid_patch: UuidPatch | None = None

    def _replay_nondet(self, tape: Tape) -> ReplayNondet:
        return ReplayNondet(tape.draws)

    def _install_uuid_patch(self, nondet: NondetSource) -> None:
        self._uuid_patch = UuidPatch(nondet)
        self._uuid_patch.install()

    def record_step(self, step: Step) -> Step:
        """Add ``step`` to this adapter's accumulated DAG (and return it)."""
        return self.dag.add(step)

    def bind(self, target: Any, tape: Tape, mode: str = "replay", **kwargs: Any) -> BindResult:
        raise NotImplementedError

    def on_step(self, event: Mapping[str, Any]) -> Step:
        raise NotImplementedError

    def teardown(self) -> None:
        if self._uuid_patch is not None:
            self._uuid_patch.restore()
            self._uuid_patch = None


# ── registry ──────────────────────────────────────────────────────────────────
#
# Same generic `Registry` (see `plugins.py`) every other tracefork seam uses.
# Built-ins register directly at import time (never via the entry-point path);
# third-party adapters are opt-in and security-gated by `load_adapter_entry_points`.

_REGISTRY: Registry[FrameworkAdapter] = Registry(ADAPTER_GROUP, kind="framework adapter")


def register_framework_adapter(adapter: FrameworkAdapter, *, name: str | None = None) -> None:
    """Register ``adapter`` under ``name`` (defaults to ``adapter.name``)."""
    _REGISTRY.register(name or adapter.name, adapter)


def get_framework_adapter(name: str) -> FrameworkAdapter:
    """Look up a registered framework adapter by name."""
    return _REGISTRY.get_or_raise(name)


def registered_framework_adapters() -> list[str]:
    """Sorted names of all registered framework adapters."""
    return _REGISTRY.names()


def load_adapter_entry_points(
    *, allow: frozenset[str] | set[str] | None = None, allow_all: bool = False
) -> list[str]:
    """Opt-in: discover third-party framework adapters advertised under the
    ``tracefork.adapters`` entry-point group (see ``plugins.py`` for the
    security-gating contract — nothing loads unless explicitly allowlisted).
    """
    return _REGISTRY.load_entry_points(allow=allow, allow_all=allow_all)
