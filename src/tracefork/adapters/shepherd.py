"""Optional Shepherd framework adapter — OpenAI-path only, synthetic-double-validated.

Shepherd (see ``docs/shepherd-gap-analysis.md``) is a privately-analyzed
~213k LOC agent-framework codebase, not a published PyPI package this repo
can ``pip install`` or CI against. That is the one real, verified constraint
that makes this adapter different from every other one in ``adapters/``:
there is no real ``shepherd`` import to guard, so — unlike ``autogen.py`` /
``openai_agents.py`` / ``adk.py`` / ``crewai.py`` — this module ships with
**no** ``shepherd_available()``/``require_shepherd()`` guard functions, no
``pytest.importorskip`` real-framework test tier, and no new pyproject
extra. ``bind()``/``on_step()`` are validated ENTIRELY against a synthetic
double that mimics the shape Shepherd's ``OpenAIProvider`` is understood to
hold (an internal ``openai``-SDK client) — never a real Shepherd package.

Two seams, reusing tracefork's *existing* byte capture — never a second one:

* **``bind``** routes Shepherd's ``OpenAIProvider``'s underlying ``openai``
  client through a ``TraceforkTransport``, the same ``client.copy(http_client=…)``
  move ``recorder.py``/``adapters/openai_agents.py``/``adapters/autogen.py``
  already use. Shepherd's exact attribute name for that client is not
  available to verify (privately-analyzed codebase, not a published API),
  so ``bind`` searches the SAME short candidate-attribute list
  ``adapters/openai_agents.py`` already uses (``_client``, ``client``,
  ``openai_client``, ``_openai_client``) rather than hard-coding one.
  **Scope, stated honestly, not silently narrowed:** only that OpenAI-path
  client is bound. Shepherd's Claude and OpenCode providers are NOT bound by
  this adapter — that is annotation-only / out of scope for this bead, and
  ``bind``'s ``notes`` field says so explicitly rather than implying
  full-provider parity with the other four adapters.
* **``on_step``** maps one Shepherd-shaped provider event (a plain dict) to a
  neutral ``Step``, using the same defensive attribute-or-mapping extraction
  (``_get``) ``adapters/openai_agents.py``'s ``TraceforkTracingCore`` uses.
  Shepherd's actual ``ProviderEvent``/``ModelRequest``/``ModelResponse`` field
  names aren't available to verify against either, so this mapping is a
  best-effort, documented-as-speculative generic shape (``id``/``parent_id``/
  ``kind``/``model``/``inputs``/``outputs``), not a claim of a verified event
  schema. Observer-only, per the design invariant in ``adapters/base.py`` — it
  feeds the step-DAG, never a second capture path.
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

# Appended to every bind() notes string (found or not) — see the module
# docstring: this adapter binds Shepherd's OpenAI-path client only.
_SCOPE_NOTE = (
    "Claude and OpenCode providers are not bound by this adapter "
    "(annotation-only, out of scope) - only Shepherd's OpenAIProvider client is."
)


# ── defensive extractor (works on dict-or-object payloads, mirrors openai_agents.py) ──


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


# ── framework-independent event core (fully offline-testable) ──────────────────


class TraceforkShepherdCore:
    """Accumulate a ``StepDAG`` from Shepherd-shaped provider events, framework-free.

    Nothing here imports a real ``shepherd`` package (none exists to import —
    see the module docstring); a test drives this directly with plain dicts.
    """

    def __init__(self, dag: StepDAG | None = None) -> None:
        self.dag = dag if dag is not None else StepDAG()

    def dispatch(self, event: Mapping[str, Any]) -> Step:
        """Map one generic ``{id, parent_id, kind, model, inputs, outputs}``-shaped
        event to a neutral ``Step``.

        An ``"end"``-typed event updates (or, if its start was never seen,
        creates a placeholder for) the step matching its id rather than being
        silently dropped; anything else (an explicit ``"start"``, or any other
        unrecognized event type) is recorded as a full step, keyed by its own
        ``kind``/event-type so nothing unknown disappears.
        """
        etype = str(_get(event, "event", "type", default="") or "")
        raw_id = _get(event, "id", "step_id", default=None)
        step_id = str(raw_id) if raw_id is not None else str(id(event))

        if etype == "end":
            existing = self.dag.by_id(step_id)
            if existing is not None:
                outputs = _get(event, "outputs", default=None)
                if outputs is not None:
                    existing.outputs = outputs
                return existing
            # No matching start was ever recorded: an honest placeholder,
            # mirroring adapters/openai_agents.py's on_span_end fallback.
            return self.dag.add(Step(step_id=step_id))

        parent_id = _get(event, "parent_id", default=None)
        kind = str(_get(event, "kind", default="") or (etype if etype != "start" else ""))
        step = Step(
            step_id=step_id,
            parent_id=str(parent_id) if parent_id is not None else None,
            kind=kind,
            name=str(_get(event, "name", default="") or ""),
            model=_get(event, "model", default=None),
            inputs=_get(event, "inputs", default=None),
            outputs=_get(event, "outputs", default=None),
        )
        return self.dag.add(step)


# ── the adapter ────────────────────────────────────────────────────────────────

# Candidate attribute names Shepherd's ``OpenAIProvider`` might store its
# underlying ``openai`` client under — the SAME list ``adapters/openai_agents.py``
# searches. Not a frozen public API (Shepherd is not a published package to
# verify one against — see the module docstring).
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


def _underlying_transports(target: Any) -> tuple[Any, Any]:  # pragma: no cover - needs Shepherd
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


class ShepherdAdapter(BaseFrameworkAdapter):
    """Bind Shepherd's OpenAI-path provider client to tracefork's transport + annotate its run.

    OpenAI-path only (see the module docstring) — Shepherd's Claude/OpenCode
    providers are explicitly NOT bound here.
    """

    name = "shepherd"

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
        """Route ``target`` (Shepherd's ``OpenAIProvider``, holding an ``openai``
        client) through tracefork.

        ``replay`` mode needs no inner transport and no live client; ``record``
        mode reuses the target's current underlying httpx transport as the inner
        so live calls still reach the network (that path needs the real,
        privately-analyzed Shepherd codebase and is not offline-testable). On
        replay, a ``ReplayNondet``-backed uuid patch (``patch_uuid=True``) makes
        framework-generated ids match the tape.
        """
        inner = inner_async = None
        if mode == "record":  # pragma: no cover - needs the real Shepherd codebase
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

        if injected:
            notes = _SCOPE_NOTE
        else:
            notes = (
                "no known OpenAI-client attribute found on target "
                f"({type(target).__name__}); nothing was injected. {_SCOPE_NOTE}"
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
        """Map one Shepherd-shaped provider event (as a dict) to a neutral ``Step``.

        See ``TraceforkShepherdCore.dispatch`` — a best-effort, documented-as-
        speculative generic mapping (Shepherd's real event schema is not
        available to verify against).
        """
        core = TraceforkShepherdCore(self.dag)
        return core.dispatch(event)


# Register the built-in adapter at import time (never via the entry-point path).
register_framework_adapter(ShepherdAdapter())


__all__ = [
    "ShepherdAdapter",
    "TraceforkShepherdCore",
]
