"""Recorder and AsyncRecorder — one-line wrappers that record any Anthropic client.

`Recorder` wraps a sync `anthropic.Anthropic` client; `AsyncRecorder` wraps an
`anthropic.AsyncAnthropic` client. Both are context managers. Inside the `with`
block, `uuid.uuid4` is patched globally so agent-generated IDs are recorded.
`datetime.datetime.now` is NOT patched here — it is a C classmethod on an
immutable type (Python 3.12+) and replacing `datetime.datetime` with a subclass
breaks pydantic's lazy schema builder inside the Anthropic SDK. Agents that need
deterministic timestamps or random draws should call `nondet.now_iso()` /
`nondet.random_float()` via `NondetSource` (see `nondet.py`) — like the clock,
`random` is not patched globally here.

An opt-in `boundary_guard=True` (or `TraceforkConfig(boundary_guard=True)`)
wraps the recording window in a `BoundaryGuard`, which hard-errors on thread/
subprocess spawn or direct `random`/clock reads that bypass `NondetSource` —
see `boundary_guard.py`. Default is off; behavior is unchanged unless a caller
opts in.

Both recorders also populate `tape.provenance` (matcher_name/boundary_guard/
nondet_mode — see `tape.py`) from values already in scope, a witness block
`ReplayVerifier` can optionally check against the replay-time configuration.

Usage (sync):
    with Recorder(client, agent_name="my-agent") as rec:
        result = my_agent(rec.client)
    tape = rec.tape

Usage (async):
    async with AsyncRecorder(async_client) as rec:
        result = await my_agent(rec.client)
    tape = rec.tape
"""

from __future__ import annotations

import uuid as _uuid_module
from collections.abc import Callable
from typing import Any

import anthropic
import httpx

from .boundary_guard import BoundaryGuard
from .checkpoint import CheckpointWriter
from .config import TraceforkConfig
from .matcher import IDENTITY_MATCHER, RequestMatcher
from .nondet import RecordingNondet
from .observability import traced_span
from .redact import Redactor
from .tape import Tape
from .transport import AsyncTraceforkTransport, TraceforkTransport


class Recorder:
    """Sync context manager that records an Anthropic client's I/O.

    ``matcher`` is an opt-in ``RequestMatcher``; the default (``None``) is the
    identity matcher (raw ``sha256`` of the request body). If a canonicalizing
    matcher is passed, the *same* matcher must be used at replay/fork/verify time
    or the fingerprints will not line up.

    ``redactor`` is an opt-in ``Redactor`` (see ``redact.py``); the default
    (``None``) records exactly as before this seam existed — byte-identical.
    When given, it wraps ``matcher`` so header/secret-env redaction runs inside
    the fingerprinting seam (record and replay still hash the same redacted
    form), scrubs the response body before it is stored, and — if the redactor
    also scrubs message content — marks ``tape.content_redacted = True``
    (forensic-only; see the README's Redaction section).

    ``config`` is an opt-in ``TraceforkConfig`` (see ``config.py``); the
    default (``None``) is likewise byte-identical to today. When given *and*
    ``redactor``/``boundary_guard`` are not passed explicitly, ``config``
    supplies them — an explicit argument always wins.

    ``boundary_guard`` is an opt-in tri-state flag (see ``boundary_guard.py``):
    ``None`` (default) defers to ``config.boundary_guard`` (itself ``False``
    by default); an explicit ``True``/``False`` always wins over ``config``.
    When effectively ``True``, a ``BoundaryGuard`` wraps the whole recording
    window (from here through ``__exit__``), hard-erroring on thread/
    subprocess spawn or direct ``random``/clock reads. Default is off —
    identical to before this flag existed.

    ``checkpoint_path`` is an opt-in crash-safety path (see ``checkpoint.py``):
    when given, each recorded exchange is durably committed to a local SQLite
    file at that path the instant it happens (not just at ``tape.save()``
    time), and a clean ``__exit__`` finalizes it. Scoped to exchanges only
    (not draws) — see ``checkpoint.py``'s module docstring. Default ``None``
    is byte-identical to before this flag existed.
    """

    def __init__(
        self,
        client: anthropic.Anthropic,
        agent_name: str = "",
        *,
        matcher: RequestMatcher | None = None,
        redactor: Redactor | None = None,
        config: TraceforkConfig | None = None,
        boundary_guard: bool | None = None,
        checkpoint_path: str | None = None,
    ) -> None:
        self._orig_client = client
        self._agent_name = agent_name
        self._matcher = matcher
        self._redactor = redactor
        self._config = config
        self._boundary_guard_flag = boundary_guard
        self._checkpoint_path = checkpoint_path
        self._nondet: RecordingNondet | None = None
        self._tape: Tape | None = None
        self._wrapped_client: anthropic.Anthropic | None = None
        self._orig_uuid4: Callable[[], _uuid_module.UUID] | None = None
        self._guard: BoundaryGuard | None = None
        self._span_cm: Any = None
        self._checkpoint: CheckpointWriter | None = None

    @property
    def client(self) -> anthropic.Anthropic:
        if self._wrapped_client is None:
            raise RuntimeError("Use Recorder as a context manager (with Recorder(client) as rec:)")
        return self._wrapped_client

    @property
    def tape(self) -> Tape:
        if self._tape is None:
            raise RuntimeError("Use Recorder as a context manager")
        return self._tape

    def __enter__(self) -> Recorder:
        # No-op unless OTel self-instrumentation is explicitly enabled (see
        # observability.py) — covers the whole recording window, enter to exit.
        self._span_cm = traced_span("tracefork.record", agent_name=self._agent_name)
        self._span_cm.__enter__()
        # RecordingNondet captures the real datetime.now and uuid.uuid4 in __init__
        # before we patch uuid.uuid4 below. Order matters.
        self._nondet = RecordingNondet()
        self._tape = Tape(agent_name=self._agent_name)
        # Share the draws list so recording nondet populates the tape's draws directly
        self._tape.draws = self._nondet.draws
        # An explicit `redactor=` always wins; otherwise fall back to `config`'s
        # (default `config=None` → `None`, byte-identical to today).
        redactor = self._redactor
        if redactor is None and self._config is not None:
            redactor = self._config.build_redactor()
        if redactor is not None:
            self._tape.content_redacted = redactor.content_redacted

        # Extract the original httpx transport to use as the recording inner transport.
        # This preserves ScriptedFakeLLM in tests and HTTPTransport in production.
        orig_inner = self._orig_client._client._transport
        effective_matcher = redactor.matcher(self._matcher) if redactor else self._matcher
        on_exchange = None
        if self._checkpoint_path is not None:
            self._checkpoint = CheckpointWriter(
                self._checkpoint_path,
                agent_name=self._agent_name,
                boundary=self._tape.boundary,
            )
            on_exchange = self._checkpoint.append_exchange
        transport = TraceforkTransport(
            "record",
            self._tape,
            orig_inner,
            matcher=effective_matcher,
            redactor=redactor,
            on_exchange=on_exchange,
        )
        # `.copy()` preserves the original client's base_url, auth_token, default
        # headers/query and timeout — only the transport and retries are swapped, so
        # a proxied or custom-base_url client still records faithfully.
        self._wrapped_client = self._orig_client.copy(
            http_client=httpx.Client(transport=transport),
            max_retries=0,
        )

        # Patch uuid.uuid4 (regular module-level function — directly assignable).
        # The Anthropic SDK may also call uuid.uuid4() internally; all draws are recorded.
        nondet = self._nondet
        self._orig_uuid4 = _uuid_module.uuid4

        def _patched_uuid4() -> _uuid_module.UUID:
            return _uuid_module.UUID(nondet.new_uuid_hex())

        _uuid_module.uuid4 = _patched_uuid4

        # Explicit `boundary_guard=` always wins; otherwise fall back to `config`'s
        # (default `config=None` → `False`, byte-identical to today).
        guard_enabled = self._boundary_guard_flag
        if guard_enabled is None:
            guard_enabled = self._config.boundary_guard if self._config is not None else False
        if guard_enabled:
            self._guard = BoundaryGuard()
            self._guard.__enter__()

        # Provenance/witness block (see tape.py): the matcher/boundary-guard/
        # nondet-mode context this tape was recorded under, for `ReplayVerifier`'s
        # opt-in mismatch check. Never fed into `digest()`.
        self._tape.provenance = {
            "matcher_name": (effective_matcher or IDENTITY_MATCHER).name,
            "boundary_guard": str(guard_enabled).lower(),
            "nondet_mode": "recording",
        }
        return self

    def __exit__(self, *args: object) -> None:
        if self._guard is not None:
            self._guard.__exit__(*args)
            self._guard = None
        _uuid_module.uuid4 = self._orig_uuid4  # type: ignore[assignment]
        # Finalize the checkpoint only on a clean exit (no exception): a crash
        # mid-recording should leave the checkpoint visibly non-finalized (see
        # checkpoint.py) rather than have __exit__ paper over it.
        if self._checkpoint is not None and args[0] is None:
            assert self._tape is not None
            self._checkpoint.finalize(self._tape)
        if self._span_cm is not None:
            self._span_cm.__exit__(*args)
            self._span_cm = None


class AsyncRecorder:
    """Async context manager that records an AsyncAnthropic client's I/O.

    See ``Recorder`` for the ``matcher`` / ``redactor`` / ``config`` /
    ``boundary_guard`` / ``checkpoint_path`` contract — identical here.
    """

    def __init__(
        self,
        client: anthropic.AsyncAnthropic,
        agent_name: str = "",
        *,
        matcher: RequestMatcher | None = None,
        redactor: Redactor | None = None,
        config: TraceforkConfig | None = None,
        boundary_guard: bool | None = None,
        checkpoint_path: str | None = None,
    ) -> None:
        self._orig_client = client
        self._agent_name = agent_name
        self._matcher = matcher
        self._redactor = redactor
        self._config = config
        self._boundary_guard_flag = boundary_guard
        self._checkpoint_path = checkpoint_path
        self._nondet: RecordingNondet | None = None
        self._tape: Tape | None = None
        self._wrapped_client: anthropic.AsyncAnthropic | None = None
        self._orig_uuid4: Callable[[], _uuid_module.UUID] | None = None
        self._guard: BoundaryGuard | None = None
        self._span_cm: Any = None
        self._checkpoint: CheckpointWriter | None = None

    @property
    def client(self) -> anthropic.AsyncAnthropic:
        if self._wrapped_client is None:
            raise RuntimeError("Use AsyncRecorder as an async context manager")
        return self._wrapped_client

    @property
    def tape(self) -> Tape:
        if self._tape is None:
            raise RuntimeError("Use AsyncRecorder as an async context manager")
        return self._tape

    async def __aenter__(self) -> AsyncRecorder:
        # No-op unless OTel self-instrumentation is explicitly enabled (see
        # observability.py) — covers the whole recording window, enter to exit.
        self._span_cm = traced_span("tracefork.record", agent_name=self._agent_name)
        self._span_cm.__enter__()
        self._nondet = RecordingNondet()
        self._tape = Tape(agent_name=self._agent_name)
        self._tape.draws = self._nondet.draws
        redactor = self._redactor
        if redactor is None and self._config is not None:
            redactor = self._config.build_redactor()
        if redactor is not None:
            self._tape.content_redacted = redactor.content_redacted

        orig_inner = self._orig_client._client._transport
        effective_matcher = redactor.matcher(self._matcher) if redactor else self._matcher
        on_exchange = None
        if self._checkpoint_path is not None:
            self._checkpoint = CheckpointWriter(
                self._checkpoint_path,
                agent_name=self._agent_name,
                boundary=self._tape.boundary,
            )
            on_exchange = self._checkpoint.append_exchange
        transport = AsyncTraceforkTransport(
            "record",
            self._tape,
            orig_inner,
            matcher=effective_matcher,
            redactor=redactor,
            on_exchange=on_exchange,
        )
        # `.copy()` preserves base_url, auth_token, default headers/query and timeout
        # (see the sync Recorder) — only the transport and retries are swapped.
        self._wrapped_client = self._orig_client.copy(
            http_client=httpx.AsyncClient(transport=transport),
            max_retries=0,
        )

        nondet = self._nondet
        self._orig_uuid4 = _uuid_module.uuid4

        def _patched_uuid4() -> _uuid_module.UUID:
            return _uuid_module.UUID(nondet.new_uuid_hex())

        _uuid_module.uuid4 = _patched_uuid4

        # See the sync Recorder for the explicit-vs-config precedence rule.
        guard_enabled = self._boundary_guard_flag
        if guard_enabled is None:
            guard_enabled = self._config.boundary_guard if self._config is not None else False
        if guard_enabled:
            self._guard = BoundaryGuard()
            self._guard.__enter__()

        # See the sync Recorder for the provenance/witness block contract.
        self._tape.provenance = {
            "matcher_name": (effective_matcher or IDENTITY_MATCHER).name,
            "boundary_guard": str(guard_enabled).lower(),
            "nondet_mode": "recording",
        }
        return self

    async def __aexit__(self, *args: object) -> None:
        if self._guard is not None:
            self._guard.__exit__(*args)
            self._guard = None
        _uuid_module.uuid4 = self._orig_uuid4  # type: ignore[assignment]
        # See the sync Recorder for the clean-exit-only finalize contract.
        if self._checkpoint is not None and args[0] is None:
            assert self._tape is not None
            self._checkpoint.finalize(self._tape)
        if self._span_cm is not None:
            self._span_cm.__exit__(*args)
            self._span_cm = None
