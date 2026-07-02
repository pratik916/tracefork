"""Recorder and AsyncRecorder — one-line wrappers that record any Anthropic client.

`Recorder` wraps a sync `anthropic.Anthropic` client; `AsyncRecorder` wraps an
`anthropic.AsyncAnthropic` client. Both are context managers. Inside the `with`
block, `uuid.uuid4` is patched globally so agent-generated IDs are recorded.
`datetime.datetime.now` is NOT patched here — it is a C classmethod on an
immutable type (Python 3.12+) and replacing `datetime.datetime` with a subclass
breaks pydantic's lazy schema builder inside the Anthropic SDK. Agents that need
deterministic timestamps should call `nondet.now_iso()` via `NondetSource`.

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

import anthropic
import httpx

from .matcher import RequestMatcher
from .nondet import RecordingNondet
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
    """

    def __init__(
        self,
        client: anthropic.Anthropic,
        agent_name: str = "",
        *,
        matcher: RequestMatcher | None = None,
        redactor: Redactor | None = None,
    ) -> None:
        self._orig_client = client
        self._agent_name = agent_name
        self._matcher = matcher
        self._redactor = redactor
        self._nondet: RecordingNondet | None = None
        self._tape: Tape | None = None
        self._wrapped_client: anthropic.Anthropic | None = None
        self._orig_uuid4: Callable[[], _uuid_module.UUID] | None = None

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
        # RecordingNondet captures the real datetime.now and uuid.uuid4 in __init__
        # before we patch uuid.uuid4 below. Order matters.
        self._nondet = RecordingNondet()
        self._tape = Tape(agent_name=self._agent_name)
        # Share the draws list so recording nondet populates the tape's draws directly
        self._tape.draws = self._nondet.draws
        if self._redactor is not None:
            self._tape.content_redacted = self._redactor.content_redacted

        # Extract the original httpx transport to use as the recording inner transport.
        # This preserves ScriptedFakeLLM in tests and HTTPTransport in production.
        orig_inner = self._orig_client._client._transport
        effective_matcher = (
            self._redactor.matcher(self._matcher) if self._redactor else self._matcher
        )
        transport = TraceforkTransport(
            "record", self._tape, orig_inner, matcher=effective_matcher, redactor=self._redactor
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
        return self

    def __exit__(self, *args: object) -> None:
        _uuid_module.uuid4 = self._orig_uuid4  # type: ignore[assignment]


class AsyncRecorder:
    """Async context manager that records an AsyncAnthropic client's I/O.

    See ``Recorder`` for the ``matcher`` / ``redactor`` contract — identical here.
    """

    def __init__(
        self,
        client: anthropic.AsyncAnthropic,
        agent_name: str = "",
        *,
        matcher: RequestMatcher | None = None,
        redactor: Redactor | None = None,
    ) -> None:
        self._orig_client = client
        self._agent_name = agent_name
        self._matcher = matcher
        self._redactor = redactor
        self._nondet: RecordingNondet | None = None
        self._tape: Tape | None = None
        self._wrapped_client: anthropic.AsyncAnthropic | None = None
        self._orig_uuid4: Callable[[], _uuid_module.UUID] | None = None

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
        self._nondet = RecordingNondet()
        self._tape = Tape(agent_name=self._agent_name)
        self._tape.draws = self._nondet.draws
        if self._redactor is not None:
            self._tape.content_redacted = self._redactor.content_redacted

        orig_inner = self._orig_client._client._transport
        effective_matcher = (
            self._redactor.matcher(self._matcher) if self._redactor else self._matcher
        )
        transport = AsyncTraceforkTransport(
            "record", self._tape, orig_inner, matcher=effective_matcher, redactor=self._redactor
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
        return self

    async def __aexit__(self, *args: object) -> None:
        _uuid_module.uuid4 = self._orig_uuid4  # type: ignore[assignment]
