"""Synthetic provider transports — offline, deterministic stand-ins for the API.

These serve opaque provider wire-format bytes (built by `tracefork.wire`, which
delegates to the registered provider adapter) so the genuine SDK parses them, but
never touch the network. The transports themselves are provider-agnostic — they
replay whatever response bytes they are handed. They are production components:
the self-validation suite (`tracefork validate`) and the test suite both drive
the recorder/fork/blame machinery through them at $0.

  - `ScriptedFakeLLM`     — returns a fixed list of responses in order.
  - `AsyncScriptedFakeLLM`— async variant.
  - `AsyncStreamingFakeLLM` — async variant that streams each scripted response
    as multiple chunks (SSE-shaped by default) instead of one buffered body —
    `proxy.py`'s tests use it as an injected fake upstream to prove record mode
    tees a streaming response while forwarding it, not only after fully
    buffering it.
  - `FaultAwareFakeLLM`   — returns a *failure* script when a fault marker
    appears in the request body, else a *normal* script; this is how an
    injected fault propagates into a flipped outcome during validation.
  - `FakeAWSPreparedRequest`/`FakeEventEmitter`/`ScriptedBedrockSender` —
    the botocore-shaped equivalents `bedrock_transport.py`'s offline tests
    drive: a duck-typed prepared request, a `HierarchicalEmitter`-shaped
    `.register()`/`.emit()` fake, and a scripted `sender` callable. None of
    these import botocore — see `bedrock_transport.py`'s module docstring for
    why the real botocore contract can be mirrored without the dependency.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

import httpx


class ScriptedFakeLLM(httpx.BaseTransport):
    """Returns scripted Anthropic wire-format responses in sequence.

    Pass a list of response bytes (from make_text_response / make_tool_use_response).
    Raises ScriptExhausted if more requests arrive than the script has responses.
    """

    class ScriptExhausted(RuntimeError):
        pass

    def __init__(self, responses: list[bytes]) -> None:
        self._responses = list(responses)
        self._i = 0
        self.requests_received: list[bytes] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.requests_received.append(request.content)
        if self._i >= len(self._responses):
            raise self.ScriptExhausted(
                f"ScriptedFakeLLM exhausted after {len(self._responses)} responses"
            )
        resp = self._responses[self._i]
        self._i += 1
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            content=resp,
        )


class AsyncScriptedFakeLLM(httpx.AsyncBaseTransport):
    """Async variant of ScriptedFakeLLM."""

    class ScriptExhausted(RuntimeError):
        pass

    def __init__(self, responses: list[bytes]) -> None:
        self._responses = list(responses)
        self._i = 0
        self.requests_received: list[bytes] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests_received.append(request.content)
        if self._i >= len(self._responses):
            raise self.ScriptExhausted(
                f"AsyncScriptedFakeLLM exhausted after {len(self._responses)} responses"
            )
        resp = self._responses[self._i]
        self._i += 1
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            content=resp,
        )


class AsyncStreamingFakeLLM(httpx.AsyncBaseTransport):
    """Streams a scripted response as multiple chunks per request.

    `chunked_responses` is a list of chunk-lists, one entry per request; each
    entry's bytes are yielded in sequence via an async generator, so a caller
    reading the response through `.aiter_bytes()` observes it arrive
    incrementally rather than as one buffered blob — the shape a real SSE
    upstream has. Raises `ScriptExhausted` if more requests arrive than the
    script has entries.
    """

    class ScriptExhausted(RuntimeError):
        pass

    def __init__(
        self,
        chunked_responses: list[list[bytes]],
        *,
        content_type: str = "text/event-stream",
    ) -> None:
        self._responses = list(chunked_responses)
        self._content_type = content_type
        self._i = 0
        self.requests_received: list[bytes] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests_received.append(request.content)
        if self._i >= len(self._responses):
            raise self.ScriptExhausted(
                f"AsyncStreamingFakeLLM exhausted after {len(self._responses)} response(s)"
            )
        chunks = self._responses[self._i]
        self._i += 1

        async def _stream() -> AsyncIterator[bytes]:
            for chunk in chunks:
                yield chunk

        return httpx.Response(
            200,
            headers={"content-type": self._content_type},
            content=_stream(),
        )


class FaultAwareFakeLLM(httpx.BaseTransport):
    """Returns different response scripts based on a fault marker in the request.

    If `fault_marker` (bytes) appears anywhere in the request body, serve
    `fault_responses`; otherwise serve `normal_responses`. Each script cycles
    independently. This lets an injected fault — which the agent echoes into a
    later request — deterministically flip the run's outcome.
    """

    def __init__(
        self,
        normal_responses: list[bytes],
        fault_responses: list[bytes],
        fault_marker: bytes,
    ) -> None:
        self._normal = list(normal_responses)
        self._fault = list(fault_responses)
        self._marker = fault_marker
        self._normal_i = 0
        self._fault_i = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if self._marker in request.content:
            resp = self._fault[self._fault_i % len(self._fault)]
            self._fault_i += 1
        else:
            resp = self._normal[self._normal_i % len(self._normal)]
            self._normal_i += 1
        return httpx.Response(200, headers={"content-type": "application/json"}, content=resp)


# ── Bedrock (botocore-shaped) fakes ─────────────────────────────────────────
#
# botocore never touches httpx (see bedrock_transport.py's module docstring),
# so the fakes above don't apply to it. These mirror the two botocore
# interfaces bedrock_transport.py depends on -- a prepared request
# (`.method`/`.url`/`.headers`/`.body`) and an event emitter
# (`.register()`/`.emit()`) -- closely enough that bedrock_transport.py's
# `BedrockTransport` works identically against these or the real botocore
# objects, entirely offline and without importing botocore.


@dataclass
class FakeAWSPreparedRequest:
    """Duck-typed stand-in for botocore's `AWSPreparedRequest`: `.method`,
    `.url`, `.headers`, `.body`. `bedrock_transport.py`'s
    `prepared_request_to_httpx()` only ever reads these four attributes."""

    method: str
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes = b""


class FakeEventEmitter:
    """Duck-typed stand-in for botocore's `HierarchicalEmitter`
    (`client.meta.events`): `.register(event_name, handler)` +
    `.emit(event_name, **kwargs) -> [(handler, response), ...]`. The list of
    `(handler, response)` pairs mirrors botocore's real `.emit()` return
    shape exactly (see `first_non_none_response` below and
    `bedrock_transport.py`'s module docstring for the citation)."""

    def __init__(self) -> None:
        self._handlers: dict[str, list[Callable[..., Any]]] = {}

    def register(self, event_name: str, handler: Callable[..., Any]) -> None:
        self._handlers.setdefault(event_name, []).append(handler)

    def emit(self, event_name: str, **kwargs: Any) -> list[tuple[Callable[..., Any], Any]]:
        return [(h, h(**kwargs)) for h in self._handlers.get(event_name, [])]


def first_non_none_response(responses: list[tuple[Any, Any]], default: Any = None) -> Any:
    """Local re-implementation of `botocore.hooks.first_non_none_response` —
    the exact function botocore's `endpoint.py` uses to interpret
    `.emit()`'s `[(handler, response), ...]` return shape and decide whether
    to skip the real network send. Reimplemented here (rather than imported)
    so callers never need botocore installed to interpret
    `FakeEventEmitter.emit()`'s output; it also works unchanged against a
    real `HierarchicalEmitter.emit()` result."""
    for _handler, response in responses:
        if response is not None:
            return response
    return default


class ScriptedBedrockSender:
    """Offline `sender` for `BedrockTransport("record", ...)`: cycles a fixed
    list of response bytes, analogous to `ScriptedFakeLLM` for the httpx seam.
    Raises `ScriptExhausted` if more requests arrive than the script has
    responses."""

    class ScriptExhausted(RuntimeError):
        pass

    def __init__(self, responses: list[bytes], *, status_code: int = 200) -> None:
        self._responses = list(responses)
        self._status_code = status_code
        self._i = 0
        self.requests_received: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests_received.append(request)
        if self._i >= len(self._responses):
            raise self.ScriptExhausted(
                f"ScriptedBedrockSender exhausted after {len(self._responses)} response(s)"
            )
        body = self._responses[self._i]
        self._i += 1
        return httpx.Response(
            self._status_code, headers={"content-type": "application/json"}, content=body
        )
