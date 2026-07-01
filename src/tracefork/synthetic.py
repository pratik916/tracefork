"""Synthetic provider transports — offline, deterministic stand-ins for the API.

These serve opaque provider wire-format bytes (built by `tracefork.wire`, which
delegates to the registered provider adapter) so the genuine SDK parses them, but
never touch the network. The transports themselves are provider-agnostic — they
replay whatever response bytes they are handed. They are production components:
the self-validation suite (`tracefork validate`) and the test suite both drive
the recorder/fork/blame machinery through them at $0.

  - `ScriptedFakeLLM`     — returns a fixed list of responses in order.
  - `AsyncScriptedFakeLLM`— async variant.
  - `FaultAwareFakeLLM`   — returns a *failure* script when a fault marker
    appears in the request body, else a *normal* script; this is how an
    injected fault propagates into a flipped outcome during validation.
"""

from __future__ import annotations

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
