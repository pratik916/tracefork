"""Shared fake Anthropic transports for offline CI testing.

All tests in this project that need Anthropic responses import from here.
Never create fake Anthropic JSON outside this file.
"""
from __future__ import annotations

import httpx

# Wire-format builders live in the package so production code (blame engine,
# CLI, validation suite) never imports from the test tree. Re-exported here
# because the test fakes are their primary consumer.
from tracefork.wire import make_text_response, make_tool_use_response

__all__ = [
    "make_text_response",
    "make_tool_use_response",
    "ScriptedFakeLLM",
    "AsyncScriptedFakeLLM",
    "FaultAwareFakeLLM",
]


class ScriptedFakeLLM(httpx.BaseTransport):
    """Returns scripted Anthropic wire-format responses in sequence.

    Pass a list of response bytes (from make_text_response / make_tool_use_response).
    Raises ScriptExhausted if the agent makes more requests than the script has responses.
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

    Used by the Plan F validation CI suite: if `fault_marker` appears anywhere
    in the request body bytes, serve `fault_responses`; otherwise serve `normal_responses`.
    Both scripts are advanced independently and reset on each new run.
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
