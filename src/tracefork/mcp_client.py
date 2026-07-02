"""Optional MCP (Model Context Protocol) record/replay adapter.

MCP is JSON-RPC 2.0 over stdio or Streamable-HTTP — not httpx — so the LLM
transport seam (`transport.py`) never sees tool traffic. This module tees an MCP
client's ``tools/call`` round-trips into the same content-addressed tape as
JSON-RPC frames, through the provider-independent `ToolTransport` in `tools.py`.

The seam sits at the `ClientSession` request/response boundary, so it records
identically whether the session's underlying transport is **stdio** or
**Streamable-HTTP** — both surface the same typed ``call_tool`` round trip.

``mcp`` is an OPTIONAL dependency (the ``mcp`` extra). Nothing here imports it at
module load: `require_mcp()` guards the one entry point that needs a live session
(record mode), so ``import tracefork`` never requires ``mcp`` and the offline test
suite drives this seam with synthetic frames — no real server, no subprocess.
Replay mode needs no live session and no ``mcp`` install at all: a tape recorded
elsewhere replays anywhere.
"""

from __future__ import annotations

from typing import Any

from .redact import Redactor
from .tape import Tape
from .tools import ToolTransport, decode_result, make_result_frame, make_tool_call_frame

MCP_IMPORT_HINT = "MCP support needs the optional 'mcp' extra: pip install 'tracefork[mcp]'"


def mcp_available() -> bool:
    """Whether the optional ``mcp`` package is importable."""
    try:
        import mcp  # noqa: F401
    except ImportError:
        return False
    return True


def require_mcp() -> None:
    """Raise a helpful `ImportError` if the optional ``mcp`` package is missing."""
    if not mcp_available():
        raise ImportError(MCP_IMPORT_HINT)


def _result_payload(result: Any) -> Any:  # pragma: no cover - needs a live mcp result
    """A JSON-serializable payload from an mcp ``CallToolResult`` (pydantic)."""
    dump = getattr(result, "model_dump", None)
    return dump(mode="json") if callable(dump) else result


class RecordingMCPSession:
    """Wrap an mcp ``ClientSession`` so each ``call_tool`` is teed into a tape.

    Record mode calls the wrapped session and stores the request + result frames;
    replay mode serves the recorded result and asserts the request frame matches
    the tape (no live session required). Redaction of tool frames reuses the same
    `Redactor` as the LLM path.
    """

    def __init__(
        self,
        tape: Tape,
        mode: str,
        *,
        session: Any = None,
        redactor: Redactor | None = None,
    ) -> None:
        if mode == "record":
            require_mcp()
            if session is None:
                raise ValueError("record mode requires a live mcp ClientSession")
        self._session = session
        self._transport = ToolTransport(mode, tape, redactor=redactor)

    @property
    def mode(self) -> str:
        return self._transport.mode

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        """Record or replay one MCP ``tools/call``.

        Record mode returns the live mcp result object; replay mode returns the
        recorded JSON-RPC ``result`` payload (a plain dict — no ``mcp`` install
        needed to consume a recorded tape)."""
        request_frame = make_tool_call_frame(None, name, arguments or {})
        if self._transport.mode == "record":
            return await self._record_call(name, arguments, request_frame)
        return decode_result(self._transport.replay_exchange(request_frame))

    async def _record_call(  # pragma: no cover - needs a live mcp ClientSession
        self, name: str, arguments: dict[str, Any] | None, request_frame: bytes
    ) -> Any:
        result = await self._session.call_tool(name, arguments)
        self._transport.record_exchange(
            request_frame, make_result_frame(None, _result_payload(result))
        )
        return result
