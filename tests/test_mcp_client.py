"""Optional MCP adapter tests — offline, synthetic frames, no real server.

`mcp` is an optional dependency, so these exercise the replay path (which needs
no `mcp` install) and the import guard, never a live session or subprocess.
"""

import pytest

from tracefork.mcp_client import RecordingMCPSession, mcp_available, require_mcp
from tracefork.nondet import DivergenceError
from tracefork.tape import Tape
from tracefork.tools import make_result_frame, make_tool_call_frame


def _seed_replay_tape():
    tape = Tape()
    tape.append_tool_exchange(
        make_tool_call_frame(None, "get_weather", {"city": "NYC"}),
        make_result_frame(None, {"content": [{"type": "text", "text": "72F"}]}),
    )
    return tape


def test_require_mcp_matches_availability():
    if mcp_available():
        require_mcp()  # no raise when installed
    else:
        with pytest.raises(ImportError, match="mcp"):
            require_mcp()


def test_record_mode_without_session_is_guarded():
    tape = Tape()
    with pytest.raises((ImportError, ValueError)):
        # ImportError if mcp missing (guard fires first), else ValueError for the
        # absent live session — either way, record mode cannot proceed.
        RecordingMCPSession(tape, "record", session=None)


async def test_replay_session_serves_recorded_tool():
    session = RecordingMCPSession(_seed_replay_tape(), "replay")
    assert session.mode == "replay"
    result = await session.call_tool("get_weather", {"city": "NYC"})
    assert result == {"content": [{"type": "text", "text": "72F"}]}


async def test_replay_session_divergence_on_mismatch():
    session = RecordingMCPSession(_seed_replay_tape(), "replay")
    with pytest.raises(DivergenceError):
        await session.call_tool("get_weather", {"city": "Paris"})


async def test_replay_session_divergence_on_extra_call():
    session = RecordingMCPSession(_seed_replay_tape(), "replay")
    await session.call_tool("get_weather", {"city": "NYC"})
    with pytest.raises(DivergenceError, match="unrecorded"):
        await session.call_tool("get_weather", {"city": "NYC"})
