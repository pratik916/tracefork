"""Transport tests — sync and async record/replay/divergence."""

import httpx
import pytest

from tracefork.nondet import DivergenceError
from tracefork.tape import Tape
from tracefork.transport import AsyncTraceforkTransport, TraceforkTransport

# --- helpers ---


def _fake_inner_response(content: bytes) -> httpx.Response:
    return httpx.Response(200, headers={"content-type": "application/json"}, content=content)


class _SyncInner(httpx.BaseTransport):
    def __init__(self, responses: list[bytes]):
        self._responses = iter(responses)

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        return _fake_inner_response(next(self._responses))


class _AsyncInner(httpx.AsyncBaseTransport):
    def __init__(self, responses: list[bytes]):
        self._responses = iter(responses)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return _fake_inner_response(next(self._responses))


def _make_request(body: bytes) -> httpx.Request:
    return httpx.Request("POST", "https://api.anthropic.com/v1/messages", content=body)


# --- sync transport ---


def test_sync_record_captures_exchange():
    tape = Tape()
    inner = _SyncInner([b"resp-1", b"resp-2"])
    t = TraceforkTransport("record", tape, inner)
    r1 = t.handle_request(_make_request(b"req-1"))
    r2 = t.handle_request(_make_request(b"req-2"))
    assert r1.read() == b"resp-1"
    assert r2.read() == b"resp-2"
    assert len(tape.exchanges) == 2
    assert tape.exchanges[0] == (b"req-1", b"resp-1")
    assert tape.exchanges[1] == (b"req-2", b"resp-2")


def test_sync_replay_serves_recorded_bytes():
    tape = Tape()
    tape.append_exchange(b"req-1", b"resp-1")
    tape.append_exchange(b"req-2", b"resp-2")
    t = TraceforkTransport("replay", tape)
    assert t.handle_request(_make_request(b"req-1")).read() == b"resp-1"
    assert t.handle_request(_make_request(b"req-2")).read() == b"resp-2"
    assert t.matched == 2
    assert t.fully_consumed()


def test_sync_replay_raises_on_request_mismatch():
    tape = Tape()
    tape.append_exchange(b"expected", b"resp")
    t = TraceforkTransport("replay", tape)
    with pytest.raises(DivergenceError, match="diverged"):
        t.handle_request(_make_request(b"different"))


def test_sync_replay_raises_on_extra_request():
    tape = Tape()
    tape.append_exchange(b"req", b"resp")
    t = TraceforkTransport("replay", tape)
    t.handle_request(_make_request(b"req"))
    with pytest.raises(DivergenceError, match="unrecorded"):
        t.handle_request(_make_request(b"req"))


def test_sync_record_requires_inner():
    with pytest.raises(ValueError, match="inner"):
        TraceforkTransport("record", Tape(), inner=None)


# --- async transport ---


@pytest.mark.asyncio
async def test_async_record_captures_exchange():
    tape = Tape()
    inner = _AsyncInner([b"resp-1"])
    t = AsyncTraceforkTransport("record", tape, inner)
    r = await t.handle_async_request(_make_request(b"req-1"))
    assert await r.aread() == b"resp-1"
    assert tape.exchanges[0] == (b"req-1", b"resp-1")


@pytest.mark.asyncio
async def test_async_replay_serves_recorded_bytes():
    tape = Tape()
    tape.append_exchange(b"req-1", b"resp-1")
    t = AsyncTraceforkTransport("replay", tape)
    r = await t.handle_async_request(_make_request(b"req-1"))
    assert await r.aread() == b"resp-1"
    assert t.fully_consumed()


@pytest.mark.asyncio
async def test_async_replay_raises_on_mismatch():
    tape = Tape()
    tape.append_exchange(b"expected", b"resp")
    t = AsyncTraceforkTransport("replay", tape)
    with pytest.raises(DivergenceError):
        await t.handle_async_request(_make_request(b"different"))
