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


# --- new_episodes transport mode ---


def test_new_episodes_replays_recorded_prefix_like_strict_replay():
    """The recorded prefix is served under the EXACT same assert logic as
    plain "replay" -- the inner transport is never consulted for it."""
    tape = Tape()
    tape.append_exchange(b"req-1", b"resp-1")
    tape.append_exchange(b"req-2", b"resp-2")
    inner = _SyncInner([])  # never touched
    t = TraceforkTransport("new_episodes", tape, inner)
    assert t.handle_request(_make_request(b"req-1")).read() == b"resp-1"
    assert t.handle_request(_make_request(b"req-2")).read() == b"resp-2"
    assert t.matched == 2
    assert t.new_episodes_recorded == 0


def test_new_episodes_prefix_divergence_still_raises():
    """The recorded-prefix assert is unmodified: a request that diverges
    from the tape inside the recorded prefix is still a hard error, exactly
    like plain "replay"."""
    tape = Tape()
    tape.append_exchange(b"expected", b"resp")
    inner = _SyncInner([])
    t = TraceforkTransport("new_episodes", tape, inner)
    with pytest.raises(DivergenceError, match="diverged"):
        t.handle_request(_make_request(b"different"))


def test_new_episodes_records_trailing_unrecorded_request_instead_of_erroring():
    tape = Tape()
    tape.append_exchange(b"req-1", b"resp-1")
    inner = _SyncInner([b"resp-2"])
    t = TraceforkTransport("new_episodes", tape, inner)
    t.handle_request(_make_request(b"req-1"))  # recorded prefix
    r2 = t.handle_request(_make_request(b"req-2"))  # beyond the prefix
    assert r2.read() == b"resp-2"
    assert tape.exchanges[-1] == (b"req-2", b"resp-2")
    assert t.new_episodes_recorded == 1
    assert t.matched == 1  # only the prefix request counts as "matched"


def test_new_episodes_requires_inner_transport():
    with pytest.raises(ValueError, match="new_episodes mode requires an inner transport"):
        TraceforkTransport("new_episodes", Tape(), inner=None)


def test_new_episodes_recorded_exchange_updates_digest_consistently_with_record_mode():
    """A trailing new_episodes exchange goes through the SAME
    `tape.append_exchange`/sha256 hash-chain path as plain "record" mode --
    the final tape's `digest()` is identical regardless of which mode
    produced it."""
    tape = Tape()
    tape.append_exchange(b"req-1", b"resp-1")
    inner = _SyncInner([b"resp-2"])
    t = TraceforkTransport("new_episodes", tape, inner)
    t.handle_request(_make_request(b"req-1"))
    t.handle_request(_make_request(b"req-2"))

    record_tape = Tape()
    record_inner = _SyncInner([b"resp-1", b"resp-2"])
    rt = TraceforkTransport("record", record_tape, record_inner)
    rt.handle_request(_make_request(b"req-1"))
    rt.handle_request(_make_request(b"req-2"))

    assert tape.exchanges == record_tape.exchanges
    assert tape.digest() == record_tape.digest()


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
