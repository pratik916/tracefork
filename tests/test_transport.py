"""Transport tests — sync and async record/replay/divergence."""

import asyncio

import httpx2
import pytest

from tracefork.nondet import DivergenceError
from tracefork.tape import Tape
from tracefork.transport import AsyncTraceforkTransport, TraceforkTransport

# --- helpers ---


def _fake_inner_response(content: bytes) -> httpx2.Response:
    return httpx2.Response(200, headers={"content-type": "application/json"}, content=content)


def _fake_inner_response_with_status(
    status: int, content_type: str, content: bytes
) -> httpx2.Response:
    return httpx2.Response(status, headers={"content-type": content_type}, content=content)


class _SyncInner(httpx2.BaseTransport):
    def __init__(self, responses: list[bytes]):
        self._responses = iter(responses)

    def handle_request(self, request: httpx2.Request) -> httpx2.Response:
        return _fake_inner_response(next(self._responses))


class _SyncInnerStatus(httpx2.BaseTransport):
    """Like `_SyncInner`, but each response carries an explicit
    (status, content_type, body) triple -- for exercising non-200/non-json
    record/replay fidelity."""

    def __init__(self, responses: list[tuple[int, str, bytes]]):
        self._responses = iter(responses)

    def handle_request(self, request: httpx2.Request) -> httpx2.Response:
        status, content_type, content = next(self._responses)
        return _fake_inner_response_with_status(status, content_type, content)


class _AsyncInner(httpx2.AsyncBaseTransport):
    def __init__(self, responses: list[bytes]):
        self._responses = iter(responses)

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        return _fake_inner_response(next(self._responses))


class _AsyncInnerStatus(httpx2.AsyncBaseTransport):
    def __init__(self, responses: list[tuple[int, str, bytes]]):
        self._responses = iter(responses)

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        status, content_type, content = next(self._responses)
        return _fake_inner_response_with_status(status, content_type, content)


def _make_request(body: bytes) -> httpx2.Request:
    return httpx2.Request("POST", "https://api.anthropic.com/v1/messages", content=body)


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


# --- non-200 status / content-type record+replay fidelity (tracefork-sis.35) ---


def test_sync_record_captures_status_and_content_type():
    tape = Tape()
    inner = _SyncInnerStatus([(429, "text/plain", b"rate limited")])
    t = TraceforkTransport("record", tape, inner)
    r = t.handle_request(_make_request(b"req-1"))
    assert r.status_code == 429
    assert r.headers["content-type"] == "text/plain"
    assert r.read() == b"rate limited"
    assert tape.response_status == [429]
    assert tape.response_content_type == ["text/plain"]


class _SyncInnerNoContentTypeHeader(httpx2.BaseTransport):
    """An inner transport whose response carries NO content-type header at
    all (not even an empty one) -- exercises the pre-existing
    `.get("content-type", "application/json")` fallback, unchanged by this
    item, now feeding `tape.response_content_type` instead of only the
    returned live response."""

    def handle_request(self, request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, content=b"{}")


def test_sync_record_still_defaults_to_json_when_inner_omits_content_type_header():
    tape = Tape()
    t = TraceforkTransport("record", tape, _SyncInnerNoContentTypeHeader())
    t.handle_request(_make_request(b"req-1"))
    assert tape.response_status == [200]
    assert tape.response_content_type == ["application/json"]


def test_sync_replay_serves_recorded_status_and_content_type():
    tape = Tape()
    tape.append_exchange(
        b"req-1", b"rate limited", response_status=429, response_content_type="text/plain"
    )
    t = TraceforkTransport("replay", tape)
    r = t.handle_request(_make_request(b"req-1"))
    assert r.status_code == 429
    assert r.headers["content-type"] == "text/plain"
    assert r.read() == b"rate limited"


def test_sync_replay_defaults_to_200_json_when_tape_predates_response_meta():
    """A `Tape` built with `exchanges=` directly (never through
    `append_exchange`) has no `response_status`/`response_content_type` at
    all -- the same shape a pre-v7 upcast or a hand-built probe tape
    (`tournament.py`) produces. Replay must still serve (200, application/json),
    never raise IndexError."""
    tape = Tape(exchanges=[(b"req-1", b"resp-1")])
    t = TraceforkTransport("replay", tape)
    r = t.handle_request(_make_request(b"req-1"))
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/json"
    assert r.read() == b"resp-1"


def test_new_episodes_prefix_replay_serves_recorded_status_and_content_type():
    tape = Tape()
    tape.append_exchange(
        b"req-1", b"not found", response_status=404, response_content_type="text/plain"
    )
    inner = _SyncInner([])  # never touched -- recorded prefix
    t = TraceforkTransport("new_episodes", tape, inner)
    r = t.handle_request(_make_request(b"req-1"))
    assert r.status_code == 404
    assert r.headers["content-type"] == "text/plain"


def test_new_episodes_trailing_record_captures_status_and_content_type():
    tape = Tape()
    tape.append_exchange(b"req-1", b"resp-1")
    inner = _SyncInnerStatus([(503, "text/html", b"<h1>down</h1>")])
    t = TraceforkTransport("new_episodes", tape, inner)
    t.handle_request(_make_request(b"req-1"))  # recorded prefix
    r2 = t.handle_request(_make_request(b"req-2"))  # trailing episode
    assert r2.status_code == 503
    assert r2.headers["content-type"] == "text/html"
    assert tape.response_status[-1] == 503
    assert tape.response_content_type[-1] == "text/html"


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


@pytest.mark.asyncio
async def test_async_record_captures_status_and_content_type():
    tape = Tape()
    inner = _AsyncInnerStatus([(429, "text/plain", b"rate limited")])
    t = AsyncTraceforkTransport("record", tape, inner)
    r = await t.handle_async_request(_make_request(b"req-1"))
    assert r.status_code == 429
    assert r.headers["content-type"] == "text/plain"
    assert tape.response_status == [429]
    assert tape.response_content_type == ["text/plain"]


@pytest.mark.asyncio
async def test_async_replay_serves_recorded_status_and_content_type():
    tape = Tape()
    tape.append_exchange(
        b"req-1", b"rate limited", response_status=429, response_content_type="text/plain"
    )
    t = AsyncTraceforkTransport("replay", tape)
    r = await t.handle_async_request(_make_request(b"req-1"))
    assert r.status_code == 429
    assert r.headers["content-type"] == "text/plain"
    assert await r.aread() == b"rate limited"


@pytest.mark.asyncio
async def test_async_replay_defaults_to_200_json_when_tape_predates_response_meta():
    tape = Tape(exchanges=[(b"req-1", b"resp-1")])
    t = AsyncTraceforkTransport("replay", tape)
    r = await t.handle_async_request(_make_request(b"req-1"))
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/json"


# --- ordered-release gate bounded wait (tracefork-sis.38) ---


@pytest.mark.asyncio
async def test_async_replay_ordered_release_gate_raises_divergence_instead_of_hanging():
    """A reordered release schedule where the OTHER exchange the gate is
    waiting on first is never actually requested must raise
    `DivergenceError` within a bounded time, not hang forever -- the sync
    transport's equivalent divergence is already an immediate hard error;
    this closes the async gate's silent-hang gap to match."""
    tape = Tape()
    tape.append_exchange(b"req-0", b"resp-0")
    tape.append_exchange(b"req-1", b"resp-1")
    # release_order says: release exchange #1 before exchange #0. We only
    # ever request exchange #0 below, so exchange #1 -- the one the gate is
    # waiting on first -- is never supplied: a genuine, permanent stall.
    t = AsyncTraceforkTransport("replay", tape, release_order=[1, 0], ordered_release_timeout=0.05)
    with pytest.raises(DivergenceError, match="ordered-release gate timed out"):
        # Outer safety-net timeout: if the fix regressed back to an
        # unbounded `.wait()`, this fails the test loudly instead of
        # hanging the whole suite.
        await asyncio.wait_for(t.handle_async_request(_make_request(b"req-0")), timeout=2.0)


@pytest.mark.asyncio
async def test_async_replay_ordered_release_gate_still_satisfies_legitimate_reordering():
    """The bounded wait must not false-positive on a genuinely-progressing
    concurrent replay: both exchanges requested concurrently, released in
    the recorded (reordered) completion order, well inside the timeout."""
    tape = Tape()
    tape.append_exchange(b"req-0", b"resp-0")
    tape.append_exchange(b"req-1", b"resp-1")
    t = AsyncTraceforkTransport("replay", tape, release_order=[1, 0], ordered_release_timeout=0.2)

    async def get(body: bytes, expected: bytes) -> None:
        r = await t.handle_async_request(_make_request(body))
        assert await r.aread() == expected

    await asyncio.wait_for(
        asyncio.gather(get(b"req-0", b"resp-0"), get(b"req-1", b"resp-1")), timeout=2.0
    )
    assert t.matched == 2
    assert t.fully_consumed()
