"""Deterministic asyncio-concurrency replay.

asyncio is deterministic except for the ORDER concurrent in-flight requests
resolve. `AsyncTraceforkTransport` records that completion order (and logs
fully-overlapping fan-out batches to `tape.async_batches`) and, on replay,
correlates each request to its recorded exchange by fingerprint and releases
responses in the recorded completion order — so a `gather`/`TaskGroup` agent
replays bit-exact. Sequential async stays byte-identical to the sync transport.
`chaos_release_order` replays a different, physically-possible interleaving.

All offline, $0, no network — exact-equality assertions (no float dust).
"""

from __future__ import annotations

import asyncio

import anthropic
import httpx
import pytest

from tests.fakes import make_text_response
from tracefork.constants import SONNET
from tracefork.nondet import DivergenceError
from tracefork.recorder import AsyncRecorder
from tracefork.tape import Tape
from tracefork.transport import (
    AsyncTraceforkTransport,
    TraceforkTransport,
    chaos_release_order,
)

# ── helpers ──────────────────────────────────────────────────────────────────


def _req(body: bytes) -> httpx.Request:
    return httpx.Request("POST", "https://api.anthropic.com/v1/messages", content=body)


class _SyncInner(httpx.BaseTransport):
    def __init__(self, responses: dict[bytes, bytes]) -> None:
        self._responses = responses

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            content=self._responses[request.content],
        )


class _DelayedAsyncInner(httpx.AsyncBaseTransport):
    """Completes each request after a fixed delay, so completion order is driven
    by the delays rather than the send order — the fan-out nondeterminism."""

    def __init__(self, delays: dict[bytes, float], responses: dict[bytes, bytes]) -> None:
        self._delays = delays
        self._responses = responses

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(self._delays[request.content])
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            content=self._responses[request.content],
        )


# Delays chosen so the completion order (gamma, beta, alpha) is the REVERSE of a
# natural send order (alpha, beta, gamma) with wide (~20ms) gaps → robust, not flaky.
_DELAYS = {b"alpha": 0.05, b"beta": 0.03, b"gamma": 0.01}
_RESPONSES = {b"alpha": b"RESP-A", b"beta": b"RESP-B", b"gamma": b"RESP-G"}


async def _record_gather_tape() -> Tape:
    tape = Tape()
    t = AsyncTraceforkTransport("record", tape, _DelayedAsyncInner(_DELAYS, _RESPONSES))

    async def send(body: bytes) -> bytes:
        resp = await t.handle_async_request(_req(body))
        return await resp.aread()

    await asyncio.gather(send(b"alpha"), send(b"beta"), send(b"gamma"))
    return tape


# ── record: completion order + batch logging ─────────────────────────────────


async def test_gather_records_completion_order_and_batch():
    tape = await _record_gather_tape()
    assert len(tape.exchanges) == 3
    # Exchanges are appended at completion, so the list IS the completion order,
    # and it differs from the send order (alpha, beta, gamma).
    recorded_bodies = [req for req, _ in tape.exchanges]
    assert recorded_bodies != [b"alpha", b"beta", b"gamma"]
    assert recorded_bodies == [b"gamma", b"beta", b"alpha"]
    # A single fully-overlapping concurrent batch, indices in completion order.
    assert tape.async_batches == [[0, 1, 2]]


# ── replay: bit-exact under a deliberately-reordered live schedule ────────────


async def test_gather_replays_bit_exact_in_recorded_order():
    tape = await _record_gather_tape()
    recorded_bodies = [req for req, _ in tape.exchanges]

    rt = AsyncTraceforkTransport("replay", tape)
    released: list[bytes] = []

    async def recv(body: bytes) -> bytes:
        resp = await rt.handle_async_request(_req(body))
        released.append(body)  # recorded at release time, before any further await
        return await resp.aread()

    # Live ARRIVAL order (alpha, beta, gamma) differs from the recorded
    # completion order — the whole point of correlation + ordered release.
    results = await asyncio.gather(recv(b"alpha"), recv(b"beta"), recv(b"gamma"))

    # Each request got ITS OWN recorded response (fingerprint correlation),
    # regardless of the arrival order — gather returns in argument order.
    assert results == [b"RESP-A", b"RESP-B", b"RESP-G"]
    # Responses were RELEASED in the recorded completion order, not arrival order.
    assert released == recorded_bodies
    assert rt.fully_consumed()


# ── sequential async stays byte-identical to the sync transport ──────────────


async def test_sequential_async_is_byte_identical_to_sync():
    pairs = {b"one": b"R1", b"two": b"R2"}

    tape_async = Tape()
    ta = AsyncTraceforkTransport(
        "record", tape_async, _DelayedAsyncInner({b"one": 0.0, b"two": 0.0}, pairs)
    )
    # One await at a time — each fully resolves before the next is sent.
    await (await ta.handle_async_request(_req(b"one"))).aread()
    await (await ta.handle_async_request(_req(b"two"))).aread()

    tape_sync = Tape()
    ts = TraceforkTransport("record", tape_sync, _SyncInner(pairs))
    ts.handle_request(_req(b"one")).read()
    ts.handle_request(_req(b"two")).read()

    # No genuine concurrency → no batch logged, and the recorded content +
    # digest are byte-for-byte the sync recording.
    assert tape_async.async_batches == []
    assert tape_async.exchanges == tape_sync.exchanges
    assert tape_async.digest() == tape_sync.digest()

    # And it still replays positionally-equivalent + bit-exact.
    rt = AsyncTraceforkTransport("replay", tape_async)
    assert (await (await rt.handle_async_request(_req(b"one"))).aread()) == b"R1"
    assert (await (await rt.handle_async_request(_req(b"two"))).aread()) == b"R2"
    assert rt.fully_consumed()


# ── divergence is still detected ──────────────────────────────────────────────


async def test_divergence_detected_on_unrecorded_request():
    tape = Tape()
    tape.append_exchange(b"recorded", b"R")
    rt = AsyncTraceforkTransport("replay", tape)
    with pytest.raises(DivergenceError):
        await rt.handle_async_request(_req(b"not-recorded"))


async def test_divergence_detected_within_a_gather():
    tape = await _record_gather_tape()
    rt = AsyncTraceforkTransport("replay", tape)

    async def recv(body: bytes) -> bytes:
        resp = await rt.handle_async_request(_req(body))
        return await resp.aread()

    # gamma+beta match; the third request diverges (never recorded).
    with pytest.raises(DivergenceError):
        await asyncio.gather(recv(b"gamma"), recv(b"beta"), recv(b"DIVERGED"))


# ── async_batches persists through both serialization surfaces ───────────────


def test_async_batches_roundtrips_and_is_not_hashed(tmp_path):
    tape = Tape(agent_name="fanout")
    tape.append_exchange(b"a", b"ra")
    tape.append_exchange(b"b", b"rb")
    tape.append_exchange(b"c", b"rc")
    tape.async_batches = [[0, 1, 2]]

    # to_bytes / from_bytes
    restored = Tape.from_bytes(tape.to_bytes())
    assert restored.async_batches == [[0, 1, 2]]
    assert restored.exchanges == tape.exchanges

    # save / load (SQLite)
    path = str(tmp_path / "fanout.tape.sqlite")
    tape.save(path)
    loaded = Tape.load(path)
    assert loaded.async_batches == [[0, 1, 2]]

    # The batch log is metadata, NOT content: a tape with an identical exchange
    # log but no batches has the SAME digest (so existing digests are unchanged).
    baseline = Tape(agent_name="fanout")
    baseline.append_exchange(b"a", b"ra")
    baseline.append_exchange(b"b", b"rb")
    baseline.append_exchange(b"c", b"rc")
    assert tape.digest() == baseline.digest()


# ── chaos-mode scheduling ─────────────────────────────────────────────────────


def test_chaos_release_order_permutes_within_batch_only():
    tape = Tape()
    for i in range(4):
        tape.append_exchange(f"r{i}".encode(), f"resp{i}".encode())
    # Steps 0,3 sequential; steps 1,2 a concurrent batch.
    tape.async_batches = [[1, 2]]

    order = chaos_release_order(tape, seed=7)
    assert sorted(order) == [0, 1, 2, 3]  # a valid permutation of range(n)
    assert order[0] == 0 and order[3] == 3  # sequential slots are pinned
    assert {order[1], order[2]} == {1, 2}  # batch reordered within its own slots
    # Seeded → reproducible.
    assert chaos_release_order(tape, seed=7) == order


def test_chaos_release_order_identity_for_sequential_tape():
    tape = Tape()
    tape.append_exchange(b"a", b"ra")
    tape.append_exchange(b"b", b"rb")
    assert chaos_release_order(tape, seed=1) == [0, 1]


async def test_chaos_replay_reorders_completion_but_serves_same_responses():
    tape = await _record_gather_tape()  # exchanges [gamma, beta, alpha], batch [[0,1,2]]
    chaos = [2, 1, 0]  # release in reverse of the recorded completion order

    rt = AsyncTraceforkTransport("replay", tape, release_order=chaos)
    released: list[bytes] = []

    async def recv(body: bytes) -> bytes:
        resp = await rt.handle_async_request(_req(body))
        released.append(body)
        return await resp.aread()

    results = await asyncio.gather(recv(b"alpha"), recv(b"beta"), recv(b"gamma"))

    # Same responses (correlated by fingerprint), but released in the chaos order.
    assert results == [b"RESP-A", b"RESP-B", b"RESP-G"]
    assert released == [tape.exchanges[i][0] for i in chaos]
    assert rt.fully_consumed()


def test_release_order_must_be_a_permutation():
    tape = Tape()
    tape.append_exchange(b"a", b"ra")
    tape.append_exchange(b"b", b"rb")
    rt = AsyncTraceforkTransport("replay", tape, release_order=[0, 0])

    async def go() -> None:
        await rt.handle_async_request(_req(b"a"))

    with pytest.raises(ValueError, match="permutation"):
        asyncio.run(go())


# ── end-to-end through the real AsyncAnthropic SDK ───────────────────────────


class _DelayedAnthropicInner(httpx.AsyncBaseTransport):
    """Serves valid Anthropic wire responses, keyed on a marker in the request
    body, with per-marker delays so 'beta' completes before 'alpha'."""

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if b"alpha" in request.content:
            await asyncio.sleep(0.04)
            body = make_text_response("respA")
        else:
            await asyncio.sleep(0.01)
            body = make_text_response("respB")
        return httpx.Response(200, headers={"content-type": "application/json"}, content=body)


async def _gather_agent(client: anthropic.AsyncAnthropic) -> list[str]:
    async def call(prompt: str) -> str:
        resp = await client.messages.create(
            model=SONNET, max_tokens=100, messages=[{"role": "user", "content": prompt}]
        )
        return resp.content[0].text

    return await asyncio.gather(call("alpha"), call("beta"))


async def test_async_recorder_gather_replays_bit_exact_via_sdk():
    rec_client = anthropic.AsyncAnthropic(
        api_key="sk-ant-fake",
        http_client=httpx.AsyncClient(transport=_DelayedAnthropicInner()),
        max_retries=0,
    )
    async with AsyncRecorder(rec_client, agent_name="gather") as rec:
        recorded = await _gather_agent(rec.client)
        tape = rec.tape

    # gather returns in argument order regardless of completion order.
    assert recorded == ["respA", "respB"]
    assert len(tape.exchanges) == 2
    assert tape.async_batches == [[0, 1]]  # a fully-overlapping fan-out of 2

    replay_transport = AsyncTraceforkTransport("replay", tape)
    replay_client = anthropic.AsyncAnthropic(
        api_key="sk-ant-replay",
        http_client=httpx.AsyncClient(transport=replay_transport),
        max_retries=0,
    )
    replayed = await _gather_agent(replay_client)

    assert replayed == recorded
    assert replay_transport.fully_consumed()
