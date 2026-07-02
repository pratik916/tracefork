"""Recording/replay httpx transports — sync and async, streaming SSE capable.

Record mode: forward to the inner transport, buffer the full response body
(works for both streaming SSE and non-streaming JSON — httpx buffers both
identically via .read()/.aread()), append to the tape, return the response.

Replay mode: for each request, fingerprint-assert it matches the tape record via
a ``RequestMatcher`` (default: raw ``sha256`` of the request body), then serve the
recorded bytes back. A replay transport has no inner transport; any unrecorded
request is a hard error. The default matched surface is the request body; request
headers (e.g. ``anthropic-beta`` / ``anthropic-version``) are out of scope for the
bit-exactness claim — see the README's determinism-boundary note. An opt-in
``RequestMatcher`` (``matcher=``) can normalize volatile fields before hashing for
providers whose raw bytes are non-deterministic; the record and replay sides MUST
share the same matcher instance or the fingerprints will not line up.

An opt-in ``Redactor`` (``redactor=``, record mode only) additionally scrubs the
response body before it is stored. Request-side redaction is applied further
upstream, inside the matcher itself (``Redactor.matcher()``), so that record and
replay hash the identical redacted form — see ``redact.py``.

Concurrency (async only). asyncio is deterministic *except* for the order in
which concurrent in-flight requests (an ``asyncio.gather``/``TaskGroup`` fan-out)
resolve — an order driven by the very I/O this transport records. So
``AsyncTraceforkTransport`` records that completion order (the ``exchanges`` list
is appended at completion; a genuinely-concurrent, fully-overlapping batch is
additionally logged to ``tape.async_batches`` for chaos scheduling) and, on
replay, **correlates each request to its recorded exchange by fingerprint (not by
positional arrival) and releases responses in the recorded completion order**
through an ordered gate. A strictly-sequential async run (one await at a time)
hits the gate with the condition already satisfied, so it is byte-identical to
before this seam existed; the sync ``TraceforkTransport`` is unaffected and stays
positional. ``chaos_release_order`` derives a seeded, physically-possible
*reordering* of a recorded schedule for race/ordering-bug analysis.
"""

from __future__ import annotations

import asyncio
import random
from collections import deque

import httpx

from .matcher import IDENTITY_MATCHER, RequestMatcher
from .nondet import DivergenceError
from .redact import Redactor
from .tape import Tape


class TraceforkTransport(httpx.BaseTransport):
    """Sync recording/replay transport."""

    def __init__(
        self,
        mode: str,
        tape: Tape,
        inner: httpx.BaseTransport | None = None,
        *,
        matcher: RequestMatcher | None = None,
        redactor: Redactor | None = None,
    ) -> None:
        assert mode in ("record", "replay")
        if mode == "record" and inner is None:
            raise ValueError("record mode requires an inner transport")
        self.mode = mode
        self.tape = tape
        self.inner = inner
        # Default is identity: raw sha256(request.content) — the pre-seam contract.
        self.matcher: RequestMatcher = matcher or IDENTITY_MATCHER
        # Default is None: no redaction — the pre-redaction contract (byte-identical).
        self.redactor = redactor
        self._i = 0
        self.matched = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if self.mode == "record":
            inner_resp = self.inner.handle_request(request)  # type: ignore[union-attr]
            resp_body = inner_resp.read()
            stored_resp = self.redactor.apply_response(resp_body) if self.redactor else resp_body
            self.tape.append_exchange(self.matcher.stored_request(request), stored_resp)
            return httpx.Response(
                inner_resp.status_code,
                headers={
                    "content-type": inner_resp.headers.get("content-type", "application/json")
                },
                content=resp_body,
                request=request,
            )

        # replay
        if self._i >= len(self.tape.exchanges):
            raise DivergenceError(
                f"replay made unrecorded request #{self._i} "
                f"(tape has {len(self.tape.exchanges)} exchanges)"
            )
        rec_req, rec_resp = self.tape.exchange(self._i)
        rec_fp = self.matcher.stored_fingerprint(rec_req)
        live_fp = self.matcher.live_fingerprint(request)
        if rec_fp != live_fp:
            raise DivergenceError(
                f"request #{self._i} diverged from tape "
                f"(recorded {rec_fp[:12]}, replay {live_fp[:12]})"
            )
        self._i += 1
        self.matched += 1
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            content=rec_resp,
            request=request,
        )

    def fully_consumed(self) -> bool:
        return self.mode == "replay" and self._i == len(self.tape.exchanges)


class AsyncTraceforkTransport(httpx.AsyncBaseTransport):
    """Async recording/replay transport with deterministic concurrent-completion
    ordering.

    Record and replay of a strictly-sequential async run are byte-identical to
    the sync ``TraceforkTransport``. The addition is fan-out handling: record
    logs the completion order of genuinely-concurrent batches, and replay serves
    each request its own recorded response (correlated by fingerprint) while
    releasing them in that recorded order. ``release_order`` (opt-in) overrides
    the replay release order with a caller-supplied permutation of the recorded
    completion order — the chaos-scheduling hook (see ``chaos_release_order``).
    """

    def __init__(
        self,
        mode: str,
        tape: Tape,
        inner: httpx.AsyncBaseTransport | None = None,
        *,
        matcher: RequestMatcher | None = None,
        redactor: Redactor | None = None,
        release_order: list[int] | None = None,
    ) -> None:
        assert mode in ("record", "replay")
        if mode == "record" and inner is None:
            raise ValueError("record mode requires an inner transport")
        self.mode = mode
        self.tape = tape
        self.inner = inner
        self.matcher: RequestMatcher = matcher or IDENTITY_MATCHER
        self.redactor = redactor
        self._release_order_param = release_order
        self._i = 0
        self.matched = 0

        # ── record-mode concurrency tracking (an "episode" is a maximal span
        #    during which the in-flight count stays > 0). No await runs between
        #    reading and mutating this state, so it stays consistent on a single
        #    asyncio loop without locks.
        self._inflight = 0
        self._episode_indices: list[int] = []
        self._episode_max_inflight = 0
        self._episode_completed = 0
        self._episode_permutable = True

        # ── replay-mode correlation + ordered release (built lazily on first
        #    request so the matcher/tape are final and the loop is running).
        self._replay_ready = False
        self._fp_queue: dict[str, deque[int]] = {}
        self._release_order: list[int] = []
        self._release_pos = 0
        self._cond: asyncio.Condition | None = None

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if self.mode == "record":
            return await self._record(request)
        return await self._replay(request)

    async def _record(self, request: httpx.Request) -> httpx.Response:
        # Episode bookkeeping (see __init__). Runs atomically up to the await.
        if self._inflight == 0:
            self._episode_indices = []
            self._episode_max_inflight = 0
            self._episode_completed = 0
            self._episode_permutable = True
        self._inflight += 1
        self._episode_max_inflight = max(self._episode_max_inflight, self._inflight)
        if self._episode_completed > 0:
            # A request entered after another in this episode already completed,
            # so the batch is not fully overlapping — unsafe to reorder in chaos.
            self._episode_permutable = False

        inner_resp = await self.inner.handle_async_request(request)  # type: ignore[union-attr]
        resp_body = await inner_resp.aread()

        stored_resp = self.redactor.apply_response(resp_body) if self.redactor else resp_body
        self.tape.append_exchange(self.matcher.stored_request(request), stored_resp)
        idx = len(self.tape.exchanges) - 1

        # Completion bookkeeping — atomic from here to the return (no await).
        self._episode_indices.append(idx)
        self._episode_completed += 1
        self._inflight -= 1
        # Episode closed (in-flight back to 0): log it only if it was a
        # genuinely-concurrent, fully-overlapping fan-out — the safely-
        # reorderable case (>=2 requests all in flight before any completed).
        if (
            self._inflight == 0
            and self._episode_max_inflight >= 2
            and self._episode_permutable
            and len(self._episode_indices) >= 2
        ):
            self.tape.async_batches.append(list(self._episode_indices))

        return httpx.Response(
            inner_resp.status_code,
            headers={"content-type": inner_resp.headers.get("content-type", "application/json")},
            content=resp_body,
            request=request,
        )

    def _prepare_replay(self) -> None:
        """Index recorded exchanges by request fingerprint (FIFO per fingerprint
        for duplicate requests) and fix the release order (recorded completion
        order by default, or the caller's chaos permutation)."""
        for idx, (req, _resp) in enumerate(self.tape.exchanges):
            self._fp_queue.setdefault(self.matcher.stored_fingerprint(req), deque()).append(idx)
        n = len(self.tape.exchanges)
        if self._release_order_param is None:
            self._release_order = list(range(n))
        else:
            order = list(self._release_order_param)
            if sorted(order) != list(range(n)):
                raise ValueError(
                    f"release_order must be a permutation of range({n}); got {order!r}"
                )
            self._release_order = order
        self._replay_ready = True

    async def _replay(self, request: httpx.Request) -> httpx.Response:
        if not self._replay_ready:
            self._prepare_replay()
        if self._cond is None:  # bind to the running loop on first use
            self._cond = asyncio.Condition()

        live_fp = self.matcher.live_fingerprint(request)
        queue = self._fp_queue.get(live_fp)
        if not queue:
            raise DivergenceError(
                f"async replay request diverged from tape (fingerprint "
                f"{live_fp[:12]} not recorded or already consumed; served "
                f"{self._i}/{len(self.tape.exchanges)})"
            )
        idx = queue.popleft()
        _rec_req, rec_resp = self.tape.exchange(idx)

        # Ordered-release gate: hold this response until it is this exchange's
        # turn in the recorded completion order. Sequential runs never wait (the
        # condition is already satisfied on arrival); a fan-out replays in the
        # recorded order regardless of the live arrival order.
        async with self._cond:
            while self._release_order[self._release_pos] != idx:
                await self._cond.wait()
            self._release_pos += 1
            self._i += 1
            self.matched += 1
            self._cond.notify_all()

        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            content=rec_resp,
            request=request,
        )

    def fully_consumed(self) -> bool:
        return self.mode == "replay" and self._i == len(self.tape.exchanges)


def chaos_release_order(tape: Tape, seed: int) -> list[int]:
    """A seeded, physically-possible reordering of ``tape``'s recorded completion
    order for chaos-mode replay.

    Reorders the completions *within* each recorded fully-overlapping concurrent
    batch (``tape.async_batches``) and leaves every sequential step in its
    recorded slot. Pass the result as ``AsyncTraceforkTransport(..., replay,
    release_order=...)`` to replay the same recorded responses under a different
    interleaving — surfacing completion-order-dependent ("race"/ordering) bugs.
    Only batches in which every request was in flight before any completed are
    recorded (and thus reordered), so the schedule can never demand a response
    before its request would have been sent — the ordered-release gate cannot
    deadlock on it.

    A tape with no recorded concurrent batches yields the identity order, so
    chaos replay of a sequential tape is a no-op.
    """
    order = list(range(len(tape.exchanges)))
    rng = random.Random(seed)
    for batch in tape.async_batches:
        slots = sorted(batch)  # the completion slots this batch occupies
        members = list(batch)  # its recorded completion order
        rng.shuffle(members)
        for slot, member in zip(slots, members, strict=True):
            order[slot] = member
    return order
