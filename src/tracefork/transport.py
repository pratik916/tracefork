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
"""

from __future__ import annotations

import httpx

from .matcher import IDENTITY_MATCHER, RequestMatcher
from .nondet import DivergenceError
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
    ) -> None:
        assert mode in ("record", "replay")
        if mode == "record" and inner is None:
            raise ValueError("record mode requires an inner transport")
        self.mode = mode
        self.tape = tape
        self.inner = inner
        # Default is identity: raw sha256(request.content) — the pre-seam contract.
        self.matcher: RequestMatcher = matcher or IDENTITY_MATCHER
        self._i = 0
        self.matched = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if self.mode == "record":
            inner_resp = self.inner.handle_request(request)  # type: ignore[union-attr]
            resp_body = inner_resp.read()
            self.tape.append_exchange(self.matcher.stored_request(request), resp_body)
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
    """Async recording/replay transport — identical logic to sync variant."""

    def __init__(
        self,
        mode: str,
        tape: Tape,
        inner: httpx.AsyncBaseTransport | None = None,
        *,
        matcher: RequestMatcher | None = None,
    ) -> None:
        assert mode in ("record", "replay")
        if mode == "record" and inner is None:
            raise ValueError("record mode requires an inner transport")
        self.mode = mode
        self.tape = tape
        self.inner = inner
        self.matcher: RequestMatcher = matcher or IDENTITY_MATCHER
        self._i = 0
        self.matched = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if self.mode == "record":
            inner_resp = await self.inner.handle_async_request(request)  # type: ignore[union-attr]
            resp_body = await inner_resp.aread()
            self.tape.append_exchange(self.matcher.stored_request(request), resp_body)
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
