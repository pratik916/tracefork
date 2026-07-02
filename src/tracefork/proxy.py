"""Localhost base-URL record/replay proxy — for clients tracefork's in-process
httpx seam can't reach: curl, Node, Go, or any Python code you don't control.

**Shape**: the client points its ``base_url``/endpoint at
``http://127.0.0.1:<port>`` instead of the provider directly. Record mode
forwards each request to the real upstream (over TLS, real network) and tees
request+response bytes into a ``Tape``; replay mode serves recorded bytes back
with **no upstream at all** — an unrecorded request (or a real request-body
change) is a hard error, mirroring ``transport.py``'s replay contract at this
layer. This is a **base-URL proxy, not a transparent TLS MITM**: it does not
generate a CA or intercept a client's ``CONNECT`` tunnel, so a client must be
able to set its own base URL (every major provider SDK, curl, and any HTTP
client can) — see the module's honest-scope note below for exactly what this
does and does not cover.

**Outside the full determinism boundary.** Everywhere else in tracefork, the
agent reads time/ids/random draws through an in-process ``NondetSource``
(``nondet.py``) that a ``Recorder`` can capture and a ``ReplayNondet`` can
serve back bit-exact. A non-Python client on the other side of a TCP socket
has no such seam — tracefork cannot see, let alone virtualize, whatever
timestamp/UUID/random material the client folds into its own request bodies
or headers. So bit-exact replay through this proxy depends entirely on the
client sending a **canonically-identical request** on both the record and the
replay run; if the client bakes in something that changes call-to-call (a
fresh idempotency key, a client-side timestamp), configure a
``RequestMatcher`` (``matcher.py``) that normalizes it away, the same seam
``transport.py`` uses for Gemini/Bedrock's volatile fields. This proxy also
does not record or replay the async concurrency-completion ordering that
``AsyncTraceforkTransport`` does for in-process ``asyncio.gather``/``TaskGroup``
fan-out — each incoming request is matched to its recorded exchange
independently, by fingerprint, not by a reconstructed schedule.

**Reused, not duplicated.** Storage/hashing is ``tape.py``'s ``Tape``,
completely unchanged; request matching for the replay-time divergence check is
``matcher.py``'s existing ``RequestMatcher`` protocol (default: the identity
matcher, i.e. raw ``sha256`` of the request body — byte-for-byte the same
contract as the in-process transport).

**Streaming (SSE).** Record mode tees the upstream response chunk-by-chunk
*while forwarding* it (``httpx``'s ``stream=True`` + ``aiter_bytes()``, wired
into a ``StreamingResponse``) rather than buffering the whole body before the
client sees the first byte, so a real streaming exchange is captured
correctly without adding latency. The tape itself only ever stores body
bytes (like every other tape in this codebase — see ``transport.py``'s replay
side hardcoding ``content-type: application/json``), so replay recovers the
SSE-vs-JSON distinction with a framing heuristic (``_sniff_content_type``:
bytes starting with ``event:``/``data:`` serve as ``text/event-stream``, else
``application/json``) rather than a persisted header.
"""

from __future__ import annotations

from collections import deque
from collections.abc import AsyncIterator

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .constants import PROXY_BOUNDARY
from .matcher import IDENTITY_MATCHER, RequestMatcher
from .nondet import DivergenceError
from .tape import Tape

# Headers that are connection-scoped or body-framing and must never be forwarded
# verbatim from an incoming request to the outgoing (upstream, or matcher-facing)
# one — a bog-standard reverse-proxy hop-by-hop list plus `host`/`content-length`,
# which the outgoing client recomputes for its own destination/body.
_HOP_BY_HOP_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "host",
        "content-length",
    }
)

# A fixed placeholder origin used only to build the `httpx.Request` object the
# matcher fingerprints a request against — never dialed. It keeps the matcher's
# view of "the request" (method, path, query, headers, body) independent of
# whatever literal host:port the client happened to connect to (e.g. the ASGI
# test transport's `http://testserver` vs a real `http://127.0.0.1:8899`), so
# the SAME client request fingerprints identically whether it arrived during a
# record run or a replay run — the only thing that must match for this proxy's
# divergence check to mean anything.
_MATCHER_ORIGIN = "http://tracefork-proxy"


def _forwardable_headers(headers) -> list[tuple[str, str]]:
    return [(k, v) for k, v in headers.items() if k.lower() not in _HOP_BY_HOP_HEADERS]


def _path_and_query(request: Request) -> str:
    path = request.url.path
    if request.url.query:
        path += "?" + request.url.query
    return path


def _matcher_request_view(request: Request, body: bytes) -> httpx.Request:
    """The `httpx.Request` the matcher fingerprints — see `_MATCHER_ORIGIN`."""
    return httpx.Request(
        request.method,
        _MATCHER_ORIGIN + _path_and_query(request),
        headers=_forwardable_headers(request.headers),
        content=body,
    )


def _sniff_content_type(body: bytes) -> str:
    """Infer a response content-type from stored bytes (see module docstring's
    Streaming section for why this is a heuristic, not persisted metadata)."""
    head = body[:16].lstrip()
    if head.startswith(b"event:") or head.startswith(b"data:"):
        return "text/event-stream"
    return "application/json"


class RecordProxy:
    """Record-mode proxy body: forwards each request to `upstream_base_url` and
    tees request+response bytes into `tape`.

    `transport` is the injectable upstream seam — a synthetic fake
    (`synthetic.py`) in tests, or omitted (real network) in production. The
    *client-facing* request (method/path/query/headers/body, built against a
    fixed placeholder origin — see `_matcher_request_view`) is what gets
    fingerprinted/stored via `matcher`, not the outgoing upstream request, so
    record and a later replay of the same client traffic fingerprint
    identically regardless of what upstream host record mode happened to talk
    to.
    """

    def __init__(
        self,
        tape: Tape,
        upstream_base_url: str,
        *,
        matcher: RequestMatcher | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.tape = tape
        self.tape.boundary = PROXY_BOUNDARY
        self.matcher: RequestMatcher = matcher or IDENTITY_MATCHER
        self._client = httpx.AsyncClient(base_url=upstream_base_url, transport=transport)

    async def handle(self, request: Request) -> Response:
        body = await request.body()
        client_req = _matcher_request_view(request, body)

        upstream_req = self._client.build_request(
            request.method,
            _path_and_query(request),
            headers=_forwardable_headers(request.headers),
            content=body,
        )
        upstream_resp = await self._client.send(upstream_req, stream=True)
        status_code = upstream_resp.status_code
        content_type = upstream_resp.headers.get("content-type", "application/octet-stream")

        tape = self.tape
        matcher = self.matcher

        async def body_iterator() -> AsyncIterator[bytes]:
            chunks: list[bytes] = []
            try:
                async for chunk in upstream_resp.aiter_bytes():
                    chunks.append(chunk)
                    yield chunk
            finally:
                await upstream_resp.aclose()
            # Only reached if the stream above ran to completion (not on a
            # client disconnect / cancellation) — a partial capture is never
            # written to the tape.
            tape.append_exchange(matcher.stored_request(client_req), b"".join(chunks))

        return StreamingResponse(body_iterator(), status_code=status_code, media_type=content_type)

    async def aclose(self) -> None:
        await self._client.aclose()


class ReplayProxy:
    """Replay-mode proxy body: serves recorded bytes for each request from
    `tape`, with NO upstream.

    Requests are correlated to their recorded exchange by matcher fingerprint —
    a FIFO queue per fingerprint, so repeated identical requests replay in
    their originally-recorded order — rather than by arrival order, since an
    HTTP server can see independent client connections arrive in any order. A
    request whose fingerprint has no (remaining) recorded match — an
    unrecorded request, or a real body/field change the matcher does not
    normalize away — is a hard error (`DivergenceError`, surfaced as HTTP 502).
    A change the matcher normalizes away (e.g. a volatile header/field via a
    `CanonicalizingMatcher`) is tolerated, matching `transport.py`'s existing
    divergence contract.
    """

    def __init__(self, tape: Tape, *, matcher: RequestMatcher | None = None) -> None:
        self.tape = tape
        self.matcher: RequestMatcher = matcher or IDENTITY_MATCHER
        self.matched = 0
        self._fp_queue: dict[str, deque[int]] = {}
        for idx, (req, _resp) in enumerate(tape.exchanges):
            fp = self.matcher.stored_fingerprint(req)
            self._fp_queue.setdefault(fp, deque()).append(idx)

    async def handle(self, request: Request) -> Response:
        body = await request.body()
        client_req = _matcher_request_view(request, body)
        live_fp = self.matcher.live_fingerprint(client_req)
        queue = self._fp_queue.get(live_fp)
        if not queue:
            raise DivergenceError(
                f"proxy replay: no recorded exchange matches request "
                f"{request.method} {request.url.path} (fingerprint {live_fp[:12]}; "
                f"served {self.matched}/{len(self.tape.exchanges)})"
            )
        idx = queue.popleft()
        _rec_req, rec_resp = self.tape.exchange(idx)
        self.matched += 1
        return Response(content=rec_resp, status_code=200, media_type=_sniff_content_type(rec_resp))

    def fully_consumed(self) -> bool:
        return self.matched == len(self.tape.exchanges)


async def _divergence_handler(request: Request, exc: Exception) -> JSONResponse:
    # Registered only for `DivergenceError` below (Starlette's exception-handler
    # type is `Exception`-typed for any handler, not the specific subclass).
    return JSONResponse(status_code=502, content={"error": str(exc)})


_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE"]


def build_record_app(
    tape: Tape,
    upstream_base_url: str,
    *,
    matcher: RequestMatcher | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    """A FastAPI ASGI app implementing the record-mode proxy over `tape`.

    `transport` is the injectable upstream seam (see `RecordProxy`). The
    returned app's `state.proxy` is the underlying `RecordProxy` — callers
    should `await app.state.proxy.aclose()` when done (the CLI's `proxy record`
    command does this after `uvicorn.run` returns).
    """
    proxy = RecordProxy(tape, upstream_base_url, matcher=matcher, transport=transport)
    app = FastAPI(title="tracefork-proxy-record", docs_url=None, redoc_url=None)
    app.state.proxy = proxy

    @app.api_route("/{full_path:path}", methods=_METHODS)
    async def _forward(request: Request, full_path: str) -> Response:
        return await proxy.handle(request)

    return app


def build_replay_app(tape: Tape, *, matcher: RequestMatcher | None = None) -> FastAPI:
    """A FastAPI ASGI app implementing the replay-mode proxy over `tape`. The
    returned app's `state.proxy` is the underlying `ReplayProxy` (its
    `fully_consumed()` reports whether every recorded exchange was served)."""
    proxy = ReplayProxy(tape, matcher=matcher)
    app = FastAPI(title="tracefork-proxy-replay", docs_url=None, redoc_url=None)
    app.state.proxy = proxy
    app.add_exception_handler(DivergenceError, _divergence_handler)

    @app.api_route("/{full_path:path}", methods=_METHODS)
    async def _serve(request: Request, full_path: str) -> Response:
        return await proxy.handle(request)

    return app
