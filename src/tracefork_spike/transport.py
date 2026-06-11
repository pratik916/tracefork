"""The recording/replay httpx transport — the model-I/O capture seam.

Record mode: forward each request to an inner transport, capture the request body
and the full response body into the tape, and return the response unchanged.

Replay mode: ignore the network entirely. For each request, pop the next recorded
exchange, assert the request body is byte-identical to what was recorded (this is the
divergence detector), and serve the recorded response bytes back. A replay transport
has no inner transport, so any unexpected/extra request is a hard error rather than a
silent network call.
"""

from __future__ import annotations

import httpx

from .nondet import DivergenceError
from .tape import Tape, sha256_hex


class TraceforkTransport(httpx.BaseTransport):
    def __init__(self, mode: str, tape: Tape, inner: httpx.BaseTransport | None = None) -> None:
        assert mode in ("record", "replay")
        if mode == "record" and inner is None:
            raise ValueError("record mode requires an inner transport")
        self.mode = mode
        self.tape = tape
        self.inner = inner
        self._i = 0
        self.matched = 0  # number of replay request-hashes that matched the tape

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        body = request.content

        if self.mode == "record":
            inner_resp = self.inner.handle_request(request)  # type: ignore[union-attr]
            resp_body = inner_resp.read()
            self.tape.append_exchange(body, resp_body)
            return httpx.Response(
                inner_resp.status_code,
                headers={"content-type": "application/json"},
                content=resp_body,
                request=request,
            )

        # replay
        if self._i >= len(self.tape.exchanges):
            raise DivergenceError(
                f"replay made an unrecorded request #{self._i} "
                f"(tape has {len(self.tape.exchanges)} exchanges)"
            )
        rec_req, rec_resp = self.tape.exchange(self._i)
        if sha256_hex(rec_req) != sha256_hex(body):
            raise DivergenceError(
                f"request #{self._i} diverged from the tape "
                f"(recorded {sha256_hex(rec_req)[:12]}, replay {sha256_hex(body)[:12]})"
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
