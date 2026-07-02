"""AWS Bedrock record/replay seam: hooks botocore's ``before-send`` short-circuit.

Bedrock is the outlier provider: ``boto3``/``botocore`` issue requests through
their own connection pool, never through httpx, so ``transport.py``'s httpx
seam cannot see them — this module is a **second, parallel seam** for
botocore, built without touching ``transport.py``/``tape.py``/``fork.py``/
``blame.py`` at all.

**The interception point.** botocore already has exactly the short-circuit
this seam needs: right before sending a request, it emits
``before-send.<service-id>.<operation>`` and if any registered handler
returns a non-``None`` value, botocore uses that as the HTTP response and
skips the real network call entirely — this is verified against botocore's
own source (``botocore/endpoint.py::Endpoint._do_get_response``)::

    event_name = f"before-send.{service_id}.{operation_model.name}"
    responses = self._event_emitter.emit(event_name, request=request)
    http_response = first_non_none_response(responses)
    if http_response is None:
        http_response = self._send(request)   # real network call, skipped if we returned non-None

The ``service_id`` for the ``bedrock-runtime`` client is ``"bedrock-runtime"``
(confirmed against botocore's ``data/bedrock-runtime/*/service-2.json``:
``"serviceId": "Bedrock Runtime"`` -> ``.hyphenize()``).

**Zero botocore imports needed for the seam itself.** Both sides of that
short-circuit are pure duck typing: the ``request`` handed to a ``before-send``
handler only needs ``.method``/``.url``/``.headers``/``.body`` (a real
``botocore.awsrequest.AWSPreparedRequest`` or the offline
``synthetic.FakeAWSPreparedRequest`` are interchangeable here), and the
response this module returns only needs ``.status_code``/``.headers``/
``.content``/``.raw`` (verified against ``botocore/endpoint.py::
convert_to_response_dict`` and ``botocore/awsrequest.py::AWSResponse`` — no
``isinstance`` check anywhere in that path). So ``import tracefork`` and the
whole offline test suite never need boto3/botocore installed — not even a
guarded lazy import — and the tests below register these hooks on a
hand-rolled fake event emitter (``synthetic.FakeEventEmitter``) that mirrors
botocore's ``HierarchicalEmitter.register``/``.emit()`` contract exactly (a
list of ``(handler, response)`` pairs, the same shape
``botocore.hooks.first_non_none_response`` consumes). The ONE place this
module ever touches botocore is ``_make_response``'s best-effort use of the
real ``botocore.awsrequest.AWSResponse`` class when it happens to be
importable (better production fidelity — case-insensitive headers, etc.) —
guarded by a plain ``try/except ImportError``, exactly the invariant CLAUDE.md
requires.

**Bit-exactness.** Request "sameness" is proven through the SAME
``CanonicalizingMatcher`` seam ``matcher.py`` already ships
(``bedrock_matcher()``, additive, pre-existing) by adapting the botocore
prepared request into an ``httpx.Request`` VIEW — never sent over httpx, pure
data — so ``bedrock_matcher()``'s existing SigV4-header-stripping logic runs
completely unmodified (see ``matcher.py``'s docstring: it already names
``x-amz-date`` as an anticipated volatile header). ``httpx`` is a hard
tracefork dependency already (transitively via ``anthropic``), so this adds no
new dependency.

**Replay contract** mirrors ``TraceforkTransport``'s exactly (positional walk
through ``tape.exchanges``, ``DivergenceError`` on an unrecorded request or a
fingerprint mismatch, ``fully_consumed()``) — just retargeted at the botocore
layer instead of httpx.

**SCOPE — streaming is NOT proven end-to-end through botocore.** This
transport CAN record/replay any bytes botocore hands it, including a raw
``application/vnd.amazon.eventstream``-framed streaming body (the seam is
byte-agnostic — see ``eventstream.py`` for the standalone, proven codec for
that framing). What is **not** exercised here is the full path of botocore's
own event-stream *parsing* machinery (``botocore.eventstream.EventStream``,
``StreamingBody``) consuming a replayed streaming response end-to-end through
a real ``bedrock-runtime`` client's ``invoke_model_with_response_stream``
call — that integration is materially deeper than this module's non-streaming
contract and is intentionally NOT claimed as proven. Land the non-streaming
path + the standalone eventstream codec; treat streaming-through-botocore as
a documented limitation, not a silently-weakened claim.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx

from .matcher import RequestMatcher, bedrock_matcher
from .nondet import DivergenceError
from .tape import Tape

#: The `serviceId` AWS's bedrock-runtime service model hyphenizes to.
DEFAULT_SERVICE_ID = "bedrock-runtime"
#: Operations this seam intercepts by default when `.register()` is called
#: with no explicit `operations=`. `InvokeModelWithResponseStream` is included
#: so a single `.register()` call also captures streaming requests' RAW
#: bytes (see the module docstring's streaming scope note for what that does
#: and doesn't prove).
DEFAULT_OPERATIONS = ("InvokeModel", "InvokeModelWithResponseStream")


def _coerce_body_bytes(body: Any) -> bytes:
    if body is None:
        return b""
    if isinstance(body, bytes):
        return body
    if isinstance(body, str):
        return body.encode("utf-8")
    # File-like / streaming body: read once, then rewind if possible so a REAL
    # send (record mode's `sender`) still sees the full body.
    data = body.read()
    if hasattr(body, "seek"):
        body.seek(0)
    return data if isinstance(data, bytes) else bytes(data)


def prepared_request_to_httpx(request: Any) -> httpx.Request:
    """View a botocore-shaped prepared request (``.method``/``.url``/
    ``.headers``/``.body``) as an ``httpx.Request`` — purely a data holder,
    never sent over httpx — so ``bedrock_matcher()`` (``matcher.py``, a
    ``CanonicalizingMatcher`` that already strips SigV4 signing headers) can
    fingerprint it with **zero new canonicalization logic**. Works
    identically against a real ``botocore.awsrequest.AWSPreparedRequest`` or
    the offline ``synthetic.FakeAWSPreparedRequest`` (duck typing)."""
    return httpx.Request(
        method=request.method,
        url=request.url,
        headers=dict(request.headers),
        content=_coerce_body_bytes(getattr(request, "body", None)),
    )


@dataclass
class _RawBody:
    """Duck-typed stand-in for the file-like ``.raw`` a real botocore
    ``AWSResponse`` carries (a urllib3-response-shaped object) — just enough
    for ``.content``/``StreamingBody`` reads: a ``.stream()`` generator
    yielding the body once, and ``.read()``."""

    data: bytes

    def stream(self, amt: int | None = None, decode_content: bool = False):
        yield self.data

    def read(self, amt: int | None = None) -> bytes:
        return self.data


@dataclass
class _FakeAWSResponse:
    """Duck-typed stand-in for ``botocore.awsrequest.AWSResponse``. botocore's
    ``before-send`` short-circuit only ever accesses ``.status_code``/
    ``.headers``/``.content``/``.raw`` on whatever a handler returns — never
    ``isinstance``-checked (verified against ``botocore/endpoint.py`` — see
    module docstring) — so this satisfies the real contract without importing
    botocore."""

    url: str
    status_code: int
    headers: dict[str, str]
    _body: bytes

    @property
    def content(self) -> bytes:
        return self._body

    @property
    def raw(self) -> _RawBody:
        return _RawBody(self._body)


def _make_response(url: str, status_code: int, headers: dict[str, str], body: bytes) -> Any:
    """Build a response object for botocore's ``before-send`` short-circuit.

    Uses the REAL ``botocore.awsrequest.AWSResponse`` when botocore happens to
    be importable (best production fidelity: case-insensitive headers,
    ``.text``, etc.) and a duck-typed equivalent otherwise — see
    ``_FakeAWSResponse``. This is the ONE place this module ever touches
    botocore, and it is fully guarded: neither ``import tracefork`` nor any
    offline test requires boto3/botocore to be installed.
    """
    try:
        from botocore.awsrequest import AWSResponse
    except ImportError:
        return _FakeAWSResponse(url=url, status_code=status_code, headers=headers, _body=body)
    return AWSResponse(url, status_code, headers, _RawBody(body))


def default_sender(client: httpx.Client | None = None) -> Callable[[httpx.Request], httpx.Response]:
    """A ``sender`` for real Bedrock record-mode: POSTs the already
    SigV4-signed prepared request via a plain ``httpx.Client``, bypassing
    botocore's own ``_send``/connection pool entirely. This is the ONE real
    network call this seam ever makes in record mode — mirroring
    ``TraceforkTransport``'s ``inner`` transport parameter, just implemented
    with httpx directly instead of forwarding to botocore's own sender (which
    has no equivalent short-circuit-and-tee point after signing)."""
    owned = client or httpx.Client()

    def _send(request: httpx.Request) -> httpx.Response:
        return owned.send(request)

    return _send


class BedrockTransport:
    """Record or replay botocore ``before-send`` events against a ``Tape``.

    ``mode="record"``: on each intercepted request, calls ``sender(httpx_req)``
    (a real network round-trip via ``default_sender()``, or a synthetic
    ``sender`` in tests), tees the CANONICALIZED request bytes
    (``matcher.stored_request``, which is what actually gets hashed/compared —
    see ``matcher.py``) plus the raw response bytes into ``tape``, and returns
    a response object built from those same bytes so botocore's short-circuit
    uses it instead of also sending the request itself.

    ``mode="replay"``: never touches the network. Walks ``tape.exchanges``
    positionally (mirroring ``TraceforkTransport``): raises ``DivergenceError``
    if replay runs past the end of the tape (an unrecorded request — a
    replay seam with no live endpoint MUST hard-error, never silently pass
    through), or if the live request's canonical fingerprint doesn't match the
    recorded one's. On a match, returns the recorded response bytes.
    """

    def __init__(
        self,
        mode: str,
        tape: Tape,
        *,
        matcher: RequestMatcher | None = None,
        sender: Callable[[httpx.Request], httpx.Response] | None = None,
    ) -> None:
        if mode not in ("record", "replay"):
            raise ValueError(f"mode must be 'record' or 'replay', got {mode!r}")
        if mode == "record" and sender is None:
            raise ValueError("record mode requires a `sender` callable")
        self.mode = mode
        self.tape = tape
        self.matcher = matcher or bedrock_matcher()
        self._sender = sender
        self._i = 0
        self.matched = 0

    def fully_consumed(self) -> bool:
        """Mirrors ``TraceforkTransport.fully_consumed()``: True once a replay
        has walked every recorded exchange."""
        return self.mode == "replay" and self._i == len(self.tape.exchanges)

    def register(
        self,
        event_emitter: Any,
        *,
        service_id: str = DEFAULT_SERVICE_ID,
        operations: tuple[str, ...] = DEFAULT_OPERATIONS,
    ) -> None:
        """Register this transport's ``before-send`` handler on
        ``event_emitter`` — a botocore ``client.meta.events``-shaped object
        (real ``HierarchicalEmitter``, or the offline
        ``synthetic.FakeEventEmitter``) — for each operation."""
        for op in operations:
            event_emitter.register(f"before-send.{service_id}.{op}", self._on_before_send)

    def _on_before_send(self, request: Any, **_kwargs: Any) -> Any:
        httpx_req = prepared_request_to_httpx(request)
        if self.mode == "record":
            return self._record(httpx_req)
        return self._replay(httpx_req)

    def _record(self, httpx_req: httpx.Request) -> Any:
        assert self._sender is not None  # enforced in __init__
        response = self._sender(httpx_req)
        body = response.content
        self.tape.append_exchange(self.matcher.stored_request(httpx_req), body)
        return _make_response(
            url=str(httpx_req.url),
            status_code=response.status_code,
            headers=dict(response.headers),
            body=body,
        )

    def _replay(self, httpx_req: httpx.Request) -> Any:
        if self._i >= len(self.tape.exchanges):
            raise DivergenceError(
                f"replay made unrecorded Bedrock request #{self._i} "
                f"(tape has {len(self.tape.exchanges)} exchanges)"
            )
        rec_req, rec_resp = self.tape.exchange(self._i)
        rec_fp = self.matcher.stored_fingerprint(rec_req)
        live_fp = self.matcher.live_fingerprint(httpx_req)
        if rec_fp != live_fp:
            raise DivergenceError(
                f"Bedrock request #{self._i} diverged from tape "
                f"(recorded {rec_fp}, replay {live_fp})"
            )
        self._i += 1
        self.matched += 1
        return _make_response(
            url=str(httpx_req.url),
            status_code=200,
            headers={"content-type": "application/json"},
            body=rec_resp,
        )
