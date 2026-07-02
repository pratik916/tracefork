"""proxy.py tests — localhost base-URL record/replay proxy.

Everything here drives the FastAPI apps in-process via `httpx.ASGITransport`
(both as the *client* side hitting the proxy, and — for record mode — as the
injected fake *upstream* the proxy forwards to via `synthetic.py`'s
`AsyncScriptedFakeLLM`/`AsyncStreamingFakeLLM`). No real network, no key.
"""

import json

import httpx
from fastapi import FastAPI
from typer.testing import CliRunner

from tracefork.cli import app as cli_app
from tracefork.constants import PROXY_BOUNDARY
from tracefork.matcher import redacting_matcher
from tracefork.proxy import build_record_app, build_replay_app
from tracefork.tape import Tape

from .fakes import AsyncScriptedFakeLLM, AsyncStreamingFakeLLM

runner = CliRunner()


def _client_for(app: FastAPI) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://proxy-under-test")


# ── record -> replay round trip ──────────────────────────────────────────────


async def test_record_then_replay_roundtrip_is_bit_exact():
    tape = Tape()
    upstream = AsyncScriptedFakeLLM([b'{"id":"resp-1"}', b'{"id":"resp-2"}'])
    record_app = build_record_app(tape, "https://upstream.example", transport=upstream)

    req1 = b'{"model":"m","messages":[{"role":"user","content":"hi"}]}'
    req2 = b'{"model":"m","messages":[{"role":"user","content":"again"}]}'

    async with _client_for(record_app) as client:
        r1 = await client.post("/v1/messages", content=req1)
        r2 = await client.post("/v1/messages", content=req2)
    await record_app.state.proxy.aclose()

    assert r1.status_code == 200
    assert r1.content == b'{"id":"resp-1"}'
    assert r2.content == b'{"id":"resp-2"}'
    assert tape.exchanges == [
        (req1, b'{"id":"resp-1"}'),
        (req2, b'{"id":"resp-2"}'),
    ]
    # a proxy-recorded tape is flagged as outside the full determinism boundary
    assert tape.boundary == PROXY_BOUNDARY

    replay_app = build_replay_app(tape)
    async with _client_for(replay_app) as client:
        rr1 = await client.post("/v1/messages", content=req1)
        rr2 = await client.post("/v1/messages", content=req2)

    assert rr1.status_code == 200
    assert rr1.content == r1.content
    assert rr2.content == r2.content
    assert replay_app.state.proxy.fully_consumed()


async def test_replay_serves_recorded_bytes_with_no_upstream():
    """Replay mode needs no upstream transport/base_url at all — the app is
    built from a tape alone and still serves the exact recorded response."""
    tape = Tape()
    tape.append_exchange(b'{"model":"m"}', b'{"id":"resp-only"}')
    replay_app = build_replay_app(tape)

    async with _client_for(replay_app) as client:
        resp = await client.post("/v1/messages", content=b'{"model":"m"}')

    assert resp.status_code == 200
    assert resp.content == b'{"id":"resp-only"}'
    assert replay_app.state.proxy.fully_consumed()


# ── divergence / hard-error contract ────────────────────────────────────────


async def test_replay_unrecorded_request_hard_errors():
    tape = Tape()
    tape.append_exchange(b'{"model":"m"}', b'{"id":"resp-1"}')
    replay_app = build_replay_app(tape)

    async with _client_for(replay_app) as client:
        ok = await client.post("/v1/messages", content=b'{"model":"m"}')
        assert ok.status_code == 200

        # a request whose body was never recorded
        unrecorded = await client.post("/v1/messages", content=b'{"model":"totally-different"}')
        assert unrecorded.status_code == 502
        assert "error" in unrecorded.json()

        # the recorded exchange is already consumed -- a second copy of the
        # SAME body is *also* unrecorded now
        exhausted = await client.post("/v1/messages", content=b'{"model":"m"}')
        assert exhausted.status_code == 502


async def test_record_replay_with_matcher_tolerates_volatile_diff_detects_real_diff():
    """`redacting_matcher()` drops a rotating `idempotency_key` body field and
    `authorization`/`x-api-key` headers. A curl/Node client that mints a fresh
    idempotency key and re-sends its bearer token on every call must still
    replay bit-exact under this matcher; a genuine model/content change must
    still hard-error."""
    m = redacting_matcher()
    tape = Tape()
    upstream = AsyncScriptedFakeLLM([b'{"id":"resp-1"}'])
    record_app = build_record_app(tape, "https://upstream.example", matcher=m, transport=upstream)

    record_body = json.dumps({"model": "m", "idempotency_key": "rec-key"}).encode()
    async with _client_for(record_app) as client:
        rec = await client.post(
            "/v1/messages", content=record_body, headers={"authorization": "Bearer rec-secret"}
        )
    await record_app.state.proxy.aclose()
    assert rec.status_code == 200
    assert len(tape.exchanges) == 1

    # volatile-only difference (idempotency key + bearer token both rotated)
    tolerant_replay_app = build_replay_app(tape, matcher=m)
    replay_body = json.dumps({"model": "m", "idempotency_key": "replay-key-different"}).encode()
    async with _client_for(tolerant_replay_app) as client:
        ok = await client.post(
            "/v1/messages", content=replay_body, headers={"authorization": "Bearer replay-secret"}
        )
    assert ok.status_code == 200
    assert ok.content == b'{"id":"resp-1"}'
    assert tolerant_replay_app.state.proxy.fully_consumed()

    # a genuine field change (the model itself) is still a hard divergence
    strict_replay_app = build_replay_app(tape, matcher=m)
    changed_body = json.dumps({"model": "different-model", "idempotency_key": "rec-key"}).encode()
    async with _client_for(strict_replay_app) as client:
        bad = await client.post("/v1/messages", content=changed_body)
    assert bad.status_code == 502
    assert "error" in bad.json()


# ── streaming (SSE) ──────────────────────────────────────────────────────────


async def test_streaming_exchange_records_and_replays():
    tape = Tape()
    chunks = [
        b'event: message_start\ndata: {"a":1}\n\n',
        b'event: content_block_delta\ndata: {"b":2}\n\n',
        b"event: message_stop\ndata: {}\n\n",
    ]
    upstream = AsyncStreamingFakeLLM([chunks])
    record_app = build_record_app(tape, "https://upstream.example", transport=upstream)

    req_body = b'{"model":"m","stream":true}'
    async with (
        _client_for(record_app) as client,
        client.stream("POST", "/v1/messages", content=req_body) as resp,
    ):
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        received = b""
        async for chunk in resp.aiter_bytes():
            received += chunk
    await record_app.state.proxy.aclose()

    full_body = b"".join(chunks)
    assert received == full_body
    assert tape.exchanges == [(req_body, full_body)]

    replay_app = build_replay_app(tape)
    async with _client_for(replay_app) as client:
        replayed = await client.post("/v1/messages", content=req_body)

    assert replayed.status_code == 200
    assert replayed.content == full_body
    assert replayed.headers["content-type"].startswith("text/event-stream")
    assert replay_app.state.proxy.fully_consumed()


# ── CLI wiring (argument validation only -- no uvicorn.run in tests) ────────


def test_cli_proxy_rejects_invalid_mode(tmp_path):
    result = runner.invoke(cli_app, ["proxy", "bogus", "--tape", str(tmp_path / "t.tape.sqlite")])
    assert result.exit_code == 1
    assert "record" in result.output and "replay" in result.output


def test_cli_proxy_record_requires_upstream(tmp_path):
    result = runner.invoke(cli_app, ["proxy", "record", "--tape", str(tmp_path / "t.tape.sqlite")])
    assert result.exit_code == 1
    assert "--upstream" in result.output


def test_cli_proxy_replay_requires_existing_tape(tmp_path):
    missing = tmp_path / "missing.tape.sqlite"
    result = runner.invoke(cli_app, ["proxy", "replay", "--tape", str(missing)])
    assert result.exit_code == 1
    assert "No tape found" in result.output


def test_cli_proxy_rejects_unknown_matcher(tmp_path):
    result = runner.invoke(
        cli_app,
        [
            "proxy",
            "record",
            "--tape",
            str(tmp_path / "t.tape.sqlite"),
            "--upstream",
            "https://x.example",
            "--matcher",
            "does-not-exist",
        ],
    )
    assert result.exit_code != 0
