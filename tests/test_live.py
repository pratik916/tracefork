"""Offline tests for tracefork-bge.61: `tail_checkpoint()` in isolation
(asyncio.run, no server) plus a TestClient-level test of the new
`/api/checkpoint/tail` route on the real server.py app."""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from tracefork.checkpoint import CheckpointWriter
from tracefork.live import tail_checkpoint
from tracefork.server import app as fastapi_app
from tracefork.tape import Tape


async def _collect(agen):
    return [frame async for frame in agen]


def test_tail_checkpoint_yields_existing_rows_then_done_when_finalized(tmp_path):
    path = str(tmp_path / "checkpoint.sqlite")
    writer = CheckpointWriter(path, agent_name="a")
    writer.append_exchange(b"req1", b"resp1")
    writer.append_exchange(b"req2", b"resp2")
    writer.finalize(Tape(agent_name="a"))

    frames = asyncio.run(_collect(tail_checkpoint(path)))
    exchange_frames = [f for f in frames if f.startswith("event: exchange")]
    done_frames = [f for f in frames if f.startswith("event: done")]
    assert len(exchange_frames) == 2
    assert '"seq": 1' in exchange_frames[0]
    assert '"seq": 2' in exchange_frames[1]
    assert len(done_frames) == 1
    assert frames[-1] == done_frames[0]


def test_tail_checkpoint_resumes_from_since_seq(tmp_path):
    path = str(tmp_path / "checkpoint.sqlite")
    writer = CheckpointWriter(path, agent_name="a")
    writer.append_exchange(b"req1", b"resp1")
    writer.append_exchange(b"req2", b"resp2")
    writer.append_exchange(b"req3", b"resp3")
    writer.finalize(Tape(agent_name="a"))

    frames = asyncio.run(_collect(tail_checkpoint(path, since_seq=1)))
    exchange_frames = [f for f in frames if f.startswith("event: exchange")]
    assert len(exchange_frames) == 2
    assert '"seq": 2' in exchange_frames[0]
    assert '"seq": 3' in exchange_frames[1]
    assert any(f.startswith("event: done") for f in frames)


def test_tail_checkpoint_stops_after_max_polls_when_not_finalized(tmp_path):
    path = str(tmp_path / "checkpoint.sqlite")
    writer = CheckpointWriter(path, agent_name="a")
    writer.append_exchange(b"req1", b"resp1")
    # NOT finalized -- must not hang.

    frames = asyncio.run(_collect(tail_checkpoint(path, max_polls=1, poll_interval=0)))
    exchange_frames = [f for f in frames if f.startswith("event: exchange")]
    done_frames = [f for f in frames if f.startswith("event: done")]
    assert len(exchange_frames) == 1
    assert len(done_frames) == 0


def test_tail_checkpoint_raises_file_not_found_for_missing_path(tmp_path):
    path = str(tmp_path / "never-existed.sqlite")

    async def _consume():
        async for _ in tail_checkpoint(path):
            pass

    with pytest.raises(FileNotFoundError):
        asyncio.run(_consume())


def test_server_tail_checkpoint_endpoint_streams_sse_and_404s_on_missing_path(tmp_path):
    path = str(tmp_path / "checkpoint.sqlite")
    writer = CheckpointWriter(path, agent_name="a")
    writer.append_exchange(b"req1", b"resp1")
    writer.finalize(Tape(agent_name="a"))

    client = TestClient(fastapi_app)
    resp = client.get("/api/checkpoint/tail", params={"path": path})
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    assert "event: exchange" in resp.text
    assert "event: done" in resp.text

    missing_resp = client.get(
        "/api/checkpoint/tail", params={"path": str(tmp_path / "nope.sqlite")}
    )
    assert missing_resp.status_code == 404
