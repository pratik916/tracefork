"""checkpoint.py's incremental read API + checkpoint_stream.py's SSE endpoint.

Scope (see checkpoint_stream.py's module docstring): a live run addressed by
its checkpoint FILE PATH only — no run_id registry, no report.html wiring,
digests only (never raw exchange bytes) over the wire.
"""

import hashlib

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tracefork.checkpoint import CheckpointWriter, checkpoint_status, read_new_exchanges
from tracefork.checkpoint_stream import router
from tracefork.tape import Tape

# ── checkpoint.py: read_new_exchanges ────────────────────────────────────────


def test_read_new_exchanges_since_seq_zero_returns_all_rows_in_order(tmp_path):
    path = str(tmp_path / "ckpt.db")
    writer = CheckpointWriter(path, agent_name="agent-a")
    writer.append_exchange(b"req-1", b"resp-1")
    writer.append_exchange(b"req-2", b"resp-2")
    writer.append_exchange(b"req-3", b"resp-3")

    rows = read_new_exchanges(path, since_seq=0)
    assert [seq for seq, _req, _resp in rows] == [1, 2, 3]
    assert rows == [
        (1, b"req-1", b"resp-1"),
        (2, b"req-2", b"resp-2"),
        (3, b"req-3", b"resp-3"),
    ]


def test_read_new_exchanges_since_seq_filters_to_newer_rows_only(tmp_path):
    path = str(tmp_path / "ckpt.db")
    writer = CheckpointWriter(path)
    writer.append_exchange(b"req-1", b"resp-1")
    writer.append_exchange(b"req-2", b"resp-2")
    writer.append_exchange(b"req-3", b"resp-3")

    rows = read_new_exchanges(path, since_seq=1)
    assert [seq for seq, _req, _resp in rows] == [2, 3]
    assert rows == [(2, b"req-2", b"resp-2"), (3, b"req-3", b"resp-3")]


def test_read_new_exchanges_since_seq_default_is_zero(tmp_path):
    path = str(tmp_path / "ckpt.db")
    writer = CheckpointWriter(path)
    writer.append_exchange(b"req-1", b"resp-1")

    assert read_new_exchanges(path) == [(1, b"req-1", b"resp-1")]


def test_read_new_exchanges_missing_file_raises(tmp_path):
    path = str(tmp_path / "does-not-exist.db")
    with pytest.raises(FileNotFoundError):
        read_new_exchanges(path)


# ── checkpoint.py: checkpoint_status ─────────────────────────────────────────


def test_checkpoint_status_before_finalize(tmp_path):
    path = str(tmp_path / "ckpt.db")
    writer = CheckpointWriter(path, agent_name="agent-b")
    writer.append_exchange(b"req-1", b"resp-1")
    writer.append_exchange(b"req-2", b"resp-2")

    status = checkpoint_status(path)
    assert status["was_finalized"] is False
    assert status["agent_name"] == "agent-b"
    assert status["exchange_count"] == 2


def test_checkpoint_status_after_finalize(tmp_path):
    path = str(tmp_path / "ckpt.db")
    writer = CheckpointWriter(path, agent_name="agent-c")
    tape = Tape(agent_name="agent-c")
    for req, resp in [(b"req-1", b"resp-1"), (b"req-2", b"resp-2")]:
        writer.append_exchange(req, resp)
        tape.append_exchange(req, resp)
    writer.finalize(tape)

    status = checkpoint_status(path)
    assert status["was_finalized"] is True
    assert status["exchange_count"] == 2


def test_checkpoint_status_missing_file_raises(tmp_path):
    path = str(tmp_path / "does-not-exist.db")
    with pytest.raises(FileNotFoundError):
        checkpoint_status(path)


# ── checkpoint_stream.py: GET /api/checkpoint/stream ─────────────────────────


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _parse_sse_events(body: str) -> list[tuple[str, str]]:
    events = []
    for block in body.strip("\n").split("\n\n"):
        if not block:
            continue
        lines = block.splitlines()
        event_line = next(line for line in lines if line.startswith("event: "))
        data_line = next(line for line in lines if line.startswith("data: "))
        events.append((event_line[len("event: ") :], data_line[len("data: ") :]))
    return events


def test_stream_finalized_checkpoint_emits_exchange_and_done_frames(tmp_path):
    """Against an ALREADY-finalized 2-exchange checkpoint the generator's
    first poll finds both rows finalized, so it terminates on that first
    poll — no sleep, no timing flakiness."""
    path = str(tmp_path / "ckpt.db")
    writer = CheckpointWriter(path, agent_name="agent-d")
    tape = Tape(agent_name="agent-d")
    pairs = [(b"req-1", b"resp-1"), (b"req-2", b"resp-2")]
    for req, resp in pairs:
        writer.append_exchange(req, resp)
        tape.append_exchange(req, resp)
    writer.finalize(tape)

    resp = _client().get("/api/checkpoint/stream", params={"path": path})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse_events(resp.text)
    assert [name for name, _data in events] == ["exchange", "exchange", "done"]

    import json

    exch1 = json.loads(events[0][1])
    exch2 = json.loads(events[1][1])
    done = json.loads(events[2][1])

    assert exch1["seq"] == 1
    assert exch1["req_sha256"] == hashlib.sha256(b"req-1").hexdigest()
    assert exch1["resp_sha256"] == hashlib.sha256(b"resp-1").hexdigest()

    assert exch2["seq"] == 2
    assert exch2["req_sha256"] == hashlib.sha256(b"req-2").hexdigest()
    assert exch2["resp_sha256"] == hashlib.sha256(b"resp-2").hexdigest()

    assert done["was_finalized"] is True
    assert done["exchange_count"] == 2


def test_stream_since_seq_skips_already_seen_rows(tmp_path):
    path = str(tmp_path / "ckpt.db")
    writer = CheckpointWriter(path)
    tape = Tape()
    pairs = [(b"req-1", b"resp-1"), (b"req-2", b"resp-2"), (b"req-3", b"resp-3")]
    for req, resp in pairs:
        writer.append_exchange(req, resp)
        tape.append_exchange(req, resp)
    writer.finalize(tape)

    resp = _client().get("/api/checkpoint/stream", params={"path": path, "since_seq": 1})
    events = _parse_sse_events(resp.text)
    assert [name for name, _data in events] == ["exchange", "exchange", "done"]

    import json

    seqs = [json.loads(data)["seq"] for name, data in events if name == "exchange"]
    assert seqs == [2, 3]


def test_stream_nonexistent_path_returns_404():
    resp = _client().get("/api/checkpoint/stream", params={"path": "/no/such/checkpoint.db"})
    assert resp.status_code == 404
