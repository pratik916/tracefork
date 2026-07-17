"""Live-tail SSE endpoint over an in-progress ``CheckpointWriter`` recording.

Addresses a live run by its checkpoint **file path** — the same sole
identifier ``CheckpointWriter``/``recover_checkpoint`` already use before a
run is ever ``TapeStore.save_tape``'d (no ``run_id`` exists yet for an
in-progress recording; that's a separate, out-of-scope follow-up — see this
module's own scope note below).

Only digests are pushed over the wire — sha256 hex of the exact request/
response bytes ``checkpoint.py``'s ``read_new_exchanges`` returns, never the
raw bytes themselves — matching the redaction-conscious pattern already used
for provenance metadata (``tape.py``'s ``provenance`` field): a live-tail
subscriber never receives a duplicate copy of potentially-sensitive exchange
payloads.

The returned ``router`` is a plain ``fastapi.APIRouter`` — additive, mount it
onto an existing app via ``app.include_router(router)`` (this is deliberately
NOT wired into ``server.py``'s app in this module, so this endpoint's own
tests mount it on a throwaway ``FastAPI()`` instance rather than importing
and booting the store-backed app).

**Scope**: this ships the full live-tail slice for a checkpoint-file-
addressed live run only. Explicitly NOT here: a ``run_id`` -> checkpoint-path
registry (so a live run could be addressed the same way a finalized one is
via ``/api/run/{run_id}``), any ``web/report.html`` ``EventSource`` client
wiring (the report UI's live-mode affordance stays fetch-based per
tracefork-bge.12/15's existing scope note), and raw exchange-body streaming.
Each is a separable follow-up, not a prerequisite this endpoint is blocked on.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from .checkpoint import checkpoint_status, read_new_exchanges

# How often the generator re-polls the checkpoint file while waiting for new
# exchanges or finalization. A live SQLite file, not a push notification —
# polling is the honest model here, same as any other WAL-tailing consumer.
POLL_INTERVAL_SECONDS = 0.25

router = APIRouter()


def _exchange_frame(seq: int, req: bytes, resp: bytes) -> bytes:
    payload = {
        "seq": seq,
        "req_sha256": hashlib.sha256(req).hexdigest(),
        "resp_sha256": hashlib.sha256(resp).hexdigest(),
    }
    return f"event: exchange\ndata: {json.dumps(payload)}\n\n".encode()


def _done_frame(status: dict[str, Any]) -> bytes:
    payload = {
        "was_finalized": status["was_finalized"],
        "exchange_count": status["exchange_count"],
    }
    return f"event: done\ndata: {json.dumps(payload)}\n\n".encode()


async def _stream_checkpoint(path: str, since_seq: int) -> AsyncIterator[bytes]:
    """Poll ``path`` for new exchanges past ``since_seq``, yielding one SSE
    ``exchange`` frame per new row (in commit order), then a terminal
    ``done`` frame — and returns (closing the stream) — the first time
    ``checkpoint_status`` reports ``was_finalized``. Runs the blocking sqlite
    reads off the event loop via ``asyncio.to_thread`` each poll."""
    seq = since_seq
    while True:
        rows = await asyncio.to_thread(read_new_exchanges, path, seq)
        for row_seq, req, resp in rows:
            yield _exchange_frame(row_seq, req, resp)
            seq = row_seq
        status = await asyncio.to_thread(checkpoint_status, path)
        if status["was_finalized"]:
            yield _done_frame(status)
            return
        await asyncio.sleep(POLL_INTERVAL_SECONDS)


@router.get("/api/checkpoint/stream")
async def stream_checkpoint(path: str, since_seq: int = 0) -> StreamingResponse:
    """``GET /api/checkpoint/stream?path=...&since_seq=0`` — 404 (mirroring
    ``server.py``'s ``get_run``/``get_branch`` KeyError->404 pattern, checked
    up front rather than surfacing mid-stream) when no checkpoint file exists
    at ``path``; otherwise an ``event: exchange``/``event: done`` SSE stream."""
    try:
        checkpoint_status(path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"no checkpoint file at {path!r}") from None
    return StreamingResponse(_stream_checkpoint(path, since_seq), media_type="text/event-stream")
