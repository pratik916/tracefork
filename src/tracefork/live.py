"""Read-only live-tail SSE endpoint over a `checkpoint.CheckpointWriter`-
backed recording (tracefork-bge.61) — the validation slice for live/attached
debugging named by this bead: an already-in-progress (or just-finished)
recording, backed by `checkpoint.py`'s crash-safe incremental writer, can be
tailed as Server-Sent Events without touching the writer's own process.

True interactive breakpoint-before-tool-call (pausing a LIVE agent process
mid-run, waiting for a client resume/mutate signal, then continuing) needs a
new bidirectional control channel and is explicitly out of scope here — see
this module's own scope note; `tail_checkpoint` only OBSERVES an
already-recording (or already-recorded) checkpoint file, it never controls it.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from .checkpoint import checkpoint_status, read_new_exchanges
from .report import _tape_to_data
from .tape import Tape


def format_sse(event: str, data: dict[str, Any]) -> str:
    """One SSE frame: ``event: <event>\\ndata: <json>\\n\\n``."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def tail_checkpoint(
    path: str,
    *,
    since_seq: int = 0,
    poll_interval: float = 0.25,
    max_polls: int | None = None,
) -> AsyncIterator[str]:
    """Poll the checkpoint file at ``path``, yielding one SSE ``event:
    exchange`` frame per exchange committed after ``since_seq`` and a
    terminal ``event: done`` frame once the checkpoint's ``was_finalized``
    flips true.

    Each exchange frame reuses ``report._tape_to_data``'s per-exchange
    preview shape (role/preview/request/response_preview) — zero new
    summarization logic — by feeding the row's raw ``(req, resp)`` bytes
    into a throwaway single-exchange ``Tape``.

    ``max_polls`` bounds iteration for a not-yet-finalized checkpoint
    (``None`` — the default — polls forever, until finalized); pass a small
    bound in tests so a live-but-never-finalized checkpoint can't hang them.
    Raises ``FileNotFoundError`` on the first iteration if ``path`` was never
    a ``CheckpointWriter`` target (mirrors ``recover_checkpoint``'s contract).
    """
    seq = since_seq
    polls = 0
    while True:
        rows = read_new_exchanges(path, since_seq=seq)
        for row_seq, req, resp in rows:
            throwaway = Tape()
            throwaway.append_exchange(req, resp)
            preview = _tape_to_data(throwaway)["exchanges"][0]
            yield format_sse("exchange", {"seq": row_seq, **preview})
            seq = row_seq

        status = checkpoint_status(path)
        if status["was_finalized"]:
            yield format_sse("done", {"exchange_count": status["exchange_count"]})
            return

        polls += 1
        if max_polls is not None and polls >= max_polls:
            return
        await asyncio.sleep(poll_interval)
