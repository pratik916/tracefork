"""Read-only query views over a recorded `Tape`'s LLM exchange log.

`fork.py`/`diff.py`/`blame.py`/`divergence.py` already treat `Tape.exchange(i)`
as the canonical "step index" into a run — this module adds no new index
space, just two read-only ways to view it:

* `state_at(tape, n)` — the fold of exchanges `[0..n]` inclusive, as a frozen
  `TapeState` (Shepherd's `state(t)` analogue).
* `slice(tape, start, end)` — the half-open range `tape.exchanges[start:end]`,
  as a tuple of `ExchangeView`.

Both decode each exchange's request/response bytes via `divergence.py`'s
existing `_json_or_b64` (parsed JSON when possible, else a lossless
`{"_raw_b64": ...}` wrapper for raw/non-JSON bytes) — no new decode logic is
introduced here, only a read-time view.

**Scope (don't overstate)**: this module deliberately covers only
`Tape.exchanges` — the LLM request/response log every other engine
(`fork.py`, `blame.py`, `diff.py`) already indexes by step. `tape.py`'s own
docstrings note that `tool_exchanges` is "a SEPARATE ordered log" and that
`async_batches`/`draws` carry no per-exchange index, so there is no
positionally-correct way to fold them into a single "step" without inventing
an unsupported cross-log correlation. A narrower-than-ideal but honest
boundary (`checkpoint.py`'s precedent) rather than a silently-wrong one.

Nothing here touches `Tape.digest()`, `to_bytes()`/`from_bytes()`, or mutates
a `Tape` in any way — both entry points are pure read-only views.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .divergence import _json_or_b64
from .tape import Tape

__all__ = ["ExchangeView", "TapeState", "state_at", "slice"]


@dataclass(frozen=True)
class ExchangeView:
    """A JSON-safe view of one recorded LLM exchange: `step_index` plus its
    request/response bodies decoded via `divergence.py`'s `_json_or_b64`
    (parsed JSON when possible, else a lossless `{"_raw_b64": ...}` wrapper)."""

    step_index: int
    request: Any
    response: Any


@dataclass(frozen=True)
class TapeState:
    """The fold of a tape's exchanges `[0..step_index]` inclusive — the state
    of the run immediately after that step's exchange completed."""

    step_index: int
    exchanges: tuple[ExchangeView, ...]


def _exchange_view(tape: Tape, i: int) -> ExchangeView:
    request_body, response_body = tape.exchange(i)
    return ExchangeView(
        step_index=i,
        request=_json_or_b64(request_body),
        response=_json_or_b64(response_body),
    )


def state_at(tape: Tape, n: int) -> TapeState:
    """The tape's state immediately after exchange `n`: every `ExchangeView`
    for steps `[0..n]` inclusive (`n + 1` of them).

    Raises `ValueError` if `n < 0`, `IndexError` if `n >= len(tape.exchanges)`
    (list-index-out-of-range semantics, not a silent clamp — unlike `slice`,
    a single requested step that doesn't exist is a caller bug, not a range
    that legitimately extends past the end).
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    n_exchanges = len(tape.exchanges)
    if n >= n_exchanges:
        raise IndexError(f"n={n} out of range for tape with {n_exchanges} exchange(s)")
    return TapeState(step_index=n, exchanges=tuple(_exchange_view(tape, i) for i in range(n + 1)))


def slice(tape: Tape, start: int, end: int) -> tuple[ExchangeView, ...]:
    """The half-open range `tape.exchanges[start:end]` as a tuple of
    `ExchangeView`.

    Out-of-range bounds clamp rather than raise (`range(len(tape.exchanges))
    [start:end]` resolves the actual indices exactly as list-slicing would,
    including a start past the end or an end past the length collapsing to an
    empty result) — except one deliberate departure from raw list-slice
    permissiveness: `start > end` raises `ValueError`, surfacing a
    reversed-bounds caller bug instead of silently returning an empty tuple.
    """
    if start > end:
        raise ValueError(f"start ({start}) must not be greater than end ({end})")
    indices = range(len(tape.exchanges))[start:end]
    return tuple(_exchange_view(tape, i) for i in indices)
