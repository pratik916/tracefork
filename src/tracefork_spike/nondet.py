"""Virtualized nondeterminism sources.

The headline tracefork claim is bit-exact replay. That is only possible if every
source of nondeterminism the agent consumes is captured at record time and served
back identically at replay time. This module is the seam: the toy agent reads the
clock and generates IDs *only* through a `NondetSource`, never through `time` /
`uuid` directly. Swap `RecordingNondet` for `ReplayNondet` and the same agent code
produces a byte-identical trajectory.

`DriftingNondet` is the negative control: it draws fresh real values during replay,
which must make the replay diverge — proving the verifier actually detects drift
rather than always passing.
"""

from __future__ import annotations

import datetime
import uuid
from typing import Protocol


class DivergenceError(RuntimeError):
    """Raised when a replay diverges from the recorded tape."""


def find_divergence(exc: BaseException | None) -> "DivergenceError | None":
    """Walk an exception's cause/context chain for a DivergenceError.

    The Anthropic SDK wraps any exception raised inside its httpx transport in an
    `APIConnectionError`, so a divergence we raise from the replay transport arrives
    as `APIConnectionError.__cause__` (possibly nested). This recovers the original."""
    seen: set[int] = set()
    cur = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, DivergenceError):
            return cur
        cur = cur.__cause__ or cur.__context__
    return None


class NondetSource(Protocol):
    def now_iso(self) -> str: ...
    def new_id(self, prefix: str) -> str: ...


class RecordingNondet:
    """Draws genuinely real values (wall clock, random UUIDs) and logs each draw."""

    def __init__(self) -> None:
        self.draws: list[tuple[str, str]] = []

    def now_iso(self) -> str:
        v = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.draws.append(("clock", v))
        return v

    def new_id(self, prefix: str) -> str:
        v = f"{prefix}_{uuid.uuid4().hex[:16]}"
        self.draws.append(("id", v))
        return v


class ReplayNondet:
    """Serves recorded draws back in order; errors on order/kind/exhaustion mismatch."""

    def __init__(self, draws: list[tuple[str, str]]) -> None:
        self._draws = list(draws)
        self._i = 0

    def _next(self, kind: str) -> str:
        if self._i >= len(self._draws):
            raise DivergenceError(
                f"replay asked for a {kind!r} draw but the tape is exhausted "
                f"(consumed {self._i}/{len(self._draws)})"
            )
        rec_kind, value = self._draws[self._i]
        if rec_kind != kind:
            raise DivergenceError(
                f"draw #{self._i}: replay asked for {kind!r}, tape has {rec_kind!r}"
            )
        self._i += 1
        return value

    def now_iso(self) -> str:
        return self._next("clock")

    def new_id(self, prefix: str) -> str:
        return self._next("id")

    def fully_consumed(self) -> bool:
        return self._i == len(self._draws)


class DriftingNondet(RecordingNondet):
    """Negative control: behaves like RecordingNondet (fresh real values) during a
    replay, which makes the rebuilt request bytes diverge from the tape."""
