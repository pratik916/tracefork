"""Virtualised nondeterminism sources.

Bit-exact replay requires capturing every nondeterminism draw at record time
and serving it back identically at replay. `RecordingNondet` draws real values
and logs them; `ReplayNondet` serves them back in order; `DriftingNondet` is
the negative control (fresh real values → forced divergence).

The SDK masks transport exceptions as `APIConnectionError`; `find_divergence`
unwraps `__cause__`/`__context__` to recover a `DivergenceError`.
"""

from __future__ import annotations

import datetime
import uuid
from typing import Protocol


class DivergenceError(RuntimeError):
    """Raised when a replay diverges from the recorded tape."""


def find_divergence(exc: BaseException | None) -> DivergenceError | None:
    """Walk an exception's cause/context chain for a DivergenceError.

    The Anthropic SDK wraps any exception raised inside its httpx transport in
    `APIConnectionError`. This recovers the original `DivergenceError`."""
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
    def new_uuid_hex(self) -> str: ...


class RecordingNondet:
    """Draws genuinely real values and logs each draw."""

    def __init__(self) -> None:
        # Capture the real datetime.now and uuid.uuid4 at init time, before
        # Recorder.__enter__ patches datetime.datetime with a subclass.
        self._real_now = datetime.datetime.now
        self._real_uuid4 = uuid.uuid4
        self.draws: list[tuple[str, str]] = []

    def now_iso(self) -> str:
        v = self._real_now(datetime.UTC).isoformat()
        self.draws.append(("clock", v))
        return v

    def new_uuid_hex(self) -> str:
        v = self._real_uuid4().hex
        self.draws.append(("uuid", v))
        return v


class ReplayNondet:
    """Serves recorded draws back in order; errors on kind/order mismatch."""

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

    def new_uuid_hex(self) -> str:
        return self._next("uuid")

    def fully_consumed(self) -> bool:
        return self._i == len(self._draws)


class DriftingNondet(RecordingNondet):
    """Negative control: draws fresh real values during replay, forcing divergence."""
