"""Virtualised nondeterminism sources.

Bit-exact replay requires capturing every nondeterminism draw at record time
and serving it back identically at replay. `RecordingNondet` draws real values
and logs them; `ReplayNondet` serves them back in order; `DriftingNondet` is
the negative control (fresh real values → forced divergence).

Four draw kinds are virtualized: clock (`now_iso`), id (`new_uuid_hex`),
random (`random_float`), and env (`get_env`) — each recorded/replayed the
same way. Like `now_iso`, `random_float` is additive/opt-in: an agent must be
handed the active `NondetSource` explicitly (see `tracefork_spike.agent` for
the pattern) and call it instead of `random.random()` directly; nothing in
`Recorder` patches `random` globally the way it does `uuid.uuid4` (see
`recorder.py`). `random_float()` logs the exact `float.hex()` representation
so replay is bit-exact with no float-formatting rounding. `get_env(name,
default=None)` logs a NUL-joined `"{flag}\0{name}\0{value}"` string: a 1-byte
set/unset `flag` ("1"/"0") lets an unset variable (`None`) round-trip
distinctly from an empty-string value, `name` is carried alongside the value
so `ReplayNondet.get_env` can assert the replayed call asks for the SAME
variable the tape recorded (a stronger check than the other three kinds
need, since only `get_env` takes an argument), and POSIX environment values
structurally cannot contain a NUL byte, so the encoding is lossless and
collision-free.

The SDK masks transport exceptions as `APIConnectionError`; `find_divergence`
unwraps `__cause__`/`__context__` to recover a `DivergenceError`.
"""

from __future__ import annotations

import datetime
import os
import random
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
    def random_float(self) -> float: ...
    def get_env(self, name: str, default: str | None = None) -> str | None: ...


class RecordingNondet:
    """Draws genuinely real values and logs each draw."""

    def __init__(self) -> None:
        # Capture the real datetime.now, uuid.uuid4, and random.random at init
        # time, before Recorder.__enter__ patches datetime.datetime with a subclass.
        self._real_now = datetime.datetime.now
        self._real_uuid4 = uuid.uuid4
        self._real_random = random.random
        self.draws: list[tuple[str, str]] = []

    def now_iso(self) -> str:
        v = self._real_now(datetime.UTC).isoformat()
        self.draws.append(("clock", v))
        return v

    def new_uuid_hex(self) -> str:
        v = self._real_uuid4().hex
        self.draws.append(("uuid", v))
        return v

    def random_float(self) -> float:
        v = self._real_random()
        # float.hex() is an exact, lossless hexadecimal representation that
        # round-trips via float.fromhex() with no precision loss — unlike
        # str(v)/repr(v), which can lose bits for some values.
        self.draws.append(("random", v.hex()))
        return v

    def get_env(self, name: str, default: str | None = None) -> str | None:
        v = os.environ.get(name, default)
        # NUL-joined "{flag}\0{name}\0{value}": a 1-byte set/unset flag lets
        # an unset (None) result round-trip distinctly from "", and carrying
        # `name` alongside lets ReplayNondet assert it's replaying the SAME
        # variable the tape recorded. POSIX env values can't contain a NUL
        # byte, so this is lossless and collision-free.
        flag = "1" if v is not None else "0"
        self.draws.append(("env", f"{flag}\0{name}\0{v if v is not None else ''}"))
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

    def random_float(self) -> float:
        return float.fromhex(self._next("random"))

    def get_env(self, name: str, default: str | None = None) -> str | None:
        packed = self._next("env")
        flag, rec_name, value = packed.split("\0", 2)
        if rec_name != name:
            raise DivergenceError(
                f"draw #{self._i - 1}: replay asked for env var {name!r}, "
                f"tape recorded {rec_name!r}"
            )
        return value if flag == "1" else None

    def fully_consumed(self) -> bool:
        return self._i == len(self._draws)


class DriftingNondet(RecordingNondet):
    """Negative control: draws fresh real values during replay, forcing divergence."""
