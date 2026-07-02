"""Opt-in runtime boundary guard for record mode.

`Recorder`/`AsyncRecorder` already virtualize clock/id/random draws through
`NondetSource` (see `nondet.py`) — but nothing stops an agent from bypassing
that seam and reading `random`/the clock directly, or forking work onto a
thread/subprocess (outside the declared single-process determinism boundary;
see `CLAUDE.md`). Today that mistake produces a tape that *looks* fine and
only reveals itself as a mysterious replay divergence later. `BoundaryGuard`
makes it fail loudly, at record time, instead.

**Scope (don't overstate).** This is a best-effort diagnostic, not a sandbox:
    * `threading.Thread.start` and `subprocess.Popen.__init__` are patched to
      hard-error unconditionally — nothing in tracefork's own recording path
      spawns either (verified empirically against the Anthropic SDK + httpx
      using tracefork's synthetic/recording transports).
    * `random.random` is patched to hard-error — the module-level entry point
      is a plain reassignable function (unlike `datetime.datetime.now`, see
      below), so it can be intercepted exactly like `uuid.uuid4` is.
    * `time.monotonic` / `time.sleep` are patched to hard-error, as a
      practical proxy for "direct clock reads/waits" — verified to never fire
      from tracefork's own recording path.
    * `datetime.datetime.now()` is deliberately **not** patched — same reason
      `Recorder` doesn't patch it (`recorder.py`'s module docstring): it's a
      classmethod on an immutable C type, and swapping `datetime.datetime`
      for a subclass breaks the Anthropic SDK's lazy pydantic schema builder.
    * `time.time()` is deliberately **not** patched: httpx's cookie-jar
      machinery (`http.cookiejar.extract_cookies`, invoked on every response,
      even when no cookies are set) calls `time.time()` unconditionally, so
      guarding it would fail on every single recorded exchange regardless of
      what the agent does — a false-positive, not a real signal.
    * `subprocess.Popen` has one more wrinkle: the Anthropic SDK's own
      `platform_headers()` helper is `functools.lru_cache`d but, uncached,
      shells out (`uname -p`, `file -b <python>`) to build an `X-Stainless-OS`
      header. `BoundaryGuard.__enter__` pre-warms that cache (best-effort;
      failures are swallowed) so the *first* real API call made under an
      active guard doesn't trip on the SDK's own housekeeping.

OPT-IN, default OFF everywhere in tracefork: nothing constructs or enters a
`BoundaryGuard` unless a caller explicitly asks for it (`Recorder(...,
boundary_guard=True)` or `TraceforkConfig(boundary_guard=True)`).
"""

from __future__ import annotations

import random as _random_module
import subprocess as _subprocess_module
import threading as _threading_module
import time as _time_module
from typing import Any


class BoundaryViolationError(RuntimeError):
    """Raised when guarded code performs an operation that bypasses
    `NondetSource` (thread/subprocess spawn, or a direct `random`/clock read)
    while a `BoundaryGuard` is active."""


def _warm_anthropic_platform_headers_cache() -> None:
    """Best-effort: force the Anthropic SDK's lru_cache'd platform-header
    derivation to run *before* the subprocess guard goes live, so its one
    known internal `subprocess.Popen` call (uncached platform detection)
    doesn't trip the guard on the first real API call. Never raises — if the
    SDK's internals have moved, this just silently no-ops and the guard still
    protects against genuine violations (only this specific SDK-internal
    false positive would resurface)."""
    try:
        import anthropic
        import anthropic._base_client as _base_client

        _base_client.platform_headers(anthropic.__version__, platform=None)
    except Exception:
        pass


class BoundaryGuard:
    """Opt-in context manager: hard-errors on boundary-bypassing operations.

    Usage::

        with BoundaryGuard():
            agent(client)  # raises BoundaryViolationError on the first violation

    Re-entrant-safe is not a goal — construct one `BoundaryGuard` per `with`
    block, mirroring `Recorder`'s own single-use-per-context-manager shape.
    """

    def __init__(self) -> None:
        self._orig_thread_start: Any = None
        self._orig_popen_init: Any = None
        self._orig_random: Any = None
        self._orig_monotonic: Any = None
        self._orig_sleep: Any = None
        self._active = False

    def __enter__(self) -> BoundaryGuard:
        _warm_anthropic_platform_headers_cache()

        self._orig_thread_start = _threading_module.Thread.start
        self._orig_popen_init = _subprocess_module.Popen.__init__
        self._orig_random = _random_module.random
        self._orig_monotonic = _time_module.monotonic
        self._orig_sleep = _time_module.sleep

        def _guarded_thread_start(*_a: Any, **_kw: Any) -> None:
            raise BoundaryViolationError(
                "threading.Thread.start() called while BoundaryGuard is active: "
                "spawning a thread crosses tracefork's declared single-process "
                "determinism boundary (see CLAUDE.md)."
            )

        def _guarded_popen_init(*_a: Any, **_kw: Any) -> None:
            raise BoundaryViolationError(
                "subprocess.Popen() called while BoundaryGuard is active: "
                "spawning a subprocess crosses tracefork's declared single-process "
                "determinism boundary (see CLAUDE.md)."
            )

        def _guarded_random() -> float:
            raise BoundaryViolationError(
                "random.random() called directly while BoundaryGuard is active: "
                "route random draws through NondetSource.random_float() so they "
                "are captured and replayed bit-exact (see nondet.py)."
            )

        def _guarded_monotonic() -> float:
            raise BoundaryViolationError(
                "time.monotonic() called directly while BoundaryGuard is active: "
                "route clock reads through NondetSource.now_iso() so they are "
                "captured and replayed bit-exact (see nondet.py)."
            )

        def _guarded_sleep(*_a: Any, **_kw: Any) -> None:
            raise BoundaryViolationError(
                "time.sleep() called directly while BoundaryGuard is active: "
                "timing-dependent waits bypass NondetSource and are not captured "
                "on the tape (see nondet.py)."
            )

        _threading_module.Thread.start = _guarded_thread_start  # type: ignore[method-assign]
        _subprocess_module.Popen.__init__ = _guarded_popen_init  # type: ignore[method-assign]
        _random_module.random = _guarded_random
        _time_module.monotonic = _guarded_monotonic
        _time_module.sleep = _guarded_sleep
        self._active = True
        return self

    def __exit__(self, *args: object) -> None:
        _threading_module.Thread.start = self._orig_thread_start  # type: ignore[method-assign]
        _subprocess_module.Popen.__init__ = self._orig_popen_init  # type: ignore[method-assign]
        _random_module.random = self._orig_random
        _time_module.monotonic = self._orig_monotonic
        _time_module.sleep = self._orig_sleep
        self._active = False
