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

**ConfinementSpec-lite.** A second, independently opt-in layer: pass
`confinement=ConfinementSpec(writable_roots=..., allowed_hosts=...)` to
additionally patch `builtins.open` (reject write-mode opens whose resolved
path falls outside `writable_roots`; reads are never restricted) and
`socket.socket.connect` (reject any host not in `allowed_hosts`) for the
guard's active window — the tail-record phase of a fork, in the intended
use (see `fork.py`'s `confinement=` kwarg). Capabilities are declared as
DATA (`ConfinementSpec` is a frozen dataclass) and verified independently
at this boundary, never derived from the agent's own tool-call arguments —
the classic confused-deputy hole. This targets a fixed local allowlist
boundary (loopback-proxy-style), not a full OS sandbox: Landlock/Seatbelt-
grade backends are an explicitly out-of-scope future escalation tier.
`confinement=None` (the default) leaves both `builtins.open` and
`socket.socket.connect` completely unpatched — byte-identical to the
guard's pre-`ConfinementSpec` behavior.
"""

from __future__ import annotations

import builtins as _builtins_module
import os as _os_module
import random as _random_module
import socket as _socket_module
import subprocess as _subprocess_module
import threading as _threading_module
import time as _time_module
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class BoundaryViolationError(RuntimeError):
    """Raised when guarded code performs an operation that bypasses
    `NondetSource` (thread/subprocess spawn, or a direct `random`/clock read)
    while a `BoundaryGuard` is active."""


class ConfinementViolationError(BoundaryViolationError):
    """Raised when guarded code performs a filesystem write outside the
    declared `ConfinementSpec.writable_roots`, or a `socket.connect` to a
    host outside `ConfinementSpec.allowed_hosts`, while a `BoundaryGuard`
    with an active `confinement=` spec is entered. Subclasses
    `BoundaryViolationError` so existing `pytest.raises(BoundaryViolationError,
    ...)` call sites keep matching."""


@dataclass(frozen=True)
class ConfinementSpec:
    """Declares the exact filesystem-write and network-egress surface a
    confined run may touch.

    `writable_roots` — absolute (or cwd-relative) directory paths; a
    write-mode `open()` is allowed only when its resolved path falls under
    one of these. Read-mode opens are never restricted, regardless of
    `writable_roots`.

    `allowed_hosts` — hostnames a `socket.connect()` may target; anything
    else is rejected before the underlying connect syscall runs (offline/$0
    even for the rejection path — no DNS/TCP attempt is made).

    Both default to empty tuples, i.e. "no writes anywhere, no egress
    anywhere" — the caller must explicitly declare a surface, matching the
    "declare capabilities as data, verify independently" principle this spec
    exists to enforce (see module docstring)."""

    writable_roots: tuple[str, ...] = ()
    allowed_hosts: tuple[str, ...] = ()


def _is_write_mode(mode: str) -> bool:
    """True for any `open()` mode that can mutate the file (`w`/`a`/`x`, or
    `+` for read+write like `r+`); false for pure-read modes (`r`, `rb`,
    `rt`, ...)."""
    return any(flag in mode for flag in ("w", "a", "x", "+"))


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

    `confinement` (default `None`) additionally patches `builtins.open` and
    `socket.socket.connect` for the guard's active window — see
    `ConfinementSpec`'s docstring. Leaving it `None` leaves both completely
    unpatched, byte-identical to pre-`ConfinementSpec` behavior.
    """

    def __init__(self, confinement: ConfinementSpec | None = None) -> None:
        self._confinement = confinement
        self._orig_thread_start: Any = None
        self._orig_popen_init: Any = None
        self._orig_random: Any = None
        self._orig_monotonic: Any = None
        self._orig_sleep: Any = None
        self._orig_open: Any = None
        self._orig_socket_connect: Any = None
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

        if self._confinement is not None:
            confinement = self._confinement
            writable_roots = tuple(Path(root).resolve() for root in confinement.writable_roots)
            allowed_hosts = confinement.allowed_hosts

            self._orig_open = _builtins_module.open
            self._orig_socket_connect = _socket_module.socket.connect

            def _guarded_open(file: Any, mode: str = "r", *a: Any, **kw: Any) -> Any:
                if _is_write_mode(mode) and isinstance(file, (str, _os_module.PathLike)):
                    target = Path(_os_module.fspath(file)).resolve()
                    if not any(target.is_relative_to(root) for root in writable_roots):
                        raise ConfinementViolationError(
                            f"open({file!r}, mode={mode!r}) denied: resolved path is "
                            "outside the declared ConfinementSpec.writable_roots "
                            "while BoundaryGuard confinement is active (see "
                            "boundary_guard.py)."
                        )
                return self._orig_open(file, mode, *a, **kw)

            def _guarded_socket_connect(sock: Any, address: Any, *a: Any, **kw: Any) -> Any:
                host = address[0] if isinstance(address, tuple) else address
                if isinstance(host, str) and host not in allowed_hosts:
                    raise ConfinementViolationError(
                        f"socket.connect({address!r}) denied: host is outside the "
                        "declared ConfinementSpec.allowed_hosts while BoundaryGuard "
                        "confinement is active (see boundary_guard.py)."
                    )
                return self._orig_socket_connect(sock, address, *a, **kw)

            _builtins_module.open = _guarded_open
            _socket_module.socket.connect = _guarded_socket_connect  # type: ignore[method-assign]

        self._active = True
        return self

    def __exit__(self, *args: object) -> None:
        _threading_module.Thread.start = self._orig_thread_start  # type: ignore[method-assign]
        _subprocess_module.Popen.__init__ = self._orig_popen_init  # type: ignore[method-assign]
        _random_module.random = self._orig_random
        _time_module.monotonic = self._orig_monotonic
        _time_module.sleep = self._orig_sleep

        if self._confinement is not None:
            _builtins_module.open = self._orig_open
            _socket_module.socket.connect = self._orig_socket_connect  # type: ignore[method-assign]

        self._active = False
