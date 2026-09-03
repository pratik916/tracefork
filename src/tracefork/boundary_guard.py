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
    * `time.sleep` is patched to hard-error unconditionally (sync and async
      alike), as a practical proxy for "direct timing-dependent waits" —
      verified to never fire from tracefork's own recording path (asyncio
      itself waits via its selector, never `time.sleep`).
    * `time.monotonic` is patched to hard-error, but the patch checks
      DYNAMICALLY, on every call, whether a live asyncio event loop is
      running *at that moment* (`asyncio.get_running_loop()`, cheap) and lets
      the call through unpatched when one is — see the `threading.Thread.
      start` note below for why: asyncio's own scheduling calls
      `time.monotonic()` unconditionally on every iteration it must genuinely
      wait, so guarding it while a loop runs would false-positive on any real
      async work, not just direct agent misuse. Per-call (not a one-time
      decision at `__enter__`) so this correctly covers a loop that starts
      *after* the guard is already active too (e.g. a sync `agent_fn` — under
      `Recorder`, or `fork.py`'s `agent_fn(client)` — that spins up its own
      `asyncio.run(...)` internally). Outside any loop, `time.monotonic`
      stays fully guarded, unchanged.
    * `datetime.datetime.now()` is deliberately **not** patched — same reason
      `Recorder` doesn't patch it (`recorder.py`'s module docstring): it's a
      classmethod on an immutable C type, and swapping `datetime.datetime`
      for a subclass breaks the Anthropic SDK's lazy pydantic schema builder.
    * `time.time()` is deliberately **not** patched: httpx's cookie-jar
      machinery (`http.cookiejar.extract_cookies`, invoked on every response,
      even when no cookies are set) calls `time.time()` unconditionally, so
      guarding it would fail on every single recorded exchange regardless of
      what the agent does — a false-positive, not a real signal.
    * `subprocess.Popen` pre-warm: `BoundaryGuard.__enter__` forces the
      Anthropic SDK's `functools.lru_cache`d `platform_headers()` to run once
      before the guard goes live (best-effort; failures are swallowed).
      Historically this mattered because an uncached call shelled out
      (`uname -p`, `file -b <python>`) to build the `X-Stainless-OS` header;
      the installed/locked `anthropic` 1.x no longer does that (its
      `get_platform()` uses `platform.system()`/`.machine()`/`.release()`
      exclusively — see that function's own source comment) so this pre-warm
      is dead weight against 1.x specifically, but it is cheap, harmless, and
      still real insurance against the declared `anthropic>=1.0,<2` range's
      *other* versions, so it stays.
    * `threading.Thread.start` has a real, current wrinkle under
      `AsyncRecorder`: anthropic 1.x's **async** client resolves its
      `X-Stainless-OS` platform once per instance via
      `AsyncAPIClient.request()`'s `self._platform = await
      asyncify(get_platform)()` — an `asyncio.to_thread` offload — on that
      instance's FIRST request. Unlike `platform_headers()` this is **not**
      an `lru_cache` — it is per-`AsyncAnthropic`-instance state (`.copy()`,
      used to route a client through tracefork's transport per
      `recorder.py`, resets it to `None` on the copy). The PRIMARY fix
      (`warm_anthropic_async_client_platform`) directly pre-resolves
      `self._platform` on the actual client instance before the guard
      activates, so the SDK never calls `asyncify(get_platform)()` — and so
      never spawns a thread — at all during the guarded window. A
      SECONDARY, narrower fallback (`warm_anthropic_async_platform_
      resolution`) pre-warms the asyncio default *executor* itself (one
      throwaway `asyncio.to_thread` call) so that if the primary fix ever
      misses (SDK internals moved) or something else offloads to a thread
      during the guarded window, at least a cold `ThreadPoolExecutor`'s
      `Thread.start()` is avoided by reusing an already-idle worker. Only
      relevant to `AsyncRecorder` (the sync `Recorder`/`BoundaryGuard` alone
      never awaits anything); `AsyncRecorder.__aenter__` applies both before
      entering the guard whenever the guard is enabled.

      This SDK call is also *why* `time.monotonic` above is conditionally
      skipped rather than merely pre-warmed like `Thread.start`: even with
      BOTH fixes above applied, the coroutine still genuinely suspends while
      waiting for the (now-reused) worker thread to signal completion — a
      real cross-thread wait, thread-pre-warming or not — and asyncio's
      OWN scheduling turns that suspend into at least one more
      `BaseEventLoop._run_once` iteration, which calls `time.monotonic()`
      unconditionally. Verified empirically that this is not specific to
      platform resolution: ANY genuinely-suspending async transport (a real
      network call, or even a bare `await asyncio.sleep(...)`) trips it the
      same way — confirming `time.monotonic` cannot be enforced while any
      event loop is genuinely running, hence the dynamic per-call skip above
      rather than a narrower, call-site-specific workaround.

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

import asyncio as _asyncio_module
import builtins as _builtins_module
import contextlib
import os as _os_module
import random as _random_module
import socket as _socket_module
import subprocess as _subprocess_module
import threading as _threading_module
import time as _time_module
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import TraceforkError

__all__ = [
    "BoundaryViolationError",
    "ConfinementViolationError",
    "ConfinementSpec",
    "BoundaryGuard",
    "warm_anthropic_async_client_platform",
    "warm_anthropic_async_platform_resolution",
]


class BoundaryViolationError(RuntimeError, TraceforkError):
    """Raised when guarded code performs an operation that bypasses
    `NondetSource` (thread/subprocess spawn, or a direct `random`/clock read)
    while a `BoundaryGuard` is active."""


class ConfinementViolationError(BoundaryViolationError):
    """Raised when guarded code performs a filesystem write outside the
    declared `ConfinementSpec.writable_roots`, or a `socket.connect` to a
    host outside `ConfinementSpec.allowed_hosts`, while a `BoundaryGuard`
    with an active `confinement=` spec is entered. Subclasses
    `BoundaryViolationError` so existing `pytest.raises(BoundaryViolationError,
    ...)` call sites keep matching.

    Since tracefork-bge.72, both raise sites (`_guarded_open`,
    `_guarded_socket_connect`, below) additionally set optional structured
    keyword-only attributes -- `violation_kind` (`"write"`/`"connect"`),
    `attempted` (the denied path/host), and whichever of
    `declared_writable_roots`/`declared_allowed_hosts` applies -- so a caller
    (see `confinement_diagnostics.py`) can build a typed diagnostic from the
    exception's own attributes instead of parsing `str(error)`. All four
    default to `None`, so `raise ConfinementViolationError("msg")` (the
    pre-bge.72 single-message-arg shape) is unaffected and every existing
    `pytest.raises(ConfinementViolationError, match=...)` call site keeps
    matching on the unchanged message text."""

    def __init__(
        self,
        message: str,
        *,
        violation_kind: str | None = None,
        attempted: str | None = None,
        declared_writable_roots: tuple[str, ...] | None = None,
        declared_allowed_hosts: tuple[str, ...] | None = None,
    ) -> None:
        super().__init__(message)
        self.violation_kind = violation_kind
        self.attempted = attempted
        self.declared_writable_roots = declared_writable_roots
        self.declared_allowed_hosts = declared_allowed_hosts


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
    derivation to run *before* the subprocess guard goes live. Historically
    this mattered because an uncached call shelled out (`uname -p`,
    `file -b <python>`); the installed/locked `anthropic` 1.x no longer does
    (see the module docstring's `subprocess.Popen` note) so this is currently
    dead weight, but it is cheap insurance across the whole declared
    `anthropic>=1.0,<2` range, so it stays. Never raises — if the SDK's
    internals have moved, this just silently no-ops and the guard still
    protects against genuine violations (only this specific SDK-internal
    false positive would resurface)."""
    try:
        import anthropic
        import anthropic._base_client as _base_client

        _base_client.platform_headers(anthropic.__version__, platform=None)
    except Exception:
        pass


def warm_anthropic_async_client_platform(async_client: Any) -> None:
    """Best-effort, PRIMARY fix for the `Thread.start()` half of anthropic
    1.x's async platform-resolution false positive: directly pre-resolve
    `async_client`'s own per-instance `self._platform` attribute *before* the
    guard goes live, so anthropic 1.x's `AsyncAPIClient.request()` —
    `async_client`'s own base class (`anthropic.AsyncAnthropic` subclasses
    `AsyncAPIClient` directly; verified against the installed `anthropic`
    1.x) — finds `self._platform` already set on its first guarded request
    and never calls `await asyncify(get_platform)()` (so never spawns a
    thread) at all.

    NOT a fix for `time.monotonic()`: that SDK call is a genuine cross-thread
    wait (the coroutine must actually suspend until the offloaded work
    signals completion, worker-thread-pre-warming or not — see
    `warm_anthropic_async_platform_resolution` below), and asyncio's OWN
    event-loop scheduling (`BaseEventLoop._run_once`) calls
    `time.monotonic()` unconditionally on every iteration it takes while
    waiting — verified empirically to trip `BoundaryGuard`'s
    `time.monotonic()` guard from ANY genuinely-suspending async transport,
    not just this one. `BoundaryGuard.__enter__` handles that separately (see
    its own `time.monotonic` note) by not guarding `time.monotonic` at all
    under a live event loop — this function only ever needed to solve
    `Thread.start()`.

    `self._platform` is *per-instance*, not a module-level cache (unlike the
    sync path's `platform_headers()` `lru_cache`) — `.copy()` (used to route
    a client through tracefork's transport, per `recorder.py`) resets it to
    `None` on the copy, so this must be called on the actual instance that
    will make the guarded request, after any `.copy()`.

    Never raises — if the SDK's internals have moved (attribute renamed, base
    class changed), this just silently no-ops and the guard still protects
    against genuine violations (only this specific SDK-internal false
    positive would resurface)."""
    try:
        from anthropic._base_client import get_platform

        if hasattr(async_client, "_platform"):
            async_client._platform = get_platform()
    except Exception:
        pass


async def warm_anthropic_async_platform_resolution() -> None:
    """Best-effort, SECONDARY defense-in-depth for the `Thread.start()` half:
    pre-warm asyncio's default executor with one throwaway worker thread
    *before* the guard goes live, so that IF
    `warm_anthropic_async_client_platform` above ever fails to reach the
    actual client instance (SDK internals moved, or some other code reaches
    `asyncify(get_platform)()` — or any other `asyncio.to_thread` call —
    during the guarded window), at least a new `Thread.start()` is avoided by
    reusing an already-started worker thread rather than spawning one on a
    cold executor — a narrower, partial fallback, not a substitute for the
    primary fix. (`time.monotonic()` is handled separately by
    `BoundaryGuard.__enter__` itself via a DYNAMIC per-call check — see its
    own note — so this function never needed to address that half, and that
    fix, unlike this one, covers every scenario including the one below.)

    Callers: `AsyncRecorder.__aenter__` awaits this (alongside the primary
    fix) before entering its guard whenever the guard is enabled. Must be a
    separate, explicitly-awaited call rather than folded into
    `BoundaryGuard.__enter__` itself (a plain sync context manager, so it
    cannot await anything). Sync-only callers (`Recorder`, and `fork.py`'s
    `ForkEngine.fork`/`fork_coalition`, which build a *sync*
    `anthropic.Anthropic` client and never run their own event loop) have no
    running loop to pre-warm and don't call this — an `agent_fn` that itself
    spins up its own event loop and constructs its OWN `AsyncAnthropic`
    inside a sync `fork()`/`fork_coalition()` call is a documented,
    UN-COVERED gap FOR Thread.start() SPECIFICALLY (a fresh loop means a
    fresh, un-warmed default executor this function cannot reach in advance,
    and fork.py has no reference to that client instance to apply the
    primary fix to either) — `time.monotonic()`'s dynamic per-call check has
    no such gap (verified empirically: it correctly covers a loop `agent_fn`
    starts only after the guard is already active, since it re-checks on
    every call rather than once at `__enter__`).

    Never raises — if `asyncio.to_thread` is unavailable or errors for any
    reason, this just silently no-ops and the guard still protects against
    genuine violations (only this specific SDK-internal false positive would
    resurface)."""
    with contextlib.suppress(Exception):
        await _asyncio_module.to_thread(lambda: None)


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
        orig_monotonic = self._orig_monotonic

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
            # Checked dynamically, PER CALL (not once at __enter__): inside a
            # live asyncio event loop — at the moment of THIS call, whether or
            # not one was running when __enter__ ran — this is asyncio's own
            # scheduling (`BaseEventLoop._run_once` calls `time.monotonic()`
            # unconditionally on every iteration it must genuinely wait), not
            # agent code; let it through rather than false-positive, the same
            # "false positive, not a real signal" reasoning the module
            # docstring's `time.time()` exclusion already documents.
            # `get_running_loop()` is a cheap, O(1) context lookup, so paying
            # it on every call is not a real cost. Outside a loop (the sync
            # `Recorder`/`fork()` path, or a sync `agent_fn` that never
            # touches asyncio) this always raises RuntimeError and the guard
            # stays fully active, unchanged.
            try:
                _asyncio_module.get_running_loop()
            except RuntimeError:
                raise BoundaryViolationError(
                    "time.monotonic() called directly while BoundaryGuard is "
                    "active: route clock reads through NondetSource.now_iso() "
                    "so they are captured and replayed bit-exact (see "
                    "nondet.py)."
                ) from None
            return float(orig_monotonic())

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
                            "boundary_guard.py).",
                            violation_kind="write",
                            attempted=str(target),
                            declared_writable_roots=confinement.writable_roots,
                        )
                return self._orig_open(file, mode, *a, **kw)

            def _guarded_socket_connect(sock: Any, address: Any, *a: Any, **kw: Any) -> Any:
                host = address[0] if isinstance(address, tuple) else address
                if isinstance(host, str) and host not in allowed_hosts:
                    raise ConfinementViolationError(
                        f"socket.connect({address!r}) denied: host is outside the "
                        "declared ConfinementSpec.allowed_hosts while BoundaryGuard "
                        "confinement is active (see boundary_guard.py).",
                        violation_kind="connect",
                        attempted=host,
                        declared_allowed_hosts=confinement.allowed_hosts,
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
