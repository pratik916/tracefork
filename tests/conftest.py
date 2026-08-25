"""Session-wide offline/$0 enforcement.

This project's loudest, most-quoted claim is that its test suite (and
`validate`, and the demo) run **fully offline, $0, no `ANTHROPIC_API_KEY`,
no network** — see `README.md` and `CLAUDE.md`. Until this file existed,
nothing actually enforced that; it was a convention documented in prose
(`tests/test_cli_smoke.py`'s own module docstring said so in so many words)
rather than a mechanism, which is exactly the kind of claim this project
elsewhere refuses to accept without proof (`tracefork validate` exists for
the identical reason, one layer up).

Two independent guards, both autouse so every test gets them for free:

1. **No real socket connect.** `socket.socket.connect`/`.connect_ex` are
   patched to raise a loud `RuntimeError` for any destination that isn't
   loopback or `AF_UNIX`. `socket.create_connection` is deliberately NOT
   patched separately — the stdlib implementation already constructs a
   `socket.socket(...)` and calls `sock.connect(sa)` on it, so patching the
   instance method alone already covers it, and patching BOTH independently
   would fight `boundary_guard.py`'s own, more specific `ConfinementSpec`
   socket guard (also a `socket.socket.connect` patch, active only for the
   duration of a confined fork's tail-record call): a blanket
   `create_connection` override would short-circuit BEFORE
   `ConfinementSpec` ever gets a chance to raise its own richer
   `ConfinementViolationError` diagnostic, since it never constructs the
   socket instance whose `.connect` boundary_guard actually patches. Loopback
   stays allowed because nothing in this suite needs it blocked (FastAPI's
   `TestClient` uses an in-process ASGI transport with no real socket at all
   — see `CLAUDE.md`'s `report.py`/`server.py` entry — but a stray
   `uvicorn.run` or a future local-server test would still want loopback to
   work). DNS resolution (`socket.getaddrinfo`) is deliberately left
   unpatched: it's the connect attempt that would actually spend money or
   leak a request, not the lookup.
2. **No real secret env vars.** `ANTHROPIC_API_KEY`/`ANTHROPIC_AUTH_TOKEN`
   (`redact.py`'s own `DEFAULT_SECRET_ENV_VARS` — the single source of
   truth, not a second hardcoded list) are deleted from the environment
   before every test via `monkeypatch`, so a real key exported in the
   shell that launched pytest can never silently reach a test. Because
   `monkeypatch` is function-scoped and shared across every fixture that
   requests it within one test, a test that deliberately needs a fake
   secret value (`tests/test_redact.py`, `tests/test_e2e.py`'s redaction
   tests) just calls `monkeypatch.setenv(...)` in its own body as normal —
   this fixture's `delenv` already ran during setup, so there is no
   ordering conflict, and `monkeypatch` unwinds both changes together at
   teardown either way.

Both guards are cheap, process-local, and change no return value any test
already asserts on — `ANTHROPIC_API_KEY=sk-ant-real uv run pytest -q` must
be byte-identical to a run with the variable unset, because the suite never
reads the real one either way.
"""

from __future__ import annotations

import socket

import pytest

from tracefork.redact import DEFAULT_SECRET_ENV_VARS

_ALLOWED_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})

_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex


class OfflineTestGuardError(RuntimeError):
    """Raised when a test attempts a real (non-loopback, non-AF_UNIX) socket
    connection. See this module's docstring."""


def _address_host(address: object) -> object:
    if isinstance(address, tuple) and address:
        return address[0]
    return address


def _is_allowed(self: socket.socket, address: object) -> bool:
    if getattr(self, "family", None) == socket.AF_UNIX:
        return True
    return _address_host(address) in _ALLOWED_HOSTS


def _guarded_connect(self: socket.socket, address):
    if _is_allowed(self, address):
        return _real_connect(self, address)
    raise OfflineTestGuardError(
        f"tracefork tests are offline by construction — blocked a real "
        f"socket connect to {address!r}. See tests/conftest.py."
    )


def _guarded_connect_ex(self: socket.socket, address):
    if _is_allowed(self, address):
        return _real_connect_ex(self, address)
    raise OfflineTestGuardError(
        f"tracefork tests are offline by construction — blocked a real "
        f"socket connect_ex to {address!r}. See tests/conftest.py."
    )


@pytest.fixture(autouse=True, scope="session")
def _block_real_network():
    socket.socket.connect = _guarded_connect
    socket.socket.connect_ex = _guarded_connect_ex
    try:
        yield
    finally:
        socket.socket.connect = _real_connect
        socket.socket.connect_ex = _real_connect_ex


@pytest.fixture(autouse=True)
def _no_real_secret_env_vars(monkeypatch):
    for key in DEFAULT_SECRET_ENV_VARS:
        monkeypatch.delenv(key, raising=False)
