"""Self-test for `tests/conftest.py`'s offline/$0 enforcement — proves the
guard actually fires rather than trusting the autouse fixture silently.
"""

from __future__ import annotations

import os
import socket

import pytest

from tests.conftest import OfflineTestGuardError
from tracefork.redact import DEFAULT_SECRET_ENV_VARS


def test_real_socket_connect_is_blocked():
    """A connect attempt to a real (non-loopback) address must raise the
    guard's own error, never actually reach the network."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(OfflineTestGuardError):
            s.connect(("example.com", 80))
    finally:
        s.close()


def test_loopback_connect_is_still_allowed():
    """The guard must not break loopback — nothing in this suite needs it
    blocked, and a future local-server test might legitimately need it."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]

        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            client.settimeout(2)
            client.connect(("127.0.0.1", port))  # must not raise
        finally:
            client.close()
    finally:
        listener.close()


def test_real_create_connection_is_blocked():
    """`socket.create_connection` isn't patched separately — it's blocked
    transitively, because the stdlib implementation itself calls the
    (patched) `socket.socket.connect` on the instance it creates. See this
    module's `_block_real_network` docstring for why a second, independent
    patch here would fight `boundary_guard.py`'s own guard."""
    with pytest.raises(OfflineTestGuardError):
        socket.create_connection(("example.com", 443), timeout=1)


@pytest.mark.parametrize("key", sorted(DEFAULT_SECRET_ENV_VARS))
def test_secret_env_vars_are_absent_by_default(key):
    """`ANTHROPIC_API_KEY`/`ANTHROPIC_AUTH_TOKEN` must never leak in from the
    shell that launched pytest — this is what makes
    `ANTHROPIC_API_KEY=sk-ant-real uv run pytest -q` byte-identical to an
    unset run."""
    assert os.environ.get(key) is None


def test_a_test_can_still_set_a_fake_secret_via_monkeypatch(monkeypatch):
    """The autouse `delenv` must not fight a test's own deliberate
    `monkeypatch.setenv` (the pattern `tests/test_redact.py` and
    `tests/test_e2e.py`'s redaction tests already use) — same `monkeypatch`
    instance, setup-then-body ordering, no conflict."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake-for-this-test-only")
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-fake-for-this-test-only"
