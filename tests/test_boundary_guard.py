"""BoundaryGuard tests — opt-in, so every test here explicitly constructs one;
`Recorder`'s own default-off behavior is asserted in test_recorder.py-adjacent
cases below (the same violating agent must NOT raise without the flag)."""

from __future__ import annotations

import builtins
import random
import socket
import subprocess
import threading
import time
from unittest import mock

import anthropic
import httpx
import pytest

from tests.fakes import ScriptedFakeLLM, make_text_response
from tracefork import Recorder
from tracefork.boundary_guard import (
    BoundaryGuard,
    BoundaryViolationError,
    ConfinementSpec,
    ConfinementViolationError,
)
from tracefork.config import TraceforkConfig

TEXT_RESP = make_text_response("Done.")


def _sync_client(fake: ScriptedFakeLLM) -> anthropic.Anthropic:
    return anthropic.Anthropic(
        api_key="sk-ant-fake",
        http_client=httpx.Client(transport=fake),
        max_retries=0,
    )


# ── Direct BoundaryGuard unit tests ─────────────────────────────────────────


def test_guard_trips_on_thread_start():
    with BoundaryGuard(), pytest.raises(BoundaryViolationError, match="Thread"):
        threading.Thread(target=lambda: None).start()


def test_guard_trips_on_subprocess_popen():
    with BoundaryGuard(), pytest.raises(BoundaryViolationError, match="Popen"):
        subprocess.run(["echo", "hi"], check=False)


def test_guard_trips_on_random_random():
    with BoundaryGuard(), pytest.raises(BoundaryViolationError, match="random"):
        random.random()


def test_guard_trips_on_time_monotonic():
    with BoundaryGuard(), pytest.raises(BoundaryViolationError, match="monotonic"):
        time.monotonic()


def test_guard_trips_on_time_sleep():
    with BoundaryGuard(), pytest.raises(BoundaryViolationError, match="sleep"):
        time.sleep(0)


def test_guard_is_silent_when_not_entered():
    """Merely constructing a BoundaryGuard (without `with`) must change nothing."""
    BoundaryGuard()
    threading.Thread(target=lambda: None).start()
    assert random.random() is not None
    assert time.monotonic() > 0


def test_guard_restores_originals_on_exit():
    orig_start = threading.Thread.start
    orig_popen = subprocess.Popen.__init__
    orig_random = random.random
    orig_monotonic = time.monotonic
    orig_sleep = time.sleep

    with BoundaryGuard():
        pass

    assert threading.Thread.start is orig_start
    assert subprocess.Popen.__init__ is orig_popen
    assert random.random is orig_random
    assert time.monotonic is orig_monotonic
    assert time.sleep is orig_sleep


def test_guard_restores_originals_after_exception():
    orig_random = random.random
    with pytest.raises(BoundaryViolationError), BoundaryGuard():
        random.random()
    assert random.random is orig_random
    # guard still usable / harmless after the exception unwound it
    assert random.random() is not None


# ── Realistic recording path under the guard: no false positives ───────────


def test_guard_active_during_real_recorder_session_no_false_positive():
    """The Anthropic SDK's own `platform_headers()` helper shells out
    (uncached) to derive an `X-Stainless-OS` header, and httpx's cookie-jar
    machinery would call time.time() per response (out of the guard's scope
    for exactly this reason). BoundaryGuard.__enter__ pre-warms the SDK's
    platform-header cache so a normal recording session doesn't trip the
    subprocess guard on its own housekeeping."""
    fake = ScriptedFakeLLM([TEXT_RESP, TEXT_RESP])
    client = _sync_client(fake)
    with Recorder(client, boundary_guard=True) as rec:
        rec.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=100,
            messages=[{"role": "user", "content": "hi"}],
        )
        rec.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=100,
            messages=[{"role": "user", "content": "hi again"}],
        )
    assert len(rec.tape.exchanges) == 2


# ── Recorder wiring: opt-in, explicit-wins-over-config, default off ────────


def _violating_agent(client: anthropic.Anthropic) -> None:
    threading.Thread(target=lambda: None).start()


def test_recorder_boundary_guard_true_catches_violation():
    fake = ScriptedFakeLLM([TEXT_RESP])
    client = _sync_client(fake)
    with pytest.raises(BoundaryViolationError), Recorder(client, boundary_guard=True) as rec:
        _violating_agent(rec.client)


def test_recorder_default_off_does_not_catch_violation():
    """Zero default behavior change: without opting in, a thread-spawning
    agent records successfully, exactly as before this feature existed."""
    fake = ScriptedFakeLLM([TEXT_RESP])
    client = _sync_client(fake)
    with Recorder(client) as rec:
        _violating_agent(rec.client)
    # No exception raised — default behavior unchanged.


def test_recorder_boundary_guard_false_overrides_config_true():
    fake = ScriptedFakeLLM([TEXT_RESP])
    client = _sync_client(fake)
    cfg = TraceforkConfig(boundary_guard=True)
    with Recorder(client, config=cfg, boundary_guard=False) as rec:
        _violating_agent(rec.client)
    # explicit False wins over config's True — no exception.


def test_recorder_config_boundary_guard_true_is_honored():
    fake = ScriptedFakeLLM([TEXT_RESP])
    client = _sync_client(fake)
    cfg = TraceforkConfig(boundary_guard=True)
    with pytest.raises(BoundaryViolationError), Recorder(client, config=cfg) as rec:
        _violating_agent(rec.client)


def test_recorder_restores_uuid4_even_after_boundary_violation():
    """__exit__ must still restore uuid.uuid4 when the guard raises inside the
    `with` block's body (i.e. the exception propagates through __exit__)."""
    import uuid as _uuid

    orig_uuid4 = _uuid.uuid4
    fake = ScriptedFakeLLM([TEXT_RESP])
    client = _sync_client(fake)
    with pytest.raises(BoundaryViolationError), Recorder(client, boundary_guard=True) as rec:
        _violating_agent(rec.client)
    assert _uuid.uuid4 is orig_uuid4


# ── TraceforkConfig wiring ───────────────────────────────────────────────


def test_config_boundary_guard_defaults_false():
    assert TraceforkConfig().boundary_guard is False


def test_config_from_env_boundary_guard(monkeypatch):
    monkeypatch.setenv("TRACEFORK_BOUNDARY_GUARD", "true")
    cfg = TraceforkConfig.from_env()
    assert cfg.boundary_guard is True


def test_config_from_env_boundary_guard_unset_matches_default(monkeypatch):
    monkeypatch.delenv("TRACEFORK_BOUNDARY_GUARD", raising=False)
    assert TraceforkConfig.from_env().boundary_guard == TraceforkConfig().boundary_guard


# ── ConfinementSpec-lite: writable-roots + network policy (tracefork-bge.17) ─


def test_confinement_none_leaves_open_and_socket_unpatched():
    """The default (`confinement=None`) must leave `builtins.open` and
    `socket.socket.connect` byte-identical to pre-bead behavior."""
    orig_open = builtins.open
    orig_connect = socket.socket.connect
    with BoundaryGuard():
        assert builtins.open is orig_open
        assert socket.socket.connect is orig_connect
    assert builtins.open is orig_open
    assert socket.socket.connect is orig_connect


def test_confinement_blocks_write_outside_writable_roots_allows_inside(tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside_file = tmp_path / "outside.txt"
    inside_file = allowed / "inside.txt"
    spec = ConfinementSpec(writable_roots=(str(allowed),))

    with BoundaryGuard(confinement=spec):
        with (
            pytest.raises(ConfinementViolationError, match="writable_roots"),
            open(outside_file, "w"),
        ):
            pass
        with open(inside_file, "w") as f:
            f.write("ok")

    assert inside_file.read_text() == "ok"
    assert not outside_file.exists()


def test_confinement_allows_reads_regardless_of_writable_roots(tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    readable = tmp_path / "readable.txt"
    readable.write_text("hello")
    spec = ConfinementSpec(writable_roots=(str(allowed),))

    with BoundaryGuard(confinement=spec), open(readable) as f:
        assert f.read() == "hello"


def test_confinement_blocks_connect_to_disallowed_host_allows_configured_host():
    """No test here ever makes a real DNS/TCP attempt: the disallowed-host
    case must raise BEFORE the underlying connect is invoked at all, and the
    allowed-host case is routed to a locally-patched fake connect (never the
    real one) so this stays offline/$0."""
    calls: list[object] = []

    def _fake_connect(_self: socket.socket, address: object) -> None:
        calls.append(address)

    spec = ConfinementSpec(allowed_hosts=("allowed.example",))
    with (
        mock.patch.object(socket.socket, "connect", _fake_connect),
        BoundaryGuard(confinement=spec),
    ):
        with pytest.raises(ConfinementViolationError, match="allowed_hosts"):
            socket.socket().connect(("disallowed.example", 80))
        socket.socket().connect(("allowed.example", 80))
    assert calls == [("allowed.example", 80)]


def test_confinement_restores_open_and_socket_connect_on_exit():
    orig_open = builtins.open
    orig_connect = socket.socket.connect
    with BoundaryGuard(confinement=ConfinementSpec()):
        pass
    assert builtins.open is orig_open
    assert socket.socket.connect is orig_connect


def test_confinement_violation_error_is_a_boundary_violation_error():
    """Existing `pytest.raises(BoundaryViolationError, ...)` patterns must
    keep working against the new, more specific error."""
    assert issubclass(ConfinementViolationError, BoundaryViolationError)
