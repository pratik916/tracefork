"""Recorder context manager tests — sync and async."""

import base64
import json
import uuid as _uuid

import anthropic
import httpx
import pytest

from tests.fakes import (
    AsyncScriptedFakeLLM,
    ScriptedFakeLLM,
    make_text_response,
    make_tool_use_response,
)
from tracefork import AsyncRecorder, Recorder
from tracefork.redact import safe_defaults

TOOL_RESP = make_tool_use_response("book_flight", {"destination": "Tokyo", "seats": 1})
TEXT_RESP = make_text_response("Done — flight booked.")


def _sync_client(fake: ScriptedFakeLLM) -> anthropic.Anthropic:
    return anthropic.Anthropic(
        api_key="sk-ant-fake",
        http_client=httpx.Client(transport=fake),
        max_retries=0,
    )


def _async_client(fake: AsyncScriptedFakeLLM) -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic(
        api_key="sk-ant-fake",
        http_client=httpx.AsyncClient(transport=fake),
        max_retries=0,
    )


def test_recorder_captures_single_turn():
    fake = ScriptedFakeLLM([TEXT_RESP])
    client = _sync_client(fake)
    with Recorder(client, agent_name="test") as rec:
        rec.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=100,
            messages=[{"role": "user", "content": "Hello"}],
        )
    tape = rec.tape
    assert len(tape.exchanges) == 1
    assert tape.agent_name == "test"
    assert tape.exchanges[0][1] == TEXT_RESP


def test_recorder_captures_two_turns():
    fake = ScriptedFakeLLM([TOOL_RESP, TEXT_RESP])
    client = _sync_client(fake)
    with Recorder(client) as rec:
        rec.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=100,
            messages=[{"role": "user", "content": "Book a flight"}],
        )
        rec.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=100,
            messages=[{"role": "user", "content": "Confirm"}],
        )
    assert len(rec.tape.exchanges) == 2


def test_recorder_patches_uuid4():
    """uuid.uuid4() is intercepted and agent-generated UUIDs appear in draws.

    The Anthropic SDK also calls uuid.uuid4() internally (e.g. for request IDs),
    so we verify the agent's UUID is present rather than asserting exactly one draw.
    """
    fake = ScriptedFakeLLM([TEXT_RESP])
    client = _sync_client(fake)
    ids_generated = []
    with Recorder(client) as rec:
        uid = _uuid.uuid4()
        ids_generated.append(uid)
        rec.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=100,
            messages=[{"role": "user", "content": "hi"}],
        )
    uuid_draws = [v for k, v in rec.tape.draws if k == "uuid"]
    assert len(uuid_draws) >= 1
    assert ids_generated[0].hex in uuid_draws


def test_recorder_restores_uuid4_after_exit():
    """uuid.uuid4 is restored after the context exits."""
    orig_uuid4 = _uuid.uuid4
    fake = ScriptedFakeLLM([TEXT_RESP])
    client = _sync_client(fake)
    with Recorder(client):
        pass
    assert _uuid.uuid4 is orig_uuid4


def test_recorder_tape_digest_is_stable():
    fake = ScriptedFakeLLM([TEXT_RESP])
    client = _sync_client(fake)
    with Recorder(client) as rec:
        rec.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=100,
            messages=[{"role": "user", "content": "hi"}],
        )
    d = rec.tape.digest()
    assert d == rec.tape.digest()  # deterministic


# ── redactor wiring reaches RecordingNondet's get_env/read_file draws too ──
# Regression: redact.py's own module docstring promises "there is no knob to
# keep a live secret on a tape", but nondet draws bypassed the Redactor
# entirely -- a secret read via NondetSource.get_env() landed verbatim on
# tape.to_bytes() even with safe_defaults() active. See nondet.py/redact.py.


def test_recorder_wires_redactor_into_recording_nondet_get_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-super-secret-value")
    fake = ScriptedFakeLLM([TEXT_RESP])
    client = _sync_client(fake)
    redactor = safe_defaults()
    with Recorder(client, redactor=redactor) as rec:
        v = rec._nondet.get_env("ANTHROPIC_API_KEY")
        rec.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=100,
            messages=[{"role": "user", "content": "hi"}],
        )
    # The live agent still gets the real value to actually use.
    assert v == "sk-ant-super-secret-value"
    # But it never lands on the tape -- neither in draws nor in the
    # serialized bytes a real `tape.to_bytes()`/save would persist.
    env_draws = [val for kind, val in rec.tape.draws if kind == "env"]
    assert env_draws == ["1\0ANTHROPIC_API_KEY\0REDACTED"]
    assert b"sk-ant-super-secret-value" not in rec.tape.to_bytes()


def test_recorder_wires_redactor_into_recording_nondet_read_file(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-super-secret-value")
    p = tmp_path / "creds.txt"
    p.write_bytes(b"ANTHROPIC_API_KEY=sk-ant-super-secret-value\n")
    fake = ScriptedFakeLLM([TEXT_RESP])
    client = _sync_client(fake)
    redactor = safe_defaults()
    with Recorder(client, redactor=redactor) as rec:
        data = rec._nondet.read_file(str(p))
        rec.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=100,
            messages=[{"role": "user", "content": "hi"}],
        )
    # The live agent still gets the real bytes.
    assert data == b"ANTHROPIC_API_KEY=sk-ant-super-secret-value\n"
    # But the STORED envelope's content_b64 -- what actually survives onto
    # the tape -- must not decode back to the live secret. (A raw substring
    # search over to_bytes() would false-negative here: content_b64 is
    # base64, which doesn't preserve the literal ASCII secret's byte
    # pattern, so this test decodes the real draw instead.)
    read_file_draws = [val for kind, val in rec.tape.draws if kind == "read_file"]
    assert len(read_file_draws) == 1
    stored = base64.b64decode(json.loads(read_file_draws[0])["content_b64"])
    assert b"sk-ant-super-secret-value" not in stored


def test_recorder_get_env_draw_untouched_without_redactor(monkeypatch):
    """Guarantee #1 from redact.py (default path unchanged) extended to the
    get_env channel: no redactor means the raw value is recorded exactly as
    before this parameter existed."""
    monkeypatch.setenv("TF_PLAIN_VAR", "plain-value")
    fake = ScriptedFakeLLM([TEXT_RESP])
    client = _sync_client(fake)
    with Recorder(client) as rec:
        rec._nondet.get_env("TF_PLAIN_VAR")
        rec.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=100,
            messages=[{"role": "user", "content": "hi"}],
        )
    env_draws = [val for kind, val in rec.tape.draws if kind == "env"]
    assert env_draws == ["1\0TF_PLAIN_VAR\0plain-value"]


@pytest.mark.asyncio
async def test_async_recorder_wires_redactor_into_recording_nondet_get_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-super-secret-value")
    fake = AsyncScriptedFakeLLM([TEXT_RESP])
    client = _async_client(fake)
    redactor = safe_defaults()
    async with AsyncRecorder(client, redactor=redactor) as rec:
        v = rec._nondet.get_env("ANTHROPIC_API_KEY")
        await rec.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=100,
            messages=[{"role": "user", "content": "hi"}],
        )
    assert v == "sk-ant-super-secret-value"
    env_draws = [val for kind, val in rec.tape.draws if kind == "env"]
    assert env_draws == ["1\0ANTHROPIC_API_KEY\0REDACTED"]
    assert b"sk-ant-super-secret-value" not in rec.tape.to_bytes()


@pytest.mark.asyncio
async def test_async_recorder_captures_exchange():
    fake = AsyncScriptedFakeLLM([TEXT_RESP])
    client = _async_client(fake)
    async with AsyncRecorder(client, agent_name="async-test") as rec:
        await rec.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=100,
            messages=[{"role": "user", "content": "Hello"}],
        )
    assert len(rec.tape.exchanges) == 1
    assert rec.tape.agent_name == "async-test"
