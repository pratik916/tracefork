"""Recorder context manager tests — sync and async."""
import uuid as _uuid

import anthropic
import httpx
import pytest

from tracefork import Recorder, AsyncRecorder
from tracefork.tape import Tape
from tests.fakes import ScriptedFakeLLM, AsyncScriptedFakeLLM, make_tool_use_response, make_text_response

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
            model="claude-sonnet-4-6", max_tokens=100,
            messages=[{"role": "user", "content": "Book a flight"}],
        )
        rec.client.messages.create(
            model="claude-sonnet-4-6", max_tokens=100,
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
            model="claude-sonnet-4-6", max_tokens=100,
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
    with Recorder(client) as rec:
        pass
    assert _uuid.uuid4 is orig_uuid4


def test_recorder_tape_digest_is_stable():
    fake = ScriptedFakeLLM([TEXT_RESP])
    client = _sync_client(fake)
    with Recorder(client) as rec:
        rec.client.messages.create(
            model="claude-sonnet-4-6", max_tokens=100,
            messages=[{"role": "user", "content": "hi"}],
        )
    d = rec.tape.digest()
    assert d == rec.tape.digest()  # deterministic


@pytest.mark.asyncio
async def test_async_recorder_captures_exchange():
    fake = AsyncScriptedFakeLLM([TEXT_RESP])
    client = _async_client(fake)
    async with AsyncRecorder(client, agent_name="async-test") as rec:
        await rec.client.messages.create(
            model="claude-sonnet-4-6", max_tokens=100,
            messages=[{"role": "user", "content": "Hello"}],
        )
    assert len(rec.tape.exchanges) == 1
    assert rec.tape.agent_name == "async-test"
