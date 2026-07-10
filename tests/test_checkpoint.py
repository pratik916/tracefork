"""CheckpointWriter / recover_checkpoint tests: crash-safe incremental recording.

Scope (see checkpoint.py's module docstring): exchanges only, not draws/nondet
— a narrower-than-ideal but honest boundary rather than a silent gap.
"""

import httpx
import pytest

from tests.fakes import AsyncScriptedFakeLLM, ScriptedFakeLLM, make_text_response
from tracefork import AsyncRecorder, Recorder
from tracefork.checkpoint import CheckpointWriter, recover_checkpoint
from tracefork.tape import Tape

TEXT_RESP = make_text_response("hi")


def test_append_without_finalize_recovers_prefix_not_finalized(tmp_path):
    path = str(tmp_path / "ckpt.db")
    writer = CheckpointWriter(path, agent_name="agent-a")
    writer.append_exchange(b"req-1", b"resp-1")
    writer.append_exchange(b"req-2", b"resp-2")
    writer.append_exchange(b"req-3", b"resp-3")

    tape, was_finalized = recover_checkpoint(path)
    assert was_finalized is False
    assert len(tape.exchanges) == 3
    assert tape.exchanges == [
        (b"req-1", b"resp-1"),
        (b"req-2", b"resp-2"),
        (b"req-3", b"resp-3"),
    ]
    assert tape.agent_name == "agent-a"


def test_finalize_marks_finalized_and_digest_matches_original(tmp_path):
    path = str(tmp_path / "ckpt.db")
    writer = CheckpointWriter(path, agent_name="agent-b")
    original = Tape(agent_name="agent-b")
    for req, resp in [(b"req-1", b"resp-1"), (b"req-2", b"resp-2")]:
        writer.append_exchange(req, resp)
        original.append_exchange(req, resp)

    writer.finalize(original)

    tape, was_finalized = recover_checkpoint(path)
    assert was_finalized is True
    assert tape.digest() == original.digest()


def test_recovered_nonfinalized_digest_matches_manual_build(tmp_path):
    path = str(tmp_path / "ckpt.db")
    writer = CheckpointWriter(path)
    pairs = [(b"a", b"1"), (b"b", b"2"), (b"c", b"3")]
    for req, resp in pairs:
        writer.append_exchange(req, resp)

    tape, was_finalized = recover_checkpoint(path)
    assert was_finalized is False

    manual = Tape()
    for req, resp in pairs:
        manual.append_exchange(req, resp)
    assert tape.digest() == manual.digest()


def test_recover_checkpoint_missing_file_raises(tmp_path):
    path = str(tmp_path / "does-not-exist.db")
    with pytest.raises(FileNotFoundError):
        recover_checkpoint(path)


def _sync_client(fake: ScriptedFakeLLM) -> "httpx.Client":
    import anthropic

    return anthropic.Anthropic(
        api_key="sk-ant-fake",
        http_client=httpx.Client(transport=fake),
        max_retries=0,
    )


def test_recorder_checkpoint_path_durable_before_exit(tmp_path):
    """Every create() call durably commits to the checkpoint file before the
    with-block exits — the crash-safety property under test."""
    ckpt_path = str(tmp_path / "live.db")
    fake = ScriptedFakeLLM([TEXT_RESP, TEXT_RESP, TEXT_RESP])
    client = _sync_client(fake)

    seen_counts = []
    with Recorder(client, agent_name="ckpt-test", checkpoint_path=ckpt_path) as rec:
        for _ in range(3):
            rec.client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=100,
                messages=[{"role": "user", "content": "hi"}],
            )
            mid_tape, mid_finalized = recover_checkpoint(ckpt_path)
            seen_counts.append((len(mid_tape.exchanges), mid_finalized))

    # After each create() call (before the with-block exits), the checkpoint
    # file already durably has that many exchanges and is not yet finalized.
    assert seen_counts == [(1, False), (2, False), (3, False)]

    # After clean __exit__, the checkpoint is finalized and matches the tape.
    final_tape, was_finalized = recover_checkpoint(ckpt_path)
    assert was_finalized is True
    assert final_tape.digest() == rec.tape.digest()


def test_recorder_without_checkpoint_path_is_unaffected(tmp_path):
    """checkpoint_path defaults to None: zero-kwarg call sites are unchanged."""
    fake = ScriptedFakeLLM([TEXT_RESP])
    client = _sync_client(fake)
    with Recorder(client, agent_name="no-ckpt") as rec:
        rec.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=100,
            messages=[{"role": "user", "content": "hi"}],
        )
    assert len(rec.tape.exchanges) == 1


def _async_client(fake: AsyncScriptedFakeLLM) -> "httpx.AsyncClient":
    import anthropic

    return anthropic.AsyncAnthropic(
        api_key="sk-ant-fake",
        http_client=httpx.AsyncClient(transport=fake),
        max_retries=0,
    )


@pytest.mark.asyncio
async def test_async_recorder_checkpoint_path_durable_before_exit(tmp_path):
    ckpt_path = str(tmp_path / "async-live.db")
    fake = AsyncScriptedFakeLLM([TEXT_RESP, TEXT_RESP])
    client = _async_client(fake)

    async with AsyncRecorder(client, agent_name="async-ckpt", checkpoint_path=ckpt_path) as rec:
        await rec.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=100,
            messages=[{"role": "user", "content": "hi"}],
        )
        mid_tape, mid_finalized = recover_checkpoint(ckpt_path)
        assert (len(mid_tape.exchanges), mid_finalized) == (1, False)

    final_tape, was_finalized = recover_checkpoint(ckpt_path)
    assert was_finalized is True
    assert final_tape.digest() == rec.tape.digest()
