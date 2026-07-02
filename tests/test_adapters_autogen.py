"""AutoGen (autogen-core / autogen-ext) adapter — offline, no framework installed.

The framework-facing thin wrapper (``InterventionHandler`` subclass) needs the
real ``autogen_core`` package and is covered by a ``pytest.importorskip`` block
that skips cleanly when absent. Everything that must work with NO framework
installed — the injection into a model client, the replay flow through the
bound httpx transport, the neutral intervention core, the event dispatch, and
the availability guards — is driven here with synthetic objects.
"""

import uuid

import pytest

from tracefork.adapters.autogen import (
    AutoGenAdapter,
    TraceforkInterventionCore,
    autogen_available,
    make_intervention_handler,
    require_autogen,
)
from tracefork.nondet import DivergenceError, ReplayNondet
from tracefork.tape import Tape
from tracefork.transport import TraceforkTransport

# ── fake model client (mimics OpenAIChatCompletionClient's client shape) ────────


class _FakeAsyncOpenAIClient:
    """Mimics ``openai.AsyncOpenAI``: ``.copy(http_client=)``."""

    def __init__(self, http_client=None):
        self.http_client = http_client

    def copy(self, *, http_client=None):
        return _FakeAsyncOpenAIClient(http_client=http_client)


class _FakeOpenAIChatCompletionClient:
    """Plain stand-in for ``autogen_ext.models.openai.OpenAIChatCompletionClient``.

    Stores its client under ``_client`` — one of ``bind``'s candidate attribute
    names, matching the base class's ``_client`` constructor parameter name."""

    def __init__(self):
        self._client = _FakeAsyncOpenAIClient()


# ── bind: injection into the underlying client ──────────────────────────────────


def test_bind_injects_client_attribute():
    tape = Tape()
    model = _FakeOpenAIChatCompletionClient()
    adapter = AutoGenAdapter()
    result = adapter.bind(model, tape, mode="replay", patch_uuid=False)
    try:
        assert result.injected_fields == ("_client",)
        assert model._client.http_client is result.http_async_client
        assert result.notes == ""
        assert isinstance(result.transport, TraceforkTransport)
    finally:
        adapter.teardown()


def test_bind_unknown_target_reports_notes():
    adapter = AutoGenAdapter()
    result = adapter.bind(object(), Tape(), mode="replay", patch_uuid=False)
    try:
        assert result.injected_fields == ()
        assert "nothing was injected" in result.notes
    finally:
        adapter.teardown()


async def test_bind_replay_serves_recorded_bytes_bit_exact():
    """The marquee: a run bound in replay mode serves tape bytes for $0, and a
    request that diverges from the tape is caught (proof, not assertion)."""
    tape = Tape()
    tape.append_exchange(b'{"model":"gpt-4o","messages":[]}', b'{"ok":true}')
    model = _FakeOpenAIChatCompletionClient()
    adapter = AutoGenAdapter()
    result = adapter.bind(model, tape, mode="replay", patch_uuid=False)
    try:
        client = model._client.http_client
        assert client is result.http_async_client
        resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            content=b'{"model":"gpt-4o","messages":[]}',
        )
        assert resp.status_code == 200
        assert resp.content == b'{"ok":true}'
    finally:
        adapter.teardown()


async def test_bind_replay_divergence_on_mismatched_request():
    tape = Tape()
    tape.append_exchange(b"RECORDED", b"RESP")
    model = _FakeOpenAIChatCompletionClient()
    adapter = AutoGenAdapter()
    result = adapter.bind(model, tape, mode="replay", patch_uuid=False)
    try:
        with pytest.raises(DivergenceError):
            await result.http_async_client.post("https://api.openai.com/v1/x", content=b"DIFFERENT")
    finally:
        adapter.teardown()


def test_bind_replay_installs_uuid_patch_and_teardown_restores():
    tape = Tape()
    tape.draws = [("uuid", "0" * 32), ("uuid", "1" * 32)]
    model = _FakeOpenAIChatCompletionClient()
    adapter = AutoGenAdapter()
    result = adapter.bind(model, tape, mode="replay")  # patch_uuid defaults True
    try:
        assert isinstance(result.nondet, ReplayNondet)
        assert uuid.uuid4().hex == "0" * 32
        assert uuid.uuid4().hex == "1" * 32
    finally:
        adapter.teardown()
    assert isinstance(uuid.uuid4(), uuid.UUID)  # real randomness restored


def test_bind_provided_nondet_is_used():
    tape = Tape()
    supplied = ReplayNondet([("uuid", "f" * 32)])
    model = _FakeOpenAIChatCompletionClient()
    adapter = AutoGenAdapter()
    result = adapter.bind(model, tape, mode="replay", nondet=supplied)
    try:
        assert result.nondet is supplied
        assert uuid.uuid4().hex == "f" * 32
    finally:
        adapter.teardown()


# ── on_step: intervention event -> Step, building the DAG ────────────────────────


def test_on_step_builds_dag_across_events():
    adapter = AutoGenAdapter()
    adapter.on_step({"event": "on_send", "id": "m1", "name": "agent/default", "message": "hi"})
    step = adapter.on_step(
        {"event": "on_response", "id": "m2", "name": "agent/default", "message": "hi back"}
    )

    dag = adapter.dag
    assert [s.step_id for s in dag.steps] == ["m1", "m2"]
    send_step = dag.by_id("m1")
    assert send_step.kind == "send"
    assert send_step.inputs == "hi"
    assert step.kind == "response"
    assert step.outputs == "hi back"


def test_on_step_publish_event():
    adapter = AutoGenAdapter()
    step = adapter.on_step({"event": "on_publish", "id": "p1", "message": {"topic": "news"}})
    assert step.kind == "publish"
    assert step.inputs == {"topic": "news"}


def test_on_step_unknown_event_is_not_dropped():
    adapter = AutoGenAdapter()
    step = adapter.on_step({"event": "on_custom", "id": "x1"})
    assert step.step_id == "x1"
    assert step.kind == "on_custom"


def test_on_step_synthesizes_id_when_missing():
    adapter = AutoGenAdapter()
    step = adapter.on_step({"event": "on_send", "message": "hi"})
    assert step.step_id  # non-empty, derived from id(event)


# ── TraceforkInterventionCore (direct) ───────────────────────────────────────────


def test_intervention_core_record_sets_fields():
    core = TraceforkInterventionCore()
    core.record("send", "s1", parent_id=None, name="agent/x", inputs="payload")
    step = core.dag.by_id("s1")
    assert step.kind == "send"
    assert step.name == "agent/x"
    assert step.inputs == "payload"


# ── availability guards ─────────────────────────────────────────────────────────


def test_require_autogen_matches_availability():
    if autogen_available():
        require_autogen()
    else:
        with pytest.raises(ImportError, match="autogen") as excinfo:
            require_autogen()
        assert excinfo.value.__cause__ is not None


def test_make_intervention_handler_guarded_or_builds():
    if not autogen_available():
        with pytest.raises(ImportError, match="autogen"):
            make_intervention_handler()
    else:  # pragma: no cover - only when autogen-core is installed
        handler = make_intervention_handler()
        assert hasattr(handler, "dag")


# ── real-framework smoke (skipped cleanly when the framework is absent) ─────────


async def test_intervention_handler_forwards_to_core_with_real_autogen():
    pytest.importorskip("autogen_core")  # pragma: no cover - needs autogen extra
    from autogen_core import AgentId  # pragma: no cover

    handler = make_intervention_handler()  # pragma: no cover
    recipient = AgentId("agent", "default")  # pragma: no cover
    kwargs = {"message_context": None, "recipient": recipient}  # pragma: no cover
    result = await handler.on_send("hi", **kwargs)  # pragma: no cover
    assert result == "hi"  # pragma: no cover
    assert len(handler.dag) == 1  # pragma: no cover
