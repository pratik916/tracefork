"""OpenAI Agents SDK adapter — offline, no framework installed.

The framework-facing thin wrapper (``TracingProcessor`` subclass) needs the
real ``agents`` package and is covered by a ``pytest.importorskip`` block that
skips cleanly when absent. Everything that must work with NO framework
installed — the injection into a model wrapper, the replay flow through the
bound httpx transport, the neutral tracing core, the event dispatch, and the
availability guards — is driven here with synthetic objects.
"""

import uuid

import pytest

from tracefork.adapters.openai_agents import (
    OpenAIAgentsAdapter,
    TraceforkTracingCore,
    bind_default_client,
    make_tracing_processor,
    openai_agents_available,
    require_openai_agents,
)
from tracefork.nondet import DivergenceError, ReplayNondet
from tracefork.tape import Tape
from tracefork.transport import TraceforkTransport

# ── fake model wrapper (mimics an Agents SDK Model wrapper's client shape) ──────


class _FakeAsyncOpenAIClient:
    """Mimics ``openai.AsyncOpenAI``: ``.copy(http_client=)``."""

    def __init__(self, http_client=None):
        self.http_client = http_client

    def copy(self, *, http_client=None):
        return _FakeAsyncOpenAIClient(http_client=http_client)


class _FakeOpenAIChatCompletionsModel:
    """Plain stand-in for ``agents.OpenAIChatCompletionsModel``.

    Stores its client under ``client`` — one of ``bind``'s candidate attribute
    names — proving the defensive search finds it without hard-coding a single
    "true" attribute name (the real SDK does not document one)."""

    def __init__(self):
        self.client = _FakeAsyncOpenAIClient()


# ── bind: injection into the underlying client ──────────────────────────────────


def test_bind_injects_client_attribute():
    tape = Tape()
    model = _FakeOpenAIChatCompletionsModel()
    adapter = OpenAIAgentsAdapter()
    result = adapter.bind(model, tape, mode="replay", patch_uuid=False)
    try:
        assert result.injected_fields == ("client",)
        assert model.client.http_client is result.http_async_client
        assert result.notes == ""
        assert isinstance(result.transport, TraceforkTransport)
    finally:
        adapter.teardown()


def test_bind_unknown_target_reports_notes():
    adapter = OpenAIAgentsAdapter()
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
    model = _FakeOpenAIChatCompletionsModel()
    adapter = OpenAIAgentsAdapter()
    result = adapter.bind(model, tape, mode="replay", patch_uuid=False)
    try:
        client = model.client.http_client
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
    model = _FakeOpenAIChatCompletionsModel()
    adapter = OpenAIAgentsAdapter()
    result = adapter.bind(model, tape, mode="replay", patch_uuid=False)
    try:
        with pytest.raises(DivergenceError):
            await result.http_async_client.post("https://api.openai.com/v1/x", content=b"DIFFERENT")
    finally:
        adapter.teardown()


def test_bind_replay_installs_uuid_patch_and_teardown_restores():
    tape = Tape()
    tape.draws = [("uuid", "0" * 32), ("uuid", "1" * 32)]
    model = _FakeOpenAIChatCompletionsModel()
    adapter = OpenAIAgentsAdapter()
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
    model = _FakeOpenAIChatCompletionsModel()
    adapter = OpenAIAgentsAdapter()
    result = adapter.bind(model, tape, mode="replay", nondet=supplied)
    try:
        assert result.nondet is supplied
        assert uuid.uuid4().hex == "f" * 32
    finally:
        adapter.teardown()


# ── on_step: tracing event -> Step, building the DAG ─────────────────────────────


def test_on_step_builds_dag_across_events():
    adapter = OpenAIAgentsAdapter()
    adapter.on_step({"event": "on_trace_start", "trace_id": "trace1", "name": "workflow"})
    adapter.on_step(
        {
            "event": "on_span_start",
            "span_id": "span1",
            "parent_id": "trace1",
            "span_data": {"type": "generation", "model": "gpt-4o"},
        }
    )
    adapter.on_step({"event": "on_span_end", "span_id": "span1", "span_data": {"output": "hello"}})
    adapter.on_step({"event": "on_trace_end", "trace_id": "trace1"})

    dag = adapter.dag
    assert [s.step_id for s in dag.steps] == ["trace1", "span1"]
    span = dag.by_id("span1")
    assert span.kind == "llm"
    assert span.parent_id == "trace1"
    assert span.model == "gpt-4o"
    assert span.outputs == "hello"
    assert [s.step_id for s in dag.llm_steps()] == ["span1"]


def test_on_step_non_llm_span_keeps_its_type_as_kind():
    adapter = OpenAIAgentsAdapter()
    step = adapter.on_step(
        {
            "event": "on_span_start",
            "span_id": "s1",
            "span_data": {"type": "function", "name": "search"},
        }
    )
    assert step.kind == "function"
    assert step.name == "search"


def test_on_step_span_falls_back_to_trace_id_when_no_parent():
    adapter = OpenAIAgentsAdapter()
    step = adapter.on_step({"event": "on_span_start", "span_id": "s1", "trace_id": "t1"})
    assert step.parent_id == "t1"


def test_on_step_unknown_event_is_not_dropped():
    adapter = OpenAIAgentsAdapter()
    step = adapter.on_step({"event": "on_custom_thing", "id": "x1"})
    assert step.step_id == "x1"
    assert step.kind == "on_custom_thing"


def test_on_step_end_event_with_no_matching_start_returns_placeholder():
    adapter = OpenAIAgentsAdapter()
    step = adapter.on_step({"event": "on_span_end", "span_id": "ghost"})
    assert step.step_id == "ghost"


# ── TraceforkTracingCore (direct) ────────────────────────────────────────────────


def test_tracing_core_span_end_extracts_output_and_falls_back_to_response():
    core = TraceforkTracingCore()
    core.on_span_start({"span_id": "s", "span_data": {"type": "response", "model": "gpt-4o-mini"}})
    core.on_span_end({"span_id": "s", "span_data": {"response": "from-response"}})
    step = core.dag.by_id("s")
    assert step.outputs == "from-response"
    assert step.model == "gpt-4o-mini"


def test_tracing_core_trace_end_without_start_returns_none():
    core = TraceforkTracingCore()
    assert core.on_trace_end({"trace_id": "missing"}) is None


# ── availability guards ─────────────────────────────────────────────────────────


def test_require_openai_agents_matches_availability():
    if openai_agents_available():
        require_openai_agents()
    else:
        with pytest.raises(ImportError, match="openai-agents") as excinfo:
            require_openai_agents()
        assert excinfo.value.__cause__ is not None


def test_make_tracing_processor_guarded_or_builds():
    if not openai_agents_available():
        with pytest.raises(ImportError, match="openai-agents"):
            make_tracing_processor()
    else:  # pragma: no cover - only when openai-agents is installed
        processor = make_tracing_processor()
        assert hasattr(processor, "dag")


def test_bind_default_client_guarded_or_builds():
    if not openai_agents_available():
        with pytest.raises(ImportError, match="openai-agents"):
            bind_default_client(Tape(), mode="replay")
    else:  # pragma: no cover - only when openai-agents is installed
        result = bind_default_client(Tape(), mode="replay")
        assert result.injected_fields == ("default_openai_client",)


# ── real-framework smoke (skipped cleanly when the framework is absent) ─────────


def test_tracing_processor_forwards_to_core_with_real_sdk():
    pytest.importorskip("agents")  # pragma: no cover - needs openai-agents extra
    processor = make_tracing_processor()  # pragma: no cover
    processor.on_trace_start({"trace_id": "t1", "name": "wf"})  # pragma: no cover
    assert len(processor.dag) == 1  # pragma: no cover
