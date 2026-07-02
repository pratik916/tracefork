"""CrewAI adapter — offline, no framework installed.

The framework-facing thin wrapper (``BaseEventListener`` subclass) needs the
real ``crewai`` package and is covered by a ``pytest.importorskip`` block that
skips cleanly when absent. Everything that must work with NO framework
installed — the injection into the litellm httpx chokepoint, the replay flow
through the bound httpx transport, the neutral event core, the event dispatch,
and the availability guards — is driven here with synthetic objects (``bind``
never imports ``litellm`` or ``crewai`` at all; it is duck-typed).
"""

import types
import uuid

import pytest

from tracefork.adapters.crewai import (
    CrewAIAdapter,
    TraceforkCrewEventCore,
    crewai_available,
    make_event_listener,
    require_crewai,
)
from tracefork.nondet import DivergenceError, ReplayNondet
from tracefork.tape import Tape
from tracefork.transport import TraceforkTransport

# ── fake litellm module (mimics the two top-level callables + client attrs) ─────


def _fake_litellm_module():
    """Mimics the ``litellm`` module's documented custom-httpx-client surface:
    ``completion``/``acompletion`` (used only as a "this looks like litellm"
    signal) plus freely assignable ``client_session``/``aclient_session``."""
    return types.SimpleNamespace(completion=lambda **kw: None, acompletion=lambda **kw: None)


# ── bind: injection into the litellm httpx chokepoint ───────────────────────────


def test_bind_injects_client_session_attrs():
    tape = Tape()
    litellm = _fake_litellm_module()
    adapter = CrewAIAdapter()
    result = adapter.bind(litellm, tape, mode="replay", patch_uuid=False)
    try:
        assert set(result.injected_fields) == {"client_session", "aclient_session"}
        assert litellm.client_session is result.http_client
        assert litellm.aclient_session is result.http_async_client
        assert result.notes == ""
        assert isinstance(result.transport, TraceforkTransport)
    finally:
        adapter.teardown()


def test_bind_unknown_target_reports_notes():
    adapter = CrewAIAdapter()
    result = adapter.bind(object(), Tape(), mode="replay", patch_uuid=False)
    try:
        assert result.injected_fields == ()
        assert "nothing was injected" in result.notes
    finally:
        adapter.teardown()


def test_bind_replay_serves_recorded_bytes_bit_exact():
    """The marquee: a run bound in replay mode serves tape bytes for $0, and a
    request that diverges from the tape is caught (proof, not assertion)."""
    tape = Tape()
    tape.append_exchange(b'{"model":"gpt-4o","messages":[]}', b'{"ok":true}')
    litellm = _fake_litellm_module()
    adapter = CrewAIAdapter()
    adapter.bind(litellm, tape, mode="replay", patch_uuid=False)
    try:
        resp = litellm.client_session.post(
            "https://api.openai.com/v1/chat/completions",
            content=b'{"model":"gpt-4o","messages":[]}',
        )
        assert resp.status_code == 200
        assert resp.content == b'{"ok":true}'
    finally:
        adapter.teardown()


def test_bind_replay_divergence_on_mismatched_request():
    tape = Tape()
    tape.append_exchange(b"RECORDED", b"RESP")
    litellm = _fake_litellm_module()
    adapter = CrewAIAdapter()
    result = adapter.bind(litellm, tape, mode="replay", patch_uuid=False)
    try:
        with pytest.raises(DivergenceError):
            result.http_client.post("https://api.openai.com/v1/x", content=b"DIFFERENT")
    finally:
        adapter.teardown()


def test_bind_replay_installs_uuid_patch_and_teardown_restores():
    tape = Tape()
    tape.draws = [("uuid", "0" * 32), ("uuid", "1" * 32)]
    litellm = _fake_litellm_module()
    adapter = CrewAIAdapter()
    result = adapter.bind(litellm, tape, mode="replay")  # patch_uuid defaults True
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
    litellm = _fake_litellm_module()
    adapter = CrewAIAdapter()
    result = adapter.bind(litellm, tape, mode="replay", nondet=supplied)
    try:
        assert result.nondet is supplied
        assert uuid.uuid4().hex == "f" * 32
    finally:
        adapter.teardown()


# ── on_step: event-bus event -> Step, building the DAG ───────────────────────────


def test_on_step_builds_dag_across_events():
    adapter = CrewAIAdapter()
    adapter.on_step({"event": "crew_kickoff_started", "id": "crew1", "name": "my-crew"})
    adapter.on_step(
        {"event": "task_started", "id": "task1", "parent_id": "crew1", "name": "research"}
    )
    adapter.on_step(
        {"event": "llm_call_started", "id": "llm1", "parent_id": "task1", "model": "gpt-4o"}
    )
    adapter.on_step({"event": "llm_call_completed", "id": "llm1", "outputs": "answer"})
    adapter.on_step({"event": "task_completed", "id": "task1", "outputs": "done"})
    adapter.on_step({"event": "crew_kickoff_completed", "id": "crew1", "outputs": "final"})

    dag = adapter.dag
    assert [s.step_id for s in dag.steps] == ["crew1", "task1", "llm1"]
    llm = dag.by_id("llm1")
    assert llm.kind == "llm"
    assert llm.parent_id == "task1"
    assert llm.model == "gpt-4o"
    assert llm.outputs == "answer"
    assert dag.by_id("task1").outputs == "done"
    assert dag.by_id("crew1").outputs == "final"
    assert [s.step_id for s in dag.llm_steps()] == ["llm1"]


def test_on_step_tool_events():
    adapter = CrewAIAdapter()
    adapter.on_step(
        {"event": "tool_usage_started", "id": "t1", "name": "search", "inputs": "query"}
    )
    step = adapter.on_step({"event": "tool_usage_finished", "id": "t1", "outputs": "result"})
    assert step.kind == "tool"
    assert step.inputs == "query"
    assert step.outputs == "result"


def test_on_step_unknown_event_is_not_dropped():
    adapter = CrewAIAdapter()
    step = adapter.on_step({"event": "flow_started", "id": "f1", "name": "flow"})
    assert step.step_id == "f1"
    assert step.kind == "flow_started"


def test_on_step_end_event_with_no_matching_start_returns_placeholder():
    adapter = CrewAIAdapter()
    step = adapter.on_step({"event": "task_completed", "id": "ghost"})
    assert step.step_id == "ghost"
    assert step.kind == "task_completed"


# ── TraceforkCrewEventCore (direct) ──────────────────────────────────────────────


def test_event_core_start_end_roundtrip():
    core = TraceforkCrewEventCore()
    core.start("task", "t1", name="research")
    core.end("t1", outputs="done")
    step = core.dag.by_id("t1")
    assert step.name == "research"
    assert step.outputs == "done"


def test_event_core_end_without_start_is_none():
    core = TraceforkCrewEventCore()
    assert core.end("missing") is None


# ── availability guards ─────────────────────────────────────────────────────────


def test_require_crewai_matches_availability():
    if crewai_available():
        require_crewai()
    else:
        with pytest.raises(ImportError, match="crewai"):
            require_crewai()


def test_make_event_listener_guarded_or_builds():
    if not crewai_available():
        with pytest.raises(ImportError, match="crewai"):
            make_event_listener()
    else:  # pragma: no cover - only when crewai is installed
        listener = make_event_listener()
        assert hasattr(listener, "dag")


# ── real-framework smoke (skipped cleanly when the framework is absent) ─────────


def test_event_listener_registers_with_real_crewai():
    pytest.importorskip("crewai")  # pragma: no cover - needs crewai extra
    listener = make_event_listener()  # pragma: no cover
    assert hasattr(listener, "setup_listeners")  # pragma: no cover
