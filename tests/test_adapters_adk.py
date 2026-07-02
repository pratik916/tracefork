"""Google ADK adapter — offline, no framework installed.

The framework-facing thin wrapper (``BasePlugin`` subclass) needs the real
``google-adk`` package and is covered by a ``pytest.importorskip`` block that
skips cleanly when absent. Everything that must work with NO framework
installed — the candidate-path injection into a duck-typed google-genai
``BaseApiClient``, the replay flow through the bound httpx transport, the
neutral event core, the event dispatch, and the availability guards — is
driven here with synthetic objects (``bind`` never imports ``google.adk`` or
``google.genai`` at all; it is duck-typed).
"""

import uuid

import pytest

from tracefork.adapters.adk import (
    AdkAdapter,
    TraceforkAdkCore,
    adk_available,
    make_plugin,
    require_adk,
)
from tracefork.nondet import DivergenceError, ReplayNondet
from tracefork.tape import Tape
from tracefork.transport import TraceforkTransport

# ── fake google-genai object graph (mimics BaseApiClient/Client/Gemini/LlmAgent) ─


class _FakeBaseApiClient:
    """Mimics ``google.genai._api_client.BaseApiClient``: plain instance attrs
    ``_httpx_client`` (sync) / ``_async_httpx_client`` (async)."""

    def __init__(self):
        self._httpx_client = None
        self._async_httpx_client = None


class _FakeGenaiClient:
    """Mimics ``google.genai.Client``: wraps a ``BaseApiClient`` under ``_api_client``."""

    def __init__(self):
        self._api_client = _FakeBaseApiClient()


class _FakeGemini:
    """Mimics an ADK ``Gemini`` model wrapper: ``.api_client`` -> a genai ``Client``."""

    def __init__(self):
        self.api_client = _FakeGenaiClient()


class _FakeLlmAgent:
    """Mimics an ADK ``LlmAgent`` whose ``.model`` already holds a resolved ``Gemini``."""

    def __init__(self):
        self.model = _FakeGemini()


# ── bind: candidate-path injection into the genai BaseApiClient ─────────────────


def test_bind_injects_into_bare_base_api_client():
    """Path (): target itself is the BaseApiClient-shaped object."""
    tape = Tape()
    holder = _FakeBaseApiClient()
    adapter = AdkAdapter()
    result = adapter.bind(holder, tape, mode="replay", patch_uuid=False)
    try:
        assert set(result.injected_fields) == {"_httpx_client", "_async_httpx_client"}
        assert holder._httpx_client is result.http_client
        assert holder._async_httpx_client is result.http_async_client
        assert result.notes == ""
        assert isinstance(result.transport, TraceforkTransport)
    finally:
        adapter.teardown()


def test_bind_injects_via_genai_client_path():
    """Path (_api_client,): target is a genai.Client."""
    tape = Tape()
    client = _FakeGenaiClient()
    adapter = AdkAdapter()
    result = adapter.bind(client, tape, mode="replay", patch_uuid=False)
    try:
        assert client._api_client._httpx_client is result.http_client
        assert client._api_client._async_httpx_client is result.http_async_client
    finally:
        adapter.teardown()


def test_bind_injects_via_gemini_wrapper_path():
    """Path (api_client, _api_client): target is an ADK Gemini model wrapper."""
    tape = Tape()
    gemini = _FakeGemini()
    adapter = AdkAdapter()
    result = adapter.bind(gemini, tape, mode="replay", patch_uuid=False)
    try:
        assert gemini.api_client._api_client._httpx_client is result.http_client
        assert gemini.api_client._api_client._async_httpx_client is result.http_async_client
    finally:
        adapter.teardown()


def test_bind_injects_via_llm_agent_path():
    """Path (model, api_client, _api_client): target is an LlmAgent-shaped object."""
    tape = Tape()
    agent = _FakeLlmAgent()
    adapter = AdkAdapter()
    result = adapter.bind(agent, tape, mode="replay", patch_uuid=False)
    try:
        held = agent.model.api_client._api_client
        assert held._httpx_client is result.http_client
        assert held._async_httpx_client is result.http_async_client
    finally:
        adapter.teardown()


def test_bind_unknown_target_reports_notes():
    adapter = AdkAdapter()
    result = adapter.bind(object(), Tape(), mode="replay", patch_uuid=False)
    try:
        assert result.injected_fields == ()
        assert "nothing was injected" in result.notes
    finally:
        adapter.teardown()


def test_bind_llm_agent_with_unresolved_string_model_reports_notes():
    """An LlmAgent whose .model is still a plain string (not yet resolved to a
    Gemini instance) has no candidate path to a BaseApiClient — honest degrade."""
    adapter = AdkAdapter()
    agent = type("FakeAgent", (), {"model": "gemini-2.5-flash"})()
    result = adapter.bind(agent, Tape(), mode="replay", patch_uuid=False)
    try:
        assert result.injected_fields == ()
        assert "nothing was injected" in result.notes
    finally:
        adapter.teardown()


def test_bind_replay_serves_recorded_bytes_bit_exact():
    """The marquee: a run bound in replay mode serves tape bytes for $0, and a
    request that diverges from the tape is caught (proof, not assertion)."""
    tape = Tape()
    tape.append_exchange(
        b'{"contents":[{"role":"user","parts":[{"text":"hi"}]}]}', b'{"candidates":[]}'
    )
    holder = _FakeBaseApiClient()
    adapter = AdkAdapter()
    adapter.bind(holder, tape, mode="replay", patch_uuid=False)
    try:
        resp = holder._httpx_client.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
            content=b'{"contents":[{"role":"user","parts":[{"text":"hi"}]}]}',
        )
        assert resp.status_code == 200
        assert resp.content == b'{"candidates":[]}'
    finally:
        adapter.teardown()


def test_bind_replay_divergence_on_mismatched_request():
    tape = Tape()
    tape.append_exchange(b"RECORDED", b"RESP")
    holder = _FakeBaseApiClient()
    adapter = AdkAdapter()
    result = adapter.bind(holder, tape, mode="replay", patch_uuid=False)
    try:
        with pytest.raises(DivergenceError):
            result.http_client.post(
                "https://generativelanguage.googleapis.com/v1beta/models/x:generateContent",
                content=b"DIFFERENT",
            )
    finally:
        adapter.teardown()


def test_bind_replay_installs_uuid_patch_and_teardown_restores():
    tape = Tape()
    tape.draws = [("uuid", "0" * 32), ("uuid", "1" * 32)]
    holder = _FakeBaseApiClient()
    adapter = AdkAdapter()
    result = adapter.bind(holder, tape, mode="replay")  # patch_uuid defaults True
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
    holder = _FakeBaseApiClient()
    adapter = AdkAdapter()
    result = adapter.bind(holder, tape, mode="replay", nondet=supplied)
    try:
        assert result.nondet is supplied
        assert uuid.uuid4().hex == "f" * 32
    finally:
        adapter.teardown()


# ── on_step: BasePlugin callback event -> Step, building the DAG ────────────────


def test_on_step_builds_dag_across_events():
    adapter = AdkAdapter()
    adapter.on_step({"event": "before_agent_callback", "id": "agent1", "name": "researcher"})
    adapter.on_step(
        {
            "event": "before_model_callback",
            "id": "llm1",
            "parent_id": "agent1",
            "model": "gemini-2.5-flash",
        }
    )
    adapter.on_step({"event": "after_model_callback", "id": "llm1", "outputs": "answer"})
    adapter.on_step({"event": "after_agent_callback", "id": "agent1", "outputs": "final"})

    dag = adapter.dag
    assert [s.step_id for s in dag.steps] == ["agent1", "llm1"]
    llm = dag.by_id("llm1")
    assert llm.kind == "llm"
    assert llm.parent_id == "agent1"
    assert llm.model == "gemini-2.5-flash"
    assert llm.outputs == "answer"
    assert dag.by_id("agent1").outputs == "final"
    assert [s.step_id for s in dag.llm_steps()] == ["llm1"]


def test_on_step_tool_events():
    adapter = AdkAdapter()
    adapter.on_step(
        {"event": "before_tool_callback", "id": "t1", "name": "search", "inputs": {"q": "x"}}
    )
    step = adapter.on_step({"event": "after_tool_callback", "id": "t1", "outputs": {"r": "y"}})
    assert step.kind == "tool"
    assert step.inputs == {"q": "x"}
    assert step.outputs == {"r": "y"}


def test_on_step_unknown_event_is_not_dropped():
    adapter = AdkAdapter()
    step = adapter.on_step({"event": "on_user_message_callback", "id": "u1", "name": "msg"})
    assert step.step_id == "u1"
    assert step.kind == "on_user_message_callback"


def test_on_step_end_event_with_no_matching_start_returns_placeholder():
    adapter = AdkAdapter()
    step = adapter.on_step({"event": "after_tool_callback", "id": "ghost"})
    assert step.step_id == "ghost"
    assert step.kind == "after_tool_callback"


# ── TraceforkAdkCore (direct) ────────────────────────────────────────────────────


def test_event_core_start_end_roundtrip():
    core = TraceforkAdkCore()
    core.start("tool", "t1", name="search")
    core.end("t1", outputs="done")
    step = core.dag.by_id("t1")
    assert step.name == "search"
    assert step.outputs == "done"


def test_event_core_end_without_start_is_none():
    core = TraceforkAdkCore()
    assert core.end("missing") is None


# ── availability guards ─────────────────────────────────────────────────────────


def test_require_adk_matches_availability():
    if adk_available():
        require_adk()
    else:
        with pytest.raises(ImportError, match="adk"):
            require_adk()


def test_make_plugin_guarded_or_builds():
    if not adk_available():
        with pytest.raises(ImportError, match="adk"):
            make_plugin()
    else:  # pragma: no cover - only when google-adk is installed
        plugin = make_plugin()
        assert hasattr(plugin, "dag")


# ── real-framework smoke (skipped cleanly when the framework is absent) ─────────


def test_plugin_registers_with_real_adk():
    pytest.importorskip("google.adk")  # pragma: no cover - needs adk extra
    plugin = make_plugin()  # pragma: no cover
    assert hasattr(plugin, "before_model_callback")  # pragma: no cover
