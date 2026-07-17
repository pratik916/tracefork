"""Shepherd adapter — offline, synthetic-double-only (see the module docstring
in ``adapters/shepherd.py`` for why: Shepherd is an unpublished
codebase, not a published package, so unlike the other adapters there is no
real-framework import to guard and no ``pytest.importorskip`` tier here).
"""

import uuid

import pytest

from tracefork.adapters.base import get_framework_adapter, registered_framework_adapters
from tracefork.adapters.shepherd import ShepherdAdapter, TraceforkShepherdCore
from tracefork.nondet import DivergenceError, ReplayNondet
from tracefork.tape import Tape
from tracefork.transport import TraceforkTransport

# ── fake OpenAIProvider (mimics the shape Shepherd's provider is understood to hold) ──


class _FakeAsyncOpenAIClient:
    """Mimics ``openai.AsyncOpenAI``: ``.copy(http_client=)``."""

    def __init__(self, http_client=None):
        self.http_client = http_client

    def copy(self, *, http_client=None):
        return _FakeAsyncOpenAIClient(http_client=http_client)


class _FakeOpenAIProvider:
    """Plain stand-in for Shepherd's ``OpenAIProvider``.

    Stores its client under ``_client`` — the FIRST candidate attribute name
    ``bind`` searches — proving the defensive search finds it without
    hard-coding a single "true" attribute name (no real Shepherd package
    exists to verify one against)."""

    def __init__(self):
        self._client = _FakeAsyncOpenAIClient()


class _FakeOpenAIProviderWithBothAttrs:
    """Carries BOTH ``_client`` and ``client`` — proves candidate search order
    is preserved (``_client`` wins, since it is searched first)."""

    def __init__(self):
        self._client = _FakeAsyncOpenAIClient()
        self.client = _FakeAsyncOpenAIClient()


# ── bind: injection into the underlying client ──────────────────────────────────


def test_bind_injects_client_attribute():
    tape = Tape()
    provider = _FakeOpenAIProvider()
    adapter = ShepherdAdapter()
    result = adapter.bind(provider, tape, mode="replay", patch_uuid=False)
    try:
        assert result.injected_fields == ("_client",)
        assert provider._client.http_client is result.http_async_client
        assert isinstance(result.transport, TraceforkTransport)
    finally:
        adapter.teardown()


def test_bind_candidate_search_order_preserved():
    tape = Tape()
    provider = _FakeOpenAIProviderWithBothAttrs()
    adapter = ShepherdAdapter()
    result = adapter.bind(provider, tape, mode="replay", patch_uuid=False)
    try:
        # "_client" precedes "client" in the candidate list, so it wins even
        # though both attributes are present.
        assert result.injected_fields == ("_client",)
        assert provider._client.http_client is result.http_async_client
        assert provider.client.http_client is None  # untouched
    finally:
        adapter.teardown()


def test_bind_unknown_target_reports_notes():
    adapter = ShepherdAdapter()
    result = adapter.bind(object(), Tape(), mode="replay", patch_uuid=False)
    try:
        assert result.injected_fields == ()
        assert "nothing was injected" in result.notes
        assert "Claude and OpenCode providers are not bound" in result.notes
    finally:
        adapter.teardown()


def test_bind_found_notes_still_state_openai_path_only_scope():
    """Even on a successful bind, notes name the Claude/OpenCode scope limit —
    the honest posture the spec calls for, not a silent narrowing."""
    tape = Tape()
    provider = _FakeOpenAIProvider()
    adapter = ShepherdAdapter()
    result = adapter.bind(provider, tape, mode="replay", patch_uuid=False)
    try:
        assert "Claude and OpenCode providers are not bound" in result.notes
    finally:
        adapter.teardown()


async def test_bind_replay_serves_recorded_bytes_bit_exact():
    """The marquee: a run bound in replay mode serves tape bytes for $0, and a
    request that diverges from the tape is caught (proof, not assertion)."""
    tape = Tape()
    tape.append_exchange(b'{"model":"gpt-4o","messages":[]}', b'{"ok":true}')
    provider = _FakeOpenAIProvider()
    adapter = ShepherdAdapter()
    result = adapter.bind(provider, tape, mode="replay", patch_uuid=False)
    try:
        client = provider._client.http_client
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
    provider = _FakeOpenAIProvider()
    adapter = ShepherdAdapter()
    result = adapter.bind(provider, tape, mode="replay", patch_uuid=False)
    try:
        with pytest.raises(DivergenceError):
            await result.http_async_client.post("https://api.openai.com/v1/x", content=b"DIFFERENT")
    finally:
        adapter.teardown()


def test_bind_replay_installs_uuid_patch_and_teardown_restores():
    tape = Tape()
    tape.draws = [("uuid", "0" * 32), ("uuid", "1" * 32)]
    provider = _FakeOpenAIProvider()
    adapter = ShepherdAdapter()
    result = adapter.bind(provider, tape, mode="replay")  # patch_uuid defaults True
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
    provider = _FakeOpenAIProvider()
    adapter = ShepherdAdapter()
    result = adapter.bind(provider, tape, mode="replay", nondet=supplied)
    try:
        assert result.nondet is supplied
        assert uuid.uuid4().hex == "f" * 32
    finally:
        adapter.teardown()


# ── on_step: generic dict event -> Step, building the DAG ───────────────────────


def test_on_step_builds_dag_across_start_and_end_events():
    adapter = ShepherdAdapter()
    adapter.on_step(
        {
            "event": "start",
            "id": "s1",
            "parent_id": None,
            "kind": "llm",
            "model": "gpt-4o",
            "inputs": {"prompt": "hi"},
        }
    )
    step = adapter.on_step({"event": "end", "id": "s1", "outputs": {"text": "hello"}})

    dag = adapter.dag
    assert [s.step_id for s in dag.steps] == ["s1"]
    assert step.step_id == "s1"
    assert step.kind == "llm"
    assert step.model == "gpt-4o"
    assert step.inputs == {"prompt": "hi"}
    assert step.outputs == {"text": "hello"}
    assert step.is_llm()
    assert [s.step_id for s in dag.llm_steps()] == ["s1"]


def test_on_step_non_llm_kind_keeps_its_kind():
    adapter = ShepherdAdapter()
    step = adapter.on_step({"event": "start", "id": "t1", "kind": "tool", "name": "search"})
    assert step.kind == "tool"
    assert step.name == "search"
    assert not step.is_llm()


def test_on_step_parent_child_linkage():
    adapter = ShepherdAdapter()
    adapter.on_step({"event": "start", "id": "root", "kind": "chain"})
    child = adapter.on_step({"event": "start", "id": "child", "parent_id": "root", "kind": "llm"})
    assert child.parent_id == "root"
    assert adapter.dag.children("root") == [child]


def test_on_step_unknown_event_is_not_dropped():
    adapter = ShepherdAdapter()
    step = adapter.on_step({"event": "tool_call", "id": "x1"})
    assert step.step_id == "x1"
    assert step.kind == "tool_call"


def test_on_step_end_event_with_no_matching_start_returns_placeholder():
    adapter = ShepherdAdapter()
    step = adapter.on_step({"event": "end", "id": "ghost", "outputs": "orphaned"})
    assert step.step_id == "ghost"
    assert step.kind == ""


def test_on_step_synthesizes_id_when_missing():
    adapter = ShepherdAdapter()
    step = adapter.on_step({"event": "start", "kind": "llm"})
    assert step.step_id  # non-empty, derived from id(event)


# ── TraceforkShepherdCore (direct) ───────────────────────────────────────────────


def test_shepherd_core_end_updates_existing_step_outputs():
    core = TraceforkShepherdCore()
    core.dispatch({"event": "start", "id": "s", "kind": "llm", "model": "gpt-4o-mini"})
    core.dispatch({"event": "end", "id": "s", "outputs": "final"})
    step = core.dag.by_id("s")
    assert step.outputs == "final"
    assert step.model == "gpt-4o-mini"


def test_shepherd_core_end_without_outputs_leaves_existing_step_untouched():
    core = TraceforkShepherdCore()
    core.dispatch({"event": "start", "id": "s", "inputs": "keep-me"})
    core.dispatch({"event": "end", "id": "s"})
    step = core.dag.by_id("s")
    assert step.inputs == "keep-me"
    assert step.outputs is None


# ── registry ─────────────────────────────────────────────────────────────────────


def test_shepherd_registered_in_framework_adapter_registry():
    assert "shepherd" in registered_framework_adapters()
    assert isinstance(get_framework_adapter("shepherd"), ShepherdAdapter)
