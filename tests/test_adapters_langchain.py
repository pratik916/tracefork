"""LangChain/LangGraph adapter — offline, no framework installed.

The framework-facing thin wrappers (``BaseCallbackHandler`` /
``BaseCheckpointSaver`` subclasses) need the real libraries and are covered by
``pytest.importorskip`` blocks that skip cleanly when absent. Everything that
must work with NO framework installed — the injection into a chat client, the
replay flow through the bound httpx transport, the neutral callback core, the
event dispatch, the checkpoint store, and the availability guards — is driven
here with synthetic objects and the ``anthropic`` SDK (a hard dependency).
"""

import types
import uuid

import anthropic
import pytest

from tracefork.adapters.langchain import (
    LangChainAdapter,
    TapeBackedCheckpointStore,
    TraceforkCallbackCore,
    langchain_available,
    langgraph_available,
    make_callback_handler,
    make_tape_backed_checkpointer,
    require_langchain,
    require_langgraph,
)
from tracefork.nondet import DivergenceError, ReplayNondet
from tracefork.tape import Tape
from tracefork.transport import TraceforkTransport

# ── fake chat models (mimic the ChatOpenAI / ChatAnthropic client shapes) ───────


class _FakeOpenAIClient:
    """Mimics an ``openai.OpenAI`` client: ``.copy(http_client=)`` + ``.chat.completions``."""

    def __init__(self, http_client=None):
        self.http_client = http_client
        self.chat = types.SimpleNamespace(completions=object())

    def copy(self, *, http_client=None):
        return _FakeOpenAIClient(http_client=http_client)


class _FakeChatOpenAI:
    def __init__(self):
        self.root_client = _FakeOpenAIClient()
        self.root_async_client = _FakeOpenAIClient()
        self.client = self.root_client.chat.completions
        self.async_client = self.root_async_client.chat.completions


class FakeChatAnthropic:
    """Plain object whose class name triggers the anthropic family detector.

    The real ``ChatAnthropic`` has NO ``http_client`` field and builds ``_client``
    lazily as a cached_property; ``bind`` seeds a real ``anthropic.Anthropic`` (a
    hard tracefork dependency) via ``object.__setattr__``. This fake deliberately
    starts with NO ``_client`` — proving ``bind`` never probes the cached property
    (which would demand an api key)."""


# ── bind: injection into the underlying client ──────────────────────────────────


def test_bind_injects_openai_client():
    tape = Tape()
    model = _FakeChatOpenAI()
    adapter = LangChainAdapter()
    result = adapter.bind(model, tape, mode="replay", patch_uuid=False)
    try:
        assert set(result.injected_fields) == {
            "root_client",
            "client",
            "root_async_client",
            "async_client",
        }
        # The model's underlying client now routes through the tracefork client.
        assert model.root_client.http_client is result.http_client
        assert model.root_async_client.http_client is result.http_async_client
        assert result.notes == ""
    finally:
        adapter.teardown()


def test_bind_injects_anthropic_client_with_real_sdk():
    tape = Tape()
    model = FakeChatAnthropic()
    adapter = LangChainAdapter()
    result = adapter.bind(model, tape, mode="replay", patch_uuid=False)
    try:
        assert set(result.injected_fields) == {"_client", "_async_client"}
        # bind seeded a real anthropic client whose httpx transport is tracefork's.
        assert isinstance(model._client, anthropic.Anthropic)
        assert isinstance(model._async_client, anthropic.AsyncAnthropic)
        assert model._client._client is result.http_client
        assert isinstance(result.transport, TraceforkTransport)
        assert model._client._client._transport is result.transport
    finally:
        adapter.teardown()


def test_bind_anthropic_replay_serves_recorded_bytes():
    tape = Tape()
    tape.append_exchange(b'{"model":"claude","messages":[]}', b'{"ok":true}')
    model = FakeChatAnthropic()
    adapter = LangChainAdapter()
    result = adapter.bind(model, tape, mode="replay", patch_uuid=False)
    try:
        resp = result.http_client.post(
            "https://api.anthropic.com/v1/messages",
            content=b'{"model":"claude","messages":[]}',
        )
        assert resp.status_code == 200
        assert resp.content == b'{"ok":true}'
    finally:
        adapter.teardown()


def test_bind_unknown_target_reports_notes():
    adapter = LangChainAdapter()
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
    tape.append_exchange(b'{"model":"gpt","messages":[]}', b'{"ok":true}')
    model = _FakeChatOpenAI()
    adapter = LangChainAdapter()
    result = adapter.bind(model, tape, mode="replay", patch_uuid=False)
    try:
        client = model.root_client.http_client
        assert client is result.http_client
        resp = client.post(
            "https://api.openai.com/v1/chat/completions",
            content=b'{"model":"gpt","messages":[]}',
        )
        assert resp.status_code == 200
        assert resp.content == b'{"ok":true}'
    finally:
        adapter.teardown()


def test_bind_replay_divergence_on_mismatched_request():
    tape = Tape()
    tape.append_exchange(b"RECORDED", b"RESP")
    model = _FakeChatOpenAI()
    adapter = LangChainAdapter()
    result = adapter.bind(model, tape, mode="replay", patch_uuid=False)
    try:
        with pytest.raises(DivergenceError):
            result.http_client.post("https://api.openai.com/v1/x", content=b"DIFFERENT")
    finally:
        adapter.teardown()


def test_bind_replay_installs_uuid_patch_and_teardown_restores():
    tape = Tape()
    tape.draws = [("uuid", "0" * 32), ("uuid", "1" * 32)]
    model = _FakeChatOpenAI()
    adapter = LangChainAdapter()
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
    model = _FakeChatOpenAI()
    adapter = LangChainAdapter()
    result = adapter.bind(model, tape, mode="replay", nondet=supplied)
    try:
        assert result.nondet is supplied
        assert uuid.uuid4().hex == "f" * 32
    finally:
        adapter.teardown()


# ── on_step: callback event -> Step, building the DAG ───────────────────────────


def test_on_step_builds_dag_across_events():
    adapter = LangChainAdapter()
    adapter.on_step(
        {"event": "on_chain_start", "run_id": "chain1", "serialized": {"name": "AgentExecutor"}}
    )
    adapter.on_step(
        {
            "event": "on_chat_model_start",
            "run_id": "llm1",
            "parent_run_id": "chain1",
            "serialized": {"id": ["x", "ChatOpenAI"], "kwargs": {"model": "gpt-4o"}},
        }
    )
    adapter.on_step(
        {
            "event": "on_llm_end",
            "run_id": "llm1",
            "response": {"generations": [[{"text": "hello"}]]},
        }
    )
    adapter.on_step({"event": "on_chain_end", "run_id": "chain1", "outputs": {"result": "done"}})

    dag = adapter.dag
    assert [s.step_id for s in dag.steps] == ["chain1", "llm1"]
    llm = dag.by_id("llm1")
    assert llm.kind == "chat_model"
    assert llm.parent_id == "chain1"
    assert llm.model == "gpt-4o"
    assert llm.outputs == "hello"
    assert dag.by_id("chain1").outputs == {"result": "done"}
    assert [s.step_id for s in dag.llm_steps()] == ["llm1"]


def test_on_step_tool_events():
    adapter = LangChainAdapter()
    adapter.on_step(
        {
            "event": "on_tool_start",
            "run_id": "t1",
            "serialized": {"name": "search"},
            "input_str": "query",
        }
    )
    step = adapter.on_step({"event": "on_tool_end", "run_id": "t1", "output": "result"})
    assert step.kind == "tool"
    assert step.inputs == "query"
    assert step.outputs == "result"


def test_on_step_unknown_event_is_not_dropped():
    adapter = LangChainAdapter()
    step = adapter.on_step({"event": "on_retriever_start", "run_id": "r1", "name": "vec"})
    assert step.step_id == "r1"
    assert step.kind == "on_retriever_start"


# ── TraceforkCallbackCore (direct) ──────────────────────────────────────────────


def test_callback_core_llm_end_extracts_text_and_model():
    core = TraceforkCallbackCore()
    core.on_llm_start({"name": "OpenAI"}, ["prompt"], run_id="l", parent_run_id=None)
    core.on_llm_end(
        {"generations": [[{"text": "answer"}]], "llm_output": {"model_name": "gpt-4o-mini"}},
        run_id="l",
    )
    step = core.dag.by_id("l")
    assert step.outputs == "answer"
    assert step.model == "gpt-4o-mini"


def test_callback_core_chat_message_content_fallback():
    core = TraceforkCallbackCore()
    core.on_chat_model_start({"name": "ChatX"}, [[]], run_id="c", parent_run_id=None)
    core.on_llm_end(
        {"generations": [[{"text": "", "message": {"content": "from-message"}}]]}, run_id="c"
    )
    assert core.dag.by_id("c").outputs == "from-message"


# ── TapeBackedCheckpointStore (the tested checkpointer core) ─────────────────────


def test_checkpoint_store_put_get_latest_and_by_id():
    store = TapeBackedCheckpointStore(Tape())
    store.put("t", "c1", {"v": 1})
    store.put("t", "c2", {"v": 2}, parent_id="c1")
    assert store.get("t").checkpoint_id == "c2"  # latest
    assert store.get("t", "c1").data == {"v": 1}
    assert store.get("t", "missing") is None
    assert store.get("other-thread") is None


def test_checkpoint_store_put_overwrites_same_id():
    store = TapeBackedCheckpointStore(Tape())
    store.put("t", "c1", {"v": 1})
    store.put("t", "c1", {"v": 99})
    assert store.get("t", "c1").data == {"v": 99}
    assert len(store.list("t")) == 1


def test_checkpoint_store_list_newest_first_with_before_and_limit():
    store = TapeBackedCheckpointStore(Tape())
    for i in range(4):
        store.put("t", f"c{i}", {"i": i})
    ids = [r.checkpoint_id for r in store.list("t")]
    assert ids == ["c3", "c2", "c1", "c0"]  # newest-first (time-travel order)
    assert [r.checkpoint_id for r in store.list("t", limit=2)] == ["c3", "c2"]
    # `before=c2` keeps only history strictly older than c2 (for resume/fork).
    assert [r.checkpoint_id for r in store.list("t", before="c2")] == ["c1", "c0"]


def test_checkpoint_store_history_is_newest_first():
    store = TapeBackedCheckpointStore(Tape())
    store.put("t", "a", 1)
    store.put("t", "b", 2)
    assert [r.checkpoint_id for r in store.history("t")] == ["b", "a"]


def test_checkpoint_store_records_tape_index_link():
    tape = Tape()
    tape.append_exchange(b"REQ", b"RESP")
    store = TapeBackedCheckpointStore(tape)
    rec = store.put("t", "c1", {"v": 1}, tape_index=0)
    assert rec.tape_index == 0
    assert store.tape is tape


# ── availability guards ─────────────────────────────────────────────────────────


def test_require_langchain_matches_availability():
    if langchain_available():
        require_langchain()
    else:
        with pytest.raises(ImportError, match="frameworks"):
            require_langchain()


def test_require_langgraph_matches_availability():
    if langgraph_available():
        require_langgraph()
    else:
        with pytest.raises(ImportError, match="frameworks"):
            require_langgraph()


def test_make_callback_handler_guarded_or_builds():
    if not langchain_available():
        with pytest.raises(ImportError, match="frameworks"):
            make_callback_handler()
    else:  # pragma: no cover - only when langchain is installed
        handler = make_callback_handler()
        assert hasattr(handler, "dag")


def test_make_tape_backed_checkpointer_guarded_or_builds():
    if not langgraph_available():
        with pytest.raises(ImportError, match="frameworks"):
            make_tape_backed_checkpointer(Tape())
    else:  # pragma: no cover - only when langgraph is installed
        cp = make_tape_backed_checkpointer(Tape())
        assert hasattr(cp, "store")


# ── real-framework smoke (skipped cleanly when frameworks are absent) ───────────


def test_callback_handler_forwards_to_core_with_real_langchain():
    pytest.importorskip("langchain_core")  # pragma: no cover - needs frameworks extra
    from uuid import uuid4  # pragma: no cover

    handler = make_callback_handler()  # pragma: no cover
    handler.on_chain_start({"name": "C"}, {"x": 1}, run_id=uuid4())  # pragma: no cover
    assert len(handler.dag) == 1  # pragma: no cover
