"""Framework adapter base seam — offline, no framework installed.

Exercises the framework-neutral half of the adapter protocol: the ``Step`` /
``StepDAG`` overlay, the run-tree normalizer, the shared httpx2-client builder
(reusing ``transport.py``), the replay uuid patch, and the registry. Nothing
here imports langchain/langgraph — this is the logic that must work with no
framework present at all.
"""

import uuid

import httpx2
import pytest

from tracefork.adapters.base import (
    BaseFrameworkAdapter,
    BindResult,
    Step,
    StepDAG,
    UuidPatch,
    build_http_clients,
    get_framework_adapter,
    load_adapter_entry_points,
    register_framework_adapter,
    registered_framework_adapters,
)
from tracefork.nondet import DivergenceError, ReplayNondet
from tracefork.tape import Tape
from tracefork.transport import TraceforkTransport

# ── Step / StepDAG ────────────────────────────────────────────────────────────


def test_step_is_llm():
    assert Step("a", kind="llm").is_llm()
    assert Step("b", kind="chat_model").is_llm()
    assert not Step("c", kind="chain").is_llm()
    assert not Step("d", kind="tool").is_llm()


def test_dag_add_overwrites_same_id():
    dag = StepDAG()
    dag.add(Step("a", kind="chain", name="first"))
    dag.add(Step("a", kind="chain", name="second"))
    assert len(dag) == 1
    assert dag.by_id("a").name == "second"


def test_dag_children_roots_and_kinds():
    dag = StepDAG.from_steps(
        [
            Step("root", parent_id=None, kind="chain"),
            Step("llm1", parent_id="root", kind="chat_model"),
            Step("tool1", parent_id="root", kind="tool"),
            Step("llm2", parent_id="tool1", kind="llm"),
        ]
    )
    assert [s.step_id for s in dag.roots()] == ["root"]
    assert [s.step_id for s in dag.children("root")] == ["llm1", "tool1"]
    assert [s.step_id for s in dag.of_kind("tool")] == ["tool1"]
    assert [s.step_id for s in dag.llm_steps()] == ["llm1", "llm2"]


def test_dag_roots_when_parent_absent():
    # A parent_id pointing outside the DAG counts as a root (dangling parent).
    dag = StepDAG.from_steps([Step("child", parent_id="ghost", kind="llm")])
    assert [s.step_id for s in dag.roots()] == ["child"]


def test_assign_tape_indices_numbers_llm_steps_in_order():
    dag = StepDAG.from_steps(
        [
            Step("root", kind="chain"),
            Step("llm1", parent_id="root", kind="chat_model"),
            Step("tool1", parent_id="root", kind="tool"),
            Step("llm2", parent_id="root", kind="llm"),
        ]
    )
    dag.assign_tape_indices()
    assert [s.tape_index for s in dag.llm_steps()] == [0, 1]
    assert dag.by_id("tool1").tape_index is None
    assert dag.by_id("root").tape_index is None


# ── from_run_tree (the normalizer) ──────────────────────────────────────────────


def test_from_run_tree_nested_mapping():
    tree = {
        "id": "root",
        "run_type": "chain",
        "name": "agent",
        "child_runs": [
            {"id": "m1", "run_type": "chat_model", "name": "llm"},
            {
                "id": "t1",
                "run_type": "tool",
                "name": "search",
                "children": [{"id": "m2", "run_type": "llm", "name": "llm2"}],
            },
        ],
    }
    dag = StepDAG.from_run_tree(tree)
    assert [s.step_id for s in dag.steps] == ["root", "m1", "t1", "m2"]
    assert dag.by_id("root").kind == "chain"
    assert dag.by_id("m1").parent_id == "root"
    assert dag.by_id("m2").parent_id == "t1"
    assert [s.step_id for s in dag.llm_steps()] == ["m1", "m2"]


def test_from_run_tree_accepts_list_of_roots():
    dag = StepDAG.from_run_tree(
        [
            {"id": "a", "kind": "chain"},
            {"id": "b", "kind": "chain"},
        ]
    )
    assert [s.step_id for s in dag.steps] == ["a", "b"]
    assert dag.by_id("a").parent_id is None


def test_from_run_tree_objects_and_alt_keys():
    class Node:
        def __init__(self, run_id, run_type, name, child_runs=()):
            self.run_id = run_id
            self.run_type = run_type
            self.name = name
            self.child_runs = list(child_runs)

    tree = Node("r", "chain", "root", [Node("c", "llm", "child")])
    dag = StepDAG.from_run_tree(tree)
    assert [s.step_id for s in dag.steps] == ["r", "c"]
    assert dag.by_id("c").parent_id == "r"
    assert dag.by_id("c").kind == "llm"


def test_from_run_tree_synthesizes_id_when_missing():
    dag = StepDAG.from_run_tree({"kind": "llm", "name": "anon"})
    assert len(dag) == 1
    assert dag.steps[0].step_id  # a synthesized hex id, non-empty


def test_from_run_tree_none_is_empty():
    assert len(StepDAG.from_run_tree(None)) == 0


# ── build_http_clients (reuses transport.py) ────────────────────────────────────


def test_build_http_clients_replay_needs_no_inner():
    tape = Tape()
    sync_c, async_c, sync_t, async_t = build_http_clients(tape, "replay")
    assert isinstance(sync_c, httpx2.Client)
    assert isinstance(async_c, httpx2.AsyncClient)
    assert isinstance(sync_t, TraceforkTransport)
    assert sync_t.mode == "replay"
    assert async_t.mode == "replay"
    sync_c.close()


def test_build_http_clients_record_requires_inner():
    tape = Tape()
    with pytest.raises(ValueError, match="record mode requires an inner transport"):
        build_http_clients(tape, "record")


# ── UuidPatch (replay determinism for framework-generated ids) ──────────────────


def test_uuid_patch_serves_recorded_ids_then_restores():
    tape = Tape()
    tape.draws = [("uuid", "0" * 32), ("uuid", "1" * 32)]
    nondet = ReplayNondet(tape.draws)
    patch = UuidPatch(nondet)
    real = uuid.uuid4()  # a genuine random uuid before patching
    patch.install()
    try:
        assert uuid.uuid4().hex == "0" * 32
        assert uuid.uuid4().hex == "1" * 32
    finally:
        patch.restore()
    after = uuid.uuid4()
    assert after != real  # back to real randomness, not the recorded stream


def test_uuid_patch_install_and_restore_idempotent():
    nondet = ReplayNondet([("uuid", "a" * 32)])
    patch = UuidPatch(nondet)
    patch.install()
    patch.install()  # no double-capture of the patched fn as "original"
    assert uuid.uuid4().hex == "a" * 32
    patch.restore()
    patch.restore()  # safe to call twice
    # After full restore, uuid is real again (does not raise / not exhausted).
    assert isinstance(uuid.uuid4(), uuid.UUID)


# ── registry ────────────────────────────────────────────────────────────────


class _DummyAdapter(BaseFrameworkAdapter):
    name = "dummy-test-adapter"

    def bind(self, target, tape, mode="replay", **kwargs):
        return BindResult(mode=mode)

    def on_step(self, event):
        return Step(step_id=str(event.get("run_id", "x")), kind=event.get("kind", ""))


def test_register_and_get_framework_adapter():
    adapter = _DummyAdapter()
    register_framework_adapter(adapter)
    assert "dummy-test-adapter" in registered_framework_adapters()
    assert get_framework_adapter("dummy-test-adapter") is adapter


def test_get_unknown_adapter_lists_registered():
    with pytest.raises(KeyError, match="no framework adapter registered"):
        get_framework_adapter("does-not-exist-xyz")


def test_load_adapter_entry_points_noop_without_allow():
    # Security-gated: nothing loads unless explicitly allowlisted (see plugins.py).
    assert load_adapter_entry_points() == []


def test_base_adapter_records_steps_into_its_dag():
    adapter = _DummyAdapter()
    adapter.record_step(Step("a", kind="chain"))
    adapter.record_step(Step("b", parent_id="a", kind="llm"))
    assert [s.step_id for s in adapter.dag.steps] == ["a", "b"]


# ── Pydantic AI adapter (offline, no framework installed) ───────────────────
#
# tracefork-sis.61: the new adapters/pydantic_ai.py is exercised here rather
# than a dedicated tests/test_adapters_pydantic_ai.py -- this lane owns only
# tests/test_adapters_base.py, and importing tracefork.adapters.pydantic_ai
# needs no real `pydantic_ai` package (same guarded-import contract this
# file's own charter already requires of everything it tests), so it fits
# this module's "must work with no framework present at all" scope.

from tracefork.adapters.pydantic_ai import (  # noqa: E402
    PydanticAIAdapter,
    pydantic_ai_available,
    require_pydantic_ai,
)


class _FakeAsyncClient:
    """Mimics ``openai.AsyncOpenAI``/``anthropic.AsyncAnthropic``: ``.copy(http_client=)``."""

    def __init__(self, http_client=None):
        self.http_client = http_client

    def copy(self, *, http_client=None):
        return _FakeAsyncClient(http_client=http_client)


class _FakeProvider:
    """Stand-in for pydantic_ai's ``OpenAIProvider``/``AnthropicProvider``."""

    def __init__(self):
        self.client = _FakeAsyncClient()


class _FakeModel:
    """Stand-in for pydantic_ai's ``OpenAIModel``/``AnthropicModel`` -- holds its
    client one level down, under ``.provider.client`` (one of ``bind``'s
    candidate holder paths), proving the search doesn't hard-code a single
    "true" attribute path -- the real package documents none."""

    def __init__(self):
        self.provider = _FakeProvider()


class _FakeAgent:
    """Stand-in for pydantic_ai's ``Agent`` -- holds a ``Model`` under ``.model``."""

    def __init__(self):
        self.model = _FakeModel()


def test_pydantic_ai_registers_itself():
    assert "pydantic_ai" in registered_framework_adapters()
    assert isinstance(get_framework_adapter("pydantic_ai"), PydanticAIAdapter)


def test_pydantic_ai_bind_injects_nested_agent_model_provider_client():
    tape = Tape()
    agent = _FakeAgent()
    adapter = PydanticAIAdapter()
    result = adapter.bind(agent, tape, mode="replay", patch_uuid=False)
    try:
        assert result.injected_fields == ("model.provider.client",)
        assert agent.model.provider.client.http_client is result.http_async_client
        assert result.notes == ""
    finally:
        adapter.teardown()


def test_pydantic_ai_bind_injects_client_attribute_directly_on_target():
    # A caller may hand `bind` an already-resolved Model/Provider-shaped
    # object exposing `.client` directly (one of `bind`'s other candidate
    # holder paths: the target itself).
    tape = Tape()
    provider = _FakeProvider()
    adapter = PydanticAIAdapter()
    result = adapter.bind(provider, tape, mode="replay", patch_uuid=False)
    try:
        assert result.injected_fields == ("client",)
        assert provider.client.http_client is result.http_async_client
    finally:
        adapter.teardown()


def test_pydantic_ai_bind_unknown_target_reports_notes():
    adapter = PydanticAIAdapter()
    result = adapter.bind(object(), Tape(), mode="replay", patch_uuid=False)
    try:
        assert result.injected_fields == ()
        assert "nothing was injected" in result.notes
    finally:
        adapter.teardown()


async def test_pydantic_ai_bind_replay_serves_recorded_bytes_bit_exact():
    """The marquee: a run bound in replay mode serves tape bytes for $0, and a
    request that diverges from the tape is caught (proof, not assertion)."""
    tape = Tape()
    tape.append_exchange(b'{"model":"gpt-4o","messages":[]}', b'{"ok":true}')
    agent = _FakeAgent()
    adapter = PydanticAIAdapter()
    result = adapter.bind(agent, tape, mode="replay", patch_uuid=False)
    try:
        client = agent.model.provider.client.http_client
        assert client is result.http_async_client
        resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            content=b'{"model":"gpt-4o","messages":[]}',
        )
        assert resp.status_code == 200
        assert resp.content == b'{"ok":true}'
    finally:
        adapter.teardown()


async def test_pydantic_ai_bind_replay_divergence_on_mismatched_request():
    tape = Tape()
    tape.append_exchange(b"RECORDED", b"RESP")
    agent = _FakeAgent()
    adapter = PydanticAIAdapter()
    result = adapter.bind(agent, tape, mode="replay", patch_uuid=False)
    try:
        with pytest.raises(DivergenceError):
            await result.http_async_client.post("https://api.openai.com/v1/x", content=b"DIFFERENT")
    finally:
        adapter.teardown()


def test_pydantic_ai_bind_replay_installs_uuid_patch_and_teardown_restores():
    tape = Tape()
    tape.draws = [("uuid", "0" * 32), ("uuid", "1" * 32)]
    agent = _FakeAgent()
    adapter = PydanticAIAdapter()
    result = adapter.bind(agent, tape, mode="replay")  # patch_uuid defaults True
    try:
        assert isinstance(result.nondet, ReplayNondet)
        assert uuid.uuid4().hex == "0" * 32
        assert uuid.uuid4().hex == "1" * 32
    finally:
        adapter.teardown()
    assert isinstance(uuid.uuid4(), uuid.UUID)  # real randomness restored


def test_pydantic_ai_bind_provided_nondet_is_used():
    tape = Tape()
    supplied = ReplayNondet([("uuid", "f" * 32)])
    agent = _FakeAgent()
    adapter = PydanticAIAdapter()
    result = adapter.bind(agent, tape, mode="replay", nondet=supplied)
    try:
        assert result.nondet is supplied
        assert uuid.uuid4().hex == "f" * 32
    finally:
        adapter.teardown()


def test_pydantic_ai_bind_record_mode_requires_inner_when_target_unresolvable():
    # object() has no known client attribute, so bind's best-effort record-mode
    # transport lookup finds nothing and build_http_clients' own "record mode
    # requires an inner transport" guard fires -- proving record mode isn't
    # silently downgraded to replay's no-inner contract.
    adapter = PydanticAIAdapter()
    with pytest.raises(ValueError, match="record mode requires an inner transport"):
        adapter.bind(object(), Tape(), mode="record")


# ── on_step: framework-neutral event -> Step ─────────────────────────────────


def test_pydantic_ai_on_step_builds_dag_from_neutral_events():
    adapter = PydanticAIAdapter()
    adapter.on_step({"id": "root", "kind": "agent_run", "name": "run"})
    adapter.on_step(
        {
            "id": "req1",
            "parent_id": "root",
            "kind": "llm",
            "name": "model_request",
            "model": "gpt-4o",
        }
    )
    adapter.on_step({"id": "tool1", "parent_id": "root", "kind": "tool", "name": "call_tools"})

    dag = adapter.dag
    assert [s.step_id for s in dag.steps] == ["root", "req1", "tool1"]
    llm_step = dag.by_id("req1")
    assert llm_step.is_llm()
    assert llm_step.model == "gpt-4o"
    assert llm_step.parent_id == "root"
    assert [s.step_id for s in dag.llm_steps()] == ["req1"]


def test_pydantic_ai_on_step_accepts_alt_key_names():
    adapter = PydanticAIAdapter()
    step = adapter.on_step({"run_id": "s1", "parent_run_id": "p1", "run_type": "llm"})
    assert step.step_id == "s1"
    assert step.parent_id == "p1"
    assert step.kind == "llm"


def test_pydantic_ai_on_step_synthesizes_id_when_missing():
    adapter = PydanticAIAdapter()
    step = adapter.on_step({"kind": "end"})
    assert step.step_id  # non-empty synthesized id
    assert step in adapter.dag.steps


# ── availability guard ────────────────────────────────────────────────────────


def test_require_pydantic_ai_matches_availability():
    if pydantic_ai_available():  # pragma: no cover - only when pydantic-ai is installed
        require_pydantic_ai()
    else:
        with pytest.raises(ImportError, match="pydantic-ai") as excinfo:
            require_pydantic_ai()
        assert excinfo.value.__cause__ is not None


# ── real-framework smoke (skipped cleanly when the framework is absent) ─────────


def test_pydantic_ai_real_package_importorskip():
    pytest.importorskip("pydantic_ai")  # pragma: no cover - needs pydantic-ai extra
    assert pydantic_ai_available()  # pragma: no cover
