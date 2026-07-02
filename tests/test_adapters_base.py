"""Framework adapter base seam — offline, no framework installed.

Exercises the framework-neutral half of the adapter protocol: the ``Step`` /
``StepDAG`` overlay, the run-tree normalizer, the shared httpx-client builder
(reusing ``transport.py``), the replay uuid patch, and the registry. Nothing
here imports langchain/langgraph — this is the logic that must work with no
framework present at all.
"""

import uuid

import httpx
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
from tracefork.nondet import ReplayNondet
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
    assert isinstance(sync_c, httpx.Client)
    assert isinstance(async_c, httpx.AsyncClient)
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
