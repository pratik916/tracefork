"""tracefork-bge.58: cross_tape_blame.py — a read-only, session-scoped
aggregation VIEW over already-persisted per-tape causal data. RunRef, the
`spawn_step_index`-aware session_topological_order interleave, and
cross_tape_causal_edges' aggregation of store.py's existing causal_edges
rows. See cross_tape_blame.py's module docstring for exactly what full-scope
tracefork-bge.58 (a genuine joint cross-tape coalition-execution engine) is
NOT attempted here.

The CLI surface (`tracefork session <cmd> --json`) is deferred to cli.py
wiring (a forbidden file for this bead) — see the ready-to-paste code handed
off in the wave's structured result. Only store.py/cross_tape_blame.py
behavior is tested here. All offline/$0."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from tracefork.blame import BlameReport, CIMethod, FlipRateResult
from tracefork.cli import app
from tracefork.cross_tape_blame import RunRef, cross_tape_causal_edges, session_topological_order
from tracefork.store import TapeStore
from tracefork.tape import Tape

runner = CliRunner()


def _tape(n_steps: int, tag: str) -> Tape:
    t = Tape(agent_name=tag)
    for i in range(n_steps):
        t.append_exchange(f"req-{tag}-{i}".encode(), f"resp-{tag}-{i}".encode())
    return t


def _blame_report(steps: list[tuple[int, bool]]) -> BlameReport:
    results = [
        FlipRateResult(
            step_index=idx,
            flip_rate=0.9 if responsible else 0.1,
            ci_lo=0.5 if responsible else 0.0,
            ci_hi=1.0 if responsible else 0.3,
            flips=9 if responsible else 1,
            trials=10,
            valid_trials=10,
            p_value=0.01 if responsible else 0.9,
            q_value=0.02 if responsible else 0.9,
            responsible=responsible,
        )
        for idx, responsible in steps
    ]
    return BlameReport(
        results=results, k=10, total_forks=len(results) * 10, ci_method=CIMethod.WILSON
    )


# ── RunRef ────────────────────────────────────────────────────────────────


def test_run_ref_is_hashable_and_usable_as_dict_key():
    a = RunRef("root", 0)
    b = RunRef("root", 0)
    c = RunRef("root", 1)
    assert a == b
    assert hash(a) == hash(b)
    assert a != c

    d = {a: "value"}
    assert d[b] == "value"  # equal RunRefs collide to the same key


# ── session_topological_order ────────────────────────────────────────────


def test_session_topological_order_splices_child_at_spawn_step_index(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        store.save_tape(_tape(3, "root"), run_id="root")
        store.save_tape(_tape(2, "child"), run_id="child")
        session_id = store.create_session(root_run_id="root")
        store.add_spawn_edge(
            session_id=session_id,
            parent_run_id="root",
            child_run_id="child",
            spawn_step_index=1,
        )

        order = session_topological_order(store, session_id)
        assert order == [
            RunRef("root", 0),
            RunRef("root", 1),
            RunRef("child", 0),
            RunRef("child", 1),
            RunRef("root", 2),
        ]
    finally:
        store.close()


def test_session_topological_order_falls_back_to_after_parent_when_step_index_none(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        store.save_tape(_tape(2, "root"), run_id="root")
        store.save_tape(_tape(2, "child"), run_id="child")
        session_id = store.create_session(root_run_id="root")
        # No spawn_step_index passed — every pre-existing caller's behavior.
        store.add_spawn_edge(session_id=session_id, parent_run_id="root", child_run_id="child")

        order = session_topological_order(store, session_id)
        assert order == [
            RunRef("root", 0),
            RunRef("root", 1),
            RunRef("child", 0),
            RunRef("child", 1),
        ]
    finally:
        store.close()


def test_session_topological_order_single_tape_session_is_its_own_steps(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        store.save_tape(_tape(2, "root"), run_id="root")
        session_id = store.create_session(root_run_id="root")
        assert session_topological_order(store, session_id) == [
            RunRef("root", 0),
            RunRef("root", 1),
        ]
    finally:
        store.close()


# ── cross_tape_causal_edges ───────────────────────────────────────────────


def test_cross_tape_causal_edges_tags_run_id_and_orders_by_topological_position(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        store.save_tape(_tape(2, "root"), run_id="root")
        store.save_tape(_tape(2, "child"), run_id="child")
        session_id = store.create_session(root_run_id="root")
        store.add_spawn_edge(
            session_id=session_id,
            parent_run_id="root",
            child_run_id="child",
            spawn_step_index=0,
        )

        store.save_blame_report("root", _blame_report([(0, False), (1, True)]))
        store.save_blame_report("child", _blame_report([(0, True), (1, False)]))

        edges = cross_tape_causal_edges(store, session_id)
        # Topological order: root[0], child[0], child[1], root[1].
        assert [(e["run_id"], e["step_index"]) for e in edges] == [
            ("root", 0),
            ("child", 0),
            ("child", 1),
            ("root", 1),
        ]
        # Each edge carries its own run_id/responsible verbatim from
        # causal_edges_for_run — no separate tagging step.
        assert edges[1]["run_id"] == "child" and edges[1]["responsible"] is True
        assert edges[3]["run_id"] == "root" and edges[3]["responsible"] is True
    finally:
        store.close()


def test_cross_tape_causal_edges_empty_when_nothing_saved(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        store.save_tape(_tape(1, "root"), run_id="root")
        session_id = store.create_session(root_run_id="root")
        assert cross_tape_causal_edges(store, session_id) == []
    finally:
        store.close()


# ── store.py: spawn_step_index / spawn_edges_for_session ────────────────────


def test_add_spawn_edge_default_spawn_step_index_is_none(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        store.save_tape(_tape(1, "root"), run_id="root")
        store.save_tape(_tape(1, "child"), run_id="child")
        session_id = store.create_session(root_run_id="root")
        store.add_spawn_edge(session_id=session_id, parent_run_id="root", child_run_id="child")

        edges = store.spawn_edges_for_session(session_id)
        assert len(edges) == 1
        assert edges[0]["spawn_step_index"] is None
        assert edges[0]["parent_run_id"] == "root"
        assert edges[0]["child_run_id"] == "child"
    finally:
        store.close()


def test_spawn_edges_for_session_round_trips_spawn_step_index(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        store.save_tape(_tape(3, "root"), run_id="root")
        store.save_tape(_tape(1, "child"), run_id="child")
        session_id = store.create_session(root_run_id="root")
        store.add_spawn_edge(
            session_id=session_id,
            parent_run_id="root",
            child_run_id="child",
            spawn_step_index=2,
        )

        edges = store.spawn_edges_for_session(session_id)
        assert edges[0]["spawn_step_index"] == 2
    finally:
        store.close()


# ── CLI: session cross-blame ─────────────────────────────────────────────


def test_cli_session_cross_blame_json_exits_zero_and_parses_ordered_edges(tmp_path):
    db = tmp_path / "store.db"
    store = TapeStore(str(db))
    try:
        store.save_tape(_tape(2, "root"), run_id="root")
        store.save_tape(_tape(2, "child"), run_id="child")
        session_id = store.create_session(root_run_id="root")
        store.add_spawn_edge(
            session_id=session_id,
            parent_run_id="root",
            child_run_id="child",
            spawn_step_index=0,
        )
        store.save_blame_report("root", _blame_report([(0, False), (1, True)]))
        store.save_blame_report("child", _blame_report([(0, True), (1, False)]))
    finally:
        store.close()

    result = runner.invoke(
        app, ["session", "cross-blame", session_id, "--json", "--store", str(db)]
    )
    assert result.exit_code == 0, result.output
    edges = json.loads(result.output)
    # Same topological order as the direct-call test above: root[0], child[0],
    # child[1], root[1].
    assert [(e["run_id"], e["step_index"]) for e in edges] == [
        ("root", 0),
        ("child", 0),
        ("child", 1),
        ("root", 1),
    ]


def test_cli_session_cross_blame_table_output_exits_zero_without_json(tmp_path):
    db = tmp_path / "store.db"
    store = TapeStore(str(db))
    try:
        store.save_tape(_tape(1, "root"), run_id="root")
        session_id = store.create_session(root_run_id="root")
        store.save_blame_report("root", _blame_report([(0, True)]))
    finally:
        store.close()

    result = runner.invoke(app, ["session", "cross-blame", session_id, "--store", str(db)])
    assert result.exit_code == 0, result.output
    assert "root" in result.output
    assert "responsible=True" in result.output


def test_cli_session_cross_blame_unknown_session_exits_nonzero(tmp_path):
    db = tmp_path / "store.db"
    TapeStore(str(db)).close()

    result = runner.invoke(app, ["session", "cross-blame", "no-such-session", "--store", str(db)])
    assert result.exit_code != 0


# ── CLI: session spawn --spawn-step ──────────────────────────────────────


def test_cli_session_spawn_step_round_trips_via_spawn_edges_for_session(tmp_path):
    db = tmp_path / "store.db"
    store = TapeStore(str(db))
    try:
        store.save_tape(_tape(3, "root"), run_id="root")
        store.save_tape(_tape(1, "child"), run_id="child")
        session_id = store.create_session(root_run_id="root")
    finally:
        store.close()

    result = runner.invoke(
        app,
        [
            "session",
            "spawn",
            session_id,
            "root",
            "child",
            "--spawn-step",
            "2",
            "--store",
            str(db),
        ],
    )
    assert result.exit_code == 0, result.output

    store = TapeStore(str(db))
    try:
        edges = store.spawn_edges_for_session(session_id)
        assert edges[0]["spawn_step_index"] == 2
    finally:
        store.close()


def test_cli_session_spawn_without_spawn_step_still_stores_none(tmp_path):
    """Omitting --spawn-step (every pre-existing caller's shape) keeps
    storing NULL, exactly as before this option existed."""
    db = tmp_path / "store.db"
    store = TapeStore(str(db))
    try:
        store.save_tape(_tape(1, "root"), run_id="root")
        store.save_tape(_tape(1, "child"), run_id="child")
        session_id = store.create_session(root_run_id="root")
    finally:
        store.close()

    result = runner.invoke(
        app, ["session", "spawn", session_id, "root", "child", "--store", str(db)]
    )
    assert result.exit_code == 0, result.output

    store = TapeStore(str(db))
    try:
        edges = store.spawn_edges_for_session(session_id)
        assert edges[0]["spawn_step_index"] is None
    finally:
        store.close()
