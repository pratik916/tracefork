"""tracefork-bge.64: session_chaos.py — session-scope schedule DERIVATION
generalizing transport.chaos_release_order to a session's multi-tape
spawn-lineage graph. Per-tape reordering reuses the REAL, unmodified
chaos_release_order (zero-diff); the new axis is session_sibling_chaos_order,
a cross-sub-agent completion-order permutation. Ships derivation only, not a
multi-tape replay driver — see session_chaos.py's module docstring.

The CLI surface (`tracefork session chaos <id> --seed N`) and the
`tracefork/__init__.py` exports are deferred to cli.py/__init__.py wiring
(forbidden files for this bead) — see the ready-to-paste code handed off in
the wave's structured result. Only store.py/session_chaos.py behavior is
tested here. All offline/$0."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from tracefork.cli import app
from tracefork.session_chaos import (
    _derive_seed,
    session_chaos_release_orders,
    session_sibling_chaos_order,
)
from tracefork.store import TapeStore
from tracefork.tape import Tape
from tracefork.transport import chaos_release_order

runner = CliRunner()


def _tape(n_exchanges: int, tag: str) -> Tape:
    t = Tape(agent_name=tag)
    for i in range(n_exchanges):
        t.append_exchange(f"req-{tag}-{i}".encode(), f"resp-{tag}-{i}".encode())
    return t


# ── session_chaos_release_orders ─────────────────────────────────────────


def test_session_chaos_release_orders_matches_real_chaos_release_order_for_batched_tape(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        tape = _tape(4, "root")
        tape.async_batches = [[1, 2]]
        store.save_tape(tape, run_id="root")
        session_id = store.create_session(root_run_id="root")

        orders = session_chaos_release_orders(store, session_id, seed=7)
        expected = chaos_release_order(tape, _derive_seed(7, "root"))
        assert orders == {"root": expected}
    finally:
        store.close()


def test_session_chaos_release_orders_identity_for_single_exchange_tape_no_batches(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        store.save_tape(_tape(1, "root"), run_id="root")
        session_id = store.create_session(root_run_id="root")

        orders = session_chaos_release_orders(store, session_id, seed=5)
        assert orders == {"root": [0]}
    finally:
        store.close()


def test_session_chaos_release_orders_deterministic_across_calls(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        tape = _tape(4, "root")
        tape.async_batches = [[1, 2]]
        store.save_tape(tape, run_id="root")
        store.save_tape(_tape(2, "child"), run_id="child")
        session_id = store.create_session(root_run_id="root")
        store.add_spawn_edge(session_id=session_id, parent_run_id="root", child_run_id="child")

        first = session_chaos_release_orders(store, session_id, seed=42)
        second = session_chaos_release_orders(store, session_id, seed=42)
        assert first == second
    finally:
        store.close()


def test_session_chaos_release_orders_covers_every_reachable_tape(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        store.save_tape(_tape(2, "root"), run_id="root")
        store.save_tape(_tape(3, "child"), run_id="child")
        session_id = store.create_session(root_run_id="root")
        store.add_spawn_edge(session_id=session_id, parent_run_id="root", child_run_id="child")

        orders = session_chaos_release_orders(store, session_id, seed=1)
        assert set(orders) == {"root", "child"}
        assert orders["root"] == [0, 1]
        assert orders["child"] == [0, 1, 2]
    finally:
        store.close()


# ── session_sibling_chaos_order ──────────────────────────────────────────


def test_session_sibling_chaos_order_permutes_children_and_omits_below_two(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        for rid in ("root", "a", "b", "c", "solo"):
            store.save_tape(_tape(1, rid), run_id=rid)
        session_id = store.create_session(root_run_id="root")
        store.add_spawn_edge(session_id=session_id, parent_run_id="root", child_run_id="a")
        store.add_spawn_edge(session_id=session_id, parent_run_id="root", child_run_id="b")
        store.add_spawn_edge(session_id=session_id, parent_run_id="root", child_run_id="c")
        store.add_spawn_edge(session_id=session_id, parent_run_id="a", child_run_id="solo")

        order = session_sibling_chaos_order(store, session_id, seed=3)
        assert sorted(order["root"]) == ["a", "b", "c"]
        # "a" has exactly one child ("solo") — omitted, not a 1-element list.
        assert "a" not in order
        assert "b" not in order
        assert "c" not in order
    finally:
        store.close()


def test_session_sibling_chaos_order_deterministic_across_calls(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        for rid in ("root", "a", "b"):
            store.save_tape(_tape(1, rid), run_id=rid)
        session_id = store.create_session(root_run_id="root")
        store.add_spawn_edge(session_id=session_id, parent_run_id="root", child_run_id="a")
        store.add_spawn_edge(session_id=session_id, parent_run_id="root", child_run_id="b")

        first = session_sibling_chaos_order(store, session_id, seed=99)
        second = session_sibling_chaos_order(store, session_id, seed=99)
        assert first == second
    finally:
        store.close()


def test_session_sibling_chaos_order_empty_when_no_fanout(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        store.save_tape(_tape(1, "root"), run_id="root")
        session_id = store.create_session(root_run_id="root")
        assert session_sibling_chaos_order(store, session_id, seed=1) == {}
    finally:
        store.close()


# ── store.py: session_spawn_children session isolation ──────────────────


def test_session_spawn_children_isolated_across_sessions_sharing_a_parent(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        for rid in ("root", "child-a", "child-b"):
            store.save_tape(_tape(1, rid), run_id=rid)

        session_1 = store.create_session(root_run_id="root")
        session_2 = store.create_session(root_run_id="root")
        store.add_spawn_edge(session_id=session_1, parent_run_id="root", child_run_id="child-a")
        store.add_spawn_edge(session_id=session_2, parent_run_id="root", child_run_id="child-b")

        assert store.session_spawn_children(session_1, "root") == ["child-a"]
        assert store.session_spawn_children(session_2, "root") == ["child-b"]
    finally:
        store.close()


def test_session_spawn_children_empty_for_leaf(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        store.save_tape(_tape(1, "root"), run_id="root")
        session_id = store.create_session(root_run_id="root")
        assert store.session_spawn_children(session_id, "root") == []
    finally:
        store.close()


# ── CLI: session chaos ────────────────────────────────────────────────────


def test_cli_session_chaos_exits_zero_and_prints_both_keys(tmp_path):
    db = tmp_path / "store.db"
    store = TapeStore(str(db))
    try:
        tape = _tape(4, "root")
        tape.async_batches = [[1, 2]]
        store.save_tape(tape, run_id="root")
        store.save_tape(_tape(2, "child"), run_id="child")
        session_id = store.create_session(root_run_id="root")
        store.add_spawn_edge(session_id=session_id, parent_run_id="root", child_run_id="child")
    finally:
        store.close()

    result = runner.invoke(app, ["session", "chaos", session_id, "--seed", "5", "--store", str(db)])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert "per_tape_release_orders" in data
    assert "sibling_chaos_order" in data
    assert set(data["per_tape_release_orders"]) == {"root", "child"}


def test_cli_session_chaos_unknown_session_exits_nonzero(tmp_path):
    db = tmp_path / "store.db"
    TapeStore(str(db)).close()

    result = runner.invoke(
        app, ["session", "chaos", "no-such-session", "--seed", "1", "--store", str(db)]
    )
    assert result.exit_code != 0
