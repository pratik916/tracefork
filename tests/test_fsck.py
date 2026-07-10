"""fsck.py tests: `store_fsck()`'s read-only structural checks over a
`TapeStore` — decode failures and orphaned branch parents, distinct from
`replay.py`'s replay-fidelity verification. All offline/$0."""

from __future__ import annotations

import sqlite3

from tracefork.fsck import store_fsck
from tracefork.store import TapeStore
from tracefork.tape import Tape


def _tape(tag: str = "x") -> Tape:
    t = Tape(agent_name=f"agent-{tag}")
    t.append_exchange(f"req-{tag}".encode(), f"resp-{tag}".encode())
    return t


def test_store_fsck_on_healthy_store_reports_all_ok(tmp_path):
    db = tmp_path / "store.db"
    store = TapeStore(str(db))
    run_a = store.save_tape(_tape("a"), run_id="run-a")
    run_b = store.save_tape(_tape("b"), run_id="run-b")
    store.save_branch(
        parent_run_id=run_a,
        divergence_step=0,
        delta_tape=_tape("a-branch"),
        mutation_desc="test branch",
    )
    store.close()

    store = TapeStore(str(db))
    result = store_fsck(store)
    store.close()

    assert result.all_ok is True
    assert len(result.rows) == 3  # 2 tapes + 1 branch
    assert {r.id for r in result.rows if r.kind == "tape"} == {run_a, run_b}


def test_store_fsck_flags_truncated_tape_others_still_pass(tmp_path):
    db = tmp_path / "store.db"
    store = TapeStore(str(db))
    run_a = store.save_tape(_tape("a"), run_id="run-a")
    run_b = store.save_tape(_tape("b"), run_id="run-b")
    store.close()

    con = sqlite3.connect(str(db))
    con.execute("UPDATE tapes SET tape_bytes=? WHERE run_id=?", (b"\x00not-a-tape", run_a))
    con.commit()
    con.close()

    store = TapeStore(str(db))
    result = store_fsck(store)
    store.close()

    by_id = {r.id: r for r in result.rows if r.kind == "tape"}
    assert by_id[run_a].passed is False
    assert "decode error" in by_id[run_a].reason
    assert by_id[run_b].passed is True
    assert result.all_ok is False


def test_store_fsck_flags_orphaned_branch_parent(tmp_path):
    db = tmp_path / "store.db"
    store = TapeStore(str(db))
    run_a = store.save_tape(_tape("a"), run_id="run-a")
    branch_id = store.save_branch(
        parent_run_id=run_a,
        divergence_step=0,
        delta_tape=_tape("a-branch"),
        mutation_desc="test branch",
    )
    store.close()

    # Force-delete the parent tape row directly, bypassing the FK
    # (foreign_keys=OFF) -- exactly the scenario `store.py`'s prune() avoids
    # by archiving branches before tapes, but raw SQL can still produce.
    con = sqlite3.connect(str(db))
    con.execute("PRAGMA foreign_keys=OFF")
    con.execute("DELETE FROM tapes WHERE run_id=?", (run_a,))
    con.commit()
    con.close()

    store = TapeStore(str(db))
    # load_branch alone still succeeds -- it never joins back to tapes.
    assert store.load_branch(branch_id)["branch_id"] == branch_id

    result = store_fsck(store)
    store.close()

    orphan_rows = [r for r in result.rows if r.kind == "branch" and r.id == branch_id]
    assert len(orphan_rows) == 1
    assert orphan_rows[0].passed is False
    assert "orphaned parent" in orphan_rows[0].reason
    assert result.all_ok is False


def test_store_fsck_on_empty_store_reports_all_ok(tmp_path):
    db = tmp_path / "store.db"
    store = TapeStore(str(db))
    result = store_fsck(store)
    store.close()

    assert result.rows == []
    assert result.all_ok is True
