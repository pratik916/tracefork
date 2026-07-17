"""Session-fork cost-model tests (tracefork-bge.60): `plan_session_fork`
walks `store.py`'s spawn-edge DAG (`session_tapes`/`spawn_children`) to
partition a session's tapes into recompute-vs-skip for a fork of one
target run, then prices both sets via `blame.py`'s existing
`BudgetGovernor.estimate` (reused unchanged, zero diff to `blame.py`) to
report minimal-recompute $ savings. Deliberately scoped to the
estimator/planner — no re-execution engine — see `session_cost.py`'s
module docstring. All offline/$0."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from tracefork import session_cost
from tracefork.cli import app, session_app
from tracefork.session_cost import SessionForkPlan, plan_session_fork
from tracefork.store import TapeStore
from tracefork.tape import Tape

runner = CliRunner()


def _tape(tag: bytes, n_exchanges: int = 1) -> Tape:
    t = Tape(agent_name="w")
    for i in range(n_exchanges):
        t.append_exchange(b"req-" + tag + str(i).encode(), b"resp-" + tag + str(i).encode())
    return t


def _diamond_session(store: TapeStore) -> str:
    """root -> a, root -> b, a -> c, b -> c. `a` has 2 exchanges, the
    others 1 each — every tape has SOME exchanges, so estimates are
    non-zero across the board."""
    store.save_tape(_tape(b"root", 1), run_id="root")
    store.save_tape(_tape(b"a", 2), run_id="a")
    store.save_tape(_tape(b"b", 1), run_id="b")
    store.save_tape(_tape(b"c", 1), run_id="c")

    session_id = store.create_session(root_run_id="root")
    store.add_spawn_edge(session_id=session_id, parent_run_id="root", child_run_id="a")
    store.add_spawn_edge(session_id=session_id, parent_run_id="root", child_run_id="b")
    store.add_spawn_edge(session_id=session_id, parent_run_id="a", child_run_id="c")
    store.add_spawn_edge(session_id=session_id, parent_run_id="b", child_run_id="c")
    return session_id


# ── (1) diamond DAG-walk partition ───────────────────────────────────────────


def test_plan_session_fork_diamond_partitions_recompute_vs_skip(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        session_id = _diamond_session(store)

        plan = plan_session_fork(store, session_id, "a", k=10, cost_per_fork_usd=0.01)

        assert set(plan.recompute_run_ids) == {"a", "c"}
        assert len(plan.recompute_run_ids) == 2  # deduplicated, no repeats
        assert set(plan.skip_run_ids) == {"root", "b"}
        assert isinstance(plan, SessionForkPlan)
        assert plan.session_id == session_id
        assert plan.target_run_id == "a"
    finally:
        store.close()


# ── (2) recompute-only cost strictly beats the whole-session naive baseline ──


def test_plan_session_fork_recompute_cost_beats_naive_baseline(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        session_id = _diamond_session(store)

        plan = plan_session_fork(store, session_id, "a", k=10, cost_per_fork_usd=0.01)

        assert plan.skip_run_ids  # non-empty precondition
        assert plan.est_usd < plan.est_usd_naive
        assert plan.savings_usd == pytest.approx(plan.est_usd_naive - plan.est_usd)
        assert plan.savings_pct == pytest.approx(plan.savings_usd / plan.est_usd_naive * 100.0)
        assert plan.savings_pct > 0.0
    finally:
        store.close()


# ── (3) forking the session root: nothing to skip, zero savings ─────────────


def test_plan_session_fork_at_root_skips_nothing_zero_savings(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        session_id = _diamond_session(store)

        plan = plan_session_fork(store, session_id, "root", k=10, cost_per_fork_usd=0.01)

        assert plan.skip_run_ids == []
        assert set(plan.recompute_run_ids) == {"root", "a", "b", "c"}
        assert abs(plan.savings_usd) < 1e-9
        assert plan.est_usd == pytest.approx(plan.est_usd_naive)
    finally:
        store.close()


# ── (4) unknown session_id -> KeyError; out-of-session target -> ValueError ──


def test_plan_session_fork_unknown_session_id_raises_key_error(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        with pytest.raises(KeyError):
            plan_session_fork(store, "no-such-session", "root")
    finally:
        store.close()


def test_plan_session_fork_target_not_in_session_raises_value_error(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        session_id = _diamond_session(store)
        store.save_tape(_tape(b"stray", 1), run_id="stray")  # stored, but not in this session

        with pytest.raises(ValueError):
            plan_session_fork(store, session_id, "stray")
    finally:
        store.close()


# ── (5) never mutates a tape; never touches to_bytes()/from_bytes() itself ──


def test_plan_session_fork_never_mutates_tapes_or_touches_tape_bytes(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        session_id = _diamond_session(store)
        before = {rid: store.load_tape(rid).digest() for rid in ("root", "a", "b", "c")}

        plan_session_fork(store, session_id, "a", k=10, cost_per_fork_usd=0.01)

        after = {rid: store.load_tape(rid).digest() for rid in ("root", "a", "b", "c")}
        assert before == after
    finally:
        store.close()

    # session_cost.py delegates all serialization to TapeStore/Tape — its own
    # code (not its prose docstrings) never references .to_bytes()/.from_bytes().
    for fn in (session_cost.plan_session_fork, session_cost._spawn_descendants):
        names = fn.__code__.co_names
        assert "to_bytes" not in names
        assert "from_bytes" not in names


# ── (6) CLI smoke: `tracefork session cost` ──────────────────────────────────

_COST_WIRED = "cost" in {c.name for c in session_app.registered_commands}


@pytest.mark.skipif(
    not _COST_WIRED,
    reason="`session cost` not yet wired into cli.py session_app (see cli_command in bead result)",
)
def test_cli_session_cost_smoke_and_error_exit_codes(tmp_path):
    db = tmp_path / "store.db"
    store = TapeStore(str(db))
    session_id = _diamond_session(store)
    store.close()

    ok = runner.invoke(
        app,
        [
            "session",
            "cost",
            session_id,
            "a",
            "--cost-per-fork-usd",
            "0.01",
            "--store",
            str(db),
        ],
    )
    assert ok.exit_code == 0, ok.output
    import json

    # stdout may carry decorative text around the JSON payload; find the
    # JSON object by locating the first '{' and parsing from there.
    payload = json.loads(ok.output[ok.output.index("{") :])
    for key in (
        "recompute_run_ids",
        "skip_run_ids",
        "est_usd",
        "est_usd_naive",
        "savings_usd",
        "savings_pct",
    ):
        assert key in payload

    bad_session = runner.invoke(
        app, ["session", "cost", "no-such-session", "a", "--store", str(db)]
    )
    assert bad_session.exit_code != 0
    assert "Traceback" not in bad_session.output

    bad_target = runner.invoke(
        app, ["session", "cost", session_id, "no-such-run", "--store", str(db)]
    )
    assert bad_target.exit_code != 0
    assert "Traceback" not in bad_target.output
