"""causal_edges persistence tests — save_blame_report/save_shapley_report round-trip,
cited_by (derived from the existing branches table, no new citation concept),
causal_closure (BFS over the fork graph's branches.parent_run_id chains), and the
CLI's additive wiring after `tracefork blame`. All offline/$0."""

from typer.testing import CliRunner

from tracefork.blame import BlameReport, CIMethod, FlipRateResult, ShapleyReport, ShapleyResult
from tracefork.cli import app
from tracefork.store import StorageBackend, TapeStore
from tracefork.tape import Tape

runner = CliRunner()


def _small_tape(tag: bytes = b"x") -> Tape:
    t = Tape(agent_name="w")
    t.append_exchange(b"req-" + tag, b"resp-" + tag)
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


# ── save_blame_report / causal_edges_for_run ────────────────────────────────


def test_save_blame_report_round_trips_every_field(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        store.save_tape(_small_tape(), run_id="run-1")
        report = _blame_report([(0, True), (1, False)])

        edge_ids = store.save_blame_report("run-1", report)
        assert len(edge_ids) == 2

        edges = store.causal_edges_for_run("run-1")
        assert len(edges) == 2
        by_step = {e["step_index"]: e for e in edges}
        assert by_step[0]["flip_rate"] == report.results[0].flip_rate
        assert by_step[0]["ci_lo"] == report.results[0].ci_lo
        assert by_step[0]["ci_hi"] == report.results[0].ci_hi
        assert by_step[0]["q_value"] == report.results[0].q_value
        assert by_step[0]["p_value"] == report.results[0].p_value
        assert by_step[0]["responsible"] is True
        assert by_step[1]["responsible"] is False
        assert by_step[0]["method"] == "blame"
        assert by_step[0]["ci_method"] == "wilson"
    finally:
        store.close()


def test_save_blame_report_replaces_not_duplicates(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        store.save_tape(_small_tape(), run_id="run-2")
        store.save_blame_report("run-2", _blame_report([(0, True), (1, False)]))
        store.save_blame_report("run-2", _blame_report([(0, False)]))

        edges = store.causal_edges_for_run("run-2")
        assert len(edges) == 1
        assert edges[0]["step_index"] == 0
        assert edges[0]["responsible"] is False
    finally:
        store.close()


def test_save_shapley_report_round_trips_necessity_and_sufficiency(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        store.save_tape(_small_tape(), run_id="run-3")
        result = ShapleyResult(
            step_index=0,
            shapley_value=0.42,
            ci_lo=0.1,
            ci_hi=0.7,
            n_samples=5,
            necessity=True,
            necessity_score=0.42,
            sufficiency=False,
            sufficiency_score=0.05,
        )
        report = ShapleyReport(results=[result], n_permutation_samples=5, k=10, total_forks=50)

        store.save_shapley_report("run-3", report)
        edges = store.causal_edges_for_run("run-3")
        assert len(edges) == 1
        edge = edges[0]
        assert edge["method"] == "shapley"
        assert edge["shapley_value"] == 0.42
        assert edge["necessity"] is True
        assert edge["sufficiency"] is False
    finally:
        store.close()


# ── cited_by ─────────────────────────────────────────────────────────────────


def test_cited_by_returns_only_branches_at_that_step(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        run_id = store.save_tape(_small_tape(), run_id="run-4")
        b0 = store.save_branch(
            parent_run_id=run_id, divergence_step=0, delta_tape=_small_tape(b"b0")
        )
        b1 = store.save_branch(
            parent_run_id=run_id, divergence_step=1, delta_tape=_small_tape(b"b1")
        )

        assert store.cited_by(run_id, 0) == [b0]
        assert store.cited_by(run_id, 1) == [b1]
        assert store.cited_by(run_id, 2) == []
    finally:
        store.close()


# ── causal_closure ───────────────────────────────────────────────────────────


def test_causal_closure_unions_responsible_steps_across_two_generations(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        # Generation 1: run A, blamed, step 0 responsible.
        store.save_tape(_small_tape(b"a"), run_id="run-a")
        store.save_blame_report("run-a", _blame_report([(0, True), (1, False)]))

        # Fork run A at step 0 into a branch, then promote that branch to its
        # own re-blamable tape ("tape B") using the branch_id as its run_id —
        # the convention `causal_closure` walks.
        branch_tape = _small_tape(b"branch")
        branch_id = store.save_branch(
            parent_run_id="run-a", divergence_step=0, delta_tape=branch_tape
        )
        store.save_tape(branch_tape, run_id=branch_id)

        # Generation 2: run B (the promoted branch), blamed, step 3 responsible.
        store.save_blame_report(branch_id, _blame_report([(3, True)]))

        closure = store.causal_closure("run-a")
        assert {(e["run_id"], e["step_index"]) for e in closure} == {
            ("run-a", 0),
            (branch_id, 3),
        }
    finally:
        store.close()


def test_causal_closure_ignores_branches_never_promoted_to_a_tape(tmp_path):
    """A branch that was only saved via `save_branch` (never separately
    `save_tape`'d under its branch_id) has no further generation to walk into —
    the closure stops at the parent's own responsible edges."""
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        store.save_tape(_small_tape(b"a"), run_id="run-a")
        store.save_blame_report("run-a", _blame_report([(0, True)]))
        store.save_branch(parent_run_id="run-a", divergence_step=0, delta_tape=_small_tape(b"x"))

        closure = store.causal_closure("run-a")
        assert {(e["run_id"], e["step_index"]) for e in closure} == {("run-a", 0)}
    finally:
        store.close()


# ── StorageBackend protocol conformance ─────────────────────────────────────


def test_tape_store_still_satisfies_storage_backend_protocol(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        assert isinstance(store, StorageBackend)
    finally:
        store.close()


# ── CLI wiring ───────────────────────────────────────────────────────────────


def test_blame_cli_persists_causal_edges(tmp_path, monkeypatch):
    """Wiring test: `tracefork blame` additionally calls `save_blame_report` after
    its existing JSON write. `BlameEngine.rank` is monkeypatched to a canned
    report so the assertion covers the CLI's wiring, not the (already-tested-
    elsewhere) blame algorithm itself — no network call is made."""
    from tracefork import validate as validate_mod
    from tracefork.blame import BlameEngine

    db = tmp_path / "store.db"
    store = TapeStore(str(db))
    run_id = store.save_tape(validate_mod._record_clean_tape(), run_id="cli-run")
    store.close()

    canned = _blame_report([(0, True), (1, False)])

    def _fake_rank(*args, **kwargs):
        return canned

    monkeypatch.setattr(BlameEngine, "rank", staticmethod(_fake_rank))

    result = runner.invoke(
        app,
        [
            "blame",
            run_id,
            "--agent",
            "tracefork.validate:synthetic_agent",
            "--store",
            str(db),
        ],
    )
    assert result.exit_code == 0, result.output

    store2 = TapeStore(str(db))
    try:
        edges = store2.causal_edges_for_run(run_id)
        assert len(edges) == 2
    finally:
        store2.close()
