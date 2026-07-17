"""corpus.py tests -- build_corpus_blame_index/detect_regressions aggregation
over TapeStore.list_runs()/causal_edges_for_run(), plus the `tracefork
corpus-blame` CLI wiring. All offline/$0.

The CLI command this bead adds lives in cli.py, an orchestrator-owned file
this bead does not edit directly (see this bead's `cli_command` result
field for the exact code to paste in). The CLI test below skips gracefully
until that wiring lands, so this file stays green pre-integration and
becomes a real enforced check the moment `corpus-blame` is registered.
"""

import json

import pytest
from typer.testing import CliRunner

from tracefork.blame import BlameReport, CIMethod, FlipRateResult, ShapleyReport, ShapleyResult
from tracefork.cli import app
from tracefork.corpus import build_corpus_blame_index, detect_regressions
from tracefork.store import TapeStore
from tracefork.tape import Tape

runner = CliRunner()


def _tape(agent_name: str, tag: str) -> Tape:
    t = Tape(agent_name=agent_name)
    t.append_exchange(f"req-{tag}".encode(), f"resp-{tag}".encode())
    return t


def _save_blame(
    store: TapeStore,
    run_id: str,
    *,
    agent_name: str,
    created_at: str,
    step_index: int,
    flip_rate: float,
    responsible: bool = False,
) -> None:
    store.save_tape(_tape(agent_name, run_id), run_id=run_id, created_at=created_at)
    report = BlameReport(
        results=[
            FlipRateResult(
                step_index=step_index,
                flip_rate=flip_rate,
                ci_lo=max(0.0, flip_rate - 0.1),
                ci_hi=min(1.0, flip_rate + 0.1),
                flips=int(flip_rate * 10),
                trials=10,
                valid_trials=10,
                responsible=responsible,
            )
        ],
        k=10,
        total_forks=10,
        ci_method=CIMethod.WILSON,
    )
    store.save_blame_report(run_id, report, created_at=created_at)


def _save_shapley(
    store: TapeStore,
    run_id: str,
    *,
    agent_name: str,
    created_at: str,
    step_index: int,
    shapley_value: float,
) -> None:
    store.save_tape(_tape(agent_name, run_id), run_id=run_id, created_at=created_at)
    report = ShapleyReport(
        results=[
            ShapleyResult(
                step_index=step_index,
                shapley_value=shapley_value,
                ci_lo=0.0,
                ci_hi=1.0,
                n_samples=5,
            )
        ],
        n_permutation_samples=5,
        k=10,
        total_forks=50,
    )
    store.save_shapley_report(run_id, report, created_at=created_at)


# ── build_corpus_blame_index ────────────────────────────────────────────────


def test_build_corpus_blame_index_aggregates_across_runs_and_methods(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        _save_blame(
            store,
            "run-a",
            agent_name="agent-a",
            created_at="2026-01-01T00:00:00+00:00",
            step_index=0,
            flip_rate=0.9,
            responsible=True,
        )
        _save_shapley(
            store,
            "run-b",
            agent_name="agent-b",
            created_at="2026-01-02T00:00:00+00:00",
            step_index=1,
            shapley_value=0.5,
        )

        index = build_corpus_blame_index(store, top_n=20)

        assert index.run_count == 2
        assert index.edge_count == 2
        assert index.by_method == {"blame": 1, "shapley": 1}
        assert [s.run_id for s in index.top_responsible] == ["run-a", "run-b"]
        assert index.top_responsible[0].score == 0.9
        assert index.top_responsible[0].agent_name == "agent-a"
        assert index.top_responsible[1].score == 0.5
        assert index.top_responsible[1].agent_name == "agent-b"
    finally:
        store.close()


def test_build_corpus_blame_index_caps_at_top_n_sorted_descending(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        for i in range(5):
            _save_blame(
                store,
                f"run-{i}",
                agent_name="agent-a",
                created_at=f"2026-01-0{i + 1}T00:00:00+00:00",
                step_index=0,
                flip_rate=0.1 * (i + 1),
            )

        index = build_corpus_blame_index(store, top_n=3)

        assert index.run_count == 5
        assert index.edge_count == 5
        assert len(index.top_responsible) == 3
        scores = [s.score for s in index.top_responsible]
        assert scores == sorted(scores, reverse=True)
        assert scores[0] == pytest.approx(0.5)
    finally:
        store.close()


# ── detect_regressions ──────────────────────────────────────────────────────


def test_detect_regressions_flags_latest_outlier_run(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        history_rates = [0.10, 0.12, 0.09, 0.11]
        for i, rate in enumerate(history_rates):
            _save_blame(
                store,
                f"run-h{i}",
                agent_name="agent-a",
                created_at=f"2026-01-0{i + 1}T00:00:00+00:00",
                step_index=2,
                flip_rate=rate,
            )
        _save_blame(
            store,
            "run-outlier",
            agent_name="agent-a",
            created_at="2026-01-05T00:00:00+00:00",
            step_index=2,
            flip_rate=0.9,
        )

        flags = detect_regressions(store, method="blame", z_threshold=2.0, min_history=3)

        assert len(flags) == 1
        flag = flags[0]
        assert flag.agent_name == "agent-a"
        assert flag.step_index == 2
        assert flag.method == "blame"
        assert flag.run_id == "run-outlier"
        assert flag.value == pytest.approx(0.9)
        assert flag.z_score >= 2.0
    finally:
        store.close()


def test_detect_regressions_stable_history_returns_no_flags(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        rates = [0.10, 0.12, 0.09, 0.11, 0.10]
        for i, rate in enumerate(rates):
            _save_blame(
                store,
                f"run-s{i}",
                agent_name="agent-a",
                created_at=f"2026-01-0{i + 1}T00:00:00+00:00",
                step_index=2,
                flip_rate=rate,
            )

        flags = detect_regressions(store, method="blame", z_threshold=2.0, min_history=3)

        assert flags == []
    finally:
        store.close()


def test_detect_regressions_skips_group_with_too_little_history(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        _save_blame(
            store,
            "run-x0",
            agent_name="agent-a",
            created_at="2026-01-01T00:00:00+00:00",
            step_index=0,
            flip_rate=0.1,
        )
        _save_blame(
            store,
            "run-x1",
            agent_name="agent-a",
            created_at="2026-01-02T00:00:00+00:00",
            step_index=0,
            flip_rate=0.9,
        )

        # Only 2 total points (1 history point) for min_history=3 -- too
        # little signal to judge, must not raise (zero-division guard) and
        # must not flag anything.
        flags = detect_regressions(store, method="blame", min_history=3)

        assert flags == []
    finally:
        store.close()


def test_detect_regressions_only_considers_requested_method(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        for i, rate in enumerate([0.10, 0.11, 0.09, 0.10]):
            _save_shapley(
                store,
                f"run-sh{i}",
                agent_name="agent-a",
                created_at=f"2026-01-0{i + 1}T00:00:00+00:00",
                step_index=0,
                shapley_value=rate,
            )
        # A wild blame outlier for the SAME (agent, step) must not leak into
        # a shapley-only regression scan.
        _save_blame(
            store,
            "run-blame-outlier",
            agent_name="agent-a",
            created_at="2026-01-05T00:00:00+00:00",
            step_index=0,
            flip_rate=0.95,
        )

        assert detect_regressions(store, method="shapley", min_history=3) == []
    finally:
        store.close()


# ── CLI wiring (tracefork corpus-blame) ─────────────────────────────────────


def _corpus_blame_registered() -> bool:
    return any(
        c.callback is not None and c.callback.__name__ == "corpus_blame"
        for c in app.registered_commands
    )


def test_corpus_blame_cli_exits_zero_and_writes_json(tmp_path):
    if not _corpus_blame_registered():
        pytest.skip("corpus-blame not yet wired into cli.py (see cli_command)")

    db = tmp_path / "store.db"
    store = TapeStore(str(db))
    _save_blame(
        store,
        "run-1",
        agent_name="agent-a",
        created_at="2026-01-01T00:00:00+00:00",
        step_index=0,
        flip_rate=0.5,
    )
    store.close()

    output = tmp_path / "corpus.json"
    result = runner.invoke(app, ["corpus-blame", "--store", str(db), "--output", str(output)])
    assert result.exit_code == 0, result.output

    data = json.loads(output.read_text())
    assert {"run_count", "edge_count", "regressions"}.issubset(data.keys())
    assert data["run_count"] == 1
    assert data["edge_count"] == 1
