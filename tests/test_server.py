"""FastAPI TestClient tests for `server.py`'s JSON/HTML endpoints that don't
already have a dedicated test file (see `test_fork_endpoint.py`,
`test_branch_queries.py`, `test_live.py`, `test_otel_locate.py`,
`test_runs_page.py`, `test_sessions.py`, `test_cost_profile.py` for the rest).

Covers two things:

  1. `GET /api/checkpoint/tail?path=` path confinement + opt-in (security
     fix): the endpoint used to take an unvalidated filesystem path on an
     unauthenticated GET and, via `open_sqlite`'s `PRAGMA journal_mode=WAL`
     plus `checkpoint.py`'s `executescript`-based "read-only" helpers, WRITE
     to whatever file it named. Nothing is tailable now unless the operator
     opts in via `init_checkpoint_dirs` (the same posture
     `fork_allowlist.py` already establishes for the click-to-fork
     endpoints), and the resolved path must stay confined to an allowlisted
     directory.

  2. `GET /api/run/{run_id}` populating `blame`/`shapley`/`branch_details`
     from data the endpoint already loads (`causal_edges`, `branches`) —
     mirroring `cli.py`'s `report` command's run_id path, which already
     derives these for free.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tracefork.blame import BlameReport, CIMethod, FlipRateResult, ShapleyReport, ShapleyResult
from tracefork.checkpoint import CheckpointWriter
from tracefork.server import app as fastapi_app
from tracefork.server import init_checkpoint_dirs, init_store
from tracefork.store import TapeStore
from tracefork.tape import Tape


@pytest.fixture(autouse=True)
def _reset_checkpoint_dirs():
    """Every test starts from the documented default: nothing allowlisted."""
    init_checkpoint_dirs([])
    yield
    init_checkpoint_dirs([])


def _small_tape(tag: bytes = b"x") -> Tape:
    t = Tape(agent_name="w")
    t.append_exchange(b"req-" + tag, b"resp-" + tag)
    return t


# ── GET /api/checkpoint/tail path confinement + opt-in ──────────────────────


def test_tail_checkpoint_403_by_default_even_for_a_real_checkpoint_file(tmp_path):
    path = str(tmp_path / "checkpoint.sqlite")
    writer = CheckpointWriter(path, agent_name="a")
    writer.append_exchange(b"req1", b"resp1")
    writer.finalize(Tape(agent_name="a"))

    client = TestClient(fastapi_app)
    resp = client.get("/api/checkpoint/tail", params={"path": path})
    assert resp.status_code == 403
    assert "allowlisted" in resp.json()["detail"]


def test_tail_checkpoint_403_for_path_outside_allowlisted_dir(tmp_path):
    allowed_dir = tmp_path / "checkpoints"
    allowed_dir.mkdir()
    third_party = tmp_path / "someone-elses.sqlite"
    writer = CheckpointWriter(str(third_party), agent_name="a")
    writer.append_exchange(b"req1", b"resp1")
    writer.finalize(Tape(agent_name="a"))

    init_checkpoint_dirs([str(allowed_dir)])
    client = TestClient(fastapi_app)
    resp = client.get("/api/checkpoint/tail", params={"path": str(third_party)})
    assert resp.status_code == 403


def test_tail_checkpoint_200_when_opted_in_and_confined(tmp_path):
    allowed_dir = tmp_path / "checkpoints"
    allowed_dir.mkdir()
    path = str(allowed_dir / "checkpoint.sqlite")
    writer = CheckpointWriter(path, agent_name="a")
    writer.append_exchange(b"req1", b"resp1")
    writer.finalize(Tape(agent_name="a"))

    init_checkpoint_dirs([str(allowed_dir)])
    client = TestClient(fastapi_app)
    resp = client.get("/api/checkpoint/tail", params={"path": path})
    assert resp.status_code == 200, resp.text
    assert "text/event-stream" in resp.headers["content-type"]
    assert "event: exchange" in resp.text
    assert "event: done" in resp.text


def test_tail_checkpoint_still_404s_on_missing_path_inside_allowed_dir(tmp_path):
    allowed_dir = tmp_path / "checkpoints"
    allowed_dir.mkdir()
    init_checkpoint_dirs([str(allowed_dir)])
    client = TestClient(fastapi_app)
    resp = client.get("/api/checkpoint/tail", params={"path": str(allowed_dir / "nope.sqlite")})
    assert resp.status_code == 404


def test_tail_checkpoint_does_not_mutate_a_third_party_sqlite_file(tmp_path):
    """The regression this whole fix targets: pointing the endpoint at a
    third-party SQLite file must never touch it — no tables gained, no
    journal_mode flip — because the request is rejected before any
    `open_sqlite` call ever happens."""
    import sqlite3

    third_party = tmp_path / "unrelated.sqlite"
    con = sqlite3.connect(str(third_party))
    con.execute("CREATE TABLE unrelated (id INTEGER)")
    con.execute("PRAGMA journal_mode=DELETE")
    con.commit()
    con.close()
    before = third_party.read_bytes()

    client = TestClient(fastapi_app)
    resp = client.get("/api/checkpoint/tail", params={"path": str(third_party)})
    assert resp.status_code == 403

    after = third_party.read_bytes()
    assert after == before


# ── GET /api/run/{run_id} blame/shapley/branch_details population ──────────


def _seed_run_with_blame_shapley_and_branch(tmp_path):
    db_path = tmp_path / "store.db"
    store = TapeStore(str(db_path))
    tape = _small_tape(b"root")
    tape.append_exchange(b"req-1", b"resp-1")
    run_id = store.save_tape(tape, run_id="run1")

    blame_report = BlameReport(
        results=[
            FlipRateResult(
                step_index=0,
                flip_rate=0.8,
                ci_lo=0.5,
                ci_hi=0.95,
                flips=8,
                trials=10,
                valid_trials=10,
                p_value=0.01,
                q_value=0.02,
                responsible=True,
            )
        ],
        k=10,
        total_forks=10,
        ci_method=CIMethod.WILSON,
    )
    store.save_blame_report(run_id, blame_report)

    shapley_report = ShapleyReport(
        results=[
            ShapleyResult(
                step_index=1,
                shapley_value=0.5,
                ci_lo=0.2,
                ci_hi=0.7,
                n_samples=5,
                necessity=True,
                sufficiency=False,
            )
        ],
        n_permutation_samples=5,
        k=5,
        total_forks=5,
    )
    store.save_shapley_report(run_id, shapley_report)

    branch_id = store.save_branch(
        parent_run_id=run_id, divergence_step=0, delta_tape=_small_tape(b"branch")
    )
    store.close()
    return db_path, run_id, branch_id


def test_get_run_populates_blame_from_causal_edges(tmp_path):
    db_path, run_id, _branch_id = _seed_run_with_blame_shapley_and_branch(tmp_path)
    init_store(str(db_path))
    client = TestClient(fastapi_app)
    resp = client.get(f"/api/run/{run_id}")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["blame"] != {}
    assert body["blame"]["0"]["flip_rate"] == 0.8
    assert body["blame"]["0"]["responsible"] is True


def test_get_run_populates_shapley_from_causal_edges(tmp_path):
    db_path, run_id, _branch_id = _seed_run_with_blame_shapley_and_branch(tmp_path)
    init_store(str(db_path))
    client = TestClient(fastapi_app)
    resp = client.get(f"/api/run/{run_id}")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["shapley"] != {}
    assert body["shapley"]["1"]["necessity"] is True
    assert body["shapley"]["1"]["shapley_value"] == 0.5


def test_get_run_populates_branch_details_keyed_by_branch_id(tmp_path):
    db_path, run_id, branch_id = _seed_run_with_blame_shapley_and_branch(tmp_path)
    init_store(str(db_path))
    client = TestClient(fastapi_app)
    resp = client.get(f"/api/run/{run_id}")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert branch_id in body["branch_details"]
    detail = body["branch_details"][branch_id]
    assert detail["divergence_step"] == 0
    assert "exchanges" in detail  # full _tape_to_data shape, not just metadata


def test_get_run_blame_and_shapley_empty_when_no_causal_edges(tmp_path):
    db_path = tmp_path / "store.db"
    store = TapeStore(str(db_path))
    run_id = store.save_tape(_small_tape(b"solo"), run_id="solo")
    store.close()

    init_store(str(db_path))
    client = TestClient(fastapi_app)
    resp = client.get(f"/api/run/{run_id}")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["blame"] == {}
    assert body["shapley"] == {}
    assert body["branch_details"] == {}
