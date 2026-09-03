"""FastAPI TestClient tests for `server.py`'s JSON/HTML endpoints that don't
already have a dedicated test file (see `test_fork_endpoint.py`,
`test_branch_queries.py`, `test_live.py`, `test_otel_locate.py`,
`test_runs_page.py`, `test_sessions.py`, `test_cost_profile.py` for the rest).

Covers three things:

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

  3. `POST /api/run/{run_id}/fork`'s `max_usd` spend cap (tracefork-sis.39):
     `do_fork` used to call `ForkEngine.fork` with no cost check at all,
     synchronously on the request-handling coroutine, so a
     click-to-fork both spent real money unconditionally and froze this
     process's ONE event loop (`uvicorn --workers 1`) — including its own
     SSE stream — for the fork's full duration. It now rejects (402) any
     fork whose `fork_allowlist.estimate_single_fork_usd` estimate exceeds
     the request's `max_usd` BEFORE executing anything, and runs the
     synchronous fork via `asyncio.to_thread` so the event loop stays free
     to serve other requests while it's in flight. See
     `test_fork_endpoint.py` for the rest of this endpoint's coverage
     (allowlist/confirm gating, branch persistence).
"""

from __future__ import annotations

import asyncio
import base64
import time

import anthropic
import httpx2
import pytest
from fastapi.testclient import TestClient

from tests.fakes import ScriptedFakeLLM, make_text_response
from tests.fixtures.fork_ui_agent import run_agent as fork_ui_run_agent
from tracefork import server
from tracefork.blame import BlameReport, CIMethod, FlipRateResult, ShapleyReport, ShapleyResult
from tracefork.checkpoint import CheckpointWriter
from tracefork.fork import ForkEngine
from tracefork.server import app as fastapi_app
from tracefork.server import init_checkpoint_dirs, init_fork_allowlist, init_store
from tracefork.store import TapeStore
from tracefork.tape import Tape
from tracefork.transport import TraceforkTransport

_FORK_UI_AGENT_PATH = "tests.fixtures.fork_ui_agent:run_agent"


@pytest.fixture(autouse=True)
def _reset_checkpoint_dirs():
    """Every test starts from the documented default: nothing allowlisted."""
    init_checkpoint_dirs([])
    yield
    init_checkpoint_dirs([])


@pytest.fixture(autouse=True)
def _reset_fork_allowlist():
    """Every test starts from the documented default: nothing allowlisted."""
    init_fork_allowlist({})
    yield
    init_fork_allowlist({})


def _small_tape(tag: bytes = b"x") -> Tape:
    t = Tape(agent_name="w")
    t.append_exchange(b"req-" + tag, b"resp-" + tag)
    return t


# ── Content-Security-Policy (v1.0.0 readiness item 42) ──────────────────────
# These pages render recorded, potentially third-party agent I/O -- a CSP
# contains any future escaping miss instead of letting it escalate to full
# tape exfiltration (see item 41's session_report.html attribute-breakout
# fix). Restrictive: no external script/style/img/connect origin anywhere.


def test_get_slash_sends_a_restrictive_csp_header(tmp_path):
    db = tmp_path / "store.db"
    init_store(str(db))
    client = TestClient(fastapi_app)
    resp = client.get("/")
    assert resp.status_code == 200
    csp = resp.headers.get("content-security-policy")
    assert csp is not None
    assert csp == server.CONTENT_SECURITY_POLICY
    for directive in ("script-src", "style-src", "img-src", "connect-src"):
        assert directive in csp
    # no external origin (http(s):// or a bare wildcard) anywhere in the policy
    assert "http://" not in csp
    assert "https://" not in csp
    assert " *" not in csp and csp.strip() != "*"


def test_get_runs_page_sends_the_same_csp_header(tmp_path):
    db = tmp_path / "store.db"
    init_store(str(db))
    client = TestClient(fastapi_app)
    resp = client.get("/runs")
    assert resp.status_code == 200
    assert resp.headers.get("content-security-policy") == server.CONTENT_SECURITY_POLICY


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


# ── POST /api/run/{run_id}/fork max_usd cap + off-loop execution ────────────
#
# See test_fork_endpoint.py for the allowlist/confirm/branch-persistence
# coverage this doesn't repeat.


def _seed_multi_step_tape(tmp_path):
    """A synthetic (not recorded via a real agent) 3-exchange tape with
    sizeable request/response bodies, so `estimate_single_fork_usd` prices a
    fork at step 0 (2 remaining tail calls) as strictly greater than $0 —
    the cap-exceeded test never needs `ForkEngine.fork` to actually run, so
    this doesn't need to be an agent-consistent tape."""
    tape = Tape(agent_name="fork_ui_agent")
    for i in range(3):
        tape.append_exchange(
            f"req-{i}-".encode() + b"a" * 400,
            f"resp-{i}-".encode() + b"b" * 400,
        )
    db = tmp_path / "store.db"
    store = TapeStore(str(db))
    run_id = store.save_tape(tape, run_id="run1")
    store.close()
    return db, run_id, tape


def _seed_last_step_fork_tape(tmp_path):
    """A real, one-exchange tape recorded via the allowlisted
    `fork_ui_agent`. Forking its only (== last) exchange is `$0`-safe: an
    empty tail means `ForkTransport` never dispatches to its inner httpx2
    transport (see `test_fork_endpoint.py`'s module docstring) — needed here
    because these tests actually execute the fork rather than rejecting it
    before it runs."""
    fake = ScriptedFakeLLM([make_text_response("4")])
    tape = Tape(agent_name="fork_ui_agent")
    transport = TraceforkTransport("record", tape, fake)
    client = anthropic.Anthropic(
        api_key="sk-ant-fake", http_client=httpx2.Client(transport=transport), max_retries=0
    )
    fork_ui_run_agent(client)
    db = tmp_path / "store.db"
    store = TapeStore(str(db))
    run_id = store.save_tape(tape, run_id="run1")
    store.close()
    return db, run_id, tape


def test_fork_rejects_when_estimate_exceeds_max_usd_and_persists_nothing(tmp_path):
    db, run_id, tape = _seed_multi_step_tape(tmp_path)
    init_store(str(db))
    init_fork_allowlist({"fork_ui_agent": _FORK_UI_AGENT_PATH})
    client = TestClient(fastapi_app)

    resp = client.post(
        f"/api/run/{run_id}/fork",
        json={
            "agent_name": "fork_ui_agent",
            "step": 0,
            "mutated_response_b64": base64.b64encode(b"{}").decode(),
            "confirm": True,
            "max_usd": 0.0,
        },
    )
    assert resp.status_code == 402, resp.text
    assert "max_usd" in resp.json()["detail"]

    store = TapeStore(str(db))
    assert store.list_branches(run_id) == []
    store.close()


def test_fork_succeeds_when_estimate_is_within_max_usd(tmp_path):
    db, run_id, tape = _seed_last_step_fork_tape(tmp_path)
    init_store(str(db))
    init_fork_allowlist({"fork_ui_agent": _FORK_UI_AGENT_PATH})
    client = TestClient(fastapi_app)
    last_step = len(tape.exchanges) - 1

    resp = client.post(
        f"/api/run/{run_id}/fork",
        json={
            "agent_name": "fork_ui_agent",
            "step": last_step,
            "mutated_response_b64": base64.b64encode(make_text_response("4 (mutated)")).decode(),
            "confirm": True,
            "max_usd": 0.01,
        },
    )
    assert resp.status_code == 200, resp.text

    store = TapeStore(str(db))
    assert len(store.list_branches(run_id)) == 1
    store.close()


async def test_fork_runs_off_the_event_loop_so_concurrent_requests_are_not_blocked(
    tmp_path, monkeypatch
):
    """The synchronous `ForkEngine.fork` call must run off the event-loop
    thread (`asyncio.to_thread` or equivalent). Proof: patch `ForkEngine.fork`
    to sleep for 0.3s (synchronously, simulating a slow real fork) then fire a
    concurrent, cheap `GET /api/runs` while the fork request is in flight — if
    `do_fork` still awaited the sleep directly on the request coroutine, this
    process's one event loop (`uvicorn --workers 1`) would be blocked and the
    concurrent GET would take ~0.3s too; off-loop execution lets it return
    almost immediately instead."""
    db, run_id, tape = _seed_last_step_fork_tape(tmp_path)
    init_store(str(db))
    init_fork_allowlist({"fork_ui_agent": _FORK_UI_AGENT_PATH})
    last_step = len(tape.exchanges) - 1

    real_fork = ForkEngine.fork

    def slow_fork(*args, **kwargs):
        time.sleep(0.3)
        return real_fork(*args, **kwargs)

    monkeypatch.setattr("tracefork.server.ForkEngine.fork", staticmethod(slow_fork))

    transport = httpx2.ASGITransport(app=fastapi_app)
    async with httpx2.AsyncClient(transport=transport, base_url="http://test") as async_client:
        fork_task = asyncio.create_task(
            async_client.post(
                f"/api/run/{run_id}/fork",
                json={
                    "agent_name": "fork_ui_agent",
                    "step": last_step,
                    "mutated_response_b64": base64.b64encode(
                        make_text_response("4 (mutated)")
                    ).decode(),
                    "confirm": True,
                    "max_usd": 5.0,
                },
            )
        )
        # A short sleep is the probe: it hands control back to the event
        # loop, which is the only way `fork_task` (just scheduled, not yet
        # run) gets to start executing at all. If `do_fork` still blocks the
        # thread inside `time.sleep(0.3)` with no `await` in between, the
        # WHOLE event loop — including this sleep's own wakeup timer — is
        # frozen until that finishes, so this `await` takes ~0.3s instead of
        # ~0.05s. Off-loop execution leaves the loop free to wake this timer
        # on schedule while the fork runs on its own thread.
        sleep_start = time.monotonic()
        await asyncio.sleep(0.05)
        sleep_elapsed = time.monotonic() - sleep_start

        runs_resp = await async_client.get("/api/runs")
        fork_resp = await fork_task

    assert runs_resp.status_code == 200
    assert fork_resp.status_code == 200, fork_resp.text
    assert sleep_elapsed < 0.2, (
        f"a 0.05s asyncio.sleep took {sleep_elapsed:.3f}s while a fork was in "
        "flight — the event loop was blocked, so the fork ran synchronously "
        "on it instead of off-loop"
    )
