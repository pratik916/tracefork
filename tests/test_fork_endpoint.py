"""Offline FastAPI TestClient tests for tracefork-bge.36's click-to-fork
endpoints (POST /api/run/{run_id}/fork/estimate, POST /api/run/{run_id}/fork),
gated by fork_allowlist.py's opt-in-only allowlist plus an explicit
confirm:true cost confirmation.

$0-safe: the allowlisted agent is forked at the tape's LAST exchange index,
whose empty TAIL means ForkTransport never dispatches to its inner httpx
transport (the same $0 property tournament.py's docstring documents) --
proven directly by asserting the persisted branch's delta_tape holds only
the single mutation-injected exchange (no counterfactual continuation was
ever recorded), with no ANTHROPIC_API_KEY set anywhere in this process.
"""

from __future__ import annotations

import base64

import anthropic
import httpx
import pytest
from fastapi.testclient import TestClient

from tests.fakes import ScriptedFakeLLM, make_text_response
from tests.fixtures.fork_ui_agent import run_agent
from tracefork.blame import BudgetGovernor
from tracefork.fork_allowlist import (
    AgentNotAllowlistedError,
    estimate_single_fork_usd,
    parse_allowlist_env,
    resolve_agent_fn,
)
from tracefork.server import app as fastapi_app
from tracefork.server import init_fork_allowlist, init_store
from tracefork.store import TapeStore
from tracefork.tape import Tape
from tracefork.transport import TraceforkTransport

AGENT_PATH = "tests.fixtures.fork_ui_agent:run_agent"


def _record_tape() -> Tape:
    fake = ScriptedFakeLLM([make_text_response("4")])
    tape = Tape(agent_name="fork_ui_agent")
    transport = TraceforkTransport("record", tape, fake)
    client = anthropic.Anthropic(
        api_key="sk-ant-fake", http_client=httpx.Client(transport=transport), max_retries=0
    )
    run_agent(client)
    return tape


def _seed(tmp_path):
    db = tmp_path / "store.db"
    store = TapeStore(str(db))
    tape = _record_tape()
    run_id = store.save_tape(tape, run_id="run1")
    store.close()
    return db, run_id, tape


@pytest.fixture(autouse=True)
def _reset_allowlist():
    """Every test starts from the documented default: nothing allowlisted."""
    init_fork_allowlist({})
    yield
    init_fork_allowlist({})


def test_estimate_403_when_not_allowlisted(tmp_path):
    db, run_id, tape = _seed(tmp_path)
    init_store(str(db))
    client = TestClient(fastapi_app)
    resp = client.post(
        f"/api/run/{run_id}/fork/estimate",
        json={
            "agent_name": "nope",
            "step": len(tape.exchanges) - 1,
            "mutated_response_b64": base64.b64encode(b"{}").decode(),
        },
    )
    assert resp.status_code == 403
    assert "not allowlisted" in resp.json()["detail"]


def test_estimate_200_no_side_effects(tmp_path):
    db, run_id, tape = _seed(tmp_path)
    init_store(str(db))
    init_fork_allowlist({"fork_ui_agent": AGENT_PATH})
    client = TestClient(fastapi_app)
    last_step = len(tape.exchanges) - 1

    resp = client.post(
        f"/api/run/{run_id}/fork/estimate",
        json={
            "agent_name": "fork_ui_agent",
            "step": last_step,
            "mutated_response_b64": base64.b64encode(b'{"a":1}').decode(),
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert isinstance(body["est_usd"], float)

    store = TapeStore(str(db))
    assert store.list_branches(run_id) == []
    store.close()


def test_fork_requires_confirm_true(tmp_path):
    db, run_id, tape = _seed(tmp_path)
    init_store(str(db))
    init_fork_allowlist({"fork_ui_agent": AGENT_PATH})
    client = TestClient(fastapi_app)
    last_step = len(tape.exchanges) - 1

    resp = client.post(
        f"/api/run/{run_id}/fork",
        json={
            "agent_name": "fork_ui_agent",
            "step": last_step,
            "mutated_response_b64": base64.b64encode(b'{"a":1}').decode(),
        },
    )
    assert resp.status_code == 400


def test_fork_creates_branch_offline_last_step(tmp_path):
    db, run_id, tape = _seed(tmp_path)
    init_store(str(db))
    init_fork_allowlist({"fork_ui_agent": AGENT_PATH})
    client = TestClient(fastapi_app)
    last_step = len(tape.exchanges) - 1
    mutated = make_text_response("4 (mutated)")

    resp = client.post(
        f"/api/run/{run_id}/fork",
        json={
            "agent_name": "fork_ui_agent",
            "step": last_step,
            "mutated_response_b64": base64.b64encode(mutated).decode(),
            "confirm": True,
        },
    )
    assert resp.status_code == 200, resp.text
    branch_id = resp.json()["branch_id"]
    # Exactly the divergence-step exchange itself (mutation-injected, $0) --
    # an empty TAIL (no counterfactual continuation, since this was the
    # tape's last step) means zero real network calls occurred, with no
    # ANTHROPIC_API_KEY set anywhere in this process.
    assert resp.json()["delta_exchanges"] == 1

    store = TapeStore(str(db))
    branch = store.load_branch(branch_id)
    assert len(branch["delta_tape"].exchanges) == 1
    assert branch["delta_tape"].exchanges[0][1] == mutated
    store.close()


def test_unknown_run_id_404s_both_routes(tmp_path):
    db = tmp_path / "store.db"
    init_store(str(db))
    client = TestClient(fastapi_app)
    body = {"agent_name": "x", "step": 0, "mutated_response_b64": base64.b64encode(b"{}").decode()}
    assert client.post("/api/run/bogus/fork/estimate", json=body).status_code == 404
    assert client.post("/api/run/bogus/fork", json={**body, "confirm": True}).status_code == 404


def test_resolve_agent_fn_lists_allowlisted_names_on_miss():
    with pytest.raises(AgentNotAllowlistedError, match=r"allowlisted: \['a', 'b'\]"):
        resolve_agent_fn("missing", {"a": "x:y", "b": "x:z"})


def test_parse_allowlist_env_roundtrip(monkeypatch):
    parsed = parse_allowlist_env("foo=pkg.mod:fn, bar=pkg2.mod2:fn2")
    assert parsed == {"foo": "pkg.mod:fn", "bar": "pkg2.mod2:fn2"}
    assert parse_allowlist_env("") == {}

    monkeypatch.delenv("TRACEFORK_FORK_AGENTS", raising=False)
    assert parse_allowlist_env(None) == {}
    monkeypatch.setenv("TRACEFORK_FORK_AGENTS", "a=m:f")
    assert parse_allowlist_env(None) == {"a": "m:f"}


def test_estimate_single_fork_usd_scales_with_remaining_tail_only():
    fake = ScriptedFakeLLM(
        [make_text_response("a"), make_text_response("b"), make_text_response("c")]
    )
    tape = Tape(agent_name="multi")
    transport = TraceforkTransport("record", tape, fake)
    client = anthropic.Anthropic(
        api_key="sk-ant-fake", http_client=httpx.Client(transport=transport), max_retries=0
    )
    messages: list[dict] = []
    for i in range(3):
        messages.append({"role": "user", "content": f"q{i}"})
        resp = client.messages.create(model="claude-sonnet-4-6", max_tokens=100, messages=messages)
        messages.append({"role": "assistant", "content": resp.content[0].text})

    est_first = estimate_single_fork_usd(tape, 0)
    est_middle = estimate_single_fork_usd(tape, 1)
    est_last = estimate_single_fork_usd(tape, 2)
    assert est_first > est_middle > est_last
    assert est_last == 0.0

    full_sweep_est = BudgetGovernor.estimate(tape, k=1).est_usd
    assert est_first != full_sweep_est
