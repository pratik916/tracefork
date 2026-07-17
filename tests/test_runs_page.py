"""TestClient-driven tests for tracefork-bge.67's multi-run dashboard page
(`GET /runs`), reusing the already-existing, already-tested `GET /api/runs`
contract -- following test_cli_smoke.py's
test_server_app_renders_ui_and_serves_run_json_same_origin pattern."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from tracefork.server import app as fastapi_app
from tracefork.server import init_store
from tracefork.store import TapeStore
from tracefork.tape import Tape

_RUNS_HTML = Path(__file__).resolve().parent.parent / "web" / "runs.html"


def _small_tape(tag: bytes) -> Tape:
    t = Tape(agent_name=f"agent-{tag.decode()}")
    t.append_exchange(b"req-" + tag, b"resp-" + tag)
    return t


def test_runs_page_served_at_get_runs(tmp_path):
    db = tmp_path / "store.db"
    init_store(str(db))
    client = TestClient(fastapi_app)

    resp = client.get("/runs")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "window.__TRACEFORK_SERVER_URL__" in resp.text


def test_runs_page_lists_seeded_runs_via_existing_api_runs_endpoint(tmp_path):
    db = tmp_path / "store.db"
    store = TapeStore(str(db))
    store.save_tape(_small_tape(b"one"), run_id="run-one", created_at="2026-01-01T00:00:00")
    store.save_tape(_small_tape(b"two"), run_id="run-two", created_at="2026-01-02T00:00:00")
    store.close()

    init_store(str(db))
    client = TestClient(fastapi_app)
    resp = client.get("/api/runs")
    assert resp.status_code == 200
    runs = resp.json()
    run_ids = {r["run_id"] for r in runs}
    assert run_ids == {"run-one", "run-two"}
    # newest-first
    assert runs[0]["run_id"] == "run-two"
    assert all("agent_name" in r and "created_at" in r for r in runs)


def test_runs_page_html_references_api_runs_and_run_query_param_link():
    content = _RUNS_HTML.read_text()
    assert "/api/runs" in content
    assert "?run_id=" in content


def test_runs_page_empty_store_returns_200(tmp_path):
    db = tmp_path / "store.db"
    init_store(str(db))
    client = TestClient(fastapi_app)

    resp = client.get("/runs")
    assert resp.status_code == 200

    api_resp = client.get("/api/runs")
    assert api_resp.status_code == 200
    assert api_resp.json() == []
