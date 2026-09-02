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


# ── app-bar cross-navigation (v1.0.0 readiness item 45) ─────────────────────
# report.html shipped with zero <a> elements -- nothing linked back to
# /runs, and web/session_report.html was unreachable from either page. The
# three templates now share a consistent app bar (see web/*.html's
# `.app-nav`/`.app-nav-link`); this section proves each page carries at
# least one resolvable link to one of the other two.

_REPORT_HTML = _RUNS_HTML.parent / "report.html"
_SESSION_REPORT_HTML = _RUNS_HTML.parent / "session_report.html"


def test_report_html_links_to_runs_page():
    content = _REPORT_HTML.read_text()
    assert 'id="nav-runs-link"' in content
    assert 'href="/runs"' in content


def test_report_html_hides_the_runs_link_in_static_mode_not_a_dead_link():
    """`/runs` is a live-server route (see server.py's `serve_runs_page`), not
    a file next to a static report export -- the link must be hidden rather
    than shipped as a dead link when window.__TRACEFORK_SERVER_URL__ is
    undefined (the static-report case)."""
    content = _REPORT_HTML.read_text()
    assert "window.__TRACEFORK_SERVER_URL__ === undefined) link.hidden = true" in content


def test_runs_html_app_nav_marks_itself_as_the_current_page():
    content = _RUNS_HTML.read_text()
    assert 'id="nav-runs-link"' in content
    assert 'aria-current="page"' in content


def test_report_html_and_runs_html_share_the_same_app_nav_css_classes():
    report_content = _REPORT_HTML.read_text()
    runs_content = _RUNS_HTML.read_text()
    for cls in (".app-nav {", ".app-nav-link {"):
        assert cls in report_content, f"report.html missing {cls!r}"
        assert cls in runs_content, f"runs.html missing {cls!r}"


def test_session_report_html_links_each_lane_back_to_its_own_run_report():
    """web/session_report.html was unreachable from either page AND itself
    linked nowhere -- each lane's run_id now links to `/?run_id=...`, the
    query-param contract report.html's loadData already reads."""
    content = _SESSION_REPORT_HTML.read_text()
    assert 'href="/?run_id=${encodeURIComponent(lane.run_id)}"' in content
    assert 'class="lane-run-id"' in content
