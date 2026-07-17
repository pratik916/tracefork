"""Offline tests for tracefork-bge.53: the OTel exemplar back-link
(trace_id/span_id -> run_id/step). `build_otel_trace`'s ids are
content-derived (sha256-truncated), not random, so the mapping is
reversible offline -- `locate_trace`/`locate_span_step` just recompute and
match, no persistence needed."""

from __future__ import annotations

from fastapi.testclient import TestClient

from tracefork.interop import build_otel_trace, locate_span_step, locate_trace
from tracefork.server import app as fastapi_app
from tracefork.server import init_store
from tracefork.store import TapeStore
from tracefork.validate import _record_clean_tape


def _seed(tmp_path):
    db = tmp_path / "store.db"
    store = TapeStore(str(db))
    tape = _record_clean_tape()
    run_id = store.save_tape(tape, run_id="run1")
    store.close()
    return db, run_id, tape


def _spans(tape):
    trace = build_otel_trace(tape)
    return trace["resourceSpans"][0]["scopeSpans"][0]["spans"]


def test_locate_trace_recovers_run_id_and_none_for_unrelated_trace(tmp_path):
    db, run_id, tape = _seed(tmp_path)
    store = TapeStore(str(db))
    trace_id = _spans(tape)[0]["traceId"]

    assert locate_trace(store, trace_id) == run_id
    assert locate_trace(store, "0" * 32) is None
    store.close()


def test_locate_span_step_recovers_each_exchange_step(tmp_path):
    _, _, tape = _seed(tmp_path)
    spans = _spans(tape)
    trace_id = spans[0]["traceId"]
    root_span_id = spans[0]["spanId"]

    # Root span has no corresponding exchange step.
    assert locate_span_step(tape, trace_id, root_span_id) is None
    # Every exchange span resolves to its own step index.
    for i, span in enumerate(spans[1:]):
        assert locate_span_step(tape, trace_id, span["spanId"]) == i


def test_server_otel_routes_redirect_with_and_without_step(tmp_path):
    db, run_id, tape = _seed(tmp_path)
    init_store(str(db))
    client = TestClient(fastapi_app, follow_redirects=False)

    spans = _spans(tape)
    trace_id = spans[0]["traceId"]
    exchange_span_id = spans[1]["spanId"]

    resp = client.get(f"/otel/{trace_id}/{exchange_span_id}")
    assert resp.status_code == 307
    assert resp.headers["location"] == f"/?run_id={run_id}&step=0"

    resp_no_step = client.get(f"/otel/{trace_id}")
    assert resp_no_step.status_code == 307
    assert resp_no_step.headers["location"] == f"/?run_id={run_id}"


def test_server_otel_unknown_trace_id_404s(tmp_path):
    db, _run_id, _tape = _seed(tmp_path)
    init_store(str(db))
    client = TestClient(fastapi_app, follow_redirects=False)

    resp = client.get(f"/otel/{'0' * 32}")
    assert resp.status_code == 404
    resp2 = client.get(f"/otel/{'0' * 32}/{'1' * 16}")
    assert resp2.status_code == 404


def test_report_html_reads_step_query_param_in_boot_sequence():
    from tracefork.report import _template_path

    content = _template_path().read_text()
    assert "URLSearchParams(location.search).get('step')" in content
