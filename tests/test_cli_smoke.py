"""All-CLI-commands smoke test — every `tracefork <command>` invoked offline,
each asserted against its real (documented) exit code, not just "doesn't
crash". Companion to `tests/test_e2e.py`'s cross-module integration tests.

Two commands (`serve`, `proxy record`/`proxy replay`) call `uvicorn.run(...)`
directly and would otherwise block forever binding a real socket — those are
driven two ways instead, both offline: (1) the CLI's own wiring (argument
resolution, `init_store`, the exact `host`/`port` passed to `uvicorn.run`) is
proven via `CliRunner` with `uvicorn.run` monkeypatched to a no-op that
records its call, so the command still returns and can assert `exit_code`;
(2) the actual serving behavior is proven via an ASGI/TestClient driving the
underlying FastAPI app object directly (`server.py`, `proxy.py`) — the exact
pattern `tests/test_proxy.py` already uses for `proxy`.

`blame` is exercised only via its offline pre-flight budget gate (`--budget
0`, which fails before any network call) — the $0 proof that blame's ENGINE
works is `tests/test_e2e.py`'s direct `BlameEngine.rank()` call and
`tracefork validate`, never the live-API CLI path.

All offline, $0 — no ANTHROPIC_API_KEY, no network, no real port bound.
"""

from __future__ import annotations

import json
from pathlib import Path

import uvicorn as uvicorn_module
from typer.testing import CliRunner

from tests.fakes import make_text_response
from tracefork.cli import app
from tracefork.store import TapeStore
from tracefork.tape import Tape
from tracefork.validate import _record_clean_tape

runner = CliRunner()

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "experiments" / "replay_fixtures"


def _seeded_store(tmp_path: Path) -> tuple[Path, str]:
    db = tmp_path / "store.db"
    store = TapeStore(str(db))
    run_id = store.save_tape(_record_clean_tape(), run_id="smoke-run")
    store.close()
    return db, run_id


# ── replay ───────────────────────────────────────────────────────────────


def test_cli_replay_agent_form_exits_zero(tmp_path):
    tape_path = tmp_path / "run.tape.sqlite"
    _record_clean_tape().save(str(tape_path))
    result = runner.invoke(
        app, ["replay", str(tape_path), "--agent", "tracefork.validate:synthetic_agent"]
    )
    assert result.exit_code == 0, result.output
    assert "PASS" in result.output


def test_cli_replay_check_fixture_corpus_exits_zero():
    result = runner.invoke(app, ["replay", "--check", str(FIXTURES_DIR)])
    assert result.exit_code == 0, result.output
    assert "fixtures passed" in result.output


def test_cli_replay_missing_args_is_the_documented_nonzero_exit():
    result = runner.invoke(app, ["replay"])
    assert result.exit_code == 1
    assert "Provide a tape path and --agent" in result.output


def test_cli_replay_receipt_includes_boundary_and_redaction_lines(tmp_path):
    """The receipt must surface `Tape.boundary`/`content_redacted` (tracefork-bge.20)
    — a forensic-only or content-redacted tape must not look identical to a
    verified one in the terminal output."""
    tape_path = tmp_path / "run.tape.sqlite"
    _record_clean_tape().save(str(tape_path))
    result = runner.invoke(
        app, ["replay", str(tape_path), "--agent", "tracefork.validate:synthetic_agent"]
    )
    assert result.exit_code == 0, result.output
    assert "boundary" in result.output
    assert "content_redacted" in result.output


# ── verify ───────────────────────────────────────────────────────────────


def test_cli_verify_agent_form_exits_zero(tmp_path):
    tape_path = tmp_path / "run.tape.sqlite"
    _record_clean_tape().save(str(tape_path))
    result = runner.invoke(
        app, ["verify", str(tape_path), "--agent", "tracefork.validate:synthetic_agent"]
    )
    assert result.exit_code == 0, result.output


def test_cli_verify_drift_is_the_documented_nonzero_exit(tmp_path):
    """A mismatched agent must be caught as drift (exit 1), not silently pass."""
    tape_path = tmp_path / "run.tape.sqlite"
    _record_clean_tape().save(str(tape_path))
    result = runner.invoke(
        app, ["verify", str(tape_path), "--agent", "tracefork.fixtures:single_turn_agent"]
    )
    assert result.exit_code == 1


def test_cli_verify_corpus_no_tapes_is_the_documented_nonzero_exit():
    result = runner.invoke(app, ["verify", "--corpus"])
    assert result.exit_code == 1
    assert "No tapes found" in result.output


def test_cli_verify_store_healthy_exits_zero_and_prints_run_id(tmp_path):
    db, run_id = _seeded_store(tmp_path)
    result = runner.invoke(app, ["verify", "--store", str(db)])
    assert result.exit_code == 0, result.output
    assert run_id in result.output
    assert "FAIL" not in result.output


def test_cli_verify_store_corrupted_row_is_nonzero_exit_and_names_run_id(tmp_path):
    import sqlite3

    db, run_id = _seeded_store(tmp_path)
    con = sqlite3.connect(str(db))
    con.execute("UPDATE tapes SET tape_bytes=? WHERE run_id=?", (b"\x00not-a-tape", run_id))
    con.commit()
    con.close()

    result = runner.invoke(app, ["verify", "--store", str(db)])
    assert result.exit_code == 1
    assert run_id in result.output
    assert "FAIL" in result.output


def test_cli_verify_store_missing_db_is_nonzero_exit(tmp_path):
    result = runner.invoke(app, ["verify", "--store", str(tmp_path / "nope.db")])
    assert result.exit_code == 1


def test_cli_verify_store_and_corpus_are_mutually_exclusive(tmp_path):
    db, _ = _seeded_store(tmp_path)
    result = runner.invoke(app, ["verify", "--store", str(db), "--corpus"])
    assert result.exit_code == 1


def test_cli_verify_receipt_includes_boundary_and_redaction_lines(tmp_path):
    """Same receipt lines as `replay` (tracefork-bge.20) — `_print_receipt` has
    exactly two call sites (replay, verify) and both must carry them."""
    tape_path = tmp_path / "run.tape.sqlite"
    _record_clean_tape().save(str(tape_path))
    result = runner.invoke(
        app, ["verify", str(tape_path), "--agent", "tracefork.validate:synthetic_agent"]
    )
    assert result.exit_code == 0, result.output
    assert "boundary" in result.output
    assert "content_redacted" in result.output


# ── fork ─────────────────────────────────────────────────────────────────


def test_cli_fork_at_last_step_is_offline_and_exits_zero(tmp_path):
    """Forking at the tape's LAST step means `tail_recorded == 0`: the CLI's
    real (network-capable) post-fork transport is constructed but never
    driven, so this is genuinely offline — not just "no assertion failed"."""
    db, run_id = _seeded_store(tmp_path)
    resp_path = tmp_path / "mutated.bytes"
    resp_path.write_bytes(make_text_response("FAIL — cancelled"))

    result = runner.invoke(
        app,
        [
            "fork",
            run_id,
            "--step",
            "1",
            "--response",
            str(resp_path),
            "--agent",
            "tracefork.validate:synthetic_agent",
            "--store",
            str(db),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Fork created" in result.output


# ── report ───────────────────────────────────────────────────────────────


def _extract_report_data(html: str) -> dict:
    marker = "window.__TRACEFORK_DATA__ = "
    start = html.find(marker) + len(marker)
    end = html.find(";\n", start)
    return json.loads(html[start:end])


def test_cli_report_writes_html_and_exits_zero(tmp_path):
    db, run_id = _seeded_store(tmp_path)
    out = tmp_path / "report.html"
    result = runner.invoke(app, ["report", run_id, "--store", str(db), "-o", str(out)])
    assert result.exit_code == 0, result.output
    assert out.exists()


def test_cli_report_with_agent_embeds_replay_receipt(tmp_path):
    db, run_id = _seeded_store(tmp_path)
    out = tmp_path / "report.html"
    result = runner.invoke(
        app,
        [
            "report",
            run_id,
            "--store",
            str(db),
            "--agent",
            "tracefork.validate:synthetic_agent",
            "-o",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    data = _extract_report_data(out.read_text())
    assert data["replay"]["bit_exact"] is True


def test_cli_report_without_run_id_or_tape_is_the_documented_nonzero_exit(tmp_path):
    result = runner.invoke(app, ["report", "--store", str(tmp_path / "store.db")])
    assert result.exit_code == 1


def test_cli_report_terminal_echo_includes_boundary_and_redaction_lines(tmp_path):
    """`report`'s terminal echo must carry the same two trust lines as the
    replay/verify receipt (tracefork-bge.20), even though it doesn't go
    through `_print_receipt` (that helper has exactly two call sites)."""
    db, run_id = _seeded_store(tmp_path)
    out = tmp_path / "report.html"
    result = runner.invoke(app, ["report", run_id, "--store", str(db), "-o", str(out)])
    assert result.exit_code == 0, result.output
    assert "boundary" in result.output
    assert "content_redacted" in result.output


# ── serve ────────────────────────────────────────────────────────────────


def test_cli_serve_wires_host_and_store_without_binding_a_real_port(tmp_path, monkeypatch):
    """`serve` calls `uvicorn.run(...)` unconditionally — there is no
    pre-serve validation branch to test via CliRunner alone. Monkeypatching
    `uvicorn.run` to a no-op that records its kwargs proves the CLI's own
    wiring (127.0.0.1, the requested port, the store path) without hanging;
    the FastAPI app object it serves is exercised directly via TestClient in
    `test_server_app_renders_ui_and_serves_run_json_same_origin` below."""
    db = tmp_path / "store.db"
    captured: dict = {}

    def fake_run(app_obj, *, host, port, workers, log_level):
        captured["host"] = host
        captured["port"] = port

    monkeypatch.setattr(uvicorn_module, "run", fake_run)

    result = runner.invoke(app, ["serve", "--store", str(db), "--port", "9911"])
    assert result.exit_code == 0, result.output
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 9911


def test_server_app_renders_ui_and_serves_run_json_same_origin(tmp_path):
    """The EXACT FastAPI app object `tracefork serve` wires up, driven via
    `TestClient` (no real socket): renders the UI, serves recorded-run JSON,
    404s on an unknown run, and sets no CORS header (same-origin only — the
    CLI-wiring test above proves the actual bind is 127.0.0.1)."""
    from fastapi.testclient import TestClient

    from tracefork.server import app as fastapi_app
    from tracefork.server import init_store

    db, run_id = _seeded_store(tmp_path)
    init_store(str(db))
    client = TestClient(fastapi_app)

    root = client.get("/")
    assert root.status_code == 200
    assert "tracefork" in root.text
    assert "access-control-allow-origin" not in {k.lower() for k in root.headers}

    runs = client.get("/api/runs")
    assert runs.status_code == 200
    assert any(r["run_id"] == run_id for r in runs.json())

    run = client.get(f"/api/run/{run_id}")
    assert run.status_code == 200
    assert run.json()["run_id"] == run_id

    missing = client.get("/api/run/does-not-exist")
    assert missing.status_code == 404


# ── blame (offline pre-flight gate only — see module docstring) ───────────


def test_cli_blame_budget_gate_blocks_overspend_before_any_network_call(tmp_path):
    db, run_id = _seeded_store(tmp_path)
    result = runner.invoke(
        app,
        [
            "blame",
            run_id,
            "--agent",
            "tracefork.validate:synthetic_agent",
            "--store",
            str(db),
            "--budget",
            "0",
        ],
    )
    assert result.exit_code == 1, result.output
    assert "exceeds budget" in result.output


# ── validate ─────────────────────────────────────────────────────────────


def test_cli_validate_runs_offline_and_exits_zero(tmp_path):
    out = tmp_path / "vr.json"
    result = runner.invoke(app, ["validate", "--k", "1", "--n-runs", "1", "--output", str(out)])
    assert result.exit_code == 0, result.output
    data = json.loads(out.read_text())
    assert data["negative_control_max_flip"] < 0.30


def test_cli_validate_check_passes_against_committed_baseline(tmp_path):
    out = tmp_path / "vr.json"
    result = runner.invoke(
        app, ["validate", "--k", "1", "--n-runs", "1", "--output", str(out), "--check"]
    )
    assert result.exit_code == 0, result.output
    assert "No regressions" in result.output


# ── bench ────────────────────────────────────────────────────────────────


def test_cli_bench_runs_offline_and_exits_zero(tmp_path):
    out = tmp_path / "bench_report.json"
    result = runner.invoke(app, ["bench", "--k", "2", "--m-samples", "1", "--output", str(out)])
    assert result.exit_code == 0, result.output
    assert "competing-fault discrimination" in result.output
    data = json.loads(out.read_text())
    assert data["n_resolved"] == 8


# ── export / ingest ─────────────────────────────────────────────────────


def test_cli_export_otel_and_ingest_round_trip_exit_zero(tmp_path):
    db, run_id = _seeded_store(tmp_path)
    trace_path = tmp_path / "trace.json"
    export_result = runner.invoke(
        app, ["export", run_id, "--store", str(db), "--otel", "-o", str(trace_path)]
    )
    assert export_result.exit_code == 0, export_result.output

    out_tape = tmp_path / "ingested.tape.sqlite"
    ingest_result = runner.invoke(app, ["ingest", str(trace_path), "--otel", "-o", str(out_tape)])
    assert ingest_result.exit_code == 0, ingest_result.output
    assert out_tape.exists()


def test_cli_export_openinference_exits_zero(tmp_path):
    db, run_id = _seeded_store(tmp_path)
    out = tmp_path / "dataset.json"
    result = runner.invoke(
        app, ["export", run_id, "--store", str(db), "--openinference", "-o", str(out)]
    )
    assert result.exit_code == 0, result.output


def test_cli_export_requires_exactly_one_format_flag_is_documented_nonzero(tmp_path):
    db, run_id = _seeded_store(tmp_path)
    result = runner.invoke(app, ["export", run_id, "--store", str(db)])
    assert result.exit_code == 1
    assert "exactly one" in result.output


# ── prune ────────────────────────────────────────────────────────────────


def test_cli_prune_dry_run_older_than_days_exits_zero_no_row_count_change(tmp_path):
    db, run_id = _seeded_store(tmp_path)
    result = runner.invoke(
        app, ["prune", "--older-than-days", "0", "--dry-run", "--store", str(db)]
    )
    assert result.exit_code == 0, result.output

    store = TapeStore(str(db))
    try:
        assert any(r["run_id"] == run_id for r in store.list_runs())
    finally:
        store.close()


def test_cli_prune_by_run_id_archives_it_and_still_exits_zero(tmp_path):
    db, run_id = _seeded_store(tmp_path)
    result = runner.invoke(app, ["prune", "--run-id", run_id, "--store", str(db)])
    assert result.exit_code == 0, result.output
    assert "Archived" in result.output

    store = TapeStore(str(db))
    try:
        assert store.list_runs() == []
    finally:
        store.close()


# ── proxy ────────────────────────────────────────────────────────────────


def test_cli_proxy_record_wires_and_saves_tape_without_binding_a_real_port(tmp_path, monkeypatch):
    """Same technique as `serve`: `uvicorn.run` never actually gets a chance
    to bind a socket, but the surrounding CLI logic (tape creation, matcher
    resolution, the `finally`-block save) all genuinely executes."""
    tape_path = tmp_path / "proxy.tape.sqlite"
    monkeypatch.setattr(uvicorn_module, "run", lambda *a, **kw: None)

    result = runner.invoke(
        app,
        [
            "proxy",
            "record",
            "--tape",
            str(tape_path),
            "--upstream",
            "https://upstream.example",
            "--port",
            "8912",
        ],
    )
    assert result.exit_code == 0, result.output
    assert tape_path.exists()


def test_cli_proxy_replay_wires_without_binding_a_real_port(tmp_path, monkeypatch):
    tape_path = tmp_path / "proxy.tape.sqlite"
    tape = Tape()
    tape.append_exchange(b'{"model":"m"}', b'{"id":"resp"}')
    tape.save(str(tape_path))
    monkeypatch.setattr(uvicorn_module, "run", lambda *a, **kw: None)

    result = runner.invoke(app, ["proxy", "replay", "--tape", str(tape_path), "--port", "8913"])
    assert result.exit_code == 0, result.output


def test_cli_proxy_rejects_invalid_mode_is_documented_nonzero(tmp_path):
    result = runner.invoke(app, ["proxy", "bogus", "--tape", str(tmp_path / "t.tape.sqlite")])
    assert result.exit_code == 1


# ── coverage ─────────────────────────────────────────────────────────────


def test_cli_coverage_prints_report_and_exits_zero(tmp_path):
    tape_path = tmp_path / "run.tape.sqlite"
    _record_clean_tape().save(str(tape_path))
    result = runner.invoke(app, ["coverage", str(tape_path)])
    assert result.exit_code == 0, result.output
    assert "boundary_guard_active" in result.output
    assert "concurrency_recorded" in result.output


def test_cli_coverage_with_agent_source_scans_and_writes_json(tmp_path):
    tape_path = tmp_path / "run.tape.sqlite"
    _record_clean_tape().save(str(tape_path))
    agent_src = tmp_path / "agent.py"
    agent_src.write_text("import random\nrandom.random()\n")
    out_path = tmp_path / "coverage.json"

    result = runner.invoke(
        app,
        [
            "coverage",
            str(tape_path),
            "--agent-source",
            str(agent_src),
            "--output",
            str(out_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "GUARDABLE" in result.output
    assert out_path.exists()

    data = json.loads(out_path.read_text())
    assert data["findings"][0]["call"] == "random.random"
