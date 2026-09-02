"""CLI tests — exercise the Typer surface offline, especially the budget money-guard
(`blame`'s pre-flight cost gate is the only thing standing between a typo and a real bill)
and the now-enforced negative control in `validate`."""

import json
import shutil
from pathlib import Path

from typer.testing import CliRunner

from tracefork.cli import app
from tracefork.constants import OTEL_INGESTED_BOUNDARY
from tracefork.store import TapeStore
from tracefork.tape import Tape
from tracefork.validate import _record_clean_tape

runner = CliRunner()

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "experiments" / "replay_fixtures"


def test_blame_budget_gate_blocks_overspend(tmp_path):
    """`blame` must refuse to spend when the estimate exceeds --budget, and it must do
    so *before* any network call — the gate fires on the pre-flight estimate."""
    db = tmp_path / "store.db"
    store = TapeStore(str(db))
    run_id = store.save_tape(_record_clean_tape(), run_id="testrun")
    store.close()

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


def test_blame_rejects_unsafe_run_id(tmp_path):
    """run_id is validated up front — this is what keeps the `blame_<run_id>.json`
    output path from being a traversal sink."""
    db = tmp_path / "store.db"
    TapeStore(str(db)).close()

    result = runner.invoke(
        app,
        [
            "blame",
            "../etc/passwd",
            "--agent",
            "tracefork.validate:synthetic_agent",
            "--store",
            str(db),
        ],
    )
    assert result.exit_code != 0


def test_blame_emits_progress_lines_per_completed_trial(tmp_path, monkeypatch):
    """tracefork-sis.51: a real blame sweep must emit one progress line per
    completed trial instead of going silent for the whole (potentially
    real-money) run. Fully offline/$0: the fixture tape has exactly ONE
    exchange, so forking its only (and therefore last) step has an empty
    tail -- see `tournament.py`'s module docstring for the same $0 trick."""
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "store.db"
    store = TapeStore(str(db))
    tape = Tape.load(str(FIXTURES_DIR / "single_turn.tape.sqlite"))
    run_id = store.save_tape(tape, run_id="single-turn-run")
    store.close()

    result = runner.invoke(
        app,
        [
            "blame",
            run_id,
            "--agent",
            "tracefork.fixtures:single_turn_agent",
            "--store",
            str(db),
            "--k",
            "3",
            "--success-re",
            "4",
            "--failure-re",
            "nope-never-matches-xyz",
        ],
    )
    assert result.exit_code == 0, result.output
    progress_lines = [ln for ln in result.output.splitlines() if "[blame] trial" in ln]
    assert len(progress_lines) == 3
    assert progress_lines[-1].strip().endswith("3/3 complete")


def test_tournament_emits_progress_lines_per_completed_trial(tmp_path, monkeypatch):
    """Same per-completed-trial progress contract, for `tournament`'s $0
    last-step comparison (its own module-documented no-tail-call case)."""
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "store.db"
    store = TapeStore(str(db))
    tape = Tape.load(str(FIXTURES_DIR / "single_turn.tape.sqlite"))
    run_id = store.save_tape(tape, run_id="single-turn-run")
    store.close()

    result = runner.invoke(
        app,
        [
            "tournament",
            run_id,
            "--agent",
            "tracefork.fixtures:single_turn_agent",
            "--candidate",
            "a:four",
            "--candidate",
            "b:5",
            "--store",
            str(db),
            "--k",
            "2",
            "--success-re",
            "4",
            "--failure-re",
            "nope-never-matches-xyz",
        ],
    )
    assert result.exit_code == 0, result.output
    progress_lines = [ln for ln in result.output.splitlines() if "[tournament] trial" in ln]
    # 2 candidates * k=2 = 4 trials.
    assert len(progress_lines) == 4
    assert progress_lines[-1].strip().endswith("4/4 complete")


def test_shapley_rank_emits_progress_per_completed_trial_across_both_phases():
    """tracefork-sis.51: `BlameEngine.shapley_rank` has no dedicated top-level
    CLI command (only `bench.py`, outside this lane, calls it), so this is a
    direct engine-level test of its `progress=` callback -- one call per
    COMPLETED trial across BOTH the internal sufficiency `rank()` pass and
    every coalition-walk trial, against ONE shared total. Fully offline/$0
    via the same single-exchange-tape trick the CLI-level tests above use."""
    from tracefork.blame import BlameEngine, StringMatchOracle
    from tracefork.fixtures import single_turn_agent
    from tracefork.wire import make_text_response

    tape = Tape.load(str(FIXTURES_DIR / "single_turn.tape.sqlite"))
    mutated = make_text_response("[tracefork] perturbed")

    def perturb_factory(step_idx):
        return mutated, None

    oracle = StringMatchOracle(success_re="4", failure_re="nope-never-matches-xyz")
    calls: list[tuple[int, int]] = []

    report = BlameEngine.shapley_rank(
        tape,
        single_turn_agent,
        oracle,
        perturb_factory=perturb_factory,
        k=2,
        m_samples=2,
        budget_usd=5.0,
        api_key="",
        progress=lambda c, t: calls.append((c, t)),
    )

    # n=1 exchange, k=2, m_samples=2 -> total = n*k*(1+m_samples) = 6.
    assert report.total_forks == 6
    assert len(calls) == 6
    assert calls == [(i, 6) for i in range(1, 7)]


def test_validate_runs_and_enforces_control(tmp_path):
    """`validate` runs fully offline; the negative control is enforced, not cosmetic."""
    out = tmp_path / "vr.json"
    result = runner.invoke(
        app,
        [
            "validate",
            "--k",
            "1",
            "--n-runs",
            "1",
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "negative control" in result.output

    data = json.loads(out.read_text())
    assert data["negative_control_max_flip"] < 0.30
    assert data["overall_top1_precision"] >= 0.7


def test_replay_check_passes_on_committed_fixture_corpus():
    result = runner.invoke(app, ["replay", "--check", str(FIXTURES_DIR)])
    assert result.exit_code == 0, result.output
    assert "fixtures passed" in result.output


def test_replay_check_fails_on_tampered_corpus(tmp_path):
    tamper_dir = tmp_path / "fixtures"
    shutil.copytree(FIXTURES_DIR, tamper_dir)

    manifest = json.loads((tamper_dir / "manifest.json").read_text())
    entry = manifest[0]
    tape_path = tamper_dir / entry["tape"]
    tape = Tape.load(str(tape_path))
    req, resp = tape.exchanges[0]
    tape.exchanges[0] = (req, resp + b" ")
    tape.save(str(tape_path))

    result = runner.invoke(app, ["replay", "--check", str(tamper_dir)])
    assert result.exit_code == 1, result.output
    assert "FAIL" in result.output


def test_replay_check_missing_manifest_exits_1(tmp_path):
    empty_dir = tmp_path / "empty_fixtures"
    empty_dir.mkdir()
    result = runner.invoke(app, ["replay", "--check", str(empty_dir)])
    assert result.exit_code == 1, result.output


def test_replay_without_check_still_requires_agent_and_tape():
    result = runner.invoke(app, ["replay"])
    assert result.exit_code == 1
    assert "Provide a tape path and --agent" in result.output


# ── export / ingest (OTel GenAI / OpenInference interop) ────────────────────


def test_export_requires_exactly_one_format_flag(tmp_path):
    db = tmp_path / "store.db"
    store = TapeStore(str(db))
    run_id = store.save_tape(_record_clean_tape(), run_id="testrun")
    store.close()

    for extra_flags in ([], ["--otel", "--openinference"]):
        result = runner.invoke(app, ["export", run_id, "--store", str(db), *extra_flags])
        assert result.exit_code == 1, result.output
        assert "exactly one" in result.output


def test_export_otel_writes_gen_ai_attributes(tmp_path):
    db = tmp_path / "store.db"
    store = TapeStore(str(db))
    run_id = store.save_tape(_record_clean_tape(), run_id="testrun")
    store.close()
    out = tmp_path / "trace.json"

    result = runner.invoke(app, ["export", run_id, "--store", str(db), "--otel", "-o", str(out)])
    assert result.exit_code == 0, result.output

    data = json.loads(out.read_text())
    spans = data["resourceSpans"][0]["scopeSpans"][0]["spans"]
    keys = {kv["key"] for span in spans for kv in span["attributes"]}
    # gen_ai.provider.name is the current OTel GenAI semconv attribute name
    # (gen_ai.system was deprecated/replaced -- tracefork-sis.50); the CLI's
    # export must never emit the deprecated one.
    assert "gen_ai.provider.name" in keys
    assert "gen_ai.system" not in keys
    assert "gen_ai.request.model" in keys


def test_export_otlp_endpoint_pushes_to_a_local_fake_collector(tmp_path):
    """tracefork-sis.60: `--otlp-endpoint` pushes the exported trace to a
    live OTLP collector -- proven against a real local HTTP server ("a local
    fake collector" per the item's own acceptance criteria), fully offline
    (127.0.0.1 only, no external network)."""
    import http.server
    import threading

    received: dict = {}

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 -- stdlib-mandated method name
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            received["path"] = self.path
            received["content_type"] = self.headers.get("Content-Type")
            received["body"] = json.loads(body)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, format_str: str, *args: object) -> None:  # noqa: A002
            pass  # silence the default stderr access log

    server = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        db = tmp_path / "store.db"
        store = TapeStore(str(db))
        run_id = store.save_tape(_record_clean_tape(), run_id="testrun")
        store.close()
        out = tmp_path / "trace.json"

        result = runner.invoke(
            app,
            [
                "export",
                run_id,
                "--store",
                str(db),
                "--otel",
                "-o",
                str(out),
                "--otlp-endpoint",
                f"http://127.0.0.1:{port}",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Pushed" in result.output
    finally:
        server.shutdown()
        thread.join(timeout=5)

    assert received["path"] == "/v1/traces"
    assert received["content_type"] == "application/json"
    exported_spans = received["body"]["resourceSpans"][0]["scopeSpans"][0]["spans"]
    assert len(exported_spans) == 3  # 1 root + 2 exchanges
    # The local JSON file and the pushed body are the exact same export.
    assert received["body"] == json.loads(out.read_text())


def test_export_otlp_endpoint_requires_otel_not_openinference(tmp_path):
    db = tmp_path / "store.db"
    store = TapeStore(str(db))
    run_id = store.save_tape(_record_clean_tape(), run_id="testrun")
    store.close()

    result = runner.invoke(
        app,
        [
            "export",
            run_id,
            "--store",
            str(db),
            "--openinference",
            "--otlp-endpoint",
            "http://127.0.0.1:1",
        ],
    )
    assert result.exit_code == 1
    assert "--otel" in result.output


def test_export_otlp_endpoint_unreachable_collector_is_clean_nonzero_exit(tmp_path):
    """A connection failure must exit 1 with a clean stderr line -- never a
    raw httpx traceback."""
    db = tmp_path / "store.db"
    store = TapeStore(str(db))
    run_id = store.save_tape(_record_clean_tape(), run_id="testrun")
    store.close()

    result = runner.invoke(
        app,
        [
            "export",
            run_id,
            "--store",
            str(db),
            "--otel",
            "--otlp-endpoint",
            "http://127.0.0.1:1",  # nothing listens on port 1 -> refused immediately
        ],
    )
    assert result.exit_code == 1
    assert "Traceback" not in result.output


def test_export_openinference_writes_llm_attributes(tmp_path):
    db = tmp_path / "store.db"
    store = TapeStore(str(db))
    run_id = store.save_tape(_record_clean_tape(), run_id="testrun")
    store.close()
    out = tmp_path / "dataset.json"

    result = runner.invoke(
        app, ["export", run_id, "--store", str(db), "--openinference", "-o", str(out)]
    )
    assert result.exit_code == 0, result.output

    data = json.loads(out.read_text())
    assert len(data["examples"]) == 2
    assert data["examples"][0]["metadata"]["openinference.span.kind"] == "LLM"


def test_export_without_run_id_or_tape_fails(tmp_path):
    db = tmp_path / "store.db"
    TapeStore(str(db)).close()
    result = runner.invoke(app, ["export", "--store", str(db), "--otel"])
    assert result.exit_code == 1
    assert "Provide a run_id or --tape" in result.output


def test_ingest_requires_exactly_one_format_flag(tmp_path):
    input_file = tmp_path / "trace.json"
    input_file.write_text("{}")
    for extra_flags in ([], ["--otel", "--openinference"]):
        result = runner.invoke(app, ["ingest", str(input_file), *extra_flags])
        assert result.exit_code == 1, result.output
        assert "exactly one" in result.output


def test_ingest_otel_builds_step_structure_and_warns_not_bit_exact(tmp_path):
    db = tmp_path / "store.db"
    store = TapeStore(str(db))
    run_id = store.save_tape(_record_clean_tape(), run_id="testrun")
    store.close()
    trace_path = tmp_path / "trace.json"
    export_result = runner.invoke(
        app, ["export", run_id, "--store", str(db), "--otel", "-o", str(trace_path)]
    )
    assert export_result.exit_code == 0, export_result.output

    out_tape = tmp_path / "ingested.tape.sqlite"
    result = runner.invoke(app, ["ingest", str(trace_path), "--otel", "-o", str(out_tape)])
    assert result.exit_code == 0, result.output
    assert "NOT $0" in result.output
    assert "blame-by-re-execution" in result.output

    ingested = Tape.load(str(out_tape))
    assert ingested.boundary == OTEL_INGESTED_BOUNDARY
    assert len(ingested.exchanges) == 2


# ── report --agent / --blame-report (divergence diagnostics + trust flags) ──


def _extract_report_data(html: str) -> dict:
    marker = "window.__TRACEFORK_DATA__ = "
    start = html.find(marker) + len(marker)
    end = html.find(";\n", start)
    return json.loads(html[start:end])


def test_report_writes_html_file(tmp_path):
    db = tmp_path / "store.db"
    store = TapeStore(str(db))
    run_id = store.save_tape(_record_clean_tape(), run_id="testrun")
    store.close()
    out = tmp_path / "report.html"

    result = runner.invoke(app, ["report", run_id, "--store", str(db), "-o", str(out)])
    assert result.exit_code == 0, result.output
    assert out.exists()
    data = _extract_report_data(out.read_text())
    assert data["replay"] == {}
    assert data["blame"] == {}


def _seed_store_with_many_branches(db: Path, n: int) -> str:
    """`n` branches, each carrying a genuinely large well-formed response (a
    real Anthropic text-message body, not a raw string `_tape_to_data` would
    otherwise fall back to a tiny `{"_raw": "sse"}` placeholder for) so each
    branch's embedded delta-tape detail is actually big -- reproducing the
    measured "branch_details dominates the payload" scenario."""
    from tracefork.wire import make_text_response

    store = TapeStore(str(db))
    tape = _record_clean_tape()
    run_id = store.save_tape(tape, run_id="testrun")
    big_resp = make_text_response("x" * 20_000)
    for i in range(n):
        delta = Tape(boundary=tape.boundary, agent_name=tape.agent_name)
        delta.append_exchange(f'{{"turn":{i}}}'.encode(), big_resp)
        store.save_branch(
            parent_run_id=run_id, divergence_step=0, delta_tape=delta, mutation_desc=f"branch {i}"
        )
    store.close()
    return run_id


def test_report_branch_details_cap_bytes_flag_truncates_many_branches(tmp_path):
    """tracefork-sis.56: `--branch-details-cap-bytes` caps the embedded
    per-branch delta-tape detail end to end (real CLI, store, and
    generate_report) so a run with many forks doesn't silently balloon the
    report."""
    db = tmp_path / "store.db"
    run_id = _seed_store_with_many_branches(db, 30)
    out = tmp_path / "report.html"

    result = runner.invoke(
        app,
        [
            "report",
            run_id,
            "--store",
            str(db),
            "-o",
            str(out),
            "--branch-details-cap-bytes",
            "50000",
        ],
    )
    assert result.exit_code == 0, result.output
    data = _extract_report_data(out.read_text())
    assert 0 < len(data["branch_details"]) < 30
    truncated = data["branch_details_truncated"]
    assert truncated is not None
    assert truncated["total_branches"] == 30
    assert truncated["cap_bytes"] == 50000


def test_report_branch_details_default_cap_applies_without_the_flag(tmp_path):
    """Omitting --branch-details-cap-bytes must still apply report.py's own
    documented default (262144 bytes) -- not embed everything unboundedly."""
    db = tmp_path / "store.db"
    run_id = _seed_store_with_many_branches(db, 30)
    out = tmp_path / "report.html"

    result = runner.invoke(app, ["report", run_id, "--store", str(db), "-o", str(out)])
    assert result.exit_code == 0, result.output
    data = _extract_report_data(out.read_text())
    assert 0 < len(data["branch_details"]) < 30
    assert data["branch_details_truncated"] is not None
    assert data["branch_details_truncated"]["cap_bytes"] == 262144


def test_report_with_agent_embeds_bit_exact_replay_receipt(tmp_path):
    """`--agent` replays the tape and embeds a bit-exactness receipt; the
    fixture's own producing agent must replay clean (no divergence)."""
    db = tmp_path / "store.db"
    store = TapeStore(str(db))
    run_id = store.save_tape(_record_clean_tape(), run_id="testrun")
    store.close()
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
    assert data["replay"]["divergence"] is None


def test_report_with_blame_report_embeds_trust_flags(tmp_path):
    db = tmp_path / "store.db"
    store = TapeStore(str(db))
    run_id = store.save_tape(_record_clean_tape(), run_id="testrun")
    store.close()
    out = tmp_path / "report.html"

    blame_path = tmp_path / f"blame_{run_id}.json"
    blame_path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "step_index": 0,
                        "flip_rate": 0.5,
                        "ci_lo": 0.2,
                        "ci_hi": 0.8,
                        "divergence_rate": 0.4,
                        "undefined": 4,
                        "trials": 10,
                        "valid_trials": 6,
                        "trustworthy": False,
                    }
                ]
            }
        )
    )

    result = runner.invoke(
        app,
        [
            "report",
            run_id,
            "--store",
            str(db),
            "--blame-report",
            str(blame_path),
            "-o",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    data = _extract_report_data(out.read_text())
    step0 = data["blame"]["0"]
    assert step0["divergence_rate"] == 0.4
    assert step0["undefined"] == 4
    assert step0["trustworthy"] is False


# ── diff (point-to-point / fork-branch diff) ─────────────────────────────────


def test_diff_branch_prints_receipt_and_exits_0_for_identical_delta(tmp_path):
    """A branch whose delta_tape re-records the SAME exchanges as the parent's
    tail (a no-op fork) diffs identical — a clean receipt, exit 0."""
    db = tmp_path / "store.db"
    store = TapeStore(str(db))
    tape = _record_clean_tape()
    run_id = store.save_tape(tape, run_id="parentrun")
    delta_tape = Tape(boundary=tape.boundary, agent_name=tape.agent_name)
    delta_tape.append_exchange(*tape.exchange(1))
    branch_id = store.save_branch(parent_run_id=run_id, divergence_step=1, delta_tape=delta_tape)
    store.close()

    result = runner.invoke(app, ["diff", run_id, branch_id, "--store", str(db)])
    assert result.exit_code == 0, result.output
    assert "identical" in result.output.lower() or "0 changed" in result.output.lower()


def test_diff_step_mode_compares_two_tapes_at_one_step(tmp_path):
    db = tmp_path / "store.db"
    store = TapeStore(str(db))
    tape = _record_clean_tape()
    run_a = store.save_tape(tape, run_id="run_a")
    run_b = store.save_tape(tape, run_id="run_b")
    store.close()

    result = runner.invoke(app, ["diff", run_a, run_b, "--step", "0", "--store", str(db)])
    assert result.exit_code == 0, result.output


# ── --version ─────────────────────────────────────────────────────────────


def test_version_flag_exits_0_and_prints_installed_package_version():
    from importlib import metadata

    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0, result.output
    assert metadata.version("tracefork") in result.output


def test_version_flag_is_eager_and_does_not_require_a_subcommand():
    """`--version` must short-circuit before Typer complains about a missing
    subcommand — it's meant to work standalone, the way `--help` already does."""
    result = runner.invoke(app, ["--version"])
    assert "Missing command" not in result.output
    assert "No such command" not in result.output


# ── validate --check regression tolerance ───────────────────────────────────


def test_validate_regressions_flags_a_real_drop_the_old_015_tolerance_would_hide():
    """`ValidationRunner` is fully deterministic (see test_faults.py's exact-
    value pin), so a 5-point precision drop between two runs at the same
    k/n_runs is a real regression, not noise — the old ±0.15 tolerance would
    have silently passed this."""
    from tracefork.cli import _validate_regressions
    from tracefork.constants import VALIDATE_CHECK_TOLERANCE

    old = {
        "top1_precision_by_class": {"corrupted_tool_output": 1.0},
        "negative_control_max_flip": 0.0,
    }
    new = {
        "top1_precision_by_class": {"corrupted_tool_output": 0.95},
        "negative_control_max_flip": 0.0,
    }

    assert VALIDATE_CHECK_TOLERANCE < 0.15, "tolerance must be tighter than the old 0.15"
    regressions = _validate_regressions(new, old, VALIDATE_CHECK_TOLERANCE)
    assert regressions, "a 0.05 drop must be flagged under the tightened tolerance"
    assert "corrupted_tool_output" in regressions[0]


def test_validate_regressions_is_clean_for_identical_reports():
    from tracefork.cli import _validate_regressions
    from tracefork.constants import VALIDATE_CHECK_TOLERANCE

    data = {
        "top1_precision_by_class": {"corrupted_tool_output": 1.0},
        "negative_control_max_flip": 0.0,
    }
    assert _validate_regressions(data, data, VALIDATE_CHECK_TOLERANCE) == []


def test_validate_regressions_flags_negative_control_regression():
    from tracefork.cli import _validate_regressions
    from tracefork.constants import VALIDATE_CHECK_TOLERANCE

    old = {"top1_precision_by_class": {}, "negative_control_max_flip": 0.0}
    new = {"top1_precision_by_class": {}, "negative_control_max_flip": 0.10}

    regressions = _validate_regressions(new, old, VALIDATE_CHECK_TOLERANCE)
    assert any("negative_control_max_flip" in r for r in regressions)


# ── CLI error routing: stderr, actionable messages, no tracebacks ──────────


def test_resolve_agent_rejects_spec_without_colon():
    import typer

    from tracefork.cli import _resolve_agent

    try:
        _resolve_agent("nocolonhere")
        raised = False
    except typer.Exit as exc:
        raised = True
        assert exc.exit_code == 1
    assert raised


def test_resolve_agent_names_expected_format_on_missing_colon(capsys):
    result = runner.invoke(
        app,
        [
            "replay",
            str(FIXTURES_DIR / "does-not-matter.tape.sqlite"),
            "--agent",
            "nocolonhere",
        ],
    )
    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "ValueError" not in result.output
    assert "pkg.module:fn" in result.output or "module:fn" in result.output.lower()


def test_resolve_agent_names_the_module_on_import_failure():
    result = runner.invoke(
        app,
        [
            "replay",
            str(FIXTURES_DIR / "does-not-matter.tape.sqlite"),
            "--agent",
            "nosuch.module:fn",
        ],
    )
    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "ModuleNotFoundError" not in result.output
    assert "nosuch.module" in result.output


def test_load_tape_or_exit_on_missing_file_has_no_traceback():
    result = runner.invoke(
        app,
        [
            "replay",
            "/nope/definitely-missing.tape.sqlite",
            "--agent",
            "tracefork.validate:synthetic_agent",
        ],
    )
    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit), (
        f"an uncaught {type(result.exception).__name__} means a real terminal run "
        "would print a raw traceback"
    )
    assert "OperationalError" not in result.output
    assert "/nope/definitely-missing.tape.sqlite" in result.output


def test_blame_closes_tapestore_even_on_early_budget_exit(tmp_path, monkeypatch):
    """`blame`'s `TapeStore` must be closed on every exit path — including the
    pre-flight budget-gate's early `raise typer.Exit(1)` — or the sqlite
    connection leaks. Asserts directly that `TapeStore.close` was actually
    invoked (rather than relying on GC timing / `ResourceWarning`, which is
    unreliable in-process: a CliRunner result can keep the frame — and its
    `db` local — alive via its captured traceback long enough that `gc.collect()`
    never reclaims it inside the test, silently passing a real leak)."""
    close_calls = []
    real_close = TapeStore.close

    def _tracking_close(self):
        close_calls.append(self)
        return real_close(self)

    monkeypatch.setattr(TapeStore, "close", _tracking_close)

    db_path = tmp_path / "store.db"
    store = TapeStore(str(db_path))
    run_id = store.save_tape(_record_clean_tape(), run_id="testrun")
    store.close()
    close_calls.clear()  # only count the `blame` command's own open/close below

    result = runner.invoke(
        app,
        [
            "blame",
            run_id,
            "--agent",
            "tracefork.validate:synthetic_agent",
            "--store",
            str(db_path),
            "--budget",
            "0",
        ],
    )
    assert result.exit_code == 1, result.output
    assert len(close_calls) == 1, (
        "blame's TapeStore.close() must run exactly once on the budget-gate exit path"
    )


def test_fork_prints_confinement_diagnostic_not_a_raw_sdk_traceback(tmp_path):
    """The Anthropic SDK wraps any exception its httpx transport raises in
    `APIConnectionError` (the same wrapping `nondet.find_divergence`
    documents and unwraps for `DivergenceError`) -- so a `ConfinementViolationError`
    raised by the guarded `socket.connect` during the tail-record call was
    reaching the user as a raw, unhandled `APIConnectionError`, not the
    intended `except ConfinementViolationError` diagnostic. `--allowed-host`
    excluding the real API host reproduces this fully offline/$0."""
    from tracefork.wire import make_text_response

    db_path = tmp_path / "store.db"
    store = TapeStore(str(db_path))
    run_id = store.save_tape(_record_clean_tape(), run_id="testrun")
    store.close()

    response_file = tmp_path / "mutated.bytes"
    response_file.write_bytes(make_text_response("mutated"))

    result = runner.invoke(
        app,
        [
            "fork",
            run_id,
            "--step",
            "0",
            "--response",
            str(response_file),
            "--agent",
            "tracefork.validate:synthetic_agent",
            "--store",
            str(db_path),
            "--allowed-host",
            "host-that-is-not-the-real-api.invalid",
        ],
    )
    assert result.exit_code == 1, result.output
    assert "Confinement violation" in result.output, result.output
    assert "APIConnectionError" not in result.output
    assert "Traceback" not in result.output


def test_fork_closes_tapestore_when_the_tail_record_call_is_confined_offline(tmp_path, monkeypatch):
    """`fork`'s `TapeStore` must be closed on an exception exit path too, not
    just the pre-flight-gate path another test already covers. Declaring
    `--allowed-host` that excludes the real API host makes
    `ConfinementSpec`'s `socket.connect` guard reject the tail-record call's
    connection attempt BEFORE any DNS/TCP happens (see `boundary_guard.py`),
    so this stays fully offline/$0 like the rest of the suite.

    Note: the SDK wraps the resulting `ConfinementViolationError` in an
    `httpx`/`anthropic` `APIConnectionError` (the same wrapping
    `nondet.find_divergence` already has to unwrap for `DivergenceError` —
    see that function's docstring), so `fork`'s own
    `except ConfinementViolationError` clause does not actually catch it
    here; that's a separate, pre-existing gap flagged in this session's
    final report, not something this test relies on. What this test proves
    is narrower and still real: whatever exits the function, `db.close()`
    must have run — asserted directly (see the `blame` test above for why
    GC/`ResourceWarning` is unreliable here) rather than inferred."""
    from tracefork.wire import make_text_response

    close_calls = []
    real_close = TapeStore.close

    def _tracking_close(self):
        close_calls.append(self)
        return real_close(self)

    monkeypatch.setattr(TapeStore, "close", _tracking_close)

    db_path = tmp_path / "store.db"
    store = TapeStore(str(db_path))
    run_id = store.save_tape(_record_clean_tape(), run_id="testrun")
    store.close()
    close_calls.clear()

    response_file = tmp_path / "mutated.bytes"
    response_file.write_bytes(make_text_response("mutated"))

    result = runner.invoke(
        app,
        [
            "fork",
            run_id,
            "--step",
            "0",
            "--response",
            str(response_file),
            "--agent",
            "tracefork.validate:synthetic_agent",
            "--store",
            str(db_path),
            "--allowed-host",
            "host-that-is-not-the-real-api.invalid",
        ],
    )
    assert result.exit_code == 1, result.output
    assert len(close_calls) == 1, (
        "fork's TapeStore.close() must run exactly once even when the tail-record call raises"
    )


def test_report_on_unknown_run_id_has_no_traceback(tmp_path):
    db_path = tmp_path / "store.db"
    from tracefork.store import TapeStore

    TapeStore(str(db_path)).close()

    result = runner.invoke(app, ["report", "no-such-run", "--store", str(db_path)])
    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "KeyError" not in result.output
    assert "no-such-run" in result.output


# ── packaging: tracefork_spike must never ship in the installed wheel ──────


def test_tracefork_spike_is_not_a_wheel_package():
    """`tracefork_spike` (Spike 0, retired) must stay source-checkout-only —
    shipping it as a second top-level installed package would claim the
    `tracefork_spike` name on every user's `sys.path` and offer a colliding,
    incompatible `Tape` class. Regression guard for `pyproject.toml`'s
    `[tool.hatch.build.targets.wheel]` `packages` list and
    `[tool.coverage.run]`'s `source` list."""
    import tomllib

    pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    data = tomllib.loads(pyproject_path.read_text())

    wheel_packages = data["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
    assert "src/tracefork_spike" not in wheel_packages
    assert "src/tracefork" in wheel_packages

    coverage_source = data["tool"]["coverage"]["run"]["source"]
    assert "src/tracefork_spike" not in coverage_source
    assert not any("tracefork_spike" in s for s in coverage_source)
