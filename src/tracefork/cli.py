"""tracefork CLI — entry point for all commands.

    tracefork <command> [args]

Commands: replay, verify, fork, report, serve, blame, validate, bench, export,
ingest, prune, proxy.
"""

from __future__ import annotations

from pathlib import Path

import typer

from tracefork.config import TraceforkConfig

app = typer.Typer(name="tracefork", help="Time-travel debugger for AI agents.")

# Module-level so `TRACEFORK_DB_PATH`/`TRACEFORK_BUDGET_USD` (if set) become the
# CLI's own option defaults below; unset (the common case), these equal today's
# hardcoded literals ("store.db", 5.0) exactly — see `config.py`.
_DEFAULT_CONFIG = TraceforkConfig.from_env()


@app.command()
def replay(
    tape_path: Path = typer.Argument(None, help="Path to a .tape.sqlite file"),  # noqa: B008
    agent: str = typer.Option(None, "--agent", "-a", help="Import path of agent fn (pkg.mod:fn)"),
    check: Path = typer.Option(  # noqa: B008
        None,
        "--check",
        help="Path to a committed fixture corpus dir (replay-as-regression gate): "
        "asserts every fixture tape replays bit-exact and its digest() matches "
        "the corpus's manifest.json",
    ),
) -> None:
    """Replay a tape and print the verification receipt, or gate a fixture corpus with --check."""
    import importlib

    from tracefork.certificate import certificate_from_verification
    from tracefork.replay import ReplayVerifier
    from tracefork.tape import Tape

    if check is not None:
        _run_replay_check(check)
        return

    if tape_path is None or agent is None:
        typer.echo("Provide a tape path and --agent, or use --check <fixtures dir>")
        raise typer.Exit(1)

    tape = Tape.load(str(tape_path))

    module_path, fn_name = agent.rsplit(":", 1)
    mod = importlib.import_module(module_path)
    agent_fn = getattr(mod, fn_name)

    result = ReplayVerifier(tape, agent_fn).verify()
    result.certificate = certificate_from_verification(result, tape)
    _print_receipt(tape_path, result)
    raise typer.Exit(0 if result.bit_exact else 1)


def _run_replay_check(fixtures_dir: Path) -> None:
    """`replay --check` body: gate a committed tape corpus. Exits 1 on any
    fixture failure (missing manifest, non-bit-exact replay, or digest drift)."""
    from tracefork.replay import run_fixture_corpus_check

    if not (fixtures_dir / "manifest.json").exists():
        typer.echo(f"No manifest.json found under {fixtures_dir}")
        raise typer.Exit(1)

    result = run_fixture_corpus_check(fixtures_dir)

    typer.echo(f"\n  tracefork replay --check {fixtures_dir}")
    typer.echo(f"  {'─' * 60}")
    for f in result.fixtures:
        status = "PASS" if f.passed else "FAIL"
        typer.echo(f"  [{status}] {f.name:<20} {f.reason}")
    n_pass = sum(1 for f in result.fixtures if f.passed)
    typer.echo(f"\n  {n_pass}/{len(result.fixtures)} fixtures passed\n")
    raise typer.Exit(0 if result.all_passed else 1)


@app.command()
def verify(
    tape_path: Path = typer.Argument(None, help="Single tape to verify"),  # noqa: B008
    agent: str = typer.Option(None, "--agent", "-a", help="Import path of agent fn"),
    corpus: bool = typer.Option(
        False, "--corpus", help="Verify all tapes in experiments/validation_tapes/"
    ),
) -> None:
    """Verify bit-exact replay. Exit 1 on drift."""
    import importlib

    from tracefork.certificate import certificate_from_verification
    from tracefork.replay import ReplayVerifier
    from tracefork.tape import Tape

    if corpus:
        corpus_dir = Path("experiments/validation_tapes")
        tapes = list(corpus_dir.glob("*.tape.sqlite"))
        if not tapes:
            typer.echo("No tapes found in experiments/validation_tapes/")
            raise typer.Exit(1)
        for tp in sorted(tapes):
            typer.echo(f"  {tp.name}: skipped (agent not specified per-tape)")
        typer.echo(f"Corpus: {len(tapes)} tapes scanned")
        raise typer.Exit(0)

    if tape_path is None or agent is None:
        typer.echo("Provide --agent and a tape path, or use --corpus")
        raise typer.Exit(1)

    tape = Tape.load(str(tape_path))
    module_path, fn_name = agent.rsplit(":", 1)
    mod = importlib.import_module(module_path)
    agent_fn = getattr(mod, fn_name)
    result = ReplayVerifier(tape, agent_fn).verify()
    result.certificate = certificate_from_verification(result, tape)
    _print_receipt(tape_path, result)
    raise typer.Exit(0 if result.bit_exact else 1)


@app.command()
def fork(
    run_id: str = typer.Argument(..., help="Parent run_id to fork from"),
    step: int = typer.Option(..., "--step", "-s", help="Exchange index to diverge at"),
    response_file: Path = typer.Option(  # noqa: B008
        ..., "--response", "-r", help="Path to .bytes file containing mutated response"
    ),
    agent: str = typer.Option(..., "--agent", "-a", help="Import path of post-fork agent fn"),
    store: Path = typer.Option(  # noqa: B008
        Path(_DEFAULT_CONFIG.db_path), "--store", help="Path to store.db"
    ),
    desc: str = typer.Option("", "--desc", "-d", help="Human description of mutation"),
) -> None:
    """Fork a run at a step with a mutated response, record the new branch."""
    import importlib

    from tracefork.fork import BranchSpec, ForkEngine
    from tracefork.store import TapeStore

    db = TapeStore(str(store))
    parent_tape = db.load_tape(run_id)

    mutated_response = response_file.read_bytes()

    module_path, fn_name = agent.rsplit(":", 1)
    mod = importlib.import_module(module_path)
    agent_fn = getattr(mod, fn_name)

    spec = BranchSpec(
        divergence_step=step,
        mutated_response=mutated_response,
        mutation_desc=desc,
    )

    branch = ForkEngine.fork(parent_tape, spec, agent_fn)

    branch_id = db.save_branch(
        parent_run_id=run_id,
        divergence_step=step,
        delta_tape=branch.delta_tape,
        mutation_desc=desc,
    )

    typer.echo("\n  Fork created")
    typer.echo(f"  branch_id       {branch_id}")
    typer.echo(f"  parent_run_id   {run_id}")
    typer.echo(f"  divergence_step {step}")
    typer.echo(f"  delta_exchanges {len(branch.delta_tape.exchanges)}")
    typer.echo(f"  description     {desc or '(none)'}\n")


@app.command()
def report(
    run_id: str = typer.Argument(None, help="run_id to report on (from store)"),
    tape_path: Path = typer.Option(  # noqa: B008
        None, "--tape", "-t", help="Path to a .tape.sqlite file"
    ),
    output: Path = typer.Option(  # noqa: B008
        Path("report.html"), "--output", "-o", help="Output HTML file"
    ),
    store: Path = typer.Option(  # noqa: B008
        Path(_DEFAULT_CONFIG.db_path), "--store", help="Path to store.db"
    ),
    agent: str = typer.Option(
        None,
        "--agent",
        "-a",
        help="Import path of the agent fn (pkg.mod:fn) that produced this tape "
        "(pkg.mod:fn); replays it and embeds a bit-exactness receipt — with a "
        "structured divergence diagnostic on drift — in the report",
    ),
    blame_report: Path = typer.Option(  # noqa: B008
        None,
        "--blame-report",
        help="Optional blame_<run_id>.json (from `tracefork blame`) to embed "
        "per-step trust flags (divergence rate, UNDEFINED trial counts) in the report",
    ),
) -> None:
    """Generate a self-contained HTML report from a tape."""
    import json as _json

    from tracefork.report import generate_report
    from tracefork.tape import Tape

    if tape_path:
        tape = Tape.load(str(tape_path))
    elif run_id:
        from tracefork.store import TapeStore

        db = TapeStore(str(store))
        tape = db.load_tape(run_id)
    else:
        typer.echo("Provide a run_id or --tape path")
        raise typer.Exit(1)

    replay_data = None
    if agent:
        import importlib

        from tracefork.replay import ReplayVerifier, verification_result_to_dict

        module_path, fn_name = agent.rsplit(":", 1)
        agent_fn = getattr(importlib.import_module(module_path), fn_name)
        result = ReplayVerifier(tape, agent_fn).verify()
        replay_data = verification_result_to_dict(result)

    blame_dict = None
    if blame_report is not None:
        blame_data = _json.loads(blame_report.read_text())
        blame_dict = {r["step_index"]: r for r in blame_data.get("results", [])}

    generate_report(tape, output, blame=blame_dict, replay=replay_data)
    typer.echo(f"Report written to {output}")


@app.command()
def serve(
    store: Path = typer.Option(  # noqa: B008
        Path(_DEFAULT_CONFIG.db_path), "--store", help="Path to store.db"
    ),
    port: int = typer.Option(7777, "--port", "-p", help="Port to listen on"),
) -> None:
    """Start the tracefork web UI server on port 7777."""
    import uvicorn

    from tracefork.server import app as fastapi_app
    from tracefork.server import init_store

    init_store(str(store))
    typer.echo(f"  tracefork serve → http://127.0.0.1:{port}")
    uvicorn.run(fastapi_app, host="127.0.0.1", port=port, workers=1, log_level="warning")


@app.command()
def blame(
    run_id: str = typer.Argument(..., help="run_id to analyze"),
    agent: str = typer.Option(
        ...,
        "--agent",
        "-a",
        help="Import path of the agent fn (pkg.mod:fn) that produced this run; "
        "it is re-run for each fork and must be deterministic up to the fork point",
    ),
    k: int = typer.Option(10, "--k", help="Forks per candidate step"),
    budget: float = typer.Option(_DEFAULT_CONFIG.budget_usd, "--budget", help="USD spend cap"),
    perturbation: str = typer.Option(
        "[tracefork] this step did not complete as recorded",
        "--perturbation",
        help="Text injected as the counterfactual response",
    ),
    success_re: str = typer.Option("SUCCESS", "--success-re", help="Regex for success outcome"),
    failure_re: str = typer.Option("FAIL", "--failure-re", help="Regex for failure outcome"),
    ci_method: str = typer.Option(
        "wilson", "--ci-method", help="Proportion CI: wilson|jeffreys|clopper_pearson|agresti_coull"
    ),
    confidence: float = typer.Option(0.95, "--confidence", help="CI confidence level (0,1)"),
    fdr_q: float = typer.Option(
        0.10, "--fdr-q", help="Benjamini-Hochberg false-discovery-rate for the responsible set"
    ),
    null_flip_rate: float = typer.Option(
        0.05, "--null-flip-rate", help="Chance-flip null the binomial test scores each step against"
    ),
    store: Path = typer.Option(  # noqa: B008
        Path(_DEFAULT_CONFIG.db_path), "--store", help="Path to store.db"
    ),
) -> None:
    """Run causal blame analysis on a recorded run.

    For each exchange, the agent is re-run with that step's response perturbed
    and the counterfactual tail recorded against the real API (budget-capped).
    The offline, $0 proof that blame correctly fingers known faults is
    `tracefork validate`.
    """
    if not run_id or not all(c.isalnum() or c in "-_" for c in run_id):
        raise typer.BadParameter("run_id must be alphanumeric (with '-' or '_')")

    import importlib
    import json
    import os

    from tracefork.blame import BlameEngine, BudgetGovernor, CIMethod, StringMatchOracle
    from tracefork.store import TapeStore
    from tracefork.wire import make_text_response

    try:
        method = CIMethod(ci_method)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    db = TapeStore(str(store))
    tape = db.load_tape(run_id)

    module_path, fn_name = agent.rsplit(":", 1)
    agent_fn = getattr(importlib.import_module(module_path), fn_name)

    oracle = StringMatchOracle(success_re=success_re, failure_re=failure_re)
    est = BudgetGovernor.estimate(tape, k=k)

    typer.echo(f"\n  Blame estimate: {est.n_forks} forks, ~${est.est_usd:.2f}")
    if est.est_usd > budget:
        typer.echo(f"  Estimated cost ${est.est_usd:.2f} exceeds budget ${budget:.2f}.")
        typer.echo("  Use --budget to increase or --k to reduce trials.")
        raise typer.Exit(1)

    mutated = make_text_response(perturbation)

    def perturb_factory(step_idx: int):
        # tail_transport=None → the counterfactual tail hits the real API.
        return mutated, None

    report = BlameEngine.rank(
        tape,
        agent_fn,
        oracle,
        perturb_factory=perturb_factory,
        k=k,
        budget_usd=budget,
        api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        ci_method=method,
        confidence=confidence,
        null_flip_rate=null_flip_rate,
        fdr_q=fdr_q,
    )

    ci_pct = round(confidence * 100)
    typer.echo(
        f"\n  run-{run_id} · blame analysis · k={k} · {report.total_forks} forks "
        f"· {method.value} {ci_pct}% CI\n"
    )
    ci_hdr = f"{ci_pct}% CI"
    typer.echo(
        f"  {'rank':<5} {'step':<8} {'flip-rate':<12} {ci_hdr:<22} "
        f"{'undef':<7} {'q-value':<10} interpretation"
    )
    typer.echo(f"  {'─' * 88}")
    for rank, r in enumerate(report.results, 1):
        ci_str = f"[{r.ci_lo:.2f}, {r.ci_hi:.2f}]"
        undef_str = f"{r.undefined}/{r.trials}"
        flag = " ⚠" if not r.trustworthy else ""
        typer.echo(
            f"  {rank:<5} step-{r.step_index:<3} {r.flip_rate:<12.2f} "
            f"{ci_str:<22} {undef_str:<7} {r.q_value:<10.3g} {r.interpretation}{flag}"
        )
    if report.responsible_set:
        steps = ", ".join(f"step-{s}" for s in report.responsible_set)
        typer.echo(f"\n  responsible set (FDR q≤{fdr_q}): {steps}")
    else:
        typer.echo(f"\n  responsible set (FDR q≤{fdr_q}): (none pass the significance bar)")
    typer.echo("")

    report_path = Path(f"blame_{run_id}.json")
    report_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "k": k,
                "ci_method": method.value,
                "confidence": confidence,
                "null_flip_rate": null_flip_rate,
                "fdr_q": fdr_q,
                "responsible_set": report.responsible_set,
                "results": [
                    {
                        "step_index": r.step_index,
                        "flip_rate": r.flip_rate,
                        "ci_lo": r.ci_lo,
                        "ci_hi": r.ci_hi,
                        "valid_trials": r.valid_trials,
                        "undefined": r.undefined,
                        "divergences": r.divergences,
                        "divergence_rate": r.divergence_rate,
                        "trustworthy": r.trustworthy,
                        "p_value": r.p_value,
                        "q_value": r.q_value,
                        "responsible": r.responsible,
                        "interpretation": r.interpretation,
                    }
                    for r in report.results
                ],
            },
            indent=2,
        )
    )
    typer.echo(f"  Report saved to {report_path}")

    edge_ids = db.save_blame_report(run_id, report)
    typer.echo(f"  Causal edges persisted   {len(edge_ids)}")


@app.command()
def validate(
    k: int = typer.Option(3, "--k", help="Forks per candidate step per run"),
    n_runs: int = typer.Option(5, "--n-runs", help="Runs per fault class"),
    output: Path = typer.Option(  # noqa: B008
        Path("validation_report.json"), "--output", "-o"
    ),
    check: bool = typer.Option(False, "--check", help="Diff vs committed report (regression gate)"),
) -> None:
    """Run fault-injection validation suite; produce validation_report.json."""
    import json as _json

    from tracefork.validate import run_all_fault_classes

    typer.echo(f"\n  tracefork validate — k={k}, n_runs={n_runs} per class")
    typer.echo(f"  {'─' * 50}")

    results = run_all_fault_classes(k=k, n_runs=n_runs)

    overall_precision = sum(r["top1_precision"] for r in results.values()) / len(results)
    max_ctrl = max(r["negative_control_max_flip"] for r in results.values())

    report_data = {
        "top1_precision_by_class": {fc: v["top1_precision"] for fc, v in results.items()},
        "overall_top1_precision": overall_precision,
        "negative_control_max_flip": max_ctrl,
        "n_runs_per_class": n_runs,
        "k": k,
        "reproduce_cmd": f"tracefork validate --k {k} --n-runs {n_runs}",
    }

    for fault_class, data in results.items():
        status = "PASS" if data["top1_precision"] >= 0.7 else "WARN"
        typer.echo(f"  [{status}] {fault_class:<35} top-1: {data['top1_precision']:.2f}")

    typer.echo(f"\n  overall top-1 precision: {overall_precision:.2f}")
    typer.echo(f"  negative control max flip: {max_ctrl:.2f} (threshold 0.30)")

    output.write_text(_json.dumps(report_data, indent=2))
    typer.echo(f"\n  Report saved to {output}\n")

    control_threshold = 0.30
    if max_ctrl >= control_threshold:
        typer.echo(
            f"  [FAIL] negative control max flip {max_ctrl:.2f} ≥ {control_threshold:.2f} "
            "— blame is firing on no-op perturbations; the precision number is not trustworthy."
        )
        raise typer.Exit(1)

    if check:
        committed = Path("experiments/validation_report_committed.json")
        if not committed.exists():
            typer.echo("  No committed report found — run without --check to create one.")
            raise typer.Exit(1)
        old = _json.loads(committed.read_text())
        regressions = []
        for fc, new_prec in report_data["top1_precision_by_class"].items():
            old_prec = old.get("top1_precision_by_class", {}).get(fc, 0.0)
            if new_prec < old_prec - 0.15:
                regressions.append(f"{fc}: {old_prec:.2f} → {new_prec:.2f}")
        old_ctrl = old.get("negative_control_max_flip", 0.0)
        if max_ctrl > old_ctrl + 0.15:
            regressions.append(f"negative_control_max_flip: {old_ctrl:.2f} → {max_ctrl:.2f}")
        if regressions:
            typer.echo("  REGRESSION detected:")
            for r_str in regressions:
                typer.echo(f"    {r_str}")
            raise typer.Exit(1)
        typer.echo("  No regressions vs committed report.")


@app.command()
def bench(
    k: int = typer.Option(3, "--k", help="Forks per candidate step per scenario"),
    m_samples: int = typer.Option(2, "--m-samples", help="Temporal-Shapley permutation samples"),
    output: Path = typer.Option(  # noqa: B008
        Path("bench_report.json"), "--output", "-o"
    ),
) -> None:
    """Long-tape competing-fault benchmark for the coalition/temporal-Shapley
    blame engine (`shapley_rank`).

    Unlike `validate` (a single planted fault vs. an inert control on a short
    tape), `bench` plants SEVERAL causally-distinct faults on one longer tape
    at once -- a true root cause, a downstream echo that must not be blamed as
    root, and a two-part necessary-not-sufficient conjunction -- and measures
    whether the engine's necessity/sufficiency classification matches ground
    truth for each, including the one case that does not resolve cleanly (a
    documented limitation, not hidden). Offline, $0. See
    `tracefork.competing_faults` and `tracefork.bench` module docstrings, and
    README -> Validation scope, for exactly what each case means and why.
    """
    import json as _json

    from tracefork.bench import run_bench

    typer.echo(f"\n  tracefork bench — k={k}, m_samples={m_samples}")
    typer.echo(f"  {'─' * 60}")

    report = run_bench(k=k, m_samples=m_samples)

    for c in report.cases:
        status = "OK" if c.resolved else "LIMITATION"
        nec = f"necessity(exp={c.expected_necessity!s:<5} act={c.actual_necessity!s:<5})"
        suff = f"sufficiency(exp={c.expected_sufficiency!s:<5} act={c.actual_sufficiency!s:<5})"
        typer.echo(f"  [{status:<10}] {c.name:<32} {nec} {suff}")
        if c.note:
            typer.echo(f"               {c.note}")

    typer.echo(
        f"\n  competing-fault discrimination: {report.accuracy:.2f} "
        f"({report.n_resolved}/{report.n_cases}), "
        f"95% CI [{report.ci_lo:.2f}, {report.ci_hi:.2f}]"
    )
    typer.echo(
        f"  context only, not reproduced here: published Who&When log-based "
        f"step-attribution top-1 anchor ~{report.who_and_when_anchor:.3f} "
        f"(see README — Validation scope)"
    )

    output.write_text(
        _json.dumps(
            {
                "k": k,
                "m_samples": m_samples,
                "accuracy": report.accuracy,
                "n_resolved": report.n_resolved,
                "n_cases": report.n_cases,
                "ci_lo": report.ci_lo,
                "ci_hi": report.ci_hi,
                "who_and_when_anchor": report.who_and_when_anchor,
                "cases": [
                    {
                        "name": c.name,
                        "step_index": c.step_index,
                        "role": c.role.value,
                        "expected_necessity": c.expected_necessity,
                        "expected_sufficiency": c.expected_sufficiency,
                        "actual_necessity": c.actual_necessity,
                        "actual_sufficiency": c.actual_sufficiency,
                        "resolved": c.resolved,
                        "note": c.note,
                    }
                    for c in report.cases
                ],
            },
            indent=2,
        )
    )
    typer.echo(f"\n  Report saved to {output}\n")

    unexpected = report.unexpected_failures()
    if unexpected:
        typer.echo("  REGRESSION: unresolved cases beyond the known limitation:")
        for c in unexpected:
            typer.echo(f"    {c.name}")
        raise typer.Exit(1)


@app.command()
def export(
    run_id: str = typer.Argument(None, help="run_id to export (from store)"),
    tape_path: Path = typer.Option(  # noqa: B008
        None, "--tape", "-t", help="Path to a .tape.sqlite file"
    ),
    otel: bool = typer.Option(
        False, "--otel", help="Emit an OTel GenAI trace (OTLP/JSON spans, gen_ai.* attributes)"
    ),
    openinference: bool = typer.Option(
        False, "--openinference", help="Emit an OpenInference-style dataset JSON (llm.* attributes)"
    ),
    blame_report: Path = typer.Option(  # noqa: B008
        None,
        "--blame-report",
        help="Optional blame_<run_id>.json (from `tracefork blame`) to attach "
        "flip-rate/CI as tracefork.blame.* attributes",
    ),
    output: Path = typer.Option(  # noqa: B008
        Path("export.json"), "--output", "-o", help="Output JSON file"
    ),
    store: Path = typer.Option(  # noqa: B008
        Path(_DEFAULT_CONFIG.db_path), "--store", help="Path to store.db"
    ),
) -> None:
    """Export a tape (+ optional blame report) for external observability tooling.

    Exactly one of --otel / --openinference is required. This is a pure data
    export (gen_ai.*/llm.* attributes as plain JSON) — no opentelemetry-sdk
    install needed to produce or consume it. See `tracefork ingest` for the
    reverse direction and its blame-only, not-bit-exact-replay caveat.
    """
    import json as _json

    from tracefork.interop import (
        blame_report_from_json,
        build_openinference_dataset,
        build_otel_trace,
    )
    from tracefork.tape import Tape

    if otel == openinference:
        typer.echo("Pass exactly one of --otel or --openinference")
        raise typer.Exit(1)

    if tape_path:
        tape = Tape.load(str(tape_path))
    elif run_id:
        from tracefork.store import TapeStore

        db = TapeStore(str(store))
        tape = db.load_tape(run_id)
    else:
        typer.echo("Provide a run_id or --tape path")
        raise typer.Exit(1)

    blame = None
    if blame_report is not None:
        blame = blame_report_from_json(_json.loads(blame_report.read_text()))

    data = (
        build_otel_trace(tape, blame=blame)
        if otel
        else build_openinference_dataset(tape, blame=blame)
    )
    output.write_text(_json.dumps(data, indent=2))
    kind = "OTel GenAI trace" if otel else "OpenInference dataset"
    typer.echo(f"  {kind} written to {output} ({len(tape.exchanges)} exchange(s))")


@app.command()
def ingest(
    input_path: Path = typer.Argument(  # noqa: B008
        ..., help="Path to an OTel OTLP/JSON trace or OpenInference dataset JSON"
    ),
    otel: bool = typer.Option(False, "--otel", help="Input is an OTel OTLP/JSON trace export"),
    openinference: bool = typer.Option(
        False, "--openinference", help="Input is an OpenInference-style dataset JSON"
    ),
    output: Path = typer.Option(  # noqa: B008
        Path("ingested.tape.sqlite"), "--output", "-o", help="Output tape file"
    ),
) -> None:
    """Build a tape's STEP STRUCTURE from an externally-produced OTel/OpenInference
    trace — for blame-by-re-execution only.

    IMPORTANT: the resulting tape is NOT bit-exact replayable. Its request
    bytes are synthesized placeholders (model id only — span attributes don't
    carry the original prompt), so `tracefork replay`/`fork` against a real
    agent will correctly diverge on the very first step. See `interop.py`'s
    module docstring for the precise scope of what an ingested tape supports.
    """
    import json as _json

    from tracefork.interop import ingest_openinference_dataset, ingest_otel_trace

    if otel == openinference:
        typer.echo("Pass exactly one of --otel or --openinference")
        raise typer.Exit(1)

    data = _json.loads(input_path.read_text())
    tape = ingest_otel_trace(data) if otel else ingest_openinference_dataset(data)
    tape.save(str(output))

    typer.echo(f"\n  Ingested {len(tape.exchanges)} exchange(s) -> {output}")
    typer.echo("  NOTE: step structure only, reconstructed from span attributes — NOT")
    typer.echo("  tracefork's own recorded bytes. Supports blame-by-re-execution, NOT $0")
    typer.echo("  bit-exact replay (`replay`/`fork` will diverge on this tape).\n")


@app.command()
def prune(
    older_than_days: float = typer.Option(
        None,
        "--older-than-days",
        help="Archive tapes with created_at older than N days ago",
    ),
    run_id: list[str] = typer.Option(  # noqa: B008
        [], "--run-id", help="Explicit run_id to archive (repeatable)"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Compute the candidate set; mutate nothing"
    ),
    store: Path = typer.Option(  # noqa: B008
        Path(_DEFAULT_CONFIG.db_path), "--store", help="Path to store.db"
    ),
) -> None:
    """Archive tapes (and their branches) — never hard-delete.

    Mirrors git gc / borg prune's mark-and-sweep-with-soft-archive
    discipline: matching rows move to tapes_archived/branches_archived and
    stay queryable there forever; reclaiming that space is a distinct,
    out-of-scope, higher-risk step. A tape matches if it's older than
    --older-than-days OR named by a repeatable --run-id; passing neither
    matches nothing. --dry-run previews the candidate set with zero writes.

    NOTE: report links for a pruned run_id go stale — server.py's
    list_runs/get_run/get_branch correctly 404 it via the existing KeyError
    path, same as any unknown run_id.

    Always exits 0: pruning is a maintenance operation, not a pass/fail gate.
    """
    import datetime as _dt

    from tracefork.store import TapeStore

    older_than_iso = None
    if older_than_days is not None:
        cutoff = _dt.datetime.now(_dt.UTC) - _dt.timedelta(days=older_than_days)
        older_than_iso = cutoff.isoformat()

    db = TapeStore(str(store))
    report = db.prune(older_than_iso=older_than_iso, run_ids=list(run_id), dry_run=dry_run)

    label = "prune [dry-run]" if dry_run else "prune"
    typer.echo(f"\n  tracefork {label}")
    typer.echo(f"  {'─' * 50}")
    if not report.tapes_archived:
        typer.echo("  no candidates matched (nothing archived)")
    else:
        for rid in report.tapes_archived:
            typer.echo(f"    {rid}")
        verb = "would archive" if dry_run else "Archived"
        typer.echo(
            f"\n  {verb} {len(report.tapes_archived)} tape(s), "
            f"{len(report.branches_archived)} branch(es)"
        )
    typer.echo("")


@app.command()
def proxy(
    mode: str = typer.Argument(..., help="record | replay"),
    tape_path: Path = typer.Option(  # noqa: B008
        ..., "--tape", "-t", help="Path to a .tape.sqlite file"
    ),
    port: int = typer.Option(8899, "--port", "-p", help="Port to listen on (binds 127.0.0.1 only)"),
    upstream: str = typer.Option(
        None,
        "--upstream",
        help="Upstream base URL, e.g. https://api.anthropic.com (record mode only)",
    ),
    matcher: str = typer.Option(
        "identity",
        "--matcher",
        help="Registered RequestMatcher name (identity|gemini|bedrock|redacting)",
    ),
) -> None:
    """Localhost base-URL record/replay proxy for non-Python / non-httpx clients
    (curl, Node, Go, ...): point the client's base URL at
    http://127.0.0.1:<port> instead of the provider directly.

    Record mode forwards each request to --upstream over the real network and
    tees request+response bytes into --tape (created fresh if it doesn't exist
    yet). Replay mode serves recorded bytes from --tape with NO upstream — an
    unrecorded request, or a real request-body change, is a hard error (HTTP
    502).

    This mode has NO in-process NondetSource (a non-Python client can't read
    one), so it sits OUTSIDE tracefork's full determinism boundary: bit-exact
    replay depends on the client sending a canonically-identical request each
    time. See the README's proxy section and `proxy.py`'s module docstring.
    """
    import asyncio

    import uvicorn

    from tracefork.matcher import get_matcher
    from tracefork.proxy import build_record_app, build_replay_app
    from tracefork.tape import Tape

    if mode not in ("record", "replay"):
        typer.echo("mode must be 'record' or 'replay'")
        raise typer.Exit(1)

    try:
        m = get_matcher(matcher)
    except KeyError as exc:
        raise typer.BadParameter(str(exc)) from exc

    if mode == "record":
        if not upstream:
            typer.echo("record mode requires --upstream <base_url>")
            raise typer.Exit(1)
        tape = Tape.load(str(tape_path)) if tape_path.exists() else Tape()
        record_app = build_record_app(tape, upstream, matcher=m)

        typer.echo(f"\n  tracefork proxy record -> http://127.0.0.1:{port} -> {upstream}")
        typer.echo(f"  tape: {tape_path}\n")
        try:
            uvicorn.run(record_app, host="127.0.0.1", port=port, workers=1, log_level="warning")
        finally:
            asyncio.run(record_app.state.proxy.aclose())
            tape.save(str(tape_path))
            typer.echo(f"\n  Tape saved to {tape_path} ({len(tape.exchanges)} exchange(s))")
        return

    if not tape_path.exists():
        typer.echo(f"No tape found at {tape_path}")
        raise typer.Exit(1)
    tape = Tape.load(str(tape_path))
    replay_app = build_replay_app(tape, matcher=m)
    typer.echo(f"\n  tracefork proxy replay -> http://127.0.0.1:{port}")
    typer.echo(f"  tape: {tape_path} ({len(tape.exchanges)} exchange(s))\n")
    uvicorn.run(replay_app, host="127.0.0.1", port=port, workers=1, log_level="warning")


def _print_receipt(tape_path: Path, result) -> None:
    from tracefork.replay import DriftDoctor

    status = "PASS" if result.bit_exact else "FAIL"
    typer.echo("\n  tracefork — replay receipt")
    typer.echo(f"  {'─' * 40}")
    typer.echo(f"  tape            {tape_path.name}")
    typer.echo(f"  exchanges       {result.matched}/{result.total} matched")
    typer.echo(f"  fingerprint     {'match' if result.fingerprints_match else 'MISMATCH'}")
    typer.echo(f"  result          {status}")
    certificate = getattr(result, "certificate", None)
    if certificate is not None:
        typer.echo(f"  certificate     {certificate.strength.value}")
    if result.divergence:
        cause = DriftDoctor.classify(result.divergence)
        typer.echo(f"  drift cause     {cause.value}")
        typer.echo(f"  at exchange     #{result.divergence.step_index}")
    typer.echo("")
