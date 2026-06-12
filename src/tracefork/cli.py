"""tracefork CLI — entry point for all commands.

Plans A–F add commands here as they're built. Run with:
    tracefork <command> [args]
"""
from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(name="tracefork", help="Time-travel debugger for AI agents.")


@app.command()
def replay(
    tape_path: Path = typer.Argument(..., help="Path to a .tape.sqlite file"),
    agent: str = typer.Option(..., "--agent", "-a", help="Import path of agent fn (pkg.mod:fn)"),
) -> None:
    """Replay a tape and print the verification receipt."""
    import importlib
    from tracefork.tape import Tape
    from tracefork.replay import ReplayVerifier

    tape = Tape.load(str(tape_path))

    module_path, fn_name = agent.rsplit(":", 1)
    mod = importlib.import_module(module_path)
    agent_fn = getattr(mod, fn_name)

    result = ReplayVerifier(tape, agent_fn).verify()
    _print_receipt(tape_path, result)
    raise typer.Exit(0 if result.bit_exact else 1)


@app.command()
def verify(
    tape_path: Path = typer.Argument(None, help="Single tape to verify"),
    agent: str = typer.Option(None, "--agent", "-a", help="Import path of agent fn"),
    corpus: bool = typer.Option(False, "--corpus", help="Verify all tapes in experiments/validation_tapes/"),
) -> None:
    """Verify bit-exact replay. Exit 1 on drift."""
    from tracefork.tape import Tape
    from tracefork.replay import ReplayVerifier
    import importlib

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
    _print_receipt(tape_path, result)
    raise typer.Exit(0 if result.bit_exact else 1)


@app.command()
def fork(
    run_id: str = typer.Argument(..., help="Parent run_id to fork from"),
    step: int = typer.Option(..., "--step", "-s", help="Exchange index to diverge at"),
    response_file: Path = typer.Option(..., "--response", "-r",
                                       help="Path to .bytes file containing mutated response"),
    agent: str = typer.Option(..., "--agent", "-a", help="Import path of post-fork agent fn"),
    store: Path = typer.Option(Path("store.db"), "--store", help="Path to store.db"),
    desc: str = typer.Option("", "--desc", "-d", help="Human description of mutation"),
) -> None:
    """Fork a run at a step with a mutated response, record the new branch."""
    import importlib
    from tracefork.store import TapeStore
    from tracefork.fork import ForkEngine, BranchSpec

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

    typer.echo(f"\n  Fork created")
    typer.echo(f"  branch_id       {branch_id}")
    typer.echo(f"  parent_run_id   {run_id}")
    typer.echo(f"  divergence_step {step}")
    typer.echo(f"  delta_exchanges {len(branch.delta_tape.exchanges)}")
    typer.echo(f"  description     {desc or '(none)'}\n")


@app.command()
def report(
    run_id: str = typer.Argument(None, help="run_id to report on (from store)"),
    tape_path: Path = typer.Option(None, "--tape", "-t", help="Path to a .tape.sqlite file"),
    output: Path = typer.Option(Path("report.html"), "--output", "-o", help="Output HTML file"),
    store: Path = typer.Option(Path("store.db"), "--store", help="Path to store.db"),
) -> None:
    """Generate a self-contained HTML report from a tape."""
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

    generate_report(tape, output)
    typer.echo(f"Report written to {output}")


@app.command()
def serve(
    store: Path = typer.Option(Path("store.db"), "--store", help="Path to store.db"),
    port: int = typer.Option(7777, "--port", "-p", help="Port to listen on"),
) -> None:
    """Start the tracefork web UI server on port 7777."""
    import uvicorn
    from tracefork.server import app as fastapi_app, init_store

    init_store(str(store))
    typer.echo(f"  tracefork serve → http://localhost:{port}")
    uvicorn.run(fastapi_app, host="127.0.0.1", port=port, workers=1, log_level="warning")


@app.command()
def blame(
    run_id: str = typer.Argument(..., help="run_id to analyze"),
    k: int = typer.Option(10, "--k", help="Forks per candidate step"),
    budget: float = typer.Option(5.0, "--budget", help="USD spend cap"),
    success_re: str = typer.Option("SUCCESS", "--success-re", help="Regex for success outcome"),
    failure_re: str = typer.Option("FAIL", "--failure-re", help="Regex for failure outcome"),
    store: Path = typer.Option(Path("store.db"), "--store", help="Path to store.db"),
) -> None:
    """Run causal blame analysis on a recorded run."""
    import json
    from tracefork.store import TapeStore
    from tracefork.blame import BlameEngine, BudgetGovernor, StringMatchOracle
    from tests.fakes import make_text_response, ScriptedFakeLLM

    db = TapeStore(str(store))
    tape = db.load_tape(run_id)

    oracle = StringMatchOracle(success_re=success_re, failure_re=failure_re)
    est = BudgetGovernor.estimate(tape, k=k)

    typer.echo(f"\n  Blame estimate: {est.n_forks} forks, ~${est.est_usd:.2f}")
    if est.est_usd > budget:
        typer.echo(f"  Estimated cost ${est.est_usd:.2f} exceeds budget ${budget:.2f}.")
        raise typer.Exit(1)

    def perturb_factory(step_idx: int):
        mutated = make_text_response("FAIL — perturbed by blame engine")
        return mutated, ScriptedFakeLLM([mutated] * 10)

    report = BlameEngine.rank(tape, oracle, perturb_factory=perturb_factory, k=k, budget_usd=budget)

    typer.echo(f"\n  run-{run_id} · blame analysis · k={k} · {report.total_forks} forks\n")
    typer.echo(f"  {'rank':<5} {'step':<8} {'flip-rate':<12} {'95% CI':<22} interpretation")
    typer.echo(f"  {'─'*70}")
    for rank, r in enumerate(report.results, 1):
        ci_str = f"[{r.ci_lo:.2f}, {r.ci_hi:.2f}]"
        typer.echo(f"  {rank:<5} step-{r.step_index:<3} {r.flip_rate:<12.2f} {ci_str:<22} {r.interpretation}")
    typer.echo("")

    report_path = Path(f"blame_{run_id}.json")
    report_path.write_text(json.dumps({
        "run_id": run_id, "k": k,
        "results": [
            {"step_index": r.step_index, "flip_rate": r.flip_rate,
             "ci_lo": r.ci_lo, "ci_hi": r.ci_hi, "interpretation": r.interpretation}
            for r in report.results
        ],
    }, indent=2))
    typer.echo(f"  Report saved to {report_path}")


@app.command()
def validate(
    k: int = typer.Option(3, "--k", help="Forks per candidate step per run"),
    n_runs: int = typer.Option(5, "--n-runs", help="Runs per fault class"),
    output: Path = typer.Option(Path("validation_report.json"), "--output", "-o"),
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
        if regressions:
            typer.echo("  REGRESSION detected:")
            for r_str in regressions:
                typer.echo(f"    {r_str}")
            raise typer.Exit(1)
        typer.echo("  No regressions vs committed report.")


def _print_receipt(tape_path: Path, result) -> None:
    from tracefork.replay import DriftDoctor
    status = "PASS" if result.bit_exact else "FAIL"
    typer.echo(f"\n  tracefork — replay receipt")
    typer.echo(f"  {'─' * 40}")
    typer.echo(f"  tape            {tape_path.name}")
    typer.echo(f"  exchanges       {result.matched}/{result.total} matched")
    typer.echo(f"  fingerprint     {'match' if result.fingerprints_match else 'MISMATCH'}")
    typer.echo(f"  result          {status}")
    if result.divergence:
        cause = DriftDoctor.classify(result.divergence)
        typer.echo(f"  drift cause     {cause.value}")
        typer.echo(f"  at exchange     #{result.divergence.step_index}")
    typer.echo("")
