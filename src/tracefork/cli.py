"""tracefork CLI — entry point for all commands.

    tracefork <command> [args]

Commands: replay, verify, fork, report, serve, blame, validate.
"""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(name="tracefork", help="Time-travel debugger for AI agents.")


@app.command()
def replay(
    tape_path: Path = typer.Argument(..., help="Path to a .tape.sqlite file"),  # noqa: B008
    agent: str = typer.Option(..., "--agent", "-a", help="Import path of agent fn (pkg.mod:fn)"),
) -> None:
    """Replay a tape and print the verification receipt."""
    import importlib

    from tracefork.replay import ReplayVerifier
    from tracefork.tape import Tape

    tape = Tape.load(str(tape_path))

    module_path, fn_name = agent.rsplit(":", 1)
    mod = importlib.import_module(module_path)
    agent_fn = getattr(mod, fn_name)

    result = ReplayVerifier(tape, agent_fn).verify()
    _print_receipt(tape_path, result)
    raise typer.Exit(0 if result.bit_exact else 1)


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
    store: Path = typer.Option(Path("store.db"), "--store", help="Path to store.db"),  # noqa: B008
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
        Path("store.db"), "--store", help="Path to store.db"
    ),
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
    store: Path = typer.Option(  # noqa: B008
        Path("store.db"), "--store", help="Path to store.db"
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
    budget: float = typer.Option(5.0, "--budget", help="USD spend cap"),
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
        Path("store.db"), "--store", help="Path to store.db"
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


def _print_receipt(tape_path: Path, result) -> None:
    from tracefork.replay import DriftDoctor

    status = "PASS" if result.bit_exact else "FAIL"
    typer.echo("\n  tracefork — replay receipt")
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
