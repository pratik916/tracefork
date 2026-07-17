"""tracefork CLI — entry point for all commands.

    tracefork <command> [args]

Commands: replay, verify, fork, coalition-fork, diff, converge, conflicts,
settlement-diff, report, receipt, release-receipt, serve, blame, tournament,
validate, bench, export, ingest, prune, proxy, coverage, corpus-blame, locate,
query, bundle-export, bundle-import, plus the `branch` sub-app
(descendants/ancestors/siblings) and the `session` sub-app (create/spawn/show/
cost/divergence/record/replay/fork/blame/cross-blame/chaos/serve).
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

    from tracefork.basis import basis_from_provenance, current_basis, format_basis_drift_warning
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

    recorded_basis = basis_from_provenance(tape.provenance)
    if recorded_basis is not None:
        warning = format_basis_drift_warning(recorded_basis, current_basis())
        if warning is not None:
            typer.echo(warning)

    module_path, fn_name = agent.rsplit(":", 1)
    mod = importlib.import_module(module_path)
    agent_fn = getattr(mod, fn_name)

    result = ReplayVerifier(tape, agent_fn).verify()
    result.certificate = certificate_from_verification(result, tape)
    _print_receipt(tape_path, result, tape)
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
        False,
        "--corpus",
        help="Gate a committed fixture corpus (replay-as-regression); see --corpus-dir",
    ),
    corpus_dir: Path = typer.Option(  # noqa: B008
        Path("experiments/replay_fixtures"),
        "--corpus-dir",
        help="Fixture corpus dir for --corpus (default: experiments/replay_fixtures, "
        "the same corpus 'replay --check' and CI already use)",
    ),
    store: Path = typer.Option(  # noqa: B008
        None,
        "--store",
        help="Path to a store.db: run a read-only structural fsck (every tape/"
        "branch decodes, every branch's parent resolves) instead of replay-"
        "verifying a single tape. Mutually exclusive with --corpus.",
    ),
) -> None:
    """Verify bit-exact replay (single tape or --corpus), or run a read-only
    structural fsck over a store with --store <db path>. Exit 1 on drift, on
    any fsck row failure, on any --corpus fixture failure, or if both
    --corpus and --store are passed."""
    import importlib

    from tracefork.certificate import certificate_from_verification
    from tracefork.replay import ReplayVerifier
    from tracefork.tape import Tape

    if corpus and store is not None:
        typer.echo("Pass at most one of --corpus or --store")
        raise typer.Exit(1)

    if store is not None:
        _run_store_fsck(store)
        return

    if corpus:
        _run_replay_check(corpus_dir)
        return

    if tape_path is None or agent is None:
        typer.echo("Provide --agent and a tape path, or use --corpus/--store")
        raise typer.Exit(1)

    tape = Tape.load(str(tape_path))
    module_path, fn_name = agent.rsplit(":", 1)
    mod = importlib.import_module(module_path)
    agent_fn = getattr(mod, fn_name)
    result = ReplayVerifier(tape, agent_fn).verify()
    result.certificate = certificate_from_verification(result, tape)
    _print_receipt(tape_path, result, tape)
    raise typer.Exit(0 if result.bit_exact else 1)


def _run_store_fsck(store_path: Path) -> None:
    """`verify --store` body: a read-only structural fsck over a `TapeStore`
    database (see `fsck.store_fsck`). Exits 1 if any tape/branch fails to
    decode or a branch's parent can't resolve (orphaned parent); never
    mutates the store."""
    from tracefork.fsck import store_fsck
    from tracefork.store import TapeStore

    if not store_path.exists():
        typer.echo(f"No store found at {store_path}")
        raise typer.Exit(1)

    db = TapeStore(str(store_path))
    try:
        result = store_fsck(db)
    finally:
        db.close()

    typer.echo(f"\n  tracefork verify --store {store_path}")
    typer.echo(f"  {'─' * 60}")
    if not result.rows:
        typer.echo("  (store is empty — nothing to check)")
    for row in result.rows:
        status = "PASS" if row.passed else "FAIL"
        typer.echo(f"  [{status}] {row.kind:<7} {row.id:<14} {row.reason}")
    n_pass = sum(1 for r in result.rows if r.passed)
    typer.echo(f"\n  {n_pass}/{len(result.rows)} row(s) passed\n")
    raise typer.Exit(0 if result.all_ok else 1)


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
    writable_root: list[str] = typer.Option(  # noqa: B008
        [],
        "--writable-root",
        help="Directory a ConfinementSpec permits writes under (repeatable); combined "
        "with --allowed-host to force BoundaryGuard confinement for the fork's "
        "re-executed agent (see boundary_guard.py). Omit both for today's unconfined "
        "default.",
    ),
    allowed_host: list[str] = typer.Option(  # noqa: B008
        [],
        "--allowed-host",
        help="Hostname a ConfinementSpec permits socket.connect to (repeatable)",
    ),
) -> None:
    """Fork a run at a step with a mutated response, record the new branch."""
    import importlib

    from tracefork.basis import basis_from_provenance, current_basis, format_basis_drift_warning
    from tracefork.boundary_guard import ConfinementSpec, ConfinementViolationError
    from tracefork.confinement_diagnostics import diagnose_confinement
    from tracefork.fork import BranchSpec, ForkEngine
    from tracefork.store import TapeStore

    db = TapeStore(str(store))
    parent_tape = db.load_tape(run_id)

    recorded_basis = basis_from_provenance(parent_tape.provenance)
    if recorded_basis is not None:
        warning = format_basis_drift_warning(recorded_basis, current_basis())
        if warning is not None:
            typer.echo(warning)

    mutated_response = response_file.read_bytes()

    module_path, fn_name = agent.rsplit(":", 1)
    mod = importlib.import_module(module_path)
    agent_fn = getattr(mod, fn_name)

    spec = BranchSpec(
        divergence_step=step,
        mutated_response=mutated_response,
        mutation_desc=desc,
    )

    confinement = (
        ConfinementSpec(writable_roots=tuple(writable_root), allowed_hosts=tuple(allowed_host))
        if writable_root or allowed_host
        else None
    )

    try:
        branch = ForkEngine.fork(parent_tape, spec, agent_fn, confinement=confinement)
    except ConfinementViolationError as exc:
        diag = diagnose_confinement(exc)
        typer.echo("\n  Confinement violation")
        typer.echo(f"  {'─' * 40}")
        typer.echo(f"  kind            {diag.violation_kind}")
        typer.echo(f"  attempted       {diag.attempted}")
        if diag.declared_writable_roots is not None:
            typer.echo(f"  writable_roots  {list(diag.declared_writable_roots)}")
        if diag.declared_allowed_hosts is not None:
            typer.echo(f"  allowed_hosts   {list(diag.declared_allowed_hosts)}")
        typer.echo(f"  message         {diag.message}")
        typer.echo("")
        raise typer.Exit(1) from exc

    branch_id = db.save_branch(
        parent_run_id=run_id,
        divergence_step=step,
        delta_tape=branch.delta_tape,
        mutation_desc=desc,
        branch_digest=branch.branch_digest,
        confinement_tier=branch.confinement_tier,
    )

    typer.echo("\n  Fork created")
    typer.echo(f"  branch_id       {branch_id}")
    typer.echo(f"  parent_run_id   {run_id}")
    typer.echo(f"  divergence_step {step}")
    typer.echo(f"  delta_exchanges {len(branch.delta_tape.exchanges)}")
    typer.echo(f"  description     {desc or '(none)'}\n")


@app.command()
def coalition_fork(
    run_id: str = typer.Argument(..., help="Parent run_id to coalition-fork from"),
    intervene: list[str] = typer.Option(  # noqa: B008
        ...,
        "--intervene",
        help="A 'step:response_file' intervention, repeatable (>=1 required) -- "
        "every step is forced to its response_file's bytes JOINTLY with every "
        "other --intervene (the coalition/Shapley do(S) primitive)",
    ),
    agent: str = typer.Option(..., "--agent", "-a", help="Import path of post-fork agent fn"),
    store: Path = typer.Option(  # noqa: B008
        Path(_DEFAULT_CONFIG.db_path), "--store", help="Path to store.db"
    ),
    desc: str = typer.Option("", "--desc", "-d", help="Human description of the coalition"),
    writable_root: list[str] = typer.Option(  # noqa: B008
        [],
        "--writable-root",
        help="Directory a ConfinementSpec permits writes under (repeatable); combined "
        "with --allowed-host to force BoundaryGuard confinement for the fork's "
        "re-executed agent (see boundary_guard.py). Omit both for today's unconfined "
        "default.",
    ),
    allowed_host: list[str] = typer.Option(  # noqa: B008
        [],
        "--allowed-host",
        help="Hostname a ConfinementSpec permits socket.connect to (repeatable)",
    ),
) -> None:
    """Fork a run at a SET of steps forced jointly, record the new branch.

    Generalizes `fork` (one divergence step) to a public what-if DSL: each
    `--intervene step:response_file` pins one intervention locus; every
    locus is resampled under the same forced-response policy in a single
    joint pass. Only the coalition's first (lowest-index) intervention is
    request-matched against the parent tape -- the genuine point of first
    divergence; every later intervention is forced unconditionally, since
    the agent's requests have already diverged by then. This pinned-locus +
    same-policy-resampling discipline is what a coalition/Shapley blame
    computation needs from its intervention primitive -- distinct from a
    naive "fork anywhere and diff", where the intervention point and the
    resampling policy are both ad hoc.
    """
    import importlib
    import json

    from tracefork.basis import basis_from_provenance, current_basis, format_basis_drift_warning
    from tracefork.boundary_guard import ConfinementSpec, ConfinementViolationError
    from tracefork.confinement_diagnostics import diagnose_confinement
    from tracefork.fork import CoalitionSpec, ForkEngine, StepIntervention
    from tracefork.store import TapeStore

    interventions = []
    for spec in intervene:
        step_str, sep, response_path = spec.partition(":")
        if not sep or not step_str or not response_path:
            raise typer.BadParameter(f"--intervene must be 'step:response_file', got {spec!r}")
        try:
            step = int(step_str)
        except ValueError as exc:
            raise typer.BadParameter(
                f"--intervene step must be an integer, got {step_str!r}"
            ) from exc
        response_file = Path(response_path)
        if not response_file.is_file():
            raise typer.BadParameter(f"--intervene response file not found: {response_path!r}")
        interventions.append(
            StepIntervention(step=step, mutated_response=response_file.read_bytes())
        )

    try:
        spec_obj = CoalitionSpec(interventions=tuple(interventions), mutation_desc=desc)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    db = TapeStore(str(store))
    parent_tape = db.load_tape(run_id)

    recorded_basis = basis_from_provenance(parent_tape.provenance)
    if recorded_basis is not None:
        warning = format_basis_drift_warning(recorded_basis, current_basis())
        if warning is not None:
            typer.echo(warning)

    module_path, fn_name = agent.rsplit(":", 1)
    mod = importlib.import_module(module_path)
    agent_fn = getattr(mod, fn_name)

    confinement = (
        ConfinementSpec(writable_roots=tuple(writable_root), allowed_hosts=tuple(allowed_host))
        if writable_root or allowed_host
        else None
    )

    try:
        branch = ForkEngine.fork_coalition(parent_tape, spec_obj, agent_fn, confinement=confinement)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    except ConfinementViolationError as exc:
        diag = diagnose_confinement(exc)
        typer.echo("\n  Confinement violation")
        typer.echo(f"  {'─' * 40}")
        typer.echo(f"  kind            {diag.violation_kind}")
        typer.echo(f"  attempted       {diag.attempted}")
        if diag.declared_writable_roots is not None:
            typer.echo(f"  writable_roots  {list(diag.declared_writable_roots)}")
        if diag.declared_allowed_hosts is not None:
            typer.echo(f"  allowed_hosts   {list(diag.declared_allowed_hosts)}")
        typer.echo(f"  message         {diag.message}")
        typer.echo("")
        raise typer.Exit(1) from exc

    # Coalition step list + description round-trip through the existing
    # free-text `mutation_desc` column (no store.py schema change) -- see
    # `store.py`'s `save_branch`/`load_branch`.
    mutation_desc_json = json.dumps(
        {"coalition_steps": list(branch.intervened_steps), "desc": desc}
    )

    branch_id = db.save_branch(
        parent_run_id=run_id,
        divergence_step=branch.divergence_step,
        delta_tape=branch.delta_tape,
        mutation_desc=mutation_desc_json,
        branch_digest=branch.branch_digest,
        confinement_tier=branch.confinement_tier,
    )

    typer.echo("\n  Coalition fork created")
    typer.echo(f"  branch_id        {branch_id}")
    typer.echo(f"  parent_run_id    {run_id}")
    typer.echo(f"  intervened_steps {list(branch.intervened_steps)}")
    typer.echo(f"  delta_exchanges  {len(branch.delta_tape.exchanges)}")
    typer.echo(f"  description      {desc or '(none)'}\n")


@app.command()
def diff(
    id_a: str = typer.Argument(..., help="parent run_id (default mode), or run_id_a (with --step)"),
    id_b: str = typer.Argument(..., help="branch_id (default mode), or run_id_b (with --step)"),
    step: int = typer.Option(
        None,
        "--step",
        help="Compare id_a/id_b as two independent tapes at this single step "
        "index, instead of the default parent-run-vs-branch mode",
    ),
    store: Path = typer.Option(  # noqa: B008
        Path(_DEFAULT_CONFIG.db_path), "--store", help="Path to store.db"
    ),
) -> None:
    """Structural diff: a branch against its parent (default), or two tapes at
    one step (--step). Reuses `divergence.py`'s structural-diff primitive,
    walked over a range of steps -- see `diff.py`."""
    from tracefork.diff import branch_diff, tape_diff
    from tracefork.store import TapeStore

    db = TapeStore(str(store))
    try:
        if step is not None:
            tape_a = db.load_tape(id_a)
            tape_b = db.load_tape(id_b)
            step_diffs = (tape_diff(tape_a, tape_b, step),)
            heading = f"tracefork diff {id_a} {id_b} --step {step}"
        else:
            parent_tape = db.load_tape(id_a)
            branch_row = db.load_branch(id_b)
            range_diff = branch_diff(
                parent_tape,
                branch_row["delta_tape"],
                divergence_step=branch_row["divergence_step"],
            )
            step_diffs = range_diff.steps
            heading = f"tracefork diff {id_a} {id_b}"
    finally:
        db.close()

    _print_diff_receipt(heading, step_diffs)
    n_changed = sum(1 for s in step_diffs if s.changed)
    raise typer.Exit(0 if n_changed == 0 else 1)


def _print_diff_receipt(heading: str, step_diffs) -> None:
    """Operator-facing receipt for `diff` — one line per step, PASS when
    unchanged, FAIL (with the field-diff count) otherwise. Mirrors
    `_run_replay_check`'s PASS/FAIL-per-row style."""
    typer.echo(f"\n  {heading}")
    typer.echo(f"  {'─' * 60}")
    for s in step_diffs:
        if s.changed:
            n = len(s.request_diffs) + len(s.response_diffs)
            typer.echo(f"  [FAIL] step {s.step_index:<4} {n} field(s) differ")
            for d in s.request_diffs:
                typer.echo(f"           request  {d.path}: {d.recorded!r} -> {d.live!r}")
            for d in s.response_diffs:
                typer.echo(f"           response {d.path}: {d.recorded!r} -> {d.live!r}")
        else:
            typer.echo(f"  [PASS] step {s.step_index:<4} identical")
    n_changed = sum(1 for s in step_diffs if s.changed)
    if n_changed == 0:
        typer.echo(f"\n  {len(step_diffs)}/{len(step_diffs)} step(s) identical\n")
    else:
        typer.echo(f"\n  {n_changed}/{len(step_diffs)} step(s) changed\n")


@app.command()
def converge(
    branch_id_a: str = typer.Argument(
        ..., help="First branch_id (must share divergence_step with branch_id_b)"
    ),
    branch_id_b: str = typer.Argument(..., help="Second branch_id"),
    store: Path = typer.Option(  # noqa: B008
        Path(_DEFAULT_CONFIG.db_path), "--store", help="Path to store.db"
    ),
) -> None:
    """Reconvergence check: did two same-divergence-step sibling branches
    (e.g. two of `blame`'s k trials, or two `tournament` variants) end up
    producing byte-identical continuations again? Reuses `fork.py`'s
    per-exchange fingerprint -- see `convergence.py`."""
    from tracefork.convergence import find_reconvergence
    from tracefork.store import TapeStore

    db = TapeStore(str(store))
    try:
        branch_a = db.load_branch(branch_id_a)
        branch_b = db.load_branch(branch_id_b)
        result = find_reconvergence(
            branch_a["delta_tape"],
            branch_a["divergence_step"],
            branch_b["delta_tape"],
            branch_b["divergence_step"],
        )
        heading = f"tracefork converge {branch_id_a} {branch_id_b}"
    finally:
        db.close()

    _print_convergence_receipt(heading, result)
    raise typer.Exit(0 if result.stable else 1)


def _print_convergence_receipt(heading: str, result) -> None:
    """Operator-facing receipt for `converge` -- one line per step, MATCH
    when both sides fingerprint identically, DIVERGED otherwise. Mirrors
    `_print_diff_receipt`'s PASS/FAIL-per-row style."""
    typer.echo(f"\n  {heading}")
    typer.echo(f"  {'─' * 60}")
    for s in result.steps:
        if s.matched:
            typer.echo(f"  [MATCH]    step {s.step_index:<4} {s.fingerprint_a[:16]}")
        else:
            typer.echo(
                f"  [DIVERGED] step {s.step_index:<4} "
                f"{s.fingerprint_a[:16]} != {s.fingerprint_b[:16]}"
            )
    if result.stable:
        typer.echo(f"\n  stable reconvergence from step {result.first_convergent_step} onward\n")
    elif result.reconverged:
        typer.echo(f"\n  coincidental match at step(s) {result.matched_steps} -- not stable\n")
    else:
        typer.echo("\n  never reconverged\n")


@app.command()
def conflicts(
    parent_run_id: str = typer.Argument(..., help="parent run_id both branches forked from"),
    branch_id_a: str = typer.Argument(..., help="first branch_id"),
    branch_id_b: str = typer.Argument(..., help="second branch_id"),
    store: Path = typer.Option(  # noqa: B008
        Path(_DEFAULT_CONFIG.db_path), "--store", help="Path to store.db"
    ),
    output: Path = typer.Option(  # noqa: B008
        None, "--output", "-o", help="Optional JSON output path"
    ),
) -> None:
    """Reviewer-sanity check: do two branches' post-divergence tool calls touch
    the same (tool_name, resource)? Loads both branches' `delta_tape`s via
    `TapeStore.load_branch` (mirrors `diff`'s branch-mode loading) and reports
    every overlap found by `effects.diff_effects` -- read-only, no merge/apply
    logic. Exits 1 iff a conflict is found, else 0.
    """
    from tracefork.effects import diff_effects
    from tracefork.store import TapeStore

    db = TapeStore(str(store))
    try:
        branch_a = db.load_branch(branch_id_a)
        branch_b = db.load_branch(branch_id_b)
    finally:
        db.close()

    report = diff_effects(branch_a["delta_tape"], branch_b["delta_tape"])

    heading = f"tracefork conflicts {parent_run_id} {branch_id_a} {branch_id_b}"
    typer.echo(f"\n  {heading}")
    typer.echo(f"  {'─' * 60}")
    typer.echo(f"  branch {branch_id_a}: {len(report.effects_a)} tool effect(s)")
    typer.echo(f"  branch {branch_id_b}: {len(report.effects_b)} tool effect(s)")
    if report.overlaps:
        typer.echo(f"\n  [FAIL] {len(report.overlaps)} overlapping tool effect(s):")
        for o in report.overlaps:
            typer.echo(f"    {o.tool_name}({o.resource!r})")
    else:
        typer.echo("\n  [PASS] no overlapping tool effects")
    typer.echo("")

    if output is not None:
        import json as _json

        def _effect_dict(e):
            return {
                "source": e.source,
                "index": e.index,
                "tool_name": e.tool_name,
                "resource": e.resource,
                "resource_is_fallback": e.resource_is_fallback,
            }

        output.write_text(
            _json.dumps(
                {
                    "parent_run_id": parent_run_id,
                    "branch_id_a": branch_id_a,
                    "branch_id_b": branch_id_b,
                    "effects_a": [_effect_dict(e) for e in report.effects_a],
                    "effects_b": [_effect_dict(e) for e in report.effects_b],
                    "overlaps": [
                        {
                            "tool_name": o.tool_name,
                            "resource": o.resource,
                            "effect_a": _effect_dict(o.effect_a),
                            "effect_b": _effect_dict(o.effect_b),
                        }
                        for o in report.overlaps
                    ],
                    "has_conflict": report.has_conflict,
                },
                indent=2,
            )
        )
        typer.echo(f"  Report saved to {output}\n")

    raise typer.Exit(1 if report.has_conflict else 0)


@app.command(name="settlement-diff")
def settlement_diff(
    run_id: str = typer.Argument(..., help="parent run_id"),
    branch_id: str = typer.Argument(..., help="branch_id to export tool-call settlement ops for"),
    store: Path = typer.Option(  # noqa: B008
        Path(_DEFAULT_CONFIG.db_path), "--store", help="Path to store.db"
    ),
    output: Path = typer.Option(  # noqa: B008
        None, "--output", "-o", help="Optional JSON output path (prints to stdout if omitted)"
    ),
) -> None:
    """Export a winning fork's post-divergence tool-call side effects as a
    portable, framework-agnostic settlement-diff artifact (tracefork-bge.69).

    Loads the parent tape + branch via `TapeStore.load_tape`/`load_branch`
    (mirrors `diff`'s and `conflicts`' branch-mode loading), decodes
    `delta_tape.tool_exchanges` via `settlement.branch_settlement_diff`, and
    writes/prints `settlement.to_settlement_json`'s in-toto-Statement-shaped
    output. Read-only export -- TraceFork never applies/settles anything
    itself; this is for an external apply/settlement layer to consume.
    Always exits 0 (like `bundle-export`/`receipt`), not a pass/fail gate
    like `diff`/`conflicts`.
    """
    import json as _json

    from tracefork.settlement import branch_settlement_diff, to_settlement_json
    from tracefork.store import TapeStore

    db = TapeStore(str(store))
    try:
        parent_tape = db.load_tape(run_id)
        branch_row = db.load_branch(branch_id)
    finally:
        db.close()

    diff = branch_settlement_diff(
        parent_tape,
        branch_row["delta_tape"],
        divergence_step=branch_row["divergence_step"],
        branch_digest=branch_row["branch_digest"],
    )
    payload = to_settlement_json(diff)
    text = _json.dumps(payload, indent=2)

    heading = f"tracefork settlement-diff {run_id} {branch_id}"
    typer.echo(f"\n  {heading}")
    typer.echo(f"  {'─' * 60}")
    typer.echo(f"  ops  {len(diff.ops)}")

    if output is not None:
        output.write_text(text)
        typer.echo(f"  Settlement diff written to {output}\n")
    else:
        typer.echo("")
        typer.echo(text)

    raise typer.Exit(0)


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
    """Generate a self-contained HTML report from a tape.

    When loaded via `run_id` (from `store`), the run's saved branches are
    looked up and embedded as the report's fork-tree panel data
    (tracefork-bge.15). The `--tape` path has no store to look branches up
    in — an honest, documented scope limit: those reports render an empty
    fork tree rather than a silently-populated one.
    """
    import json as _json

    from tracefork.report import generate_report
    from tracefork.tape import Tape

    branches: list[dict] | None = None
    if tape_path:
        tape = Tape.load(str(tape_path))
    elif run_id:
        from tracefork.store import TapeStore

        db = TapeStore(str(store))
        tape = db.load_tape(run_id)
        branches = db.list_branches(run_id)
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

    generate_report(tape, output, blame=blame_dict, replay=replay_data, branches=branches)
    typer.echo(f"Report written to {output}")
    _print_trust_lines(tape)


@app.command()
def receipt(
    run_id: str = typer.Argument(None, help="run_id to build a trust receipt for (from store)"),
    tape_path: Path = typer.Option(  # noqa: B008
        None, "--tape", "-t", help="Path to a .tape.sqlite file"
    ),
    store: Path = typer.Option(  # noqa: B008
        Path(_DEFAULT_CONFIG.db_path), "--store", help="Path to store.db"
    ),
    agent: str = typer.Option(
        None,
        "--agent",
        "-a",
        help="Import path of the agent fn (pkg.mod:fn) that produced this tape "
        "(pkg.mod:fn); re-replayed ($0) to embed replay evidence in the receipt",
    ),
    validation_report: Path = typer.Option(  # noqa: B008
        Path("validation_report.json"),
        "--validation-report",
        help="Path to a validation_report.json (from `tracefork validate`); "
        "embedded if it exists, an explicit absent marker otherwise",
    ),
    bench_report_path: Path = typer.Option(  # noqa: B008
        Path("bench_report.json"),
        "--bench-report",
        help="Path to a bench_report.json (from `tracefork bench`); embedded "
        "if it exists, an explicit absent marker otherwise",
    ),
    output: Path = typer.Option(  # noqa: B008
        Path("receipt.json"), "--output", "-o", help="Output receipt JSON file"
    ),
    shield_output: Path = typer.Option(  # noqa: B008
        None,
        "--shield-output",
        help="Optional output path for a Shields.io endpoint-badge JSON derived from the receipt",
    ),
) -> None:
    """Build a shareable, JSON-safe trust receipt for a tape (tracefork-bge.26).

    Composes already-computed evidence — a fresh ($0) replay via --agent,
    plus `validation_report.json`/`bench_report.json` off disk if present —
    into one attestation-shaped JSON document (see `tracefork.receipt`).
    Missing evidence is always an explicit `{"available": false}` marker,
    never silently omitted or defaulted to a verified state. Pass
    `--shield-output` to also write a Shields.io-style endpoint-badge JSON
    (green only when replay is bit-exact AND validate clears the precision
    bar; a content-redacted tape never badges green).
    """
    import json as _json

    from tracefork.receipt import build_shield_json, build_trust_receipt
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

    replay_result = None
    if agent:
        import importlib

        from tracefork.replay import ReplayVerifier

        module_path, fn_name = agent.rsplit(":", 1)
        agent_fn = getattr(importlib.import_module(module_path), fn_name)
        replay_result = ReplayVerifier(tape, agent_fn).verify()

    validate_data = None
    if validation_report.exists():
        validate_data = _json.loads(validation_report.read_text())

    bench_data = None
    if bench_report_path.exists():
        bench_data = _json.loads(bench_report_path.read_text())

    receipt_dict = build_trust_receipt(
        tape,
        replay=replay_result,
        validate_report=validate_data,
        bench_report=bench_data,
    )
    output.write_text(_json.dumps(receipt_dict, indent=2))
    typer.echo(f"  Receipt written to {output}")
    typer.echo(f"  tape_fingerprint  {receipt_dict['tape_fingerprint']}")

    if shield_output is not None:
        shield_dict = build_shield_json(receipt_dict)
        shield_output.write_text(_json.dumps(shield_dict, indent=2))
        typer.echo(f"  Shield badge written to {shield_output}")


@app.command(name="release-receipt")
def release_receipt(
    version: str = typer.Argument(..., help="Release version this receipt is for (e.g. v0.3.0)"),
    junit_xml: Path = typer.Option(  # noqa: B008
        Path("junit.xml"),
        "--junit-xml",
        help="Path to a JUnit XML report (from `pytest --junit-xml`); embedded if it "
        "exists, an explicit absent marker otherwise",
    ),
    coverage_json: Path = typer.Option(  # noqa: B008
        Path("coverage.json"),
        "--coverage-json",
        help="Path to a `coverage json` report; embedded if it exists, an explicit "
        "absent marker otherwise",
    ),
    validation_report: Path = typer.Option(  # noqa: B008
        Path("validation_report.json"),
        "--validation-report",
        help="Path to a validation_report.json (from `tracefork validate`); embedded "
        "if it exists, an explicit absent marker otherwise",
    ),
    bench_report_path: Path = typer.Option(  # noqa: B008
        Path("bench_report.json"),
        "--bench-report",
        help="Path to a bench_report.json (from `tracefork bench`); embedded if it "
        "exists, an explicit absent marker otherwise",
    ),
    replay_fixtures_dir: Path = typer.Option(  # noqa: B008
        Path("experiments/replay_fixtures"),
        "--replay-fixtures",
        help="Committed tape-fixture corpus directory to freshly replay-check ($0, deterministic)",
    ),
    output_dir: Path = typer.Option(  # noqa: B008
        Path("docs/release_receipts"),
        "--output-dir",
        help="Directory the signed receipt JSON is written to, as <version>.json",
    ),
) -> None:
    """Compose+sign a per-release trust receipt (tracefork-bge.50).

    Reads junit.xml/coverage.json/validation_report.json/bench_report.json off
    disk if present (an explicit absent marker otherwise), runs a fresh ($0,
    deterministic) replay-fixture-corpus check and CI-calibration sweep, and
    composes+signs everything into one content-addressed release receipt
    written to <output-dir>/<version>.json (committed, unlike the other
    gitignored runtime JSONs). Signs with HMAC-SHA256 when
    `TRACEFORK_RELEASE_SIGNING_KEY` is set in the environment (an honest
    symmetric attestation, not a DSSE/asymmetric signature); unsigned
    otherwise. Exits 1 if the replay corpus didn't fully pass or calibration
    has coverage regressions, 0 otherwise.
    """
    import json as _json
    import os

    from tracefork.ci_calibration import run_calibration
    from tracefork.release_receipt import (
        build_release_receipt,
        parse_coverage_summary,
        parse_junit_test_summary,
        sign_release_receipt,
    )
    from tracefork.replay import run_fixture_corpus_check

    test_summary = parse_junit_test_summary(junit_xml) if junit_xml.exists() else None
    coverage_summary = parse_coverage_summary(coverage_json) if coverage_json.exists() else None

    validate_data = None
    if validation_report.exists():
        validate_data = _json.loads(validation_report.read_text())

    bench_data = None
    if bench_report_path.exists():
        bench_data = _json.loads(bench_report_path.read_text())

    replay_corpus_result = run_fixture_corpus_check(replay_fixtures_dir)
    calibration_report = run_calibration()

    receipt = build_release_receipt(
        version=version,
        test_summary=test_summary,
        coverage_summary=coverage_summary,
        validate_report=validate_data,
        bench_report=bench_data,
        replay_corpus=replay_corpus_result,
        calibration=calibration_report,
    )

    signing_key_str = os.environ.get("TRACEFORK_RELEASE_SIGNING_KEY")
    signing_key = signing_key_str.encode("utf-8") if signing_key_str else None
    receipt = sign_release_receipt(receipt, signing_key=signing_key)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{version}.json"
    output_path.write_text(_json.dumps(receipt, indent=2))
    typer.echo(f"  Release receipt written to {output_path}")
    typer.echo(f"  receipt_digest  {receipt['receipt_digest']}")

    if not replay_corpus_result.all_passed:
        typer.echo("  replay corpus check FAILED")
    if not calibration_report.all_within_tolerance():
        typer.echo("  calibration has coverage regressions")

    ok = replay_corpus_result.all_passed and calibration_report.all_within_tolerance()
    raise typer.Exit(0 if ok else 1)


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
    field: str = typer.Option(
        None,
        "--field",
        help="JSON field path ($.a.b[0].c) to grade instead of the whole output "
        "text (scopes success/failure regex matching to one field's value; "
        "see tracefork.field_oracle.FieldDiffOracle)",
    ),
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

    from tracefork import narrative
    from tracefork.blame import BlameEngine, BudgetGovernor, CIMethod, Oracle, StringMatchOracle
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

    if field:
        from tracefork.field_oracle import FieldDiffOracle

        oracle: Oracle = FieldDiffOracle(
            field_path=field, success_re=success_re, failure_re=failure_re
        )
    else:
        oracle = StringMatchOracle(success_re=success_re, failure_re=failure_re)
    est = BudgetGovernor.estimate(tape, k=k)

    typer.echo(f"\n  Blame estimate: {est.n_forks} forks, ~${est.est_usd:.2f}")
    if est.est_usd > budget:
        typer.echo(f"  Estimated cost ${est.est_usd:.2f} exceeds budget ${budget:.2f}.")
        typer.echo("  Use --budget to increase or --k to reduce trials.")
        raise typer.Exit(1)

    risk = BudgetGovernor.confinement_risk(tape, k=k)
    typer.echo(f"  {risk.note}")

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
                "confinement_risk": (
                    {
                        "projected_trials": report.confinement_risk.projected_trials,
                        "confined": report.confinement_risk.confined,
                        "note": report.confinement_risk.note,
                    }
                    if report.confinement_risk is not None
                    else None
                ),
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

    narrative_path = Path(f"blame_{run_id}.md")
    narrative_path.write_text(narrative.explain_blame_report(report))
    typer.echo(f"  Narrative saved to {narrative_path}")

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
def bundle_export(
    run_id: str = typer.Argument(..., help="run_id to export (and its direct branches)"),
    output: Path = typer.Option(  # noqa: B008
        Path("bundle.db"), "--output", "-o", help="Output bundle store.db path"
    ),
    store: Path = typer.Option(  # noqa: B008
        Path(_DEFAULT_CONFIG.db_path), "--store", help="Path to source store.db"
    ),
) -> None:
    """Export a run and its direct branches into a portable, self-contained
    store.db bundle — a scp-able artifact analogous to `git bundle`.

    Raw blob copy, byte-for-byte: no Tape decode/re-encode round trip (unlike
    `export --otel`/`--openinference`, which are lossy observability data,
    not a replayable tape). See `tracefork bundle-import` for the reverse
    direction.
    """
    from tracefork.bundle import export_bundle
    from tracefork.store import TapeStore

    db = TapeStore(str(store))
    try:
        result = export_bundle(db, run_id, str(output))
    except KeyError as exc:
        typer.echo(str(exc))
        raise typer.Exit(1) from exc
    finally:
        db.close()

    typer.echo(f"\n  Bundle written to {output}")
    typer.echo(f"  run_id     {result.run_id}")
    typer.echo(f"  branch(es) {len(result.branch_ids)}\n")


@app.command()
def bundle_import(
    bundle_path: Path = typer.Argument(  # noqa: B008
        ..., help="Path to a bundle store.db (from `tracefork bundle-export`)"
    ),
    store: Path = typer.Option(  # noqa: B008
        Path(_DEFAULT_CONFIG.db_path), "--store", help="Path to destination store.db"
    ),
) -> None:
    """Import every run + its direct branches from a bundle store.db into
    --store, through the CAS-guarded save_tape/save_branch write path (never
    raw INSERT) — a genuine content collision on an existing run_id/branch_id
    raises an error instead of silently overwriting. Reusing the same ids
    with byte-identical content is an idempotent no-op. See `tracefork
    bundle-export` for the reverse direction.
    """
    from tracefork.bundle import import_bundle
    from tracefork.store import TapeConflictError, TapeStore

    if not bundle_path.exists():
        typer.echo(f"No bundle found at {bundle_path}")
        raise typer.Exit(1)

    db = TapeStore(str(store))
    try:
        result = import_bundle(db, str(bundle_path))
    except TapeConflictError as exc:
        typer.echo(str(exc))
        raise typer.Exit(1) from exc
    finally:
        db.close()

    typer.echo(f"\n  Imported {len(result.run_ids)} run(s), {len(result.branch_ids)} branch(es)")
    typer.echo(f"  from {bundle_path} -> {store}\n")


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


@app.command()
def coverage(
    tape_path: Path = typer.Argument(..., help="Path to a .tape.sqlite file"),  # noqa: B008
    agent_source: Path = typer.Option(  # noqa: B008
        None,
        "--agent-source",
        help="Path to the agent's Python source file: a best-effort static "
        "AST scan (never imported or executed) for known-uncapturable "
        "nondeterminism call sites not covered by BoundaryGuard",
    ),
    output: Path = typer.Option(  # noqa: B008
        None, "--output", "-o", help="Optional JSON output path"
    ),
) -> None:
    """Print a determinism-coverage report for a recorded tape.

    Reports which nondeterminism draw kinds occurred, whether concurrency was
    recorded, and whether `BoundaryGuard` was active at record time -- plus,
    with --agent-source, a best-effort static AST scan for known-uncapturable
    nondeterminism calls not covered by the active guard. See
    `coverage.py`'s module docstring for the scan's scope limits.
    """
    from tracefork.coverage import coverage_report
    from tracefork.tape import Tape

    tape = Tape.load(str(tape_path))
    source = agent_source.read_text() if agent_source is not None else None
    result = coverage_report(tape, agent_source=source)

    typer.echo(f"\n  tracefork coverage — {tape_path.name}")
    typer.echo(f"  {'─' * 50}")
    typer.echo(f"  boundary_guard_active   {result.boundary_guard_active}")
    typer.echo(f"  concurrency_recorded    {result.concurrency_recorded}")
    typer.echo("  draw counts:")
    if result.draw_counts:
        for kind, count in sorted(result.draw_counts.items()):
            typer.echo(f"    {kind:<10} {count}")
    else:
        typer.echo("    (no draws recorded)")

    if agent_source is not None:
        typer.echo(f"\n  AST scan of {agent_source.name} (best-effort lint):")
        if result.findings:
            for f in result.findings:
                tag = "GUARDABLE" if f.guardable else "informational"
                typer.echo(f"    line {f.lineno:<4} [{tag:<13}] {f.call} -- {f.note}")
        else:
            typer.echo("    no known-uncapturable call sites found")
    typer.echo("")

    if output is not None:
        import json as _json

        output.write_text(
            _json.dumps(
                {
                    "draw_counts": result.draw_counts,
                    "concurrency_recorded": result.concurrency_recorded,
                    "boundary_guard_active": result.boundary_guard_active,
                    "findings": [
                        {
                            "call": f.call,
                            "lineno": f.lineno,
                            "guardable": f.guardable,
                            "caught": f.caught,
                            "note": f.note,
                        }
                        for f in result.findings
                    ],
                },
                indent=2,
            )
        )
        typer.echo(f"  Report saved to {output}\n")


@app.command()
def corpus_blame(
    top_n: int = typer.Option(20, "--top-n", help="Cap on the top_responsible list"),
    method: str = typer.Option(
        "blame", "--method", help="Method to run regression detection over (blame|shapley)"
    ),
    z_threshold: float = typer.Option(
        2.0, "--z-threshold", help="Regression flag threshold (|z-score| >= this)"
    ),
    min_history: int = typer.Option(
        3, "--min-history", help="Minimum prior history points required before flagging a step"
    ),
    store: Path = typer.Option(  # noqa: B008
        Path(_DEFAULT_CONFIG.db_path), "--store", help="Path to store.db"
    ),
    output: Path = typer.Option(  # noqa: B008
        None, "--output", "-o", help="Optional JSON output path"
    ),
) -> None:
    """Print a corpus-wide blame/Shapley index plus a z-score regression flag list.

    Aggregates every run's persisted `causal_edges` rows
    (`store.py`'s `save_blame_report`/`save_shapley_report`) into a
    corpus-wide top-responsible index, and flags any `(agent_name,
    step_index)` whose latest run's flip_rate/shapley_value is a
    statistical outlier against that step's own prior history (see
    `corpus.py`'s module docstring). A diagnostic report, not a gate --
    always exits 0.
    """
    import json as _json
    from dataclasses import asdict

    from tracefork.corpus import build_corpus_blame_index, detect_regressions
    from tracefork.store import TapeStore

    db = TapeStore(str(store))
    try:
        index = build_corpus_blame_index(db, top_n=top_n)
        flags = detect_regressions(
            db, method=method, z_threshold=z_threshold, min_history=min_history
        )
    finally:
        db.close()

    typer.echo("\n  tracefork corpus-blame")
    typer.echo(f"  {'─' * 50}")
    typer.echo(f"  runs     {index.run_count}")
    typer.echo(f"  edges    {index.edge_count}")
    for m, count in sorted(index.by_method.items()):
        typer.echo(f"    {m:<10} {count}")

    typer.echo("\n  top responsible:")
    if index.top_responsible:
        for s in index.top_responsible:
            typer.echo(
                f"    {s.agent_name:<16} run={s.run_id:<12} step={s.step_index:<4} "
                f"{s.method:<8} score={s.score:.3f}"
            )
    else:
        typer.echo("    (no causal_edges recorded)")

    typer.echo(f"\n  regressions (method={method}, |z|>={z_threshold}):")
    if flags:
        for f in flags:
            typer.echo(
                f"    {f.agent_name:<16} step={f.step_index:<4} run={f.run_id:<12} "
                f"value={f.value:.3f} mean={f.history_mean:.3f} z={f.z_score:+.2f}"
            )
    else:
        typer.echo("    (none)")
    typer.echo("")

    if output is not None:
        output.write_text(
            _json.dumps(
                {
                    "run_count": index.run_count,
                    "edge_count": index.edge_count,
                    "by_method": index.by_method,
                    "top_responsible": [asdict(s) for s in index.top_responsible],
                    "regressions": [asdict(f) for f in flags],
                },
                indent=2,
            )
        )
        typer.echo(f"  Report saved to {output}\n")


@app.command()
def tournament(
    run_id: str = typer.Argument(..., help="run_id to analyze"),
    agent: str = typer.Option(
        ...,
        "--agent",
        "-a",
        help="Import path of the agent fn (pkg.mod:fn) that produced this run; "
        "it is re-run for each variant trial and must be deterministic up to "
        "the fixed step",
    ),
    candidate: list[str] = typer.Option(  # noqa: B008
        ...,
        "--candidate",
        help="A 'name:text' candidate continuation, repeatable (>=1 required) -- "
        "'text' becomes the forced response at --step",
    ),
    step: int = typer.Option(
        -1,
        "--step",
        help="Fixed step index to compare candidates at (default: the tape's "
        "last exchange -- a $0 comparison, no tail calls)",
    ),
    k: int = typer.Option(10, "--k", help="Forks per candidate variant"),
    budget: float = typer.Option(_DEFAULT_CONFIG.budget_usd, "--budget", help="USD spend cap"),
    success_re: str = typer.Option("SUCCESS", "--success-re", help="Regex for success outcome"),
    failure_re: str = typer.Option("FAIL", "--failure-re", help="Regex for failure outcome"),
    ci_method: str = typer.Option(
        "wilson", "--ci-method", help="Proportion CI: wilson|jeffreys|clopper_pearson|agresti_coull"
    ),
    confidence: float = typer.Option(0.95, "--confidence", help="CI confidence level (0,1)"),
    fdr_q: float = typer.Option(
        0.10, "--fdr-q", help="Benjamini-Hochberg false-discovery-rate for the winner test"
    ),
    store: Path = typer.Option(  # noqa: B008
        Path(_DEFAULT_CONFIG.db_path), "--store", help="Path to store.db"
    ),
) -> None:
    """Rank pre-specified candidate continuations at one fixed step.

    Unlike `blame` (which compares a single perturbation against the parent
    outcome, across every step), a tournament compares N candidate responses
    against EACH OTHER at one step you choose, ranked by success rate with a
    Wilson CI and a Benjamini-Hochberg-corrected significance test of the top
    candidate against every runner-up. When `--step` targets the tape's last
    exchange (the default), the comparison is $0 -- see `tournament.py`'s
    module docstring.
    """
    if not run_id or not all(c.isalnum() or c in "-_" for c in run_id):
        raise typer.BadParameter("run_id must be alphanumeric (with '-' or '_')")

    import importlib
    import json
    import os

    from tracefork.blame import BudgetExceededError, CIMethod, StringMatchOracle
    from tracefork.store import TapeStore
    from tracefork.tournament import TournamentEngine, Variant
    from tracefork.wire import make_text_response

    try:
        method = CIMethod(ci_method)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    db = TapeStore(str(store))
    tape = db.load_tape(run_id)

    step_index = step if step >= 0 else len(tape.exchanges) - 1
    if step_index < 0 or step_index >= len(tape.exchanges):
        raise typer.BadParameter(f"--step {step_index} out of range [0, {len(tape.exchanges)})")

    variants = []
    for spec in candidate:
        name, sep, text = spec.partition(":")
        if not sep or not name or not text:
            raise typer.BadParameter(f"--candidate must be 'name:text', got {spec!r}")
        variants.append(Variant(name=name, response=make_text_response(text)))

    module_path, fn_name = agent.rsplit(":", 1)
    agent_fn = getattr(importlib.import_module(module_path), fn_name)

    oracle = StringMatchOracle(success_re=success_re, failure_re=failure_re)
    est = TournamentEngine.estimate(tape, step_index=step_index, n_variants=len(variants), k=k)

    typer.echo(f"\n  Tournament estimate: {est.n_forks} forks, ~${est.est_usd:.2f}")
    if est.est_usd > budget:
        typer.echo(f"  Estimated cost ${est.est_usd:.2f} exceeds budget ${budget:.2f}.")
        typer.echo("  Use --budget to increase or --k to reduce trials.")
        raise typer.Exit(1)

    try:
        report = TournamentEngine.run(
            tape,
            step_index=step_index,
            variants=variants,
            agent_fn=agent_fn,
            oracle=oracle,
            k=k,
            budget_usd=budget,
            api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            ci_method=method,
            confidence=confidence,
            fdr_q=fdr_q,
        )
    except BudgetExceededError as exc:
        typer.echo(f"  {exc}")
        raise typer.Exit(1) from exc

    ci_pct = round(confidence * 100)
    typer.echo(
        f"\n  run-{run_id} · tournament at step-{step_index} · k={k} · "
        f"{report.total_forks} forks · {method.value} {ci_pct}% CI\n"
    )
    ci_hdr = f"{ci_pct}% CI"
    typer.echo(f"  {'rank':<5} {'variant':<16} {'score':<10} {ci_hdr:<22} {'q-value':<10} winner")
    typer.echo(f"  {'─' * 76}")
    for rank, r in enumerate(report.results, 1):
        ci_str = f"[{r.ci_lo:.2f}, {r.ci_hi:.2f}]"
        flag = " *" if r.significant_winner else ""
        typer.echo(
            f"  {rank:<5} {r.name:<16} {r.score:<10.2f} {ci_str:<22} {r.q_value:<10.3g}{flag}"
        )

    winner = report.winner()
    if winner is not None:
        typer.echo(f"\n  winner (FDR q≤{fdr_q}): {winner.name}")
    else:
        typer.echo(f"\n  winner (FDR q≤{fdr_q}): (no candidate significantly beats every other)")
    typer.echo("")

    report_path = Path(f"tournament_{run_id}.json")
    report_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "step_index": step_index,
                "k": k,
                "ci_method": method.value,
                "confidence": confidence,
                "fdr_q": fdr_q,
                "winner": winner.name if winner is not None else None,
                "results": [
                    {
                        "name": r.name,
                        "score": r.score,
                        "ci_lo": r.ci_lo,
                        "ci_hi": r.ci_hi,
                        "successes": r.successes,
                        "trials": r.trials,
                        "valid_trials": r.valid_trials,
                        "undefined": r.undefined,
                        "divergences": r.divergences,
                        "q_value": r.q_value,
                        "significant_winner": r.significant_winner,
                    }
                    for r in report.results
                ],
            },
            indent=2,
        )
    )
    typer.echo(f"  Report saved to {report_path}")


@app.command()
def locate(
    value: str = typer.Argument(None, help="Substring to locate in the tape (or its fork lineage)"),
    run_id: str = typer.Argument(
        None, help="run_id to search (from store); omit when using --tape"
    ),
    tape_path: Path = typer.Option(  # noqa: B008
        None,
        "--tape",
        "-t",
        help="Path to a .tape.sqlite file (single-tape mode, no store/lineage)",
    ),
    store: Path = typer.Option(  # noqa: B008
        Path(_DEFAULT_CONFIG.db_path), "--store", help="Path to store.db"
    ),
    no_lineage: bool = typer.Option(
        False,
        "--no-lineage",
        help="Restrict the search to the root tape only, skip the fork-lineage BFS",
    ),
) -> None:
    """Locate a substring inside a tape (or its fork lineage) and print an
    offline-checkable receipt (tracefork-bge.62): which exchange kind/index/
    side it was found in, plus blob_sha256/tape_digest -- hashes
    `Tape.digest()` itself already folds in, so any reader can re-hash the
    raw exchange bytes themselves and compare, no need to trust this command.

    `--tape` searches a single tape file, no store/lineage involved. Without
    `--tape`, `run_id` is looked up in `--store` and the search also
    BFS-walks its fork lineage (direct branches, fork-of-forks, ...) unless
    `--no-lineage` is passed. Exit 0 if found, 1 otherwise.
    """
    from tracefork.locate import locate_in_lineage, locate_value
    from tracefork.tape import Tape

    if not tape_path and not run_id:
        typer.echo("Provide a run_id or --tape path")
        raise typer.Exit(1)
    if not value:
        typer.echo("Provide a value to search for")
        raise typer.Exit(1)

    if tape_path:
        heading = f"tracefork locate {value!r} --tape {tape_path}"
        tape = Tape.load(str(tape_path))
        hit = locate_value(tape, value)
        _print_locate_receipt(heading, hit, branch_id=None, depth=None)
        raise typer.Exit(0 if hit is not None else 1)

    from tracefork.store import TapeStore

    heading = f"tracefork locate {value!r} {run_id}"
    db = TapeStore(str(store))
    try:
        hits = locate_in_lineage(db, run_id, value, follow_lineage=not no_lineage)
    except KeyError as exc:
        typer.echo(str(exc))
        raise typer.Exit(1) from exc
    finally:
        db.close()

    found = hits[0] if hits else None
    _print_locate_receipt(
        heading,
        found.hit if found else None,
        branch_id=found.branch_id if found else None,
        depth=found.depth if found else None,
    )
    raise typer.Exit(0 if found is not None else 1)


def _print_locate_receipt(heading: str, hit, *, branch_id: str | None, depth: int | None) -> None:
    """Operator-facing receipt for `locate` -- PASS with the found location
    plus blob_sha256/tape_digest lines, or FAIL 'not found'. Mirrors
    `_print_diff_receipt`'s PASS/FAIL style."""
    typer.echo(f"\n  {heading}")
    typer.echo(f"  {'─' * 60}")
    if hit is None:
        typer.echo("  [FAIL] not found\n")
        return
    where = "root tape" if branch_id is None else f"branch {branch_id} (depth {depth})"
    typer.echo(f"  [PASS] found in {where}")
    typer.echo(f"  kind          {hit.kind}")
    typer.echo(f"  index         {hit.index}")
    typer.echo(f"  side          {hit.side}")
    typer.echo(f"  blob_sha256   {hit.blob_sha256}")
    typer.echo(f"  tape_digest   {hit.tape_digest}\n")


@app.command()
def query(
    store: Path = typer.Option(  # noqa: B008
        Path(_DEFAULT_CONFIG.db_path), "--store", help="Path to store.db"
    ),
    cmd: str = typer.Option(
        None,
        "--cmd",
        "-c",
        help="Run one query line and exit (scriptable); omit to open the interactive REPL",
    ),
) -> None:
    """Query a store's tapes/branches/causal graph: state/diff/causes/tree.

    With --cmd, runs ONE query line and exits (exit 1 on a bad query --
    scriptable, and the only way to CI-test this command without blocking
    on stdin, same rationale as `serve`/`proxy`'s monkeypatch-uvicorn
    tests). Without --cmd, opens an interactive `repl.QueryShell` over
    --store. Grammar: `state <run_id> <step>` / `diff <a> <b> [--step N]` /
    `causes <run_id> <step|--closure>` / `tree <run_id>` -- see
    `tracefork.query`'s module docstring.
    """
    from tracefork.query import QueryError, dispatch
    from tracefork.repl import run_repl
    from tracefork.store import TapeStore

    db = TapeStore(str(store))
    try:
        if cmd is not None:
            try:
                typer.echo(dispatch(db, cmd))
            except QueryError as exc:
                typer.echo(f"error: {exc}")
                raise typer.Exit(1) from exc
        else:
            run_repl(db)
    finally:
        db.close()


# ── session (orchestration spawn-lineage) sub-app ───────────────────────────

branch_app = typer.Typer(
    name="branch",
    help="Branch DAG relationship queries (descendants/ancestors/siblings).",
)
app.add_typer(branch_app, name="branch")


@branch_app.command("descendants")
def branch_descendants_cmd(
    run_id: str = typer.Argument(..., help="run_id (or branch_id) to walk descendants from"),
    store: Path = typer.Option(  # noqa: B008
        Path(_DEFAULT_CONFIG.db_path), "--store", help="Path to store.db"
    ),
) -> None:
    """Every branch reachable from RUN_ID via fork-of-fork chains (BFS over
    `branches.parent_run_id`, promoted branches only recursed into)."""
    from tracefork.store import TapeStore

    db = TapeStore(str(store))
    try:
        descendants = db.branch_descendants(run_id)
    finally:
        db.close()

    typer.echo(f"\n  Descendants of {run_id} ({len(descendants)}):")
    for branch_id in descendants:
        typer.echo(f"    {branch_id}")
    typer.echo("")


@branch_app.command("ancestors")
def branch_ancestors_cmd(
    run_id: str = typer.Argument(..., help="branch_id to walk ancestors from"),
    store: Path = typer.Option(  # noqa: B008
        Path(_DEFAULT_CONFIG.db_path), "--store", help="Path to store.db"
    ),
) -> None:
    """Walk RUN_ID's parent_run_id chain upward, nearest-parent-first."""
    from tracefork.store import TapeStore

    db = TapeStore(str(store))
    try:
        ancestors = db.branch_ancestors(run_id)
    finally:
        db.close()

    typer.echo(f"\n  Ancestors of {run_id} ({len(ancestors)}):")
    for parent_id in ancestors:
        typer.echo(f"    {parent_id}")
    typer.echo("")


@branch_app.command("siblings")
def branch_siblings_cmd(
    run_id: str = typer.Argument(..., help="branch_id to find siblings of"),
    store: Path = typer.Option(  # noqa: B008
        Path(_DEFAULT_CONFIG.db_path), "--store", help="Path to store.db"
    ),
) -> None:
    """Every other branch forked from RUN_ID's own parent_run_id."""
    from tracefork.store import TapeStore

    db = TapeStore(str(store))
    try:
        siblings = db.branch_siblings(run_id)
    finally:
        db.close()

    typer.echo(f"\n  Siblings of {run_id} ({len(siblings)}):")
    for sibling_id in siblings:
        typer.echo(f"    {sibling_id}")
    typer.echo("")


session_app = typer.Typer(
    name="session",
    help="Orchestration session / spawn-lineage commands "
    "(create/spawn/show/cost/divergence/record/replay/fork/blame/"
    "cross-blame/chaos/serve).",
)
app.add_typer(session_app, name="session")


@session_app.command("create")
def session_create(
    root_run_id: str = typer.Argument(..., help="run_id of the session's root tape"),
    store: Path = typer.Option(  # noqa: B008
        Path(_DEFAULT_CONFIG.db_path), "--store", help="Path to store.db"
    ),
) -> None:
    """Create a new orchestration session rooted at ROOT_RUN_ID.

    ROOT_RUN_ID must already be a stored tape (`FOREIGN KEY` to `tapes`); an
    unknown run_id is rejected rather than silently accepted.
    """
    import sqlite3

    from tracefork.store import TapeStore

    db = TapeStore(str(store))
    try:
        session_id = db.create_session(root_run_id=root_run_id)
    except sqlite3.IntegrityError as exc:
        typer.echo(f"  {exc}")
        raise typer.Exit(1) from exc
    finally:
        db.close()

    typer.echo("\n  Session created")
    typer.echo(f"  session_id   {session_id}")
    typer.echo(f"  root_run_id  {root_run_id}\n")


@session_app.command("spawn")
def session_spawn(
    session_id: str = typer.Argument(..., help="Session id to add the spawn edge to"),
    parent_run_id: str = typer.Argument(..., help="Delegating run_id"),
    child_run_id: str = typer.Argument(..., help="Spawned/delegated-to run_id"),
    reason: str = typer.Option("", "--reason", help="Human-readable spawn reason"),
    spawn_step: int = typer.Option(
        None, "--spawn-step", help="Step index of parent_run_id this child was spawned at"
    ),
    store: Path = typer.Option(  # noqa: B008
        Path(_DEFAULT_CONFIG.db_path), "--store", help="Path to store.db"
    ),
) -> None:
    """Record a spawn/delegation edge from PARENT_RUN_ID to CHILD_RUN_ID
    within SESSION_ID.

    SESSION_ID/PARENT_RUN_ID/CHILD_RUN_ID must already exist (a live session
    and two stored tapes) — each enforced by its own `FOREIGN KEY`.
    """
    import sqlite3

    from tracefork.store import TapeStore

    db = TapeStore(str(store))
    try:
        edge_id = db.add_spawn_edge(
            session_id=session_id,
            parent_run_id=parent_run_id,
            child_run_id=child_run_id,
            spawn_reason=reason,
            spawn_step_index=spawn_step,
        )
    except sqlite3.IntegrityError as exc:
        typer.echo(f"  {exc}")
        raise typer.Exit(1) from exc
    finally:
        db.close()

    typer.echo("\n  Spawn edge recorded")
    typer.echo(f"  edge_id        {edge_id}")
    typer.echo(f"  session_id     {session_id}")
    typer.echo(f"  parent_run_id  {parent_run_id}")
    typer.echo(f"  child_run_id   {child_run_id}\n")


@session_app.command("show")
def session_show(
    session_id: str = typer.Argument(..., help="Session id to show"),
    store: Path = typer.Option(  # noqa: B008
        Path(_DEFAULT_CONFIG.db_path), "--store", help="Path to store.db"
    ),
) -> None:
    """Show a session's root tape and every tape reachable via spawn edges
    (a BFS over `spawn_edges`, see `store.py`'s `session_tapes`)."""
    from tracefork.store import TapeStore

    db = TapeStore(str(store))
    try:
        try:
            session = db.get_session(session_id)
        except KeyError as exc:
            typer.echo(f"  {exc}")
            raise typer.Exit(1) from exc
        tapes = db.session_tapes(session_id)
    finally:
        db.close()

    typer.echo("\n  Session")
    typer.echo(f"  session_id   {session['session_id']}")
    typer.echo(f"  root_run_id  {session['root_run_id']}")
    typer.echo(f"  created_at   {session['created_at']}")
    typer.echo(f"  tapes ({len(tapes)}):")
    for rid in tapes:
        typer.echo(f"    {rid}")
    typer.echo("")


@session_app.command("cost")
def session_cost(
    session_id: str = typer.Argument(..., help="Session id to plan a fork within"),
    target_run_id: str = typer.Argument(
        ..., help="run_id to fork; prices its transitive spawn subtree"
    ),
    k: int = typer.Option(1, "--k", help="Forks per step (mirrors BudgetGovernor.estimate's k)"),
    model: str = typer.Option(
        None, "--model", help="Model id for pricing (default: auto-detect from the tape)"
    ),
    cost_per_fork_usd: float = typer.Option(
        None,
        "--cost-per-fork-usd",
        help="Flat per-fork USD cost override instead of token-based pricing",
    ),
    store: Path = typer.Option(  # noqa: B008
        Path(_DEFAULT_CONFIG.db_path), "--store", help="Path to store.db"
    ),
) -> None:
    """Plan a minimal-recompute fork of TARGET_RUN_ID within SESSION_ID.

    Walks the spawn-edge DAG (`store.py`'s `session_tapes`/`spawn_children`) to
    partition the session's tapes into a recompute set (TARGET_RUN_ID plus its
    transitive spawn descendants) and a skip set (everything else, genuinely
    independent upstream), prices both sets via `blame.py`'s existing
    `BudgetGovernor.estimate`, and prints the resulting plan as JSON.
    """
    import json

    from tracefork.session_cost import plan_session_fork
    from tracefork.store import TapeStore

    db = TapeStore(str(store))
    try:
        try:
            plan = plan_session_fork(
                db,
                session_id,
                target_run_id,
                k=k,
                model=model,
                cost_per_fork_usd=cost_per_fork_usd,
            )
        except KeyError as exc:
            typer.echo(f"  {exc}")
            raise typer.Exit(1) from exc
        except ValueError as exc:
            typer.echo(f"  {exc}")
            raise typer.Exit(1) from exc
    finally:
        db.close()

    typer.echo(
        json.dumps(
            {
                "session_id": plan.session_id,
                "target_run_id": plan.target_run_id,
                "recompute_run_ids": plan.recompute_run_ids,
                "skip_run_ids": plan.skip_run_ids,
                "est_usd": plan.est_usd,
                "est_usd_naive": plan.est_usd_naive,
                "savings_usd": plan.savings_usd,
                "savings_pct": plan.savings_pct,
            },
            indent=2,
        )
    )


@session_app.command("divergence")
def session_divergence(
    session_id: str = typer.Argument(..., help="Session id to check for divergence"),
    agents_manifest: Path = typer.Option(  # noqa: B008
        ...,
        "--agents-manifest",
        help="Path to a JSON {run_id: 'module:fn'} agent manifest",
    ),
    store: Path = typer.Option(  # noqa: B008
        Path(_DEFAULT_CONFIG.db_path), "--store", help="Path to store.db"
    ),
) -> None:
    """Replay every SESSION_ID spawn-linked tape that AGENTS_MANIFEST maps an
    agent_fn for (see `session_replay.session_divergence_rollup`), in BFS/
    session order, and report the earliest genuine divergence, if any.

    AGENTS_MANIFEST is a JSON object `{run_id: "module:fn"}`, resolved via
    the same importlib/rsplit(':', 1) pattern `verify`/`replay --check`
    already use. A reachable run_id absent from the manifest is skipped, not
    crashed on.

    Exits 1 iff a real divergence was found before the session was fully
    consistent; 0 otherwise — including when every reachable tape was
    skipped for lack of a manifest entry (an honest no-op, not a false
    pass/fail).
    """
    import json

    from tracefork import session_replay
    from tracefork.replay import DriftDoctor
    from tracefork.store import TapeStore

    manifest = json.loads(agents_manifest.read_text())
    agent_fns = session_replay.resolve_agent_manifest(manifest)

    db = TapeStore(str(store))
    try:
        try:
            result = session_replay.session_divergence_rollup(db, session_id, agent_fns)
        except KeyError as exc:
            typer.echo(f"  {exc}")
            raise typer.Exit(1) from exc
    finally:
        db.close()

    typer.echo("\n  tracefork session divergence")
    typer.echo(f"  {'─' * 60}")
    typer.echo(f"  session_id   {session_id}")
    typer.echo(f"  checked      {', '.join(result.checked_run_ids) or '(none)'}")
    typer.echo(f"  skipped      {', '.join(result.skipped_run_ids) or '(none)'}")
    if result.diverged_run_id is None:
        typer.echo("  result       no divergence\n")
        raise typer.Exit(0)

    cause = DriftDoctor.classify(result.divergence) if result.divergence else None
    typer.echo(f"  result       DIVERGED at run_id={result.diverged_run_id}")
    if result.divergence is not None:
        typer.echo(f"  step_index   {result.divergence.step_index}")
    if cause is not None:
        typer.echo(f"  drift cause  {cause.value}")
    typer.echo("")
    raise typer.Exit(1)


@session_app.command("record")
def session_record(
    root_run_id: str = typer.Argument(..., help="run_id of the session's root tape"),
    spawn: list[str] = typer.Option(  # noqa: B008
        [],
        "--spawn",
        help="A 'PARENT:CHILD[:REASON]' spawn edge, repeatable -- batch-creates "
        "the session and registers every edge in one call (see session_ops.record_session)",
    ),
    store: Path = typer.Option(  # noqa: B008
        Path(_DEFAULT_CONFIG.db_path), "--store", help="Path to store.db"
    ),
) -> None:
    """Batch-create a session rooted at ROOT_RUN_ID and register every
    `--spawn` edge in one call.

    Equivalent to N+1 separate `session create`/`session spawn` calls; each
    edge is still individually FK-validated by `TapeStore.add_spawn_edge`
    (a dangling parent/child run_id raises `sqlite3.IntegrityError`,
    propagated unchanged).
    """
    import sqlite3

    from tracefork.session_ops import record_session
    from tracefork.store import TapeStore

    db = TapeStore(str(store))
    try:
        try:
            session_id, edges = record_session(db, root_run_id, spawn)
        except sqlite3.IntegrityError as exc:
            typer.echo(f"  {exc}")
            raise typer.Exit(1) from exc
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
    finally:
        db.close()

    typer.echo("\n  Session created")
    typer.echo(f"  session_id   {session_id}")
    typer.echo(f"  root_run_id  {root_run_id}")
    typer.echo(f"  spawn_edges  {len(edges)}")
    for edge in edges:
        typer.echo(
            f"    {edge.parent_run_id} -> {edge.child_run_id} ({edge.spawn_reason or '(none)'})"
        )
    typer.echo("")


@session_app.command("replay")
def session_replay_cmd(
    session_id: str = typer.Argument(..., help="Session id to check for bit-exact replay"),
    agent: str = typer.Option(
        ...,
        "--agent",
        "-a",
        help="Import path of the SAME agent fn for every tape in the session",
    ),
    store: Path = typer.Option(  # noqa: B008
        Path(_DEFAULT_CONFIG.db_path), "--store", help="Path to store.db"
    ),
) -> None:
    """Replay every tape reachable within SESSION_ID with the SAME --agent
    and report the earliest genuine divergence, if any (see
    `session_ops.build_uniform_agent_manifest` +
    `session_replay.session_divergence_rollup`).

    Exits 1 on a genuine divergence or an unknown session_id, 0 if every
    reachable tape replayed bit-exact.
    """
    import importlib

    from tracefork import session_replay
    from tracefork.replay import DriftDoctor
    from tracefork.session_ops import build_uniform_agent_manifest
    from tracefork.store import TapeStore

    module_path, fn_name = agent.rsplit(":", 1)
    agent_fn = getattr(importlib.import_module(module_path), fn_name)

    db = TapeStore(str(store))
    try:
        try:
            run_ids = db.session_tapes(session_id)
        except KeyError as exc:
            typer.echo(f"  {exc}")
            raise typer.Exit(1) from exc
        manifest = build_uniform_agent_manifest(run_ids, agent_fn)
        result = session_replay.session_divergence_rollup(db, session_id, manifest)
    finally:
        db.close()

    typer.echo("\n  tracefork session replay")
    typer.echo(f"  {'─' * 60}")
    typer.echo(f"  session_id   {session_id}")
    typer.echo(f"  checked      {', '.join(result.checked_run_ids) or '(none)'}")
    typer.echo(f"  skipped      {', '.join(result.skipped_run_ids) or '(none)'}")
    if result.diverged_run_id is None:
        typer.echo("  result       every tape replayed bit-exact\n")
        raise typer.Exit(0)

    cause = DriftDoctor.classify(result.divergence) if result.divergence else None
    typer.echo(f"  result       DIVERGED at run_id={result.diverged_run_id}")
    if result.divergence is not None:
        typer.echo(f"  step_index   {result.divergence.step_index}")
    if cause is not None:
        typer.echo(f"  drift cause  {cause.value}")
    typer.echo("")
    raise typer.Exit(1)


@session_app.command("fork")
def session_fork(
    session_id: str = typer.Argument(..., help="Session id RUN_ID must belong to"),
    run_id: str = typer.Argument(..., help="Parent run_id (member of session_id) to fork from"),
    step: int = typer.Option(..., "--step", "-s", help="Exchange index to diverge at"),
    response_file: Path = typer.Option(  # noqa: B008
        ..., "--response", "-r", help="Path to .bytes file containing mutated response"
    ),
    agent: str = typer.Option(..., "--agent", "-a", help="Import path of post-fork agent fn"),
    store: Path = typer.Option(  # noqa: B008
        Path(_DEFAULT_CONFIG.db_path), "--store", help="Path to store.db"
    ),
    desc: str = typer.Option("", "--desc", "-d", help="Human description of mutation"),
    writable_root: list[str] = typer.Option(  # noqa: B008
        [],
        "--writable-root",
        help="Directory a ConfinementSpec permits writes under (repeatable); see "
        "`tracefork fork --writable-root`",
    ),
    allowed_host: list[str] = typer.Option(  # noqa: B008
        [],
        "--allowed-host",
        help="Hostname a ConfinementSpec permits socket.connect to (repeatable)",
    ),
) -> None:
    """Fork RUN_ID within SESSION_ID, guarding session membership first.

    Delegates to the top-level `fork` command after confirming RUN_ID is
    reachable within SESSION_ID (`session_ops.ensure_run_in_session`) --
    same fork engine, no second implementation.
    """
    from tracefork.session_ops import ensure_run_in_session
    from tracefork.store import TapeStore

    db = TapeStore(str(store))
    try:
        try:
            ensure_run_in_session(db, session_id, run_id)
        except (KeyError, ValueError) as exc:
            typer.echo(f"  {exc}")
            raise typer.Exit(1) from exc
    finally:
        db.close()

    fork(
        run_id=run_id,
        step=step,
        response_file=response_file,
        agent=agent,
        store=store,
        desc=desc,
        writable_root=writable_root,
        allowed_host=allowed_host,
    )


@session_app.command("blame")
def session_blame(
    session_id: str = typer.Argument(..., help="Session id RUN_ID must belong to"),
    run_id: str = typer.Argument(..., help="run_id (member of session_id) to analyze"),
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
    field: str = typer.Option(
        None,
        "--field",
        help="JSON field path ($.a.b[0].c) to grade instead of the whole output text",
    ),
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
    """Run causal blame analysis on RUN_ID within SESSION_ID, guarding session
    membership first.

    Delegates to the top-level `blame` command after confirming RUN_ID is
    reachable within SESSION_ID (`session_ops.ensure_run_in_session`) --
    same blame engine, no second implementation.
    """
    from tracefork.session_ops import ensure_run_in_session
    from tracefork.store import TapeStore

    db = TapeStore(str(store))
    try:
        try:
            ensure_run_in_session(db, session_id, run_id)
        except (KeyError, ValueError) as exc:
            typer.echo(f"  {exc}")
            raise typer.Exit(1) from exc
    finally:
        db.close()

    blame(
        run_id=run_id,
        agent=agent,
        k=k,
        budget=budget,
        perturbation=perturbation,
        success_re=success_re,
        failure_re=failure_re,
        field=field,
        ci_method=ci_method,
        confidence=confidence,
        fdr_q=fdr_q,
        null_flip_rate=null_flip_rate,
        store=store,
    )


@session_app.command("cross-blame")
def session_cross_blame(
    session_id: str = typer.Argument(
        ..., help="Session id to build the cross-tape causal view for"
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Emit JSON instead of a formatted table"
    ),
    store: Path = typer.Option(  # noqa: B008
        Path(_DEFAULT_CONFIG.db_path), "--store", help="Path to store.db"
    ),
) -> None:
    """Session-wide causal view: aggregates every already-persisted
    causal_edges row across SESSION_ID's tapes
    (cross_tape_blame.cross_tape_causal_edges), ordered by cross-tape
    topological position. Read-only, $0 -- never invokes fork/blame itself,
    only reads already-persisted rows a prior `blame`/`session blame` run
    wrote."""
    import json

    from tracefork.cross_tape_blame import cross_tape_causal_edges
    from tracefork.store import TapeStore

    db = TapeStore(str(store))
    try:
        try:
            db.get_session(session_id)
        except KeyError as exc:
            typer.echo(f"  {exc}")
            raise typer.Exit(1) from exc
        edges = cross_tape_causal_edges(db, session_id)
    finally:
        db.close()

    if json_output:
        typer.echo(json.dumps(edges, indent=2))
        return

    typer.echo(f"\n  Cross-tape causal view for session {session_id} ({len(edges)} edges):")
    for edge in edges:
        typer.echo(
            f"    {edge['run_id']}:{edge['step_index']} "
            f"{edge['method']} responsible={edge['responsible']}"
        )
    typer.echo("")


@session_app.command("chaos")
def session_chaos_cmd(
    session_id: str = typer.Argument(..., help="Session id to derive chaos schedules for"),
    seed: int = typer.Option(..., "--seed", help="Base seed for the derived schedules"),
    store: Path = typer.Option(  # noqa: B008
        Path(_DEFAULT_CONFIG.db_path), "--store", help="Path to store.db"
    ),
) -> None:
    """Derive (never replay-drive) per-tape chaos release orders and
    cross-sub-agent sibling completion orders for SESSION_ID -- an analysis
    surface only (see session_chaos.py's module docstring)."""
    import json

    from tracefork.session_chaos import session_chaos_release_orders, session_sibling_chaos_order
    from tracefork.store import TapeStore

    db = TapeStore(str(store))
    try:
        try:
            db.get_session(session_id)
        except KeyError as exc:
            typer.echo(f"  {exc}")
            raise typer.Exit(1) from exc
        per_tape = session_chaos_release_orders(db, session_id, seed)
        sibling = session_sibling_chaos_order(db, session_id, seed)
    finally:
        db.close()

    typer.echo(
        json.dumps({"per_tape_release_orders": per_tape, "sibling_chaos_order": sibling}, indent=2)
    )


@session_app.command("serve")
def session_serve(
    session_id: str = typer.Argument(..., help="Session id to open the web UI to"),
    store: Path = typer.Option(  # noqa: B008
        Path(_DEFAULT_CONFIG.db_path), "--store", help="Path to store.db"
    ),
    port: int = typer.Option(7777, "--port", "-p", help="Port to listen on"),
) -> None:
    """Start the tracefork web UI server and print SESSION_ID's deep link.

    Exits 1 (never starting the server) if SESSION_ID is unknown.
    """
    import uvicorn

    from tracefork.server import app as fastapi_app
    from tracefork.server import init_store
    from tracefork.session_ops import session_deep_link_path
    from tracefork.store import TapeStore

    db = TapeStore(str(store))
    try:
        db.get_session(session_id)
    except KeyError as exc:
        typer.echo(f"  {exc}")
        raise typer.Exit(1) from exc
    finally:
        db.close()

    init_store(str(store))
    deep_link = session_deep_link_path(session_id)
    typer.echo(f"\n  tracefork session serve → http://127.0.0.1:{port}{deep_link}")
    uvicorn.run(fastapi_app, host="127.0.0.1", port=port, workers=1, log_level="warning")


def _print_receipt(tape_path: Path, result, tape) -> None:
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
    _print_trust_lines(tape)
    typer.echo("")


def _print_trust_lines(tape) -> None:
    """Print the two trust/provenance lines (`Tape.boundary`/`content_redacted`)
    shared by the replay/verify receipt and the `report` command's terminal
    echo (tracefork-bge.20) — a forensic-only or content-redacted tape must
    not look identical to a verified one. Both fields are envelope metadata,
    never fed into `Tape.digest()` (see `tape.py`); this is a trust warning,
    not a pass/fail input.
    """
    typer.echo(f"  boundary        {tape.boundary}")
    typer.echo(f"  content_redacted {tape.content_redacted}")
