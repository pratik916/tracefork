"""Generate the demo tracefork report (examples/demo_report.html) — the artifact
docs/demo.png is a screenshot of, and every distinctive claim in README.md's headline
paragraph should be visible in.

Records a synthetic 7-exchange run (`tracefork.competing_faults`'s long-tape fixture — a
root-cause fault, a downstream echo of it, decoy neutral steps, and a two-part
necessary-not-sufficient conjunction, all on one tape), then populates every panel the
workbench UI renders: a bit-exact replay receipt, three forked branches (a real fork tree),
causal blame AND temporal-Shapley necessity/sufficiency, a per-model/per-tool cost profile,
and the run's causal closure. Fully offline, $0 — no server, no API key, no network. Open the
resulting HTML in any browser.

    uv run python examples/demo_report.py
    open examples/demo_report.html
"""

from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from tracefork import (
    BlameEngine,
    BranchSpec,
    ForkEngine,
    ReplayVerifier,
    StringMatchOracle,
    TapeStore,
)

# Internal-use only, below this line: these come from tracefork's test/demo scaffolding
# (competing_faults.py — see docs/stability.md's "explicitly internal" list) and from two
# real product modules (cost_profile.py, replay.py) that aren't yet re-exported at the top
# level — imported the same deep-module way `cli.py`'s own `report`/`blame`/`bench` commands
# already do, not a shortcut unique to this demo. A real integration should look like
# examples/record_your_agent.py instead, which imports only from top-level tracefork plus
# real dependencies.
from tracefork.competing_faults import (
    N_TURNS,
    SCENARIO_ROOT_ECHO,
    RuleBasedTail,
    StepRole,
    build_competing_fault_tape,
    competing_fault_agent,
    make_perturb_factory,
    mutated_response_for,
)
from tracefork.cost_profile import compute_cost_profile, cost_profile_to_dict
from tracefork.replay import verification_result_to_dict
from tracefork.report import _tape_to_data, generate_report

RUN_ID = "demo-booking-run"

# Three forked branches, spread across the tape, telling three different stories:
#   step 0 (ROOT)  -> the root cause alone -> FAILs immediately.
#   step 1 (ECHO)  -> a downstream re-expression of the same marker -> also FAILs,
#                     but is NOT causally necessary once the root itself is considered
#                     (that's exactly what the Shapley necessity/sufficiency badge shows,
#                     and single-step flip-rate alone can't tell these two branches apart).
#   step 3 (GATE)  -> only half of a two-part conjunction -> stays SUCCESS: a fork that
#                     demonstrates a plausible-looking perturbation that DOESN'T flip the
#                     outcome, because it's necessary but not sufficient on its own.
_FORK_PLAN: list[tuple[int, StepRole, str]] = [
    (0, StepRole.ROOT, "root-cause fault at the tool-call step (necessary AND sufficient)"),
    (
        1,
        StepRole.ECHO,
        "downstream echo of the same fault, one step later (sufficient, not necessary)",
    ),
    (
        3,
        StepRole.GATE,
        "one half of a two-part conjunction fault (payload half absent -> stays benign)",
    ),
]


def _now() -> str:
    return datetime.now(UTC).isoformat()


def main() -> None:
    tape = build_competing_fault_tape()
    assert len(tape.exchanges) == N_TURNS >= 6, "demo tape must have >= 6 exchanges"

    # 1. Bit-exact replay receipt (offline, $0 — re-runs the same agent against the tape).
    replay_result = ReplayVerifier(tape, competing_fault_agent).verify()
    replay_data = verification_result_to_dict(replay_result)

    with tempfile.TemporaryDirectory() as tmp:
        db = TapeStore(str(Path(tmp) / "demo_store.db"))
        try:
            db.save_tape(tape, run_id=RUN_ID, created_at=_now())

            # 2. Three forked branches -> a real fork tree, not "no forked branches".
            for i, (step, role, desc) in enumerate(_FORK_PLAN):
                spec = BranchSpec(
                    divergence_step=step,
                    mutated_response=mutated_response_for(role),
                    mutation_desc=desc,
                )
                post_fork = RuleBasedTail(N_TURNS - (step + 1))
                branch = ForkEngine.fork(
                    tape, spec, competing_fault_agent, post_fork_transport=post_fork
                )
                db.save_branch(
                    parent_run_id=RUN_ID,
                    divergence_step=step,
                    delta_tape=branch.delta_tape,
                    mutation_desc=desc,
                    branch_digest=branch.branch_digest,
                    confinement_tier=branch.confinement_tier,
                    created_at=_now(),
                )
                print(f"branch {i}: diverge@{step} ({role.value}) -> {desc}")

            # 3. Causal blame (flip-rate + Wilson CIs) AND temporal Shapley
            #    (necessity/sufficiency) over the ROOT/ECHO scenario — the fixture that
            #    demonstrates exactly why naive flip-rate isn't enough (root and echo tie
            #    on flip-rate alone; Shapley correctly separates them).
            oracle = StringMatchOracle(success_re=r"SUCCESS", failure_re=r"FAIL")
            perturb_factory = make_perturb_factory(SCENARIO_ROOT_ECHO)

            blame_report = BlameEngine.rank(
                tape,
                competing_fault_agent,
                oracle,
                perturb_factory=perturb_factory,
                k=20,
                budget_usd=1_000_000.0,
            )
            shapley_report = BlameEngine.shapley_rank(
                tape,
                competing_fault_agent,
                oracle,
                perturb_factory=perturb_factory,
                k=5,
                m_samples=3,
                budget_usd=1_000_000.0,
            )
            db.save_blame_report(RUN_ID, blame_report, created_at=_now())
            db.save_shapley_report(RUN_ID, shapley_report, created_at=_now())

            blame_dict = {
                r.step_index: {
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
                for r in blame_report.results
            }
            shapley_dict = {
                r.step_index: {
                    "step_index": r.step_index,
                    "shapley_value": r.shapley_value,
                    "ci_lo": r.ci_lo,
                    "ci_hi": r.ci_hi,
                    "necessity": r.necessity,
                    "necessity_score": r.necessity_score,
                    "sufficiency": r.sufficiency,
                    "sufficiency_score": r.sufficiency_score,
                    "interpretation": r.interpretation,
                }
                for r in shapley_report.results
            }

            # 4. Pull everything the store now knows about this run back out, the same
            #    way cli.py's `report` command does when loaded via run_id.
            branches = db.list_branches(RUN_ID)
            causal_edges = db.causal_edges_for_run(RUN_ID)
            causal_closure = db.causal_closure(RUN_ID)
            branch_details: dict[str, dict] = {}
            for b in branches:
                branch_row = db.load_branch(b["branch_id"])
                branch_details[b["branch_id"]] = {
                    **_tape_to_data(branch_row["delta_tape"]),
                    "divergence_step": branch_row["divergence_step"],
                    "mutation_desc": branch_row["mutation_desc"],
                    "branch_digest": branch_row["branch_digest"],
                    "parent_run_id": branch_row["parent_run_id"],
                }
        finally:
            db.close()

    # 5. Per-model/per-tool cost profile.
    cost_profile_dict = cost_profile_to_dict(compute_cost_profile(tape))

    out = Path(__file__).parent / "demo_report.html"
    generate_report(
        tape,
        out,
        blame=blame_dict,
        replay=replay_data,
        branches=branches,
        causal_edges=causal_edges,
        branch_details=branch_details,
        shapley=shapley_dict,
        cost_profile=cost_profile_dict,
        causal_closure=causal_closure,
        run_id=RUN_ID,
    )

    print(f"\nparent outcome: {'SUCCESS' if blame_report.parent_outcome else 'FAIL'}")
    print("blame:", json.dumps(blame_dict, indent=2))
    print(
        f"\nbranches: {len(branches)}  causal_edges: {len(causal_edges)}  "
        f"causal_closure: {len(causal_closure)}"
    )
    bit_exact = replay_data["bit_exact"]
    matched, total = replay_data["matched"], replay_data["total"]
    print(f"replay: bit_exact={bit_exact} ({matched}/{total})")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
