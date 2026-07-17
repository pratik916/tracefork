"""Deterministic markdown narratives over already-computed `blame.py` results.

Four pure functions template `FlipRateResult`/`ShapleyResult`/`BlameReport`/
`ShapleyReport` dataclass fields into human-readable sentences and
markdown documents. No new computation happens here: every number rendered
was already computed by `BlameEngine.rank()`/`BlameEngine.shapley_rank()`,
and the causal-strength wording reuses each result's own
`.interpretation` string (produced by `blame.py`'s `_interpret()`) rather
than re-deriving the 0.7/0.3 thresholds a second time. Formatting is fixed
(`.0%` for rates/CI bounds/scores, `.3g` for q-values) so calling these
functions twice on the same input produces byte-identical output — safe to
diff, hash, or snapshot-test.

`explain_blame_report`/`explain_shapley_report` are wired additively into
`cli.py`'s `blame` command, writing a `blame_<run_id>.md` companion to the
existing `blame_<run_id>.json` report — no existing CLI output changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tracefork.blame import BlameReport, FlipRateResult, ShapleyReport, ShapleyResult


def explain_flip_result(r: FlipRateResult) -> str:
    """One deterministic sentence summarizing a single step's flip-rate result."""
    responsible = "responsible" if r.responsible else "not responsible"
    trustworthy = "trustworthy" if r.trustworthy else "untrustworthy"
    return (
        f"Step {r.step_index}: flip rate {r.flip_rate:.0%} "
        f"(CI [{r.ci_lo:.0%}, {r.ci_hi:.0%}], q={r.q_value:.3g}) — "
        f"{r.interpretation}; {responsible}, {trustworthy}."
    )


def explain_shapley_result(r: ShapleyResult) -> str:
    """One deterministic sentence summarizing a single step's Shapley result."""
    necessity = "necessary" if r.necessity else "not necessary"
    sufficiency = "sufficient" if r.sufficiency else "not sufficient"
    return (
        f"Step {r.step_index}: Shapley value {r.shapley_value:.0%} "
        f"(CI [{r.ci_lo:.0%}, {r.ci_hi:.0%}]) — {r.interpretation}; "
        f"{necessity} ({r.necessity_score:.0%}), {sufficiency} ({r.sufficiency_score:.0%})."
    )


def explain_blame_report(report: BlameReport) -> str:
    """Markdown doc: header, one bullet per step (report order), closing summary.

    The closing line names `report.responsible_set` but orders its members the
    same way `BlameReport.responsible()` already does (ascending q-value, then
    descending flip-rate) rather than `responsible_set`'s own ascending-index
    order, so the highlighted subset reads most-significant-first.
    """
    ci_pct = round(report.confidence * 100)
    lines = [
        "# Blame report",
        "",
        f"k={report.k}, {report.total_forks} total forks, "
        f"{report.ci_method.value} {ci_pct}% CI, FDR q≤{report.fdr_q}.",
        "",
    ]
    for r in report.results:
        lines.append(f"- {explain_flip_result(r)}")
    lines.append("")
    if report.responsible_set:
        ordered = report.responsible()
        steps = ", ".join(f"step {r.step_index} (q={r.q_value:.3g})" for r in ordered)
        lines.append(f"**Responsible set** (FDR q≤{report.fdr_q}): {steps}.")
    else:
        lines.append("**Responsible set**: no step cleared the significance bar.")
    return "\n".join(lines) + "\n"


def explain_shapley_report(report: ShapleyReport) -> str:
    """Markdown doc: header, one bullet per step (report order), closing summary.

    Mirrors `explain_blame_report`'s shape for `ShapleyReport`, which has no
    `responsible_set`/`responsible()` of its own: the closing line instead
    names the steps with `necessity=True`, ordered by descending Shapley
    value (then ascending step index) -- the closest analogue this report
    has to the blame report's FDR-controlled responsible set.
    """
    lines = [
        "# Shapley report",
        "",
        f"{report.n_permutation_samples} permutation samples, k={report.k}, "
        f"{report.total_forks} total forks.",
        "",
    ]
    for r in report.results:
        lines.append(f"- {explain_shapley_result(r)}")
    lines.append("")
    necessary = sorted(
        (r for r in report.results if r.necessity),
        key=lambda r: (-r.shapley_value, r.step_index),
    )
    if necessary:
        steps = ", ".join(f"step {r.step_index}" for r in necessary)
        lines.append(f"**Necessary steps**: {steps}.")
    else:
        lines.append("**Necessary steps**: no step cleared the necessity bar.")
    return "\n".join(lines) + "\n"
