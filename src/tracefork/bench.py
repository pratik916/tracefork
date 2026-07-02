"""Offline benchmark harness: runs `blame.py`'s coalition/temporal-Shapley
engine (`BlameEngine.shapley_rank`) over the long-tape competing-fault fixture
(`competing_faults.py`) and reports, per planted causal archetype, whether the
engine's (necessity, sufficiency) classification matches ground truth --
including the ONE case that does NOT resolve cleanly (see
`competing_faults.py`'s module docstring), reported honestly rather than
excluded.

This is contextualized against the published Who&When (ICML 2025) log-based
step-attribution anchor (~14.2% top-1) as scale-of-the-gap CONTEXT ONLY. It is
NOT a re-run of that benchmark: no external dataset is downloaded anywhere in
this repository (the offline/$0 invariant is non-negotiable -- see CLAUDE.md),
so there is no tracefork score on Who&When's actual data to report. The
`WHO_AND_WHEN_LOG_BASED_TOP1_ANCHOR` constant is the published number, cited,
not reproduced -- see README -> Validation scope for the exact citation and
scope of this comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .blame import ShapleyReport, wilson_ci
from .competing_faults import (
    SCENARIO_ALL,
    SCENARIO_GATE_PAYLOAD,
    SCENARIO_ROOT_ECHO,
    StepRole,
    run_shapley,
)

# Published anchor: Zhang et al., "Who&When: Uncover the Whodunit and When of
# LLM Multi-Agent Failures" (ICML 2025) reports ~14.2% top-1 accuracy for
# log-based (single-pass, no re-execution) step attribution on their
# multi-agent failure benchmark. Cited here as CONTEXT for the scale of the
# attribution gap tracefork's causal (fork-and-remeasure) approach targets --
# tracefork has not been run on Who&When's actual data (see module docstring).
WHO_AND_WHEN_LOG_BASED_TOP1_ANCHOR = 0.142

KNOWN_LIMITATION_CASES = frozenset({"gate_half_of_conjunction"})


@dataclass
class CaseResult:
    """One planted causal archetype's ground truth vs. the engine's reading."""

    name: str
    step_index: int
    role: StepRole
    expected_necessity: bool
    expected_sufficiency: bool
    actual_necessity: bool
    actual_sufficiency: bool
    resolved: bool
    note: str = ""


@dataclass
class BenchReport:
    cases: list[CaseResult] = field(default_factory=list)
    n_resolved: int = 0
    n_cases: int = 0
    accuracy: float = 0.0
    ci_lo: float = 0.0
    ci_hi: float = 0.0
    who_and_when_anchor: float = WHO_AND_WHEN_LOG_BASED_TOP1_ANCHOR

    def unexpected_failures(self) -> list[CaseResult]:
        """Cases that did NOT resolve and are NOT the one documented,
        known limitation -- a genuine regression if this is ever non-empty."""
        return [c for c in self.cases if not c.resolved and c.name not in KNOWN_LIMITATION_CASES]


def _case(
    name: str,
    report: ShapleyReport,
    step_index: int,
    role: StepRole,
    expected_necessity: bool,
    expected_sufficiency: bool,
    *,
    note: str = "",
) -> CaseResult:
    r = next(x for x in report.results if x.step_index == step_index)
    resolved = r.necessity == expected_necessity and r.sufficiency == expected_sufficiency
    return CaseResult(
        name=name,
        step_index=step_index,
        role=role,
        expected_necessity=expected_necessity,
        expected_sufficiency=expected_sufficiency,
        actual_necessity=r.necessity,
        actual_sufficiency=r.sufficiency,
        resolved=resolved,
        note=note,
    )


def run_bench(*, k: int = 3, m_samples: int = 2) -> BenchReport:
    """Run the three competing-fault scenarios and score all nine cases
    against their planted ground truth. See `competing_faults.py`'s module
    docstring for exactly why each case is expected to resolve the way it is."""
    root_echo = run_shapley(SCENARIO_ROOT_ECHO, k=k, m_samples=m_samples)
    gate_payload = run_shapley(SCENARIO_GATE_PAYLOAD, k=k, m_samples=m_samples)
    all_active = run_shapley(SCENARIO_ALL, k=k, m_samples=m_samples)

    cases = [
        _case("root", root_echo, 0, StepRole.ROOT, True, True),
        _case("downstream_echo", root_echo, 1, StepRole.ECHO, False, True),
        _case("neutral_decoy_root_echo_run", root_echo, 2, StepRole.NEUTRAL, False, False),
        _case(
            "gate_half_of_conjunction",
            gate_payload,
            3,
            StepRole.GATE,
            True,
            False,
            note=(
                "KNOWN LIMITATION: single-ordering temporal Shapley credits only the "
                "later-joining half of a symmetric AND-conjunction (see shapley_rank's "
                "docstring); step3 is genuinely necessary (removing it, with step4's "
                "fault still present, restores success) but the engine's own marginal "
                "at step3's coalition position is measured BEFORE step4 completes the "
                "AND, so necessity reads False here. See README -> Validation scope."
            ),
        ),
        _case("payload_completes_conjunction", gate_payload, 4, StepRole.PAYLOAD, True, False),
        _case("neutral_decoy_gate_payload_run", gate_payload, 2, StepRole.NEUTRAL, False, False),
        _case(
            "overdetermined_gate",
            all_active,
            3,
            StepRole.GATE,
            False,
            False,
            note=(
                "Correctly NOT necessary here: step0's independent fault already "
                "determines failure, so removing step3's fault alone would not "
                "restore success -- an over-determined run, correctly attributed."
            ),
        ),
        _case("overdetermined_payload", all_active, 4, StepRole.PAYLOAD, False, False),
        _case("root_under_competing_load", all_active, 0, StepRole.ROOT, True, True),
    ]

    n_resolved = sum(1 for c in cases if c.resolved)
    n = len(cases)
    accuracy = n_resolved / n if n else 0.0
    ci_lo, ci_hi = wilson_ci(n_resolved, n)
    return BenchReport(
        cases=cases, n_resolved=n_resolved, n_cases=n, accuracy=accuracy, ci_lo=ci_lo, ci_hi=ci_hi
    )
