"""N-way concurrent-sibling true-negative-discrimination fixture tests — all
offline, zero API keys.

See `tracefork/concurrent_validate.py`'s module docstring for the full design
rationale: `competing_faults.py`'s 2-member GATE/PAYLOAD conjunction is
symmetric (both members guilty by design) and so cannot prove the coalition/
temporal-Shapley engine correctly discriminates against an INNOCENT sibling
sitting in the same unordered batch as a guilty one. This module's fixture
plants exactly one fault among `n_branches` genuinely-concurrent siblings and
sweeps every possible guilty position.
"""

import pytest

from tracefork.blame import StringMatchOracle
from tracefork.concurrent_validate import (
    ConcurrentValidationReport,
    build_multi_branch_tape,
    make_single_branch_perturb_factory,
    run_concurrent_branch_validation,
    run_shapley_multi_branch,
    run_shapley_negative_control,
)


def _by_step(report, step_index: int):
    return next(r for r in report.results if r.step_index == step_index)


# ── the parent (clean) tape ─────────────────────────────────────────────────


def test_multi_branch_tape_records_a_real_batch_for_every_sibling():
    """`build_multi_branch_tape` must record its concurrency through the REAL
    `AsyncTraceforkTransport` machinery, not a hand-constructed
    `tape.async_batches` -- one setup exchange, three concurrent branch
    exchanges, a merge exchange, and a final exchange."""
    tape = build_multi_branch_tape(n_branches=3)
    assert len(tape.exchanges) == 6
    assert tape.async_batches == [[1, 2, 3]]


def test_multi_branch_parent_tape_grades_success():
    tape = build_multi_branch_tape(n_branches=3)
    oracle = StringMatchOracle(success_re=r"SUCCESS", failure_re=r"FAIL")
    assert oracle.grade(tape.exchanges[-1][1].decode(errors="replace")) is True


def test_make_single_branch_perturb_factory_rejects_out_of_range_step():
    with pytest.raises(ValueError):
        make_single_branch_perturb_factory(0, n_branches=3)
    with pytest.raises(ValueError):
        make_single_branch_perturb_factory(4, n_branches=3)


# ── the crux: true-negative discrimination among concurrent siblings ───────


@pytest.mark.parametrize("faulty_step", [1, 2, 3])
def test_guilty_sibling_is_ranked_and_innocent_siblings_are_not(faulty_step):
    """For every possible guilty-branch position, `shapley_rank` (with the
    tape's real `async_batches` forwarded) must rank the guilty sibling #1
    with `necessity=True`, and every OTHER sibling in the SAME unordered batch
    must read `necessity=False` -- the true-negative-discrimination proof the
    symmetric 2-way GATE/PAYLOAD fixture cannot give."""
    report = run_shapley_multi_branch(faulty_step, n_branches=3, k=3, m_samples=2)

    top = report.top()
    assert top is not None
    assert top.step_index == faulty_step

    guilty = _by_step(report, faulty_step)
    assert guilty.necessity is True

    for other_step in (1, 2, 3):
        if other_step == faulty_step:
            continue
        innocent = _by_step(report, other_step)
        assert innocent.necessity is False, f"step{other_step} should not be necessary"


# ── the validation runner ───────────────────────────────────────────────────


def test_run_concurrent_branch_validation_reaches_perfect_top1_precision():
    report = run_concurrent_branch_validation(n_branches=3, k=3, m_samples=2)
    assert isinstance(report, ConcurrentValidationReport)
    assert report.n_branches == 3
    assert report.n_runs == 3
    assert report.top1_correct == 3
    assert report.top1_precision == 1.0


def test_negative_control_yields_no_necessity_and_near_zero_shapley():
    """No marker anywhere: every step, including every batch member, must
    read `necessity=False` with a near-zero `shapley_value` -- otherwise the
    validation runner's perfect precision score would be meaningless."""
    report = run_shapley_negative_control(n_branches=3, k=3, m_samples=2)
    for step_index in range(len(report.results)):
        r = _by_step(report, step_index)
        assert r.necessity is False, f"step{step_index} should not be necessary"
        assert r.shapley_value == pytest.approx(0.0, abs=1e-9)


def test_run_concurrent_branch_validation_negative_control_max_shapley_near_zero():
    report = run_concurrent_branch_validation(n_branches=3, k=3, m_samples=2)
    assert report.negative_control_max_shapley == pytest.approx(0.0, abs=1e-9)
