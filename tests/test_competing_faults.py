"""Long-tape competing-fault fixture tests — all offline, zero API keys.

See `tracefork/competing_faults.py`'s module docstring for the full design
rationale (why each scenario is expected to resolve the way it does, and
exactly which case is a documented, honest limitation rather than a bug).
"""

import pytest

from tracefork.competing_faults import (
    GATE_MARKER,
    N_TURNS,
    PAYLOAD_MARKER,
    ROOT_MARKER,
    SCENARIO_ALL,
    SCENARIO_GATE_PAYLOAD,
    SCENARIO_ROOT_ECHO,
    StepRole,
    _fails,
    build_competing_fault_tape,
    build_concurrent_gate_payload_tape,
    make_perturb_factory,
    run_shapley,
    run_shapley_concurrent,
)


def _by_step(report, step_index: int):
    return next(r for r in report.results if r.step_index == step_index)


# ── the failure rule itself ─────────────────────────────────────────────────


def test_fails_rule_root_marker_alone_triggers():
    assert _fails(b"... " + ROOT_MARKER + b" ...") is True


def test_fails_rule_gate_alone_does_not_trigger():
    assert _fails(b"... " + GATE_MARKER + b" ...") is False


def test_fails_rule_payload_alone_does_not_trigger():
    assert _fails(b"... " + PAYLOAD_MARKER + b" ...") is False


def test_fails_rule_gate_and_payload_together_trigger():
    assert _fails(b"... " + GATE_MARKER + b" ... " + PAYLOAD_MARKER + b" ...") is True


def test_fails_rule_no_markers_never_triggers():
    assert _fails(b"nothing interesting here") is False


def test_make_perturb_factory_rejects_non_activatable_roles():
    with pytest.raises(ValueError):
        make_perturb_factory(frozenset({StepRole.NEUTRAL}))


# ── the parent (clean) tape ─────────────────────────────────────────────────


def test_parent_tape_has_seven_exchanges_and_grades_success():
    tape = build_competing_fault_tape()
    assert len(tape.exchanges) == N_TURNS
    from tracefork.blame import StringMatchOracle

    oracle = StringMatchOracle(success_re=r"SUCCESS", failure_re=r"FAIL")
    last_resp = tape.exchanges[-1][1]
    # The provider adapter round-trips the response; just check the marker of
    # success survives the record.
    assert oracle.grade(last_resp.decode(errors="replace")) is True


# ── SCENARIO_ROOT_ECHO: root vs. downstream echo, on a long/noisy tape ─────
#
# This re-demonstrates test_blame.py::test_temporal_shapley_discriminates_root_from_echo
# (a 2-step tape) on a 7-step tape with unrelated decoy steps around the pair,
# proving the discrimination isn't an artifact of a trivially short fixture.


def test_root_is_necessary_and_sufficient():
    report = run_shapley(SCENARIO_ROOT_ECHO, k=3, m_samples=2)
    root = _by_step(report, 0)
    assert root.necessity is True
    assert root.sufficiency is True
    assert root.shapley_value == 1.0


def test_downstream_echo_is_sufficient_but_not_necessary():
    """step1 ties step0 under naive single-step flip-rate (both independently
    "sufficient": forcing either one alone, with the rest clean, flips the
    run) -- exactly the tie `rank()` cannot break. Temporal-Shapley must NOT
    blame it as the root: once step0's fault is already in the coalition,
    step1 contributes nothing further."""
    report = run_shapley(SCENARIO_ROOT_ECHO, k=3, m_samples=2)
    echo = _by_step(report, 1)
    assert echo.sufficiency is True
    assert echo.necessity is False
    assert echo.shapley_value == 0.0
    assert report.top().step_index == 0  # the root wins, not the echo


def test_decoy_step_is_a_true_negative_amid_root_and_echo():
    report = run_shapley(SCENARIO_ROOT_ECHO, k=3, m_samples=2)
    decoy = _by_step(report, 2)
    assert decoy.necessity is False
    assert decoy.sufficiency is False
    assert decoy.shapley_value == 0.0


# ── SCENARIO_GATE_PAYLOAD: a genuine two-part AND-conjunction ─────────────


def test_payload_is_necessary_not_sufficient():
    """step4 alone (clean prefix) never flips the outcome (sufficiency False)
    -- the AND-gate needs step3's half too -- but once step3's half is
    already in the coalition, adding step4 completes the AND and flips it
    (necessity True): the fixture's clean necessary-not-sufficient case."""
    report = run_shapley(SCENARIO_GATE_PAYLOAD, k=3, m_samples=2)
    payload = _by_step(report, 4)
    assert payload.sufficiency is False
    assert payload.necessity is True
    assert report.top().step_index == 4


def test_gate_alone_is_not_sufficient():
    report = run_shapley(SCENARIO_GATE_PAYLOAD, k=3, m_samples=2)
    gate = _by_step(report, 3)
    assert gate.sufficiency is False


def test_temporal_order_undercredits_the_earlier_half_of_a_conjunction():
    """DOCUMENTED LIMITATION (not a bug to silently fix): step3 (GATE) is
    genuinely, causally necessary in this scenario -- with step4's fault held
    fixed, reverting step3's fault restores success, since the AND-gate needs
    both. But `shapley_rank`'s necessity check is a TEMPORAL-ORDER-RESTRICTED
    Shapley walk with exactly one valid permutation (see its docstring): a
    step's marginal is measured at ITS OWN coalition position, which for
    step3 is BEFORE step4 (index 4 > 3) ever joins. So step3's own marginal
    contribution is 0 (the AND isn't complete yet) and `necessity` reads
    False here, even though a full accounting would call it necessary too.

    This pins the CURRENT, real behaviour so a future change to the engine
    that silently regresses (or silently "fixes" without deliberate design)
    is visible in a diff, rather than reasoning about it only in prose. See
    `competing_faults.py`'s module docstring and README -> Validation scope.
    """
    report = run_shapley(SCENARIO_GATE_PAYLOAD, k=3, m_samples=2)
    gate = _by_step(report, 3)
    assert gate.necessity is False
    assert gate.shapley_value == 0.0


def test_decoy_steps_are_true_negatives_amid_the_conjunction():
    report = run_shapley(SCENARIO_GATE_PAYLOAD, k=3, m_samples=2)
    for step_index in (0, 1, 2, 5):
        r = _by_step(report, step_index)
        assert r.necessity is False, f"step{step_index} should not be necessary"
        assert r.sufficiency is False, f"step{step_index} should not be sufficient"


# ── SCENARIO_ALL: every fault live at once — an over-determined run ───────


def test_root_still_correctly_attributed_under_competing_load():
    """With ROOT, ECHO, GATE, and PAYLOAD all live simultaneously, the run is
    over-determined (ROOT alone already causes failure). The engine must
    still isolate ROOT as the responsible cause, not spread blame across
    every technically-present fault."""
    report = run_shapley(SCENARIO_ALL, k=3, m_samples=2)
    root = _by_step(report, 0)
    assert root.necessity is True
    assert root.sufficiency is True
    assert report.top().step_index == 0


def test_gate_and_payload_correctly_not_necessary_when_overdetermined():
    """Correct (not a limitation): once ROOT's fault alone already guarantees
    failure, removing GATE's or PAYLOAD's fault alone does not restore
    success (ROOT's fault is untouched) -- so neither reads `necessity=True`
    here, and that is the RIGHT answer for this specific run, not a miss."""
    report = run_shapley(SCENARIO_ALL, k=3, m_samples=2)
    for step_index in (3, 4):
        r = _by_step(report, step_index)
        assert r.necessity is False


# ── genuinely-concurrent GATE/PAYLOAD (tracefork-bge.10) ────────────────────


def test_concurrent_tape_records_a_real_batch_for_gate_and_payload():
    """`build_concurrent_gate_payload_tape` must record its concurrency
    through the REAL `AsyncTraceforkTransport` machinery, not a hand-
    constructed `tape.async_batches` -- steps 3 (GATE) and 4 (PAYLOAD),
    recorded in that completion order (see the module's `_CONCURRENT_DELAYS`)."""
    tape = build_concurrent_gate_payload_tape()
    assert len(tape.exchanges) == N_TURNS
    assert tape.async_batches == [[3, 4]]


def test_concurrent_parent_tape_grades_success():
    tape = build_concurrent_gate_payload_tape()
    from tracefork.blame import StringMatchOracle

    oracle = StringMatchOracle(success_re=r"SUCCESS", failure_re=r"FAIL")
    assert oracle.grade(tape.exchanges[-1][1].decode(errors="replace")) is True


def test_concurrent_gate_and_payload_both_resolve_necessary():
    """The fix: forwarding the tape's REAL `async_batches` into `shapley_rank`
    makes both halves of the conjunction converge to necessity=True, unlike
    `test_temporal_order_undercredits_the_earlier_half_of_a_conjunction`'s
    sequential-tape limitation, which this leaves completely untouched."""
    report = run_shapley_concurrent(SCENARIO_GATE_PAYLOAD, k=3, m_samples=2)
    gate = _by_step(report, 3)
    payload = _by_step(report, 4)
    assert gate.necessity is True
    assert payload.necessity is True
    assert gate.sufficiency is False
    assert payload.sufficiency is False
    assert gate.shapley_value == pytest.approx(0.5)
    assert payload.shapley_value == pytest.approx(0.5)


def test_concurrent_decoys_still_true_negatives():
    report = run_shapley_concurrent(SCENARIO_GATE_PAYLOAD, k=3, m_samples=2)
    for step_index in (0, 1, 2, 5):
        r = _by_step(report, step_index)
        assert r.necessity is False, f"step{step_index} should not be necessary"
        assert r.sufficiency is False, f"step{step_index} should not be sufficient"


def test_sequential_gate_payload_limitation_is_unaffected_by_the_fix():
    """`run_shapley` (the untouched, fully-sequential fixture) must keep
    reproducing the exact documented limitation -- the fix is additive, not a
    silent change to the existing scenario."""
    report = run_shapley(SCENARIO_GATE_PAYLOAD, k=3, m_samples=2)
    gate = _by_step(report, 3)
    payload = _by_step(report, 4)
    assert gate.necessity is False
    assert payload.necessity is True
