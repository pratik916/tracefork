"""Parameterized causal-archetype generator tests — all offline, zero API keys.

See `tracefork/archetypes.py`'s module docstring for the full design rationale
(why each archetype is expected to resolve the way it does) and
`tests/test_competing_faults.py` for the fixed-fixture precedent this
generalizes: same `_by_step` helper, same "never grade the tape's last
exchange" invariant, same exact-value assertions (every fake transport here is
fully deterministic, so there is no trial-to-trial noise to tolerate).
"""

import pytest

from tracefork.archetypes import (
    OR_BOTH,
    OR_CAUSE_A,
    OR_CAUSE_B,
    run_long_relay,
    run_n_way_conjunction,
    run_or_redundancy,
)


def _by_step(report, step_index: int):
    return next(r for r in report.results if r.step_index == step_index)


# ── run_or_redundancy: two independently-sufficient OR-causes ──────────────


def test_or_redundancy_earlier_cause_is_necessary_and_sufficient():
    result = run_or_redundancy(pos_a=0, pos_b=3, n_turns=6, k=3, m_samples=2)
    cause_a = _by_step(result.report, 0)
    assert cause_a.necessity is True
    assert cause_a.sufficiency is True
    assert cause_a.shapley_value == 1.0
    assert result.report.top().step_index == 0
    assert result.matches_expected()


def test_or_redundancy_later_cause_is_sufficient_not_necessary():
    """`pos_b`'s marker ties `pos_a` under naive single-step flip-rate (both
    independently "sufficient": forcing either one alone, with the rest
    clean, flips the run) -- but temporal-Shapley must NOT credit it as
    necessary once `pos_a`'s fault is already in the coalition."""
    result = run_or_redundancy(pos_a=0, pos_b=3, n_turns=6, k=3, m_samples=2)
    cause_b = _by_step(result.report, 3)
    assert cause_b.necessity is False
    assert cause_b.sufficiency is True
    assert cause_b.shapley_value == 0.0
    assert result.matches_expected()


def test_or_redundancy_isolated_scenario_each_independently_sufficient():
    """With only ONE cause lit up at a time (the other's marker never even
    appears in the tape), that ONE position reads BOTH necessity=True and
    sufficiency=True -- proving each cause is independently sufficient on
    its own merit, not merely because the other happens to be present."""
    result_a = run_or_redundancy(pos_a=0, pos_b=3, n_turns=6, active=OR_CAUSE_A, k=3, m_samples=2)
    cause_a = _by_step(result_a.report, 0)
    assert cause_a.necessity is True
    assert cause_a.sufficiency is True
    other_b = _by_step(result_a.report, 3)
    assert other_b.necessity is False
    assert other_b.sufficiency is False
    assert result_a.matches_expected()

    result_b = run_or_redundancy(pos_a=0, pos_b=3, n_turns=6, active=OR_CAUSE_B, k=3, m_samples=2)
    cause_b = _by_step(result_b.report, 3)
    assert cause_b.necessity is True
    assert cause_b.sufficiency is True
    other_a = _by_step(result_b.report, 0)
    assert other_a.necessity is False
    assert other_a.sufficiency is False
    assert result_b.matches_expected()


def test_or_redundancy_rejects_non_activatable_roles():
    with pytest.raises(ValueError):
        run_or_redundancy(pos_a=0, pos_b=3, n_turns=6, active=frozenset({"not-a-cause"}))


def test_or_redundancy_both_is_the_default_active_set():
    assert OR_BOTH == OR_CAUSE_A | OR_CAUSE_B


# ── run_n_way_conjunction: parameterized k-part AND ────────────────────────


@pytest.mark.parametrize("arity", [2, 3, 4, 5], ids=lambda a: f"arity={a}")
def test_n_way_conjunction_only_last_part_is_necessary(arity):
    result = run_n_way_conjunction(arity, k=3, m_samples=2)
    last = _by_step(result.report, arity - 1)
    assert last.necessity is True
    assert last.sufficiency is False
    assert last.shapley_value == 1.0
    assert result.report.top().step_index == arity - 1
    for i in range(arity - 1):
        earlier = _by_step(result.report, i)
        assert earlier.necessity is False, f"part_{i} should not be necessary"
    assert result.matches_expected()


def test_n_way_conjunction_earlier_parts_never_necessary_or_sufficient():
    """DOCUMENTED LIMITATION (not a bug), scaled up from
    `test_competing_faults.py::test_temporal_order_undercredits_the_earlier_half_of_a_conjunction`'s
    arity=2 case: every part except the last-joining one reads
    necessity=False despite each being genuinely necessary for the full AND,
    and NO part alone (single-step forcing, rest clean) is ever sufficient
    for a >= 2-part AND."""
    result = run_n_way_conjunction(4, k=3, m_samples=2)
    for i in range(3):
        r = _by_step(result.report, i)
        assert r.necessity is False
        assert r.sufficiency is False
        assert r.shapley_value == 0.0


def test_n_way_conjunction_rejects_arity_below_two():
    with pytest.raises(ValueError):
        run_n_way_conjunction(1)


# ── run_long_relay: root propagated through a parameterized chain ─────────


@pytest.mark.parametrize("n_relay", [1, 5, 10], ids=lambda n: f"n_relay={n}")
def test_long_relay_root_invariant_to_chain_length(n_relay):
    result = run_long_relay(n_relay, k=3, m_samples=2)
    root = _by_step(result.report, 0)
    assert root.necessity is True
    assert root.sufficiency is True
    assert root.shapley_value == 1.0
    assert result.report.top().step_index == 0
    assert result.matches_expected()


def test_long_relay_decoys_are_true_negatives():
    result = run_long_relay(5, k=3, m_samples=2)
    for i in range(1, 6):
        r = _by_step(result.report, i)
        assert r.necessity is False, f"relay step {i} should not be necessary"
        assert r.sufficiency is False, f"relay step {i} should not be sufficient"
        assert r.shapley_value == 0.0


def test_long_relay_rejects_negative_chain_length():
    with pytest.raises(ValueError):
        run_long_relay(-1)


# ── shared "never grade the tape's last exchange" invariant ───────────────


def test_archetype_rejects_role_position_at_final_slot():
    """Mirrors `competing_faults.py`'s documented rule that the tape's LAST
    exchange is never a valid role position: placing a cause at the final
    slot must raise before any tape is even recorded."""
    with pytest.raises(ValueError):
        run_or_redundancy(pos_a=0, pos_b=5, n_turns=6, k=3, m_samples=2)
