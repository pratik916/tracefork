"""Offline tests for `query.py`'s pure `dispatch` function -- the
load-bearing coverage for this bead, independent of any CLI/interactive-loop
plumbing (see `test_cli_smoke.py` for the CLI wiring proof).

`query.py` adds zero new engine logic: every verb is a thin call-through to
an already-shipped, already-tested primitive (`report._tape_to_data`,
`diff.branch_diff`/`diff.tape_diff`, `store.py`'s
`causal_edges_for_run`/`cited_by`/`causal_closure`/`list_branches`). These
tests assert `dispatch()`'s text output agrees with what those primitives
return directly on the same fixtures.
"""

import anthropic
import httpx
import pytest

from tests.fakes import ScriptedFakeLLM, make_text_response
from tracefork.blame import BlameReport, CIMethod, FlipRateResult
from tracefork.diff import branch_diff, tape_diff
from tracefork.fork import BranchSpec, ForkEngine
from tracefork.query import QueryError, dispatch
from tracefork.report import _tape_to_data
from tracefork.store import TapeStore
from tracefork.tape import Tape
from tracefork.transport import TraceforkTransport

RESP_A = make_text_response("Response A")
RESP_B = make_text_response("Response B — mutated")
RESP_C = make_text_response("Response C — final turn")


def _conversation_agent(client: anthropic.Anthropic) -> str:
    """Two-turn agent; turn2's history embeds turn1's reply text (mirrors
    `tests/test_diff.py`'s fixture)."""
    r1 = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=100,
        messages=[{"role": "user", "content": "turn1"}],
    )
    first = r1.content[0].text
    r2 = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=100,
        messages=[
            {"role": "user", "content": "turn1"},
            {"role": "assistant", "content": first},
            {"role": "user", "content": "turn2"},
        ],
    )
    return r2.content[0].text


def _build_two_turn_tape() -> Tape:
    """Parent run: turn1 -> RESP_A, turn2 -> RESP_C (2 exchanges)."""
    fake = ScriptedFakeLLM([RESP_A, RESP_C])
    tape = Tape()
    transport = TraceforkTransport("record", tape, fake)
    client = anthropic.Anthropic(
        api_key="sk-ant-fake",
        http_client=httpx.Client(transport=transport),
        max_retries=0,
    )
    _conversation_agent(client)
    return tape


def _small_tape(tag: bytes = b"x") -> Tape:
    t = Tape(agent_name="w")
    t.append_exchange(b"req-" + tag, b"resp-" + tag)
    return t


def _blame_report(steps: list[tuple[int, bool]]) -> BlameReport:
    results = [
        FlipRateResult(
            step_index=idx,
            flip_rate=0.9 if responsible else 0.1,
            ci_lo=0.5 if responsible else 0.0,
            ci_hi=1.0 if responsible else 0.3,
            flips=9 if responsible else 1,
            trials=10,
            valid_trials=10,
            p_value=0.01 if responsible else 0.9,
            q_value=0.02 if responsible else 0.9,
            responsible=responsible,
        )
        for idx, responsible in steps
    ]
    return BlameReport(
        results=results, k=10, total_forks=len(results) * 10, ci_method=CIMethod.WILSON
    )


# ── state ────────────────────────────────────────────────────────────────


def test_dispatch_state_returns_shaped_exchange_json(tmp_path):
    tape = _build_two_turn_tape()
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        store.save_tape(tape, run_id="run-1")
        expected = _tape_to_data(tape)["exchanges"][0]

        result = dispatch(store, "state run-1 0")

        assert expected["role"] in result
        assert repr(expected["preview"]) in result
    finally:
        store.close()


def test_dispatch_state_out_of_range_step_raises_query_error(tmp_path):
    tape = _build_two_turn_tape()
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        store.save_tape(tape, run_id="run-1")

        with pytest.raises(QueryError, match=str(len(tape.exchanges))):
            dispatch(store, "state run-1 99")
    finally:
        store.close()


# ── diff ─────────────────────────────────────────────────────────────────


def test_dispatch_diff_branch_mode_matches_diff_module_output(tmp_path):
    parent_tape = _build_two_turn_tape()
    spec = BranchSpec(divergence_step=0, mutated_response=RESP_B)
    branch = ForkEngine.fork(
        parent_tape, spec, _conversation_agent, post_fork_transport=ScriptedFakeLLM([RESP_C])
    )

    store = TapeStore(str(tmp_path / "store.db"))
    try:
        parent_id = store.save_tape(parent_tape, run_id="parent-1")
        branch_id = store.save_branch(
            parent_run_id=parent_id,
            divergence_step=branch.divergence_step,
            delta_tape=branch.delta_tape,
            mutation_desc=branch.mutation_desc,
            branch_digest=branch.branch_digest,
        )

        expected = branch_diff(parent_tape, branch)

        result = dispatch(store, f"diff {parent_id} {branch_id}")

        assert expected.changed_steps  # sanity: this fixture does diverge
        for step in expected.steps:
            status = "FAIL" if step.changed else "PASS"
            assert f"[{status}] step {step.step_index}" in result
    finally:
        store.close()


def test_dispatch_diff_step_mode_compares_two_independent_tapes(tmp_path):
    tape_a = _small_tape(b"a")
    tape_b = _small_tape(b"b")
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        store.save_tape(tape_a, run_id="run-a")
        store.save_tape(tape_b, run_id="run-b")

        expected = tape_diff(tape_a, tape_b, 0)

        result = dispatch(store, "diff run-a run-b --step 0")

        assert expected.changed  # different tags -> genuinely differing bodies
        status = "FAIL" if expected.changed else "PASS"
        assert f"[{status}] step {expected.step_index}" in result
    finally:
        store.close()


# ── causes ───────────────────────────────────────────────────────────────


def test_dispatch_causes_filters_by_step_and_lists_citers(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        store.save_tape(_small_tape(), run_id="run-1")
        store.save_blame_report("run-1", _blame_report([(0, False), (1, True)]))
        branch_id = store.save_branch(
            parent_run_id="run-1", divergence_step=1, delta_tape=_small_tape(b"branch")
        )

        result = dispatch(store, "causes run-1 1")

        assert "responsible=True" in result
        assert "responsible=False" not in result  # step 0's edge is filtered out
        assert branch_id in result
    finally:
        store.close()


def test_dispatch_causes_closure_flag_walks_fork_graph(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        store.save_tape(_small_tape(b"a"), run_id="run-a")
        store.save_blame_report("run-a", _blame_report([(0, True), (1, False)]))

        branch_tape = _small_tape(b"branch")
        branch_id = store.save_branch(
            parent_run_id="run-a", divergence_step=0, delta_tape=branch_tape
        )
        store.save_tape(branch_tape, run_id=branch_id)
        store.save_blame_report(branch_id, _blame_report([(3, True)]))

        expected = store.causal_closure("run-a")
        assert {(e["run_id"], e["step_index"]) for e in expected} == {
            ("run-a", 0),
            (branch_id, 3),
        }

        result = dispatch(store, "causes run-a --closure")

        for edge in expected:
            assert f"{edge['run_id']}:{edge['step_index']}" in result
    finally:
        store.close()


# ── tree ─────────────────────────────────────────────────────────────────


def test_dispatch_tree_lists_branch_summary_fields(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        store.save_tape(_small_tape(), run_id="run-1")
        branch_id = store.save_branch(
            parent_run_id="run-1",
            divergence_step=2,
            delta_tape=_small_tape(b"branch"),
            mutation_desc="mutated response",
            created_at="2026-01-01T00:00:00",
            branch_digest="digest-abc123",
        )

        expected = store.list_branches("run-1")
        assert len(expected) == 1

        result = dispatch(store, "tree run-1")

        row = expected[0]
        assert row["branch_id"] == branch_id
        assert str(row["divergence_step"]) in result
        assert row["mutation_desc"] in result
        assert row["created_at"] in result
        assert row["branch_digest"] in result
        assert branch_id in result
    finally:
        store.close()


# ── errors ───────────────────────────────────────────────────────────────


def test_dispatch_unknown_verb_and_unknown_run_id_raise_query_error(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        with pytest.raises(QueryError, match="state.*diff.*causes.*tree"):
            dispatch(store, "bogus x")

        with pytest.raises(QueryError) as excinfo:
            dispatch(store, "state no-such-run 0")
        assert "no-such-run" in str(excinfo.value)
        assert not isinstance(excinfo.value, KeyError)
    finally:
        store.close()
