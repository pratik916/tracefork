"""Settlement-diff export tests -- offline, no API keys.

`settlement.py` decodes a winning fork's post-divergence
`delta_tape.tool_exchanges` into `SettlementOp`s and renders them as a
portable, in-toto-Statement-shaped JSON artifact. These tests exercise both
halves of the dual-input contract (`diff.py`'s pattern: a live `fork.Branch`
vs. a plain store-reloaded `Tape` + `divergence_step`), the empty/degenerate
cases, the JSON round-trip, and a `Tape.digest()` byte-stability regression
guard (this module must never feed anything into `digest()`).
"""

from __future__ import annotations

import json

from tracefork.fork import Branch
from tracefork.settlement import (
    SETTLEMENT_DIFF_KIND,
    SettlementDiff,
    SettlementOp,
    branch_settlement_diff,
    to_settlement_json,
)
from tracefork.tape import Tape
from tracefork.tools import make_request_frame, make_result_frame, make_tool_call_frame


def _build_parent_tape() -> Tape:
    tape = Tape()
    tape.append_exchange(b'{"model":"claude-sonnet-4-6"}', b'{"text":"hi"}')
    return tape


def _build_delta_tape_with_tool_calls(parent: Tape) -> Tape:
    """A delta_tape carrying two synthetic `tools/call` exchanges -- built
    from `tools.py`'s own frame constructors, never a hand-rolled JSON
    string, so this test exercises the real wire shape."""
    delta = Tape(boundary=parent.boundary, agent_name=parent.agent_name)
    delta.append_tool_exchange(
        make_tool_call_frame(1, "read_file", {"path": "/etc/hosts"}),
        make_result_frame(1, {"content": "127.0.0.1 localhost"}),
    )
    delta.append_tool_exchange(
        make_tool_call_frame(2, "write_file", {"path": "/tmp/out.txt", "content": "done"}),
        make_result_frame(2, {"ok": True}),
    )
    return delta


def _expected_ops() -> tuple[SettlementOp, ...]:
    return (
        SettlementOp(
            tool_name="read_file",
            arguments={"path": "/etc/hosts"},
            result={"content": "127.0.0.1 localhost"},
            step_index=0,
        ),
        SettlementOp(
            tool_name="write_file",
            arguments={"path": "/tmp/out.txt", "content": "done"},
            result={"ok": True},
            step_index=1,
        ),
    )


# ── branch_settlement_diff: live fork.Branch input ───────────────────────────


def test_branch_settlement_diff_live_branch_ops_match_in_order():
    parent_tape = _build_parent_tape()
    delta_tape = _build_delta_tape_with_tool_calls(parent_tape)
    branch = Branch(
        parent_tape=parent_tape,
        divergence_step=0,
        delta_tape=delta_tape,
        branch_digest="branchdigest123",
    )

    diff = branch_settlement_diff(parent_tape, branch)

    assert isinstance(diff, SettlementDiff)
    assert diff.parent_tape_digest == parent_tape.digest()
    assert diff.branch_digest == "branchdigest123"
    assert diff.divergence_step == 0
    assert diff.ops == _expected_ops()


# ── branch_settlement_diff: plain store-reloaded Tape input ──────────────────


def test_branch_settlement_diff_plain_tape_produces_byte_identical_ops():
    parent_tape = _build_parent_tape()
    delta_tape = _build_delta_tape_with_tool_calls(parent_tape)

    diff = branch_settlement_diff(parent_tape, delta_tape, divergence_step=0)

    assert diff.ops == _expected_ops()
    # No branch object here -- branch_digest defaults to "" unless supplied.
    assert diff.branch_digest == ""


def test_branch_settlement_diff_plain_tape_accepts_explicit_branch_digest():
    parent_tape = _build_parent_tape()
    delta_tape = _build_delta_tape_with_tool_calls(parent_tape)

    diff = branch_settlement_diff(
        parent_tape, delta_tape, divergence_step=0, branch_digest="stored-digest"
    )

    assert diff.branch_digest == "stored-digest"
    assert diff.ops == _expected_ops()


def test_branch_settlement_diff_plain_tape_without_divergence_step_raises():
    parent_tape = _build_parent_tape()
    delta_tape = _build_delta_tape_with_tool_calls(parent_tape)

    try:
        branch_settlement_diff(parent_tape, delta_tape)
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "divergence_step" in str(exc)


# ── empty / degenerate tool_exchanges ─────────────────────────────────────────


def test_branch_settlement_diff_empty_tool_exchanges_yields_empty_ops():
    parent_tape = _build_parent_tape()
    delta_tape = Tape(boundary=parent_tape.boundary, agent_name=parent_tape.agent_name)

    diff = branch_settlement_diff(parent_tape, delta_tape, divergence_step=0)

    assert diff.ops == ()


def test_branch_settlement_diff_skips_non_tool_call_frames_without_crashing():
    """A tool_exchanges entry that isn't a `tools/call` request (e.g. a
    plain JSON-RPC result frame with no `method`) is skipped, not a crash --
    mirrors `effects.py`'s own best-effort decode discipline."""
    parent_tape = _build_parent_tape()
    delta_tape = Tape(boundary=parent_tape.boundary, agent_name=parent_tape.agent_name)
    delta_tape.append_tool_exchange(
        make_request_frame(1, "not_a_tool_call", {"foo": "bar"}),
        make_result_frame(1, {"ignored": True}),
    )
    delta_tape.append_tool_exchange(
        make_tool_call_frame(2, "read_file", {"path": "/etc/hosts"}),
        make_result_frame(2, {"content": "127.0.0.1 localhost"}),
    )

    diff = branch_settlement_diff(parent_tape, delta_tape, divergence_step=0)

    assert len(diff.ops) == 1
    assert diff.ops[0].tool_name == "read_file"
    assert diff.ops[0].step_index == 1  # position within tool_exchanges, unshifted


# ── to_settlement_json ────────────────────────────────────────────────────────


def test_to_settlement_json_round_trips_and_preserves_every_field():
    parent_tape = _build_parent_tape()
    delta_tape = _build_delta_tape_with_tool_calls(parent_tape)
    diff = branch_settlement_diff(
        parent_tape, delta_tape, divergence_step=3, branch_digest="abc123"
    )

    payload = to_settlement_json(diff)
    round_tripped = json.loads(json.dumps(payload))

    assert round_tripped["kind"] == SETTLEMENT_DIFF_KIND == "tracefork.settlement_diff/v1"
    assert round_tripped["subject"]["parent_tape_digest"] == diff.parent_tape_digest
    assert round_tripped["subject"]["branch_digest"] == "abc123"
    assert round_tripped["predicate"]["divergence_step"] == 3
    ops_json = round_tripped["predicate"]["ops"]
    assert len(ops_json) == len(diff.ops)
    for op_json, op in zip(ops_json, diff.ops, strict=True):
        assert op_json["tool_name"] == op.tool_name
        assert op_json["arguments"] == op.arguments
        assert op_json["result"] == op.result
        assert op_json["step_index"] == op.step_index


def test_to_settlement_json_empty_ops_round_trips_cleanly():
    parent_tape = _build_parent_tape()
    delta_tape = Tape(boundary=parent_tape.boundary, agent_name=parent_tape.agent_name)
    diff = branch_settlement_diff(parent_tape, delta_tape, divergence_step=0)

    round_tripped = json.loads(json.dumps(to_settlement_json(diff)))

    assert round_tripped["predicate"]["ops"] == []


# ── digest()-byte-stability regression guard ─────────────────────────────────


def test_branch_settlement_diff_never_changes_parent_tape_digest():
    """This module must never feed anything into `Tape.digest()` -- a
    regression here would silently break the hash chain byte-stability
    invariant every existing tape/replay/fork test depends on."""
    parent_tape = _build_parent_tape()
    delta_tape = _build_delta_tape_with_tool_calls(parent_tape)
    digest_before = parent_tape.digest()

    branch_settlement_diff(parent_tape, delta_tape, divergence_step=0)

    assert parent_tape.digest() == digest_before
