"""Tests for `locate.py` — the offline-checkable evidence locator
(tracefork-bge.62): `locate_value`'s single-tape substring scan,
`locate_in_lineage`'s BFS over a run's fork lineage (direct branch and
fork-of-fork depths, `follow_lineage=False` scoping), the offline-checkable
blob-hash property, and the `tracefork locate` CLI surface.

The CLI tests exercise `tracefork.cli.app` directly via `typer.testing
.CliRunner`. `cli.py` is a shared, orchestrator-owned file in this bead's
workflow (this bead only hands off ready-to-paste command code, see the
bead's `cli_command` result field) — until that command is wired in, the
CLI tests below skip themselves (rather than failing on a `locate.py`-side
non-issue) via `_LOCATE_CLI_WIRED`, and self-heal the moment `cli.py` gains
the command.
"""

import hashlib

from typer.testing import CliRunner

from tracefork.cli import app
from tracefork.locate import LocateHit, TapeHit, locate_in_lineage, locate_value
from tracefork.store import TapeStore
from tracefork.tape import Tape, sha256_hex

runner = CliRunner()

_LOCATE_CLI_WIRED = any(
    (cmd.name or cmd.callback.__name__) == "locate" for cmd in app.registered_commands
)
_SKIP_REASON = (
    "cli.py's `locate` command isn't wired yet (tracefork-bge.62 hands off "
    "cli_command for the orchestrator to paste in) -- self-heals once it is."
)


def _skip_unless_wired():
    import pytest

    if not _LOCATE_CLI_WIRED:
        pytest.skip(_SKIP_REASON)


def _tape_with(*exchanges: tuple[bytes, bytes], tool_exchanges: tuple = ()) -> Tape:
    t = Tape(agent_name="locate-test")
    for req, resp in exchanges:
        t.append_exchange(req, resp)
    for req, resp in tool_exchanges:
        t.append_tool_exchange(req, resp)
    return t


# ── locate_value ─────────────────────────────────────────────────────────


def test_locate_value_returns_none_when_value_never_occurs():
    tape = _tape_with((b"req-0", b"resp-0"), (b"req-1", b"resp-1"))
    assert locate_value(tape, "needle") is None


def test_locate_value_finds_first_match_index_ascending_across_exchanges():
    tape = _tape_with(
        (b"req-0", b"resp-0"),
        (b"req-1-needle", b"resp-1"),
        (b"req-2-needle", b"resp-2"),
    )
    hit = locate_value(tape, "needle")
    assert hit is not None
    assert hit.kind == "exchange"
    assert hit.index == 1  # the FIRST occurrence, not the second at index 2
    assert hit.side == "request"


def test_locate_value_prefers_request_before_response_at_the_same_index():
    tape = _tape_with((b"req-0", b"resp-0-needle"), (b"req-1-needle", b"resp-1"))
    hit = locate_value(tape, "needle")
    assert hit is not None
    # index 0's response matches, but index 1's request also matches and comes
    # first in index-ascending order -- request-before-response only matters
    # when BOTH sides of the SAME index match.
    assert hit.index == 0
    assert hit.side == "response"


def test_locate_value_request_before_response_when_both_sides_of_one_exchange_match():
    tape = _tape_with((b"req-0-needle", b"resp-0-needle"))
    hit = locate_value(tape, "needle")
    assert hit is not None
    assert hit.index == 0
    assert hit.side == "request"


def test_locate_value_scans_tool_exchanges_after_exchanges():
    tape = _tape_with(
        (b"req-0", b"resp-0"),
        tool_exchanges=((b"tool-req-needle", b"tool-resp"),),
    )
    hit = locate_value(tape, "needle")
    assert hit is not None
    assert hit.kind == "tool_exchange"
    assert hit.index == 0
    assert hit.side == "request"


def test_locate_value_exchanges_take_priority_over_tool_exchanges():
    tape = _tape_with(
        (b"req-0", b"resp-0-needle"),
        tool_exchanges=((b"tool-req-needle", b"tool-resp-needle"),),
    )
    hit = locate_value(tape, "needle")
    assert hit is not None
    assert hit.kind == "exchange"
    assert hit.index == 0


def test_locate_value_blob_sha256_is_independently_reproducible():
    """The offline-checkable-receipt property the bead is named for: any
    reader can re-hash the raw bytes at `tape.exchange(i)` themselves and
    get the same `blob_sha256`, with no need to trust this module."""
    tape = _tape_with((b"req-0", b"resp-0"), (b"req-1-needle-here", b"resp-1"))
    hit = locate_value(tape, "needle-here")
    assert hit is not None
    assert hit.kind == "exchange"
    req, resp = tape.exchange(hit.index)
    raw = req if hit.side == "request" else resp
    assert hit.blob_sha256 == hashlib.sha256(raw).hexdigest()
    assert hit.blob_sha256 == sha256_hex(raw)


def test_locate_value_tape_digest_matches_tapes_own_digest():
    tape = _tape_with((b"req-0", b"resp-0-needle"))
    hit = locate_value(tape, "needle")
    assert hit is not None
    assert hit.tape_digest == tape.digest()


# ── locate_in_lineage ────────────────────────────────────────────────────


def test_locate_in_lineage_finds_value_in_root_tape_at_depth_zero(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        root = _tape_with((b"req-0", b"resp-0-needle"))
        run_id = store.save_tape(root, run_id="root-run")

        hits = locate_in_lineage(store, run_id, "needle")
        assert len(hits) == 1
        assert hits[0].depth == 0
        assert hits[0].branch_id is None
        assert hits[0].hit.index == 0
    finally:
        store.close()


def test_locate_in_lineage_finds_value_only_in_direct_branch_delta_tape(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        root = _tape_with((b"req-0", b"resp-0"))
        run_id = store.save_tape(root, run_id="root-run")

        branch_delta = _tape_with((b"req-0", b"resp-0"), (b"req-1-branch-needle", b"resp-1"))
        branch_id = store.save_branch(
            parent_run_id=run_id,
            divergence_step=1,
            delta_tape=branch_delta,
            mutation_desc="test branch",
        )

        hits = locate_in_lineage(store, run_id, "branch-needle")
        assert len(hits) == 1
        found = hits[0]
        assert found.depth == 1
        assert found.branch_id == branch_id
        assert found.hit.index == 1
    finally:
        store.close()


def test_locate_in_lineage_finds_value_in_fork_of_fork_at_depth_two(tmp_path):
    """Mirrors test_storage.py's test_branches_forked_from_finds_fork_of_fork
    fixture shape: branch A's delta_tape is promoted to its own tape (under
    run_id == branch_a_id) so branch B can be saved under
    parent_run_id=branch_a_id -- the promotion convention this bead's BFS
    relies on."""
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        root = _tape_with((b"req-0", b"resp-0"))
        run_id = store.save_tape(root, run_id="root-run")

        branch_a_delta = _tape_with((b"req-0", b"resp-0"), (b"req-1-a", b"resp-1-a"))
        branch_a_id = store.save_branch(
            parent_run_id=run_id,
            divergence_step=1,
            delta_tape=branch_a_delta,
            mutation_desc="branch a",
        )
        # Promote A's delta_tape to its own tape (the promotion convention
        # causal_closure/branches_forked_from already document).
        store.save_tape(branch_a_delta, run_id=branch_a_id)

        branch_b_delta = _tape_with(
            (b"req-0", b"resp-0"),
            (b"req-1-a", b"resp-1-a"),
            (b"req-2-fork-of-fork-needle", b"resp-2"),
        )
        branch_b_id = store.save_branch(
            parent_run_id=branch_a_id,
            divergence_step=2,
            delta_tape=branch_b_delta,
            mutation_desc="branch b",
        )

        hits = locate_in_lineage(store, run_id, "fork-of-fork-needle")
        assert len(hits) == 1
        found = hits[0]
        assert found.depth == 2
        assert found.branch_id == branch_b_id
        assert found.hit.index == 2
    finally:
        store.close()


def test_locate_in_lineage_follow_lineage_false_misses_branch_only_value(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        root = _tape_with((b"req-0", b"resp-0"))
        run_id = store.save_tape(root, run_id="root-run")

        branch_delta = _tape_with((b"req-0", b"resp-0-branch-only-needle"))
        store.save_branch(
            parent_run_id=run_id,
            divergence_step=0,
            delta_tape=branch_delta,
            mutation_desc="branch only",
        )

        hits = locate_in_lineage(store, run_id, "branch-only-needle", follow_lineage=False)
        assert hits == []
    finally:
        store.close()


def test_locate_in_lineage_returns_empty_list_when_value_absent_everywhere(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        root = _tape_with((b"req-0", b"resp-0"))
        run_id = store.save_tape(root, run_id="root-run")
        store.save_branch(
            parent_run_id=run_id,
            divergence_step=0,
            delta_tape=_tape_with((b"req-0", b"resp-0")),
            mutation_desc="no match here",
        )

        assert locate_in_lineage(store, run_id, "nowhere-to-be-found") == []
    finally:
        store.close()


def test_locate_in_lineage_raises_keyerror_for_unknown_root_run_id(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        try:
            locate_in_lineage(store, "no-such-run", "anything")
        except KeyError:
            pass
        else:
            raise AssertionError("expected KeyError for an unknown root_run_id")
    finally:
        store.close()


def test_locate_hit_and_tape_hit_are_frozen_dataclasses():
    hit = TapeHit(kind="exchange", index=0, side="request", blob_sha256="abc", tape_digest="def")
    located = LocateHit(depth=0, branch_id=None, hit=hit)
    try:
        located.depth = 1  # type: ignore[misc]
    except (AttributeError, TypeError):
        pass
    else:
        raise AssertionError("LocateHit should be frozen")


# ── CLI: `tracefork locate` ──────────────────────────────────────────────


def test_cli_locate_finds_value_via_tape_path_prints_receipt(tmp_path):
    _skip_unless_wired()
    tape = _tape_with((b"req-0", b"resp-0-cli-needle"))
    tape_path = tmp_path / "one.tape.sqlite"
    tape.save(str(tape_path))

    result = runner.invoke(app, ["locate", "cli-needle", "--tape", str(tape_path)])
    assert result.exit_code == 0, result.output
    assert "blob_sha256" in result.output
    assert "tape_digest" in result.output


def test_cli_locate_not_found_exits_1_and_prints_not_found(tmp_path):
    _skip_unless_wired()
    tape = _tape_with((b"req-0", b"resp-0"))
    tape_path = tmp_path / "one.tape.sqlite"
    tape.save(str(tape_path))

    result = runner.invoke(app, ["locate", "absent-value", "--tape", str(tape_path)])
    assert result.exit_code == 1, result.output
    assert "not found" in result.output.lower()


def test_cli_locate_run_id_mode_against_a_store(tmp_path):
    _skip_unless_wired()
    db = tmp_path / "store.db"
    store = TapeStore(str(db))
    run_id = store.save_tape(_tape_with((b"req-0", b"resp-0-store-needle")), run_id="run-1")
    store.close()

    result = runner.invoke(app, ["locate", "store-needle", run_id, "--store", str(db)])
    assert result.exit_code == 0, result.output
    assert "blob_sha256" in result.output


def test_cli_locate_no_args_errors_with_usage_message():
    _skip_unless_wired()
    result = runner.invoke(app, ["locate"])
    assert result.exit_code == 1, result.output
    assert "run_id" in result.output or "--tape" in result.output
