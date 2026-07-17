"""Confinement-tier tests (tracefork-bge.56): `compute_confinement_tier()`'s
tier selection, `Branch.confinement_tier` wiring through `fork()`/
`fork_coalition()`/`rebase()`, `store.py`'s migration + round-trip, and proof
that the tier never perturbs `delta_tape.digest()` -- it is Branch/store-level
metadata only, exactly like `branch_digest`/`parent_tape_digest`. All offline,
no API keys.
"""

import anthropic
import httpx

from tests.fakes import ScriptedFakeLLM, make_text_response
from tracefork.boundary_guard import ConfinementSpec
from tracefork.constants import (
    CONFINEMENT_TIER_DECLARED,
    CONFINEMENT_TIER_GUARDED,
    CONFINEMENT_TIER_NONE,
)
from tracefork.fork import BranchSpec, CoalitionSpec, ForkEngine, compute_confinement_tier
from tracefork.store import TapeStore
from tracefork.tape import Tape, open_sqlite
from tracefork.transport import TraceforkTransport

RESP_A = make_text_response("Response A")
RESP_B = make_text_response("Response B -- mutated")
RESP_C = make_text_response("Response C -- final turn")


def _conversation_agent(client: anthropic.Anthropic) -> str:
    """Two-turn agent; turn2's history embeds turn1's reply text."""
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


# ── compute_confinement_tier() pure tier selection ──────────────────────────


def test_compute_confinement_tier_unconfined_default():
    assert compute_confinement_tier(False, None) == CONFINEMENT_TIER_NONE


def test_compute_confinement_tier_boundary_guard_only():
    assert compute_confinement_tier(True, None) == CONFINEMENT_TIER_GUARDED


def test_compute_confinement_tier_declared_wins_with_boundary_guard_false():
    spec = ConfinementSpec(writable_roots=("/tmp",))
    assert compute_confinement_tier(False, spec) == CONFINEMENT_TIER_DECLARED


def test_compute_confinement_tier_declared_wins_with_boundary_guard_true():
    """`confinement` forces the guard active regardless of `boundary_guard`
    (see `fork()`'s docstring) -- so DECLARED must win over GUARDED too."""
    spec = ConfinementSpec(writable_roots=("/tmp",))
    assert compute_confinement_tier(True, spec) == CONFINEMENT_TIER_DECLARED


# ── ForkEngine.fork() wiring ─────────────────────────────────────────────────


def test_fork_no_kwargs_produces_none_tier():
    parent_tape = _build_two_turn_tape()
    spec = BranchSpec(divergence_step=1, mutated_response=RESP_B)
    branch = ForkEngine.fork(
        parent_tape, spec, _conversation_agent, post_fork_transport=ScriptedFakeLLM([])
    )
    assert branch.confinement_tier == CONFINEMENT_TIER_NONE


def test_fork_boundary_guard_true_produces_guarded_tier_with_unchanged_digest():
    """The tier flips to GUARDED, but `delta_tape.digest()` (the hash-chained
    content) is byte-identical to the default no-kwargs fork -- proving the
    tier is pure metadata never fed into `digest()`."""
    parent_tape = _build_two_turn_tape()
    spec = BranchSpec(divergence_step=1, mutated_response=RESP_B)

    baseline = ForkEngine.fork(
        parent_tape, spec, _conversation_agent, post_fork_transport=ScriptedFakeLLM([])
    )
    guarded = ForkEngine.fork(
        parent_tape,
        spec,
        _conversation_agent,
        post_fork_transport=ScriptedFakeLLM([]),
        boundary_guard=True,
    )

    assert baseline.confinement_tier == CONFINEMENT_TIER_NONE
    assert guarded.confinement_tier == CONFINEMENT_TIER_GUARDED
    assert guarded.delta_tape.digest() == baseline.delta_tape.digest()


def test_fork_confinement_produces_declared_tier_with_unchanged_digest(tmp_path):
    """A `confinement=ConfinementSpec(...)` fork produces CONFINEMENT_TIER_DECLARED
    with `delta_tape.digest()` UNCHANGED vs. the same fork without confinement --
    the tier is pure metadata, never hash-chained."""
    parent_tape = _build_two_turn_tape()
    spec = BranchSpec(divergence_step=1, mutated_response=RESP_B)

    baseline = ForkEngine.fork(
        parent_tape, spec, _conversation_agent, post_fork_transport=ScriptedFakeLLM([])
    )
    declared = ForkEngine.fork(
        parent_tape,
        spec,
        _conversation_agent,
        post_fork_transport=ScriptedFakeLLM([]),
        confinement=ConfinementSpec(writable_roots=(str(tmp_path),)),
    )

    assert baseline.confinement_tier == CONFINEMENT_TIER_NONE
    assert declared.confinement_tier == CONFINEMENT_TIER_DECLARED
    assert declared.delta_tape.digest() == baseline.delta_tape.digest()


# ── ForkEngine.fork_coalition() wiring ───────────────────────────────────────


def test_fork_coalition_boundary_guard_true_produces_guarded_tier():
    parent_tape = _build_two_turn_tape()
    spec = CoalitionSpec.single(0, RESP_B)
    branch = ForkEngine.fork_coalition(
        parent_tape,
        spec,
        _conversation_agent,
        post_fork_transport=ScriptedFakeLLM([RESP_C]),
        boundary_guard=True,
    )
    assert branch.confinement_tier == CONFINEMENT_TIER_GUARDED


def test_fork_coalition_no_kwargs_produces_none_tier():
    parent_tape = _build_two_turn_tape()
    spec = CoalitionSpec.single(0, RESP_B)
    branch = ForkEngine.fork_coalition(
        parent_tape, spec, _conversation_agent, post_fork_transport=ScriptedFakeLLM([RESP_C])
    )
    assert branch.confinement_tier == CONFINEMENT_TIER_NONE


# ── ForkEngine.rebase() wiring (no confinement kwarg -- NONE/GUARDED only) ──


def test_rebase_boundary_guard_true_produces_guarded_tier():
    parent_tape = _build_two_turn_tape()
    old_branch = ForkEngine.fork(
        parent_tape,
        BranchSpec(divergence_step=1, mutated_response=RESP_B),
        _conversation_agent,
        post_fork_transport=ScriptedFakeLLM([]),
    )
    new_parent_tape = _build_two_turn_tape()  # a fresh, byte-identical parent

    rebased = ForkEngine.rebase(
        old_branch,
        new_parent_tape,
        _conversation_agent,
        post_fork_transport=ScriptedFakeLLM([]),
        boundary_guard=True,
    )
    assert rebased.confinement_tier == CONFINEMENT_TIER_GUARDED


def test_rebase_no_kwargs_produces_none_tier():
    parent_tape = _build_two_turn_tape()
    old_branch = ForkEngine.fork(
        parent_tape,
        BranchSpec(divergence_step=1, mutated_response=RESP_B),
        _conversation_agent,
        post_fork_transport=ScriptedFakeLLM([]),
    )
    new_parent_tape = _build_two_turn_tape()

    rebased = ForkEngine.rebase(
        old_branch, new_parent_tape, _conversation_agent, post_fork_transport=ScriptedFakeLLM([])
    )
    assert rebased.confinement_tier == CONFINEMENT_TIER_NONE


# ── store.py: save_branch/load_branch/find_branch_by_digest/list_branches ──


def test_confinement_tier_round_trips_through_store(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        parent_tape = _small_tape(b"parent")
        run_id = store.save_tape(parent_tape, run_id="parent-run")
        branch_id = store.save_branch(
            parent_run_id=run_id,
            divergence_step=0,
            delta_tape=_small_tape(b"branch"),
            branch_digest="digest-declared",
            confinement_tier=CONFINEMENT_TIER_DECLARED,
        )

        loaded = store.load_branch(branch_id)
        assert loaded["confinement_tier"] == CONFINEMENT_TIER_DECLARED

        by_digest = store.find_branch_by_digest("digest-declared")
        assert by_digest is not None
        assert by_digest["confinement_tier"] == CONFINEMENT_TIER_DECLARED

        summaries = store.list_branches(run_id)
        assert len(summaries) == 1
        assert summaries[0]["confinement_tier"] == CONFINEMENT_TIER_DECLARED
    finally:
        store.close()


def test_save_branch_omitting_confinement_tier_defaults_to_empty_string(tmp_path):
    """Every existing caller that omits `confinement_tier` keeps storing ''
    exactly as before this parameter existed."""
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        run_id = store.save_tape(_small_tape(b"parent"), run_id="parent-run")
        branch_id = store.save_branch(
            parent_run_id=run_id,
            divergence_step=0,
            delta_tape=_small_tape(b"branch"),
        )
        loaded = store.load_branch(branch_id)
        assert loaded["confinement_tier"] == ""
        summaries = store.list_branches(run_id)
        assert summaries[0]["confinement_tier"] == ""
    finally:
        store.close()


def test_confinement_tier_migration_adds_column_without_losing_rows(tmp_path):
    """A store.db built with the OLD schema (no `confinement_tier` column)
    neither crashes nor loses rows when opened by the new `TapeStore` -- the
    column gets added via a guarded `ALTER TABLE`, mirroring
    `test_storage.py`'s existing
    `test_branch_digest_migration_adds_column_without_losing_rows` pattern."""
    db_path = str(tmp_path / "old_store.db")

    # Build an old-schema store.db by hand (no confinement_tier column at all).
    old_con = open_sqlite(db_path)
    old_con.executescript(
        """
        CREATE TABLE IF NOT EXISTS tapes (
            run_id       TEXT PRIMARY KEY,
            agent_name   TEXT NOT NULL,
            tape_bytes   BLOB NOT NULL,
            created_at   TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS branches (
            branch_id         TEXT PRIMARY KEY,
            parent_run_id     TEXT NOT NULL,
            divergence_step   INTEGER NOT NULL,
            delta_tape_bytes  BLOB NOT NULL,
            mutation_desc     TEXT NOT NULL DEFAULT '',
            created_at        TEXT NOT NULL,
            branch_digest     TEXT NOT NULL DEFAULT '',
            parent_tape_digest          TEXT NOT NULL DEFAULT '',
            divergence_exchange_digest  TEXT NOT NULL DEFAULT '',
            intervened_steps_json       TEXT NOT NULL DEFAULT '[]',
            FOREIGN KEY(parent_run_id) REFERENCES tapes(run_id)
        );
        """
    )
    old_tape = _small_tape(b"pre-existing")
    old_con.execute(
        "INSERT INTO tapes(run_id, agent_name, tape_bytes, created_at) VALUES(?,?,?,?)",
        ("old-run", "w", old_tape.to_bytes(), "2020-01-01T00:00:00+00:00"),
    )
    old_con.execute(
        """INSERT INTO branches
           (branch_id, parent_run_id, divergence_step, delta_tape_bytes, mutation_desc, created_at)
           VALUES(?,?,?,?,?,?)""",
        (
            "old-branch",
            "old-run",
            0,
            _small_tape(b"pre-branch").to_bytes(),
            "",
            "2020-01-01T00:00:00+00:00",
        ),
    )
    cols_before = {row[1] for row in old_con.execute("PRAGMA table_info(branches)").fetchall()}
    assert "confinement_tier" not in cols_before

    old_con.commit()
    old_con.close()

    # Opening with the new TapeStore must not crash and must not lose rows.
    store = TapeStore(db_path)
    try:
        assert store.load_tape("old-run").exchanges == old_tape.exchanges
        loaded_branch = store.load_branch("old-branch")
        assert loaded_branch["parent_run_id"] == "old-run"
        assert loaded_branch["confinement_tier"] == ""  # migrated column defaults to ''

        cols_after = {
            row[1] for row in store._con.execute("PRAGMA table_info(branches)").fetchall()
        }
        assert "confinement_tier" in cols_after
    finally:
        store.close()
