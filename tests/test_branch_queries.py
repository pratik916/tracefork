"""tracefork-bge.43: branch_descendants / branch_ancestors / branch_siblings —
read-only TapeStore queries over the existing `branches` table, no schema/DDL
change, no digest involvement. Mirrors test_causal_edges.py's `causal_closure`
promotion convention (a branch only recurses further once its `delta_tape` is
itself promoted to a tape via `save_tape(delta_tape, run_id=branch_id)`).

CLI (`tracefork branch descendants|ancestors|siblings`) and server
(`GET /api/branch/{run_id}/related`) surfaces are deferred to cli.py/server.py
wiring (forbidden files for this bead) — see the ready-to-paste code handed
off in the wave's structured result. Only TapeStore-level behavior is tested
here. All offline/$0."""

from __future__ import annotations

from typer.testing import CliRunner

from tracefork.cli import app
from tracefork.store import TapeStore
from tracefork.tape import Tape

runner = CliRunner()


def _small_tape(tag: bytes = b"x") -> Tape:
    t = Tape(agent_name="w")
    t.append_exchange(b"req-" + tag, b"resp-" + tag)
    return t


# ── branch_descendants ───────────────────────────────────────────────────────


def test_branch_descendants_empty_for_leaf_never_forked(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        store.save_tape(_small_tape(b"root"), run_id="root")
        assert store.branch_descendants("root") == []
    finally:
        store.close()


def test_branch_descendants_returns_direct_branches_one_level(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        store.save_tape(_small_tape(b"root"), run_id="root")
        b1 = store.save_branch(
            parent_run_id="root", divergence_step=0, delta_tape=_small_tape(b"b1")
        )
        b2 = store.save_branch(
            parent_run_id="root", divergence_step=0, delta_tape=_small_tape(b"b2")
        )
        descendants = store.branch_descendants("root")
        assert set(descendants) == {b1, b2}
        assert len(descendants) == 2
    finally:
        store.close()


def test_branch_descendants_walks_fork_of_fork_chain(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        store.save_tape(_small_tape(b"root"), run_id="root")
        child = store.save_branch(
            parent_run_id="root", divergence_step=0, delta_tape=_small_tape(b"child")
        )
        # Promote the branch to its own tape (the causal_closure convention)
        # so a fork-of-fork chain is reachable at all.
        store.save_tape(_small_tape(b"child"), run_id=child)
        grandchild = store.save_branch(
            parent_run_id=child, divergence_step=0, delta_tape=_small_tape(b"grandchild")
        )

        descendants = store.branch_descendants("root")
        assert child in descendants
        assert grandchild in descendants
    finally:
        store.close()


def test_branch_descendants_dead_end_when_branch_never_promoted(tmp_path):
    """Mirrors test_causal_closure_ignores_branches_never_promoted_to_a_tape:
    a branch's own children never surface if the branch itself was never
    promoted to a re-forkable tape."""
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        store.save_tape(_small_tape(b"root"), run_id="root")
        child = store.save_branch(
            parent_run_id="root", divergence_step=0, delta_tape=_small_tape(b"child")
        )
        # NOT promoted: child never becomes its own tape, so a "grandchild"
        # branch of it can't legally exist via save_branch's FK — nothing
        # further to find. branch_descendants still reports the one hop.
        descendants = store.branch_descendants("root")
        assert descendants == [child]
    finally:
        store.close()


# ── branch_ancestors ─────────────────────────────────────────────────────────


def test_branch_ancestors_empty_for_root_tape(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        store.save_tape(_small_tape(b"root"), run_id="root")
        assert store.branch_ancestors("root") == []
    finally:
        store.close()


def test_branch_ancestors_one_level(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        store.save_tape(_small_tape(b"root"), run_id="root")
        child = store.save_branch(
            parent_run_id="root", divergence_step=0, delta_tape=_small_tape(b"child")
        )
        assert store.branch_ancestors(child) == ["root"]
    finally:
        store.close()


def test_branch_ancestors_two_generations_nearest_first(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        store.save_tape(_small_tape(b"root"), run_id="root")
        child = store.save_branch(
            parent_run_id="root", divergence_step=0, delta_tape=_small_tape(b"child")
        )
        store.save_tape(_small_tape(b"child"), run_id=child)
        grandchild = store.save_branch(
            parent_run_id=child, divergence_step=0, delta_tape=_small_tape(b"grandchild")
        )
        assert store.branch_ancestors(grandchild) == [child, "root"]
    finally:
        store.close()


# ── branch_siblings ──────────────────────────────────────────────────────────


def test_branch_siblings_empty_for_root_tape(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        store.save_tape(_small_tape(b"root"), run_id="root")
        assert store.branch_siblings("root") == []
    finally:
        store.close()


def test_branch_siblings_empty_for_only_child(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        store.save_tape(_small_tape(b"root"), run_id="root")
        only_child = store.save_branch(
            parent_run_id="root", divergence_step=0, delta_tape=_small_tape(b"only")
        )
        assert store.branch_siblings(only_child) == []
    finally:
        store.close()


def test_branch_siblings_excludes_self_when_multiple_share_parent(tmp_path):
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        store.save_tape(_small_tape(b"root"), run_id="root")
        b1 = store.save_branch(
            parent_run_id="root", divergence_step=0, delta_tape=_small_tape(b"b1")
        )
        b2 = store.save_branch(
            parent_run_id="root", divergence_step=0, delta_tape=_small_tape(b"b2")
        )
        b3 = store.save_branch(
            parent_run_id="root", divergence_step=0, delta_tape=_small_tape(b"b3")
        )

        siblings_of_b1 = store.branch_siblings(b1)
        assert set(siblings_of_b1) == {b2, b3}
        assert b1 not in siblings_of_b1
        # Matches list_branches' own created_at DESC ordering exactly.
        assert siblings_of_b1 == [
            b["branch_id"] for b in store.list_branches("root") if b["branch_id"] != b1
        ]
    finally:
        store.close()


# ── CLI: branch descendants / ancestors / siblings ──────────────────────────


def test_cli_branch_descendants_ancestors_siblings_exit_zero_and_print_ids(tmp_path):
    db = tmp_path / "store.db"
    store = TapeStore(str(db))
    try:
        store.save_tape(_small_tape(b"root"), run_id="root")
        b1 = store.save_branch(
            parent_run_id="root", divergence_step=0, delta_tape=_small_tape(b"b1")
        )
        b2 = store.save_branch(
            parent_run_id="root", divergence_step=0, delta_tape=_small_tape(b"b2")
        )
    finally:
        store.close()

    descendants_result = runner.invoke(app, ["branch", "descendants", "root", "--store", str(db)])
    assert descendants_result.exit_code == 0, descendants_result.output
    assert b1 in descendants_result.output
    assert b2 in descendants_result.output

    ancestors_result = runner.invoke(app, ["branch", "ancestors", b1, "--store", str(db)])
    assert ancestors_result.exit_code == 0, ancestors_result.output
    assert "root" in ancestors_result.output

    siblings_result = runner.invoke(app, ["branch", "siblings", b1, "--store", str(db)])
    assert siblings_result.exit_code == 0, siblings_result.output
    assert b2 in siblings_result.output


def test_cli_branch_descendants_unknown_run_id_exits_zero_with_empty_list(tmp_path):
    """The store methods return [] for an unknown id rather than raising, so
    the CLI command has nothing to gate on -- exits 0 with a "(0)" heading."""
    db = tmp_path / "store.db"
    TapeStore(str(db)).close()

    result = runner.invoke(app, ["branch", "descendants", "does-not-exist", "--store", str(db)])
    assert result.exit_code == 0, result.output
    assert "(0)" in result.output


# ── server.py: GET /api/branch/{run_id}/related ─────────────────────────────


def test_server_get_branch_related_returns_lists_for_known_id(tmp_path):
    """A known branch_id gets its real ancestors/siblings/descendants back --
    never a 404, mirroring test_sessions.py's session-JSON server test."""
    from fastapi.testclient import TestClient

    from tracefork.server import app as fastapi_app
    from tracefork.server import init_store

    db = tmp_path / "store.db"
    store = TapeStore(str(db))
    store.save_tape(_small_tape(b"root"), run_id="root")
    b1 = store.save_branch(parent_run_id="root", divergence_step=0, delta_tape=_small_tape(b"b1"))
    b2 = store.save_branch(parent_run_id="root", divergence_step=0, delta_tape=_small_tape(b"b2"))
    store.close()

    init_store(str(db))
    client = TestClient(fastapi_app)

    ok = client.get(f"/api/branch/{b1}/related")
    assert ok.status_code == 200
    body = ok.json()
    assert body["run_id"] == b1
    assert body["ancestors"] == ["root"]
    assert body["siblings"] == [b2]
    assert body["descendants"] == []


def test_server_get_branch_related_returns_200_with_empty_lists_on_unknown_id(tmp_path):
    """This endpoint never 404s -- an unknown/root/leaf id returns 200 with
    empty lists, since the underlying store methods return [] rather than
    raising (unlike get_branch/get_session's KeyError -> 404 pattern)."""
    from fastapi.testclient import TestClient

    from tracefork.server import app as fastapi_app
    from tracefork.server import init_store

    db = tmp_path / "store.db"
    TapeStore(str(db)).close()

    init_store(str(db))
    client = TestClient(fastapi_app)

    resp = client.get("/api/branch/does-not-exist/related")
    assert resp.status_code == 200
    assert resp.json() == {
        "run_id": "does-not-exist",
        "descendants": [],
        "ancestors": [],
        "siblings": [],
    }


# ── save_tape's default created_at (tracefork item 28) ──────────────────────


def test_save_tape_without_created_at_defaults_to_real_timestamp_not_prune_candidate(
    tmp_path,
):
    """save_tape(tape, run_id=...) with no created_at used to default to '',
    and prune()'s candidate filter is a plain string comparison
    (created_at < older_than_iso) -- '' sorts before every non-empty
    ISO-8601 string, so a tape recorded seconds ago was "older than the
    year 2000" by that comparison. save_tape must now default created_at to
    a real current-time ISO timestamp so a freshly-saved tape is never
    returned as a prune candidate against a decades-old cutoff."""
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        store.save_tape(_small_tape(b"fresh"), run_id="freshtape")

        row = store._con.execute(
            "SELECT created_at FROM tapes WHERE run_id=?", ("freshtape",)
        ).fetchone()
        assert row[0] != ""

        report = store.prune(older_than_iso="2000-01-01T00:00:00+00:00", dry_run=True)
        assert "freshtape" not in report.tapes_archived
    finally:
        store.close()


def test_save_tape_explicit_created_at_still_honored(tmp_path):
    """The default is purely additive: a caller that DOES pass created_at
    keeps getting exactly that value, unmodified -- so the 209 existing
    call sites that already thread a real created_at through (e.g.
    bundle.py) are unaffected."""
    store = TapeStore(str(tmp_path / "store.db"))
    try:
        store.save_tape(
            _small_tape(b"old"), run_id="oldtape", created_at="1999-01-01T00:00:00+00:00"
        )
        row = store._con.execute(
            "SELECT created_at FROM tapes WHERE run_id=?", ("oldtape",)
        ).fetchone()
        assert row[0] == "1999-01-01T00:00:00+00:00"

        report = store.prune(older_than_iso="2000-01-01T00:00:00+00:00", dry_run=True)
        assert "oldtape" in report.tapes_archived
    finally:
        store.close()
