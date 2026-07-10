"""fsck.py — store-level integrity check (read-only), ``tracefork verify --store``.

Distinct from ``replay.py``'s replay-FIDELITY verification (does an agent
reproduce a tape bit-exact?): ``store_fsck()`` is a structural check over the
store itself, in the spirit of ``git fsck``'s corrupt / missing / dangling
object model — does every stored tape and branch still DECODE, and does
every branch's parent still resolve? — without ever running an agent and
without ever mutating the store (unlike ``TapeStore.prune()``).

Walks every tape and branch via ``TapeStore``'s existing PUBLIC surface
(``list_runs``/``load_tape``/``list_branches``/``load_branch``), exercising
the exact same ``Tape.from_bytes`` (JSON + zstd, never pickle) decode path a
real consumer would hit, plus two small ``TapeStore``-only read helpers this
bead adds (``all_branch_parents``/``stored_digest`` — see ``store.py``) for
the orphaned-parent check and the opportunistic digest recompute.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .store import TapeStore


@dataclass
class FsckRow:
    """Outcome of fsck-checking one stored tape or branch row."""

    kind: str  # "tape" | "branch"
    id: str  # run_id (kind="tape") or branch_id (kind="branch")
    passed: bool
    reason: str  # "" when passed


@dataclass
class StoreFsckResult:
    """Outcome of one ``store_fsck()`` call — mirrors ``replay.py``'s
    ``CorpusCheckResult`` dataclass-list-plus-``all_passed`` shape."""

    rows: list[FsckRow] = field(default_factory=list)

    @property
    def all_ok(self) -> bool:
        return all(r.passed for r in self.rows)


def store_fsck(store: TapeStore) -> StoreFsckResult:
    """Read-only structural fsck over ``store``.

    Every tape must decode via ``load_tape`` (a decode error is reported as
    corruption in the returned row, never raised — one bad row must not
    abort the scan of the rest of the store); every branch under a
    still-live parent must decode via ``load_branch``; and every branch's
    ``parent_run_id`` must resolve to a live tape — an "orphaned parent"
    failure (the git-fsck dangling-object case) reported even when
    ``load_branch`` alone would still succeed, e.g. after a parent tape row
    was force-deleted with ``foreign_keys=OFF``, bypassing the FK this
    schema normally enforces.

    Never mutates ``store``. If the store's ``tapes`` table has a ``digest``
    column (see ``TapeStore.stored_digest``, which returns ``None`` when the
    column doesn't exist), a tape's recomputed ``Tape.digest()`` is compared
    against the stored value as a stronger signal; the column's absence is
    not itself a failure, only a missed opportunity for that stronger check.
    """
    rows: list[FsckRow] = []
    runs = store.list_runs()
    run_ids = {r["run_id"] for r in runs}

    for run in runs:
        run_id = run["run_id"]
        rows.append(_check_tape(store, run_id))
        for branch_meta in store.list_branches(run_id):
            rows.append(_check_branch(store, branch_meta["branch_id"]))

    for branch_id, parent_run_id in sorted(store.all_branch_parents()):
        if parent_run_id not in run_ids:
            rows.append(
                FsckRow(
                    kind="branch",
                    id=branch_id,
                    passed=False,
                    reason=f"orphaned parent: parent_run_id {parent_run_id!r} not found",
                )
            )

    return StoreFsckResult(rows=rows)


def _check_tape(store: TapeStore, run_id: str) -> FsckRow:
    try:
        tape = store.load_tape(run_id)
    except Exception as exc:  # decode error = corruption, not a crash
        return FsckRow(kind="tape", id=run_id, passed=False, reason=f"decode error: {exc}")

    stored_digest = store.stored_digest(run_id)
    if stored_digest is not None:
        recomputed = tape.digest()
        if stored_digest != recomputed:
            return FsckRow(
                kind="tape",
                id=run_id,
                passed=False,
                reason=(
                    f"digest mismatch: stored {stored_digest[:12]}…, recomputed {recomputed[:12]}…"
                ),
            )
    return FsckRow(kind="tape", id=run_id, passed=True, reason="")


def _check_branch(store: TapeStore, branch_id: str) -> FsckRow:
    try:
        store.load_branch(branch_id)
    except Exception as exc:
        return FsckRow(kind="branch", id=branch_id, passed=False, reason=f"decode error: {exc}")
    return FsckRow(kind="branch", id=branch_id, passed=True, reason="")
