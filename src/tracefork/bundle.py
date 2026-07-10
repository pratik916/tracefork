"""bundle.py — lossless tape+branch trajectory export/import (portable bundle).

A bundle is literally a second, smaller ``store.db``: same DDL, same
``Tape.to_bytes()`` envelope, zero new serialization format. This mirrors
``git bundle``'s model — a self-contained, versioned, transitive-closure
export that downstream tooling can open directly with no new reader code
path — a scoped-down valid ``store.db`` IS the bundle, rather than a bespoke
archive format.

``export_bundle`` copies a run's ``tapes``/``branches`` BLOB columns
byte-for-byte (via :meth:`TapeStore.raw_tape_row`/``raw_branch_rows`` /
``install_raw_tape_row``/``install_raw_branch_row`` — see ``store.py``) into
a fresh ``store.db`` file: no ``Tape.from_bytes``/``to_bytes`` decode-reencode
round trip, so the bundle's stored bytes are identical to the source store's,
not merely digest-equal.

``import_bundle`` goes the other way through the EXISTING (CAS-guarded)
``save_tape``/``save_branch`` write path — never a raw ``INSERT`` — so a
collision on import (an existing ``run_id``/``branch_id`` with genuinely
different content) is caught as a :class:`~tracefork.store.TapeConflictError`
instead of silently clobbering the prior data; reusing the exact same
``run_id``/``branch_id`` with byte-identical content is an idempotent no-op.
``save_branch``'s ``branch_id=`` parameter (added alongside this module) lets
import preserve a branch's id across stores, matching the run_id it
diverged from.

Only the run's DIRECT branches are exported — the branch DAG's next
generation (a branch promoted to its own tape via
``save_tape(delta_tape, run_id=branch_id)``) is a separate run and gets its
own bundle if wanted. Offline/$0: pure local file I/O, no network.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .store import TapeStore


@dataclass
class BundleExportResult:
    """Outcome of one :func:`export_bundle` call."""

    run_id: str
    branch_ids: list[str] = field(default_factory=list)
    output_path: str = ""


@dataclass
class BundleImportResult:
    """Outcome of one :func:`import_bundle` call — every run_id/branch_id
    that was imported (whether freshly written or an idempotent no-op on
    already-identical content)."""

    run_ids: list[str] = field(default_factory=list)
    branch_ids: list[str] = field(default_factory=list)


def export_bundle(store: TapeStore, run_id: str, output_path: str) -> BundleExportResult:
    """Export ``run_id`` and its direct branches from ``store`` into a fresh,
    self-contained ``store.db`` file at ``output_path`` — a scp-able bundle
    any ``TapeStore`` can open directly, no new reader code path.

    Raw blob copy, byte-for-byte: never decodes/re-encodes tape content (see
    module docstring). Raises ``KeyError`` if ``run_id`` isn't found in
    ``store``.
    """
    tape_row = store.raw_tape_row(run_id)
    if tape_row is None:
        raise KeyError(f"run_id {run_id!r} not found")
    branch_rows = store.raw_branch_rows(run_id)

    bundle = TapeStore(output_path)
    try:
        bundle.install_raw_tape_row(tape_row)
        for row in branch_rows:
            bundle.install_raw_branch_row(row)
    finally:
        bundle.close()

    return BundleExportResult(
        run_id=run_id,
        branch_ids=[row[0] for row in branch_rows],
        output_path=output_path,
    )


def import_bundle(target: TapeStore, bundle_path: str) -> BundleImportResult:
    """Import every run + its direct branches from the bundle ``store.db`` at
    ``bundle_path`` into ``target``, through :meth:`TapeStore.save_tape` /
    :meth:`TapeStore.save_branch` — never a raw ``INSERT`` — so a genuine
    content collision on an existing ``run_id``/``branch_id`` raises
    :class:`~tracefork.store.TapeConflictError` instead of silently
    overwriting; reusing the same ids with byte-identical content is an
    idempotent no-op (see module docstring).
    """
    bundle = TapeStore(bundle_path)
    try:
        imported_run_ids: list[str] = []
        imported_branch_ids: list[str] = []
        for run in bundle.list_runs():
            run_id = run["run_id"]
            tape = bundle.load_tape(run_id)
            target.save_tape(tape, run_id=run_id, created_at=run["created_at"])
            imported_run_ids.append(run_id)

            for branch_meta in bundle.list_branches(run_id):
                branch_id = branch_meta["branch_id"]
                branch = bundle.load_branch(branch_id)
                target.save_branch(
                    parent_run_id=run_id,
                    divergence_step=branch["divergence_step"],
                    delta_tape=branch["delta_tape"],
                    mutation_desc=branch["mutation_desc"],
                    created_at=branch["created_at"],
                    branch_id=branch_id,
                )
                imported_branch_ids.append(branch_id)
    finally:
        bundle.close()

    return BundleImportResult(run_ids=imported_run_ids, branch_ids=imported_branch_ids)
