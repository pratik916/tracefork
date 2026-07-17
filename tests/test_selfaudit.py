"""selfaudit.py tests: the architecture-fitness gate proving tracefork's own
`src/tracefork/` source never directly calls a `BoundaryGuard`-patched shape
(or `uuid.uuid4`) outside the tiny, comment-justified `SANCTIONED_CALL_SITES`
allowlist. All offline/$0 -- `ast.parse` only, never imports/executes a
scanned file."""

from __future__ import annotations

from pathlib import Path

import tracefork
from tracefork.selfaudit import (
    SANCTIONED_CALL_SITES,
    ArchitectureViolation,
    audit_package,
    scan_file_for_violations,
)


def _package_root() -> Path:
    return Path(tracefork.__file__).parent


def test_no_unsanctioned_nondeterminism_calls_in_tracefork_source():
    # The machine-checked version of the prose invariant: "the agent reads
    # nondeterminism only through NondetSource" / "nothing in tracefork's
    # own recording path spawns a thread/subprocess" (boundary_guard.py's
    # own module docstring). Run against the REAL current src/tracefork
    # tree, not a fixture.
    violations = audit_package(_package_root())
    assert violations == []


def test_scan_flags_a_synthetic_violation(tmp_path):
    # Positive control: proves the scan isn't vacuously green, mirroring the
    # repo's existing negative-control discipline (DriftingNondet, replay
    # divergence tests).
    bad_file = tmp_path / "unsanctioned.py"
    bad_file.write_text("import subprocess\n\ndef spawn():\n    subprocess.Popen(['echo', 'hi'])\n")

    violations = scan_file_for_violations(bad_file, tmp_path)

    assert violations == [
        ArchitectureViolation(file="unsanctioned.py", lineno=4, call="subprocess.Popen.__init__"),
    ]


def test_scan_never_imports_or_executes_the_scanned_file(tmp_path):
    # A module-level side effect that would prove import/exec if it fired.
    bad_file = tmp_path / "would_explode.py"
    bad_file.write_text("raise RuntimeError('scanned file must never execute')\n")

    violations = scan_file_for_violations(bad_file, tmp_path)

    assert violations == []


def test_sanctioned_uuid_call_in_store_py_is_not_flagged():
    # store.py's 4 genuine `uuid.uuid4().hex[:12]` storage-key-default call
    # sites (run_id/branch_id/session_id/edge_id) are sanctioned -- proves
    # the allowlist path is actually exercised, not just the zero-tolerance
    # path every other shape takes.
    package_root = _package_root()
    store_py = package_root / "store.py"

    violations = scan_file_for_violations(store_py, package_root)

    assert violations == []
    assert store_py.read_text().count("uuid.uuid4()") == 4


def test_sanctioned_call_sites_reference_real_files():
    # Guards against allowlist rot: a renamed/deleted sanctioned file would
    # silently reopen a zero-tolerance shape.
    package_root = _package_root()
    for filenames in SANCTIONED_CALL_SITES.values():
        for filename in filenames:
            assert (package_root / filename).is_file(), (
                f"SANCTIONED_CALL_SITES references {filename!r}, "
                "which does not exist under src/tracefork/"
            )
