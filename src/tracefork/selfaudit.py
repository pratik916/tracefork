"""selfaudit.py — architecture-fitness gate: a static AST scan proving
tracefork's own source never directly calls the nondeterminism/boundary-
crossing operations `BoundaryGuard` patches (see `boundary_guard.py`), plus
`uuid.uuid4` (patched globally by `recorder.py`, not `BoundaryGuard`, but the
same class of "reads nondeterminism outside `NondetSource`" violation) —
outside a tiny, explicitly comment-justified `SANCTIONED_CALL_SITES`
allowlist.

Turns the prose invariant ("the agent reads nondeterminism only through
`NondetSource`"; "nothing in tracefork's own recording path spawns a
thread/subprocess" — both already asserted in `boundary_guard.py`'s module
docstring) into a machine-checked one, the same way `coverage.py` turns
"is this replay actually complete?" into a checkable artifact.

Reuses `coverage.py`'s existing `_call_path`/`_dotted_path` dotted-path
resolver (imported read-only; `coverage.py` itself is untouched) instead of
duplicating that AST-walking logic here.

**Scope (don't overstate).** This is the same best-effort lint `coverage.py`
already documents: it matches calls by their literal dotted-attribute shape
and will miss aliasing (`import uuid as _uuid_module; _uuid_module.uuid4()`
— exactly what `recorder.py`/`adapters/base.py` do at their own patch
points) or indirection through a variable. It is a static fitness gate over
tracefork's own `src/tracefork/` tree, not a runtime sandbox — `BoundaryGuard`
is the runtime enforcement; this is the "prove no call site needs it in the
first place" complement.

Read-only: only `ast.parse`s each file's source *text*. Never imports or
executes any file under the scanned package — offline/$0-safe.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from tracefork.coverage import _call_path

# The 5 call-shapes `BoundaryGuard.__enter__` actually patches
# (`boundary_guard.py`), plus `uuid.uuid4` (patched globally by
# `recorder.py`/`adapters/base.py`, not `BoundaryGuard`, but read outside
# `NondetSource` all the same) -- keyed by the trailing dotted-path
# components a matching `ast.Call` resolves to, mapped to a human-readable
# label. Kept in lockstep with those modules.
_AUDITED_CALL_SHAPES: dict[tuple[str, ...], str] = {
    ("Thread", "start"): "threading.Thread.start",
    ("Popen",): "subprocess.Popen.__init__",
    ("random", "random"): "random.random",
    ("time", "monotonic"): "time.monotonic",
    ("time", "sleep"): "time.sleep",
    ("uuid", "uuid4"): "uuid.uuid4",
}

# Explicitly sanctioned direct call sites, keyed by the shape's label
# (matching `_AUDITED_CALL_SHAPES` values) -> a tuple of filenames (relative
# to the scanned `package_root`, e.g. `Path(tracefork.__file__).parent`)
# allowed to call it directly. A shape with no entry here has ZERO
# tolerance: any direct call anywhere in the scanned tree is a violation.
# Each entry below is comment-justified, not a bare allowlist.
SANCTIONED_CALL_SITES: dict[str, tuple[str, ...]] = {
    # store.py calls `uuid.uuid4().hex[:12]` 4x (run_id/branch_id/
    # session_id/edge_id storage-key defaults, see `store.py`'s
    # `import uuid` at module scope) -- a storage-layer identifier
    # generator, never read by an agent or fed into replay/nondeterminism.
    # Verified empirically against the real tree: none of the other 5
    # BoundaryGuard-patched shapes are ever directly CALLED anywhere in
    # tracefork's own source today (`boundary_guard.py` itself only
    # reads/reassigns them as attributes -- e.g. `_subprocess_module.Popen
    # .__init__ = _guarded_popen_init` -- never invokes the real ones), so
    # every other shape's allowlist stays empty.
    "uuid.uuid4": ("store.py",),
}


@dataclass(frozen=True)
class ArchitectureViolation:
    """One unsanctioned direct call to a `_AUDITED_CALL_SHAPES` shape found
    in tracefork's own source by the static scan."""

    file: str
    lineno: int
    call: str


def scan_file_for_violations(path: Path, package_root: Path) -> list[ArchitectureViolation]:
    """Read-only `ast.parse`-only scan of one `.py` file for unsanctioned
    direct calls to `_AUDITED_CALL_SHAPES`. Never imports or executes
    `path` -- offline/$0-safe, mirroring `coverage.py`'s
    `scan_source_for_nondeterminism_calls`'s same never-import guarantee.

    `package_root` is the directory `path`'s reported `file` field (and the
    `SANCTIONED_CALL_SITES` lookup) is relative to -- typically
    `Path(tracefork.__file__).parent`. `path` outside `package_root` reports
    its filename unchanged rather than raising.
    """
    source = path.read_text()
    tree = ast.parse(source)
    try:
        rel = str(path.relative_to(package_root))
    except ValueError:
        rel = str(path)

    violations: list[ArchitectureViolation] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        call_path = _call_path(node)
        if call_path is None:
            continue
        for key, label in _AUDITED_CALL_SHAPES.items():
            if len(call_path) < len(key) or call_path[-len(key) :] != key:
                continue
            if rel in SANCTIONED_CALL_SITES.get(label, ()):
                break
            violations.append(ArchitectureViolation(file=rel, lineno=node.lineno, call=label))
            break

    return violations


def audit_package(package_root: Path) -> list[ArchitectureViolation]:
    """Scan every `*.py` file under `package_root` (recursively, so
    `adapters/`/`providers/` are covered) for unsanctioned direct calls to
    `_AUDITED_CALL_SHAPES`. Returns every violation found, sorted by
    `(file, lineno)` for stable output. An empty list is the passing case --
    the machine-checked version of "nothing in tracefork's own source
    bypasses `NondetSource`/`BoundaryGuard`".

    Read-only: only `ast.parse`s each file's source text. Never imports or
    executes anything under `package_root`.
    """
    violations: list[ArchitectureViolation] = []
    for path in package_root.rglob("*.py"):
        violations.extend(scan_file_for_violations(path, package_root))
    return sorted(violations, key=lambda v: (v.file, v.lineno))
