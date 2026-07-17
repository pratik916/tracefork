"""Determinism-coverage report for a recorded tape.

Bit-exact replay is only as good as what actually got captured. This turns
"is this replay actually complete?" from a question you have to reason about
by hand into a checkable artifact: which nondeterminism draw kinds
(`nondet.py`) occurred on a tape, whether concurrency was recorded
(`transport.py`'s `async_batches`), whether `BoundaryGuard` (`boundary_guard.py`)
was active at record time, and — given the agent's source text — a static
best-effort scan for call sites shaped like the operations `BoundaryGuard`
can, or (documented) cannot, intercept.

Nondeterminism-coverage-as-instrumentation (in the spirit of the NonDex /
flaky-test divergence-tracing lineage): report what fraction of executed
nondeterminism call-sites were actually intercepted vs. merely assumed
stable, rather than collapsing everything into a single boolean
"is this deterministic" flag.

Both surfaces here are read-only:

* `tape_draw_coverage` only reads fields off an already-loaded `Tape` — it
  never touches `digest()`/`to_bytes()`/`from_bytes()`.
* `scan_source_for_nondeterminism_calls` only `ast.parse`s the given source
  *text*. It never imports or executes it — offline/$0-safe even over
  unreviewed agent code.

**Scope (don't overstate).** The AST scan is a best-effort lint, not
exhaustive static analysis: it matches calls by their literal dotted-
attribute shape (`random.random()`, `threading.Thread(...).start()`, a bare
`subprocess.Popen(...)` constructor call, `time.monotonic()`/`time.sleep()`,
`datetime.datetime.now()`, `time.time()`). It will miss aliasing (`import
random as r; r.random()`), indirection through a variable
(`t = threading.Thread(...); t.start()`), or calls reached via `getattr`.
The guardable-call list is grounded exactly in what `boundary_guard.py`
itself patches; `datetime.datetime.now()` and `time.time()` are carried
forward as that module's own documented, permanent exclusions and are
**never** flagged as violations, only as informational findings, no matter
whether a `BoundaryGuard` was active.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

from tracefork.tape import Tape

#: The five draw kinds `nondet.py` actually records (`RecordingNondet.draws`).
#: Kept in sync with `nondet.py` -- do not invent new kinds here.
DRAW_KINDS: tuple[str, ...] = ("clock", "uuid", "random", "env", "read_file")

# Call sites `BoundaryGuard.__enter__` actually patches (boundary_guard.py),
# keyed by the trailing dotted-path components a matching `ast.Call` resolves
# to, mapped to a human-readable label. Kept in lockstep with that module.
_GUARDABLE_CALLS: dict[tuple[str, ...], str] = {
    ("Thread", "start"): "threading.Thread.start",
    ("Popen",): "subprocess.Popen.__init__",
    ("random", "random"): "random.random",
    ("time", "monotonic"): "time.monotonic",
    ("time", "sleep"): "time.sleep",
}

# Call sites `boundary_guard.py`'s own module docstring documents as
# deliberately NOT patched -- always informational, never a violation,
# regardless of whether a guard was active on this tape.
_DOCUMENTED_NON_GUARDABLE_CALLS: dict[tuple[str, ...], str] = {
    ("datetime", "now"): "datetime.datetime.now()",
    ("time", "time"): "time.time()",
}


@dataclass(frozen=True)
class CallFinding:
    """One nondeterminism-shaped call site found by the static AST scan.

    `guardable` reflects whether the call matches something
    `BoundaryGuard.__enter__` actually patches (see `_GUARDABLE_CALLS`);
    `caught` is `guardable AND` the tape's recorded `boundary_guard_active`
    state -- a documented non-guardable call (`guardable=False`) is always
    `caught=False`, regardless of whether a guard was active, since
    `BoundaryGuard` never intercepts it either way.
    """

    call: str
    lineno: int
    guardable: bool
    caught: bool
    note: str


@dataclass(frozen=True)
class CoverageReport:
    """Combined determinism-coverage report for one tape (+ optional agent
    source scan)."""

    draw_counts: dict[str, int]
    concurrency_recorded: bool
    boundary_guard_active: bool
    findings: list[CallFinding] = field(default_factory=list)


def tape_draw_coverage(tape: Tape) -> tuple[dict[str, int], bool, bool]:
    """Tally an already-loaded tape's draw kinds, whether concurrency was
    recorded, and whether `BoundaryGuard` was active at record time.

    Read-only: only reads `tape.draws`/`tape.async_batches`/`tape.provenance`
    -- never touches `digest()`/`to_bytes()`/`from_bytes()`. Returns
    `(draw_counts, concurrency_recorded, boundary_guard_active)`.
    `draw_counts` only contains entries for kinds that actually occurred
    (no zero-filled kinds), keyed by exactly `nondet.py`'s five kinds.
    """
    draw_counts: dict[str, int] = {}
    for kind, _value in tape.draws:
        if kind in DRAW_KINDS:
            draw_counts[kind] = draw_counts.get(kind, 0) + 1
    concurrency_recorded = len(tape.async_batches) > 0
    boundary_guard_active = tape.provenance.get("boundary_guard", "").lower() == "true"
    return draw_counts, concurrency_recorded, boundary_guard_active


def _dotted_path(node: ast.expr) -> tuple[str, ...] | None:
    """Resolve a `Name`/`Attribute` chain (e.g. `random.random`) to its
    dotted-path components. Returns `None` for any other expression shape."""
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Attribute):
        base = _dotted_path(node.value)
        if base is None:
            return None
        return (*base, node.attr)
    return None


def _call_path(node: ast.Call) -> tuple[str, ...] | None:
    """Resolve a `Call` node's callee to a dotted-path tuple, chaining through
    a `Thread(...).start()`-shaped call (the inner call's own callee is
    collapsed to its last component, e.g. `Thread`, then `start` appended)."""
    func = node.func
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Call):
        inner = _call_path(func.value)
        if inner is None:
            return None
        return (inner[-1], func.attr)
    return _dotted_path(func)


def scan_source_for_nondeterminism_calls(
    source: str, boundary_guard_active: bool
) -> list[CallFinding]:
    """Best-effort static AST lint over agent source *text* for call sites
    shaped like the direct nondeterminism/boundary-crossing operations
    `BoundaryGuard` can (or, documented, cannot) intercept.

    Read-only and NEVER imports or executes `source` -- `ast.parse` only, so
    this is safe to run over untrusted/unreviewed agent code, offline/$0.
    See the module docstring for the scan's best-effort-lint scope limit.
    """
    tree = ast.parse(source)
    findings: list[CallFinding] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        path = _call_path(node)
        if path is None:
            continue

        matched = False
        for key, label in _GUARDABLE_CALLS.items():
            if len(path) >= len(key) and path[-len(key) :] == key:
                findings.append(
                    CallFinding(
                        call=label,
                        lineno=node.lineno,
                        guardable=True,
                        caught=boundary_guard_active,
                        note=(
                            "intercepted by the active BoundaryGuard"
                            if boundary_guard_active
                            else "would be intercepted if BoundaryGuard were "
                            "active (see boundary_guard.py)"
                        ),
                    )
                )
                matched = True
                break
        if matched:
            continue

        for key, label in _DOCUMENTED_NON_GUARDABLE_CALLS.items():
            if len(path) >= len(key) and path[-len(key) :] == key:
                findings.append(
                    CallFinding(
                        call=label,
                        lineno=node.lineno,
                        guardable=False,
                        caught=False,
                        note=(
                            "documented BoundaryGuard exclusion -- never "
                            "guardable, guard active or not (see boundary_guard.py)"
                        ),
                    )
                )
                break

    return findings


def coverage_report(tape: Tape, agent_source: str | None = None) -> CoverageReport:
    """Build the combined determinism-coverage report for `tape`.

    `agent_source`, if given, is the agent's Python source *text* (never a
    file the tape references implicitly, never imported/executed) run
    through `scan_source_for_nondeterminism_calls` with the tape's own
    recorded `boundary_guard_active` state. Omit it to get just the
    tape-side tally with an empty `findings` list.
    """
    draw_counts, concurrency_recorded, boundary_guard_active = tape_draw_coverage(tape)
    findings = (
        scan_source_for_nondeterminism_calls(agent_source, boundary_guard_active)
        if agent_source is not None
        else []
    )
    return CoverageReport(
        draw_counts=draw_counts,
        concurrency_recorded=concurrency_recorded,
        boundary_guard_active=boundary_guard_active,
        findings=findings,
    )
