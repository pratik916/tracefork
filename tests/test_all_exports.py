"""Package-wide `__all__` completeness — every `tracefork` submodule must
declare an explicit `__all__`, and (with a short, individually-justified
list of exceptions) it must exactly match that module's own locally-defined
public surface: never silently missing a name, never silently leaking an
import as if it were a deliberate re-export.

Companion to `test_init.py`, which guards the same drift invariant for the
top-level `tracefork` package only (a curated aggregator, by design). This
walks every submodule.

"Locally defined" = bound via a top-level `def`/`class`/`async def` or a
top-level `name = ...` / `name: T = ...` assignment, in THIS file, with a
name that doesn't start with `_` — exactly what `ast.parse` sees among the
module's own top-level statements. This deliberately excludes anything
merely `from .other import Name`-ed in for internal use: a submodule's
`__all__` is its own vocabulary, not everything reachable off it.

Two structural exemptions from the locality check, not a per-name list:
  - any `__init__.py` (an aggregator package — `tracefork/__init__.py`,
    `adapters/__init__.py`, `providers/__init__.py` — exists specifically
    to curate a re-export surface, the same pattern `test_init.py` already
    covers for the top-level package on its own terms);
  - `tracefork_spike` is a different, un-related package (Spike 0), not
    walked here at all.
Everything else goes through the strict locality check, with a short,
named `KNOWN_REEXPORT_EXCEPTIONS` for the handful of individually-reviewed
cases where a submodule deliberately re-exports one imported name because
its own public API structurally exposes it (a dataclass field type, or an
exception its own function raises directly) — see the comment by each
entry. Any new mismatch this test finds should be fixed by narrowing the
module's own `__all__` (the default), not by growing this list.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import pkgutil

import tracefork

KNOWN_REEXPORT_EXCEPTIONS: dict[str, set[str]] = {
    # `diff.py`'s own `StepDiff`/`RangeDiff` dataclasses structurally expose
    # `divergence.FieldDiff` values (and `MISSING` as their sentinel), so
    # diff.py re-exports both rather than making a caller reach into
    # `divergence.py` for a type diff.py's own return values are built from.
    "tracefork.diff": {"FieldDiff", "MISSING"},
    # `tests/test_report_session.py` imports `_session_to_data` directly as
    # a tested unit (a documented, deliberate exception to the "no leading
    # underscore in __all__" convention every other module follows).
    "tracefork.report_session": {"_session_to_data"},
    # `TournamentEngine.run()` raises `blame.BudgetExceededError` directly
    # (the same exception `BlameEngine.rank()` raises), so tournament.py
    # re-exports it rather than sending a caller to `blame.py` to catch it.
    "tracefork.tournament": {"BudgetExceededError"},
}


def _own_public_names(source: str) -> set[str]:
    tree = ast.parse(source)
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("_"):
                names.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and not t.id.startswith("_"):
                    names.add(t.id)
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and not node.target.id.startswith("_")
        ):
            names.add(node.target.id)
    return names


def _all_submodules() -> list[str]:
    return [m.name for m in pkgutil.walk_packages(tracefork.__path__, prefix="tracefork.")]


def test_every_submodule_declares_all() -> None:
    missing = [
        name for name in _all_submodules() if not hasattr(importlib.import_module(name), "__all__")
    ]
    assert not missing, f"modules missing __all__: {missing}"


def test_every_all_entry_resolves_with_no_duplicates() -> None:
    bad: list[str] = []
    for name in _all_submodules():
        mod = importlib.import_module(name)
        all_list = getattr(mod, "__all__", [])
        if len(set(all_list)) != len(all_list):
            bad.append(f"{name}: duplicate entries in __all__")
        for n in all_list:
            if not hasattr(mod, n):
                bad.append(f"{name}: __all__ names {n!r}, which is not bound on the module")
    assert not bad, "\n".join(bad)


def test_all_matches_locally_defined_public_names() -> None:
    """`__all__` must equal exactly this module's own top-level, non-underscore
    def/class/assign names — the "what does this module itself export" filter
    tooling (pdoc/sphinx-autosummary-style) would apply — except an
    aggregator `__init__.py` (structurally exempt, see module docstring) or
    an individually-reviewed `KNOWN_REEXPORT_EXCEPTIONS` entry."""
    mismatches: list[str] = []
    for name in _all_submodules():
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None or not spec.origin.endswith(".py"):
            continue
        if spec.origin.endswith("__init__.py"):
            continue
        with open(spec.origin, encoding="utf-8") as f:
            source = f.read()
        expected = _own_public_names(source)
        mod = importlib.import_module(name)
        declared = set(getattr(mod, "__all__", []))
        exceptions = KNOWN_REEXPORT_EXCEPTIONS.get(name, set())
        missing_from_all = expected - declared
        extra_in_all = declared - expected - exceptions
        if missing_from_all:
            mismatches.append(f"{name}: public names not in __all__: {sorted(missing_from_all)}")
        if extra_in_all:
            mismatches.append(
                f"{name}: __all__ names not locally defined "
                f"(undeclared re-export): {sorted(extra_in_all)}"
            )
    assert not mismatches, "\n".join(mismatches)
