"""Structured confinement-violation diagnostics: turn a bare
`ConfinementViolationError` into an operator-facing explanation.

`boundary_guard.py`'s `ConfinementViolationError` (see its class docstring)
is raised from exactly two sites -- `_guarded_open`'s write-outside-
`writable_roots` check and `_guarded_socket_connect`'s host-outside-
`allowed_hosts` check -- and, since tracefork-bge.72, both raise sites set
optional structured attributes on the exception (`violation_kind`/
`attempted`/`declared_writable_roots`/`declared_allowed_hosts`) alongside
its unchanged message string. This module is the confinement-violation
analogue of `divergence.py`'s `DivergenceDiagnostic`/`diagnose`/
`diagnostic_to_dict` triad: it reads those attributes straight off the
already-raised exception (never re-parses `str(error)`) into a typed,
JSON-safe view a CLI or web report can echo.

Nothing here changes what `boundary_guard.py` raises or when; this is
purely a read-time diagnostic built from an already-raised
`ConfinementViolationError`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .boundary_guard import ConfinementViolationError


@dataclass(frozen=True)
class ConfinementDiagnostic:
    """A structured, operator-facing explanation of one confinement violation.

    `violation_kind` is `"write"` (a `builtins.open` write-mode call outside
    `declared_writable_roots`) or `"connect"` (a `socket.connect` to a host
    outside `declared_allowed_hosts`) -- or `None` if the source exception
    never set the attribute (e.g. one raised by hand with the pre-bge.72
    single-message-arg shape). `attempted` is the denied path (write) or
    host (connect). Exactly one of `declared_writable_roots`/
    `declared_allowed_hosts` is populated, matching `violation_kind`; the
    other is `None`.
    """

    violation_kind: str | None
    attempted: str | None
    declared_writable_roots: tuple[str, ...] | None
    declared_allowed_hosts: tuple[str, ...] | None
    message: str


def diagnose_confinement(error: ConfinementViolationError) -> ConfinementDiagnostic:
    """Build a `ConfinementDiagnostic` from an already-raised
    `ConfinementViolationError`.

    Reads the exception's own structured attributes (set at both
    `boundary_guard.py` raise sites since tracefork-bge.72) -- never
    `str(error)` -- so this stays correct even if the message text is
    reworded later.
    """
    return ConfinementDiagnostic(
        violation_kind=error.violation_kind,
        attempted=error.attempted,
        declared_writable_roots=error.declared_writable_roots,
        declared_allowed_hosts=error.declared_allowed_hosts,
        message=str(error),
    )


def confinement_diagnostic_to_dict(diag: ConfinementDiagnostic) -> dict[str, Any]:
    """JSON-safe view of a `ConfinementDiagnostic` for the CLI / web report."""
    return {
        "violation_kind": diag.violation_kind,
        "attempted": diag.attempted,
        "declared_writable_roots": (
            list(diag.declared_writable_roots) if diag.declared_writable_roots is not None else None
        ),
        "declared_allowed_hosts": (
            list(diag.declared_allowed_hosts) if diag.declared_allowed_hosts is not None else None
        ),
        "message": diag.message,
    }
