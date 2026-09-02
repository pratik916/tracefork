"""``TraceforkError`` — the common base every tracefork-raised exception
subclasses, so a caller can write ``except tracefork.TraceforkError`` to catch
any of them generically instead of enumerating each specific class.

A zero-internal-import leaf module (same discipline as ``nondet.py``): nothing
here imports another tracefork module, so every module that raises a
tracefork-specific error can import ``TraceforkError`` from here with no risk
of a circular import.

Additive-only: every existing specific exception class (``BudgetExceededError``,
``BoundaryViolationError``, ``DivergenceError``, etc.) keeps its current base
class(es) via multiple inheritance, so any existing ``except SpecificError`` or
``except RuntimeError`` clause is completely unaffected — this only ADDS a new
common ancestor, it never removes or reorders an existing one in a way that
changes ``isinstance`` results callers already rely on.
"""

from __future__ import annotations

__all__ = ["TraceforkError"]


class TraceforkError(Exception):
    """Base class for every exception tracefork itself raises.

    Not raised directly — always one of its specific subclasses
    (``BudgetExceededError``, ``BoundaryViolationError``,
    ``ConfinementViolationError``, ``ProofEnvelopeError``,
    ``CheckpointPathNotAllowedError``, ``EventStreamError``,
    ``DivergenceError``, ``ReadFileTooLargeError``, ``ProvenanceMismatchError``,
    ``QueryError``, ``TapeConflictError``, ``ForkPointDriftError``,
    ``AgentNotAllowlistedError``). Catch this to handle any tracefork-raised
    error generically; catch a specific subclass to handle just that one.
    """
