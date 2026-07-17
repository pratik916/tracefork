"""Field-scoped oracle: grade one JSON field's value, not the whole output.

``StringMatchOracle`` (in ``blame.py``) regex-matches the *entire* graded
text, so any unrelated field containing a stray "SUCCESS"/"FAIL" substring
can flip a verdict that has nothing to do with the field an operator actually
cares about. This module adds one additive, OPT-IN ``Oracle`` implementation
on top of the protocol defined in ``blame.py``:

``FieldDiffOracle`` takes a ``field_path`` — a ``$.``-style key path (leading
``$`` optional) mirroring ``divergence.py``'s existing ``FieldDiff.path``
convention (e.g. ``$.result.status`` or ``$.items[0].value``) — plus
``success_re``/``failure_re`` (the same contract ``StringMatchOracle``
already uses). ``grade()`` parses ``output`` as JSON, resolves the path
against the parsed document via a small local tokenizer/walker (dict-key and
list-index steps), stringifies the resolved leaf (``str`` values as-is,
everything else via ``json.dumps(..., sort_keys=True)``), and regex-matches
success/failure exactly like ``StringMatchOracle`` — so grading is scoped to
ONE field's value: field-level provenance-of-value ("which step caused THIS
field to end up as X"), distinguishable from noise in unrelated fields. Any
resolution failure — non-JSON ``output``, a missing key, an out-of-range
index, or indexing a non-container — returns ``None`` (ambiguous), never
raises.

Registers itself into ``blame.ORACLE_REGISTRY`` under ``"field_diff"`` at
import time, the same opt-in pattern ``judge.py`` already establishes for
``LLMJudgeOracle`` — importing this module is itself the opt-in.
``BlameEngine.rank()`` is untouched: it only ever calls ``oracle.grade(str)``,
so it is oracle-agnostic by construction. Pure stdlib (``json``, ``re``) — no
new dependency, no network.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .blame import Oracle, register_oracle

__all__ = ["FieldDiffOracle"]

# ── field-path tokenizer ─────────────────────────────────────────────────────
#
# Matches one dotted key segment (optionally preceded by its separating dot)
# or one bracketed list index per iteration, e.g. "result.status" ->
# ["result", "status"] and "items[0].value" -> ["items", 0, "value"].
_PATH_TOKEN_RE = re.compile(r"\.?([^.\[\]]+)|\[(\d+)\]")


def _tokenize_field_path(field_path: str) -> list[str | int]:
    """Tokenize a ``$.``-style key path into dict-key (``str``) and
    list-index (``int``) steps. Leading ``$`` and/or ``.`` are stripped;
    neither is a token itself."""
    path = field_path.strip()
    if path.startswith("$"):
        path = path[1:]
    if path.startswith("."):
        path = path[1:]
    tokens: list[str | int] = []
    for match in _PATH_TOKEN_RE.finditer(path):
        key, index = match.groups()
        if key is not None:
            tokens.append(key)
        else:
            assert index is not None
            tokens.append(int(index))
    return tokens


def _resolve_field_path(doc: Any, tokens: list[str | int]) -> Any:
    """Walk ``tokens`` against a parsed JSON document. Raises ``KeyError``
    (missing dict key), ``IndexError`` (out-of-range list index), or
    ``TypeError`` (indexing a non-container, or a dict/list-kind mismatch)
    on any resolution failure — the caller maps all three to ``None``."""
    value = doc
    for token in tokens:
        if isinstance(token, int):
            if not isinstance(value, list):
                raise TypeError(f"cannot index non-list with [{token}]")
            value = value[token]
        else:
            if not isinstance(value, dict):
                raise TypeError(f"cannot index non-dict with .{token}")
            value = value[token]
    return value


# ── FieldDiffOracle ──────────────────────────────────────────────────────────


class FieldDiffOracle:
    """Grades by resolving one JSON field and regex-matching its value.

    See the module docstring for the full contract. ``field_path`` mirrors
    ``divergence.FieldDiff.path``'s ``$.``-style convention; ``success_re``/
    ``failure_re`` are the same contract ``StringMatchOracle`` uses.
    """

    def __init__(self, *, field_path: str, success_re: str, failure_re: str) -> None:
        self._tokens = _tokenize_field_path(field_path)
        self._success = re.compile(success_re)
        self._failure = re.compile(failure_re)

    def grade(self, output: str) -> bool | None:
        try:
            doc = json.loads(output)
        except ValueError:
            return None
        try:
            value = _resolve_field_path(doc, self._tokens)
        except (KeyError, IndexError, TypeError):
            return None
        text = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
        if self._success.search(text):
            return True
        if self._failure.search(text):
            return False
        return None


register_oracle("field_diff", FieldDiffOracle)

# Static conformance check only (no runtime inheritance — the same duck-typed
# pattern `StringMatchOracle` already uses): mypy rejects this assignment if
# `FieldDiffOracle` ever drifts from the `Oracle` protocol's `grade` signature.
_ORACLE_CONFORMANCE: type[Oracle] = FieldDiffOracle
