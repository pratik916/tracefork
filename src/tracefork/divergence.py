"""Structured divergence diagnostics: turn a bare sha256 mismatch into a diff.

`transport.py`'s replay-mode divergence check is a hard-error proof — exactly
what makes it stronger than VCR-style matching — but a bare "fingerprint abc123
!= def456" tells an operator nothing about *what* changed. This module adds a
diagnostic layer on top, without touching the proof itself:

* `diff_json`/`diff_request_bytes` — a pure structural diff (key path ->
  recorded vs live) between two JSON request bodies.
* `diagnose` — builds a `DivergenceDiagnostic` for one replay step by diffing
  the *stored* (matcher-canonicalized) form of the recorded request against
  the same matcher's canonicalization of the live request that failed to
  match it. Volatile fields a `CanonicalizingMatcher` strips before hashing
  (rotating auth, idempotency keys, ...) are therefore excluded from
  `field_diffs` by construction — diffing the canonical forms, not the raw
  bytes, is what keeps a tolerated normalization from reading as divergence.

Nothing here changes what `transport.py` hashes or raises; this is purely a
read-time diagnostic built from a tape and a live `httpx.Request`.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import httpx

    from .matcher import RequestMatcher
    from .tape import Tape

# Sentinel for "this key/index is present on only one side" — a plain string
# (not `None`) so a real recorded/live value of `null` is never confused with
# "absent".
MISSING = "∅"  # ∅


@dataclass(frozen=True)
class FieldDiff:
    """One differing leaf in a structural request-body comparison.

    ``path`` is a JS-like key path (e.g. ``$.messages[0].content``);
    ``recorded``/``live`` are the two JSON-safe values found there, or the
    `MISSING` sentinel when the key/index exists on only one side.
    """

    path: str
    recorded: Any
    live: Any


def _json_or_b64(body: bytes) -> Any:
    """A JSON-serializable view of arbitrary bytes: parsed JSON when possible,
    else a lossless base64 wrapper (mirrors ``matcher._canonical_body``'s
    non-JSON fallback) so unequal non-JSON bytes never diff-compare equal."""
    if not body:
        return None
    try:
        return json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return {"_raw_b64": base64.b64encode(body).decode("ascii")}


def diff_json(recorded: Any, live: Any, path: str = "$") -> list[FieldDiff]:
    """Recursively find every leaf path where ``recorded`` and ``live`` differ.

    Dict keys are compared by name (a key present on only one side reads as
    `MISSING` on the other); lists are compared positionally. Two
    JSON-semantically-equal values yield an empty list even if the bytes they
    were parsed from differed (whitespace/key order) — `json.loads` already
    normalizes that away before this function ever sees it, which is exactly
    what lets a caller distinguish a real content change from non-semantic
    byte drift.
    """
    if isinstance(recorded, dict) and isinstance(live, dict):
        diffs: list[FieldDiff] = []
        for key in sorted(set(recorded) | set(live)):
            child = f"{path}.{key}"
            if key not in recorded:
                diffs.append(FieldDiff(child, MISSING, live[key]))
            elif key not in live:
                diffs.append(FieldDiff(child, recorded[key], MISSING))
            else:
                diffs.extend(diff_json(recorded[key], live[key], child))
        return diffs
    if isinstance(recorded, list) and isinstance(live, list):
        diffs = []
        for i in range(max(len(recorded), len(live))):
            child = f"{path}[{i}]"
            if i >= len(recorded):
                diffs.append(FieldDiff(child, MISSING, live[i]))
            elif i >= len(live):
                diffs.append(FieldDiff(child, recorded[i], MISSING))
            else:
                diffs.extend(diff_json(recorded[i], live[i], child))
        return diffs
    if recorded != live:
        return [FieldDiff(path, recorded, live)]
    return []


def diff_request_bytes(recorded: bytes, live: bytes) -> list[FieldDiff]:
    """Structural diff between two request-body byte strings (JSON or raw)."""
    return diff_json(_json_or_b64(recorded), _json_or_b64(live))


@dataclass(frozen=True)
class DivergenceDiagnostic:
    """A structured, operator-facing explanation of one replay divergence.

    ``is_real_divergence`` is `False` only when the two canonical forms are
    semantically identical despite the fingerprint mismatch — the
    non-semantic byte-level case (whitespace/key order) that only a raw-byte
    matcher (the default `IdentityMatcher`) can produce, since it hashes raw
    bytes rather than a JSON-normalized form. Under a `CanonicalizingMatcher`,
    volatile fields are already stripped from both sides before this
    diagnostic ever runs, so any surfaced `field_diffs` there are always real.
    """

    step_index: int
    recorded_fingerprint: str
    live_fingerprint: str
    matcher_name: str
    normalized_fields: tuple[str, ...]
    field_diffs: tuple[FieldDiff, ...]
    is_real_divergence: bool
    message: str


def diagnose(
    tape: Tape,
    step_index: int,
    live_request: httpx.Request,
    matcher: RequestMatcher | None = None,
) -> DivergenceDiagnostic | None:
    """Build a `DivergenceDiagnostic` for a replay divergence at `step_index`.

    Diffs the recorded exchange's *stored* bytes (exactly what's on the tape
    — canonicalized already, if `matcher` canonicalizes) against
    `matcher.stored_request(live_request)`, i.e. the live request run through
    the identical canonicalization. Returns `None` when there is no
    corresponding recorded exchange to compare against (e.g. the live run
    made more requests than were recorded — `step_index` out of range).
    """
    from .matcher import IDENTITY_MATCHER

    if step_index < 0 or step_index >= len(tape.exchanges):
        return None

    active_matcher = matcher or IDENTITY_MATCHER
    recorded_stored, _resp = tape.exchange(step_index)
    live_stored = active_matcher.stored_request(live_request)

    recorded_fp = active_matcher.stored_fingerprint(recorded_stored)
    live_fp = active_matcher.live_fingerprint(live_request)
    diffs = diff_request_bytes(recorded_stored, live_stored)
    normalized = tuple(sorted(getattr(active_matcher, "volatile_body_fields", frozenset())))
    matcher_name = getattr(active_matcher, "name", "identity")

    is_real = len(diffs) > 0
    if is_real:
        message = f"{len(diffs)} field(s) differ from the recorded request"
    else:
        message = (
            f"fingerprints differ ({recorded_fp[:12]} vs {live_fp[:12]}) but no semantic "
            f"field changed — a non-semantic byte-level difference (whitespace/key order) "
            f"under the '{matcher_name}' matcher, not a real content divergence"
        )

    return DivergenceDiagnostic(
        step_index=step_index,
        recorded_fingerprint=recorded_fp,
        live_fingerprint=live_fp,
        matcher_name=matcher_name,
        normalized_fields=normalized,
        field_diffs=tuple(diffs),
        is_real_divergence=is_real,
        message=message,
    )


def diagnostic_to_dict(diag: DivergenceDiagnostic) -> dict[str, Any]:
    """JSON-safe view of a `DivergenceDiagnostic` for the web report / CLI."""
    return {
        "step_index": diag.step_index,
        "recorded_fingerprint": diag.recorded_fingerprint,
        "live_fingerprint": diag.live_fingerprint,
        "matcher_name": diag.matcher_name,
        "normalized_fields": list(diag.normalized_fields),
        "is_real_divergence": diag.is_real_divergence,
        "message": diag.message,
        "field_diffs": [
            {"path": d.path, "recorded": d.recorded, "live": d.live} for d in diag.field_diffs
        ],
    }
