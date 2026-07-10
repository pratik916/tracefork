"""Shareable per-run trust receipt: a JSON-safe attestation-shaped summary of
a tape's replay/validate/bench evidence, plus a Shields.io-style endpoint
badge derived from it.

`build_trust_receipt()` is pure composition over already-computed evidence —
it never runs blame, never touches the network, and never re-derives numbers
`replay.py`/`validate.py`/`bench.py` don't already produce (offline/$0: the
one thing it re-runs is a fresh `ReplayVerifier.verify()`, itself $0). Its
shape mirrors an in-toto Statement (subject-by-digest + predicate): unsigned
today, upgradeable later to a DSSE-signed envelope without a rewrite. Missing
evidence is always an EXPLICIT ``{"available": False}`` marker, never a
silently-omitted key or a defaulted "verified" state — a receipt must never
overstate what was actually checked.

`build_shield_json()` derives a Shields.io endpoint-schema badge
(https://shields.io/badges/endpoint-badge) from a receipt: green only when
replay was bit-exact AND `validate`'s precision clears
`_HIGH_PRECISION_THRESHOLD`, red when replay ran and detected drift, yellow
otherwise — including any content-redacted tape, which must never badge
green (a redacted tape's replay guarantee is forensic-only, see
`redact.py` / `tracefork-bge.20`). The badge message embeds the receipt's own
`tape_fingerprint` prefix so a stale badge (tape regenerated, badge not
refreshed) is visually detectable rather than silently misleading.
"""

from __future__ import annotations

import datetime as _dt
from typing import TYPE_CHECKING, Any

from .replay import VerificationResult, verification_result_to_dict

if TYPE_CHECKING:
    from .tape import Tape

#: Bumped only on a breaking shape change; consumers should tolerate unknown
#: keys within a major version.
SCHEMA_VERSION = "tracefork/trust-receipt/v1"

# Mirrors the PASS/WARN bar `cli.py`'s `validate` command already prints
# against (`top1_precision >= 0.7`) — not a new threshold, the same one made
# badge-visible.
_HIGH_PRECISION_THRESHOLD = 0.7


def _absent() -> dict[str, Any]:
    """Explicit 'this evidence was not supplied' marker. Used instead of
    omitting the key so a receipt reader can never mistake missing evidence
    for a verified-good state."""
    return {"available": False}


def build_trust_receipt(
    tape: Tape,
    *,
    replay: VerificationResult | None = None,
    validate_report: dict[str, Any] | None = None,
    bench_report: dict[str, Any] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Compose a JSON-safe trust receipt for `tape` from already-computed evidence.

    `replay` is a `ReplayVerifier.verify()` result — converted through
    `verification_result_to_dict`, the exact same conversion `cli.py`'s
    `report` command already applies, never a parallel reimplementation.
    `validate_report` / `bench_report` are the parsed JSON dicts
    `tracefork validate` / `tracefork bench` already write to disk
    (`validation_report.json` / `bench_report.json`) — read and embedded
    verbatim, never recomputed.

    Any of the three left `None` renders as an explicit
    `{"available": False}` marker rather than an omitted key or a defaulted
    claim — a receipt must never overstate what was actually checked.

    `generated_at` defaults to the current UTC time in ISO-8601; pass an
    explicit value for a deterministic receipt (e.g. in tests).
    """
    replay_dict: dict[str, Any]
    if replay is None:
        replay_dict = _absent()
    else:
        replay_dict = {"available": True, **verification_result_to_dict(replay)}

    validate_dict: dict[str, Any] = (
        _absent() if validate_report is None else {"available": True, **validate_report}
    )
    bench_dict: dict[str, Any] = (
        _absent() if bench_report is None else {"available": True, **bench_report}
    )

    return {
        "schemaVersion": SCHEMA_VERSION,
        "tape_fingerprint": tape.digest()[:16],
        "boundary": tape.boundary,
        "content_redacted": tape.content_redacted,
        "replay": replay_dict,
        "validate": validate_dict,
        "bench": bench_dict,
        "generated_at": generated_at or _dt.datetime.now(_dt.UTC).isoformat(),
    }


def build_shield_json(receipt: dict[str, Any]) -> dict[str, Any]:
    """Derive a Shields.io endpoint-schema badge dict from `receipt`.

    Colors, in priority order:
      * ``yellow`` — `receipt["content_redacted"]` is `True`. A redacted
        tape's replay guarantee is forensic-only (see `redact.py`); it must
        never badge green regardless of the other evidence.
      * ``red`` — replay evidence is present and `bit_exact` is `False`
        (a detected divergence).
      * ``brightgreen`` — replay is present and bit-exact AND validate is
        present with `overall_top1_precision` >= `_HIGH_PRECISION_THRESHOLD`.
      * ``yellow`` — every other case (missing replay/validate evidence, or
        validate present but below the precision bar).

    The message always embeds the receipt's `tape_fingerprint` prefix so a
    stale badge (regenerated tape, unrefreshed badge) is visible at a glance.
    """
    replay = receipt["replay"]
    validate = receipt["validate"]
    fp8 = receipt["tape_fingerprint"][:8]

    replay_available = bool(replay.get("available"))
    bit_exact = replay_available and replay.get("bit_exact") is True
    drift_detected = replay_available and replay.get("bit_exact") is False

    validate_available = bool(validate.get("available"))
    precision = validate.get("overall_top1_precision") if validate_available else None
    high_precision = precision is not None and precision >= _HIGH_PRECISION_THRESHOLD

    if receipt["content_redacted"]:
        color = "yellow"
        message = f"content redacted · {fp8}"
    elif drift_detected:
        color = "red"
        message = f"replay drift · {fp8}"
    elif bit_exact and high_precision:
        color = "brightgreen"
        message = f"bit-exact · {precision:.0%} precision · {fp8}"
    else:
        missing = [
            name
            for name, available in (
                ("replay", replay_available),
                ("validate", validate_available),
            )
            if not available
        ]
        message = (
            f"unverified ({', '.join(missing)}) · {fp8}" if missing else f"low precision · {fp8}"
        )
        color = "yellow"

    return {
        "schemaVersion": 1,
        "label": "tracefork",
        "message": message,
        "color": color,
    }
