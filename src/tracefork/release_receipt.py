"""Shareable per-*release* trust receipt: a JSON-safe, content-addressed,
optionally-signed attestation composing a release's already-computed
evidence — test/coverage summaries, `validate`/`bench` reports, a fresh
replay-fixture-corpus check, and a fresh CI-calibration sweep — into one
document.

`build_release_receipt()` mirrors `receipt.py`'s exact philosophy at the
repo/release level instead of the tape level: pure composition over
already-computed (or freshly, $0-computed) evidence, never a parallel
reimplementation of anything `validate.py`/`bench.py`/`replay.py`/
`ci_calibration.py` already produce. Missing evidence is always an EXPLICIT
``{"available": False}`` marker, never a silently-omitted key or a defaulted
"passing" claim — a receipt must never overstate what was actually checked.
`test_summary`/`coverage_summary` are dicts parsed off disk by
`parse_junit_test_summary`/`parse_coverage_summary`; `validate_report`/
`bench_report` are the raw dicts `tracefork validate`/`tracefork bench`
already write; `replay_corpus`/`calibration` are the REAL
`replay.CorpusCheckResult`/`ci_calibration.CalibrationReport` dataclasses,
shaped into JSON-safe dicts by `corpus_check_result_to_dict`/
`calibration_report_to_dict` — mirroring `replay.py`'s own
`verification_result_to_dict` conversion pattern exactly, never a duplicate
of the dataclasses' own logic.

The composed body (everything except `receipt_digest` itself) is hashed via
canonical ``json.dumps(sort_keys=True)`` -> sha256 into ``receipt_digest`` —
the same Merkle-style content-address idiom as `tape.Tape.digest()` /
`fork.py`'s `branch_digest`: identical inputs always produce an identical
digest, byte-stable across builds.

`sign_release_receipt()` HMAC-SHA256-signs `receipt_digest` when a
`signing_key` (e.g. from ``TRACEFORK_RELEASE_SIGNING_KEY``) is supplied;
``signing_key=None`` (the default — unset env var) yields an explicit
``{"available": False}`` signature marker, never a silently-omitted
"signed" claim. This is documented HONESTLY as a symmetric HMAC
attestation — a "was this exact receipt produced by someone holding the
shared key" check — **not** a DSSE/asymmetric signature or a public,
independently-verifiable proof; a real PKI/secrets-provisioning signing
tier is out of scope here (avoids that scope-blow while staying
offline/$0, stdlib `hmac` + `hashlib` only). Same "typed ceiling, never
overclaim" discipline as `certificate.py`'s `ReplayCertificate`.
`verify_release_receipt_signature()` checks a signature back against a
candidate key.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import hmac
import json
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from .ci_calibration import CalibrationReport
from .replay import CorpusCheckResult

#: Bumped only on a breaking shape change; consumers should tolerate unknown
#: keys within a major version.
SCHEMA_VERSION = "tracefork/release-receipt/v1"

#: The one HMAC algorithm this module signs with — see the module docstring
#: for why this is an honest symmetric attestation, not an asymmetric
#: signature.
_SIGNATURE_ALGORITHM = "HMAC-SHA256"


def _absent() -> dict[str, Any]:
    """Explicit 'this evidence was not supplied' marker. Used instead of
    omitting the key so a receipt reader can never mistake missing evidence
    for a verified-good state. Mirrors `receipt.py`'s `_absent()` exactly."""
    return {"available": False}


def parse_junit_test_summary(path: Path) -> dict[str, Any]:
    """Parse a JUnit XML report's own top-level ``<testsuite>`` counts.

    Accepts both a bare ``<testsuite ...>`` root (some runners) and a
    ``<testsuites><testsuite .../>...</testsuites>`` root (pytest's
    ``--junit-xml``), summing counts across every child ``<testsuite>`` in
    the wrapped form. Deliberately does NOT reparse individual
    ``<testcase>`` elements — that per-testcase cross-check against a
    required-id manifest is `scripts/check_executed_evidence.py`'s job, not
    this summary parser's.

    Returns ``{"tests": int, "failures": int, "errors": int, "skipped": int,
    "time": float}``.

    Uses stdlib ``xml.etree.ElementTree`` deliberately (no new dependency),
    the same choice `scripts/check_executed_evidence.py` already documents:
    ``path`` is a ``junit.xml`` pytest itself just wrote in the same
    CI/release run that invokes this parser, never an externally-sourced or
    attacker-controlled document, so the XXE/billion-laughs trust boundary
    `defusedxml` guards against does not apply here.
    """
    root = ET.parse(path).getroot()
    suites = root.findall("testsuite") if root.tag == "testsuites" else [root]
    return {
        "tests": sum(int(s.get("tests", 0)) for s in suites),
        "failures": sum(int(s.get("failures", 0)) for s in suites),
        "errors": sum(int(s.get("errors", 0)) for s in suites),
        "skipped": sum(int(s.get("skipped", 0)) for s in suites),
        "time": sum(float(s.get("time", 0.0)) for s in suites),
    }


def parse_coverage_summary(path: Path) -> dict[str, Any]:
    """Read the ``totals`` block of a ``coverage json`` report (the
    `coverage.py` tool's own JSON output format — unrelated to this
    package's `coverage.py` determinism-coverage module).

    Returns ``{"percent_covered": float, "num_statements": int,
    "covered_lines": int, "missing_lines": int}``, read verbatim from the
    report's ``totals`` block — never recomputed.
    """
    data = json.loads(path.read_text())
    totals = data["totals"]
    return {
        "percent_covered": totals["percent_covered"],
        "num_statements": totals["num_statements"],
        "covered_lines": totals["covered_lines"],
        "missing_lines": totals["missing_lines"],
    }


def corpus_check_result_to_dict(result: CorpusCheckResult) -> dict[str, Any]:
    """JSON-safe view of a `replay.CorpusCheckResult`, mirroring
    `replay.py`'s own `verification_result_to_dict` conversion pattern."""
    return {
        "all_passed": result.all_passed,
        "fixtures": [
            {
                "name": f.name,
                "tape_path": f.tape_path,
                "agent": f.agent,
                "passed": f.passed,
                "reason": f.reason,
                "digest": f.digest,
            }
            for f in result.fixtures
        ],
    }


def calibration_report_to_dict(report: CalibrationReport) -> dict[str, Any]:
    """JSON-safe view of a `ci_calibration.CalibrationReport`, mirroring
    `replay.py`'s `verification_result_to_dict` conversion pattern."""
    return {
        "seed": report.seed,
        "all_within_tolerance": report.all_within_tolerance(),
        "results": [
            {
                "method": r.method.value,
                "true_p": r.true_p,
                "n_trials": r.n_trials,
                "n_repeats": r.n_repeats,
                "confidence": r.confidence,
                "coverage": r.coverage,
                "tolerance": r.tolerance,
                "within_tolerance": r.within_tolerance,
            }
            for r in report.results
        ],
    }


def build_release_receipt(
    *,
    version: str,
    test_summary: dict[str, Any] | None = None,
    coverage_summary: dict[str, Any] | None = None,
    validate_report: dict[str, Any] | None = None,
    bench_report: dict[str, Any] | None = None,
    replay_corpus: CorpusCheckResult | None = None,
    calibration: CalibrationReport | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Compose a JSON-safe, content-addressed release receipt from
    already-computed (or freshly, $0-computed) evidence.

    `test_summary`/`coverage_summary` are dicts produced by
    `parse_junit_test_summary`/`parse_coverage_summary`; `validate_report`/
    `bench_report` are the parsed JSON dicts `tracefork validate`/
    `tracefork bench` already write to disk, read and embedded verbatim.
    `replay_corpus`/`calibration` are the REAL `CorpusCheckResult`/
    `CalibrationReport` dataclasses, converted via
    `corpus_check_result_to_dict`/`calibration_report_to_dict`.

    Any of the six left `None` renders as an explicit
    `{"available": False}` marker rather than an omitted key or a defaulted
    claim — a receipt must never overstate what was actually checked.

    `generated_at` defaults to the current UTC time in ISO-8601; pass an
    explicit value for a byte-stable, reproducible receipt (e.g. in tests).

    The returned dict's `receipt_digest` is the sha256 hex digest of the
    canonical (``sort_keys=True``) JSON encoding of every OTHER field — a
    Merkle-style content address, the same idiom as `tape.Tape.digest()` /
    `fork.py`'s `branch_digest`: identical inputs always produce an
    identical digest.
    """
    test_dict = _absent() if test_summary is None else {"available": True, **test_summary}
    coverage_dict = (
        _absent() if coverage_summary is None else {"available": True, **coverage_summary}
    )
    validate_dict = _absent() if validate_report is None else {"available": True, **validate_report}
    bench_dict = _absent() if bench_report is None else {"available": True, **bench_report}
    replay_corpus_dict = (
        _absent()
        if replay_corpus is None
        else {"available": True, **corpus_check_result_to_dict(replay_corpus)}
    )
    calibration_dict = (
        _absent()
        if calibration is None
        else {"available": True, **calibration_report_to_dict(calibration)}
    )

    body: dict[str, Any] = {
        "schemaVersion": SCHEMA_VERSION,
        "version": version,
        "test": test_dict,
        "coverage": coverage_dict,
        "validate": validate_dict,
        "bench": bench_dict,
        "replay_corpus": replay_corpus_dict,
        "calibration": calibration_dict,
        "generated_at": generated_at or _dt.datetime.now(_dt.UTC).isoformat(),
    }
    receipt_digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()
    return {**body, "receipt_digest": receipt_digest}


def sign_release_receipt(receipt: dict[str, Any], *, signing_key: bytes | None) -> dict[str, Any]:
    """Return a copy of `receipt` with a ``signature`` field added.

    ``signing_key=None`` (e.g. ``TRACEFORK_RELEASE_SIGNING_KEY`` unset)
    yields an explicit ``{"available": False}`` marker — never a silently
    omitted key or a defaulted "signed" claim. Otherwise HMAC-SHA256-signs
    `receipt["receipt_digest"]` and returns
    ``{"available": True, "algorithm": "HMAC-SHA256", "value": <hex mac>}``.

    An HMAC is a SYMMETRIC attestation — "produced by someone holding this
    shared key" — not a DSSE/asymmetric signature; see the module
    docstring.
    """
    if signing_key is None:
        return {**receipt, "signature": _absent()}
    mac = hmac.new(
        signing_key, receipt["receipt_digest"].encode("utf-8"), hashlib.sha256
    ).hexdigest()
    return {
        **receipt,
        "signature": {
            "available": True,
            "algorithm": _SIGNATURE_ALGORITHM,
            "value": mac,
        },
    }


def verify_release_receipt_signature(receipt: dict[str, Any], *, signing_key: bytes) -> bool:
    """Check `receipt["signature"]` against `signing_key`.

    Returns `False` when the signature is absent, when `receipt_digest` was
    mutated after signing (the recomputed HMAC no longer matches), or when
    `signing_key` doesn't match the key the receipt was actually signed
    with. Uses `hmac.compare_digest` for a constant-time comparison.
    """
    signature = receipt.get("signature")
    if not signature or not signature.get("available"):
        return False
    expected = hmac.new(
        signing_key, receipt["receipt_digest"].encode("utf-8"), hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature.get("value", ""))
