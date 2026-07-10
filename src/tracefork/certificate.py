"""Typed replay certificates: a constructor-enforced ceiling on what a
replay verification is allowed to claim.

`VerificationResult.bit_exact` (see `replay.py`) is a bare boolean a caller
could relay uncritically. `ReplayCertificate` makes the claim's *strength* a
typed value whose `__post_init__` refuses to construct an overclaiming
instance — e.g. `BIT_EXACT_FULL_REPLAY` requires every recorded exchange to
have matched *and* the recorded and replayed tape fingerprints to be
identical, not merely a `True` flag reported by whoever built it. This is a
proof-envelope split (predicate vs. signed claim) in the spirit of
in-toto/SLSA attestations and Certificate-Transparency-style chained witness
records: the claimed strength is checked against the numbers it claims to
summarize on every construction, so a caller cannot report
`BIT_EXACT_FULL_REPLAY` for a partial or hash-only match.

Three tiers, weakest to strongest, mirroring stochastic-LLM reality (a
sampled model's output can be hash-matched without ever being byte-for-byte
replayable end to end):

- `UNVERIFIED` — no claim; never raises regardless of the numbers.
- `HASH_MATCHED` — at least one exchange matched a recorded hash.
- `BIT_EXACT_FULL_REPLAY` — every exchange matched *and* the recorded and
  replayed fingerprints are identical and non-empty.

`certificate_from_verification()` is the sole function in the package that
derives a `ReplayCertificate` from real verification data (an already-run
`ReplayVerifier.verify()` result plus the `Tape` it was checked against).
Tests construct `ReplayCertificate` directly only to exercise the
constructor's guard.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .replay import VerificationResult
    from .tape import Tape


class CertificateStrength(enum.Enum):
    """Reproducibility tiers a `ReplayCertificate` can claim, weakest first."""

    UNVERIFIED = "unverified"
    HASH_MATCHED = "hash_matched"
    BIT_EXACT_FULL_REPLAY = "bit_exact_full_replay"


class ProofEnvelopeError(ValueError):
    """Raised when a `ReplayCertificate`'s claimed strength isn't justified
    by its own `matched`/`total`/fingerprint fields."""


@dataclass(frozen=True)
class ReplayCertificate:
    """A constructor-enforced claim about how strongly a replay was verified.

    `strength` is checked against `matched`/`total`/the fingerprint pair on
    every construction — the ceiling is mechanical, not advisory. See the
    module docstring for exactly what each tier requires; violating one
    raises `ProofEnvelopeError` instead of silently constructing an
    overclaiming certificate.
    """

    strength: CertificateStrength
    matched: int
    total: int
    recorded_fingerprint: str
    replayed_fingerprint: str

    def __post_init__(self) -> None:
        if self.strength is CertificateStrength.UNVERIFIED:
            return
        if self.strength is CertificateStrength.HASH_MATCHED:
            if self.matched <= 0:
                raise ProofEnvelopeError(
                    f"HASH_MATCHED requires matched > 0, got matched={self.matched}"
                )
            return
        # BIT_EXACT_FULL_REPLAY: the headline claim, held to the tightest bar.
        if self.total <= 0 or self.matched != self.total:
            raise ProofEnvelopeError(
                "BIT_EXACT_FULL_REPLAY requires every exchange matched "
                f"(matched={self.matched}, total={self.total})"
            )
        if not self.recorded_fingerprint or self.recorded_fingerprint != self.replayed_fingerprint:
            raise ProofEnvelopeError(
                "BIT_EXACT_FULL_REPLAY requires the recorded and replayed "
                "fingerprints to match and be non-empty "
                f"(recorded={self.recorded_fingerprint!r}, "
                f"replayed={self.replayed_fingerprint!r})"
            )


def certificate_from_verification(result: VerificationResult, tape: Tape) -> ReplayCertificate:
    """Derive a `ReplayCertificate` from an already-computed `VerificationResult`.

    The recorded fingerprint and exchange total are recomputed from `tape`
    itself (`tape.digest()` / `len(tape.exchanges)`) rather than trusted from
    `result`'s own fields, so the certificate is checked against the tape's
    actual content rather than merely echoing back what the verification run
    reported about itself.

    This is the only function in the package that produces a
    `ReplayCertificate` from real verification data.
    """
    recorded_fp = tape.digest()
    total = len(tape.exchanges)
    matched = result.matched
    replayed_fp = result.replayed_fingerprint

    if (
        result.bit_exact
        and result.fingerprints_match
        and total > 0
        and matched == total
        and replayed_fp == recorded_fp
        and recorded_fp != ""
    ):
        strength = CertificateStrength.BIT_EXACT_FULL_REPLAY
    elif matched > 0:
        strength = CertificateStrength.HASH_MATCHED
    else:
        strength = CertificateStrength.UNVERIFIED

    return ReplayCertificate(
        strength=strength,
        matched=matched,
        total=total,
        recorded_fingerprint=recorded_fp,
        replayed_fingerprint=replayed_fp,
    )
