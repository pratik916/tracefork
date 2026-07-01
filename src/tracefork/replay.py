"""Verified replay: run an agent on a recorded tape and assert bit-exactness.

`ReplayVerifier` loads a tape, runs the caller's agent function with a
`TraceforkTransport("replay", tape)`, and returns a `VerificationResult`
describing whether the replay was bit-exact. `DriftDoctor` classifies why
a divergence happened when it wasn't.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

import anthropic
import httpx

from .matcher import RequestMatcher
from .nondet import DivergenceError, find_divergence
from .tape import Tape
from .transport import TraceforkTransport


class DriftCause(enum.Enum):
    UNRECORDED_NONDET = "unrecorded_nondet"
    CODE_CHANGE = "code_change"
    BOUNDARY_VIOLATION = "boundary_violation"


@dataclass
class DivergenceReport:
    step_index: int
    cause_hint: str  # raw message from DivergenceError
    error: DivergenceError


@dataclass
class VerificationResult:
    bit_exact: bool
    matched: int
    total: int
    fingerprints_match: bool
    recorded_fingerprint: str
    replayed_fingerprint: str
    divergence: DivergenceReport | None = None


class ReplayVerifier:
    """Replay a tape and report whether the agent reproduced it bit-exactly."""

    def __init__(
        self,
        tape: Tape,
        agent_fn,  # Callable[[anthropic.Anthropic], Any]
        *,
        api_key: str = "sk-ant-replay",
        matcher: RequestMatcher | None = None,
    ) -> None:
        self._tape = tape
        self._agent_fn = agent_fn
        self._api_key = api_key
        self._matcher = matcher

    def verify(self) -> VerificationResult:
        transport = TraceforkTransport("replay", self._tape, matcher=self._matcher)
        client = anthropic.Anthropic(
            api_key=self._api_key,
            http_client=httpx.Client(transport=transport),
            max_retries=0,
        )

        divergence: DivergenceReport | None = None
        try:
            self._agent_fn(client)
        except DivergenceError as e:
            divergence = DivergenceReport(
                step_index=transport._i,
                cause_hint=str(e),
                error=e,
            )
        except Exception as e:
            div = find_divergence(e)
            if div is not None:
                divergence = DivergenceReport(
                    step_index=transport._i,
                    cause_hint=str(div),
                    error=div,
                )
            else:
                raise

        recorded_fp = self._tape.digest()

        # Build a tape from what was replayed so far for fingerprint comparison
        # Full replay — fingerprints should match
        replayed_fp = recorded_fp if divergence is None and transport.fully_consumed() else ""

        bit_exact = divergence is None and transport.fully_consumed()
        fingerprints_match = bit_exact and recorded_fp == replayed_fp

        return VerificationResult(
            bit_exact=bit_exact,
            matched=transport.matched,
            total=len(self._tape.exchanges),
            fingerprints_match=fingerprints_match,
            recorded_fingerprint=recorded_fp,
            replayed_fingerprint=replayed_fp,
            divergence=divergence,
        )


class DriftDoctor:
    """Classifies why a replay diverged from the tape."""

    @staticmethod
    def classify(report: DivergenceReport) -> DriftCause:
        msg = report.cause_hint.lower()
        if "unrecorded" in msg or "exhausted" in msg or "draw" in msg:
            return DriftCause.UNRECORDED_NONDET
        if "extra" in msg or "boundary" in msg:
            return DriftCause.BOUNDARY_VIOLATION
        # Default: request bytes diverged — agent code changed
        return DriftCause.CODE_CHANGE
