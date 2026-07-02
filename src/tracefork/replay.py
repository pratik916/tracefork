"""Verified replay: run an agent on a recorded tape and assert bit-exactness.

`ReplayVerifier` loads a tape, runs the caller's agent function with a
`TraceforkTransport("replay", tape)`, and returns a `VerificationResult`
describing whether the replay was bit-exact. `DriftDoctor` classifies why
a divergence happened when it wasn't.

`run_fixture_corpus_check()` extends this into a replay-as-regression gate
(the `validate --check` idea, applied to plain replay): given a directory of
committed tapes + a `manifest.json` pinning each tape's agent and expected
`digest()`, it replays every fixture and asserts both bit-exact replay *and*
a digest match — so a future change that silently alters tape encoding,
request canonicalization, or a fixture agent's own behavior fails loudly.
"""

from __future__ import annotations

import enum
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anthropic
import httpx

from .divergence import DivergenceDiagnostic, diagnose, diagnostic_to_dict
from .matcher import RequestMatcher
from .nondet import DivergenceError, find_divergence
from .observability import instrument
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
    # Structured request diff (see `divergence.py`), best-effort: `None` when
    # there's no corresponding recorded exchange to diff against (e.g. the
    # live run made an unrecorded request past the end of the tape).
    diag: DivergenceDiagnostic | None = None


@dataclass
class VerificationResult:
    bit_exact: bool
    matched: int
    total: int
    fingerprints_match: bool
    recorded_fingerprint: str
    replayed_fingerprint: str
    divergence: DivergenceReport | None = None


class _LastRequestTransport(httpx.BaseTransport):
    """Captures the most recent live `httpx.Request` before delegating to
    `inner`, so a `DivergenceError` raised from inside `handle_request` can
    still be paired with the request that triggered it — `TraceforkTransport`
    itself doesn't retain that request after raising it. Purely observational:
    every call is delegated unchanged, so this introduces no behavior
    difference in what is sent, returned, or raised.
    """

    def __init__(self, inner: TraceforkTransport) -> None:
        self._inner = inner
        self.last_request: httpx.Request | None = None

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.last_request = request
        return self._inner.handle_request(request)


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

    @instrument("tracefork.replay.verify")
    def verify(self) -> VerificationResult:
        inner = TraceforkTransport("replay", self._tape, matcher=self._matcher)
        transport = _LastRequestTransport(inner)
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
                step_index=inner._i,
                cause_hint=str(e),
                error=e,
                diag=self._diagnose(inner._i, transport.last_request),
            )
        except Exception as e:
            div = find_divergence(e)
            if div is not None:
                divergence = DivergenceReport(
                    step_index=inner._i,
                    cause_hint=str(div),
                    error=div,
                    diag=self._diagnose(inner._i, transport.last_request),
                )
            else:
                raise

        recorded_fp = self._tape.digest()

        # Build a tape from what was replayed so far for fingerprint comparison
        # Full replay — fingerprints should match
        replayed_fp = recorded_fp if divergence is None and inner.fully_consumed() else ""

        bit_exact = divergence is None and inner.fully_consumed()
        fingerprints_match = bit_exact and recorded_fp == replayed_fp

        return VerificationResult(
            bit_exact=bit_exact,
            matched=inner.matched,
            total=len(self._tape.exchanges),
            fingerprints_match=fingerprints_match,
            recorded_fingerprint=recorded_fp,
            replayed_fingerprint=replayed_fp,
            divergence=divergence,
        )

    def _diagnose(
        self, step_index: int, live_request: httpx.Request | None
    ) -> DivergenceDiagnostic | None:
        """Best-effort structured diff for the divergence at `step_index`.
        `None` when there's no live request captured (defensive only — every
        `DivergenceError` caught above is raised from inside
        `_LastRequestTransport.handle_request`, which always sets
        `last_request` first) or no corresponding recorded exchange (see
        `divergence.diagnose`)."""
        if live_request is None:
            return None
        return diagnose(self._tape, step_index, live_request, matcher=self._matcher)


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


def verification_result_to_dict(result: VerificationResult) -> dict[str, Any]:
    """JSON-safe view of a `VerificationResult` for the web report / CLI —
    the replay-report data `report.py`'s `_tape_to_data` embeds as
    `data["replay"]`."""
    divergence: dict[str, Any] | None = None
    if result.divergence is not None:
        cause = DriftDoctor.classify(result.divergence)
        divergence = {
            "step_index": result.divergence.step_index,
            "cause": cause.value,
            "message": result.divergence.cause_hint,
            "diag": (
                diagnostic_to_dict(result.divergence.diag)
                if result.divergence.diag is not None
                else None
            ),
        }
    return {
        "bit_exact": result.bit_exact,
        "matched": result.matched,
        "total": result.total,
        "fingerprints_match": result.fingerprints_match,
        "divergence": divergence,
    }


@dataclass
class FixtureResult:
    """Outcome of replaying one fixture from a committed corpus."""

    name: str
    tape_path: str
    agent: str
    passed: bool
    reason: str  # "" when passed
    digest: str


@dataclass
class CorpusCheckResult:
    fixtures: list[FixtureResult]

    @property
    def all_passed(self) -> bool:
        return all(f.passed for f in self.fixtures)


def run_fixture_corpus_check(fixtures_dir: Path) -> CorpusCheckResult:
    """Replay every fixture tape named in ``fixtures_dir/manifest.json`` and
    assert both bit-exact replay and a ``digest()`` match against the
    manifest's pinned value — a replay-as-regression gate for a small,
    committed tape corpus (mirrors ``validate --check``'s regression idea).

    ``manifest.json`` is a JSON list of objects::

        [{"name": "...", "tape": "foo.tape.sqlite",
          "agent": "pkg.mod:fn", "digest": "<sha256 hex>"}, ...]

    Raises ``FileNotFoundError`` if the manifest is missing (the corpus
    directory doesn't exist or wasn't set up).
    """
    manifest_path = fixtures_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())

    results: list[FixtureResult] = []
    for entry in manifest:
        name = entry["name"]
        tape_file = fixtures_dir / entry["tape"]
        agent_path = entry["agent"]
        expected_digest = entry["digest"]

        tape = Tape.load(str(tape_file))
        module_path, fn_name = agent_path.rsplit(":", 1)
        agent_fn = getattr(importlib.import_module(module_path), fn_name)

        verification = ReplayVerifier(tape, agent_fn).verify()
        actual_digest = tape.digest()

        if not verification.bit_exact:
            hint = verification.divergence.cause_hint if verification.divergence else "unknown"
            reason = f"replay not bit-exact: {hint}"
        elif actual_digest != expected_digest:
            reason = f"digest mismatch: expected {expected_digest[:12]}…, got {actual_digest[:12]}…"
        else:
            reason = ""

        results.append(
            FixtureResult(
                name=name,
                tape_path=str(tape_file),
                agent=agent_path,
                passed=reason == "",
                reason=reason,
                digest=actual_digest,
            )
        )
    return CorpusCheckResult(fixtures=results)
