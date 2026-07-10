"""ReplayCertificate / proof-envelope tests — all offline, no API keys.

Covers the constructor-enforced ceiling (a caller cannot construct a
`ReplayCertificate` whose claimed `CertificateStrength` overclaims what its
own matched/total/fingerprint fields justify) and `certificate_from_verification`,
the sole function that derives a certificate from real verification data.
"""

import anthropic
import httpx
import pytest

from tests.fakes import ScriptedFakeLLM, make_text_response
from tracefork.certificate import (
    CertificateStrength,
    ProofEnvelopeError,
    ReplayCertificate,
    certificate_from_verification,
)
from tracefork.nondet import DriftingNondet, RecordingNondet
from tracefork.replay import ReplayVerifier
from tracefork.tape import Tape
from tracefork.transport import TraceforkTransport

TEXT_RESP = make_text_response("Done.")


def _record_tape(responses: list[bytes]) -> Tape:
    fake = ScriptedFakeLLM(responses)
    tape = Tape()
    transport = TraceforkTransport("record", tape, fake)
    client = anthropic.Anthropic(
        api_key="sk-ant-fake",
        http_client=httpx.Client(transport=transport),
        max_retries=0,
    )
    client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=100,
        messages=[{"role": "user", "content": "Hello"}],
    )
    return tape


def _agent_fn(client: anthropic.Anthropic) -> str:
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=100,
        messages=[{"role": "user", "content": "Hello"}],
    )
    return resp.content[0].text


# ── constructor-enforced ceiling ────────────────────────────────────────────


def test_bit_exact_full_replay_succeeds_when_fully_matched_and_fingerprints_equal():
    cert = ReplayCertificate(
        strength=CertificateStrength.BIT_EXACT_FULL_REPLAY,
        matched=3,
        total=3,
        recorded_fingerprint="abc123",
        replayed_fingerprint="abc123",
    )
    assert cert.strength is CertificateStrength.BIT_EXACT_FULL_REPLAY


def test_bit_exact_full_replay_raises_when_matched_less_than_total():
    with pytest.raises(ProofEnvelopeError):
        ReplayCertificate(
            strength=CertificateStrength.BIT_EXACT_FULL_REPLAY,
            matched=3,
            total=5,
            recorded_fingerprint="abc123",
            replayed_fingerprint="abc123",
        )


def test_bit_exact_full_replay_raises_on_mismatched_fingerprints_even_if_fully_matched():
    with pytest.raises(ProofEnvelopeError):
        ReplayCertificate(
            strength=CertificateStrength.BIT_EXACT_FULL_REPLAY,
            matched=3,
            total=3,
            recorded_fingerprint="abc123",
            replayed_fingerprint="def456",
        )


def test_hash_matched_raises_when_matched_is_zero():
    with pytest.raises(ProofEnvelopeError):
        ReplayCertificate(
            strength=CertificateStrength.HASH_MATCHED,
            matched=0,
            total=3,
            recorded_fingerprint="abc123",
            replayed_fingerprint="",
        )


def test_hash_matched_succeeds_with_at_least_one_match():
    cert = ReplayCertificate(
        strength=CertificateStrength.HASH_MATCHED,
        matched=1,
        total=3,
        recorded_fingerprint="abc123",
        replayed_fingerprint="",
    )
    assert cert.strength is CertificateStrength.HASH_MATCHED


def test_unverified_never_raises_regardless_of_numbers():
    # Numbers that would blow up any other tier are fine for UNVERIFIED — it
    # makes no claim at all.
    cert = ReplayCertificate(
        strength=CertificateStrength.UNVERIFIED,
        matched=0,
        total=0,
        recorded_fingerprint="",
        replayed_fingerprint="mismatch",
    )
    assert cert.strength is CertificateStrength.UNVERIFIED


# ── certificate_from_verification: the sole real-data producer ─────────────


def test_certificate_from_verification_on_clean_replay_yields_bit_exact_full_replay():
    tape = _record_tape([TEXT_RESP])
    result = ReplayVerifier(tape, _agent_fn).verify()
    cert = certificate_from_verification(result, tape)
    assert cert.strength is CertificateStrength.BIT_EXACT_FULL_REPLAY
    assert cert.matched == cert.total == 1
    assert cert.recorded_fingerprint == cert.replayed_fingerprint


def test_certificate_from_verification_on_negative_control_is_not_bit_exact():
    """Proof-not-assert, applied to the certificate: replaying with
    `DriftingNondet` (fresh draws every call, the package's negative control)
    MUST NOT yield `BIT_EXACT_FULL_REPLAY` — if it did, the certificate's
    headline claim would be as vacuous as a bare unchecked boolean."""

    def nondet_agent_factory(nondet):
        def agent(client: anthropic.Anthropic) -> str:
            v = nondet.random_float()
            resp = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=100,
                messages=[{"role": "user", "content": f"roll: {v.hex()}"}],
            )
            return resp.content[0].text

        return agent

    rec_nondet = RecordingNondet()
    tape = _record_tape_with_nondet_agent(nondet_agent_factory(rec_nondet))
    tape.draws = rec_nondet.draws

    result = ReplayVerifier(tape, nondet_agent_factory(DriftingNondet())).verify()
    cert = certificate_from_verification(result, tape)

    assert cert.strength is not CertificateStrength.BIT_EXACT_FULL_REPLAY


def _record_tape_with_nondet_agent(agent) -> Tape:
    fake = ScriptedFakeLLM([TEXT_RESP])
    tape = Tape()
    transport = TraceforkTransport("record", tape, fake)
    client = anthropic.Anthropic(
        api_key="sk-ant-fake",
        http_client=httpx.Client(transport=transport),
        max_retries=0,
    )
    agent(client)
    return tape
