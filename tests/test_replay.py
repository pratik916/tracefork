"""Replay + DriftDoctor tests — all offline, no API keys."""

import json
from pathlib import Path

import anthropic
import httpx

from tests.fakes import ScriptedFakeLLM, make_text_response, make_tool_use_response
from tracefork.replay import DriftCause, DriftDoctor, ReplayVerifier, run_fixture_corpus_check
from tracefork.tape import Tape
from tracefork.transport import TraceforkTransport

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "experiments" / "replay_fixtures"

TEXT_RESP = make_text_response("Done.")
TOOL_RESP = make_tool_use_response("book_flight", {"destination": "Tokyo", "seats": 1})


def _record_tape(responses: list[bytes]) -> Tape:
    """Record a tape using ScriptedFakeLLM; return tape."""
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


def test_verifier_passes_on_exact_replay():
    tape = _record_tape([TEXT_RESP])
    result = ReplayVerifier(tape, _agent_fn).verify()
    assert result.bit_exact is True
    assert result.matched == 1
    assert result.total == 1
    assert result.fingerprints_match is True
    assert result.divergence is None


def test_verifier_fails_on_code_change():
    """If the agent builds a different request, replay should diverge."""
    tape = _record_tape([TEXT_RESP])

    def different_agent(client: anthropic.Anthropic) -> str:
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=100,
            messages=[{"role": "user", "content": "Completely different prompt"}],
        )
        return resp.content[0].text

    result = ReplayVerifier(tape, different_agent).verify()
    assert result.bit_exact is False
    assert result.divergence is not None


def test_verifier_matched_count():
    """With two exchanges recorded, both must match for bit_exact=True."""
    fake_rec = ScriptedFakeLLM([TOOL_RESP, TEXT_RESP])
    tape = Tape()
    rec_transport = TraceforkTransport("record", tape, fake_rec)
    client = anthropic.Anthropic(
        api_key="sk-ant-fake",
        http_client=httpx.Client(transport=rec_transport),
        max_retries=0,
    )

    def two_turn_agent(c: anthropic.Anthropic) -> None:
        c.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=100,
            messages=[{"role": "user", "content": "turn1"}],
        )
        c.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=100,
            messages=[
                {"role": "user", "content": "turn1"},
                {"role": "assistant", "content": "..."},
                {"role": "user", "content": "turn2"},
            ],
        )

    two_turn_agent(client)

    result = ReplayVerifier(tape, two_turn_agent).verify()
    assert result.matched == 2
    assert result.bit_exact is True


def test_drift_doctor_classifies_code_change():
    tape = _record_tape([TEXT_RESP])

    def changed_agent(client: anthropic.Anthropic) -> str:
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=100,
            messages=[{"role": "user", "content": "different"}],
        )
        return resp.content[0].text

    result = ReplayVerifier(tape, changed_agent).verify()
    assert result.divergence is not None
    cause = DriftDoctor.classify(result.divergence)
    assert cause == DriftCause.CODE_CHANGE


def test_fingerprints_match_on_exact_replay():
    tape = _record_tape([TEXT_RESP])
    result = ReplayVerifier(tape, _agent_fn).verify()
    assert result.fingerprints_match is True
    assert result.recorded_fingerprint == result.replayed_fingerprint


# ── run_fixture_corpus_check — replay --check's library-level gate ─────────


def test_fixture_corpus_check_passes_on_committed_corpus():
    result = run_fixture_corpus_check(FIXTURES_DIR)
    assert result.fixtures, "expected at least one committed fixture"
    assert result.all_passed, [f.reason for f in result.fixtures if not f.passed]
    for f in result.fixtures:
        assert f.reason == ""
        assert len(f.digest) == 64  # sha256 hex


def test_fixture_corpus_check_fails_on_tampered_tape(tmp_path):
    """Corrupting a committed tape's exchange bytes must fail the gate — either
    because replay is no longer bit-exact, or because the digest no longer
    matches the manifest's pinned value."""
    manifest = json.loads((FIXTURES_DIR / "manifest.json").read_text())
    entry = manifest[0]

    tamper_dir = tmp_path / "fixtures"
    tamper_dir.mkdir()
    tampered_tape_path = tamper_dir / entry["tape"]

    tape = Tape.load(str(FIXTURES_DIR / entry["tape"]))
    req, resp = tape.exchanges[0]
    tape.exchanges[0] = (req, resp + b" ")  # corrupt the recorded response bytes
    tape.save(str(tampered_tape_path))

    # Manifest keeps the ORIGINAL (now-stale) digest — exactly what a real
    # regression (or a corrupted commit) would look like.
    (tamper_dir / "manifest.json").write_text(json.dumps([entry]))

    result = run_fixture_corpus_check(tamper_dir)
    assert result.all_passed is False
    assert result.fixtures[0].passed is False
    assert result.fixtures[0].reason != ""


def test_fixture_corpus_check_passes_after_regenerating_manifest_digest(tmp_path):
    """Sanity check on the tamper test above: if the manifest digest is
    updated to match the (tampered) tape, but the agent-produced request no
    longer matches what's on the tampered tape, the bit-exactness check alone
    must still catch it."""
    manifest = json.loads((FIXTURES_DIR / "manifest.json").read_text())
    entry = manifest[0]

    tamper_dir = tmp_path / "fixtures"
    tamper_dir.mkdir()
    tampered_tape_path = tamper_dir / entry["tape"]

    tape = Tape.load(str(FIXTURES_DIR / entry["tape"]))
    req, resp = tape.exchanges[0]
    tape.exchanges[0] = (req + b" ", resp)  # corrupt the recorded REQUEST bytes
    tape.save(str(tampered_tape_path))

    tampered_entry = dict(entry)
    tampered_entry["digest"] = tape.digest()  # digest matches the tampered tape...
    (tamper_dir / "manifest.json").write_text(json.dumps([tampered_entry]))

    result = run_fixture_corpus_check(tamper_dir)
    # ...but the fixture agent no longer reproduces the (corrupted) request,
    # so replay still isn't bit-exact.
    assert result.all_passed is False
