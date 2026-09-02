import os
import pathlib
import tempfile

import pytest

from tracefork.tape import Tape, sha256_hex

_FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures"


def _make_tape() -> Tape:
    t = Tape(agent_name="test-agent")
    t.draws = [("clock", "2026-01-01T00:00:00+00:00"), ("uuid", "abc123")]
    t.append_exchange(b"request-1", b"response-1")
    t.append_exchange(b"request-2", b"response-2")
    return t


def test_sha256_hex_is_deterministic():
    assert sha256_hex(b"hello") == sha256_hex(b"hello")
    assert sha256_hex(b"hello") != sha256_hex(b"world")


def test_digest_is_deterministic():
    assert _make_tape().digest() == _make_tape().digest()


def test_digest_changes_on_draws():
    t1 = _make_tape()
    t2 = _make_tape()
    t2.draws[0] = ("clock", "2026-01-02T00:00:00+00:00")
    assert t1.digest() != t2.digest()


def test_digest_changes_on_exchange():
    t1 = _make_tape()
    t2 = _make_tape()
    t2.exchanges[0] = (b"different", b"response-1")
    assert t1.digest() != t2.digest()


def test_save_load_roundtrip():
    tape = _make_tape()
    with tempfile.NamedTemporaryFile(suffix=".tape.sqlite", delete=False) as f:
        path = f.name
    try:
        tape.save(path)
        loaded = Tape.load(path)
        assert loaded.digest() == tape.digest()
        assert loaded.draws == tape.draws
        assert loaded.exchanges == tape.exchanges
        assert loaded.agent_name == tape.agent_name
        assert loaded.boundary == tape.boundary
    finally:
        os.unlink(path)


def test_dedup_identical_blobs():
    tape = Tape()
    tape.append_exchange(b"same-request", b"same-response")
    tape.append_exchange(b"same-request", b"same-response")
    with tempfile.NamedTemporaryFile(suffix=".tape.sqlite", delete=False) as f:
        path = f.name
    try:
        tape.save(path)
        import sqlite3

        con = sqlite3.connect(path)
        blob_count = con.execute("SELECT COUNT(*) FROM blobs").fetchone()[0]
        con.close()
        # 2 unique blobs (same-request, same-response) not 4
        assert blob_count == 2
    finally:
        os.unlink(path)


def test_meta_roundtrip():
    tape = Tape(agent_name="my-agent", boundary="single-process-asyncio-v1")
    with tempfile.NamedTemporaryFile(suffix=".tape.sqlite", delete=False) as f:
        path = f.name
    try:
        tape.save(path)
        loaded = Tape.load(path)
        assert loaded.agent_name == "my-agent"
        assert loaded.boundary == "single-process-asyncio-v1"
    finally:
        os.unlink(path)


def test_to_bytes_from_bytes_roundtrip():
    tape = _make_tape()
    data = tape.to_bytes()
    restored = Tape.from_bytes(data)
    assert restored.digest() == tape.digest()
    assert restored.draws == tape.draws
    assert restored.exchanges == tape.exchanges
    assert restored.agent_name == tape.agent_name


# ── provenance (v5): matcher/boundary_guard/nondet-mode witness block ───────


def test_provenance_roundtrips_through_to_bytes_from_bytes_exactly():
    tape = _make_tape()
    tape.provenance = {
        "matcher_name": "identity",
        "boundary_guard": "false",
        "nondet_mode": "recording",
    }
    restored = Tape.from_bytes(tape.to_bytes())
    assert restored.provenance == tape.provenance


def test_digest_excludes_provenance():
    """Two tapes identical except for `provenance` must hash EQUAL — the single
    most load-bearing invariant for this field (explicit test, not implied by
    the general digest tests above)."""
    t1 = _make_tape()
    t2 = _make_tape()
    t2.provenance = {
        "matcher_name": "bedrock",
        "boundary_guard": "true",
        "nondet_mode": "recording",
    }
    assert t1.provenance != t2.provenance
    assert t1.digest() == t2.digest()


# ── item 4: draw-framing hash-chain collision ────────────────────────────────


def test_digest_does_not_collide_on_ambiguous_draw_framing():
    """Reproduces the exact collision from the readiness plan: an unescaped,
    variable-length 'D:kind:value\\n' delimiter lets a draw VALUE containing an
    embedded 'D:'-prefixed line forge what looks like a second draw record, so
    two structurally different tapes hashed equal. Framing each field through a
    fixed-width digest (as exchanges already do) must make them differ."""
    t1 = Tape(draws=[("env", "1\x00MY\x00a\nD:clock:2020-01-01T00:00:00")])
    t2 = Tape(draws=[("env", "1\x00MY\x00a"), ("clock", "2020-01-01T00:00:00")])
    assert t1.digest() != t2.digest()


def test_digest_no_collision_for_adversarial_draw_values():
    """Property-style check: draw values containing the framing delimiters
    themselves (':', '\\n', 'D:', 'X:', 'T:') must not let one draw list's
    digest collide with a structurally different draw list's digest."""
    adversarial_lists: list[list[tuple[str, str]]] = [
        [("env", "a\nD:clock:b")],
        [("env", "a"), ("clock", "b")],
        [("env", "a:b\nX:c:d")],
        [("env", "a:b"), ("__x", "c:d")],
        [("clock", "a\nT:x:y")],
        [("clock", "a"), ("__t", "x:y")],
        [("k", "v")],
        [("k:v", "")],
        [("", "k:v")],
        [("a", "b"), ("c", "d")],
        [("a:b", "c"), ("d", "")],
    ]
    digests = []
    for draws in adversarial_lists:
        t = Tape(draws=draws)
        digests.append(t.digest())
    for i in range(len(digests)):
        for j in range(i + 1, len(digests)):
            assert digests[i] != digests[j], (
                f"draw lists {adversarial_lists[i]!r} and "
                f"{adversarial_lists[j]!r} collided: {digests[i]}"
            )


def test_digest_still_deterministic_and_sensitive_after_framing_fix():
    """The framing fix must not weaken the existing digest properties: same
    content -> same digest, and a real content change still changes it."""
    assert _make_tape().digest() == _make_tape().digest()
    t1 = _make_tape()
    t2 = _make_tape()
    t2.draws[0] = ("clock", "2026-01-02T00:00:00+00:00")
    assert t1.digest() != t2.digest()


# ── item 5: provenance / request_urls persistence through save()/load() ─────


def test_provenance_and_request_urls_roundtrip_through_save_load():
    tape = _make_tape()
    tape.provenance = {
        "matcher_name": "redacting",
        "boundary_guard": "true",
        "nondet_mode": "recording",
    }
    tape.request_urls = ["https://api.example/v1/messages", "https://api.example/v1/messages"]
    with tempfile.NamedTemporaryFile(suffix=".tape.sqlite", delete=False) as f:
        path = f.name
    try:
        tape.save(path)
        loaded = Tape.load(path)
        assert loaded.provenance == tape.provenance
        assert loaded.request_urls == tape.request_urls
        # digest must still be unaffected by either field (metadata, per invariant)
        assert loaded.digest() == tape.digest()
    finally:
        os.unlink(path)


def test_load_legacy_db_without_provenance_or_request_url_meta_rows():
    """A `.tape.sqlite` written before this fix (or by any code path that
    still only writes the five original meta rows) has no `provenance` /
    `request_urls` meta rows at all. `load()` must upcast such a DB to the
    documented defaults instead of raising."""
    import sqlite3

    tape = _make_tape()
    with tempfile.NamedTemporaryFile(suffix=".tape.sqlite", delete=False) as f:
        path = f.name
    try:
        tape.save(path)
        # Simulate a legacy DB by deleting the new meta rows this fix adds.
        con = sqlite3.connect(path)
        con.execute("DELETE FROM meta WHERE key IN ('provenance', 'request_urls')")
        con.commit()
        con.close()

        loaded = Tape.load(path)
        assert loaded.provenance == {}
        assert loaded.request_urls == [""] * len(loaded.exchanges)
    finally:
        os.unlink(path)


def test_replay_verifier_raises_provenance_mismatch_after_save_load_roundtrip():
    """The acceptance test from the readiness plan: record a tape with a
    non-default matcher, save it to disk, reload it, and confirm
    `ReplayVerifier.verify()` raises `ProvenanceMismatchError` up front rather
    than the loaded tape silently losing its provenance block and the
    mismatch instead surfacing as a generic byte divergence."""
    from tracefork.replay import ProvenanceMismatchError, ReplayVerifier

    tape = _make_tape()
    tape.provenance = {"matcher_name": "redacting", "boundary_guard": "false"}
    with tempfile.NamedTemporaryFile(suffix=".tape.sqlite", delete=False) as f:
        path = f.name
    try:
        tape.save(path)
        loaded = Tape.load(path)
        verifier = ReplayVerifier(loaded, lambda client: None)  # default matcher: identity
        with pytest.raises(ProvenanceMismatchError):
            verifier.verify()
    finally:
        os.unlink(path)


# ── golden v2/v3 tape-format fixtures ───────────────────────────────────────
#
# tests/fixtures/legacy_tape_v{2,3}.blob are committed binary fixtures for the
# historical v2 (content-addressed zstd container, pre-tool-log) and v3 (adds
# the JSON-RPC tool-exchange log, pre-concurrency-batch-log) on-disk envelope
# formats. They were hand-built with struct/json/hashlib/zstandard directly
# (see the generator this test module's sibling comment references),
# independent of tape.py's current `_encode_v6` -- unlike `legacy_tape_v1.blob`
# (tested in test_storage.py), which pins the v1 JSON+base64 format, these pin
# the v2/v3 binary-container decode branches (`_decode_v2_binary`/
# `_decode_v3_binary`), which until now were only exercised indirectly via
# hand-synthesized bytes from the *current* encoder.


def test_golden_v2_fixture_decodes_via_upcaster_chain():
    data = (_FIXTURES_DIR / "legacy_tape_v2.blob").read_bytes()
    tape = Tape.from_bytes(data)
    assert tape.boundary == "single-process-asyncio-v1"
    assert tape.agent_name == "golden-agent-v2"
    assert tape.draws == [("clock", "2026-01-01T00:00:00+00:00"), ("uuid", "abc123")]
    assert tape.exchanges == [
        (b"request-1", b"response-1"),
        (b"request-2", b"response-2"),
        (b"request-1", b"response-1"),
    ]
    # v2 predates the tool log / concurrency-batch log / provenance /
    # request-URL witness log -- all upcast to their documented empty default.
    assert tape.tool_exchanges == []
    assert tape.async_batches == []
    assert tape.provenance == {}
    assert tape.request_urls == [""] * len(tape.exchanges)
    assert tape.content_redacted is False


def test_golden_v2_fixture_digest_matches_equivalent_fresh_tape():
    data = (_FIXTURES_DIR / "legacy_tape_v2.blob").read_bytes()
    loaded = Tape.from_bytes(data)
    fresh = Tape(agent_name="golden-agent-v2", boundary="single-process-asyncio-v1")
    fresh.draws = [("clock", "2026-01-01T00:00:00+00:00"), ("uuid", "abc123")]
    fresh.append_exchange(b"request-1", b"response-1")
    fresh.append_exchange(b"request-2", b"response-2")
    fresh.append_exchange(b"request-1", b"response-1")
    assert loaded.digest() == fresh.digest()


def test_golden_v3_fixture_decodes_via_upcaster_chain():
    data = (_FIXTURES_DIR / "legacy_tape_v3.blob").read_bytes()
    tape = Tape.from_bytes(data)
    assert tape.boundary == "single-process-asyncio-v1"
    assert tape.agent_name == "golden-agent-v3"
    assert tape.draws == [("clock", "2026-01-01T00:00:00+00:00"), ("uuid", "def456")]
    assert tape.exchanges == [
        (b"request-1", b"response-1"),
        (b"request-2", b"response-2"),
    ]
    # v3 introduces the tool log but predates the concurrency-batch log /
    # provenance / request-URL witness log -- those upcast to empty defaults.
    assert tape.tool_exchanges == [(b"tool-request-1", b"tool-response-1")]
    assert tape.async_batches == []
    assert tape.provenance == {}
    assert tape.request_urls == [""] * len(tape.exchanges)
    assert tape.content_redacted is False


def test_golden_v3_fixture_digest_matches_equivalent_fresh_tape():
    data = (_FIXTURES_DIR / "legacy_tape_v3.blob").read_bytes()
    loaded = Tape.from_bytes(data)
    fresh = Tape(agent_name="golden-agent-v3", boundary="single-process-asyncio-v1")
    fresh.draws = [("clock", "2026-01-01T00:00:00+00:00"), ("uuid", "def456")]
    fresh.append_exchange(b"request-1", b"response-1")
    fresh.append_exchange(b"request-2", b"response-2")
    fresh.append_tool_exchange(b"tool-request-1", b"tool-response-1")
    assert loaded.digest() == fresh.digest()


def test_golden_v2_and_v3_fixtures_carry_their_declared_envelope_version():
    from tracefork.constants import TAPE_MAGIC

    v2 = (_FIXTURES_DIR / "legacy_tape_v2.blob").read_bytes()
    v3 = (_FIXTURES_DIR / "legacy_tape_v3.blob").read_bytes()
    assert v2[: len(TAPE_MAGIC)] == TAPE_MAGIC
    assert v3[: len(TAPE_MAGIC)] == TAPE_MAGIC
    import struct

    (v2_version,) = struct.unpack_from(">H", v2, len(TAPE_MAGIC))
    (v3_version,) = struct.unpack_from(">H", v3, len(TAPE_MAGIC))
    assert v2_version == 2
    assert v3_version == 3
