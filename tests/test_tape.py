import tempfile, os
from tracefork.tape import Tape, sha256_hex

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
