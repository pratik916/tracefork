"""nondet.py — direct class tests for the file channel (`read_file`),
mirroring tests/test_nondet.py's pattern for the random channel: record/
replay round-trip, kind- and path-mismatch divergence, size-cap enforcement,
DriftingNondet freshness, interleaving with the other four kinds, and
coverage.py's tally of the new "read_file" draw kind."""

from __future__ import annotations

import base64
import hashlib
import json
import random

import pytest

from tracefork.coverage import tape_draw_coverage
from tracefork.nondet import (
    DivergenceError,
    DriftingNondet,
    ReadFileTooLargeError,
    RecordingNondet,
    ReplayNondet,
)
from tracefork.tape import Tape

# ── Direct class tests ──────────────────────────────────────────────────────


def test_recording_nondet_read_file_logs_path_size_sha256_and_returns_real_bytes(
    tmp_path,
):
    p = tmp_path / "config.json"
    p.write_bytes(b'{"key": "value"}')

    nd = RecordingNondet()
    data = nd.read_file(str(p))

    assert data == b'{"key": "value"}'
    assert len(nd.draws) == 1
    kind, packed = nd.draws[0]
    assert kind == "read_file"
    envelope = json.loads(packed)
    assert envelope["path"] == str(p)
    assert envelope["size"] == len(data)
    assert envelope["sha256"] == hashlib.sha256(data).hexdigest()
    assert base64.b64decode(envelope["content_b64"]) == data


def test_read_file_record_replay_round_trip_is_exact(tmp_path):
    p = tmp_path / "state.bin"
    p.write_bytes(b"\x00\x01\xff\xfe binary content here")

    nd = RecordingNondet()
    recorded = nd.read_file(str(p))

    replay = ReplayNondet(nd.draws)
    replayed = replay.read_file(str(p))

    assert replayed == recorded
    assert replay.fully_consumed()


def test_read_file_replay_does_not_touch_filesystem(tmp_path, monkeypatch):
    p = tmp_path / "gone.txt"
    p.write_bytes(b"will be deleted before replay")

    nd = RecordingNondet()
    recorded = nd.read_file(str(p))

    p.unlink()  # prove replay never reads the real file again
    replay = ReplayNondet(nd.draws)
    assert replay.read_file(str(p)) == recorded


def test_replay_read_file_rejects_kind_mismatch():
    replay = ReplayNondet([("uuid", "deadbeef")])
    with pytest.raises(DivergenceError, match="read_file"):
        replay.read_file("/any/path")


def test_replay_read_file_exhausted_tape_raises():
    replay = ReplayNondet([])
    with pytest.raises(DivergenceError, match="exhausted"):
        replay.read_file("/any/path")


def test_replay_read_file_rejects_path_mismatch(tmp_path):
    """Only get_env and read_file take an argument, so replay must
    additionally assert the requested path matches the recorded one -- a
    stronger check than clock/uuid/random need."""
    p = tmp_path / "recorded.txt"
    p.write_bytes(b"content")

    nd = RecordingNondet()
    nd.read_file(str(p))

    replay = ReplayNondet(nd.draws)
    with pytest.raises(DivergenceError, match="other/path"):
        replay.read_file("/other/path")


def test_read_file_over_cap_raises_and_appends_nothing(tmp_path):
    p = tmp_path / "big.bin"
    p.write_bytes(b"x" * 100)

    nd = RecordingNondet(max_read_file_bytes=50)
    with pytest.raises(ReadFileTooLargeError, match=r"100.*50"):
        nd.read_file(str(p))
    assert nd.draws == []


def test_read_file_at_exactly_cap_succeeds(tmp_path):
    p = tmp_path / "exact.bin"
    p.write_bytes(b"x" * 50)

    nd = RecordingNondet(max_read_file_bytes=50)
    data = nd.read_file(str(p))
    assert data == b"x" * 50
    assert len(nd.draws) == 1


def test_drifting_nondet_read_file_reads_fresh_content(tmp_path):
    """DriftingNondet inherits RecordingNondet.read_file -- it must re-read
    the real file fresh, not replay a fixed/recorded one."""
    p = tmp_path / "drift.txt"
    p.write_bytes(b"recorded content")
    recorded = RecordingNondet().read_file(str(p))

    p.write_bytes(b"changed content")
    drifted = DriftingNondet().read_file(str(p))

    assert drifted != recorded
    assert drifted == b"changed content"


def test_interleaved_all_five_kinds_round_trip_in_order(tmp_path, monkeypatch):
    """All five draw kinds share one ordered log; replay must serve each
    kind back in the order it was recorded, regardless of interleaving."""
    p = tmp_path / "interleave.txt"
    p.write_bytes(b"file payload")
    monkeypatch.setenv("TF_RF_INTERLEAVE", "env_value")
    random.seed(1)

    nd = RecordingNondet()
    clock1 = nd.now_iso()
    rand1 = nd.random_float()
    uuid1 = nd.new_uuid_hex()
    env1 = nd.get_env("TF_RF_INTERLEAVE")
    file1 = nd.read_file(str(p))
    rand2 = nd.random_float()

    assert [k for k, _ in nd.draws] == [
        "clock",
        "random",
        "uuid",
        "env",
        "read_file",
        "random",
    ]

    replay = ReplayNondet(nd.draws)
    assert replay.now_iso() == clock1
    assert replay.random_float() == rand1
    assert replay.new_uuid_hex() == uuid1
    assert replay.get_env("TF_RF_INTERLEAVE") == env1
    assert replay.read_file(str(p)) == file1
    assert replay.random_float() == rand2
    assert replay.fully_consumed()


# ── coverage.py: tape_draw_coverage tallies the new "read_file" kind ───────


def test_tape_draw_coverage_reports_nonzero_read_file_count():
    tape = Tape(
        draws=[
            ("read_file", json.dumps({"path": "a", "size": 1})),
            ("read_file", json.dumps({"path": "b", "size": 2})),
        ]
    )
    draw_counts, _concurrency, _guard = tape_draw_coverage(tape)
    assert draw_counts == {"read_file": 2}


def test_tape_draw_coverage_pre_existing_tape_has_no_zero_filled_read_file_entry():
    tape = Tape(draws=[("clock", "a"), ("uuid", "b"), ("random", "c")])
    draw_counts, _concurrency, _guard = tape_draw_coverage(tape)
    assert draw_counts == {"clock": 1, "uuid": 1, "random": 1}
    assert "read_file" not in draw_counts
