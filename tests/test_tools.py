"""JSON-RPC tool-frame seam tests — all offline, synthetic frames, $0.

Mirrors test_transport.py / test_fork.py for the tool-I/O seam: record→replay
determinism, request-frame mismatch → divergence, fork-swap of a tool output,
redaction of tool frames, the native (non-MCP) tool seam, and tape persistence
of tool exchanges.
"""

import os
import tempfile

import pytest

from tracefork.nondet import DivergenceError
from tracefork.redact import Redactor, regex_redactor
from tracefork.tape import Tape
from tracefork.tools import (
    NativeToolSeam,
    ToolForkTransport,
    ToolTransport,
    canonical_frame,
    decode_result,
    frame_fingerprint,
    frame_id,
    make_result_frame,
    make_tool_call_frame,
    retarget_frame_id,
)

# --- helpers ---


def _weather_call(frame_id_=1, city="NYC"):
    return make_tool_call_frame(frame_id_, "get_weather", {"city": city})


def _weather_result(frame_id_=1, temp=72):
    return make_result_frame(frame_id_, {"content": [{"type": "text", "text": f"{temp}F"}]})


def _scripted_inner(responses):
    """A record-mode inner that returns queued response frames in order."""
    it = iter(responses)

    def inner(_request_frame):
        return next(it)

    return inner


# --- frame utilities ---


def test_canonical_frame_drops_volatile_id():
    a = _weather_call(frame_id_=1)
    b = _weather_call(frame_id_=999)
    assert a != b  # raw bytes differ (different id)
    assert canonical_frame(a) == canonical_frame(b)  # identity ignores id
    assert frame_fingerprint(a) == frame_fingerprint(b)


def test_canonical_frame_distinguishes_semantic_change():
    assert frame_fingerprint(_weather_call(city="NYC")) != frame_fingerprint(
        _weather_call(city="LA")
    )


def test_retarget_frame_id_sets_response_id():
    resp = _weather_result(frame_id_=1)
    retargeted = retarget_frame_id(resp, 42)
    assert frame_id(retargeted) == 42
    assert decode_result(retargeted) == decode_result(resp)


def test_frame_id_and_retarget_tolerate_non_json():
    assert frame_id(b"not json") is None
    assert retarget_frame_id(b"not json", 5) == b"not json"
    assert canonical_frame(b"not json") == b"not json"


# --- record / replay ---


def test_record_captures_tool_exchanges():
    tape = Tape()
    t = ToolTransport(
        "record", tape, inner=_scripted_inner([_weather_result(1), _weather_result(2)])
    )
    r1 = t.handle_frame(_weather_call(1))
    t.handle_frame(_weather_call(2, city="LA"))
    assert decode_result(r1) == {"content": [{"type": "text", "text": "72F"}]}
    assert len(tape.tool_exchanges) == 2
    assert tape.tool_exchanges[0] == (_weather_call(1), _weather_result(1))


def test_record_then_replay_is_deterministic():
    tape = Tape()
    rec = ToolTransport(
        "record", tape, inner=_scripted_inner([_weather_result(1), _weather_result(2)])
    )
    rec.handle_frame(_weather_call(1))
    rec.handle_frame(_weather_call(2, city="LA"))

    rep = ToolTransport("replay", tape)
    assert decode_result(rep.handle_frame(_weather_call(1))) == {
        "content": [{"type": "text", "text": "72F"}]
    }
    assert decode_result(rep.handle_frame(_weather_call(2, city="LA"))) == {
        "content": [{"type": "text", "text": "72F"}]
    }
    assert rep.matched == 2
    assert rep.fully_consumed()


def test_replay_retargets_response_id_to_live_request():
    """A rotated JSON-RPC id must not diverge, and the served response's id is
    retargeted so the client can still correlate it."""
    tape = Tape()
    tape.append_tool_exchange(_weather_call(1), _weather_result(1))
    rep = ToolTransport("replay", tape)
    served = rep.handle_frame(_weather_call(777))  # same call, different id
    assert frame_id(served) == 777
    assert decode_result(served) == {"content": [{"type": "text", "text": "72F"}]}


def test_replay_request_frame_mismatch_is_divergence():
    tape = Tape()
    tape.append_tool_exchange(_weather_call(1, city="NYC"), _weather_result(1))
    rep = ToolTransport("replay", tape)
    with pytest.raises(DivergenceError, match="diverged"):
        rep.handle_frame(_weather_call(1, city="Paris"))


def test_replay_extra_call_is_divergence():
    tape = Tape()
    tape.append_tool_exchange(_weather_call(1), _weather_result(1))
    rep = ToolTransport("replay", tape)
    rep.handle_frame(_weather_call(1))
    with pytest.raises(DivergenceError, match="unrecorded"):
        rep.handle_frame(_weather_call(1))


def test_record_without_inner_raises():
    tape = Tape()
    t = ToolTransport("record", tape)
    with pytest.raises(ValueError, match="inner"):
        t.handle_frame(_weather_call(1))


# --- fork (mutator) ---


def _two_call_parent():
    tape = Tape()
    rec = ToolTransport(
        "record", tape, inner=_scripted_inner([_weather_result(1), _weather_result(2)])
    )
    rec.handle_frame(_weather_call(1, city="NYC"))
    rec.handle_frame(_weather_call(2, city="LA"))
    return tape


def test_fork_swaps_tool_output_at_divergence_step():
    parent = _two_call_parent()
    mutated = make_result_frame(2, {"content": [{"type": "text", "text": "STORM"}]})
    delta = Tape()
    fork = ToolForkTransport(parent, divergence_step=1, mutated_response=mutated, delta_tape=delta)

    # step 0: prefix, replayed from parent for $0
    served0 = fork.handle_frame(_weather_call(1, city="NYC"))
    assert decode_result(served0) == {"content": [{"type": "text", "text": "72F"}]}
    # step 1: mutation injected
    served1 = fork.handle_frame(_weather_call(2, city="LA"))
    assert decode_result(served1) == {"content": [{"type": "text", "text": "STORM"}]}

    assert fork.prefix_replayed == 1
    assert fork.tail_recorded == 0
    assert len(delta.tool_exchanges) == 1
    assert decode_result(delta.tool_exchanges[0][1]) == {
        "content": [{"type": "text", "text": "STORM"}]
    }


def test_fork_at_step_zero_records_counterfactual_tail():
    parent = _two_call_parent()
    mutated = make_result_frame(1, {"content": [{"type": "text", "text": "MUTATED"}]})
    delta = Tape()
    tail_resp = make_result_frame(2, {"content": [{"type": "text", "text": "TAIL"}]})
    fork = ToolForkTransport(
        parent,
        divergence_step=0,
        mutated_response=mutated,
        delta_tape=delta,
        inner=_scripted_inner([tail_resp]),
    )
    fork.handle_frame(_weather_call(1, city="NYC"))  # mutation
    served_tail = fork.handle_frame(_weather_call(2, city="LA"))  # tail, recorded fresh

    assert fork.prefix_replayed == 0
    assert fork.tail_recorded == 1
    assert decode_result(served_tail) == {"content": [{"type": "text", "text": "TAIL"}]}
    assert len(delta.tool_exchanges) == 2


def test_fork_prefix_mismatch_is_divergence():
    parent = _two_call_parent()
    delta = Tape()
    fork = ToolForkTransport(parent, divergence_step=1, mutated_response=b"{}", delta_tape=delta)
    with pytest.raises(DivergenceError, match="prefix"):
        fork.handle_frame(_weather_call(1, city="WRONG"))


# --- redaction ---


def test_redaction_scrubs_tool_frames_both_sides():
    """A secret in both the request args and the response is scrubbed on the
    stored tape, and record→replay still verifies (redaction applied both sides)."""
    redactor = Redactor(
        request_filters=(regex_redactor(r"SECRET"),),
        response_filters=(regex_redactor(r"SECRET"),),
    )
    tape = Tape()
    req = make_tool_call_frame(1, "lookup", {"token": "SECRET"})
    resp = make_result_frame(1, {"content": "value=SECRET"})
    rec = ToolTransport("record", tape, inner=_scripted_inner([resp]), redactor=redactor)
    live_resp = rec.handle_frame(req)

    # stored bytes are scrubbed on both request and response
    stored_req, stored_resp = tape.tool_exchanges[0]
    assert b"SECRET" not in stored_req
    assert b"SECRET" not in stored_resp
    assert b"REDACTED" in stored_req
    # caller still saw the unredacted live response (matches the LLM seam)
    assert b"SECRET" in live_resp

    # replay with the SAME redactor verifies (identical transform on both sides)
    rep = ToolTransport("replay", tape, redactor=redactor)
    served = rep.handle_frame(req)
    assert b"SECRET" not in served


# --- native (non-MCP) tool seam ---


def test_native_tool_seam_round_trip():
    tape = Tape()
    rec = NativeToolSeam(tape, "record")
    assert rec.mode == "record"

    @rec.tool("add")
    def add(a, b):
        return a + b

    assert add(2, 3) == 5
    assert add(10, 1) == 11
    assert len(tape.tool_exchanges) == 2

    calls = {"n": 0}
    rep = NativeToolSeam(tape, "replay")

    @rep.tool("add")
    def add_replay(a, b):  # must NOT be called on replay
        calls["n"] += 1
        return -999

    assert add_replay(2, 3) == 5
    assert add_replay(10, 1) == 11
    assert calls["n"] == 0  # served from tape, real fn never ran


def test_native_tool_seam_divergence_on_arg_change():
    tape = Tape()
    rec = NativeToolSeam(tape, "record")

    @rec.tool("add")
    def add(a, b):
        return a + b

    add(2, 3)

    rep = NativeToolSeam(tape, "replay")

    @rep.tool("add")
    def add_replay(a, b):
        return a + b

    with pytest.raises(DivergenceError):
        add_replay(4, 4)  # different args → request frame mismatch


# --- tape persistence + digest ---


def test_tool_exchanges_survive_to_bytes_roundtrip():
    tape = Tape(agent_name="tool-agent")
    tape.append_exchange(b"llm-req", b"llm-resp")
    tape.append_tool_exchange(_weather_call(1), _weather_result(1))
    restored = Tape.from_bytes(tape.to_bytes())
    assert restored.exchanges == tape.exchanges
    assert restored.tool_exchanges == tape.tool_exchanges
    assert restored.digest() == tape.digest()


def test_tool_exchanges_survive_sqlite_roundtrip():
    tape = Tape(agent_name="tool-agent")
    tape.append_exchange(b"llm-req", b"llm-resp")
    tape.append_tool_exchange(_weather_call(1), _weather_result(1))
    tape.append_tool_exchange(_weather_call(2, city="LA"), _weather_result(2))
    with tempfile.NamedTemporaryFile(suffix=".tape.sqlite", delete=False) as f:
        path = f.name
    try:
        tape.save(path)
        loaded = Tape.load(path)
        assert loaded.tool_exchanges == tape.tool_exchanges
        assert loaded.exchanges == tape.exchanges
        assert loaded.digest() == tape.digest()
    finally:
        os.unlink(path)


def test_digest_changes_with_tool_exchanges_but_empty_is_neutral():
    llm_only = Tape()
    llm_only.append_exchange(b"req", b"resp")
    baseline = llm_only.digest()

    with_tool = Tape()
    with_tool.append_exchange(b"req", b"resp")
    with_tool.append_tool_exchange(_weather_call(1), _weather_result(1))

    # empty tool log leaves the digest identical; a tool exchange changes it
    assert llm_only.digest() == baseline
    assert with_tool.digest() != baseline
