"""BedrockTransport tests — the botocore `before-send` record/replay seam.

Offline, $0, zero botocore dependency (see bedrock_transport.py's module
docstring): drives the seam entirely through synthetic.py's botocore-shaped
fakes (`FakeAWSPreparedRequest`, `FakeEventEmitter`, `ScriptedBedrockSender`).
One optional test proves the SAME seam also works against a REAL botocore
`HierarchicalEmitter`/`AWSRequest` when botocore happens to be installed
(skipped otherwise via `pytest.importorskip`).

Proves, per the bead's hard invariants:
  1. record -> replay round trip is bit-exact.
  2. a canonical-request mismatch (changed body) raises DivergenceError.
  3. a fresh-signature/timestamp-only request does NOT raise (SigV4
     canonicalization via the pre-existing matcher.bedrock_matcher()).
  4. an unrecorded request at replay hard-errors (no live endpoint).
  5. tape.py is reused unchanged: to_bytes/from_bytes round-trips the tape
     BedrockTransport wrote, and its digest() computes normally.
"""

import pytest

from tracefork.bedrock_transport import BedrockTransport, prepared_request_to_httpx
from tracefork.nondet import DivergenceError
from tracefork.synthetic import (
    FakeAWSPreparedRequest,
    FakeEventEmitter,
    ScriptedBedrockSender,
    first_non_none_response,
)
from tracefork.tape import Tape

INVOKE_URL = (
    "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-sonnet-4-6/invoke"
)
EVENT_NAME = "before-send.bedrock-runtime.InvokeModel"


def _prepared(
    body: bytes, *, date: str = "20260101T000000Z", token: str = "tok-A", url: str = INVOKE_URL
) -> FakeAWSPreparedRequest:
    return FakeAWSPreparedRequest(
        method="POST",
        url=url,
        headers={
            "content-type": "application/json",
            "authorization": (
                f"AWS4-HMAC-SHA256 Credential=AKIAEXAMPLE/{date[:8]}/us-east-1/"
                f"bedrock/aws4_request, SignedHeaders=content-type;host, Signature=deadbeef"
            ),
            "x-amz-date": date,
            "x-amz-security-token": token,
        },
        body=body,
    )


def _req_body(text: str = "hi") -> bytes:
    return (
        b'{"anthropic_version": "bedrock-2023-05-31", "max_tokens": 100, '
        b'"messages": [{"role": "user", "content": "%b"}]}' % text.encode()
    )


RESP_BODY = (
    b'{"id": "msg_1", "type": "message", "role": "assistant", '
    b'"model": "anthropic.claude-sonnet-4-6", "content": [{"type": "text", "text": "hello"}], '
    b'"stop_reason": "end_turn", "usage": {"input_tokens": 10, "output_tokens": 5}}'
)


# ── construction / mode validation ──────────────────────────────────────────


def test_invalid_mode_rejected():
    with pytest.raises(ValueError, match="mode must be"):
        BedrockTransport("bogus", Tape())


def test_record_mode_requires_sender():
    with pytest.raises(ValueError, match="requires a `sender`"):
        BedrockTransport("record", Tape())


# ── prepared_request_to_httpx: duck typing works on the fake ───────────────


def test_prepared_request_to_httpx_carries_method_url_headers_body():
    prepared = _prepared(_req_body("hi"))
    httpx_req = prepared_request_to_httpx(prepared)
    assert httpx_req.method == "POST"
    assert str(httpx_req.url) == INVOKE_URL
    assert httpx_req.content == _req_body("hi")
    assert httpx_req.headers["x-amz-date"] == "20260101T000000Z"


# ── record -> replay bit-exact round trip ───────────────────────────────────


def test_record_then_replay_is_bit_exact():
    tape = Tape()
    sender = ScriptedBedrockSender([RESP_BODY])
    recorder = BedrockTransport("record", tape, sender=sender)
    emitter = FakeEventEmitter()
    recorder.register(emitter)

    result = first_non_none_response(emitter.emit(EVENT_NAME, request=_prepared(_req_body("hi"))))
    assert result is not None
    assert result.content == RESP_BODY
    assert len(tape.exchanges) == 1
    assert sender.requests_received[0].content == _req_body("hi")

    # Round-trip the tape through to_bytes/from_bytes -- proves tape.py is
    # reused completely unchanged, not reimplemented for Bedrock.
    reloaded = Tape.from_bytes(tape.to_bytes())
    assert reloaded.digest() == tape.digest()

    replayer = BedrockTransport("replay", reloaded)
    replay_emitter = FakeEventEmitter()
    replayer.register(replay_emitter)
    # Replay with a DIFFERENT signature/date/token -- must still match (see
    # the SigV4-volatility tests below for the isolated proof).
    replay_result = first_non_none_response(
        replay_emitter.emit(
            EVENT_NAME, request=_prepared(_req_body("hi"), date="20260702T121212Z", token="tok-B")
        )
    )
    assert replay_result is not None
    assert replay_result.content == RESP_BODY
    assert replayer.fully_consumed()
    assert replayer.matched == 1


def test_streaming_operation_also_registered_by_default():
    tape = Tape()
    tape.append_exchange(b"canonical-req", RESP_BODY)
    replayer = BedrockTransport("replay", tape)
    emitter = FakeEventEmitter()
    replayer.register(emitter)
    assert "before-send.bedrock-runtime.InvokeModelWithResponseStream" in emitter._handlers
    assert "before-send.bedrock-runtime.InvokeModel" in emitter._handlers


# ── divergence: unrecorded request (no live endpoint) hard-errors ──────────


def test_replay_hard_errors_on_unrecorded_request():
    replayer = BedrockTransport("replay", Tape())
    with pytest.raises(DivergenceError, match="unrecorded"):
        replayer._on_before_send(_prepared(_req_body("hi")))


def test_replay_hard_errors_past_end_of_tape():
    tape = Tape()
    sender = ScriptedBedrockSender([RESP_BODY])
    recorder = BedrockTransport("record", tape, sender=sender)
    emitter = FakeEventEmitter()
    recorder.register(emitter)
    emitter.emit(EVENT_NAME, request=_prepared(_req_body("hi")))

    replayer = BedrockTransport("replay", tape)
    replay_emitter = FakeEventEmitter()
    replayer.register(replay_emitter)
    replay_emitter.emit(EVENT_NAME, request=_prepared(_req_body("hi")))
    assert replayer.fully_consumed()
    with pytest.raises(DivergenceError, match="unrecorded"):
        replay_emitter.emit(EVENT_NAME, request=_prepared(_req_body("hi")))


# ── divergence: a genuine body/target change IS caught ──────────────────────


def test_replay_diverges_on_body_change():
    tape = Tape()
    sender = ScriptedBedrockSender([RESP_BODY])
    recorder = BedrockTransport("record", tape, sender=sender)
    emitter = FakeEventEmitter()
    recorder.register(emitter)
    emitter.emit(EVENT_NAME, request=_prepared(_req_body("hi")))

    replayer = BedrockTransport("replay", tape)
    with pytest.raises(DivergenceError, match="diverged"):
        replayer._on_before_send(_prepared(_req_body("a completely different message")))


def test_replay_diverges_on_different_model_path():
    tape = Tape()
    sender = ScriptedBedrockSender([RESP_BODY])
    recorder = BedrockTransport("record", tape, sender=sender)
    emitter = FakeEventEmitter()
    recorder.register(emitter)
    emitter.emit(EVENT_NAME, request=_prepared(_req_body("hi"), url=INVOKE_URL))

    other_model_url = INVOKE_URL.replace("claude-sonnet-4-6", "claude-haiku-4-5")
    replayer = BedrockTransport("replay", tape)
    with pytest.raises(DivergenceError, match="diverged"):
        replayer._on_before_send(_prepared(_req_body("hi"), url=other_model_url))


# ── SigV4 volatility: fresh signature/timestamp alone must NOT diverge ─────


def test_replay_does_not_diverge_on_fresh_signature_date_and_token_alone():
    tape = Tape()
    sender = ScriptedBedrockSender([RESP_BODY])
    recorder = BedrockTransport("record", tape, sender=sender)
    emitter = FakeEventEmitter()
    recorder.register(emitter)
    emitter.emit(
        EVENT_NAME, request=_prepared(_req_body("hi"), date="20260101T000000Z", token="tok-A")
    )

    replayer = BedrockTransport("replay", tape)
    replay_emitter = FakeEventEmitter()
    replayer.register(replay_emitter)
    # Same body, same URL/model -- only the SigV4 signing material rotated.
    result = first_non_none_response(
        replay_emitter.emit(
            EVENT_NAME,
            request=_prepared(_req_body("hi"), date="20260702T121212Z", token="tok-ROTATED"),
        )
    )
    assert result is not None
    assert result.content == RESP_BODY
    assert replayer.matched == 1


# ── real botocore integration (optional, skipped when unavailable) ─────────


def test_register_and_replay_against_real_botocore_emitter():
    pytest.importorskip("botocore")
    from botocore.awsrequest import AWSRequest
    from botocore.hooks import HierarchicalEmitter
    from botocore.hooks import first_non_none_response as real_fnnr

    tape = Tape()
    sender = ScriptedBedrockSender([RESP_BODY])
    recorder = BedrockTransport("record", tape, sender=sender)
    emitter = HierarchicalEmitter()
    recorder.register(emitter)

    aws_req = AWSRequest(
        method="POST",
        url=INVOKE_URL,
        data=_req_body("hi"),
        headers={"x-amz-date": "20260101T000000Z", "content-type": "application/json"},
    )
    prepared = aws_req.prepare()
    responses = emitter.emit(EVENT_NAME, request=prepared)
    result = real_fnnr(responses)
    assert result is not None
    assert result.content == RESP_BODY
    assert len(tape.exchanges) == 1

    replayer = BedrockTransport("replay", tape)
    replay_emitter = HierarchicalEmitter()
    replayer.register(replay_emitter)
    aws_req2 = AWSRequest(
        method="POST",
        url=INVOKE_URL,
        data=_req_body("hi"),
        # Different date -> a real, freshly-signed replay attempt.
        headers={"x-amz-date": "20260702T121212Z", "content-type": "application/json"},
    )
    replay_result = real_fnnr(replay_emitter.emit(EVENT_NAME, request=aws_req2.prepare()))
    assert replay_result is not None
    assert replay_result.content == RESP_BODY
    assert replayer.fully_consumed()
