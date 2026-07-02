"""AWS event-stream binary framing codec — standalone round-trip proof.

Offline, stdlib-only (no botocore). Proves the encoder/decoder pair is
self-consistent (encode -> decode -> assert bytes/semantics) even though full
streaming-response replay through botocore is out of scope — see
`bedrock_transport.py`'s module docstring.
"""

import pytest

from tracefork.eventstream import (
    EventStreamError,
    EventStreamMessage,
    decode_message,
    decode_messages,
    encode_message,
)


def test_round_trip_single_message_with_bedrock_style_headers():
    headers = {
        ":event-type": "chunk",
        ":content-type": "application/json",
        ":message-type": "event",
    }
    payload = b'{"bytes": "eyJkZWx0YSI6ICJoZWxsbyJ9"}'
    frame = encode_message(headers, payload)
    message, rest = decode_message(frame)
    assert message == EventStreamMessage(headers=headers, payload=payload)
    assert rest == b""


def test_round_trip_empty_payload_and_empty_headers():
    frame = encode_message({}, b"")
    message, rest = decode_message(frame)
    assert message.headers == {}
    assert message.payload == b""
    assert rest == b""


def test_decode_messages_handles_multiple_back_to_back_frames():
    frames = [
        encode_message({":event-type": "chunk"}, b"first"),
        encode_message({":event-type": "chunk"}, b"second"),
        encode_message({":event-type": "chunk"}, b"third"),
    ]
    buffer = b"".join(frames)
    messages = decode_messages(buffer)
    assert [m.payload for m in messages] == [b"first", b"second", b"third"]
    assert all(m.headers[":event-type"] == "chunk" for m in messages)


def test_decode_message_returns_correct_remainder_for_partial_buffer():
    frame_a = encode_message({":event-type": "chunk"}, b"aaa")
    frame_b = encode_message({":event-type": "chunk"}, b"bbb")
    message, rest = decode_message(frame_a + frame_b)
    assert message.payload == b"aaa"
    assert rest == frame_b


def test_unicode_payload_and_header_values_round_trip():
    headers = {":event-type": "chunk", "x-note": "café ✅"}
    payload = "hello 世界".encode()
    frame = encode_message(headers, payload)
    message, _ = decode_message(frame)
    assert message.headers == headers
    assert message.payload == payload


def test_corrupted_prelude_crc_raises():
    frame = bytearray(encode_message({":event-type": "chunk"}, b"payload"))
    frame[0] ^= 0xFF  # flip a bit in total_length, invalidating prelude_crc
    with pytest.raises(EventStreamError, match="prelude CRC mismatch"):
        decode_message(bytes(frame))


def test_corrupted_payload_raises_message_crc_mismatch():
    frame = bytearray(encode_message({":event-type": "chunk"}, b"payload"))
    # Flip a byte inside the payload region (well past the 12-byte prelude and
    # small header block) without touching total_length/headers_length.
    frame[-5] ^= 0xFF
    with pytest.raises(EventStreamError, match="message CRC mismatch"):
        decode_message(bytes(frame))


def test_truncated_prelude_raises():
    with pytest.raises(EventStreamError, match="truncated prelude"):
        decode_message(b"\x00\x01\x02")


def test_truncated_message_raises():
    frame = encode_message({":event-type": "chunk"}, b"payload")
    with pytest.raises(EventStreamError, match="truncated message"):
        decode_message(frame[:-3])


def test_decode_messages_on_empty_buffer_returns_empty_list():
    assert decode_messages(b"") == []
