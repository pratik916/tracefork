"""AWS ``application/vnd.amazon.eventstream`` binary framing codec.

Bedrock's streaming responses (``InvokeModelWithResponseStream``/
``ConverseStream``) use AWS's binary event-stream framing — **not** SSE, so
none of the ``"data: "``/bare-``"data:"`` SSE parsers in
``providers/anthropic.py``/``providers/gemini.py`` apply. This module is a
small, self-contained (stdlib-only: ``struct`` + ``binascii.crc32``) ENCODER
and DECODER of that wire format, so the offline synthetic Bedrock fake can
emit and read real frames and prove round-trip fidelity — including the
frame's CRC checks — with **no botocore dependency at all**. That keeps this
module importable (and its round-trip test runnable) even when boto3/botocore
are not installed, satisfying the same "offline and $0" contract as the rest
of the package.

The algorithm mirrors ``botocore.eventstream``'s decoder (``MessagePrelude``,
``EventStreamBuffer``, ``_validate_checksum``) — verified against that
module's source — but is an independent implementation; nothing here imports
``botocore``.

Frame layout (big-endian throughout, per AWS's event-stream spec)::

    total_length    uint32   -- the whole message, prelude through message CRC
    headers_length  uint32   -- byte length of the header block below
    prelude_crc     uint32   -- CRC-32 of the two fields above
    <headers_length bytes of headers>
    <payload bytes>
    message_crc     uint32   -- CRC-32 of every byte from `total_length`
                                through the end of the payload (i.e. the whole
                                message except this trailing field)

Each header is ``[1-byte name length][name][1-byte type tag][type-specific
value]``. This codec only emits/reads the ``string`` type (tag 7, a 2-byte
length-prefixed UTF-8 value) — the only type Bedrock's own event headers
(``:event-type``, ``:content-type``, ``:message-type``) use.
"""

from __future__ import annotations

import struct
from binascii import crc32
from dataclasses import dataclass

#: Bytes covered by the prelude CRC: `total_length` (4B) + `headers_length` (4B).
_PRELUDE_CRC_INPUT_LEN = 8
#: Full prelude including its own trailing CRC field.
_PRELUDE_LEN = _PRELUDE_CRC_INPUT_LEN + 4
_STRING_TYPE = 7


class EventStreamError(Exception):
    """A malformed event-stream frame: bad CRC, truncated prelude/payload, or
    an unsupported header type tag."""


@dataclass(frozen=True)
class EventStreamMessage:
    """One decoded event-stream message: its headers and raw payload bytes."""

    headers: dict[str, str]
    payload: bytes


def _encode_header(name: str, value: str) -> bytes:
    name_bytes = name.encode("utf-8")
    if len(name_bytes) > 0xFF:
        raise ValueError(f"header name too long ({len(name_bytes)} bytes): {name!r}")
    value_bytes = value.encode("utf-8")
    if len(value_bytes) > 0xFFFF:
        raise ValueError(f"header value too long ({len(value_bytes)} bytes) for header {name!r}")
    return (
        struct.pack("!B", len(name_bytes))
        + name_bytes
        + struct.pack("!B", _STRING_TYPE)
        + struct.pack("!H", len(value_bytes))
        + value_bytes
    )


def encode_message(headers: dict[str, str], payload: bytes) -> bytes:
    """Encode one event-stream message. All header values are strings (the
    only type Bedrock's own frames use — see the module docstring)."""
    header_bytes = b"".join(_encode_header(k, v) for k, v in headers.items())
    total_length = _PRELUDE_LEN + len(header_bytes) + len(payload) + 4
    prelude_crc_input = struct.pack("!II", total_length, len(header_bytes))
    prelude_crc = crc32(prelude_crc_input) & 0xFFFFFFFF
    body = struct.pack("!I", prelude_crc) + header_bytes + payload
    message_crc = crc32(prelude_crc_input + body) & 0xFFFFFFFF
    return prelude_crc_input + body + struct.pack("!I", message_crc)


def _decode_header(data: bytes) -> tuple[str, str, bytes]:
    (name_len,) = struct.unpack_from("!B", data, 0)
    off = 1
    name = data[off : off + name_len].decode("utf-8")
    off += name_len
    (type_tag,) = struct.unpack_from("!B", data, off)
    off += 1
    if type_tag != _STRING_TYPE:
        raise EventStreamError(f"unsupported header type tag {type_tag} for header {name!r}")
    (value_len,) = struct.unpack_from("!H", data, off)
    off += 2
    value = data[off : off + value_len].decode("utf-8")
    off += value_len
    return name, value, data[off:]


def decode_message(data: bytes) -> tuple[EventStreamMessage, bytes]:
    """Decode ONE message from the front of ``data``; return ``(message, rest)``
    where ``rest`` is whatever bytes follow it (empty if none)."""
    if len(data) < _PRELUDE_LEN:
        raise EventStreamError(f"truncated prelude: need {_PRELUDE_LEN} bytes, got {len(data)}")
    total_length, headers_length = struct.unpack_from("!II", data, 0)
    (prelude_crc,) = struct.unpack_from("!I", data, _PRELUDE_CRC_INPUT_LEN)
    computed_prelude_crc = crc32(data[:_PRELUDE_CRC_INPUT_LEN]) & 0xFFFFFFFF
    if computed_prelude_crc != prelude_crc:
        raise EventStreamError(
            f"prelude CRC mismatch: expected {prelude_crc:#010x}, "
            f"computed {computed_prelude_crc:#010x}"
        )
    if len(data) < total_length:
        raise EventStreamError(f"truncated message: need {total_length} bytes, got {len(data)}")
    message_crc_offset = total_length - 4
    (message_crc,) = struct.unpack_from("!I", data, message_crc_offset)
    computed_message_crc = crc32(data[:message_crc_offset]) & 0xFFFFFFFF
    if computed_message_crc != message_crc:
        raise EventStreamError(
            f"message CRC mismatch: expected {message_crc:#010x}, "
            f"computed {computed_message_crc:#010x}"
        )
    headers_start = _PRELUDE_LEN
    headers_end = headers_start + headers_length
    headers: dict[str, str] = {}
    remaining = data[headers_start:headers_end]
    while remaining:
        name, value, remaining = _decode_header(remaining)
        headers[name] = value
    payload = data[headers_end:message_crc_offset]
    return EventStreamMessage(headers=headers, payload=payload), data[total_length:]


def decode_messages(data: bytes) -> list[EventStreamMessage]:
    """Decode a buffer containing zero or more back-to-back event-stream
    messages (e.g. the full body of a buffered streaming response)."""
    messages: list[EventStreamMessage] = []
    remaining = data
    while remaining:
        message, remaining = decode_message(remaining)
        messages.append(message)
    return messages
