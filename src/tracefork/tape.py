"""Content-addressed, zstd-compressed, persistable tape.

A tape is the recorded artifact of one agent run: ordered HTTP exchanges
(request body + response body) and nondeterminism draws. Blobs are stored
content-addressed (keyed by sha256) and zstd-compressed so identical bytes
are stored once. `digest()` is a hash chain over all draws + exchanges.

Two serialization surfaces, both content-addressed + zstd, both versioned:
  * `save()`/`load()` — SQLite file (blobs table + event log).
  * `to_bytes()`/`from_bytes()` — a single self-describing blob (the form
    `TapeStore` persists). A magic marker + uint16 version prefix the envelope;
    `from_bytes` dispatches on the version through a read-time upcaster chain.
    Legacy blobs (no marker: the original JSON + base64 encoding) still load as
    format version 1 — detect-and-fall-back, never crash. The version header is
    envelope metadata only: it is NOT fed into `digest()`, so the hash chain of
    any given tape content is byte-identical across format versions.
"""

from __future__ import annotations

import base64
import hashlib
import json
import sqlite3
import struct
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import zstandard as zstd

from .constants import BOUNDARY_V1, TAPE_FORMAT_VERSION, TAPE_MAGIC

_ZCTX = zstd.ZstdCompressor(level=3)
_DCTX = zstd.ZstdDecompressor()

_MAGIC_LEN = len(TAPE_MAGIC)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def open_sqlite(path: str) -> sqlite3.Connection:
    """Open a hardened SQLite connection: WAL, relaxed sync, a busy-timeout so
    concurrent writers wait instead of raising ``database is locked``, and
    foreign-key enforcement. ``isolation_level=None`` puts us in autocommit mode
    so writers can take an explicit ``BEGIN IMMEDIATE`` write lock up front."""
    con = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA busy_timeout=5000")
    con.execute("PRAGMA foreign_keys=ON")
    return con


@dataclass
class Tape:
    exchanges: list[tuple[bytes, bytes]] = field(default_factory=list)
    draws: list[tuple[str, str]] = field(default_factory=list)
    boundary: str = BOUNDARY_V1
    agent_name: str = ""
    # Set by `Recorder`/`AsyncRecorder` when an opt-in content-redacting `Redactor`
    # scrubbed message CONTENT (as opposed to just headers/secret env values) on
    # this tape. Forensic-only: redacted response content changes what the agent
    # sees on replay, and redacted request content weakens the divergence check,
    # so a `content_redacted` tape is not guaranteed bit-exact-replayable — see
    # `redact.py` and the README's Redaction section. Never touched by `digest()`
    # (metadata, like `boundary`/`agent_name`, not hash-chained content).
    content_redacted: bool = False

    def append_exchange(self, request_body: bytes, response_body: bytes) -> None:
        self.exchanges.append((request_body, response_body))

    def exchange(self, i: int) -> tuple[bytes, bytes]:
        return self.exchanges[i]

    def digest(self) -> str:
        """sha256 hash chain over draws then exchanges — the tape fingerprint."""
        h = hashlib.sha256()
        for kind, value in self.draws:
            h.update(b"D:" + kind.encode() + b":" + value.encode() + b"\n")
        for req, resp in self.exchanges:
            h.update(b"X:" + sha256_hex(req).encode() + b":" + sha256_hex(resp).encode() + b"\n")
        return h.hexdigest()

    def to_bytes(self) -> bytes:
        """Serialize to the current versioned envelope (magic + uint16 version +
        a content-addressed, zstd-compressed binary container). Shared blobs are
        stored once by sha256, and there is no base64, so the ~1.33x base64
        blow-up of the legacy format is gone. Read back with `from_bytes`."""
        return _encode_v2(self)

    @classmethod
    def from_bytes(cls, data: bytes) -> Tape:
        """Deserialize a tape blob produced by any released `to_bytes`.

        Dispatches on the envelope version and runs the read-time upcaster chain
        up to the current schema. Legacy blobs with no magic marker (the original
        JSON + base64 encoding) are detected and loaded as format version 1."""
        f = _decode(data)
        tape = cls(
            boundary=f["boundary"],
            agent_name=f["agent_name"],
            content_redacted=f.get("content_redacted", False),
        )
        tape.draws = [tuple(pair) for pair in f["draws"]]
        tape.exchanges = f["exchanges"]
        return tape

    def save(self, path: str) -> None:
        con = open_sqlite(path)
        try:
            con.executescript("""
                DROP TABLE IF EXISTS blobs;
                DROP TABLE IF EXISTS events;
                DROP TABLE IF EXISTS meta;
                CREATE TABLE blobs (hash TEXT PRIMARY KEY, data BLOB NOT NULL);
                CREATE TABLE events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL, a TEXT NOT NULL, b TEXT NOT NULL
                );
                CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """)
            con.execute("BEGIN IMMEDIATE")
            for kind, value in self.draws:
                con.execute("INSERT INTO events (kind, a, b) VALUES ('draw', ?, ?)", (kind, value))
            for req, resp in self.exchanges:
                rh, sh = sha256_hex(req), sha256_hex(resp)
                con.execute(
                    "INSERT OR IGNORE INTO blobs VALUES (?, ?)",
                    (rh, _ZCTX.compress(req)),
                )
                con.execute(
                    "INSERT OR IGNORE INTO blobs VALUES (?, ?)",
                    (sh, _ZCTX.compress(resp)),
                )
                con.execute("INSERT INTO events (kind, a, b) VALUES ('exchange', ?, ?)", (rh, sh))
            con.execute("INSERT INTO meta VALUES ('boundary', ?)", (self.boundary,))
            con.execute("INSERT INTO meta VALUES ('agent_name', ?)", (self.agent_name,))
            con.execute("INSERT INTO meta VALUES ('schema_version', '1')")
            con.execute(
                "INSERT INTO meta VALUES ('content_redacted', ?)",
                (str(int(self.content_redacted)),),
            )
            con.execute("COMMIT")
        finally:
            con.close()

    @classmethod
    def load(cls, path: str) -> Tape:
        con = open_sqlite(path)
        try:
            raw_blobs = dict(con.execute("SELECT hash, data FROM blobs").fetchall())
            blobs = {k: _DCTX.decompress(bytes(v)) for k, v in raw_blobs.items()}
            meta = dict(con.execute("SELECT key, value FROM meta").fetchall())
            tape = cls(
                boundary=meta.get("boundary", BOUNDARY_V1),
                agent_name=meta.get("agent_name", ""),
                content_redacted=bool(int(meta.get("content_redacted", "0"))),
            )
            for kind, a, b in con.execute("SELECT kind, a, b FROM events ORDER BY seq").fetchall():
                if kind == "draw":
                    tape.draws.append((a, b))
                elif kind == "exchange":
                    tape.exchanges.append((blobs[a], blobs[b]))
            return tape
        finally:
            con.close()


# ── to_bytes / from_bytes codec ─────────────────────────────────────────────
#
# The envelope is:  TAPE_MAGIC | uint16 version | version-specific body.
# `from_bytes` reads the version, decodes the body to a canonical `_Fields`
# dict, then walks the read-time upcaster chain up to TAPE_FORMAT_VERSION.
# `_Fields` is the version-independent shape every decoder yields and every
# upcaster transforms — decoupling on-disk encodings from the in-memory tape.
_Fields = dict[str, Any]


def _encode_v2(tape: Tape) -> bytes:
    """Version-2 body: a JSON header (boundary, agent_name, draws, the exchange
    hash pairs, and the dedup'd blob-hash order) followed by, for each unique
    blob in order, a uint32 length + its zstd-compressed bytes. Content-addressed
    (each distinct request/response stored once) and base64-free."""
    order: list[str] = []
    seen: dict[str, bytes] = {}
    for req, resp in tape.exchanges:
        for blob in (req, resp):
            h = sha256_hex(blob)
            if h not in seen:
                seen[h] = blob
                order.append(h)
    header = {
        "boundary": tape.boundary,
        "agent_name": tape.agent_name,
        "draws": tape.draws,
        "exchanges": [[sha256_hex(req), sha256_hex(resp)] for req, resp in tape.exchanges],
        "blob_hashes": order,
        "content_redacted": tape.content_redacted,
    }
    header_json = json.dumps(header).encode()
    parts: list[bytes] = [
        TAPE_MAGIC,
        struct.pack(">H", TAPE_FORMAT_VERSION),
        struct.pack(">I", len(header_json)),
        header_json,
    ]
    for h in order:
        comp = _ZCTX.compress(seen[h])
        parts.append(struct.pack(">I", len(comp)))
        parts.append(comp)
    return b"".join(parts)


def _decode_v1_json(body: bytes) -> _Fields:
    """Legacy format: a JSON object with base64-encoded exchange bodies."""
    d = json.loads(body)
    return {
        "boundary": d["boundary"],
        "agent_name": d["agent_name"],
        "draws": [tuple(pair) for pair in d["draws"]],
        "exchanges": [
            (base64.b64decode(req), base64.b64decode(resp)) for req, resp in d["exchanges"]
        ],
        "content_redacted": d.get("content_redacted", False),
    }


def _decode_v2_binary(body: bytes) -> _Fields:
    """Content-addressed zstd container written by `_encode_v2`."""
    (header_len,) = struct.unpack_from(">I", body, 0)
    off = 4
    header = json.loads(body[off : off + header_len])
    off += header_len
    blobs: dict[str, bytes] = {}
    for h in header["blob_hashes"]:
        (blob_len,) = struct.unpack_from(">I", body, off)
        off += 4
        blobs[h] = _DCTX.decompress(body[off : off + blob_len])
        off += blob_len
    return {
        "boundary": header["boundary"],
        "agent_name": header["agent_name"],
        "draws": [tuple(pair) for pair in header["draws"]],
        "exchanges": [(blobs[req], blobs[resp]) for req, resp in header["exchanges"]],
        "content_redacted": header.get("content_redacted", False),
    }


def _upcast_v1_to_v2(fields: _Fields) -> _Fields:
    """v1 -> v2 is an encoding-only change (JSON+base64 -> zstd binary container);
    the logical tape schema is unchanged, so the fields carry forward as-is. This
    seam exists so a future *logical* migration slots in as one more upcaster."""
    return fields


_DECODERS: dict[int, Callable[[bytes], _Fields]] = {
    1: _decode_v1_json,
    2: _decode_v2_binary,
}
_UPCASTERS: dict[int, Callable[[_Fields], _Fields]] = {
    1: _upcast_v1_to_v2,
}


def _read_envelope(data: bytes) -> tuple[int, bytes]:
    """Split an envelope into (version, body). A blob without the magic marker is
    a legacy JSON encoding — format version 1 — so it keeps loading."""
    if data[:_MAGIC_LEN] == TAPE_MAGIC:
        (version,) = struct.unpack_from(">H", data, _MAGIC_LEN)
        return version, data[_MAGIC_LEN + 2 :]
    return 1, data


def _decode(data: bytes) -> _Fields:
    version, body = _read_envelope(data)
    decoder = _DECODERS.get(version)
    if decoder is None:
        raise ValueError(
            f"unsupported tape format version {version} "
            f"(this build reads up to {TAPE_FORMAT_VERSION})"
        )
    fields = decoder(body)
    v = version
    while v < TAPE_FORMAT_VERSION:  # read-time upcaster chain
        fields = _UPCASTERS[v](fields)
        v += 1
    return fields
