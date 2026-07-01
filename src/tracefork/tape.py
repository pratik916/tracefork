"""Content-addressed, zstd-compressed, persistable tape.

A tape is the recorded artifact of one agent run: ordered HTTP exchanges
(request body + response body) and nondeterminism draws. Blobs are stored
content-addressed (keyed by sha256) and zstd-compressed so identical bytes
are stored once. `digest()` is a hash chain over all draws + exchanges.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass, field

import zstandard as zstd

from .constants import BOUNDARY_V1

_ZCTX = zstd.ZstdCompressor(level=3)
_DCTX = zstd.ZstdDecompressor()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class Tape:
    exchanges: list[tuple[bytes, bytes]] = field(default_factory=list)
    draws: list[tuple[str, str]] = field(default_factory=list)
    boundary: str = BOUNDARY_V1
    agent_name: str = ""

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
        """Serialize tape to JSON bytes (base64-encoded for binary exchange fields)."""
        import base64
        import json

        return json.dumps(
            {
                "exchanges": [
                    [base64.b64encode(req).decode(), base64.b64encode(resp).decode()]
                    for req, resp in self.exchanges
                ],
                "draws": self.draws,
                "boundary": self.boundary,
                "agent_name": self.agent_name,
            }
        ).encode()

    @classmethod
    def from_bytes(cls, data: bytes) -> Tape:
        """Deserialize tape from JSON bytes produced by to_bytes()."""
        import base64
        import json

        d = json.loads(data)
        tape = cls(
            boundary=d["boundary"],
            agent_name=d["agent_name"],
        )
        tape.draws = [tuple(pair) for pair in d["draws"]]
        tape.exchanges = [
            (base64.b64decode(req), base64.b64decode(resp)) for req, resp in d["exchanges"]
        ]
        return tape

    def save(self, path: str) -> None:
        con = sqlite3.connect(path)
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
            con.commit()
        finally:
            con.close()

    @classmethod
    def load(cls, path: str) -> Tape:
        con = sqlite3.connect(path)
        try:
            raw_blobs = dict(con.execute("SELECT hash, data FROM blobs").fetchall())
            blobs = {k: _DCTX.decompress(bytes(v)) for k, v in raw_blobs.items()}
            meta = dict(con.execute("SELECT key, value FROM meta").fetchall())
            tape = cls(
                boundary=meta.get("boundary", BOUNDARY_V1),
                agent_name=meta.get("agent_name", ""),
            )
            for kind, a, b in con.execute("SELECT kind, a, b FROM events ORDER BY seq").fetchall():
                if kind == "draw":
                    tape.draws.append((a, b))
                elif kind == "exchange":
                    tape.exchanges.append((blobs[a], blobs[b]))
            return tape
        finally:
            con.close()
