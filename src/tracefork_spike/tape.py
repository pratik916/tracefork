"""Content-addressed, persistable tape.

A tape is the recorded artifact of one agent run: the ordered HTTP exchanges
(request body + response body) plus the ordered nondeterminism draws. Response and
request bodies are stored content-addressed (keyed by sha256), so identical bytes are
stored once; an ordered event log preserves sequence. `digest()` is a hash chain over
the whole tape — the single fingerprint reported in the receipt.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass, field


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class Tape:
    exchanges: list[tuple[bytes, bytes]] = field(default_factory=list)
    draws: list[tuple[str, str]] = field(default_factory=list)

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

    # --- persistence: content-addressed blobs + ordered event log ---------------

    def save(self, path: str) -> None:
        con = sqlite3.connect(path)
        try:
            con.executescript(
                """
                DROP TABLE IF EXISTS blobs;
                DROP TABLE IF EXISTS events;
                CREATE TABLE blobs (hash TEXT PRIMARY KEY, data BLOB NOT NULL);
                CREATE TABLE events (seq INTEGER PRIMARY KEY AUTOINCREMENT,
                                     kind TEXT NOT NULL, a TEXT NOT NULL, b TEXT NOT NULL);
                """
            )
            for kind, value in self.draws:
                con.execute("INSERT INTO events (kind, a, b) VALUES ('draw', ?, ?)", (kind, value))
            for req, resp in self.exchanges:
                rh, sh = sha256_hex(req), sha256_hex(resp)
                con.execute("INSERT OR IGNORE INTO blobs (hash, data) VALUES (?, ?)", (rh, req))
                con.execute("INSERT OR IGNORE INTO blobs (hash, data) VALUES (?, ?)", (sh, resp))
                con.execute("INSERT INTO events (kind, a, b) VALUES ('exchange', ?, ?)", (rh, sh))
            con.commit()
        finally:
            con.close()

    @classmethod
    def load(cls, path: str) -> "Tape":
        con = sqlite3.connect(path)
        try:
            blobs = dict(con.execute("SELECT hash, data FROM blobs").fetchall())
            tape = cls()
            for kind, a, b in con.execute("SELECT kind, a, b FROM events ORDER BY seq").fetchall():
                if kind == "draw":
                    tape.draws.append((a, b))
                elif kind == "exchange":
                    tape.exchanges.append((bytes(blobs[a]), bytes(blobs[b])))
            return tape
        finally:
            con.close()
