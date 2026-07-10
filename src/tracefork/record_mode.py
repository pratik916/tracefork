"""``RecordMode`` — a named vocabulary for the record/replay policy tracefork
already enforces, modeled on VCR's ``record_mode`` cassette options.

Today "record" and "replay" are two separate, explicit code paths a caller
picks directly: ``Recorder``/``AsyncRecorder`` for recording, and
``TraceforkTransport("replay", ...)`` (used by ``ReplayVerifier``,
``ForkEngine``'s prefix-replay phase, and ``BlameEngine``) for replay. Replay
mode already hard-errors on any request beyond the tape's recorded exchanges
— see ``transport.py``'s module docstring — but that contract has never had a
name. This module gives it one, so callers and ``TraceforkConfig`` can refer
to the policy by a typed value instead of a stringly-typed literal.

``ONCE``, ``NONE``, and ``ALL`` resolve to today's two literal transport modes
(``"record"`` / ``"replay"``) via ``resolve_transport_mode`` below.
``NEW_EPISODES`` (replay recorded exchanges but *record* any new trailing
ones instead of erroring) resolves to a THIRD literal, ``"new_episodes"`` —
``TraceforkTransport``'s additive third mode (see ``transport.py``'s module
docstring): the recorded prefix still replays under the exact strict-replay
assert logic, and any request beyond it is forwarded to the transport's inner
transport and recorded, mirroring ``"record"`` mode.
"""

from __future__ import annotations

from enum import StrEnum


class RecordMode(StrEnum):
    """VCR-style cassette policy names for the record-vs-replay choice.

    * ``ONCE`` (default) — record if no tape exists yet, else replay strict
      (error on any unrecorded/extra request). This is exactly today's
      behavior when a caller reaches for ``Recorder`` the first time and
      ``TraceforkTransport("replay", ...)`` on every run after — nothing
      about the runtime changes, this just names the pattern.
    * ``NONE`` — always replay strict, never touch the network. Equivalent to
      ``TraceforkTransport("replay", tape)`` unconditionally; useful for
      CI/replay-only environments where an accidental live call must be a
      hard error rather than a silent new recording.
    * ``ALL`` — always (re-)record against the live backend, ignoring any
      existing tape. Equivalent to ``Recorder``/``TraceforkTransport("record",
      ...)`` unconditionally.
    * ``NEW_EPISODES`` — replay recorded exchanges but record any new
      trailing ones instead of erroring, regardless of ``tape_exists``
      (there is always a recorded prefix to attempt to replay, even if
      empty). Resolves to ``TraceforkTransport``'s ``"new_episodes"`` literal
      (see ``transport.py``).
    """

    ONCE = "once"
    NONE = "none"
    ALL = "all"
    NEW_EPISODES = "new_episodes"


def resolve_transport_mode(mode: RecordMode, *, tape_exists: bool) -> str:
    """Map a ``RecordMode`` + tape-existence fact to today's literal transport
    mode string (``"record"`` or ``"replay"``).

    This function *names* what was previously an implicit caller decision for
    ``ONCE``/``NONE``/``ALL``; ``transport.py``'s ``"replay"`` literal keeps
    hard-erroring on any unrecorded request exactly as it did before this
    module existed. ``RecordMode.NEW_EPISODES`` resolves to the THIRD,
    additive literal ``"new_episodes"`` (see ``transport.py``'s module
    docstring) regardless of ``tape_exists``.
    """
    if mode is RecordMode.NONE:
        return "replay"
    if mode is RecordMode.ALL:
        return "record"
    if mode is RecordMode.ONCE:
        return "replay" if tape_exists else "record"
    return "new_episodes"
