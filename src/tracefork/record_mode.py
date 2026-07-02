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

Only ``ONCE``, ``NONE``, and ``ALL`` resolve to today's two literal transport
modes (``"record"`` / ``"replay"``) via ``resolve_transport_mode`` below.
``NEW_EPISODES`` (replay recorded exchanges but *record* any new trailing
ones instead of erroring) is reserved for future work: wiring it in requires
extending ``TraceforkTransport``'s replay branch with a live fallback
transport, which is out of scope for this change (``transport.py`` is
untouched — see ``CLAUDE.md``). Selecting it raises ``NotImplementedError``
rather than silently behaving like one of the other three.
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
      trailing ones instead of erroring. **Reserved**: not implemented in
      this release (see module docstring); ``resolve_transport_mode`` raises
      ``NotImplementedError`` if selected.
    """

    ONCE = "once"
    NONE = "none"
    ALL = "all"
    NEW_EPISODES = "new_episodes"


def resolve_transport_mode(mode: RecordMode, *, tape_exists: bool) -> str:
    """Map a ``RecordMode`` + tape-existence fact to today's literal transport
    mode string (``"record"`` or ``"replay"``).

    This function *names* what was previously an implicit caller decision; it
    does not change ``transport.py``, which keeps hard-erroring on any
    unrecorded request during replay exactly as it did before this module
    existed. Raises ``NotImplementedError`` for ``RecordMode.NEW_EPISODES``
    (see module docstring).
    """
    if mode is RecordMode.NONE:
        return "replay"
    if mode is RecordMode.ALL:
        return "record"
    if mode is RecordMode.ONCE:
        return "replay" if tape_exists else "record"
    raise NotImplementedError(
        f"{mode!r} is reserved for future work: it requires extending "
        "TraceforkTransport's replay path with a live fallback transport, which is "
        "out of scope for this change (transport.py is untouched)."
    )
