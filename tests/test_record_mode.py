"""`RecordMode` semantics — the named vocabulary for tracefork's record/replay
split. These tests pin down that `resolve_transport_mode`'s default (`ONCE`)
reproduces today's behavior exactly, and that `NEW_EPISODES` resolves to
`transport.py`'s additive `"new_episodes"` transport literal.

Offline, zero API keys.
"""

from __future__ import annotations

from tracefork.record_mode import RecordMode, resolve_transport_mode


def test_once_records_when_no_tape_exists():
    assert resolve_transport_mode(RecordMode.ONCE, tape_exists=False) == "record"


def test_once_replays_when_tape_exists():
    assert resolve_transport_mode(RecordMode.ONCE, tape_exists=True) == "replay"


def test_none_always_replays_regardless_of_tape_existence():
    assert resolve_transport_mode(RecordMode.NONE, tape_exists=False) == "replay"
    assert resolve_transport_mode(RecordMode.NONE, tape_exists=True) == "replay"


def test_all_always_records_regardless_of_tape_existence():
    assert resolve_transport_mode(RecordMode.ALL, tape_exists=False) == "record"
    assert resolve_transport_mode(RecordMode.ALL, tape_exists=True) == "record"


def test_new_episodes_resolves_to_new_episodes_transport_mode_regardless_of_tape_existence():
    """NEW_EPISODES is no longer reserved: it maps to `transport.py`'s third
    literal (replay-recorded-prefix + record-trailing-episodes) whether or
    not a tape already exists — that distinction only matters inside the
    transport itself, not at this resolution layer."""
    assert resolve_transport_mode(RecordMode.NEW_EPISODES, tape_exists=True) == "new_episodes"
    assert resolve_transport_mode(RecordMode.NEW_EPISODES, tape_exists=False) == "new_episodes"


def test_record_mode_values_are_stable_strings():
    """`StrEnum` values double as the CLI/env-var literal spelling — pin the
    exact strings so a future refactor can't silently rename them."""
    assert RecordMode.ONCE == "once"
    assert RecordMode.NONE == "none"
    assert RecordMode.ALL == "all"
    assert RecordMode.NEW_EPISODES == "new_episodes"


def test_record_mode_constructible_from_its_own_value_string():
    for mode in RecordMode:
        assert RecordMode(mode.value) is mode
