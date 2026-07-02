"""`RecordMode` semantics — the named vocabulary for tracefork's existing
record/replay split. `transport.py` itself is untouched by this module (see
its docstring); these tests pin down that `resolve_transport_mode`'s default
(`ONCE`) reproduces today's behavior exactly, and that the reserved
`NEW_EPISODES` mode fails loudly rather than silently misbehaving.

Offline, zero API keys.
"""

from __future__ import annotations

import pytest

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


def test_new_episodes_is_reserved_and_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        resolve_transport_mode(RecordMode.NEW_EPISODES, tape_exists=True)
    with pytest.raises(NotImplementedError):
        resolve_transport_mode(RecordMode.NEW_EPISODES, tape_exists=False)


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
