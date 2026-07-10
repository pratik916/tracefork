"""Property-based (Hypothesis) proofs for `tape.py`'s two load-bearing claims.

`test_tape.py` pins those claims against a handful of fixed inputs; this module
generalizes them over generated content:

1. **Round-trip stability** — `Tape.to_bytes()` / `Tape.from_bytes()` is
   lossless for arbitrary exchanges/draws/tool_exchanges/metadata, and the
   restored tape's `digest()` is byte-identical to the original's.
2. **Metadata exclusion from `digest()`** — varying ONLY the fields the module
   docstring/inline comments declare non-hashed (`boundary`, `agent_name`,
   `content_redacted`, `async_batches`, `provenance`) while holding
   `exchanges`/`draws`/`tool_exchanges` fixed must never change `digest()`.

Deterministic and offline: `derandomize=True` derives each test's examples
from a fixed seed (no network, no example-database file to commit — see the
`settings` comment below), and `max_examples` is bounded so this stays well
within CI's time budget. Pure in-process sha256/json/zstd work, $0.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from tracefork.tape import Tape

# `derandomize=True` seeds every example from a hash of the test itself, so a
# run is bit-for-bit reproducible across machines/CI without needing an
# on-disk example database (Hypothesis disables the database entirely in this
# mode) — simpler than committing a `.hypothesis/` cache and equally
# deterministic. `deadline=None` avoids flaking on a loaded CI runner; the
# work here is pure in-memory hashing/serialization, never slow in practice.
_SETTINGS = settings(max_examples=50, derandomize=True, deadline=None)

_bytes = st.binary(max_size=64)
_exchange = st.tuples(_bytes, _bytes)
_exchanges = st.lists(_exchange, max_size=6)
_draw = st.tuples(st.text(max_size=16), st.text(max_size=16))
_draws = st.lists(_draw, max_size=6)
_async_batches = st.lists(st.lists(st.integers(min_value=0, max_value=20), max_size=4), max_size=4)
_provenance = st.dictionaries(st.text(max_size=12), st.text(max_size=12), max_size=4)
_name = st.text(max_size=24)
_flag = st.booleans()


@st.composite
def _tapes(draw: st.DrawFn) -> Tape:
    """Build an arbitrary in-memory `Tape` covering every field `to_bytes`
    serializes: exchanges, draws, tool_exchanges, agent_name, boundary,
    content_redacted, async_batches, and provenance."""
    return Tape(
        exchanges=draw(_exchanges),
        draws=draw(_draws),
        boundary=draw(_name),
        agent_name=draw(_name),
        content_redacted=draw(_flag),
        tool_exchanges=draw(_exchanges),
        async_batches=draw(_async_batches),
        provenance=draw(_provenance),
    )


@_SETTINGS
@given(_tapes())
def test_to_bytes_from_bytes_roundtrip_is_lossless_and_digest_stable(tape: Tape) -> None:
    """`Tape.from_bytes(tape.to_bytes())` restores every field exactly, and the
    restored tape's `digest()` matches the original's — the general form of
    `test_tape.py::test_to_bytes_from_bytes_roundtrip`."""
    restored = Tape.from_bytes(tape.to_bytes())
    assert restored == tape
    assert restored.digest() == tape.digest()


@_SETTINGS
@given(
    exchanges=_exchanges,
    draws=_draws,
    tool_exchanges=_exchanges,
    boundary_a=_name,
    boundary_b=_name,
    agent_name_a=_name,
    agent_name_b=_name,
    content_redacted_a=_flag,
    content_redacted_b=_flag,
    async_batches_a=_async_batches,
    async_batches_b=_async_batches,
    provenance_a=_provenance,
    provenance_b=_provenance,
)
def test_digest_excludes_metadata_fields(
    exchanges: list[tuple[bytes, bytes]],
    draws: list[tuple[str, str]],
    tool_exchanges: list[tuple[bytes, bytes]],
    boundary_a: str,
    boundary_b: str,
    agent_name_a: str,
    agent_name_b: str,
    content_redacted_a: bool,
    content_redacted_b: bool,
    async_batches_a: list[list[int]],
    async_batches_b: list[list[int]],
    provenance_a: dict[str, str],
    provenance_b: dict[str, str],
) -> None:
    """Two tapes sharing the same content (exchanges/draws/tool_exchanges) but
    arbitrary, independently-generated `boundary`/`agent_name`/
    `content_redacted`/`async_batches`/`provenance` must hash EQUAL — the
    general form of `test_tape.py::test_digest_excludes_provenance`, extended
    to every metadata field the module documents as non-hashed."""
    t1 = Tape(
        exchanges=exchanges,
        draws=draws,
        tool_exchanges=tool_exchanges,
        boundary=boundary_a,
        agent_name=agent_name_a,
        content_redacted=content_redacted_a,
        async_batches=async_batches_a,
        provenance=provenance_a,
    )
    t2 = Tape(
        exchanges=exchanges,
        draws=draws,
        tool_exchanges=tool_exchanges,
        boundary=boundary_b,
        agent_name=agent_name_b,
        content_redacted=content_redacted_b,
        async_batches=async_batches_b,
        provenance=provenance_b,
    )
    assert t1.digest() == t2.digest()
