"""Session-wide replay divergence rollup: for one orchestration session (see
`store.py`'s `sessions`/`spawn_edges` schema, tracefork-bge.12), replay every
spawn-linked tape the caller supplies an `agent_fn` for, and report the
EARLIEST (BFS/session order) genuine divergence — or `None` when every
mapped tape replayed bit-exact.

Pure composition, zero new engine logic: `store.session_tapes()` (the
already-shipped deterministic BFS) supplies the walk order, and each mapped
tape is replayed via the EXISTING `replay.ReplayVerifier(tape,
agent_fn).verify()`, unchanged. `session_divergence_rollup` returns as soon
as it hits the first tape (in `session_tapes()` order) whose
`VerificationResult.bit_exact` is `False` — later tapes in the session are
never even loaded, let alone replayed.

`agent_fns` is a caller-supplied `run_id -> callable` mapping — the same
shape `replay.run_fixture_corpus_check`'s manifest already uses — rather
than an agent_fn auto-derived from any of the `adapters/*.py` frameworks'
own metadata; reconstructing a replayable agent_fn generically from
LangChain/CrewAI/AutoGen/ADK/OpenAI-Agents adapter state is explicitly out
of scope for this module (a future bead's job). A run_id that's reachable in
the session's spawn graph but ABSENT from `agent_fns` is recorded in
`SessionDivergenceResult.skipped_run_ids` — never silently dropped, never
raised on, and never counted as either a pass or a divergence.

`resolve_agent_manifest()` is the small `importlib` helper mirroring
`replay.run_fixture_corpus_check`'s `"module:fn"` string -> callable
resolution, for the CLI's `--agents-manifest <path.json>` surface.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .replay import DivergenceReport, ReplayVerifier

if TYPE_CHECKING:
    from .store import TapeStore

__all__ = [
    "SessionDivergenceResult",
    "session_divergence_rollup",
    "resolve_agent_manifest",
]


@dataclass
class SessionDivergenceResult:
    """Outcome of `session_divergence_rollup` for one session.

    `checked_run_ids` are every run_id actually replayed via
    `ReplayVerifier.verify()` (in `session_tapes()` order) before the walk
    stopped — including the diverging run_id itself, when there is one.
    `skipped_run_ids` are reachable run_ids the walk reached but had no
    `agent_fns` entry for, so they were never loaded or replayed. Because the
    walk stops at the first divergence, `checked_run_ids + skipped_run_ids`
    covers the full `session_tapes()` order only when `diverged_run_id` is
    `None`; otherwise it covers only the prefix walked before the return.
    """

    session_id: str
    checked_run_ids: list[str]
    skipped_run_ids: list[str]
    diverged_run_id: str | None
    divergence: DivergenceReport | None = None


def session_divergence_rollup(
    store: TapeStore, session_id: str, agent_fns: dict[str, Any]
) -> SessionDivergenceResult:
    """Replay every `session_tapes(session_id)`-reachable tape that
    `agent_fns` maps a callable for, in BFS/session order, via the existing
    `ReplayVerifier`.

    Returns as soon as the first tape whose replay is not bit-exact is
    found — `SessionDivergenceResult.diverged_run_id` names it and
    `.divergence` carries the `DivergenceReport` (feed it to
    `replay.DriftDoctor.classify` for a `DriftCause`). Returns
    `diverged_run_id=None`/`divergence=None` when every mapped tape replayed
    bit-exact (whether or not every reachable run_id had a mapping).

    A reachable run_id with no `agent_fns` entry lands in
    `skipped_run_ids` and is neither loaded nor replayed — never a crash,
    never mistaken for a pass or a divergence.

    Raises `KeyError` (via `store.session_tapes` -> `store.get_session`) for
    an unknown `session_id`, exactly as `session_tapes` itself does.
    """
    order = store.session_tapes(session_id)
    checked: list[str] = []
    skipped: list[str] = []

    for run_id in order:
        agent_fn = agent_fns.get(run_id)
        if agent_fn is None:
            skipped.append(run_id)
            continue

        tape = store.load_tape(run_id)
        result = ReplayVerifier(tape, agent_fn).verify()
        checked.append(run_id)

        if not result.bit_exact:
            return SessionDivergenceResult(
                session_id=session_id,
                checked_run_ids=checked,
                skipped_run_ids=skipped,
                diverged_run_id=run_id,
                divergence=result.divergence,
            )

    return SessionDivergenceResult(
        session_id=session_id,
        checked_run_ids=checked,
        skipped_run_ids=skipped,
        diverged_run_id=None,
        divergence=None,
    )


def resolve_agent_manifest(manifest: dict[str, str]) -> dict[str, Any]:
    """Resolve a `{run_id: "module:fn"}` mapping (the parsed JSON of the
    CLI's `--agents-manifest <path.json>` file) into a `{run_id: callable}`
    `agent_fns` mapping for `session_divergence_rollup`, via the same
    `importlib`/`rsplit(':', 1)` pattern `replay.run_fixture_corpus_check`
    already uses to resolve its own `manifest.json` agent entries.
    """
    resolved: dict[str, Any] = {}
    for run_id, agent_path in manifest.items():
        module_path, fn_name = agent_path.rsplit(":", 1)
        resolved[run_id] = getattr(importlib.import_module(module_path), fn_name)
    return resolved
