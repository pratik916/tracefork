"""Multi-agent session replay + fork-board report (tracefork-bge.33): one
lane per spawn-linked tape in an orchestration session, plus a shared
index-based scrubber — the offline CLI static-generator path
(`tracefork session board`).

Reuses `report.py`'s existing `_tape_to_data`/`_safe_json` verbatim (zero
edits to `report.py`) for each lane's tape data, in
`store.TapeStore.session_tapes`'s already-shipped BFS order. Each lane is
additionally annotated with `store.spawn_parent`/`spawn_children` and, only
when that run_id appears in an optional `agent_map: dict[run_id, callable]`
(the SAME resolved-callable shape `session_replay.resolve_agent_manifest`
already produces from a `{run_id: "module:fn"}` JSON file — reused, not
reinvented), a REAL `replay.ReplayVerifier(tape, agent_fn).verify()` receipt
via `verification_result_to_dict` — generalizing `cli.py`'s existing
`report` command's single-agent `--agent` resolution to per-lane, since a
session's tapes come from different agent fns. A run_id absent from
`agent_map` renders `replay={}`, the same falsy/neutral pattern `report.py`
already establishes for every OTHER report — never a fabricated status.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from .replay import ReplayVerifier, verification_result_to_dict
from .report import _safe_json, _tape_to_data

if TYPE_CHECKING:
    from .store import TapeStore

_INJECT_MARKER = "</head>"

__all__ = ["_session_to_data", "generate_session_report"]


def _session_template_path() -> Path:
    """Locate ``web/session_report.html`` — same dual wheel/source-checkout
    lookup as ``report._template_path``."""
    here = Path(__file__).parent
    for cand in (
        here / "web" / "session_report.html",  # installed wheel (force-included)
        here.parent.parent / "web" / "session_report.html",  # repo root
    ):
        if cand.exists():
            return cand
    raise FileNotFoundError(
        "web/session_report.html not found (looked in the package and the repo root)"
    )


def _session_to_data(
    store: TapeStore, session_id: str, *, agent_map: dict[str, Any] | None = None
) -> dict:
    """Assemble the fork-board JSON: session metadata + one lane per
    ``store.session_tapes(session_id)``'s BFS-ordered run_id.

    ``agent_map`` (optional) is a ``{run_id: agent_fn}`` mapping of ALREADY-
    RESOLVED callables (see ``session_replay.resolve_agent_manifest``); a
    run_id present in it gets a real, freshly-computed replay receipt, a
    run_id absent from it gets the neutral ``replay={}`` empty state.
    """
    session = store.get_session(session_id)
    run_ids = store.session_tapes(session_id)
    agent_map = agent_map or {}

    lanes = []
    for run_id in run_ids:
        tape = store.load_tape(run_id)
        lane_data = _tape_to_data(tape)
        replay_data: dict = {}
        agent_fn = agent_map.get(run_id)
        if agent_fn is not None:
            result = ReplayVerifier(tape, agent_fn).verify()
            replay_data = verification_result_to_dict(result)
        lanes.append(
            {
                **lane_data,
                "run_id": run_id,
                "spawn_parent": store.spawn_parent(run_id),
                "spawn_children": store.spawn_children(run_id),
                "replay": replay_data,
            }
        )

    return {
        "session_id": session["session_id"],
        "root_run_id": session["root_run_id"],
        "created_at": session["created_at"],
        "lanes": lanes,
    }


def generate_session_report(
    store: TapeStore,
    session_id: str,
    output_path: Path,
    *,
    agent_map: dict[str, Any] | None = None,
) -> None:
    """Write a self-contained fork-board HTML report to ``output_path``.

    The session data is injected before ``</head>`` as
    ``window.__TRACEFORK_SESSION_DATA__`` (mirrors ``report.generate_report``'s
    ``window.__TRACEFORK_DATA__`` injection, including the same
    ``</script>``-breakout escaping via ``report._safe_json``).
    """
    html = _session_template_path().read_text()
    data = _session_to_data(store, session_id, agent_map=agent_map)
    inject = f"\n<script>\nwindow.__TRACEFORK_SESSION_DATA__ = {_safe_json(data)};\n</script>\n"
    html = html.replace(_INJECT_MARKER, inject + _INJECT_MARKER, 1)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
