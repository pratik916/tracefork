"""Report generator: produces a self-contained HTML file from a Tape.

In static mode, the entire tape data is serialized as JSON and injected
into the HTML template as `window.__TRACEFORK_DATA__ = {...}`.
"""

from __future__ import annotations

import json
from pathlib import Path

from .providers import get_adapter

_INJECT_MARKER = "</head>"


def _template_path() -> Path:
    """Locate ``web/report.html`` in both an installed wheel and a source checkout.

    A built wheel force-includes the file at ``tracefork/web/report.html`` (next to
    this module); an editable/source checkout keeps it at the repo root. Resolved at
    call time so importing this module never depends on the file's location.
    """
    here = Path(__file__).parent
    for cand in (
        here / "web" / "report.html",  # installed wheel (force-included)
        here.parent.parent / "web" / "report.html",  # repo root (src/tracefork -> repo)
    ):
        if cand.exists():
            return cand
    raise FileNotFoundError("web/report.html not found (looked in the package and the repo root)")


def _runs_template_path() -> Path:
    """Locate ``web/runs.html`` (the multi-run dashboard page,
    tracefork-bge.67) — same dual wheel/source-checkout lookup as
    :func:`_template_path`."""
    here = Path(__file__).parent
    for cand in (
        here / "web" / "runs.html",  # installed wheel (force-included)
        here.parent.parent / "web" / "runs.html",  # repo root (src/tracefork -> repo)
    ):
        if cand.exists():
            return cand
    raise FileNotFoundError("web/runs.html not found (looked in the package and the repo root)")


def _tape_to_data(
    tape,
    blame: dict | None = None,
    replay: dict | None = None,
    branches: list[dict] | None = None,
    causal_edges: list[dict] | None = None,
    branch_details: dict[str, dict] | None = None,
    shapley: dict | None = None,
    cost_profile: dict | None = None,
    causal_closure: list[dict] | None = None,
    run_id: str | None = None,
) -> dict:
    """Convert a Tape to the JSON shape expected by the web UI."""
    adapter = get_adapter("anthropic")
    exchanges = []
    for req_bytes, resp_bytes in tape.exchanges:
        try:
            req_json = json.loads(req_bytes.decode())
        except Exception:
            req_json = {"_raw": req_bytes.hex()}

        try:
            resp_json = json.loads(resp_bytes.decode())
        except Exception:
            # Streaming response — let the provider adapter extract the first
            # JSON object from the SSE framing (or fall back to an opaque marker).
            resp_json = adapter.parse_sse(resp_bytes) or {"_raw": "sse"}

        # Determine role from response
        role = "unknown"
        if isinstance(resp_json, dict):
            if resp_json.get("type") == "message":
                role = "assistant"
            elif resp_json.get("role") == "user":
                role = "user"
        if "messages" in req_json:
            msgs = req_json["messages"]
            if msgs:
                role = msgs[-1].get("role", role)

        # Preview: first 80 chars of last user message or response text
        preview = ""
        try:
            if isinstance(resp_json, dict) and resp_json.get("content"):
                for block in resp_json["content"]:
                    if block.get("type") == "text":
                        preview = block["text"][:80]
                        break
                    if block.get("type") == "tool_use":
                        tool_input_preview = json.dumps(block.get("input", {}))[:60]
                        preview = f"→ {block.get('name', 'tool')}({tool_input_preview})"
                        break
        except Exception:
            pass

        exchanges.append(
            {
                "role": role,
                "preview": preview,
                "request": req_json,
                "response_preview": resp_json,
            }
        )

    return {
        "agent_name": tape.agent_name,
        "exchanges": exchanges,
        "blame": blame or {},
        # Replay-report data (see `replay.verification_result_to_dict`): bit-exactness
        # receipt + a structured divergence diagnostic on drift. `{}` (falsy) when no
        # replay was run — the UI renders a neutral "no replay data" state for that.
        "replay": replay or {},
        "created_at": "",
        "fingerprint": tape.digest()[:16],
        # Trust/provenance metadata — never fed into `digest()` (see `tape.py`),
        # so surfacing it here is purely informational. `boundary` distinguishes
        # a bit-exact-replayable tape (`constants.BOUNDARY_V1`) from a
        # forensic-only one (`OTEL_INGESTED_BOUNDARY`/`PROXY_BOUNDARY`); the web
        # UI renders both as a trust badge (see `web/report.html`'s
        # `renderProvenanceBadges`).
        "boundary": tape.boundary,
        "content_redacted": tape.content_redacted,
        # Fork-tree panel data (see `store.list_branches`) — the run's
        # branch summaries (branch_id/divergence_step/mutation_desc/
        # created_at/branch_digest), no `delta_tape` fetch needed to render
        # the tree. `[]` (falsy) when none were passed, the same neutral
        # empty-state pattern `replay={}` already establishes.
        "branches": branches or [],
        # Persisted causal edges (see `store.causal_edges_for_run`) — blame
        # and Shapley results already computed and saved by `tracefork blame`,
        # a free read with no recompute. `[]` (falsy) when none were passed,
        # same neutral empty-state pattern as `branches`.
        "causal_edges": causal_edges or [],
        # branch_id -> full `_tape_to_data(branch['delta_tape'])` dict plus
        # divergence_step/mutation_desc/branch_digest/parent_run_id — the
        # exact shape `server.py`'s `/api/branch/{id}` already returns, baked
        # into the static report so a fork-tree click needs no live server.
        # `{}` (falsy) when none were passed, same neutral empty-state
        # pattern as `branches`/`causal_edges`.
        "branch_details": branch_details or {},
        # Per-step Shapley necessity/sufficiency quadrant (ShapleyResult/
        # causal_edges shape, step_index-keyed) — the Timeline panel's
        # inline quadrant badge (see `web/report.html`'s
        # `shapleyQuadrantHtml`). `{}` (falsy) when not passed, same neutral
        # empty-state pattern as `blame`/`branches`.
        "shapley": shapley or {},
        # Per-model/per-tool cost dashboard (see `cost_profile.py`) — the
        # shape `cost_profile.cost_profile_to_dict` returns. `{}` (falsy)
        # when not passed, same neutral empty-state pattern as `shapley`.
        "cost_profile": cost_profile or {},
        # External-anchor vocabulary (tracefork-bge.70): responsible=1 blame
        # edges reachable via fork-promotion lineage (`store.causal_closure`)
        # that can belong to OTHER run_ids -- Shepherd's "causal parent
        # outside the current slice's own content". `[]` (falsy) when not
        # passed, same neutral empty-state pattern as `branches`.
        "causal_closure": causal_closure or [],
        # This run's own id, so the UI can tell an external-anchor entry
        # (`edge["run_id"] != data["run_id"]`) apart from one that's already
        # covered by this run's own `blame` rows. `None` when not passed.
        "run_id": run_id,
    }


def _safe_json(data: dict) -> str:
    """Serialize `data` and escape HTML-significant chars so recorded agent I/O
    (which can contain ``</script>``) cannot break out of the inline <script>.

    Replacing ``< > &`` with their ``\\uXXXX`` forms yields valid JSON string
    escapes, so the loader's parse still works. The JS line separators
    U+2028/U+2029 are already emitted as ``\\u`` escapes by ``ensure_ascii=True``.
    """
    return (
        json.dumps(data, indent=2)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def generate_report(
    tape,
    output_path: Path,
    *,
    blame: dict | None = None,
    replay: dict | None = None,
    branches: list[dict] | None = None,
    causal_edges: list[dict] | None = None,
    branch_details: dict[str, dict] | None = None,
    shapley: dict | None = None,
    cost_profile: dict | None = None,
    causal_closure: list[dict] | None = None,
    run_id: str | None = None,
) -> None:
    """Write a self-contained HTML report to `output_path`.

    The tape data is injected before </head> so the UI loads it synchronously.
    `replay` (optional) is the JSON-safe dict from
    `tracefork.replay.verification_result_to_dict` — a bit-exactness receipt
    plus a structured divergence diagnostic when the replay drifted.
    `branches` (optional) is the run's branch summaries — the shape
    `tracefork.store.TapeStore.list_branches` returns — rendered as the
    fork-tree panel; `None`/omitted embeds an empty list (see
    `web/report.html`'s `renderForkTree`).
    `causal_edges` (optional) is the run's persisted blame/Shapley edges —
    the shape `tracefork.store.TapeStore.causal_edges_for_run` returns —
    cross-referenced against `branches` to highlight causally-significant
    fork points. `branch_details` (optional) is branch_id -> full delta-tape
    report data (the shape `server.py`'s `/api/branch/{id}` returns) so a
    static report's fork-tree clicks render real data with no live server.
    `shapley` (optional) is a step_index-keyed dict of Shapley
    necessity/sufficiency results (the `ShapleyResult`/`causal_edges` shape)
    rendered as a small inline quadrant badge per Timeline exchange.
    `cost_profile` (optional) is the JSON-safe dict from
    `cost_profile.cost_profile_to_dict`, rendered as the report's cost/profile
    dashboard panel.
    `causal_closure` (optional) is `tracefork.store.TapeStore.causal_closure`'s
    result — responsible blame edges reachable via fork-promotion lineage,
    possibly from other run_ids — rendered as external-anchor entries.
    `run_id` (optional) is this run's own id, so the UI can distinguish an
    external-anchor entry from one already covered by this run's own blame
    rows.
    """
    html = _template_path().read_text()
    data = _tape_to_data(
        tape,
        blame,
        replay,
        branches,
        causal_edges,
        branch_details,
        shapley,
        cost_profile,
        causal_closure,
        run_id,
    )
    inject = f"\n<script>\nwindow.__TRACEFORK_DATA__ = {_safe_json(data)};\n</script>\n"
    html = html.replace(_INJECT_MARKER, inject + _INJECT_MARKER, 1)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
