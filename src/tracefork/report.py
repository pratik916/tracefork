"""Report generator: produces a self-contained HTML file from a Tape.

In static mode, the entire tape data is serialized as JSON and injected
into the HTML template as `window.__TRACEFORK_DATA__ = {...}`.
"""

from __future__ import annotations

import base64
import gzip
import json
from pathlib import Path
from typing import Any

from .providers import get_adapter
from .tape import Tape

__all__ = [
    "DEFAULT_COMPRESSION_STEP_THRESHOLD",
    "DEFAULT_BRANCH_DETAILS_CAP_BYTES",
    "generate_report",
]


_INJECT_MARKER = "</head>"

# A 400-exchange run's report can run into the hundreds of MB uncompressed
# (see `_maybe_compressed_inject_script`'s docstring) -- nobody hits this in
# the demo/quickstart (a handful of exchanges), but it is a real ceiling on
# the product's usable scale. 50 is a documented, tunable default: small
# enough that a genuinely long real run gets compressed automatically,
# large enough that the demo/quickstart/every offline test's small tapes
# never take the gzip path (keeping their output byte-for-byte identical to
# before this feature existed). `generate_report`'s
# `compression_step_threshold` parameter lets a caller widen or narrow it.
DEFAULT_COMPRESSION_STEP_THRESHOLD = 50

# `branch_details` (branch_id -> full delta-tape report data) is, by far, the
# largest contributor to a report's payload once a run has more than a
# handful of forks -- measured: a run with 100 branches produced a 1.67 MB
# report of which 1.35 MB (81%) was branch_details alone, with no cap and no
# opt-out (tracefork-sis.56). 256 KiB is a documented, tunable default: large
# enough to embed a healthy number of real branches whole, small enough that
# a pathological fork count can no longer balloon the report unboundedly.
# `generate_report`/`cli.py`'s `report --branch-details-cap-bytes` both let a
# caller widen or narrow it.
DEFAULT_BRANCH_DETAILS_CAP_BYTES = 256 * 1024


def _cap_branch_details(
    branch_details: dict[str, dict[str, Any]], cap_bytes: int
) -> tuple[dict[str, dict[str, Any]], dict[str, Any] | None]:
    """Keep `branch_details` entries (in dict/insertion order) until their
    cumulative COMPACT JSON size would exceed `cap_bytes`, dropping the rest.

    Returns `(kept, None)` unchanged when the full dict already fits (the
    common case for a run with few branches — no truncation, no marker).
    Otherwise returns `(kept, marker)` where `marker` is a small,
    JSON-safe truncation notice (`included`/`omitted`/`total_branches`/
    `cap_bytes`) a consumer can render or act on, instead of the excess
    branches silently vanishing with no trace.

    Per-entry size is measured via `json.dumps` with NO indentation — a
    conservative proxy for that entry's contribution to the final,
    `indent=2`-pretty-printed payload `_safe_json` emits (which is always
    somewhat LARGER due to indentation/newlines), so the true embedded size
    stays close to, and never wildly under, `cap_bytes` — a hint sized to be
    conservative, not a byte-exact guarantee.
    """
    total = len(branch_details)
    if total == 0:
        return branch_details, None
    if len(json.dumps(branch_details)) <= cap_bytes:
        return branch_details, None

    kept: dict[str, dict[str, Any]] = {}
    running = 2  # the enclosing `{}` of the eventual dict
    for branch_id, detail in branch_details.items():
        entry_size = len(json.dumps({branch_id: detail})) - 2  # minus its own `{}`
        if kept and running + entry_size > cap_bytes:
            break
        running += entry_size
        kept[branch_id] = detail

    return kept, {
        "included": len(kept),
        "omitted": total - len(kept),
        "total_branches": total,
        "cap_bytes": cap_bytes,
    }


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
    tape: Tape,
    blame: dict[str, Any] | None = None,
    replay: dict[str, Any] | None = None,
    branches: list[dict[str, Any]] | None = None,
    causal_edges: list[dict[str, Any]] | None = None,
    branch_details: dict[str, dict[str, Any]] | None = None,
    shapley: dict[str, Any] | None = None,
    cost_profile: dict[str, Any] | None = None,
    causal_closure: list[dict[str, Any]] | None = None,
    run_id: str | None = None,
    branch_details_cap_bytes: int = DEFAULT_BRANCH_DETAILS_CAP_BYTES,
) -> dict[str, Any]:
    """Convert a Tape to the JSON shape expected by the web UI.

    `branch_details_cap_bytes` (default `DEFAULT_BRANCH_DETAILS_CAP_BYTES`)
    caps how much of `branch_details` actually gets embedded — see
    `_cap_branch_details`. A `branch_details` dict that already fits under
    the cap round-trips unchanged.
    """
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

    capped_branch_details, branch_details_truncated = _cap_branch_details(
        branch_details or {}, branch_details_cap_bytes
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
        # pattern as `branches`/`causal_edges`. Capped at
        # `branch_details_cap_bytes` (see `_cap_branch_details`,
        # tracefork-sis.56) — a run with many branches no longer silently
        # balloons the report; `branch_details_truncated` (below) says so.
        "branch_details": capped_branch_details,
        # `None` (falsy) when `branch_details` already fit under the cap
        # whole; otherwise a small notice (`included`/`omitted`/
        # `total_branches`/`cap_bytes`) a consumer can render instead of the
        # excess branches silently vanishing with no trace.
        "branch_details_truncated": branch_details_truncated,
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


def _safe_json(data: dict[str, Any]) -> str:
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


def _gzip_b64(data: dict[str, Any]) -> str:
    """gzip-compress `data`'s COMPACT (no `indent=`) JSON encoding and
    base64-encode the result for embedding in a <script> tag.

    Unlike `_safe_json`, no `< > &` escaping is needed: the base64 alphabet
    (`A-Za-z0-9+/=`) contains none of those characters, so a `</script>`
    breakout is structurally impossible here regardless of what the
    recorded agent I/O contained -- the escaping problem `_safe_json` exists
    to solve doesn't apply to this path at all.
    """
    raw = json.dumps(data).encode("utf-8")
    compressed = gzip.compress(raw, compresslevel=9)
    return base64.b64encode(compressed).decode("ascii")


def _inject_script(data: dict[str, Any], compression_step_threshold: int) -> str:
    """Build the `<script>` block that seeds the report's data, taking the
    gzip+base64 path (`window.__TRACEFORK_DATA_GZIP_B64__`) once
    `len(data["exchanges"])` reaches `compression_step_threshold`, the plain
    path (`window.__TRACEFORK_DATA__`, byte-for-byte the same as before this
    feature existed) otherwise.

    Measured (this repo's own fixtures, a real multi-turn conversation --
    see README.md's "Scale envelope (measured)" table for the full numbers):
    at 400 exchanges, gzip+base64 shrinks the injected payload by roughly
    50x versus the equivalent `_safe_json` text. `web/report.html`'s
    `loadData` decodes the compressed path via the standard
    `DecompressionStream('gzip')` Web API -- no new dependency, matching
    this project's "no CDN, no library" discipline for `web/*.html`.
    """
    if len(data.get("exchanges", [])) >= compression_step_threshold:
        payload = json.dumps(_gzip_b64(data))  # a base64 string -- json.dumps just quotes it
        return f"\n<script>\nwindow.__TRACEFORK_DATA_GZIP_B64__ = {payload};\n</script>\n"
    return f"\n<script>\nwindow.__TRACEFORK_DATA__ = {_safe_json(data)};\n</script>\n"


def generate_report(
    tape: Tape,
    output_path: Path,
    *,
    blame: dict[str, Any] | None = None,
    replay: dict[str, Any] | None = None,
    branches: list[dict[str, Any]] | None = None,
    causal_edges: list[dict[str, Any]] | None = None,
    branch_details: dict[str, dict[str, Any]] | None = None,
    shapley: dict[str, Any] | None = None,
    cost_profile: dict[str, Any] | None = None,
    causal_closure: list[dict[str, Any]] | None = None,
    run_id: str | None = None,
    branch_details_cap_bytes: int = DEFAULT_BRANCH_DETAILS_CAP_BYTES,
    compression_step_threshold: int = DEFAULT_COMPRESSION_STEP_THRESHOLD,
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
    `branch_details_cap_bytes` (default `DEFAULT_BRANCH_DETAILS_CAP_BYTES`,
    tracefork-sis.56) caps how much of `branch_details` is actually
    embedded — see `_cap_branch_details`'s docstring. A run with many
    branches no longer silently balloons the report; exceeding it embeds a
    `branch_details_truncated` notice alongside the (still fully valid,
    just partial) `branch_details` dict.
    `compression_step_threshold` (default `DEFAULT_COMPRESSION_STEP_THRESHOLD`)
    is the exchange count at or above which the tape payload is gzip+base64
    compressed instead of embedded as plain (HTML-escaped) JSON — see
    `_inject_script`'s docstring. A report under the threshold is
    byte-for-byte identical to what this function produced before this
    feature existed.
    """
    html = _template_path().read_text(encoding="utf-8")
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
        branch_details_cap_bytes,
    )
    inject = _inject_script(data, compression_step_threshold)
    html = html.replace(_INJECT_MARKER, inject + _INJECT_MARKER, 1)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
