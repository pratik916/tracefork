"""FastAPI server for tracefork live mode.

Serves the report HTML at / and JSON endpoints at /api/run/{run_id},
/api/branch/{branch_id}, and /api/session/{session_id}. Single-threaded
(uvicorn --workers 1).
"""

from __future__ import annotations

import base64
import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel

from .cost_profile import compute_cost_profile, cost_profile_to_dict
from .fork import BranchSpec, ForkEngine
from .fork_allowlist import AgentNotAllowlistedError, estimate_single_fork_usd, resolve_agent_fn
from .interop import locate_span_step, locate_trace
from .live import tail_checkpoint
from .report import _runs_template_path, _tape_to_data, _template_path
from .store import ForkPointDriftError, TapeStore

# No CORS middleware: the UI is served same-origin by this app and uvicorn
# binds to 127.0.0.1 (see the `serve` CLI command), so cross-origin access is
# neither needed nor desirable.
app = FastAPI(title="tracefork", docs_url=None, redoc_url=None)

_store: TapeStore | None = None


def get_store() -> TapeStore:
    if _store is None:
        raise RuntimeError("Store not initialized — call init_store() first")
    return _store


def init_store(db_path: str = "store.db") -> None:
    global _store
    _store = TapeStore(db_path)


# ── click-to-fork (tracefork-bge.36) ────────────────────────────────────────
#
# Nothing is fork-able through these endpoints unless the operator names it
# via `init_fork_allowlist` (wired from `cli.py`'s `serve --allow-fork-agent`
# flag) — an empty allowlist (the default) 403s every request.
_fork_allowlist: dict[str, str] = {}


def init_fork_allowlist(allowlist: dict[str, str]) -> None:
    global _fork_allowlist
    _fork_allowlist = dict(allowlist)


class ForkEstimateRequest(BaseModel):
    agent_name: str
    step: int
    mutated_response_b64: str


class ForkRequest(BaseModel):
    agent_name: str
    step: int
    mutated_response_b64: str
    confirm: bool = False
    mutation_desc: str = ""


@app.post("/api/run/{run_id}/fork/estimate")
async def estimate_fork(run_id: str, body: ForkEstimateRequest) -> JSONResponse:
    """Price ONE targeted fork at `body.step` with no side effects (no fork
    executed, no branch persisted) — see `fork_allowlist.estimate_single_fork_usd`."""
    store = get_store()
    try:
        tape = store.load_tape(run_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"run {run_id!r} not found") from None
    try:
        resolve_agent_fn(body.agent_name, _fork_allowlist)
    except AgentNotAllowlistedError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from None
    base64.b64decode(body.mutated_response_b64)  # validate the encoding up front
    est_usd = estimate_single_fork_usd(tape, body.step)
    return JSONResponse({"agent_name": body.agent_name, "step": body.step, "est_usd": est_usd})


@app.post("/api/run/{run_id}/fork")
async def do_fork(run_id: str, body: ForkRequest) -> JSONResponse:
    """Execute a REAL fork (`fork.ForkEngine.fork`) and persist the branch —
    requires `body.confirm is True` (an explicit cost-confirmation gate,
    mirroring the report's UI-level "Confirm & Fork" step) and an
    allowlisted `body.agent_name` (see `fork_allowlist.py`)."""
    store = get_store()
    try:
        tape = store.load_tape(run_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"run {run_id!r} not found") from None
    try:
        agent_fn = resolve_agent_fn(body.agent_name, _fork_allowlist)
    except AgentNotAllowlistedError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from None
    if not body.confirm:
        raise HTTPException(status_code=400, detail="confirm must be true to execute a real fork")

    mutated_response = base64.b64decode(body.mutated_response_b64)
    spec = BranchSpec(
        divergence_step=body.step,
        mutated_response=mutated_response,
        mutation_desc=body.mutation_desc,
    )
    branch = ForkEngine.fork(tape, spec, agent_fn)
    branch_id = store.save_branch(
        parent_run_id=run_id,
        divergence_step=body.step,
        delta_tape=branch.delta_tape,
        mutation_desc=body.mutation_desc,
        branch_digest=branch.branch_digest,
    )
    return JSONResponse(
        {
            "branch_id": branch_id,
            "parent_run_id": run_id,
            "divergence_step": body.step,
            "delta_exchanges": len(branch.delta_tape.exchanges),
            "mutation_desc": body.mutation_desc,
        }
    )


@app.get("/", response_class=HTMLResponse)
async def serve_ui() -> HTMLResponse:
    html = _template_path().read_text()
    # Empty server URL → the UI fetches same-origin (works on any --port).
    inject = "\n<script>\nwindow.__TRACEFORK_SERVER_URL__ = '';\n</script>\n"
    html = html.replace("</head>", inject + "</head>", 1)
    return HTMLResponse(html)


@app.get("/runs", response_class=HTMLResponse)
async def serve_runs_page() -> HTMLResponse:
    """Multi-run dashboard / run-picker page (tracefork-bge.67): a plain
    table over the already-existing `/api/runs` endpoint, linking each row
    to `/?run_id=<id>` (the query-param contract `report.html`'s `loadData`
    already reads). Same live-mode injection as `serve_ui`, so `runs.html`'s
    own boot logic can tell it's being served same-origin."""
    html = _runs_template_path().read_text()
    inject = "\n<script>\nwindow.__TRACEFORK_SERVER_URL__ = '';\n</script>\n"
    html = html.replace("</head>", inject + "</head>", 1)
    return HTMLResponse(html)


@app.get("/api/runs")
async def list_runs() -> JSONResponse:
    store = get_store()
    return JSONResponse(store.list_runs())


@app.get("/api/run/{run_id}")
async def get_run(run_id: str) -> JSONResponse:
    store = get_store()
    try:
        tape = store.load_tape(run_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"run {run_id!r} not found") from None
    data = _tape_to_data(tape)
    data["run_id"] = run_id
    # Additive: the run's saved branches (tracefork-bge.15's fork-tree panel
    # data) — `list_branches` is the no-`delta_tape`-fetch summary shape, so
    # this costs no extra `load_branch` round trips. Existing `/api/run`
    # consumers that ignore the new key are unaffected.
    data["branches"] = store.list_branches(run_id)
    # Additive: the run's persisted causal edges (blame + Shapley results
    # `tracefork blame` already saved via `save_blame_report`/
    # `save_shapley_report`) — drives the fork-tree panel's causal heatmap
    # overlay (tracefork-bge.35). Existing `/api/run` consumers that ignore
    # the new key are unaffected.
    data["causal_edges"] = store.causal_edges_for_run(run_id)
    # Additive: per-model/per-tool cost dashboard (tracefork-bge.52), same
    # one-line pattern as the `branches`/`causal_edges` lines above.
    data["cost_profile"] = cost_profile_to_dict(compute_cost_profile(tape))
    # Additive: external-anchor vocabulary (tracefork-bge.70) — responsible
    # blame edges reachable via fork-promotion lineage, possibly from other
    # run_ids. `data["run_id"]` is already set above.
    data["causal_closure"] = store.causal_closure(run_id)
    return JSONResponse(data)


@app.get("/api/branch/{branch_id}")
async def get_branch(branch_id: str) -> JSONResponse:
    store = get_store()
    try:
        branch = store.load_branch(branch_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"branch {branch_id!r} not found") from None
    except ForkPointDriftError as exc:
        # The cited fork point has drifted since the branch was made (see
        # store.py's load_branch docstring) — a hard, citable conflict, not a
        # missing resource or an unhandled 500.
        raise HTTPException(status_code=409, detail=str(exc)) from None
    data = _tape_to_data(branch["delta_tape"])
    data["branch_id"] = branch_id
    data["parent_run_id"] = branch["parent_run_id"]
    data["divergence_step"] = branch["divergence_step"]
    data["mutation_desc"] = branch["mutation_desc"]
    data["branch_digest"] = branch["branch_digest"]
    return JSONResponse(data)


@app.get("/api/branch/{run_id}/related")
async def get_branch_related(run_id: str) -> JSONResponse:
    """Branch DAG relationship queries for RUN_ID -- descendants, ancestors,
    siblings. No 404: the store methods themselves return empty lists for an
    unknown/root/leaf id rather than raising, so this always returns 200."""
    store = get_store()
    return JSONResponse(
        {
            "run_id": run_id,
            "descendants": store.branch_descendants(run_id),
            "ancestors": store.branch_ancestors(run_id),
            "siblings": store.branch_siblings(run_id),
        }
    )


@app.get("/otel/{trace_id}")
async def otel_locate_trace(trace_id: str) -> RedirectResponse:
    """OTel exemplar back-link (tracefork-bge.53): redirect a
    `build_otel_trace`-exported `trace_id` to the report view for the run it
    was derived from. Localhost-only by construction — this app only ever
    binds 127.0.0.1 with no CORS middleware (see the module docstring)."""
    store = get_store()
    run_id = locate_trace(store, trace_id)
    if run_id is None:
        raise HTTPException(status_code=404, detail=f"trace_id {trace_id!r} not found")
    return RedirectResponse(url=f"/?run_id={run_id}")


@app.get("/otel/{trace_id}/{span_id}")
async def otel_locate_span(trace_id: str, span_id: str) -> RedirectResponse:
    """Same as `otel_locate_trace`, additionally resolving `span_id` to its
    exchange step so the report auto-selects it. A `span_id` that doesn't
    resolve to any exchange (e.g. the trace's own root span) redirects to
    the run with no `step` param, same as `otel_locate_trace`."""
    store = get_store()
    run_id = locate_trace(store, trace_id)
    if run_id is None:
        raise HTTPException(status_code=404, detail=f"trace_id {trace_id!r} not found")
    tape = store.load_tape(run_id)
    step = locate_span_step(tape, trace_id, span_id)
    url = f"/?run_id={run_id}" + (f"&step={step}" if step is not None else "")
    return RedirectResponse(url=url)


@app.get("/api/checkpoint/tail")
async def tail_checkpoint_endpoint(
    path: str, since_seq: int = 0, poll_interval: float = 0.25, max_polls: int | None = None
) -> StreamingResponse:
    """Live-tail SSE endpoint (tracefork-bge.61) over a
    `checkpoint.CheckpointWriter`-backed recording at `path`. Read-only —
    never controls the writer's own process (see `live.py`'s module
    docstring for the interactive-breakpoint scope note)."""
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"no checkpoint file at {path!r}")
    return StreamingResponse(
        tail_checkpoint(
            path, since_seq=since_seq, poll_interval=poll_interval, max_polls=max_polls
        ),
        media_type="text/event-stream",
    )


@app.get("/api/session/{session_id}")
async def get_session(session_id: str) -> JSONResponse:
    """A session's root run_id/created_at plus every tape reachable via its
    spawn edges (`TapeStore.session_tapes`'s BFS) — out of scope for
    report.html's UI itself (see store.py's module docstring), an
    additive JSON surface for a future/external consumer of the
    orchestration graph."""
    store = get_store()
    try:
        session = store.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"session {session_id!r} not found") from None
    session["tapes"] = store.session_tapes(session_id)
    return JSONResponse(session)
