"""FastAPI server for tracefork live mode.

Serves the report HTML at / and JSON endpoints at /api/run/{run_id}
and /api/branch/{branch_id}. Single-threaded (uvicorn --workers 1).
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

from .report import _tape_to_data, _template_path
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


@app.get("/", response_class=HTMLResponse)
async def serve_ui() -> HTMLResponse:
    html = _template_path().read_text()
    # Empty server URL → the UI fetches same-origin (works on any --port).
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
