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


def _tape_to_data(tape, blame: dict | None = None, replay: dict | None = None) -> dict:
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
        "content_redacted": tape.content_redacted,
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
) -> None:
    """Write a self-contained HTML report to `output_path`.

    The tape data is injected before </head> so the UI loads it synchronously.
    `replay` (optional) is the JSON-safe dict from
    `tracefork.replay.verification_result_to_dict` — a bit-exactness receipt
    plus a structured divergence diagnostic when the replay drifted.
    """
    html = _template_path().read_text()
    data = _tape_to_data(tape, blame, replay)
    inject = f"\n<script>\nwindow.__TRACEFORK_DATA__ = {_safe_json(data)};\n</script>\n"
    html = html.replace(_INJECT_MARKER, inject + _INJECT_MARKER, 1)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
