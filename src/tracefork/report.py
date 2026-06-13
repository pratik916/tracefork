"""Report generator: produces a self-contained HTML file from a Tape.

In static mode, the entire tape data is serialized as JSON and injected
into the HTML template as `window.__TRACEFORK_DATA__ = {...}`.
"""
from __future__ import annotations

import json
from pathlib import Path


_TEMPLATE_PATH = Path(__file__).parent.parent.parent / "web" / "report.html"
_INJECT_MARKER = "</head>"


def _tape_to_data(tape, blame: dict | None = None) -> dict:
    """Convert a Tape to the JSON shape expected by the web UI."""
    exchanges = []
    for i, (req_bytes, resp_bytes) in enumerate(tape.exchanges):
        try:
            req_json = json.loads(req_bytes.decode())
        except Exception:
            req_json = {"_raw": req_bytes.hex()}

        try:
            resp_json = json.loads(resp_bytes.decode())
        except Exception:
            # SSE stream — extract first data line
            lines = resp_bytes.decode(errors="replace").splitlines()
            data_lines = [l[6:] for l in lines if l.startswith("data: ") and l != "data: [DONE]"]
            try:
                resp_json = json.loads(data_lines[0]) if data_lines else {"_raw": "sse"}
            except Exception:
                resp_json = {"_raw": "sse"}

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
                        preview = f"→ {block.get('name', 'tool')}({json.dumps(block.get('input', {}))[:60]})"
                        break
        except Exception:
            pass

        exchanges.append({
            "role": role,
            "preview": preview,
            "request": req_json,
            "response_preview": resp_json,
        })

    return {
        "agent_name": tape.agent_name,
        "exchanges": exchanges,
        "blame": blame or {},
        "created_at": "",
        "fingerprint": tape.digest()[:16],
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
) -> None:
    """Write a self-contained HTML report to `output_path`.

    The tape data is injected before </head> so the UI loads it synchronously.
    """
    html = _TEMPLATE_PATH.read_text()
    data = _tape_to_data(tape, blame)
    inject = f"\n<script>\nwindow.__TRACEFORK_DATA__ = {_safe_json(data)};\n</script>\n"
    html = html.replace(_INJECT_MARKER, inject + _INJECT_MARKER, 1)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
