"""Regression test for examples/demo_report.py — the artifact docs/demo.png is a
screenshot of, and every distinctive claim in README.md's headline paragraph should be
visible in (v1.0.0 "rebuild the demo report" item).

Runs the real script end to end (offline, $0) and asserts the generated report's embedded
data is actually rich — >= 6 exchanges, and non-empty replay/branches/causal_edges/shapley/
cost_profile — so this can never silently regress back to the near-empty 2-exchange,
blame-only report a prior version of this script produced.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import examples.demo_report as demo_report

_DATA_RE = re.compile(r"window\.__TRACEFORK_DATA__ = (.*?);\n</script>", re.S)


def _embedded_data(html_path: Path) -> dict:
    html = html_path.read_text(encoding="utf-8")
    match = _DATA_RE.search(html)
    assert match is not None, "report HTML has no window.__TRACEFORK_DATA__ payload"
    return json.loads(match.group(1))


def test_demo_report_produces_a_rich_report(tmp_path, monkeypatch):
    # Redirect the script's fixed `examples/demo_report.html` output path into tmp_path so
    # this test never depends on (or clobbers) the real committed demo_report.html, then
    # restore it via monkeypatch's own teardown.
    monkeypatch.setattr(demo_report, "__file__", str(tmp_path / "demo_report.py"))

    demo_report.main()

    out = tmp_path / "demo_report.html"
    assert out.exists(), "demo_report.main() did not write demo_report.html next to itself"
    data = _embedded_data(out)

    assert len(data["exchanges"]) >= 6, "demo tape must have >= 6 exchanges"

    assert data["replay"], "replay receipt must be populated"
    assert data["replay"]["bit_exact"] is True
    assert data["replay"]["matched"] == data["replay"]["total"] == len(data["exchanges"])

    assert data["branches"], "fork tree must have real branches, not an empty list"
    assert len(data["branches"]) >= 3

    assert data["causal_edges"], "causal_edges must be populated (persisted blame + shapley)"
    assert any(e["method"] == "blame" for e in data["causal_edges"])
    assert any(e["method"] == "shapley" for e in data["causal_edges"])

    assert data["branch_details"], "branch_details must be populated"
    assert set(data["branch_details"]) == {b["branch_id"] for b in data["branches"]}

    assert data["shapley"], "shapley must be populated"
    # The demo's whole point: naive flip-rate ties root(0)/echo(1), but Shapley separates
    # them by necessity -- assert that story actually holds in the generated data.
    assert data["shapley"]["0"]["necessity"] is True
    assert data["shapley"]["1"]["necessity"] is False

    assert data["cost_profile"], "cost_profile must be populated"
    assert data["cost_profile"]["total_cost_usd"] > 0

    assert data["causal_closure"], "causal_closure must be populated"

    assert data["run_id"], "run_id must be populated"

    # NOT asserted here on purpose: `data["created_at"]` is still hardcoded to `""` by
    # `report.py`'s `_tape_to_data` (line ~120) regardless of what this script does -- a
    # pre-existing, separately-scoped defect (see item 28's text: "distinct from the
    # report.py:120 display-only hardcoded ''"), not something `examples/demo_report.py` can
    # fix from the outside (`generate_report`/`_tape_to_data` take no `created_at` parameter).
    # `web/report.html`'s run-meta line (`agent_name · N exchanges · created_at`) will keep
    # showing a dangling trailing separator until that's fixed in `report.py`/`web/
    # report.html`, neither of which this file owns.
