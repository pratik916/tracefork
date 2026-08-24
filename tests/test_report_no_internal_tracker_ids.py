"""v1.0.0 readiness item 15: internal `tracefork-bge.NN` bead-tracker ids must
never leak into the shipped web reports -- they're an internal planning
artifact, not something a portfolio reader (or `view-source:`) should see.

Offline, string-assertion test matching this suite's existing convention for
web/*.html (no JS runtime/headless browser -- see test_report_scrubber.py).
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TRACKER_ID_RE = re.compile(r"tracefork-bge\.\d+")


def test_report_html_has_no_internal_tracker_ids():
    content = (REPO_ROOT / "web" / "report.html").read_text()
    matches = TRACKER_ID_RE.findall(content)
    assert matches == [], f"internal tracker ids leaked into report.html: {matches}"


def test_runs_html_has_no_internal_tracker_ids():
    content = (REPO_ROOT / "web" / "runs.html").read_text()
    matches = TRACKER_ID_RE.findall(content)
    assert matches == [], f"internal tracker ids leaked into runs.html: {matches}"


def test_session_report_html_has_no_internal_tracker_ids():
    content = (REPO_ROOT / "web" / "session_report.html").read_text()
    matches = TRACKER_ID_RE.findall(content)
    assert matches == [], f"internal tracker ids leaked into session_report.html: {matches}"
