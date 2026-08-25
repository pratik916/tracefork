#!/usr/bin/env bash
# scripts/e2e.sh — the single-receipt end-to-end gate.
#
# Runs everything CI checks, plus the self-validation / replay-fixture-corpus
# / competing-fault-benchmark proofs and a packaging smoke test, as ONE script
# — so a single PASS banner is the whole "does tracefork work, end to end?"
# answer. Offline and $0 throughout (see CLAUDE.md's Commands section): no
# ANTHROPIC_API_KEY, no network, anywhere in this script.
#
#   bash scripts/e2e.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "==> uv sync --extra dev"
uv sync --extra dev

echo "==> uv run ruff check ."
uv run ruff check .

echo "==> uv run ruff format --check ."
uv run ruff format --check .

echo "==> uv run mypy src/tracefork"
uv run mypy src/tracefork

echo "==> uv run python scripts/check_readme_links.py"
uv run python scripts/check_readme_links.py

echo "==> uv run pytest -q --cov --cov-report=term-missing --junit-xml=junit.xml"
uv run pytest -q --cov --cov-report=term-missing --junit-xml=junit.xml

echo "==> checking README.md/CONTRIBUTING.md/CLAUDE.md test-count claims vs junit.xml"
uv run python - <<'PYEOF'
# Fails the gate if a tracked doc's "N tests"/"N offline tests" claim disagrees
# with the real count pytest just collected (junit.xml's <testsuite tests="...">),
# so a future edit to the suite can't silently desync the docs again (v1.0.0
# doc-drift item). SPIKE0.md is deliberately excluded: it scopes its own "8
# tests" claim to tests/test_spike0.py alone, not the whole suite.
import re
import sys
import xml.etree.ElementTree as ET

root = ET.parse("junit.xml").getroot()
if root.tag == "testsuite":
    total = int(root.attrib["tests"])
else:
    total = sum(int(ts.attrib.get("tests", 0)) for ts in root.findall("testsuite"))

pattern = re.compile(r"(\d{2,5}) (?:offline )?tests\b")
bad = []
for path in ("README.md", "CONTRIBUTING.md", "CLAUDE.md"):
    text = open(path, encoding="utf-8").read()
    for n in pattern.findall(text):
        if int(n) != total:
            bad.append((path, n))

if bad:
    for path, n in bad:
        print(f"test-count mismatch in {path}: claims {n} tests, junit.xml has {total}", file=sys.stderr)
    sys.exit(1)
print(f"test-count claims OK ({total} tests)")
PYEOF

echo "==> uv run python scripts/check_executed_evidence.py"
uv run python scripts/check_executed_evidence.py

echo "==> uv run tracefork validate --check"
uv run tracefork validate --check

echo "==> uv run tracefork replay --check experiments/replay_fixtures"
uv run tracefork replay --check experiments/replay_fixtures

echo "==> uv run tracefork bench"
uv run tracefork bench

echo "==> rm -rf dist && uv build && twine check dist/*"
rm -rf dist
uv build
uv run --with twine twine check dist/*

echo "==> uv run python scripts/wheel_smoke.py --wheel dist/tracefork-*.whl"
uv run python scripts/wheel_smoke.py --wheel dist/tracefork-*.whl

echo ""
echo "══════════════════════════════════════════════════════════════════"
echo "  PASS — tracefork end-to-end receipt: every gate green, \$0 spent."
echo "══════════════════════════════════════════════════════════════════"
