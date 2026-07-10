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

echo "==> uv run pytest -q --cov --cov-report=term-missing --junit-xml=junit.xml"
uv run pytest -q --cov --cov-report=term-missing --junit-xml=junit.xml

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

echo ""
echo "══════════════════════════════════════════════════════════════════"
echo "  PASS — tracefork end-to-end receipt: every gate green, \$0 spent."
echo "══════════════════════════════════════════════════════════════════"
