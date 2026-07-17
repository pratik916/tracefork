#!/usr/bin/env bash
# scripts/mutation.sh — nightly mutation-testing pass (tracefork-bge.49).
#
# Mutmut-mutates the four proof-bearing modules named in [tool.mutmut] of
# pyproject.toml (transport.py/tape.py/matcher.py/blame.py) and summarizes
# how many mutants the offline test suite actually kills. Informational only
# — see scripts/mutation_summary.py's module docstring for why this never
# gates a merge: this script always exits 0 by design (mutmut's own exit
# code is deliberately not propagated). Offline/$0 throughout: no
# ANTHROPIC_API_KEY, no network beyond the local test run.
#
#   bash scripts/mutation.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "==> uv sync --extra dev --extra mutation"
uv sync --extra dev --extra mutation

echo "==> uv run pytest -q --cov (seed coverage data for --use-coverage test selection)"
uv run pytest -q --cov

echo "==> uv run mutmut run --use-coverage"
uv run mutmut run --use-coverage || true

echo "==> uv run mutmut junitxml > mutmut-junit.xml"
uv run mutmut junitxml > mutmut-junit.xml

echo "==> uv run python scripts/mutation_summary.py --junit-xml mutmut-junit.xml"
uv run python scripts/mutation_summary.py --junit-xml mutmut-junit.xml || true

echo ""
echo "══════════════════════════════════════════════════════════════════"
echo "  mutation pass complete — informational only, see mutmut-junit.xml"
echo "══════════════════════════════════════════════════════════════════"
