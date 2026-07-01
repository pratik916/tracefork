## What & why

<!-- Summarize the change and the motivation behind it. -->

## Checklist

- [ ] `uv run pytest -q` is green (65 offline tests, $0, no key)
- [ ] `uv run ruff check .` is green
- [ ] `uv run ruff format --check .` is green
- [ ] `uv run mypy src/tracefork` is green
- [ ] `uv run tracefork validate --check` is green (if this PR touches `blame.py`, `fork.py`, or `faults.py`)
- [ ] Determinism-boundary invariants respected: the agent reads time/ids only via
      `NondetSource`; no new networked test; the verifier still proves via hash-check
      (not assert); the drift negative control still fails as expected
- [ ] Commit messages are conventional-commit style, with no `Co-Authored-By: Claude`
      or other AI-authorship trailer
