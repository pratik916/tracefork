# Contributing to tracefork

Thanks for considering a contribution. tracefork is a small, offline-first project —
please keep changes in that spirit.

## Dev setup

Python **3.12** via [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
```

This installs the runtime deps (`anthropic`, `zstandard`, `typer`, `fastapi`, `uvicorn`)
plus the dev toolchain (`pytest`, `pytest-asyncio`, `pytest-cov`, `ruff`, `mypy`).

## Running everything locally

All of the following are offline and **$0** — no `ANTHROPIC_API_KEY`, no network:

```bash
uv run pytest -q                        # full offline suite (1641 tests, $0, no key)
uv run tracefork validate               # self-validation: blame vs injected, known faults
uv run tracefork validate --check       # regression-gate vs experiments/validation_report_committed.json
uv run ruff check .                     # lint
uv run ruff format --check .            # format check
uv run mypy src/tracefork               # type check
uv run python examples/demo_report.py   # generate the demo report (examples/demo_report.html)
```

Run the single-receipt gate before opening a PR — it is the same one thing CI checks,
so a green run here means CI will be green too:

```bash
bash scripts/e2e.sh
```

That's `uv sync`, lint, format check, type check, the full test suite with coverage,
`scripts/check_executed_evidence.py` (CI's sentinel against faked-passing evidence),
`tracefork validate --check`, the replay-fixture-corpus gate, `tracefork bench`, a
package build + `twine check`, and a wheel-install smoke test — one PASS/FAIL verdict.
For a faster inner loop while iterating, the individual commands it wraps still work on
their own:

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy src/tracefork && uv run pytest -q
```

CI (`.github/workflows/ci.yml`) runs most of the same checks as `scripts/e2e.sh` — lint,
format check, type check, tests + coverage with `--junit-xml`,
`scripts/check_executed_evidence.py`, the `validate --check` regression gate, a package
build, `twine check`, and the wheel-install smoke test — across Python 3.12/3.13/3.14,
on every pull request. It does **not** separately run the replay-fixture-corpus gate
(`tracefork replay --check`) or `tracefork bench`; `scripts/e2e.sh` is the strictly
stronger local gate that also covers those two, which is why it — not the CI YAML — is
the one command to run before opening a PR.

## Invariants a PR must respect

These are load-bearing for the project's claims and are enforced by review, not just CI:

- **Offline and $0 stays non-negotiable** for the whole test suite, the spike, `validate`,
  and the demo. If your change needs a new kind of model response or behavior, add it to
  the offline fakes in `src/tracefork/synthetic.py` rather than hitting the real API. The
  only intentionally networked path is `blame` against a real run, and that is
  budget-capped by `BudgetGovernor`.
- **The agent reads time/ids only through `NondetSource`.** Any direct
  `datetime.now()` / `uuid` / `random` call in agent code breaks the determinism boundary
  and invalidates the bit-exactness claim that replay, fork, and blame all depend on.
- **The verifier proves, it does not assert.** Every replayed request body is
  sha256-checked against the tape (`transport.py`); don't weaken this to a soft
  comparison. The drift negative control (`DriftingNondet`) must keep failing — if a
  change makes it pass, that's a regression in the divergence detector, not a fix.

If you're touching `src/tracefork/recorder.py`, `transport.py`, `fork.py`, or `blame.py`,
re-read the relevant section of `CLAUDE.md` first — it documents the seams these files
depend on.

- **Adding, removing, or changing the signature of anything in `tracefork.__all__`, a CLI
  command/flag, or the tape on-disk format is a SemVer-covered change.** Read
  [`docs/stability.md`](docs/stability.md) first — it defines exactly what's covered and the
  deprecation procedure a breaking change must follow (one minor release with a
  `DeprecationWarning`/deprecation notice, removal only in the next major).

## Commit style

- [Conventional commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `chore:`,
  `docs:`, `ci:`, `test:`, `refactor:`).
- No `Co-Authored-By: Claude` trailer or any other AI-authorship marker on commits or PR
  descriptions — attribute commits to yourself only.

## PR flow

1. Branch off `main`.
2. Make your change; keep it additive where possible (see `CLAUDE.md` for the project's
   architecture and invariants).
3. Make sure the full local gate above is green.
4. Open a PR against `main`. CI must pass before merge.

## Questions

Open an issue using the templates in `.github/ISSUE_TEMPLATE/`, or see `SECURITY.md` if
you're reporting a vulnerability rather than a bug.
