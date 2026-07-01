# CLAUDE.md

This file guides Claude Code when working in the `tracefork` repository.

## What this is

`tracefork` is a time-travel debugger for AI agents: record an agent run to a
content-addressed **tape**, replay it **bit-exact for $0** (hash-verified), fork any
step, and measure causal blame with confidence intervals — the instrument itself
validated against runs with injected, known root-cause faults.

**Current state: v1 built.** All five product pillars work offline and are tested
(65 tests, $0): streaming-capable record/replay with drift detection, the three-phase
fork engine, the causal blame engine with Wilson CIs and a budget governor, the
single-file web report/UI, and the fault-injection self-validation suite (5 fault
classes at 1.00 top-1 precision). `src/tracefork_spike/` keeps the original Spike 0 that
de-risked the load-bearing assumption (bit-exact, no-key replay within a declared
determinism boundary). Design/feature list: `../ideas/2026-06-11-tracefork-features.md`;
spike finding: `SPIKE0.md`.

## Commands

Python is **3.12 via uv**. The tests, the spike, `validate`, the demo, and
record/replay/fork are offline and $0 — **no `ANTHROPIC_API_KEY`, no network**. Only
`blame` against a *real* run hits the live API (budget-capped). Always prefix `uv run`.

```bash
uv sync --extra dev                  # install (anthropic, zstandard, typer, fastapi, uvicorn + pytest)
uv run pytest -q                     # full offline suite (65 tests)
uv run pytest tests/test_faults.py::test_validation_runner_fingers_fault_step -q   # one test
uv run tracefork validate            # self-validation: blame vs injected, known faults
uv run tracefork validate --check    # regression-gate vs experiments/validation_report_committed.json
uv run python examples/demo_report.py   # write examples/demo_report.html (the README screenshot)
uv run python -m tracefork_spike     # the original Spike 0 bit-exact replay receipt
uv run tracefork --help              # replay, verify, fork, report, serve, blame, validate
```

## Architecture (the parts that span files)

The spine is a **record/replay seam at the Anthropic SDK's httpx boundary**, plus a
**nondeterminism-virtualization seam** the agent reads time/ids through. Bit-exactness
is the contract between them.

The product lives in `src/tracefork/`:

- `nondet.py` — `NondetSource` is the *only* way the agent gets time/ids.
  `RecordingNondet` draws real values and logs them; `ReplayNondet` serves them back in
  order; `DriftingNondet` is the negative control (fresh values → forced divergence).
  `find_divergence()` unwraps a `DivergenceError` from the `APIConnectionError` the SDK
  wraps transport exceptions in — **keep this; without it a real divergence looks like a
  network blip.**
- `transport.py` — `TraceforkTransport` (sync) + `AsyncTraceforkTransport` (async) are the
  capture seam, streaming-SSE capable (buffer via `.read()`/`.aread()`). Record mode tees
  request+response bytes into the tape; replay mode serves recorded bytes and
  sha256-asserts each request body matches (the divergence detector). A replay transport
  has **no inner transport**, so any unrecorded request is a hard error.
- `tape.py` — `Tape` is content-addressed (sha256 blobs) + an ordered event log,
  JSON+base64 in memory, persistable to SQLite, with a hash-chain `digest()` fingerprint.
  (`to_bytes`/`from_bytes` are JSON, **not pickle** — no arbitrary-code-execution risk.)
- `recorder.py` — `Recorder` context manager wraps a real `anthropic.Anthropic` at its
  `_client._transport` seam (via `client.copy(http_client=...)`, so base_url / auth_token /
  default headers are preserved). Patches `uuid.uuid4` globally; **does not** patch
  `datetime.datetime` (immutable C type in 3.12+, and a subclass breaks the SDK's pydantic
  schema builder) — agents needing deterministic clocks read `NondetSource` directly.
- `fork.py` — `ForkTransport` runs three phases: prefix-replay ($0, request asserted to
  match the parent), mutation-injection (same request, swapped response), tail-record (the
  counterfactual continuation). `Branch` carries `prefix_replayed`/`tail_recorded` counts.
  `ForkEngine.fork()` re-runs the **same** agent that produced the tape.
- `store.py` — `TapeStore`, SQLite persistence for tapes + the branch DAG.
- `blame.py` — `BlameEngine.rank()` forks each step `k` times, re-runs the agent, grades
  via an `Oracle`, counts flips vs. the parent outcome; `wilson_ci()` for intervals;
  `BudgetGovernor` estimates tail-call cost from `constants.PRICING_TABLE` before spend and
  `rank()` raises `BudgetExceededError` if the estimate exceeds `budget_usd`.
- `faults.py` / `validate.py` — 5 fault classes (valid JSON, marker **inside** a content
  field) + the self-validation runner; a synthetic agent echoes each response forward so an
  injected fault propagates to a fault-aware tail. `run_all_fault_classes()` scores top-1.
  **Scope (don't overstate):** the fixture is a positive-vs-inert control on a short tape —
  it proves the engine is genuinely causal (not a fixed-slot artifact), not that it
  discriminates among competing causes on long tapes. See README → Validation scope.
- `report.py` / `server.py` / `web/report.html` — the single-file, dependency-free
  three-panel UI; `report.py` injects tape JSON (HTML-escaped against `</script>`
  breakout), `server.py` is FastAPI same-origin (no CORS, binds 127.0.0.1).
- `wire.py` / `synthetic.py` — Anthropic wire-format builders and the offline
  Scripted/FaultAware fake transports, in the **package** so production never imports from
  `tests/`; `tests/fakes.py` re-exports them.
- `cli.py` — Typer entry point for all seven commands.

`src/tracefork_spike/` holds the original Spike 0 (`fake_llm.py`, `agent.py`, `spike.py`):
record → save → load → replay → verify + negative control, with its own tests.

## Invariants / conventions

- **Offline and $0 is non-negotiable** for the whole test suite, the spike, `validate`,
  and the demo — no key, no network. The synthetic transports (`synthetic.py`) are the
  seam; add to them rather than reaching for the real API. (`blame` on a real run is the
  one budget-capped exception.)
- **The agent must read time/ids only through `NondetSource`** — any direct
  `datetime.now()` / `uuid` / `random` breaks the determinism boundary and the
  bit-exactness claim.
- **The verifier proves, not asserts** — every request body is hash-checked against the
  tape; the negative control must keep failing (drift detected) or the proof is vacuous.
- **Declared determinism boundary (v1):** single-process (sync **or** asyncio), clock +
  id nondeterminism captured through `NondetSource`. Threads/subprocess are out of scope;
  fork and blame additionally assume the agent rebuilds its prefix deterministically (the
  property replay proves) — see `SPIKE0.md`.
- **No `Co-Authored-By: Claude` trailer** on commits in this repo (public portfolio repo,
  sole-author attribution).
- **Model IDs / pricing / SDK usage:** consult the `claude-api` skill before writing or
  editing any Anthropic integration code rather than relying on memory.
- `docs/superpowers/`, `.beads/`, `planning/` are gitignored local scaffolding (but
  `docs/demo.png` is committed). Runtime artifacts (`store.db`, `report.html`,
  `blame_*.json`, `validation_report.json`, `examples/demo_report.html`) are gitignored.
