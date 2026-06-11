# CLAUDE.md

This file guides Claude Code when working in the `tracefork` repository.

## What this is

`tracefork` is a time-travel debugger for AI agents: record an agent run to a
content-addressed **tape**, replay it **bit-exact for $0** (hash-verified), fork any
step, and measure causal blame with confidence intervals — the instrument itself
validated against runs with injected, known root-cause faults.

**Current state: Spike 0 only.** Spike 0 de-risks the one load-bearing assumption
(bit-exact, offline, no-key replay within a declared determinism boundary). The full
product (fork UI, causal engine, fault-injection validation) is not built yet. The
design/feature list is `../ideas/2026-06-11-tracefork-features.md`; the spike finding is
`SPIKE0.md`.

## Commands

Python is **3.12 via uv**. Everything is offline and $0 — **no `ANTHROPIC_API_KEY`,
no network**. Always prefix with `uv run`.

```bash
uv sync --extra dev                  # install (anthropic + pytest)
uv run python -m tracefork_spike     # run the spike, print the bit-exact replay receipt
uv run pytest -q                     # full offline suite (8 tests)
uv run pytest tests/test_spike0.py::test_negative_control_drift_is_detected -q   # one test
```

## Architecture (the parts that span files)

The spine is a **record/replay seam at the Anthropic SDK's httpx boundary**, plus a
**nondeterminism-virtualization seam** the agent reads time/ids through. Bit-exactness
is the contract between them.

- `nondet.py` — `NondetSource` is the *only* way the agent gets time/ids.
  `RecordingNondet` draws real values and logs them; `ReplayNondet` serves them back in
  order; `DriftingNondet` is the negative control (fresh values → forced divergence).
  `find_divergence()` unwraps a `DivergenceError` from the `APIConnectionError` the SDK
  wraps transport exceptions in — **keep this; without it a real divergence looks like a
  network blip.**
- `transport.py` — `TraceforkTransport(httpx.BaseTransport)` is the capture seam. Record
  mode tees request+response bytes into the tape; replay mode serves recorded response
  bytes and sha256-asserts each request body matches the tape (the divergence detector).
  A replay transport has **no inner transport**, so any unrecorded request is a hard
  error, never a silent network call.
- `tape.py` — `Tape` is content-addressed (sha256 blobs) + an ordered event log,
  persistable to SQLite, with a hash-chain `digest()` fingerprint.
- `fake_llm.py` — `FakeAnthropicTransport` emits real Anthropic **wire-format** JSON so
  the genuine SDK parses it; this is what makes the spike offline/$0. Swapping it for the
  real network is a one-line change with the record/replay machinery unchanged.
- `agent.py` — a toy tool-use agent on the **real** `anthropic` SDK; its `book_flight`
  tool injects genuine nondeterminism (timestamp + id) that must be virtualized for
  replay to be byte-exact.
- `spike.py` — orchestration: record → save → load → replay → verify + negative control,
  and prints the receipt. `record_replay_verify()` returns a structured dict (used by
  the CLI and tests).

## Invariants / conventions

- **Offline and $0 is non-negotiable** for the spike and its tests — no key, no network.
  The fake endpoint is the seam; add to it rather than reaching for the real API.
- **The agent must read time/ids only through `NondetSource`** — any direct
  `datetime.now()` / `uuid` / `random` breaks the determinism boundary and the
  bit-exactness claim.
- **The verifier proves, not asserts** — every request body is hash-checked against the
  tape; the negative control must keep failing (drift detected) or the proof is vacuous.
- **Declared determinism boundary (v1):** single-process, synchronous, clock + id
  nondeterminism. Threads/asyncio/subprocess are out of scope — see `SPIKE0.md`.
- **No `Co-Authored-By: Claude` trailer** on commits in this repo (public portfolio repo,
  sole-author attribution).
- **Model IDs / pricing / SDK usage:** consult the `claude-api` skill before writing or
  editing any Anthropic integration code rather than relying on memory.
- `docs/superpowers/`, `.beads/`, `planning/` are gitignored local scaffolding.
