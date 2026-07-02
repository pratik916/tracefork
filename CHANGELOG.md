# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **OTel GenAI / OpenInference interop** (`interop.py`, `tracefork export`/`ingest`):
  adopts `gen_ai.*` attribute names (pinned semconv version) for the normalized
  provider view; exports a tape + blame report as an OTel GenAI trace (OTLP/JSON
  spans) or an OpenInference-style dataset, both plain JSON — no `opentelemetry-sdk`
  install required; ingests either format back into a tape's step structure for
  **blame-by-re-execution**, explicitly **not** $0 bit-exact replay (an ingested
  tape's `boundary` is marked `OTEL_INGESTED_BOUNDARY` and diverges on
  `replay`/`fork` by design — proven in `tests/test_interop.py`).
- **Opt-in observability extra** (`pip install 'tracefork[observability]'`,
  `observability.py`): a structlog JSON logging pipeline and OTel
  self-instrumentation of record/replay/fork/blame, off by default and double
  opt-in even when installed — the offline/$0 core and its test suite need neither
  package.

## [0.1.0] - 2026-07-02

### Added

- **Record/replay** at the Anthropic SDK's httpx transport boundary
  (`TraceforkTransport` / `AsyncTraceforkTransport`), streaming-SSE capable, with
  bit-exact replay proven by sha256-checking every replayed request body against the
  recorded tape — and drift detection that fails loudly on divergence rather than
  silently falling back to the network.
- **Content-addressed tape format** (`Tape`) — sha256 blobs plus an ordered event log,
  JSON + base64 (never pickle), persistable to SQLite, with a hash-chain `digest()`
  fingerprint.
- **Nondeterminism virtualization** (`NondetSource`) — the only path through which an
  agent reads time and ids, with `RecordingNondet`, `ReplayNondet`, and a `DriftingNondet`
  negative control that proves the divergence detector actually detects divergence.
- **Three-phase fork engine** (`ForkEngine`, `ForkTransport`) — prefix-replay ($0),
  mutation-injection (swap a response), and tail-record (the recorded counterfactual
  continuation), re-running the same agent that produced the original tape.
- **Causal blame engine** (`BlameEngine`) — forks each step `k` times, re-runs the agent,
  grades outcomes via an `Oracle`, and ranks steps by flip-rate with Wilson score
  confidence intervals; a `BudgetGovernor` estimates dollar cost from the pricing table
  and refuses to exceed a caller-supplied budget before making any real API calls.
- **Fault-injection self-validation suite** (`faults.py`, `validate.py`) — five fault
  classes with markers embedded in valid Anthropic JSON, scored end-to-end offline
  against a synthetic fault-aware agent: **1.00 top-1 precision** across all five classes,
  with an enforced negative-control threshold so the proof isn't vacuous.
- **Single-file web report/UI** (`report.py`, `server.py`, `web/report.html`) — a
  dependency-free, three-panel HTML report (timeline, exchange detail, blame ranking)
  either rendered statically or served live via FastAPI (`serve`, 127.0.0.1, no CORS).
- **CLI** (`cli.py`, Typer) — `replay`, `verify`, `fork`, `blame`, `report`, `serve`,
  `validate`.
- `src/tracefork_spike/` — the original Spike 0 that de-risked bit-exact, no-key replay
  within the declared determinism boundary.

[Unreleased]: https://github.com/pratik916/tracefork/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/pratik916/tracefork/releases/tag/v0.1.0
