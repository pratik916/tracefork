# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Framework adapters** (`adapters/`, opt-in `frameworks` extra): a minimal
  adapter protocol — `bind()` (route the framework's underlying LLM client through
  the *existing* `TraceforkTransport` + `NondetSource`), `on_step()` (map a
  framework callback event to a neutral `Step`), `teardown()` — plus a
  `StepDAG`/`from_run_tree` normalizer that overlays a run's structure on the tape
  (byte seam stays at httpx; callbacks are observer-only annotation). Ships a
  **LangChain/LangGraph** adapter: injects the tracefork transport into
  `ChatOpenAI` (`root_client.copy(http_client=…)`) and `ChatAnthropic` (no
  `http_client` field — a fresh `anthropic` client seeded via `object.__setattr__`
  before its cached-property client is built), a `BaseCallbackHandler` step
  collector, and a **tape-backed LangGraph checkpointer** for bit-exact, $0
  time-travel replay. Registered via the plugin registry (`tracefork.adapters`
  entry-point group, same security gating as every other seam). `langchain-*`
  /`langgraph` are optional and all imports are guarded — `import tracefork` and
  the full offline suite run with none installed; the framework-facing wrappers
  are exercised against the real library when present and skipped otherwise.
- **Three more framework adapters**, each its own optional extra, each binding at
  the framework's actual model-call chokepoint (never a second capture path):
  **OpenAI Agents SDK** (`adapters/openai_agents.py`, `pip install
  'tracefork[openai-agents]'`) — defensive attribute-search injection into a model
  wrapper's underlying `openai` client, plus `bind_default_client()` for the SDK's
  own documented `agents.set_default_openai_client()`, and a real `TracingProcessor`
  (`make_tracing_processor()`) for span/trace step visibility; **CrewAI**
  (`adapters/crewai.py`, `pip install 'tracefork[crewai]'`) — targets LiteLLM's
  documented `client_session`/`aclient_session` custom-httpx-client surface (CrewAI
  routes every model call through LiteLLM, never touching httpx itself), plus a
  `crewai_event_bus` listener (`make_event_listener()`) over crew/agent/task/tool/
  LLM-call boundaries; **AutoGen** (`adapters/autogen.py`, `pip install
  'tracefork[autogen]'`, `autogen-core`/`autogen-ext`) — the same defensive
  client-attribute injection for an AutoGen model client, plus a message-level
  `InterventionHandler` (`make_intervention_handler()`) that is pass-through only
  (never `DropMessage`), so it stays an annotation layer. All three are guarded the
  same way as LangChain: `import tracefork` and the whole offline suite run with
  none of them installed; each framework-neutral core is offline-tested against
  synthetic events, and the thin real wrapper classes are only reachable — and only
  smoke-tested — when the framework is actually installed (`pytest.importorskip`).
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
