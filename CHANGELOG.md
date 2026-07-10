# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Typed `ReplayCertificate` / proof envelope** (new `certificate.py`) — a
  frozen dataclass whose `strength` (`UNVERIFIED` / `HASH_MATCHED` /
  `BIT_EXACT_FULL_REPLAY`) is checked against its own `matched`/`total`/
  fingerprint fields in `__post_init__`: `BIT_EXACT_FULL_REPLAY` raises
  `ProofEnvelopeError` unless every exchange matched *and* the recorded and
  replayed fingerprints are identical, so a caller can no longer overclaim
  bit-exactness with a bare boolean. `certificate_from_verification(result,
  tape)` is the sole function that derives a certificate from a real
  `ReplayVerifier.verify()` result, recomputing the recorded fingerprint from
  `tape.digest()` rather than trusting the result's own field. Purely
  additive: `VerificationResult` gains an optional `certificate` field
  (`None` unless explicitly attached — `ReplayVerifier.verify()` itself is
  unchanged), and `tracefork replay`/`tracefork verify` attach one and print
  an extra `certificate` line in the receipt; every other caller (fixture
  corpus check, the web report) is byte-for-byte unaffected.

- **`tracefork prune`** (`store.py`, `cli.py`) — a retention/GC command for
  tapes and branches, mirroring git gc / borg prune's mark-and-sweep-with-
  soft-archive discipline: nothing is ever hard-deleted. `TapeStore.prune(*,
  older_than_iso=None, run_ids=None, dry_run=False) -> PruneReport` archives a
  matching tape's branches, then the tape row, into new
  `branches_archived`/`tapes_archived` tables (both stay queryable forever)
  inside one `BEGIN IMMEDIATE`/`_write_lock` transaction — branches first, so
  the live `branches -> tapes` foreign key is never violated. `dry_run=True`
  computes the candidate set with zero writes. The new `tracefork prune
  [--older-than-days N] [--run-id id ...] [--dry-run]` CLI command always
  exits 0. `save_tape`/`save_branch`/`load_tape`/`load_branch`/`list_runs`/
  `list_branches` signatures are completely unchanged; `StorageBackend` is
  deliberately not extended with `prune` (kept `TapeStore`-only for now).

### Fixed

- **CAS-guarded `save_tape`** (`store.py`) — reusing a `run_id` no longer
  silently clobbers the previously-stored tape. `save_tape` now installs or
  verifies-same-content (git's object-store model): identical content
  (compared via `Tape.digest()`, never raw envelope bytes) is an idempotent
  no-op; genuinely different content raises the new `TapeConflictError`
  instead of overwriting, unless the new explicit `overwrite=True` is passed.
  The check-then-write stays inside the existing `BEGIN IMMEDIATE` transaction
  and write lock, so it's TOCTOU-free with no second lock added.

## [0.2.1] - 2026-07-03

### Added

- **Google ADK adapter** (`adapters/adk.py`, opt-in `adk` extra) — `bind()` routes
  an ADK `LlmAgent`/`Gemini`'s underlying `google-genai` client through the
  existing `TraceforkTransport` via a short candidate-path search for the
  `google.genai` `BaseApiClient`'s private `_httpx_client`/`_async_httpx_client`
  attributes (the target itself, a `genai.Client`, an ADK `Gemini` model wrapper,
  or an `LlmAgent` whose `.model` already holds one) — the same Gemini
  `generateContent` wire format `providers/gemini.py` already parses. Step
  visibility is a real `BasePlugin` (`make_plugin()`) over ADK's documented
  agent/model/tool before/after callback boundaries, registered once on the
  `Runner` rather than threaded through every agent — observer-only, never a
  second capture path.
- **Curated `all` extra** (`pip install 'tracefork[all]'`) — a self-referential
  convenience bundle over `providers` + `bedrock` + `mcp` + `observability` (the
  internally-consistent, stable-wire family). Deliberately excludes the five
  independently-capped, fast-moving framework stacks (`frameworks`,
  `openai-agents`, `crewai`, `autogen`, `adk`) so one future cap collision on any
  single framework can't `ResolutionImpossible` the whole `all` install.

### Changed

- **Relaxed the `frameworks` extra's version caps to floors** — dropped the
  speculative `<2` upper bounds on `langchain-core`/`langchain-openai`/
  `langchain-anthropic`/`langgraph`. LangChain 1.0 GA commits to no breaking
  changes until 2.0, so a `<2` cap was speculative and, more importantly, useless
  against the real observed failure mode: intra-1.x patch regressions ship
  *inside* the allowed range regardless of the cap. The remaining internal-
  coupling caps (`openai-agents`, `crewai`, `autogen`, `adk` — each an adapter
  injecting into private/undocumented framework internals) are kept, each now
  with an inline "hint, not a guard" rationale and a revisit TODO.
- **Adapter import guards now chain the root cause** (`raise ImportError(HINT)
  from exc`) in `adapters/{langchain,openai_agents,crewai,autogen,adk}.py`'s
  `require_*()` functions, so an installed-but-broken dependency's real
  `ImportError` is preserved instead of being masked as "not installed".

## [0.2.0] - 2026-07-02

Generic, multi-provider, production-hardening release. The bit-exact, $0,
hash-verified replay substrate from 0.1.0 is the contract and is unchanged —
everything below is additive around it: all engine internals stay byte-stable,
`digest()` is format-stable, and every existing tape still loads and replays.

### Added

- **Provider abstraction seam** (`providers/`): a `ProviderAdapter` protocol + registry
  and a `gen_ai.*`-style `NormalizedResponse` view. Anthropic is now a *registered*
  adapter, not a hardcoded assumption; raw request/response **bytes** stay the immutable
  replay + hash contract. (#7)
- **More provider backends** — **OpenAI** and **Google Gemini** (httpx-based, captured
  through the existing transport seam, with provider-generic fault injection), and **AWS
  Bedrock** (a separate botocore `before-send` seam, a stdlib `vnd.amazon.eventstream`
  frame codec, and SigV4-canonical request matching so a fresh signature/timestamp is not
  a false divergence). (#12, #21)
- **Pluggable canonicalization / divergence matcher** (`matcher.py`): identity +
  canonicalizing matchers so volatile material (Gemini `?key=`, Bedrock `x-amz-date`,
  rotating auth) hashes equal while a real request-body change still diverges. (#9)
- **Plugin & extension architecture** (`plugins.py`): entry-point registries for
  providers, matchers, storage backends, transports, oracles, and adapters — each gated by
  the same explicit allowlist. (#11)
- **MCP + native tool-call record/replay** (`tools.py`, `mcp_client.py`): a JSON-RPC tee so
  tool exchanges share the tape with LLM exchanges and replay bit-exact. (#14)
- **Framework adapters** (`adapters/`, opt-in extras): a minimal
  `bind()`/`on_step()`/`teardown()` protocol + a `StepDAG` normalizer, with
  **LangChain/LangGraph** (including a tape-backed LangGraph checkpointer) plus **OpenAI
  Agents SDK**, **CrewAI**, and **AutoGen** adapters — each binding at the framework's real
  model-call chokepoint (never a second capture path) and each guarded so `import
  tracefork` and the full offline suite run with none installed. (#16, #23)
- **OTel GenAI / OpenInference interop** (`interop.py`, `tracefork export`/`ingest`): export
  a tape (+ blame report) as OTel GenAI spans or an OpenInference dataset, and ingest either
  back into a tape's step structure for **blame-by-re-execution** (explicitly *not* $0
  bit-exact replay — an ingested tape diverges on `replay`/`fork` by design). Plain JSON, no
  `opentelemetry-sdk` required. Plus an **opt-in `observability` extra** (structlog + OTel
  self-instrumentation of record/replay/fork/blame), off by default. (#15)
- **Deterministic asyncio concurrency replay** (`transport.py`): records the completion
  order of concurrent fan-out (`gather`/`TaskGroup`) and re-imposes it on replay, so
  concurrent agents replay bit-exact — not just single-call-at-a-time ones. (#17)
- **Causal blame depth**: coalition / temporal-Shapley attribution with necessity +
  sufficiency, and a long-tape competing-fault benchmark (`tracefork bench`) with Wilson
  CIs. The one temporal-order Shapley limitation is documented, not hidden. (#18, #24)
- **Oracle rigor** (`judge.py`, opt-in): an LLM-judge oracle with position-swap averaging +
  a self-judge guard, gold-set calibration (FPR/FNR/Cohen's kappa), and Rogan-Gladen
  flip-rate debiasing — all offline-testable via an injected judge function. (#19)
- **Divergence diagnostics & debug UX** (`divergence.py`, web report): a structured
  divergence diff, a report rewind panel, blame trust flags, and real-vs-tolerated
  divergence messaging. (#20)
- **Localhost record/replay proxy** (`tracefork proxy`): a base-URL MITM-style proxy (binds
  127.0.0.1) for non-Python / non-httpx clients (curl, Node, Go), reusing the tape +
  matcher. Record/replay only — outside the in-process `NondetSource` determinism
  boundary. (#22)
- **Hardening seams**: an opt-in record-time `BoundaryGuard` (thread/subprocess spawn,
  direct `random`/`time` bypasses), `random_float()` virtualization through `NondetSource`,
  a `replay --check` fixture-corpus regression gate, and opt-in secret/PII redaction on
  record. (#10, #13)
- **CLI**: new `bench`, `export`, `ingest`, and `proxy` commands (joining
  `replay`/`verify`/`fork`/`report`/`serve`/`blame`/`validate`), plus a full end-to-end
  integration suite and a single `scripts/e2e.sh` receipt. (#25)

### Changed

- **Versioned tape envelope** (`tape.py`): a `TAPE_MAGIC` + uint16 version header over a
  zstd + content-addressed binary container (no base64, no pickle), with a read-time
  upcaster chain (v1→v4). The version header is **not** part of `digest()`, so the hash
  chain is byte-stable across format versions and legacy tapes still load and replay. (#6)
- **Blame statistics**: three-valued trials (flip / no-flip / undefined) and pluggable CI
  methods. (#8)
- **SQLite persistence hardened**: WAL, `synchronous=NORMAL`, `busy_timeout`,
  `foreign_keys=ON`, and writers take `BEGIN IMMEDIATE`. (#6)

### Fixed

- Wilson CI boundaries snap to analytically-exact 0/1 to avoid ~1e-17 platform float-dust
  that differed across CI runners. (#8)

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

[Unreleased]: https://github.com/pratik916/tracefork/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/pratik916/tracefork/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/pratik916/tracefork/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/pratik916/tracefork/releases/tag/v0.1.0
