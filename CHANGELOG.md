# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`tracefork coalition-fork`: coalition-forking as a public what-if CLI DSL**
  (`cli.py`, new command; zero new engine code) — promotes the already-tested
  `CoalitionSpec`/`ForkEngine.fork_coalition` to a documented CLI surface: a
  repeatable `--intervene step:response_file` pins one intervention locus per
  flag, jointly forced in a single pass (the coalition/Shapley `do(S)`
  primitive's pinned-locus + same-policy-resampling guarantee, vs. a naive
  "fork anywhere and diff"). Persists through the existing
  `TapeStore.save_branch` unchanged — no schema change — by JSON-encoding the
  coalition's step list and description into the existing free-text
  `mutation_desc` column. Malformed `--intervene` syntax, an out-of-range
  step, or duplicate step indices (`CoalitionSpec.__post_init__`'s existing
  `ValueError`) all surface as a clean `typer.BadParameter`, never a raw
  traceback. The existing single-step `fork` command is untouched.

- **Tournament API: rank N candidate continuations at one fixed step**
  (`tournament.py`, new) — `TournamentEngine.run()` forks each `Variant` `k`
  times at the same `step_index` (best-of-N argmax, but statistically
  validated), reusing `ForkEngine.fork` unchanged, `BudgetGovernor.estimate`/
  `BudgetExceededError` for a pre-spend budget gate, and `blame.py`'s Wilson
  CI (`proportion_ci`)/`binom_sf_ge`/`benjamini_hochberg` for scoring — never
  a parallel cost model or CI reimplementation. A winner is declared only
  when the top variant is significantly better than EVERY runner-up (each
  tested one-sided against the top's observed rate, jointly BH-corrected at
  `fdr_q`), so two indistinguishable variants never produce a spurious
  winner. Forking the tape's LAST exchange has an empty tail — the common
  "best-of-N final answers" case is genuinely $0. New `tracefork tournament`
  CLI command prints a ranked table (variant, score, Wilson CI, q-value,
  winner flag) and writes `tournament_<run_id>.json`.

- **Monte Carlo coverage-calibration harness for blame's confidence intervals**
  (`ci_calibration.py`, new) — closes the "no experiment proves a nominal 95%
  CI actually achieves ~95% coverage" gap using the canonical Brown-Cai-
  DasGupta (2001) methodology: fix a KNOWN ground-truth flip probability,
  simulate thousands of Bernoulli replicates via `random.Random(seed)`,
  compute the candidate interval for each via `blame.py`'s REAL
  `proportion_ci`/`wilson_ci`/`CIMethod` (never a parallel reimplementation),
  and measure empirical coverage against a documented tolerance band (4x the
  Monte Carlo error of the coverage estimate itself, `monte_carlo_error`).
  `simulate_coverage()` runs one `(method, true_p, n_trials)` cell;
  `run_calibration()` sweeps the full grid into a `CalibrationReport`
  mirroring `bench.py`'s dataclass-report shape (`.regressions()`,
  `.all_within_tolerance()`). Confirms the qualitative ordering the
  docstrings already claim — Clopper-Pearson is conservative
  (coverage >= nominal) by construction, Wilson may dip slightly below
  nominal at small n — and stays sane at the `true_p in {0, 1}` boundary via
  `wilson_ci`'s documented boundary-snapping. Pure math, deterministic given
  a seed, offline/$0, no fork/agent/API calls; standalone diagnostic, not
  wired into any gate yet.

- **Lossless tape+branch bundle export/import** (`bundle.py`, new; `store.py`,
  `cli.py`) — `tracefork bundle-export <run_id> [--output bundle.db]
  [--store store.db]` / `tracefork bundle-import <bundle.db> [--store
  store.db]`, a portable, scp-able artifact analogous to `git bundle`: a
  bundle is literally a second, smaller `store.db` (same DDL, same
  `Tape.to_bytes()` envelope), not a bespoke archive format. `export_bundle`
  copies a run's + its direct branches' `tapes`/`branches` BLOB columns
  byte-for-byte via two new `TapeStore`-only helper pairs
  (`raw_tape_row`/`raw_branch_rows` read, `install_raw_tape_row`/
  `install_raw_branch_row` write) — zero `Tape.from_bytes`/`to_bytes`
  decode-reencode round trip, so the bundle's stored bytes are identical to
  the source store's, not merely digest-equal. `import_bundle` goes through
  the EXISTING CAS-guarded `save_tape`/`save_branch` write path — never a raw
  `INSERT` — so a collision on import (an existing `run_id`/`branch_id` with
  genuinely different content) raises `TapeConflictError` instead of
  silently clobbering; reusing the same ids with byte-identical content is
  an idempotent no-op. `save_branch` gains an optional `branch_id=` parameter
  (defaults to `None`, generating a fresh uuid exactly as before) so import
  can preserve a branch's id across stores, guarded by the same
  install-or-verify-same-content check as `save_tape`. Offline/$0, pure local
  file I/O.

- **Crash-safe incremental (checkpointed) recording** (`checkpoint.py`, new;
  `transport.py`, `recorder.py`) — a crash before `tape.save()`/`to_bytes()`
  previously lost the entire in-memory recording. `CheckpointWriter` durably
  commits each recorded exchange to a local SQLite file (its own `BEGIN
  IMMEDIATE`/`COMMIT`, reusing `tape.open_sqlite`'s hardened connection
  factory) the instant it happens; `recover_checkpoint(path)` returns
  `(tape, was_finalized)` — an honest linear prefix with `was_finalized=False`
  if recovered mid-crash, or the complete tape with `was_finalized=True` after
  a clean `finalize()`. Scoped to exchanges only, not nondeterminism draws — a
  narrower-than-ideal but honest boundary, documented as such rather than
  silently under-delivering; a cleanly finalized checkpoint still has the full
  draw log via `Tape.save`. Wired in via an opt-in, keyword-only `on_exchange`
  hook on `TraceforkTransport`/`AsyncTraceforkTransport` (fires once,
  immediately after `tape.append_exchange`, in the record branch only —
  never replay, never the async ordered-release/chaos machinery) and an
  opt-in `checkpoint_path=` on `Recorder`/`AsyncRecorder` (constructs the
  writer, passes its `append_exchange` as the hook, calls `finalize(tape)` on
  a clean `__exit__`/`__aexit__` only). Both default to `None`/unset — every
  existing zero-kwarg call site is byte-identical to before.

- **Store-level fsck** (`fsck.py`, new; `store.py`, `cli.py`) — `tracefork
  verify --store <db>` runs a read-only, git-fsck-style structural check over
  a `TapeStore` database, distinct from replay-fidelity verification: every
  tape must decode via the existing public `load_tape`, every branch under a
  still-live parent must decode via `load_branch` (a decode error is a
  per-row failure, never a crash — one bad row doesn't abort the scan of the
  rest of the store), and every branch's `parent_run_id` must resolve to a
  live tape — an orphaned-parent failure reported even when `load_branch`
  alone would still succeed (e.g. after a parent tape row was force-deleted
  with `foreign_keys=OFF`). `store.py` gains two small `TapeStore`-only read
  helpers: `stored_digest` (gated on `PRAGMA table_info(tapes)`, so a future
  `digest` column is an opportunistic stronger check, never a hard
  dependency) and `all_branch_parents` (every `(branch_id, parent_run_id)`
  pair regardless of parent liveness, since `list_branches` alone can't
  surface an orphan). `--store` is mutually exclusive with the existing
  `--corpus` option on `verify`. Read-only: never mutates the store, unlike
  `prune()`.

- **Boundary/provenance/redaction trust badge in report + CLI** (`report.py`,
  `cli.py`, `web/report.html`) — `Tape.boundary` and `Tape.content_redacted`
  were persisted but never surfaced, so a forensic-only or content-redacted
  tape looked identical to a verified one. `_tape_to_data()` now embeds
  `boundary` alongside the existing `content_redacted`; `_print_receipt`
  (replay/verify) and the `report` command's terminal echo now print both as
  two trust lines via a shared `_print_trust_lines` helper; `web/report.html`
  gains a header badge (`renderProvenanceBadges`, SLSA-style leveled trust,
  not a single verified/unverified boolean): `BOUNDARY_V1` renders green
  ("verified boundary", reusing `.tag-pass`), `OTEL_INGESTED_BOUNDARY`/
  `PROXY_BOUNDARY` render a new yellow `.tag-warn` ("forensic-only boundary"),
  and `content_redacted=true` renders a separate orange `.tag-redacted`
  warning. Both fields stay forensic-only — neither is fed into
  `Tape.digest()` — the badge is a trust warning, not a pass/fail input.
  Additive only: `_print_receipt` still has exactly two call sites (replay,
  verify), `web/report.html` stays a single dependency-free file.

- **Determinism-coverage report** (`coverage.py`, `cli.py`) — `tracefork
  coverage <tape> [--agent-source FILE]` turns "is this replay actually
  complete?" into a checkable artifact: `tape_draw_coverage` tallies
  `nondet.py`'s draw kinds and whether concurrency (`async_batches`) /
  `BoundaryGuard` (`provenance["boundary_guard"]`) were recorded/active for
  an already-loaded `Tape`; `scan_source_for_nondeterminism_calls` is a
  read-only, best-effort `ast.parse`-only lint (never imports or executes
  the given source) over agent source text for call sites shaped like what
  `boundary_guard.py` itself patches, vs. that module's own documented,
  permanent exclusions (`datetime.datetime.now()`/`time.time()`, always
  informational, never a violation). New capability, no existing consumer
  to break; `Tape.digest()`/`to_bytes()`/`from_bytes()` untouched.

- **Recording provenance/witness block on `Tape`** (`tape.py`, `recorder.py`,
  `replay.py`) — `Tape.provenance` (matcher fingerprint, boundary-guard state,
  nondet mode) is a small `dict[str, str]` `Recorder`/`AsyncRecorder` populate
  from values already in scope; tape format version bumps to 5 with a
  `_decode_v5_binary`/`_upcast_v4_to_v5` pair mirroring the existing v3->v4
  pattern, so a pre-v5 tape upcasts to `provenance={}` with an unchanged
  digest. `ReplayVerifier.verify()` optionally compares the tape's recorded
  `matcher_name` against the matcher actually used at replay and raises a
  distinct `ProvenanceMismatchError` instead of a generic byte-diff
  divergence — opt-in, firing only when `provenance` is non-empty. `digest()`
  needs no change: like `boundary`/`agent_name`/`async_batches`, `provenance`
  is never hashed into it.

- **Persistent causal graph store: `causal_edges` table** (`store.py`,
  `cli.py`) — `TapeStore.save_blame_report()`/`save_shapley_report()` persist
  every `FlipRateResult`/`ShapleyResult` (flip_rate, Wilson CI, BH-FDR
  `q_value`/`responsible`, or Shapley value + necessity/sufficiency) instead
  of letting a blame run be computed and discarded; upsert-by-replace keyed on
  `edge_id=f"{run_id}:{step_index}:{method}"` means a re-blame replaces the
  prior row set rather than accumulating stale rows. `causal_edges_for_run()`
  reads them back; `cited_by(run_id, step)` derives citing branch ids
  directly from the existing `branches` table (no new citation concept);
  `causal_closure(run_id)` BFS-walks `branches.parent_run_id` chains — where a
  branch was itself promoted to its own tape via `save_tape(delta_tape,
  run_id=branch_id)` — unioning each generation's `responsible=1` edges into
  one causal graph strictly stronger than a bare caused_by DAG. `causal_edges`
  has no `FOREIGN KEY` to `tapes` (unlike `branches`, `prune()` need not know
  about it). `StorageBackend` gains the same 5 signatures; `tracefork blame`
  calls `save_blame_report()` additively after its existing
  `blame_<run_id>.json` write. Store-level metadata only — never fed into
  `Tape.digest()`.

- **Confined fork/blame execution via `boundary_guard=` on `ForkEngine`/
  `BlameEngine`** (`fork.py`, `blame.py`) — `ForkEngine.fork()` and
  `fork_coalition()` gain an opt-in `boundary_guard: bool = False` kwarg that
  wraps *only* the re-executed `agent_fn(client)` call in a fresh
  `BoundaryGuard` (see `boundary_guard.py`), confining that one trial's own
  tool-call/thread/random/subprocess surface without touching the
  prefix-replay/mutation-injection transport logic. `BlameEngine.rank()` and
  `shapley_rank()` thread the same flag down through `_run_trial()`/
  `_run_coalition_trials()` — including `shapley_rank()`'s internal
  sufficiency pass (a `rank()` call sharing the same `agent_fn`) — into every
  `ForkEngine` call they issue. A trial that trips the guard is caught by the
  existing broad exception handling and counted `UNDEFINED`, never a silent
  `NO_FLIP`. Default `False` leaves every existing call byte-identical;
  `cli.py`'s `fork` command and `faults.py`/`validate.py`/
  `competing_faults.py`/`bench.py` pass no kwarg and are unaffected.

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

- **Untrustworthy steps excluded from blame's BH correction, and
  `blame_report_from_json` recomputes rather than trusts q_value/responsible**
  (`blame.py`, `interop.py`) — `BlameEngine.rank()`'s Benjamini-Hochberg FDR
  step now runs only over the p-values of steps with `trustworthy=True`; a
  step with too few valid trials (a spuriously low raw p-value from a
  handful of all-flip trials, say) never occupies a BH correction slot and
  always keeps `q_value=1.0`/`responsible=False`, regardless of how
  significant its raw p-value looks. Excluding it also tightens neighboring
  trustworthy steps' q-values relative to the old whole-list BH (proving the
  candidate count `m` actually changed, not a post-hoc overwrite); the
  common all-trustworthy case is unaffected (byte-identical output).
  `interop.py`'s `blame_report_from_json` now recomputes
  `q_value`/`responsible`/`responsible_set` from the decoded `p_value`s via
  the same `benjamini_hochberg` call, rather than trusting those fields from
  the JSON payload — closing the one round-trip boundary where they could be
  forged independently of `p_value`.

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
