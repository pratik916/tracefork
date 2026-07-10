# CLAUDE.md

This file guides Claude Code when working in the `tracefork` repository.

## What this is

`tracefork` is a time-travel debugger for AI agents: record an agent run to a
content-addressed **tape**, replay it **bit-exact for $0** (hash-verified), fork any
step, and measure causal blame with confidence intervals — the instrument itself
validated against runs with injected, known root-cause faults.

**Current state: v1 built.** All five product pillars work offline and are tested
(672 tests, $0): streaming-capable record/replay with drift detection, the three-phase
fork engine, the causal blame engine with Wilson CIs and a budget governor, the
single-file web report/UI, and the fault-injection self-validation suite (5 fault
classes at 1.00 top-1 precision, plus a longer competing-fault fixture that measures
whether the coalition/temporal-Shapley engine discriminates *several* simultaneously
planted causes — `tracefork bench`; see README → Validation scope for exactly what each
number does and doesn't claim). `src/tracefork_spike/` keeps the original Spike 0 that
de-risked the load-bearing assumption (bit-exact, no-key replay within a declared
determinism boundary). Design/feature list: `../ideas/2026-06-11-tracefork-features.md`;
spike finding: `SPIKE0.md`.

## Commands

Python is **3.12 via uv**. The tests, the spike, `validate`, the demo, and
record/replay/fork are offline and $0 — **no `ANTHROPIC_API_KEY`, no network**. Only
`blame` against a *real* run hits the live API (budget-capped). Always prefix `uv run`.

```bash
uv sync --extra dev                  # install (anthropic, zstandard, typer, fastapi, uvicorn + pytest)
uv run pytest -q                     # full offline suite (672 tests)
uv run pytest tests/test_faults.py::test_validation_runner_fingers_fault_step -q   # one test
uv run tracefork validate            # self-validation: blame vs injected, known faults
uv run tracefork validate --check    # regression-gate vs experiments/validation_report_committed.json
uv run tracefork bench                # long-tape competing-fault discrimination benchmark
uv run python examples/demo_report.py   # write examples/demo_report.html (the README screenshot)
uv run python -m tracefork_spike     # the original Spike 0 bit-exact replay receipt
uv run tracefork --help              # replay, verify, fork, report, serve, blame, tournament, validate,
                                      # bench, proxy, export, ingest, prune, coverage, bundle-export,
                                      # bundle-import
uv run tracefork replay --check experiments/replay_fixtures   # replay-as-regression gate
bash scripts/e2e.sh                  # single-receipt gate: sync, lint, type-check, tests+coverage,
                                      # validate --check, replay --check, bench, build+twine, one PASS banner
```

## Architecture (the parts that span files)

The spine is a **record/replay seam at the Anthropic SDK's httpx boundary**, plus a
**nondeterminism-virtualization seam** the agent reads time/ids through. Bit-exactness
is the contract between them.

The product lives in `src/tracefork/`:

- `nondet.py` — `NondetSource` is the *only* way the agent gets time/ids/random draws
  (`now_iso`/`new_uuid_hex`/`random_float`). `RecordingNondet` draws real values and logs
  them; `ReplayNondet` serves them back in order; `DriftingNondet` is the negative control
  (fresh values → forced divergence). `random_float()` logs the exact `float.hex()` string
  (lossless round-trip, no float-formatting dust) and, like `now_iso()`, is additive/opt-in
  — an agent must be handed the active `NondetSource` explicitly; nothing patches `random`
  globally the way `Recorder` patches `uuid.uuid4`. `find_divergence()` unwraps a
  `DivergenceError` from the `APIConnectionError` the SDK wraps transport exceptions in —
  **keep this; without it a real divergence looks like a network blip.**
- `boundary_guard.py` — `BoundaryGuard`, an **opt-in** (default off) record-mode guard:
  hard-errors on `threading.Thread.start`/`subprocess.Popen` (crossing the single-process
  boundary) and direct `random.random`/`time.monotonic`/`time.sleep` (bypassing
  `NondetSource`) instead of letting the tape fail replay later, mysteriously. Deliberately
  does **not** guard `datetime.datetime.now()` (same immutable-C-type reason `recorder.py`
  doesn't patch it) or `time.time()` (httpx's cookie-jar machinery calls it on every
  response — guarding it would false-positive on every exchange). Pre-warms the Anthropic
  SDK's `platform_headers()` `lru_cache` on `__enter__` so its one internal
  `subprocess.Popen` call (uncached platform detection) doesn't trip the guard on the first
  real API call. Wired into `Recorder`/`AsyncRecorder` via `boundary_guard=` (tri-state:
  explicit wins over `TraceforkConfig.boundary_guard`, both default `False`).
- `transport.py` — `TraceforkTransport` (sync) + `AsyncTraceforkTransport` (async) are the
  capture seam, streaming-SSE capable (buffer via `.read()`/`.aread()`). Record mode tees
  request+response bytes into the tape; replay mode serves recorded bytes and
  sha256-asserts each request body matches (the divergence detector). A replay transport
  has **no inner transport**, so any unrecorded request is a hard error. **Async
  concurrency:** the async transport records the completion order of concurrent fan-out
  (`asyncio.gather`/`TaskGroup`) — appending exchanges at completion and logging each
  fully-overlapping batch to `tape.async_batches` — and on replay **correlates each request
  to its recorded exchange by fingerprint (not positional arrival) and releases responses in
  the recorded completion order** via an ordered gate, so a fan-out agent replays bit-exact.
  A strictly-sequential async run never waits at the gate → byte-identical to before; the
  sync transport is untouched (stays positional). `chaos_release_order(tape, seed)` derives a
  seeded, physically-possible reordering of a recorded schedule (chaos-mode replay) for
  race/ordering-bug analysis; **do NOT** add awaits/ordering to the sequential or sync path.
  A keyword-only `on_exchange` hook (opt-in, default `None`) fires once, immediately
  after `tape.append_exchange`, in the record branch only — the crash-safe-checkpoint
  seam (see `checkpoint.py`); never fired on replay or the async ordered-release/chaos path.
- `tape.py` — `Tape` is content-addressed (sha256 blobs) + an ordered event log,
  persistable to SQLite, with a hash-chain `digest()` fingerprint. `to_bytes`/`from_bytes`
  emit a **versioned envelope** (`TAPE_MAGIC` + uint16 version, then a zstd + content-
  addressed binary container — no base64); `from_bytes` dispatches on the version through a
  read-time upcaster chain and still loads legacy header-less JSON blobs as v1. The version
  header is envelope metadata, **not** part of `digest()`, so the hash chain is byte-stable
  across format versions. **v4** adds the `async_batches` concurrency-batch log (recorded
  completion order of fully-overlapping fan-out); like `boundary`/`agent_name` it is
  persisted but **never** fed into `digest()` (the completion order is already fingerprinted
  by the `exchanges` list ordering), so every existing and every sequential/sync tape's
  digest is byte-identical and v1/v2/v3 tapes upcast to an empty batch log. **v5** adds `provenance`
  (matcher_name/boundary_guard/nondet_mode, populated by
  `Recorder`/`AsyncRecorder`); like `async_batches` it is metadata only,
  **never** fed into `digest()`, and v1-v4 tapes upcast to `provenance={}`
  — see `replay.py`'s opt-in `ProvenanceMismatchError` check. It's a JSON
  header + zstd blobs, **not pickle** — no
  arbitrary-code-execution risk. `open_sqlite()` is the one hardened connection factory
  (WAL, `synchronous=NORMAL`, `busy_timeout`, `foreign_keys=ON`); writers take `BEGIN
  IMMEDIATE`. `store.py` reuses it and serializes its write fan-out with a lock.
- `recorder.py` — `Recorder` context manager wraps a real `anthropic.Anthropic` at its
  `_client._transport` seam (via `client.copy(http_client=...)`, so base_url / auth_token /
  default headers are preserved). Patches `uuid.uuid4` globally; **does not** patch
  `datetime.datetime` (immutable C type in 3.12+, and a subclass breaks the SDK's pydantic
  schema builder) — agents needing deterministic clocks/random read `NondetSource` directly.
  Optionally wraps the recording window in a `BoundaryGuard` (see `boundary_guard.py`).
  An opt-in `checkpoint_path=` wires a `CheckpointWriter` (see `checkpoint.py`) into
  the transport's `on_exchange` hook so each exchange is durably committed as it
  happens, and calls `finalize(tape)` on a clean `__exit__`/`__aexit__` only —
  default `None` is byte-identical to before this flag existed.
- `checkpoint.py` — `CheckpointWriter`/`recover_checkpoint`: opt-in crash-safe
  incremental recording. Each recorded exchange is committed to a local SQLite
  file with its own `BEGIN IMMEDIATE`/`COMMIT` (reusing `tape.open_sqlite`) the
  instant it happens, so a crash before `tape.save()` loses at most the exchange
  in flight, never the already-recorded prefix. `recover_checkpoint(path)` returns
  `(tape, was_finalized)`: a mid-crash recovery is an honest linear prefix with
  `was_finalized=False`; a clean `finalize(tape)` writes the complete tape (draws
  included) via `Tape.save` and flips `was_finalized=True`. Scoped to exchanges
  only, not draws — a narrower-than-ideal but honest, documented boundary. Wired
  into `transport.py`'s record branch via a keyword-only `on_exchange` hook (never
  fired on replay or the async ordered-release/chaos path) and into
  `Recorder`/`AsyncRecorder` via `checkpoint_path=`.
- `fork.py` — `ForkTransport` runs three phases: prefix-replay ($0, request asserted to
  match the parent), mutation-injection (same request, swapped response), tail-record (the
  counterfactual continuation). `Branch` carries `prefix_replayed`/`tail_recorded` counts.
  `ForkEngine.fork()` re-runs the **same** agent that produced the tape. Every `Branch` also
  carries a content-addressed `branch_digest` (`compute_branch_digest`): `sha256(parent_tape
  .digest() + delta_tape.digest() + repr(intervened_steps))`, computed in `fork()`/
  `fork_coalition()` right before constructing the returned `Branch` — Merkle-DAG identity
  (a node's hash folds in its children's), so identical (parent, delta, intervened_steps)
  always produce the same digest and `store.py` can key branches by content, resolve
  fork-of-fork chains, and answer inverse-citation queries as reachability walks. Branch/
  store-level metadata only — `Tape.digest()` itself is completely untouched.
- `diff.py` — generalized point-to-point / fork-branch diff, purely a
  sequence-of-steps orchestration layer on top of `divergence.py`'s existing
  single-step structural-diff primitive (`diff_json`/`diff_request_bytes`/
  `MISSING`); adds no new diff logic of its own. `branch_diff(parent_tape,
  branch, from_step=None)` walks a branch's `delta_tape` against its parent
  from the divergence step onward; `branch` is either a live `fork.Branch`
  (its `.delta_tape`/`.divergence_step` read directly) or a plain `Tape` (a
  store-reloaded `delta_tape`, with `divergence_step=` passed explicitly) —
  decoupled from `TapeStore` itself either way. `tape_diff(tape_a, tape_b,
  step)` compares two independent tapes at one step index, no parent/child
  relationship assumed. A step present on only one side (a `delta_tape`
  shorter than the parent's tail) reports via `MISSING`, never a crash. CLI:
  `tracefork diff <parent_run_id> <branch_id>` (branch mode) or
  `<run_id_a> <run_id_b> --step N` (tape mode).
- `store.py` — `TapeStore`, SQLite persistence for tapes + the branch DAG.
  `save_tape` is install-or-verify-same-content (git's object-store model):
  reusing a `run_id` with byte-identical content (compared via `Tape.digest()`,
  never raw bytes) is an idempotent no-op; genuinely different content raises
  `TapeConflictError` instead of silently clobbering the prior tape, unless the
  caller passes `overwrite=True`. The check-then-write stays inside the
  existing `BEGIN IMMEDIATE`/`_write_lock` transaction — no second lock.
  `save_branch` gains an optional `branch_id=` parameter (default `None`,
  generating a fresh uuid exactly as every existing caller already gets):
  passing it applies the same install-or-verify-same-content CAS guard as
  `save_tape` (idempotent no-op on identical content, `TapeConflictError` on
  a genuine collision) so a caller that needs to preserve a branch's id
  across stores (`bundle.py`'s `import_bundle`) gets the same safety
  `save_tape` already has, instead of the raw `sqlite3.IntegrityError` a bare
  collision would otherwise raise. `prune()` is a
  soft-archive-only retention pass (git gc / borg prune's mark-and-sweep-with-
  soft-archive discipline, never a hard delete): a tape matching
  `older_than_iso` (lexical `created_at` cutoff) and/or an explicit `run_ids`
  allowlist has its branches copied into `branches_archived` and deleted from
  the live `branches` table FIRST, then its row copied into `tapes_archived`
  and deleted from the live `tapes` table — the order the live `branches` ->
  `tapes` foreign key requires — all inside one `BEGIN IMMEDIATE`/
  `_write_lock` transaction. `dry_run=True` computes the candidate set with
  zero writes. Archived rows are never deleted by anything; reclaiming that
  space is a distinct, out-of-scope, higher-risk step. `StorageBackend` is
  deliberately NOT extended with `prune` (kept `TapeStore`-only for now); the
  `tracefork prune` CLI command always exits 0 (a maintenance op, not a gate).
  `save_blame_report`/`save_shapley_report` persist `blame.py`'s
  `FlipRateResult`/`ShapleyResult`s into a `causal_edges` table (upsert-by-
  replace on `edge_id=f"{run_id}:{step_index}:{method}"`, so a re-blame
  replaces rather than duplicates) instead of letting every blame run be
  computed and discarded; `causal_edges_for_run`/`cited_by`/`causal_closure`
  read them back — `cited_by` derives citing branch ids straight from the
  existing `branches` table, and `causal_closure` BFS-walks
  `branches.parent_run_id` chains (where a branch was promoted to its own
  tape via `save_tape(delta_tape, run_id=branch_id)`) unioning each
  generation's `responsible=1` edges. `StorageBackend` gains the same 5
  signatures; `causal_edges` has no FK to `tapes` (unlike `branches`,
  `prune()` doesn't need to know about it) and is never fed into
  `Tape.digest()`. `cli.py`'s `blame` command calls `save_blame_report`
  additively after its existing JSON write. `stored_digest`/`all_branch_parents`
  are two small `TapeStore`-only read helpers (same "not on `StorageBackend`
  yet" precedent as `prune`) for `fsck.py`: `stored_digest` gates on `PRAGMA
  table_info(tapes)` so a future `digest` column is an opportunistic stronger
  check, never a hard dependency; `all_branch_parents` returns every
  `(branch_id, parent_run_id)` pair regardless of parent liveness, since
  `list_branches(parent_run_id)` alone can never surface a branch whose parent
  tape row was force-deleted directly (`foreign_keys=OFF`). `raw_tape_row`/
  `raw_branch_rows` (read) and `install_raw_tape_row`/`install_raw_branch_row`
  (write) are four more small `TapeStore`-only helpers, for `bundle.py`'s
  byte-for-byte export: the read pair returns a stored row's raw BLOB column
  exactly as stored (zero `Tape.from_bytes`/`to_bytes` decode-reencode); the
  write pair is a plain `INSERT OR REPLACE` with no CAS guard, deliberately
  not general-purpose (unlike `save_tape`/`save_branch`) — safe only because
  `export_bundle` always points it at a fresh bundle file, never a live store.
  The `branches` table also carries `fork.py`'s content-addressed
  `branch_digest` (a column + index; Branch/store-level metadata only, never
  fed into `Tape.digest()`), migrated onto a pre-existing `store.db` via a
  `PRAGMA table_info`-guarded `ALTER TABLE` in `TapeStore.__init__` (a fresh
  database gets the column straight from `CREATE TABLE`; an old one is
  altered in place, no row lost). `save_branch`/`load_branch` gain a
  `branch_digest=` parameter/return key defaulting to `''`, so every
  existing caller is unaffected; `find_branch_by_digest` resolves the branch
  with a given digest, and `branches_forked_from` is the inverse-citation
  query — which branches used a given digest's branch as their own parent,
  once that branch's `delta_tape` is itself promoted to a tape via
  `save_tape(delta_tape, run_id=branch_id)` (the same promotion convention
  `causal_closure` already relies on) — enabling fork-of-fork chains as a
  plain reachability walk.
- `bundle.py` — lossless tape+branch trajectory export/import: a bundle is
  literally a second, smaller `store.db` (same DDL, same
  `Tape.to_bytes()` envelope) — `git bundle`'s model, a scoped-down valid
  store any `TapeStore` can open directly rather than a bespoke archive
  format. `export_bundle(store, run_id, output_path)` copies `run_id` and its
  DIRECT branches' BLOB columns byte-for-byte via `store.py`'s four raw-row
  helpers above — never a `Tape` decode/re-encode round trip, so the bundle's
  bytes are identical to the source store's, not merely digest-equal.
  `import_bundle(target, bundle_path)` goes the other way through the
  EXISTING CAS-guarded `save_tape`/`save_branch` write path (never a raw
  `INSERT`), preserving each branch's id via `save_branch`'s new `branch_id=`
  parameter — a genuine content collision on an existing `run_id`/`branch_id`
  raises `TapeConflictError` instead of silently clobbering; reusing the same
  ids with byte-identical content is an idempotent no-op. `tracefork
  bundle-export <run_id> [-o bundle.db] [--store store.db]` / `tracefork
  bundle-import <bundle.db> [--store store.db]` are the CLI surface,
  distinct from the lossy OTel/OpenInference `export`/`ingest` commands
  (`interop.py`) — a bundle is bit-exact replayable, an OTel/OpenInference
  export is not. Offline/$0, pure local file I/O.
- `fsck.py` — `store_fsck()` is a read-only, git-fsck-style STRUCTURAL check
  over a `TapeStore` (distinct from `replay.py`'s replay-FIDELITY
  verification): every tape must decode via `load_tape`, every branch under a
  still-live parent must decode via `load_branch` (a decode error is reported
  as a `FsckRow` failure, never raised, so one bad row doesn't abort the
  scan), and every branch's `parent_run_id` must resolve to a live tape — an
  orphaned-parent failure reported even when `load_branch` alone would still
  succeed. Mirrors `replay.py`'s `CorpusCheckResult` dataclass-list-plus-
  `all_passed` shape (`StoreFsckResult.rows` / `.all_ok`). Never mutates the
  store. `tracefork verify --store <db>` is the CLI surface, mutually
  exclusive with `--corpus`.
- `blame.py` — `BlameEngine.rank()` forks each step `k` times, re-runs the agent, grades
  via an `Oracle`, counts flips vs. the parent outcome; `wilson_ci()` for intervals;
  `BudgetGovernor` estimates tail-call cost from `constants.PRICING_TABLE` before spend and
  `rank()` raises `BudgetExceededError` if the estimate exceeds `budget_usd`.
- `ci_calibration.py` — standalone Monte Carlo coverage-calibration harness for
  `blame.py`'s proportion CIs: fix a KNOWN ground-truth flip probability,
  simulate thousands of Bernoulli(`n_trials`) replicates over
  `random.Random(seed)`, compute the candidate interval per replicate via
  `blame.py`'s REAL `proportion_ci`/`wilson_ci`/`CIMethod` (never a parallel
  reimplementation of the interval math), and measure empirical coverage —
  the canonical Brown-Cai-DasGupta (2001) calibration methodology.
  `simulate_coverage()` runs one `(method, true_p, n_trials)` cell;
  `run_calibration()` sweeps a grid into a `CalibrationReport` mirroring
  `bench.py`'s dataclass-report shape. `monte_carlo_error()` is the standard
  error of the coverage estimate itself, which sets both the tolerance band
  a cell is judged against and the minimum trustworthy replicate count
  (`DEFAULT_N_REPEATS = 2000`). Pure math, deterministic given a seed,
  offline/$0 — no fork/agent/API calls. Standalone diagnostic today (not
  wired into any CLI command or gate).
- `judge.py` — OPT-IN, additive on top of `blame.py`'s `Oracle` protocol (never imported by the
  default $0 path): `LLMJudgeOracle` is a binary-rubric judge with few-shot examples, a
  configurable ("cross-family") judge model with a self-judge guard, position-swap averaging
  (grades twice with the candidate output moved in the prompt; disagreement or low average
  confidence abstains via `None`) — testable OFFLINE via an injected `judge_fn` (prompt -> raw
  text), never a real API call in tests. `calibrate_oracle()` measures any `Oracle`'s FPR/FNR/
  Cohen's kappa against a labeled gold set (`kappa_alert` below 0.6). `rogan_gladen_correct()` /
  `debias_flip_rate()` debias an observed flip-rate for judge FPR/FNR (Rogan-Gladen 1978) and
  widen a step's CI via delta-method propagation of BOTH k-sampling noise and finite-gold-set
  judge noise. Pure math, reuses `blame.z_from_confidence`; registers `"llm_judge"` into
  `blame.ORACLE_REGISTRY` at import time (importing the module is itself the opt-in).
- `faults.py` / `validate.py` — 5 fault classes (valid JSON, marker **inside** a content
  field) + the self-validation runner; a synthetic agent echoes each response forward so an
  injected fault propagates to a fault-aware tail. `run_all_fault_classes()` scores top-1.
  **Scope (don't overstate):** the fixture is a positive-vs-inert control on a short tape —
  it proves the engine is genuinely causal (not a fixed-slot artifact), not that it
  discriminates among competing causes on long tapes. See README → Validation scope.
- `competing_faults.py` / `bench.py` — the longer-tape ANSWER to that scope note: a
  7-exchange tape with several causally-DISTINCT faults planted at once (a root,
  a downstream echo, and a two-part necessary-not-sufficient AND-conjunction), scored
  against `blame.py`'s coalition/temporal-Shapley engine (`shapley_rank`) via `tracefork
  bench`. 8/9 planted cases resolve correctly; the 9th is a documented, NOT hidden,
  limitation of single-ordering temporal Shapley (it under-credits the earlier half of a
  symmetric conjunction) — see `competing_faults.py`'s module docstring and README →
  Validation scope. Cites, but does not reproduce, the published Who&When (ICML 2025)
  ~14.2% log-based step-attribution anchor as context only — no external dataset is ever
  downloaded (offline/$0 invariant applies here too). Zero-diff over the engines: both
  modules only call `blame.py`'s existing public API.
- `tournament.py` — `TournamentEngine.run()` ranks N pre-specified `Variant`
  candidate continuations at ONE fixed `step_index` (a different axis from
  `blame.py`'s per-step-across-runs comparison): each variant is forked `k`
  times via `ForkEngine.fork` (unchanged, zero-diff), graded by an `Oracle`,
  and scored by its own success rate (no baseline to flip away from) with a
  Wilson CI (`blame.proportion_ci`, reused). `TournamentEngine.estimate`
  prices the run via `BudgetGovernor.estimate` (never a parallel cost model,
  reused verbatim via probe tapes shaped for this engine's "N variants at one
  step" cost, unlike blame's "every step once") BEFORE any trial runs, and
  `run()` raises `blame.py`'s own `BudgetExceededError` if that estimate
  exceeds `budget_usd`. Forking the tape's LAST exchange has an empty tail —
  `ForkTransport` never calls its inner transport there — so the common
  "best-of-N final answers" comparison is genuinely $0. A winner is declared
  only when the top variant is significantly better than EVERY runner-up:
  each runner-up's trial count is tested one-sided (`blame.binom_sf_ge`,
  reused) against the top's observed rate as the null, and the p-values are
  jointly corrected via Benjamini-Hochberg (`blame.benjamini_hochberg`,
  reused) at `fdr_q` — so two variants with the same underlying success
  probability don't produce a spurious winner. `tracefork tournament` is the
  CLI surface (new command only; `report.py`/`server.py` untouched — a
  tournament result is a new artifact, not yet wired into the report UI).
- `report.py` / `server.py` / `web/report.html` — the single-file, dependency-free
  three-panel UI; `report.py` injects tape JSON (HTML-escaped against `</script>`
  breakout), `server.py` is FastAPI same-origin (no CORS, binds 127.0.0.1).
- `wire.py` / `synthetic.py` — Anthropic wire-format builders and the offline
  Scripted/FaultAware fake transports, in the **package** so production never imports from
  `tests/`; `tests/fakes.py` re-exports them.
- `replay.py` — `ReplayVerifier` (per-tape) and `run_fixture_corpus_check()`, which extends
  the `validate --check` idea to plain replay: gates a committed tape corpus
  (`experiments/replay_fixtures/` + its `manifest.json`) by asserting both bit-exact replay
  and a `digest()` match per fixture — `tracefork replay --check <dir>`. `fixtures.py` holds
  the tiny deterministic agents the corpus is built from (kept out of `validate.py` so the
  corpus doesn't couple to fault-testing concerns); `scripts/gen_replay_fixtures.py`
  (re)generates the corpus offline. `ReplayVerifier.verify()` also runs an opt-in provenance
  check: if the tape's `provenance` (see `tape.py`) is non-empty and recorded a
  `matcher_name`, a mismatch against the matcher actually used at replay raises a distinct
  `ProvenanceMismatchError` instead of a generic byte-diff divergence; empty provenance
  skips the check entirely.
- `certificate.py` — `ReplayCertificate`, a frozen dataclass whose `strength`
  (`UNVERIFIED`/`HASH_MATCHED`/`BIT_EXACT_FULL_REPLAY`) is constructor-enforced against its
  own `matched`/`total`/fingerprint fields (`ProofEnvelopeError` on overclaim) — a typed
  ceiling on the bit-exactness claim instead of a bare boolean a caller could misreport.
  `certificate_from_verification(result, tape)` is the sole producer wired to real data,
  recomputing the recorded fingerprint from `tape.digest()` rather than trusting `result`'s
  own field. Additive only: `VerificationResult.certificate` defaults to `None` and
  `ReplayVerifier.verify()` never sets it; `tracefork replay`/`verify` attach one and print
  an extra receipt line, every other caller is unaffected.
- `proxy.py` — `RecordProxy`/`ReplayProxy` (wired into FastAPI apps via
  `build_record_app`/`build_replay_app`) are a **localhost base-URL record/replay proxy**
  for clients the in-process httpx seam can't reach (curl, Node, Go, non-wrapped Python):
  point the client's `base_url` at `http://127.0.0.1:<port>` instead of the provider.
  Record forwards to a real (or, in tests, injected-fake) upstream and tees request+response
  bytes into a `Tape`, streaming SSE chunk-by-chunk while forwarding; replay serves recorded
  bytes with **no upstream**, matching each request to its recorded exchange by
  `matcher.py`'s existing `RequestMatcher` fingerprint (an unrecorded request, or a real
  body change, is a hard HTTP 502). Reuses `tape.py`/`matcher.py` unchanged — no in-process
  `NondetSource` exists on this path, so a proxy-recorded tape (`Tape.boundary =
  constants.PROXY_BOUNDARY`) sits outside the full single-process determinism boundary; see
  the module docstring and README → Localhost record/replay proxy.
- `adapters/` — opt-in framework adapters (`bind()` routes a framework's LLM client
  through the *existing* `TraceforkTransport`/`NondetSource`; `on_step()` maps a
  framework callback/event to a neutral `Step`/`StepDAG` overlay — observer-only,
  never a second capture path). `base.py` owns the protocol/registry;
  `langchain.py` (`ChatOpenAI`/`ChatAnthropic` + a tape-backed LangGraph
  checkpointer), `openai_agents.py` (defensive client-attribute injection +
  `agents.set_default_openai_client()` + a `TracingProcessor`),
  `crewai.py` (LiteLLM's `client_session`/`aclient_session` — CrewAI's actual httpx
  chokepoint — + a `crewai_event_bus` listener), `autogen.py` (defensive
  client-attribute injection + an `InterventionHandler` message-level seam), and
  `adk.py` (Google ADK: candidate-path injection into the `google-genai`
  `BaseApiClient`'s private `_httpx_client`/`_async_httpx_client` — reached
  through the target itself, a `genai.Client`, an ADK `Gemini` model wrapper, or
  an `LlmAgent` — + a `BasePlugin` registered once on the `Runner` for
  agent/model/tool before/after boundaries) are the concrete adapters, each its
  own optional extra with every framework import guarded, so `import tracefork`
  and the whole offline suite work with none of them installed. Where a
  framework's exact internal attribute names/event shapes aren't a documented
  stable API, injection is defensive (a short candidate list, never one
  hard-coded name) — see each module's docstring.
- `coverage.py` — determinism-coverage report for an already-loaded `Tape`:
  `tape_draw_coverage` tallies `nondet.py`'s three draw kinds
  (`clock`/`uuid`/`random`, only kinds that occurred — no zero-filled
  entries) and whether concurrency (`async_batches`) / `BoundaryGuard`
  (`provenance["boundary_guard"]`) were recorded/active; plus, given the
  agent's source *text*, `scan_source_for_nondeterminism_calls` is a
  best-effort `ast.parse`-only lint (never imports/executes the source) for
  call sites shaped like what `boundary_guard.py` itself patches
  (`Thread.start`/`Popen.__init__`/`random.random`/`time.monotonic`/
  `time.sleep`) vs. that module's own documented, permanent exclusions
  (`datetime.datetime.now()`/`time.time()`, always informational, never a
  violation regardless of guard state). `tracefork coverage <tape>
  [--agent-source FILE]` is the CLI surface. Read-only: never touches
  `Tape.digest()`/`to_bytes()`/`from_bytes()`.
- `cli.py` — Typer entry point for all sixteen commands.

`src/tracefork_spike/` holds the original Spike 0 (`fake_llm.py`, `agent.py`, `spike.py`):
record → save → load → replay → verify + negative control, with its own tests.

Most test files prove ONE module (or one seam) in isolation. Two are deliberately
cross-module, added once every feature bead had merged: `tests/test_e2e.py` chains
record → `TapeStore` save/load → replay → fork → blame → validate through the SAME
tape at every stage (not fresh fixtures per stage), plus the negative control,
asyncio-concurrency determinism, and every cross-feature path (MCP/tool exchanges
sharing a tape with LLM exchanges, redaction, OTel/OpenInference export→ingest,
plugin-registry resolution, `BoundaryGuard`, `divergence.py` diagnostics, the
base-URL proxy, `bench`, and the Bedrock/OpenAI/Gemini provider seams with their
documented scope boundaries called out explicitly, not papered over) — all
offline/$0. `tests/test_cli_smoke.py` invokes every one of the sixteen CLI
subcommands and asserts its real exit code; `serve`/`proxy record`/`proxy replay`
call `uvicorn.run()` directly, so those are driven by monkeypatching `uvicorn.run`
to a no-op (proving the CLI's own wiring without binding a socket) plus a
`TestClient`/ASGI-transport hit against the underlying FastAPI app for actual
serving behavior. `scripts/e2e.sh` runs the whole gate — sync, lint, format,
mypy, tests+coverage, `validate --check`, `replay --check`, `bench`, build+twine
— as one script with a single PASS/FAIL verdict. Both test files are additive
only: zero-diff over `transport.py`/`tape.py`/`fork.py`/`blame.py`/`matcher.py`.

## Invariants / conventions

- **Offline and $0 is non-negotiable** for the whole test suite, the spike, `validate`,
  and the demo — no key, no network. The synthetic transports (`synthetic.py`) are the
  seam; add to them rather than reaching for the real API. (`blame` on a real run is the
  one budget-capped exception.)
- **The agent must read time/ids/random only through `NondetSource`** — any direct
  `datetime.now()` / `uuid` / `random` breaks the determinism boundary and the
  bit-exactness claim. `BoundaryGuard` (opt-in, default off) turns a subset of these
  violations (thread/subprocess spawn, direct `random`/`time.monotonic`/`time.sleep`) into
  a loud record-time error instead of a mysterious later replay failure — see
  `boundary_guard.py` for exactly which calls it can and can't intercept.
- **The verifier proves, not asserts** — every request body is hash-checked against the
  tape; the negative control must keep failing (drift detected) or the proof is vacuous.
- **Declared determinism boundary (v1):** single-process (sync **or** asyncio), clock +
  id nondeterminism captured through `NondetSource`, **plus concurrency-graph determinism**
  — the completion order of concurrent asyncio fan-out is recorded and re-imposed on replay
  (see `transport.py`), so `gather`/`TaskGroup` agents replay bit-exact, not just
  single-call-at-a-time ones. Threads/subprocess are out of scope; fork and blame
  additionally assume the agent rebuilds its prefix deterministically (the property replay
  proves) — see `SPIKE0.md`.
- **No `Co-Authored-By: Claude` trailer** on commits in this repo (public portfolio repo,
  sole-author attribution).
- **Model IDs / pricing / SDK usage:** consult the `claude-api` skill before writing or
  editing any Anthropic integration code rather than relying on memory.
- `docs/superpowers/`, `.beads/`, `planning/` are gitignored local scaffolding (but
  `docs/demo.png` is committed). Runtime artifacts (`store.db`, `report.html`,
  `blame_*.json`, `validation_report.json`, `examples/demo_report.html`) are gitignored.
