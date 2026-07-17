# CLAUDE.md

This file guides Claude Code when working in the `tracefork` repository.

## What this is

`tracefork` is a time-travel debugger for AI agents: record an agent run to a
content-addressed **tape**, replay it **bit-exact for $0** (hash-verified), fork any
step, and measure causal blame with confidence intervals — the instrument itself
validated against runs with injected, known root-cause faults.

**Current state: v1 built.** All five product pillars work offline and are tested
(1306 tests, $0): streaming-capable record/replay with drift detection, the three-phase
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
uv run pytest -q                     # full offline suite (1306 tests)
uv run pytest tests/test_faults.py::test_validation_runner_fingers_fault_step -q   # one test
uv run tracefork validate            # self-validation: blame vs injected, known faults
uv run tracefork validate --check    # regression-gate vs experiments/validation_report_committed.json
uv run tracefork bench                # long-tape competing-fault discrimination benchmark
uv run python examples/demo_report.py   # write examples/demo_report.html (the README screenshot)
uv run python -m tracefork_spike     # the original Spike 0 bit-exact replay receipt
uv run tracefork --help              # replay, verify, fork, coalition-fork, converge, conflicts, diff,
                                      # report, receipt, settlement-diff, serve, blame, tournament,
                                      # validate, bench, proxy, export, ingest, prune, coverage,
                                      # corpus-blame, locate, query, bundle-export, bundle-import;
                                      # sub-apps: `branch` (descendants/ancestors/siblings) and
                                      # `session` (create/spawn/show/cost/divergence/cross-blame/chaos/
                                      # record/replay/fork/blame/serve/board)
uv run tracefork replay --check experiments/replay_fixtures   # replay-as-regression gate
bash scripts/e2e.sh                  # single-receipt gate: sync, lint, type-check, tests+coverage,
                                      # validate --check, replay --check, bench, build+twine, one PASS banner
```

## Architecture (the parts that span files)

The spine is a **record/replay seam at the Anthropic SDK's httpx boundary**, plus a
**nondeterminism-virtualization seam** the agent reads time/ids through. Bit-exactness
is the contract between them.

The product lives in `src/tracefork/`:

- `nondet.py` — `NondetSource` is the *only* way the agent gets time/ids/random/env/file
  draws (`now_iso`/`new_uuid_hex`/`random_float`/`get_env`/`read_file`). `RecordingNondet`
  draws real values and logs them; `ReplayNondet` serves them back in order; `DriftingNondet`
  is the negative control (fresh values → forced divergence). `random_float()` logs the exact
  `float.hex()` string (lossless round-trip, no float-formatting dust) and, like `now_iso()`,
  is additive/opt-in — an agent must be handed the active `NondetSource` explicitly; nothing
  patches `random` globally the way `Recorder` patches `uuid.uuid4`. `get_env(name,
  default=None)` logs a NUL-joined `"{flag}\0{name}\0{value}"` string so an unset variable
  round-trips distinctly from `""` and `ReplayNondet.get_env` can assert the replayed call
  names the SAME variable the tape recorded. `read_file(path)` pre-checks
  `os.path.getsize` against a `max_read_file_bytes` cap (`DEFAULT_MAX_READ_FILE_BYTES`, 256
  KiB) *before* touching the file — over-cap raises `ReadFileTooLargeError` with no
  partial/truncated draw ever landing on the tape — then logs a JSON envelope
  (`path`/`size`/`sha256`/`content_b64`) so `ReplayNondet.read_file` can assert the same path
  and return the exact bytes with zero filesystem access on replay. `read_file` stores
  content raw/unredacted today (redaction through `redact.py` is a deliberate follow-up); the
  size cap is the shipped mitigating control in the meantime. `find_divergence()` unwraps a
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
  explicit wins over `TraceforkConfig.boundary_guard`, both default `False`). A second,
  independently opt-in `confinement: ConfinementSpec | None = None` parameter
  additionally patches `builtins.open` (reject write-mode opens resolving outside
  `writable_roots`; reads always allowed) and `socket.socket.connect` (reject hosts
  outside `allowed_hosts`, raised before any DNS/TCP attempt) for the guard's active
  window, restored symmetrically on exit; `confinement=None` (the default) leaves both
  completely unpatched. `ConfinementViolationError` subclasses `BoundaryViolationError`.
  Capabilities are declared as data (`ConfinementSpec`, a frozen dataclass) and verified
  independently at this boundary rather than derived from the agent's own tool-call
  args (the confused-deputy hole); this is a fixed local allowlist, not a full OS
  sandbox — Landlock/Seatbelt-grade backends are an explicit future tier.
  `ForkEngine.fork()`/`fork_coalition()` (`fork.py`) take a matching `confinement=`
  kwarg that forces the guard active for the re-executed agent even when
  `boundary_guard=False`, confining a fork's tail-record phase to a declared
  writable/network surface. Since tracefork-bge.72, `ConfinementViolationError` additionally
  sets optional structured keyword attributes at both its raise sites (`_guarded_open`/
  `_guarded_socket_connect`) — `violation_kind` (`"write"`/`"connect"`), `attempted`, and
  whichever of `declared_writable_roots`/`declared_allowed_hosts` applies — so
  `confinement_diagnostics.py` can build a typed diagnostic straight off the exception
  rather than parsing `str(error)`; all default to `None`, so the pre-bge.72
  single-message-arg shape and every existing `match=`-based test still passes.
- `confinement_diagnostics.py` — `ConfinementDiagnostic`/`diagnose_confinement(error)` turn an
  already-raised `ConfinementViolationError` into a typed, JSON-safe explanation, reading the
  exception's own structured attributes above (never re-parsing `str(error)`).
  `confinement_diagnostic_to_dict()` is the CLI/web JSON view. Read-only: builds nothing new,
  changes no raise site. `cli.py`'s `fork`/`coalition-fork` commands catch
  `ConfinementViolationError` and print this diagnostic before exiting 1.
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
  — see `replay.py`'s opt-in `ProvenanceMismatchError` check. **v6** adds `request_urls`
  (parallel-indexed to `exchanges`: the URL captured at each exchange's real capture seam in
  `transport.py`/`bedrock_transport.py`) so a provider whose model id lives in the URL path
  rather than the body (Gemini/Bedrock) can still be resolved, via `providers/*.detect_model`'s
  `request_url` fallback and its downstream consumers (`blame.py`'s `_detect_model`,
  `interop.py`); like `async_batches`/`provenance` it is metadata only, **never** fed into
  `digest()`, so every tape's digest is unchanged and a pre-v6 tape upcasts to
  `[""] * len(exchanges)`. It's a JSON
  header + zstd blobs, **not pickle** — no
  arbitrary-code-execution risk. `open_sqlite()` is the one hardened connection factory
  (WAL, `synchronous=NORMAL`, `busy_timeout`, `foreign_keys=ON`); writers take `BEGIN
  IMMEDIATE`. `store.py` reuses it and serializes its write fan-out with a lock.
- `tapequery.py` — `state_at(tape, n)`/`slice(tape, start, end)`: read-only views over
  `Tape.exchanges` only (not `tool_exchanges`/`async_batches`/`draws` — an honest, documented
  scope boundary, mirroring `checkpoint.py`'s precedent), decoding each exchange's
  request/response via `divergence.py`'s existing `_json_or_b64`. `state_at` folds exchanges
  `[0..n]` inclusive into a frozen `TapeState` (raises on an out-of-range `n`); `slice`
  returns the half-open `[start, end)` range as `ExchangeView` tuples (clamps out-of-range
  bounds like list-slicing, but raises on `start > end`). No new decode logic, no
  `digest()`/`to_bytes()` interaction, no CLI wiring yet.
- `providers/` — the `ProviderAdapter` protocol (`detect_model`/`parse_response`/build
  helpers) plus `AnthropicAdapter`/`OpenAIAdapter`/`GeminiAdapter`/`BedrockAdapter`,
  registered via `register_adapter`/`get_adapter`/`registered_providers`. `detect_model
  (request_bytes, request_url=None)` gained an optional `request_url` fallback: Anthropic/
  OpenAI ignore it (their body already carries a real `model` field); `GeminiAdapter`/
  `BedrockAdapter` parse the model id out of the URL path (`models/{id}:generateContent`,
  `/model/{id}/invoke`) when the body has none — sourced from `tape.py`'s new
  `request_urls[i]` (v6). A new `ProviderCapabilities` registry (`register_capabilities`/
  `get_capabilities`/`registered_capabilities` — a plain dict like the adapter registry,
  since nothing here needs third-party discovery) advertises `model_detectable`/
  `converse_response` per provider as advisory metadata, deliberately NOT part of the
  `ProviderAdapter` Protocol itself; an unregistered name gets a conservative all-`False`
  default rather than a `KeyError`. `blame.py`'s `_detect_model` falls through every OTHER
  registered adapter (each tried with its exchange's `request_url`) when the default
  (Anthropic) adapter finds nothing on any exchange, so a Bedrock/Gemini tape auto-resolves
  its real model instead of silently defaulting to Sonnet pricing.
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
- `basis.py` — `RecordBasis` (`tracefork_version`/`git_sha`) is the "what build recorded this
  tape?" witness, layered onto `Tape.provenance` exactly like `matcher_name`/
  `boundary_guard`/`nondet_mode` — two more optional string keys, never fed into `digest()`.
  `current_basis()` captures the running package version (`importlib.metadata`) and a
  best-effort `git rev-parse HEAD` (swallowing every failure to `""`). `Recorder`/
  `AsyncRecorder`'s opt-in `record_basis=True` writes it via `basis_to_provenance_keys`;
  `cli.py`'s `replay`/`fork`/`coalition-fork` commands read it back via
  `basis_from_provenance` and print a non-fatal `format_basis_drift_warning` when the
  replaying build differs from the recording one — diagnostic context, never a hard error or
  an exit-code change, distinct in kind from `replay.py`'s `ProvenanceMismatchError`.
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
- `checkpoint_stream.py` — a standalone `fastapi.APIRouter` (`GET /api/checkpoint/stream?
  path=&since_seq=`) live-tailing a `checkpoint.py`-backed in-progress recording as
  Server-Sent Events, pushing only sha256 digests of each new exchange's request/response
  bytes (never the raw bytes) plus a terminal `done` frame once `checkpoint_status` reports
  `was_finalized`. Deliberately **not** mounted onto `server.py`'s app in this module — its
  own tests mount `router` on a throwaway `FastAPI()` instance instead; `live.py`'s
  `tail_checkpoint` (below) is the function actually wired into `server.py`'s live endpoint.
  No `run_id`-registry, no `web/report.html` wiring — each a separate, out-of-scope
  follow-up per the module's own scope note.
- `live.py` — `tail_checkpoint(path, since_seq=, poll_interval=, max_polls=)` is the async
  generator `server.py`'s `GET /api/checkpoint/tail` actually streams: polls a
  `checkpoint.py`-backed recording and yields one SSE `event: exchange` frame per
  newly-committed exchange (reusing `report._tape_to_data`'s per-exchange preview shape via
  a throwaway single-exchange `Tape` — zero new summarization logic) plus a terminal
  `event: done` frame once finalized. Read-only observation of an already-recording (or
  already-recorded) checkpoint file — never controls the writer's process; true interactive
  breakpoint-before-tool-call control is explicit future scope. `checkpoint_stream.py`'s
  router (above) covers the same idea as a standalone, unmounted `APIRouter`; the two were
  built independently and are not yet consolidated.
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
  store-level metadata only — `Tape.digest()` itself is completely untouched. Every `Branch`
  also carries `parent_tape_digest` (the parent tape's own `digest()` at fork time) and
  `divergence_exchange_digest` (`compute_divergence_exchange_digest`: sha256 of the exact
  request+response bytes at the first divergence point) — a Certificate-Transparency-style
  citable fork point, re-verified independently on every `store.py` `load_branch` (see
  below) rather than trusted once at write time. Every `Branch` also carries
  `confinement_tier` (`compute_confinement_tier(boundary_guard, confinement)`, one of
  `constants.CONFINEMENT_TIER_NONE`/`_GUARDED`/`_DECLARED`) — an axis orthogonal to a tape's
  own `boundary` tiers, describing how confined the re-executed agent was during the fork's
  tail-record phase; a declared `ConfinementSpec` wins regardless of `boundary_guard`'s value
  (passing it forces the guard active). Branch/store-level metadata only, same discipline as
  `branch_digest` — never fed into `Tape.digest()`; persisted by `store.py`'s
  `save_branch(confinement_tier=)`/returned by `load_branch`/`list_branches`.
- `convergence.py` — `find_reconvergence(delta_tape_a, divergence_step_a, delta_tape_b,
  divergence_step_b)` checks whether two SAME-`divergence_step` sibling branches (e.g. two of
  `blame.rank`'s k trials, or two `tournament` variants) end up byte-identical again, reusing
  `fork.py`'s `compute_divergence_exchange_digest` verbatim as the per-step fingerprint — no
  new hashing. `ConvergenceResult.reconverged` is any match; `.stable` is the genuine signal
  (every step from the first match onward matches, ruling out a coincidental single-step
  collision that immediately reverts). Raises `ValueError` for branches with different
  `divergence_step`s (no well-defined shared alignment) rather than silently comparing
  misaligned offsets. `tracefork converge <branch_a> <branch_b>` is the CLI surface (exit 0
  iff stable).
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
- `effects.py` — `extract_effects(tape)` normalizes two independent "tool call" shapes
  already on any tape — Anthropic `tool_use` content blocks in `Tape.exchanges` and
  JSON-RPC `tools/call` frames in `Tape.tool_exchanges` — into one `Effect(source, index,
  tool_name, resource)` shape, via a pluggable per-tool-name `EFFECT_EXTRACTOR_REGISTRY`/
  `register_effect_extractor()` (a plain dict, deliberately not `plugins.py`'s heavier
  entry-point `Registry`) falling back to a small key-probe (`path`/`file_path`/`url`/...)
  then a canonical-JSON stringification. `diff_effects(tape_a, tape_b)` flags every
  `(tool_name, resource)` pair touched by both sides as an `EffectOverlap` — a read-only
  reviewer-sanity signal, no merge/apply logic. Needs no step-range slicing: pass a branch's
  `delta_tape` directly (mirrors `diff.py`'s `Tape`-in-`Tape`-out contract). `tracefork
  conflicts <parent> <branch_a> <branch_b>` is the CLI surface (exit 1 iff an overlap found).
- `settlement.py` — `branch_settlement_diff(parent_tape, branch, *, divergence_step=,
  branch_digest=)` decodes a winning fork's post-divergence `delta_tape.tool_exchanges`
  JSON-RPC frames into `SettlementOp`s (`tool_name`/`arguments`/`result`/`step_index`), then
  `to_settlement_json()` renders them as an in-toto-Statement-shaped, digest-keyed
  (`parent_tape_digest`+`branch_digest`) dict for an EXTERNAL apply/settlement layer to
  consume — TraceFork itself never applies/settles anything, this is export-only. Takes
  either a live `fork.Branch` or a store-reloaded plain `Tape` (mirrors `diff.py`'s
  dual-input contract). Decodes frames inline rather than reusing `effects.py`'s `Effect`
  shape on purpose: `Effect.resource` collapses arguments to one comparable string for
  conflict-detection, while a `SettlementOp` needs the full `arguments`/`result` — same
  underlying frames, two different shapes for two different jobs. `tracefork settlement-diff
  <run_id> <branch_id>` is the CLI surface (always exits 0, an export not a gate).
- `query.py` — a small query-language layer (`state <run_id> <step>` / `diff <a> <b>
  [--step N]` / `causes <run_id> <step|--closure>` / `tree <run_id>`) adding ZERO new engine
  logic: `dispatch(store, line)` parses one line and calls straight through to
  already-shipped read primitives — `report._tape_to_data`, `diff.branch_diff`/
  `diff.tape_diff`, and `store.py`'s `causal_edges_for_run`/`cited_by`/`causal_closure`/
  `list_branches` — formatting their existing return shapes as text. `QueryError` wraps any
  bad syntax/out-of-range step/unknown verb/unknown id into one clean, printable message,
  never a raw `KeyError`/`ValueError`/`IndexError`. Pure string-returning function, no I/O,
  no `cmd`/`typer` import — `repl.py` (below) is the interactive wrapper.
- `repl.py` — `QueryShell(cmd.Cmd)` is the thin, stdlib-only interactive loop over
  `query.dispatch`: every input line is handed straight to `dispatch()` and its result (or
  `error: ...` on a `QueryError`) is printed — no verb-specific `do_*` method, so the grammar
  stays defined in exactly one place (`query.py`). `run_repl(store)` runs the loop. `tracefork
  query --store <db>` opens it; `--cmd <line>` runs one query non-interactively and exits
  (scriptable, CI-testable without blocking on stdin).
- `locate.py` — `locate_value(tape, value)` scans `tape.exchanges` then `tape.tool_exchanges`
  (the same order `Tape.digest()` chains) for `value` as a UTF-8 substring, returning a
  `TapeHit` carrying `sha256_hex` of the matched raw bytes plus the tape's own `digest()` —
  an offline-checkable receipt anyone can independently re-verify, no new hash scheme.
  `locate_in_lineage(store, root_run_id, value, follow_lineage=True)` BFS-walks a run's fork
  lineage (`store.list_branches`/`load_branch`, the same promotion-convention traversal
  `causal_closure`/`branches_forked_from` already document) collecting every `LocateHit`
  (depth-tagged) where `value` occurs. Entirely read-only. `tracefork locate <value>
  <run_id>` (or `--tape` for a single file, no lineage) is the CLI surface (exit 0 iff found).
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
  additively after its existing JSON write. `sessions`/`spawn_edges` are a
  SEPARATE new schema for cross-agent orchestration/delegation lineage — NOT
  `Tape.async_batches` (a single agent's own per-run asyncio fan-out;
  unrelated, never conflate the two) and NOT the fork/counterfactual
  `branches` DAG. A session is rooted at one `run_id`; each `spawn_edges`
  row is a `parent_run_id -> child_run_id` delegation edge (+ optional
  `spawn_reason`), both FK-checked against `tapes(run_id)` (an unknown
  run_id raises `sqlite3.IntegrityError`, never silently accepted).
  Modeling delegation as its OWN graph — distinct from the causal/execution
  graph `causal_edges` already covers — follows 2026 delegated-execution
  observability practice: collapsing the two breaks under async fan-out and
  re-delegation, which `fork.py`/`blame.py` do constantly.
  `create_session`/`add_spawn_edge` mirror `save_tape`/`save_branch`'s
  `BEGIN IMMEDIATE` + `self._write_lock` write discipline; `session_tapes`
  BFS-walks the spawn graph reachable from a session's root (deduplicated —
  a run reached via more than one path, e.g. a diamond, appears once);
  `spawn_children`/`spawn_parent` answer the direct-neighbor queries. A NEW,
  separate `runtime_checkable` `SessionStore` Protocol (not merged into
  `StorageBackend`, so no third-party `StorageBackend` implementer breaks)
  names this seam; `TapeStore` satisfies both. `stored_digest`/`all_branch_parents`
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
  plain reachability walk. `branches` also carries `fork.py`'s
  `parent_tape_digest`/`divergence_exchange_digest`, migrated in the SAME
  guarded `ALTER TABLE` pass as `branch_digest` (one migration, not two);
  `save_branch` gains matching optional parameters (default `''`, every
  existing caller — including `cli.py`'s fork command, which does not pass
  them — unaffected). `load_branch` is the re-verification point: when a
  branch recorded a non-empty `parent_tape_digest`, it recomputes the parent
  tape's CURRENT digest and compares it against the stored value, raising
  `ForkPointDriftError` (hard error, never silently logged and continued) on
  a mismatch; an empty `parent_tape_digest` has nothing to re-verify against
  and skips the check. `server.py`'s `get_branch` catches
  `ForkPointDriftError` and maps it to HTTP 409, alongside its existing
  `KeyError` → 404. `branch_descendants(run_id)`/`branch_ancestors(run_id)`/
  `branch_siblings(run_id)` are three more read-only DAG-relationship queries over
  `branches` (BFS downward via `parent_run_id`, promoted branches only recursed into; walk
  upward via `branch_id -> parent_run_id`; and the other branches sharing one's own parent,
  reusing `list_branches` — no new SQL), backing `cli.py`'s `branch` sub-app
  (`descendants`/`ancestors`/`siblings`) and `server.py`'s `GET /api/branch/{run_id}/related`.
  The `branches` table also carries `fork.py`'s `confinement_tier` (migrated in the same
  guarded `ALTER TABLE` pass as the other branch-metadata columns; `save_branch`/
  `load_branch`/`find_branch_by_digest`/`list_branches` all thread it, defaulting to `''`).
  `spawn_edges` gains a nullable `spawn_step_index` column (which step of the parent this
  child was spawned at — `None` for a pre-existing edge, a real distinct value
  `cross_tape_blame.py` documents its own fallback for, never coerced to `0`);
  `session_spawn_children(session_id, run_id)` is `spawn_children`'s session-scoped sibling
  (never leaking a same-`parent_run_id` spawn edge across two different sessions), and
  `spawn_edges_for_session(session_id)` returns a session's full edge set (including
  `spawn_step_index`) in one call, powering `cross_tape_blame.session_topological_order`.
- `corpus.py` — read-only aggregation over `store.py`'s already-persisted `causal_edges`
  rows, adding zero new SQL: `build_corpus_blame_index(store, top_n=)` joins every run's
  edges (`list_runs()`/`causal_edges_for_run()`) into a corpus-wide top-responsible index
  (`CorpusEdgeSummary.score` — `flip_rate` for a `"blame"` edge, `shapley_value` for
  `"shapley"`, each on its own native scale). `detect_regressions(store, method=,
  z_threshold=2.0, min_history=3)` groups by `(agent_name, step_index, method)`, sorts by the
  RUN's `created_at` (lexical-ISO, the same convention `prune()`'s `older_than_iso` relies
  on), and flags the latest point as a `RegressionFlag` when its z-score against that step's
  own prior history (population mean/stdev, stdlib `statistics`) clears the threshold —
  skipped for too-little history or zero-variance history. `tracefork corpus-blame` is the
  CLI surface (a diagnostic, always exits 0); a full web panel is deferred.
- `cross_tape_blame.py` — deliberately NOT a joint cross-tape coalition-execution engine
  (that needs `fork.py`'s `CoalitionForkTransport` to relax its single-linear-causal-ordering
  assumption first — see `docs/session-cross-tape-design-spike.md`). What it ships instead:
  `RunRef(run_id, step_index)` names one step of one tape within a session;
  `session_topological_order(store, session_id)` interleaves every tape reachable in a
  session (`store.session_tapes`'s BFS) into one cross-tape step order, splicing each spawned
  child's full step range in at its recorded `spawn_edges.spawn_step_index` (falling back to
  "entirely after its parent's own steps" when that column is `None` — a pre-this-bead edge,
  or any caller that omits it); `cross_tape_causal_edges(store, session_id)` aggregates every
  tape's already-persisted `causal_edges` rows in that cross-tape order — zero new execution,
  a read-only view over data the graph store already has. `tracefork session cross-blame
  <session_id>` is the CLI surface.
- `session_ops.py` — small, directly-testable helpers backing `cli.py`'s `session` sub-app,
  zero engine-module changes: `parse_spawn_spec`/`record_session` batch-parse `--spawn
  PARENT:CHILD[:REASON]` specs and register a session + all its spawn edges in one call
  (looping the existing `create_session`/`add_spawn_edge`); `ensure_run_in_session(db,
  session_id, run_id)` is the membership guard `session fork`/`session blame` call before
  delegating, unmodified, to the top-level `fork`/`blame` command functions (Typer's
  `@app.command()` returns the callback unmodified, so this is in-process composition, not a
  second engine); `build_uniform_agent_manifest` maps every tape in a session to the same
  agent fn for `session replay`'s common one-agent case, reusing
  `session_replay.session_divergence_rollup` unchanged; `session_deep_link_path` formats
  `session serve`'s only new surface (the pre-existing `GET /api/session/{id}` route).
- `session_replay.py` — `session_divergence_rollup(store, session_id, agent_fns)` replays
  every `session_tapes()`-reachable tape that `agent_fns` (a caller-supplied `run_id ->
  callable` map, the same shape `replay.run_fixture_corpus_check`'s manifest uses) maps a
  callable for, via the EXISTING `replay.ReplayVerifier`, unchanged — returning as soon as
  the first non-bit-exact tape is found (later tapes are never loaded). A reachable run_id
  absent from `agent_fns` lands in `skipped_run_ids`, never silently dropped or miscounted as
  a pass. `resolve_agent_manifest()` resolves a `{run_id: "module:fn"}` JSON manifest into
  callables via the same `importlib` pattern `run_fixture_corpus_check` uses. `tracefork
  session divergence --agents-manifest <path.json>` is the CLI surface.
- `session_cost.py` — `plan_session_fork(store, session_id, target_run_id, k=, model=,
  cost_per_fork_usd=)` prices a minimal-recompute fork: BFS's `target_run_id`'s transitive
  spawn descendants (`_spawn_descendants`, over `store.spawn_children`) as the conservative
  "recompute set" (any fork is assumed to potentially invalidate its ENTIRE spawn subtree,
  since `spawn_edges` has no per-step parent/child association yet), prices it and the naive
  "recompute everything in the session" baseline via `blame.py`'s existing
  `BudgetGovernor.estimate` (summed per tape, zero new pricing math), and reports
  `savings_usd`/`savings_pct`. Estimator/planner only — never re-executes anything; actually
  threading a fork's counterfactual output into each recompute-set descendant is a separate,
  larger follow-on. `tracefork session cost <session_id> <target_run_id>` is the CLI surface.
- `session_chaos.py` — schedule DERIVATION only, generalizing `transport.chaos_release_order`
  to a session's spawn-lineage graph; never a replay driver (no multi-tape
  orchestration-replay harness exists to drive it — a materially larger, separate effort).
  `session_chaos_release_orders(store, session_id, seed)` calls the real, unmodified
  `chaos_release_order` per reachable tape with a seed derived from the base seed + that
  tape's own `run_id` (`_derive_seed`, stable regardless of BFS discovery order).
  `session_sibling_chaos_order(store, session_id, seed)` is the new axis: for each parent
  with 2+ spawn children (scoped to `session_spawn_children`, never leaking across sessions),
  a seed-shuffled permutation of those children's `run_id`s — the delegation-graph analogue
  of `chaos_release_order`'s within-batch shuffle, one level up. `tracefork session chaos
  --seed N` is the CLI surface.
- `report_session.py` — `generate_session_report(store, session_id, output_path, agent_map=)`
  is the offline CLI static-generator for a multi-agent session fork-board
  (`web/session_report.html`): one lane per `store.session_tapes()`-ordered run_id, reusing
  `report.py`'s `_tape_to_data`/`_safe_json` verbatim (zero edits to `report.py`) plus each
  lane's `spawn_parent`/`spawn_children`. A run_id present in the optional `agent_map` (the
  same resolved-callable shape `session_replay.resolve_agent_manifest` produces) gets a real,
  freshly-computed `replay.ReplayVerifier` receipt; one absent renders the neutral
  `replay={}` empty state, never a fabricated status. `tracefork session board <session_id>`
  is the CLI surface. Live-mode serving through `server.py` is a deliberate, documented
  follow-on — this ships the static path only.
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
  `BudgetGovernor.confinement_risk(tape, k=, coalition_samples=, async_batches=,
  confinement=)` is pure disclosure (never a gate, unlike `estimate()`'s cost check): a
  `ConfinementRisk(projected_trials, confined, note)` naming how many trials a
  `rank()`/`shapley_rank()` run will execute and whether they run under a `ConfinementSpec`
  (see `boundary_guard.py`); both methods accept an additive `confinement=None` kwarg
  forwarded to every `ForkEngine.fork()`/`fork_coalition()` trial (byte-identical when
  omitted) and attach the resulting risk to `BlameReport.confinement_risk`/
  `ShapleyReport.confinement_risk`. `_detect_model` now falls through every registered
  non-default provider adapter (see `providers/`) when Anthropic's own body-based detection
  finds nothing on any exchange, so a Bedrock/Gemini tape auto-resolves its real model via
  `request_urls` instead of defaulting to Sonnet pricing.
- `narrative.py` — four pure functions (`explain_flip_result`/`explain_shapley_result`/
  `explain_blame_report`/`explain_shapley_report`) template `blame.py`'s
  `FlipRateResult`/`ShapleyResult`/`BlameReport`/`ShapleyReport` fields into deterministic
  markdown/sentences — no new computation, reusing each result's own `.interpretation` string
  rather than re-deriving the 0.7/0.3 thresholds. Fixed number formatting (`.0%`/`.3g`) makes
  two calls on the same input byte-identical — safe to diff/hash/snapshot-test. Wired
  additively into `cli.py`'s `blame` command, writing a `blame_<run_id>.md` companion
  alongside the existing `blame_<run_id>.json`.
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
- `field_oracle.py` — `FieldDiffOracle(field_path=, success_re=, failure_re=)` is an
  additive, OPT-IN `Oracle` (same protocol as `blame.py`'s `StringMatchOracle`) that grades
  one JSON field's value instead of the whole output text: resolves a `$.`-style key path
  (mirrors `divergence.py`'s `FieldDiff.path` convention) via a small local tokenizer/walker,
  then regex-matches success/failure against the resolved leaf — scoping grading to one
  field so an unrelated field's stray "SUCCESS"/"FAIL" substring can't flip a verdict. Any
  resolution failure (non-JSON output, missing key, out-of-range index) returns `None`
  (ambiguous), never raises. Registers itself into `blame.ORACLE_REGISTRY` under
  `"field_diff"` at import time, the same opt-in pattern `judge.py` establishes above.
  `cli.py`'s `blame --field <path>` is the CLI surface.
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
  bench`. 10/11 planted cases resolve correctly; the one that doesn't is a documented,
  NOT hidden, limitation of single-ordering temporal Shapley over a strictly SEQUENTIAL
  tape (it under-credits the earlier half of a symmetric conjunction) — see
  `competing_faults.py`'s module docstring and README → Validation scope. Cites, but does
  not reproduce, the published Who&When (ICML 2025) ~14.2% log-based step-attribution
  anchor as context only — no external dataset is ever downloaded (offline/$0 invariant
  applies here too). Zero-diff over the engines: both modules only call `blame.py`'s
  existing public API. `shapley_rank`'s optional `async_batches` parameter
  (`tracefork-bge.10`) closes that limitation for tapes recorded through
  `AsyncTraceforkTransport`: a batch's members genuinely raced, so `_batch_blocks`/
  `_block_orderings` compute each member's EXACT marginal contribution averaged over
  every one of the batch's internal join orders (the local Shapley value of that small
  sub-game), independent of `m_samples`; a falsy `async_batches` (every pre-existing
  caller) is byte-for-byte the original single-ordering algorithm.
  `build_concurrent_gate_payload_tape`/`run_shapley_concurrent` record the SAME
  GATE/PAYLOAD conjunction through a REAL `asyncio.gather` (not a hand-constructed
  `tape.async_batches`) and both halves now resolve `necessity=True`.
  `BudgetGovernor.estimate` gains a matching `async_batches` parameter that inflates its
  pre-flight cost estimate by the batch's `b!` — a safe, if loose, upper bound on the new
  walk's true (per-position-varying) cost, so real spend never outruns it.
- `archetypes.py` — generalizes `competing_faults.py`'s fixed 7-step fixture into a
  PARAMETERIZED fault-scenario generator: `_ArchetypeTail`/`make_linear_agent`/
  `_make_perturb_factory` genericize its `RuleBasedTail`/agent loop/`make_perturb_factory` to
  caller-chosen chain length and role placement. Three hand-derivable archetypes, each
  verified against `shapley_rank`'s exact documented semantics: `run_or_redundancy(pos_a,
  pos_b, n_turns)` (two independently-sufficient OR-causes with no shared marker text — the
  OR-mirror of `competing_faults.py`'s ROOT/ECHO AND case); `run_n_way_conjunction(arity)` (a
  parameterized k-part AND, generalizing the 2-part GATE/PAYLOAD case across `arity=2..5`);
  `run_long_relay(n_relay)` (a lone root cause propagated through a parameterized-length
  inert chain, proving attribution is invariant to propagation length). Rescoped to exactly
  these 3 archetypes with concrete parameters, not an open-ended scenario DSL. Zero-diff over
  the engines, offline/$0 like `competing_faults.py`; no CLI wiring yet.
- `concurrent_validate.py` — answers the true-negative gap `competing_faults.py`'s symmetric
  2-member GATE/PAYLOAD fixture can't: `build_multi_branch_tape(n_branches=3)` records,
  through a REAL `AsyncTraceforkTransport`/`asyncio.gather`, `n_branches` genuinely-concurrent
  sibling calls (so `tape.async_batches` carries a real, non-hand-constructed batch entry),
  then `make_single_branch_perturb_factory(faulty_step)` lights up exactly ONE sibling's fault
  marker. Forwarding the real `async_batches` into `BlameEngine.shapley_rank` proves the
  engine both credits the guilty sibling (`necessity=True`) AND correctly reads
  `necessity=False` on its innocent siblings in the SAME unordered batch — a true-negative
  proof the symmetric fixture structurally cannot give. `run_concurrent_branch_validation
  (n_branches=)` sweeps every branch position plus a negative control, mirroring
  `validate.py`'s top-1-precision-plus-negative-control shape. Zero-diff over the engines,
  offline/$0; no CLI wiring yet.
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
- `fork_allowlist.py` — the security gate for `server.py`'s click-to-fork endpoints: nothing
  is fork-able through `POST /api/run/{id}/fork[/estimate]` unless the operator explicitly
  allowlists it (`TRACEFORK_FORK_AGENTS` env var, or `serve --allow-fork-agent
  name=module:fn`, repeatable) — the same opt-in-only posture `plugins.py`'s entry-point
  registry already establishes; merely running the server must never be enough to let a
  request re-execute arbitrary agent code. `parse_allowlist_env`/`resolve_agent_fn` parse and
  resolve it, raising `AgentNotAllowlistedError` (naming what IS allowlisted) rather than a
  bare `KeyError`/`ImportError`. `estimate_single_fork_usd(tape, step)` prices ONE targeted
  fork (k=1, only the `n - 1 - step` tail calls a real click-to-fork bills) via the existing
  `providers.get_adapter`/`pricing.get_rates` seams — deliberately not
  `BudgetGovernor.estimate`, which prices a full multi-step sweep and would badly overstate a
  single fork's cost.
- `cost_profile.py` — `compute_cost_profile(tape, provider=)` aggregates a tape's exchanges
  into per-model (`ModelCost`) and per-tool (`ToolCost`) cost rows for the report's cost
  dashboard panel, built entirely on the existing `providers.get_adapter`/`pricing.get_rates`
  seams `blame.BudgetGovernor.estimate` already uses — no new pricing logic.
  `_normalize_exchange` mirrors `blame._avg_tokens`'s exact fallback (adapter
  `parse_response`, filling missing model/token fields via `detect_model`/`constants.SONNET`
  and a ~4-bytes-per-token estimate on parse failure). Per-tool cost attribution is an honest
  over-count for multi-tool exchanges: Anthropic bills per-exchange, not per-tool-call, so an
  exchange invoking more than one tool attributes its FULL modeled cost to each tool name it
  called — no token-level sub-split is invented. `cost_profile_to_dict()` is the JSON-safe
  view `cli.py`'s `report` command and `server.py`'s `GET /api/run/{id}` both embed.
- `report.py` / `server.py` / `web/report.html` — the single-file, dependency-free
  four-panel UI; `report.py` injects tape JSON (HTML-escaped against `</script>`
  breakout), `server.py` is FastAPI same-origin (no CORS, binds 127.0.0.1).
  `server.py` also exposes an additive `GET /api/session/{session_id}`
  (`store.py`'s `sessions`/`spawn_edges`, mirroring `get_run`'s
  404-on-`KeyError` pattern) — explicitly out of scope for `report.html`'s
  UI itself (tracefork-bge.12), a JSON-only surface for now. The fourth
  panel is a fork-tree view (tracefork-bge.15) built on the now-landed
  `branch_digest`/DAG metadata (tracefork-bge.21) and `store.py`'s
  already-persisted branch fields: `_tape_to_data`/`generate_report` gain an
  optional `branches: list[dict] | None = None` parameter defaulting to `[]`
  (the same falsy empty-state pattern `replay={}` already establishes);
  `cli.py`'s `report` command threads `store.list_branches(run_id)` through
  when loading via `run_id`, leaving `branches=None` (documented scope
  limit, no store to query) on the `--tape` path; `server.py`'s `get_run`
  adds `data["branches"] = store.list_branches(run_id)` additively.
  `web/report.html`'s `renderForkTree` draws a git-graph-style row layout
  (branches ranked by `divergence_step`, edge-labeled with `mutation_desc`/
  `branch_digest`) as inline SVG — no new JS library, no CDN — following
  `renderTimeline`'s existing vanilla-JS pattern; a live-mode node click
  fetches `/api/branch/{id}`, a static report shows an honest
  'live-mode-only' affordance instead. The panel renders one level (a run's
  direct branches) — a full multi-level fork-of-fork DAG render is future
  scope (see `store.py`'s `branches_forked_from`). Additive since:
  `_tape_to_data`/`generate_report` also accept `causal_edges`/`branch_details`/`shapley`/
  `cost_profile`/`causal_closure`/`run_id` (each falsy-default, same empty-state pattern as
  `branches`) — `causal_edges` (persisted blame/Shapley rows) and `branch_details` (each
  branch's full delta-tape report data keyed by `branch_id`, an explicit
  `{"error": "fork_point_drift"}` marker when `load_branch` raises `ForkPointDriftError`
  rather than aborting the whole report) drive `renderForkTree`'s causal-heatmap overlay
  (cross-referencing a branch's `divergence_step` against `causal_edges`) and let a static
  report's fork-tree clicks render real data with zero live server; `shapley` renders a
  per-exchange necessity/sufficiency quadrant badge (`shapleyQuadrantHtml`) in the Timeline
  panel; `cost_profile` (see `cost_profile.py`) renders a per-model/per-tool cost dashboard
  panel; `causal_closure` renders "external anchor" entries — responsible edges reachable via
  fork-promotion lineage that may belong to OTHER run_ids — distinguished from this run's own
  blame rows via `run_id`. `cli.py`'s `report` command populates all of these when loaded via
  `run_id` (still empty on the `--tape` path, the same documented scope limit); `server.py`'s
  `GET /api/run/{id}` populates `causal_edges`/`cost_profile`/`causal_closure` the same way.
  `web/report.html` also ships a Timeline scrubber (play/pause + a tick-per-exchange slider).
  `server.py` additionally exposes: click-to-fork endpoints `POST /api/run/{id}/fork
  [/estimate]` (gated by `fork_allowlist.py`'s allowlist plus an explicit `confirm: true`,
  never on by default); `GET /api/branch/{id}/related` (`store.py`'s
  `branch_descendants`/`branch_ancestors`/`branch_siblings`); `GET /otel/{trace_id}
  [/{span_id}]` (`interop.py`'s `locate_trace`/`locate_span_step` — an OTel exemplar
  back-link redirecting to the report view, localhost-only); `GET /api/checkpoint/tail`
  (`live.py`'s `tail_checkpoint`, a live-tail SSE endpoint over an in-progress
  `checkpoint.py` recording — read-only, no `web/report.html` wiring yet); and `GET /runs`
  (`web/runs.html`, a plain multi-run dashboard/picker linking to `/?run_id=`).
  `report_session.py`'s session fork-board (`web/session_report.html`) is a separate
  template reusing `report.py`'s helpers, not part of this file.
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
- `receipt.py` — `build_trust_receipt(tape, replay=, validate_report=,
  bench_report=)` is pure composition, no new engine logic: a JSON-safe,
  in-toto-Statement-shaped (subject-by-digest + predicate, unsigned today,
  upgradeable to DSSE later) dict combining `tape.digest()[:16]`/`boundary`/
  `content_redacted` with already-computed evidence — a fresh ($0)
  `ReplayVerifier.verify()` result (via `verification_result_to_dict`, the
  same conversion `cli.py`'s `report` command already applies) plus the
  parsed `validation_report.json`/`bench_report.json` dicts `validate`/
  `bench` already write. Any evidence left `None` renders as an explicit
  `{"available": False}` marker, never an omitted key or a defaulted
  "verified" claim. `build_shield_json(receipt)` derives a Shields.io
  endpoint-badge dict: green (`brightgreen`) only when replay is bit-exact
  AND validate's `overall_top1_precision` clears the same 0.7 bar
  `cli.py`'s `validate` command already prints against, red on a detected
  replay divergence, yellow otherwise — a `content_redacted` tape (see
  `redact.py` / tracefork-bge.20) never badges green regardless of the
  other evidence. The badge message embeds the receipt's own fingerprint
  prefix so a stale badge is visible at a glance. `tracefork receipt`
  (mirrors `report`'s run_id/--tape/--agent loading) is the CLI surface,
  writing `receipt.json` (+ an optional Shields.io badge JSON via
  `--shield-output`); offline/$0 — it only re-runs replay and reads
  already-generated JSON off disk, never triggers a live blame call.
- `release_receipt.py` — `build_release_receipt(version=, test_summary=, coverage_summary=,
  validate_report=, bench_report=, replay_corpus=, calibration=)` mirrors `receipt.py`'s
  exact philosophy at the repo/release level instead of the tape level: pure composition over
  already-computed (or freshly, $0-computed via `replay.run_fixture_corpus_check`/
  `ci_calibration.run_calibration`) evidence, never a parallel reimplementation. Any evidence
  left `None` renders as an explicit `{"available": False}` marker, never an omitted key or a
  defaulted "passing" claim. The composed body hashes (canonical `json.dumps(sort_keys=True)`
  → sha256) into `receipt_digest`, the same Merkle-style content-address idiom as
  `Tape.digest()`/`fork.py`'s `branch_digest`. `sign_release_receipt(receipt, signing_key=)`
  HMAC-SHA256-signs it when `TRACEFORK_RELEASE_SIGNING_KEY` is set — documented honestly as
  a symmetric attestation, **not** a DSSE/asymmetric signature; `verify_release_receipt_
  signature()` checks one back. `tracefork release-receipt <version>` writes the signed
  receipt to `docs/release_receipts/<version>.json` (exit 1 if the replay corpus or
  calibration sweep isn't clean).
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
  hard-coded name) — see each module's docstring. `shepherd.py` is a sixth,
  OpenAI-path-only adapter for the Shepherd agent framework — since there is no
  published `shepherd` package to
  `pip install`/guard against, it ships with no `shepherd_available()`/`require_shepherd()`
  guard and no real-framework test tier, validated entirely against a synthetic double
  (`TraceforkShepherdCore`) shaped like Shepherd's `OpenAIProvider`; `bind()` only routes
  that OpenAI-path client (Shepherd's Claude/OpenCode providers are explicitly out of scope,
  stated in `bind()`'s own `notes` field).
- `coverage.py` — determinism-coverage report for an already-loaded `Tape`:
  `tape_draw_coverage` tallies `nondet.py`'s five draw kinds
  (`clock`/`uuid`/`random`/`env`/`read_file`, only kinds that occurred — no zero-filled
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
- `selfaudit.py` — an architecture-fitness gate: `audit_package(package_root)`
  `ast.parse`-scans (never imports/executes) every `.py` file under `src/tracefork/` for
  direct calls to the exact 5 call-shapes `BoundaryGuard` patches
  (`Thread.start`/`Popen.__init__`/`random.random`/`time.monotonic`/`time.sleep`) plus
  `uuid.uuid4` (patched globally by `recorder.py`, not `BoundaryGuard`, but the same class of
  violation), outside a tiny, comment-justified `SANCTIONED_CALL_SITES` allowlist (today:
  only `store.py`'s `uuid.uuid4().hex[:12]` id-generator calls). Turns the prose invariant
  ("tracefork's own source never bypasses `NondetSource`/`BoundaryGuard`") into a
  machine-checked one, reusing `coverage.py`'s existing `_call_path` dotted-path resolver
  rather than duplicating it. Same best-effort scope limits as `coverage.py`'s own lint
  (misses aliasing/indirection). Not yet wired into `scripts/e2e.sh`.
- `cli.py` — Typer entry point for all top-level commands: `replay`/`verify`/`fork`/
  `coalition-fork`/`diff`/`converge`/`conflicts`/`settlement-diff`/`report`/`receipt`/
  `release-receipt`/`serve`/`blame`/`tournament`/`validate`/`bench`/`export`/`ingest`/
  `prune`/`proxy`/`coverage`/`corpus-blame`/`locate`/`query`/`bundle-export`/`bundle-import`,
  plus the `branch` sub-app (`descendants`/`ancestors`/`siblings`, see `store.py`) and the
  `session` sub-app (`create`/`spawn`/`show`/`board`/`cost`/`divergence`/`record`/`replay`/
  `fork`/`blame`/`cross-blame`/`chaos`/`serve`) for `store.py`'s orchestration-session/
  spawn-lineage schema. `fork`/`coalition-fork` gain repeatable `--writable-root`/
  `--allowed-host` flags building a `ConfinementSpec` (forwarded as `confinement=`), catching
  `ConfinementViolationError` and printing `confinement_diagnostics.diagnose_confinement`'s
  fields before exiting 1. `blame` gains `--field <path>` (routes to
  `field_oracle.FieldDiffOracle` instead of `StringMatchOracle`), prints
  `BudgetGovernor.confinement_risk`'s disclosure line, and additionally writes a
  `narrative.explain_blame_report` markdown companion. `replay`/`fork`/`coalition-fork` all
  print a non-fatal `basis.format_basis_drift_warning` when a tape's recorded build
  (`basis.py`) differs from the current one. `report` additionally embeds `causal_edges`/
  `branch_details`/`shapley`/`cost_profile`/`causal_closure` (see the `report.py` entry
  above) when loaded via `run_id`. `serve` gains `--allow-fork-agent name=module:fn`
  (repeatable), wiring `fork_allowlist.py`'s allowlist into `server.py`'s click-to-fork
  endpoints.

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
offline/$0. `tests/test_cli_smoke.py` invokes every one of the CLI
subcommands and asserts its real exit code; `serve`/`proxy record`/`proxy replay`
call `uvicorn.run()` directly, so those are driven by monkeypatching `uvicorn.run`
to a no-op (proving the CLI's own wiring without binding a socket) plus a
`TestClient`/ASGI-transport hit against the underlying FastAPI app for actual
serving behavior. `scripts/e2e.sh` runs the whole gate — sync, lint, format,
mypy, tests+coverage, `validate --check`, `replay --check`, `bench`, build+twine
— as one script with a single PASS/FAIL verdict. Both test files are additive
only: zero-diff over `transport.py`/`tape.py`/`fork.py`/`blame.py`/`matcher.py`.
`scripts/check_executed_evidence.py` is the executed-evidence CI sentinel
(tracefork-bge.25): both `scripts/e2e.sh` and `.github/workflows/ci.yml` run
pytest with `--junit-xml=junit.xml`, then this script cross-checks the
JUnit report against `tests/required_test_ids.txt` (a curated
classname::name manifest of this repo's most safety-critical tests — the
negative control, `Tape.digest()`/round-trip tests, bench's
`unexpected_failures` regression, the `ReplayCertificate` negative control,
checkpoint/bundle round-trips, and the Hypothesis property tests) and hard-
fails (exit 1) if any required id is absent or present-but-skipped, so a
narrowed `-k` selection or a silent skip can no longer pass CI/e2e green on a
bare 0 exit code alone. Pure stdlib `xml.etree.ElementTree` parsing of a
report pytest itself just wrote in the same run (not untrusted input) — see
the module's own docstring for that trust-boundary note. Renaming/removing a
manifested test is a maintenance obligation: update the manifest in the same
change, or the sentinel correctly starts failing.

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
