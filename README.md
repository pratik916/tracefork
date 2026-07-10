# tracefork

[![CI](https://github.com/pratik916/tracefork/actions/workflows/ci.yml/badge.svg)](https://github.com/pratik916/tracefork/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**A time-travel debugger for AI agents.** Record a run to a content-addressed tape, replay
it bit-for-bit for **$0** — proven by hash, not asserted — fork any step to explore a
counterfactual, and measure which step is causally responsible for a failure, with
confidence intervals.

![tracefork report — timeline, exchange detail, and causal blame panel](docs/demo.png)

*The report: a run's timeline with blame badges (left), the request/response for the
selected exchange (center), and the causal-blame ranking with 95% CIs (right). Generated
offline, for $0, by [`examples/demo_report.py`](examples/demo_report.py).*

---

## Why it's different

Most agent-observability tools show you a trace and ask you to eyeball it. tracefork treats
a run like a recording you can rewind, branch, and reason about causally:

- **Record** — every model call is teed into a content-addressed **tape** at the Anthropic
  SDK's HTTP seam, along with the clock/id/random draws the agent reads.
- **Replay** — the tape replays **bit-exact for $0**: every request body is sha256-checked
  against the tape, so replay is *proven* identical, not asserted. No network, no API key.
- **Fork** — swap a different response into any step and run the *same* agent forward. The
  unchanged prefix replays for free; only the new tail costs anything.
- **Blame** — resample those forks across every step and rank each by **flip-rate** (how
  often perturbing it changes the outcome), with **Wilson-score** confidence intervals so a
  small sample can't masquerade as certainty.

And, crucially, the instrument is **held to ground truth**: `tracefork validate` injects
faults with *known* root causes and confirms blame fingers the right step (**1.00 top-1**
across five injection mechanisms, with an *enforced* negative control); `tracefork bench`
plants several competing causes on one tape and checks the engine tells them apart (**10/11**,
the single exception named — not hidden). Both run offline, for $0, in seconds. See
[Validation scope](#validation-scope) for exactly what each number does and doesn't claim.

## Quickstart (offline, $0, no API key)

Python **3.12** via [uv](https://docs.astral.sh/uv/). Everything below is offline.

```bash
uv sync --extra dev

uv run pytest -q                       # full offline suite (890 tests, $0)
uv run tracefork validate              # blame vs injected, known-root-cause faults
uv run tracefork bench                 # discrimination among competing causes
uv run python examples/demo_report.py  # write examples/demo_report.html (the screenshot)
uv run python -m tracefork_spike       # the original bit-exact replay receipt
```

Or run the whole gate — lint, format, type-check, tests+coverage, the self-validation and
replay-corpus regression gates, the benchmark, and a package build — as one script with a
single PASS/FAIL verdict:

```bash
bash scripts/e2e.sh
```

`tracefork validate` prints:

```
  [PASS] corrupted_tool_output   top-1: 1.00
  [PASS] misleading_retrieval    top-1: 1.00
  [PASS] wrong_system_prompt     top-1: 1.00
  [PASS] dropped_message         top-1: 1.00
  [PASS] poisoned_argument       top-1: 1.00

  overall top-1 precision: 1.00
  negative control max flip: 0.00 (threshold 0.30)
```

## CLI

```bash
uv run tracefork --help
```

| Command | What it does |
|---|---|
| `replay <tape> --agent pkg:fn` | Replay a tape and print the bit-exact verification receipt. |
| `verify <tape> --agent pkg:fn` | Verify replay; exit non-zero on drift (CI gate). Also `--corpus` (fixture-corpus gate) and `--store` (structural fsck). |
| `fork <run_id> --step N --response f --agent pkg:fn` | Fork a run at step N with a mutated response; record the counterfactual branch. |
| `blame <run_id> --agent pkg:fn [--k 10] [--budget 5.0]` | Rank every step by causal flip-rate with 95% CIs (re-runs the agent; budget-capped). |
| `validate` / `bench` | Self-validation against injected faults / competing-cause benchmark. |
| `report <run_id> -o out.html` / `serve` | Render the self-contained HTML report, or serve the live UI (127.0.0.1). |

Also available: `coalition-fork`, `tournament`, `diff`, `receipt`, `coverage`, `prune`,
`bundle-export`/`import`, `export`/`ingest` (OTel / OpenInference), `proxy`, and `session`.
Run `--help` on any command for details.

> Replay, verify, fork, and the offline demos need no key. `blame` on a *real* run re-runs
> the agent's counterfactual tails against the live API, so it's budget-capped — the offline,
> $0 proof that blame works is `tracefork validate`.

## Install

```bash
pip install tracefork          # core: offline/$0, no provider or framework SDKs
pip install 'tracefork[all]'   # + providers, Bedrock, MCP, observability
```

Framework adapters are separate extras so one library's version churn can't block the rest:
`frameworks` (LangChain/LangGraph), `openai-agents`, `crewai`, `autogen`, `adk`. Providers
(`openai`, `google-genai`) come via `providers`; AWS via `bedrock`. Every framework import is
guarded — `import tracefork` and the full test suite run with none of them installed.

> Single-quote bracketed installs — unquoted `[...]` is glob-expanded by zsh into `no matches found`.

## How it works

The spine is a **record/replay seam at the Anthropic SDK's httpx boundary** plus a
**nondeterminism-virtualization seam** the agent reads time/ids through. Bit-exactness is the
contract between them.

- **`transport.py`** — record mode tees request+response bytes into the tape (streaming SSE
  and JSON alike); replay mode serves recorded bytes and sha256-asserts every request body
  matches. A replay transport has no inner transport, so an unrecorded request is a hard
  error, never a silent network call.
- **`tape.py`** — content-addressed (sha256) blobs + an ordered event log, persistable to
  SQLite, with a hash-chain `digest()` fingerprint.
- **`nondet.py`** — `NondetSource` is the only way the agent gets time/ids/random;
  `RecordingNondet` logs draws, `ReplayNondet` serves them back, `DriftingNondet` is the
  negative control that must keep failing.
- **`fork.py`** — three phases: prefix-replay ($0, asserted to match), mutation-injection
  (swapped response), tail-record (the counterfactual, recorded fresh).
- **`blame.py`** — forks each step `k` times, grades the outcome via an `Oracle`, counts
  flips vs. the parent; `wilson_ci()` for intervals, `BudgetGovernor` to cap spend.

Deeper design notes — every module, the load-bearing invariants, the async concurrency-graph
determinism, and the honest boundaries of each seam — live in
[`CLAUDE.md`](CLAUDE.md) and each module's docstring.

## Determinism boundary (honest scope)

Bit-exact replay holds within a declared boundary: **single-process (sync or asyncio), with
clock/id/random nondeterminism captured through `NondetSource`** — including the completion
order of concurrent asyncio fan-out, which is recorded and re-imposed on replay. An agent
that reads `datetime.now()`/`uuid`/`random` directly, or spans threads/subprocesses, steps
outside it — and the verifier *detects* the resulting drift rather than papering over it. An
opt-in `BoundaryGuard` turns the catchable violations into a loud error at record time. See
[`SPIKE0.md`](SPIKE0.md) for how the boundary was de-risked.

## Validation scope

Read this before trusting any accuracy number here. The load-bearing, *proven* claim is the
bit-exact, hash-verified replay substrate (`replay --check`, `verify`, the spike receipt). The
causal/blame claims are validated on controlled, labeled fixtures — **not** real-world traces.

- **`tracefork validate` — is the engine genuinely causal?** Yes, on a short control: injecting
  an outcome-flipping fault at *any* step makes the engine rank that step #1 (verified by also
  injecting at a non-root step), so 1.00 top-1 is not a fixed-slot artifact. The negative
  control is enforced with a hard threshold. It does **not** claim discrimination among
  competing causes.
- **`tracefork bench` — does it discriminate competing causes?** Mostly: a longer tape plants a
  root cause, a downstream echo that must not be blamed as root, and a necessary-not-sufficient
  conjunction. **10/11 resolve as planted.** The one exception — single-ordering temporal
  Shapley under-crediting the earlier half of a *symmetric* conjunction on a strictly sequential
  tape — is reported by `bench` itself (`[LIMITATION]`), pinned by a test, and closed when the
  same conjunction is recorded through a real `asyncio.gather`.

tracefork has **not** been run against any external benchmark — no dataset is ever downloaded
(offline/$0 is non-negotiable). Read the numbers as: *"the instrument reliably finds a single
planted cause, and — with one named exception — discriminates among several on one longer run,"*
not as a score on real multi-agent traces.

## Integrations & advanced features

Each is opt-in and documented in its module docstring (and, where noted, a dedicated doc):

- **Framework adapters** — LangChain/LangGraph (incl. tape-backed LangGraph time-travel),
  OpenAI Agents SDK, CrewAI, AutoGen, Google ADK. Each keeps the byte seam at the httpx
  transport and uses the framework layer only for step structure.
- **Providers** — OpenAI, Gemini, and AWS Bedrock (a second, parallel botocore seam with SigV4
  canonicalization and a standalone event-stream codec).
- **Localhost proxy** (`proxy.py`) — a base-URL record/replay proxy for non-Python clients
  (curl, Node, Go).
- **Redaction** (`redact.py`) — opt-in secret/PII scrubbing; metadata redaction stays
  bit-exact-replayable, content redaction is marked forensic-only.
- **OTel / OpenInference interop** (`interop.py`) — export a run as a `gen_ai.*` trace or
  OpenInference dataset; ingest an external trace's step structure (blame-by-re-execution, not
  bit-exact replay).
- **Trust receipt** (`tracefork receipt`) — an in-toto-shaped, JSON-safe evidence document +
  Shields.io badge, with absent evidence marked explicitly rather than defaulted to "verified."
- **Prune / retention** (`tracefork prune`) — soft-archive-only tape retention (never a hard
  delete).
- **Plugin API** — a `Registry` + entry-point loader for custom matchers/oracles/providers/
  serializers/adapters; nothing loads without explicit opt-in. See
  [`docs/plugin-api.md`](docs/plugin-api.md).

## Layout

```
src/tracefork/        transport, tape, nondet, recorder, matcher, redact, fork, store,
                      blame, faults, validate, competing_faults, bench, report, server,
                      certificate, coverage, checkpoint, bundle, fsck, diff, receipt,
                      tournament, interop, observability, proxy, bedrock_transport,
                      eventstream, cli, adapters/, providers/
src/tracefork_spike/  the original bit-exact record/replay spike
web/report.html       the single-file UI (timeline, exchange detail, blame, fork tree)
examples/             runnable demo that produces the report above
tests/                890 offline tests ($0, no key)
experiments/          committed reference reports for the regression gates
```

## Testing

```bash
uv run pytest -q                       # all 890 offline tests
uv run tracefork validate --check      # regression-gate vs committed report
uv run tracefork replay --check experiments/replay_fixtures  # replay-corpus gate
uv run tracefork bench                 # competing-cause discrimination
```

## Contributing

Contributions welcome — see [`CONTRIBUTING.md`](CONTRIBUTING.md) for dev setup, the invariants
a PR must respect, and commit/PR conventions. The whole dev loop is offline and $0. Please also
read the [Code of Conduct](CODE_OF_CONDUCT.md).

## Security

See [`SECURITY.md`](SECURITY.md). In short: tapes are JSON + binary blobs (never pickle, so
loading one can't execute code), and `tracefork serve` binds to 127.0.0.1 only.

## License

MIT — see [`LICENSE`](LICENSE).
