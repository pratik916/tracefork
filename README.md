# tracefork

[![CI](https://github.com/pratik916/tracefork/actions/workflows/ci.yml/badge.svg)](https://github.com/pratik916/tracefork/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**A time-travel debugger for AI agents that doesn't just replay a failed run — it
proves the replay is bit-for-bit real, lets you fork any step, and measures *which*
step caused the failure, with confidence intervals.**

![tracefork report — timeline, exchange detail, and causal blame panel](docs/demo.png)

*The three-panel report: a run's timeline (left) with causal-blame badges, the
request/response for the selected exchange (center), and the blame ranking with 95%
confidence intervals (right). Generated offline, for $0, by
[`examples/demo_report.py`](examples/demo_report.py).*

---

## The idea

Every agent-observability tool shows you a trace and asks you to eyeball it. tracefork
treats an agent run like a recording you can rewind, branch, and reason about causally:

- **Record** every model call into a content-addressed **tape** at the HTTP seam of the
  Anthropic SDK, capturing the sources of nondeterminism (clock, ids) the agent reads.
- **Replay** the tape **bit-exact for $0** — every replayed request's *body* is
  sha256-checked against the tape, so it's *proven* identical, not asserted. (The matched
  surface is the request body; request headers such as `anthropic-beta` are out of scope —
  see [Determinism boundary](#determinism-boundary-v1-honest-scope).) No network, no key.
- **Fork** any step: swap in a different model response and let the *same* agent run
  forward from there. The unchanged prefix replays for free; only the new tail costs
  anything.
- **Blame**: resample those forks across every step and rank each by its **flip-rate** —
  how often perturbing it changes the run's outcome — with **Wilson score** confidence
  intervals so a small sample can't masquerade as certainty.
- **Validate the instrument itself**: inject faults with *known* root causes and confirm
  the blame engine fingers the right step. The engine is genuinely causal — it ranks
  whichever step actually flips the outcome #1, not a fixed slot — and across five
  injection mechanisms it hits **1.00 top-1 precision** offline against a flat negative
  control (which is now *enforced*, not just printed). See
  [Validation scope](#validation-scope) for exactly what that number does and doesn't claim.

That last pillar is the point: a debugger that claims to find root causes has to be
held to ground truth. `tracefork validate` is that proof, and it runs in under a second
with no API key.

## Quickstart (offline, $0, no API key)

Python **3.12** via [uv](https://docs.astral.sh/uv/). Everything below is offline and
makes no network calls.

```bash
uv sync --extra dev

# 1. The full offline test suite (65 tests).
uv run pytest -q

# 2. The instrument validates itself against injected, known-root-cause faults.
uv run tracefork validate

# 3. Generate the demo report shown above, then open it in any browser.
uv run python examples/demo_report.py
open examples/demo_report.html      # macOS; or just open the file

# 4. The original Spike 0 receipt: record → persist → replay → prove bit-exact.
uv run python -m tracefork_spike
```

`tracefork validate` prints:

```
  [PASS] corrupted_tool_output               top-1: 1.00
  [PASS] misleading_retrieval                top-1: 1.00
  [PASS] wrong_system_prompt                 top-1: 1.00
  [PASS] dropped_message                     top-1: 1.00
  [PASS] poisoned_argument                   top-1: 1.00

  overall top-1 precision: 1.00
  negative control max flip: 0.00 (threshold 0.30)
```

## The CLI

```bash
uv run tracefork --help
```

| Command | What it does |
|---|---|
| `replay  <tape> --agent pkg.mod:fn` | Replay a tape and print the bit-exact verification receipt. |
| `verify  <tape> --agent pkg.mod:fn` | Verify replay; exit non-zero on drift (CI gate). |
| `fork    <run_id> --step N --response f --agent pkg.mod:fn` | Fork a run at step N with a mutated response; record the counterfactual branch. |
| `blame   <run_id> --agent pkg.mod:fn [--k 10] [--budget 5.0]` | Rank every step by causal flip-rate with 95% CIs (re-runs the agent; budget-capped). |
| `report  <run_id> \| --tape <tape> -o out.html` | Render the self-contained three-panel HTML report. |
| `serve   [--store store.db] [--port 7777]` | Serve the live web UI (same-origin, 127.0.0.1). |
| `validate [--k 3] [--n-runs 5] [--check]` | Run the fault-injection suite; `--check` gates against the committed report. |

Replay, verify, fork, and the offline demos need no key. `blame` against a *real* run
re-runs the agent's counterfactual tails against the live API, which is why it's
budget-capped — the offline, $0 proof that blame works is `tracefork validate`.

## How it works

The spine is a **record/replay seam at the Anthropic SDK's httpx boundary** plus a
**nondeterminism-virtualization seam** the agent reads time and ids through. Bit-exactness
is the contract between them.

- **`transport.py`** — `TraceforkTransport` (sync) / `AsyncTraceforkTransport` (async).
  Record mode tees request+response bytes into the tape (buffering streaming SSE and
  plain JSON identically via `.read()`/`.aread()`); replay mode serves recorded bytes and
  sha256-asserts every request body matches the tape. A replay transport has **no inner
  transport**, so an unrecorded request is a hard error, never a silent network call.
- **`tape.py`** — content-addressed (sha256) blobs + an ordered event log, persistable to
  SQLite, with a hash-chain `digest()` fingerprint.
- **`nondet.py`** — `NondetSource` is the only way the agent gets time/ids;
  `RecordingNondet` logs real draws, `ReplayNondet` serves them back, `DriftingNondet` is
  the negative control. `find_divergence()` unwraps the `DivergenceError` the SDK buries
  inside an `APIConnectionError` so a real divergence isn't mistaken for a network blip.
- **`fork.py`** — `ForkTransport` runs three phases: **prefix-replay** (served from the
  parent tape for $0, request asserted to match — the agent must be deterministic up to
  the fork point), **mutation-injection** (same request, swapped response), and
  **tail-record** (the counterfactual continuation recorded fresh). A `Branch` carries
  `prefix_replayed`/`tail_recorded` counters that quantify the savings.
- **`blame.py`** — forks each step `k` times, re-runs the agent, grades the outcome via an
  `Oracle`, and counts flips vs. the parent outcome. `wilson_ci()` gives the interval;
  `BudgetGovernor` estimates fork count and dollar cost before any spend.
- **`faults.py` / `validate.py`** — five fault classes, each producing *valid* Anthropic
  JSON with a marker embedded inside a content field. A synthetic agent echoes each
  response into its next request, so an injected fault propagates through a fork to a
  fault-aware tail and flips the outcome — letting the blame engine be scored against
  ground truth entirely offline.
- **`report.py` / `server.py` / `web/report.html`** — a single, dependency-free HTML file
  (vanilla JS, no npm) rendered statically by `report` or served live by `serve`.

## Determinism boundary (v1, honest scope)

Bit-exact replay holds within a declared boundary: **single-process, clock + id
nondeterminism, captured through `NondetSource`**. An agent that reads `datetime.now()` /
`uuid` / `random` directly, or runs its loop across threads/subprocesses, steps outside
that boundary — and the verifier will *detect* the resulting drift rather than paper over
it. Forking and blame assume the agent rebuilds its prefix deterministically (the same
property replay proves). See [`SPIKE0.md`](SPIKE0.md) for how the boundary was de-risked.

## Validation scope

What `tracefork validate` proves, stated precisely: the blame engine is **genuinely
causal** — inject an outcome-flipping fault at *any* step and the engine ranks that step
first (verified by also injecting at a non-root step), so the 1.00 is not a tautology or a
fixed-slot artifact. The five "fault classes" carry two real injection mechanisms (a
corrupted tool argument and a replaced text message) via a marker that survives the SDK's
JSON round-trip, and the negative control — a no-op perturbation that must not flip the
outcome — is enforced with a hard threshold (the run fails if it ever exceeds 0.30).

What it does **not** yet claim: discrimination among *several competing* plausible causes
on a long run. The fixture is a short tape where one step gets a flip-capable perturbation
and the rest get an inert one — a clean positive-vs-control, but an easy one. A longer tape
with a decoy step that changes the transcript without changing the outcome is the next
iteration; until then, read 1.00 as "the instrument reliably finds the planted cause," not
"it resolves ambiguous multi-cause blame."

## Layout

```
src/tracefork/      transport, tape, nondet, recorder, fork, store,
                    blame, faults, validate, report, server, wire, synthetic, cli
src/tracefork_spike/  the original bit-exact record/replay spike
web/report.html     the single-file three-panel UI
examples/           runnable demo that produces the report above
tests/              65 offline tests ($0, no key)
experiments/        committed reference report for `validate --check`
```

## Testing

```bash
uv run pytest -q                                   # all 65 offline tests
uv run pytest tests/test_faults.py -q              # the self-validation chain
uv run tracefork validate --check                  # regression-gate vs committed report
```

## Contributing

Contributions are welcome — see [`CONTRIBUTING.md`](CONTRIBUTING.md) for dev setup,
the invariants a PR must respect, and commit/PR conventions. The whole dev loop
(tests, `validate`, lint, type-check) is offline and $0, so you can run the full gate
with no API key. Please also read the [Code of Conduct](CODE_OF_CONDUCT.md).

## Security

See [`SECURITY.md`](SECURITY.md) for how to report a vulnerability. In short: tapes
are JSON + base64 (never pickle, so loading one can't execute code), and `tracefork
serve` binds to 127.0.0.1 only.

## License

MIT — see [`LICENSE`](LICENSE).
