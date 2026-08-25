# Spike 0 — bit-exact record/replay (the headline-risk de-risk)

> **Status: superseded by v1 (2026-08-25).** This document records the *original* Spike 0
> de-risk and its (deliberately narrow) proven scope. Two things this spike named as future
> work have since shipped in v1 and are proven, not just planned: **streaming SSE**
> capture/replay (`transport.py` tees response bytes chunk-by-chunk in both record and
> replay mode) and **asyncio concurrency-graph determinism** (the completion order of
> concurrent `asyncio.gather` fan-out is recorded and re-imposed on replay — see
> `README.md`'s Determinism boundary section). Where this document still says either is
> "out of scope," read that as historical — true when Spike 0 was written, superseded by
> v1. Test counts and the pasted receipt below are as of Spike 0's original run and are not
> re-verified against the current suite; see `README.md`/`CLAUDE.md` for current numbers.

**Question:** Can we record a tool-using Anthropic-SDK agent run and replay it
**bit-exact**, with proof, for **$0 and no network**, within a *declared* determinism
boundary? This is the one assumption the whole tracefork project rests on (feature
list receipt **R1**). If replay is only "mostly exact," the product collapses.

**Verdict: YES, within the scoped boundary.** Proven end-to-end, offline, in CI.

```
  tracefork — Spike 0: bit-exact record/replay
  ----------------------------------------------------
  recorded exchanges ........ 2
  nondeterminism draws ...... 2  (clock + id, virtualized)
  request hashes matched .... 2/2
  tape fingerprint .......... <sha256, illustrative — matches replay fingerprint>…
  replay fingerprint ........ <sha256, illustrative — matches tape fingerprint>…
  network calls / spend ..... 0 / $0.00
  agent final answer ........ 'Done — your flight to Tokyo is booked. Confirmation CONF_…'

    [x] replayed trajectory byte-identical to recorded
    [x] tape fingerprint matches after save/load round-trip
    [x] every replayed request hash matched the tape
    [x] every recorded nondeterminism draw consumed
    [x] negative control: drift was DETECTED, not silently passed

  RESULT: PASS
```

Run it yourself: `uv run python -m tracefork_spike` (receipt) and
`uv run pytest tests/test_spike0.py -q` (8 tests — this file's own suite; `uv run
pytest -q` with no path runs the whole project, currently far more than 8). No
`ANTHROPIC_API_KEY` required.

## What it proves

1. **The transport seam is real and provider-real.** Recording happens at the
   `anthropic` SDK's `http_client` boundary (`httpx.Client(transport=...)`). The
   genuine SDK runs a genuine tool-use loop; the fake endpoint emits real Anthropic
   *wire-format* JSON, so swapping in the real network later is a one-line change with
   the record/replay machinery unchanged.
2. **Nondeterminism virtualization is load-bearing and works.** The toy agent's
   `book_flight` tool stamps a wall-clock `booked_at` and a fresh `confirmation_id`
   on every run. Those values flow into the *next* request body, so replay can only be
   byte-exact because every draw is captured at record time and served back in order.
3. **The verifier proves exactness, it doesn't assert it.** Every replayed request
   body is sha256-checked against the tape; the whole tape has a hash-chain
   fingerprint that survives a SQLite save/load round-trip (content-addressed blobs +
   ordered event log).
4. **Drift is detected, not silently passed (negative control).** Replaying with
   *drifting* nondeterminism (fresh real clock/id) makes the turn-2 request diverge,
   and the replay transport raises `DivergenceError`. A tampered tape (one extra byte
   in a recorded request) is likewise rejected. This is the faithfulnessbench
   "the instrument can report failure" discipline applied here.

## The one real surprise (worth keeping)

**The Anthropic SDK masks any exception raised inside its httpx transport as
`anthropic.APIConnectionError`.** A `DivergenceError` raised from the replay transport
arrives wrapped in `APIConnectionError.__cause__`. If we hadn't unwrapped it
(`nondet.find_divergence` walks the `__cause__`/`__context__` chain), a genuine replay
divergence would have masqueraded as a network blip — exactly the kind of silent
dishonesty this project exists to prevent. v1 must keep this unwrap at the boundary.

## The declared determinism boundary (v1 scope)

In-scope and proven here: a **single-process, synchronous** agent whose only
nondeterminism sources are **clock** and **id generation**, both routed through the
`NondetSource` seam, talking to the API through the SDK's injectable httpx transport.

Deliberately out of scope for Spike 0 (tracked, not solved *at the time*):

- **Streaming SSE bytes.** Used non-streaming here. Capture/replay is mechanically
  identical (same transport seam, tee the response bytes); streaming only adds the
  SDK's SSE accumulator on top of identical bytes. Called "next spike step" here —
  **superseded in v1**: `transport.py` now tees and replays SSE chunk-by-chunk, in
  scope and tested, not a future step.
- **Cross-process replay.** Record and replay run in one process, so request
  serialization is trivially stable. Persisted-tape replay across processes needs the
  SDK's body serialization to be byte-stable across runs — likely true (pydantic), but
  must be proven separately. (Still the case in v1: replay stays single-process.)
- **Threads / asyncio scheduling / subprocess** nondeterminism — explicitly outside
  the v1 boundary *as originally scoped here*. **Superseded in v1** for asyncio
  specifically: concurrent `asyncio.gather` fan-out completion order is captured
  through `NondetSource` and re-imposed on replay (see `README.md`'s Determinism
  boundary section) — proven in scope, not just planned. Threads and subprocesses
  remain genuinely out of scope.

## Biggest open architectural risk (flagged, not yet spiked)

The feature list names the **Claude Agent SDK** (`claude-agent-sdk`) as the v1 target.
That SDK executes the agent loop in a **subprocess** (the Claude Code runtime), so the
in-process httpx seam used here would **not** capture its model calls. Two viable paths,
to be decided before the full build:

- **(A) Target the Anthropic API SDK directly** (what Spike 0 proves) — capture is
  clean and in-process; we provide the agent loop. Lower risk, slightly less
  "batteries-included."
- **(B) Intercept at the subprocess boundary** for the Claude Agent SDK — a
  fundamentally harder capture problem (proxy the subprocess's transport / protocol).

Recommendation: **build v1 on path (A)** (Anthropic SDK + our own loop), keep path (B)
as a stretch. Spike 0 validates (A) end-to-end.

## Files

`src/tracefork_spike/`: `nondet.py` (the virtualization seam + drift unwrap),
`tape.py` (content-addressed, hash-chained, SQLite-persistable tape), `fake_llm.py`
(offline Anthropic-wire-format endpoint), `transport.py` (record/replay capture seam),
`agent.py` (toy tool-use agent on the real SDK), `spike.py` (orchestration + receipt).
Tests in `tests/test_spike0.py`.
