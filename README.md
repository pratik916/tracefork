# tracefork

**A time-travel debugger for AI agents that doesn't just replay failures — it proves,
bit-for-bit, that the replay is real, then measures which step caused the failure.**

> Status: **Spike 0 complete.** The headline assumption — bit-exact, $0, no-network
> replay of an agent run within a declared determinism boundary — is proven end-to-end.
> See [`SPIKE0.md`](SPIKE0.md). Full design lives in
> `../ideas/2026-06-11-tracefork-features.md`.

## The idea

Every agent-observability tool shows you a trace and asks you to eyeball it. tracefork
records every model call, tool result, and source of nondeterminism into a
content-addressed **tape**, so any run replays **bit-exact for $0** — hash-verified, not
asserted. Then you can fork any step (edit a tool result, re-run from the divergence
point) and a causal engine resamples the forks to tell you, with confidence intervals,
*which* step actually caused the outcome — validated against runs with injected,
known root-cause faults.

## What runs today (Spike 0)

The de-risking spike: record a real tool-using Anthropic-SDK agent, persist the tape to
SQLite, reload it, replay with zero network, and **prove** the replay is byte-identical
— including a negative control that confirms drift is *detected*, not silently passed.

```bash
# Python 3.12 via uv; tests/spike need no ANTHROPIC_API_KEY and make no network calls.
uv sync --extra dev
uv run python -m tracefork_spike     # prints the bit-exact replay receipt
uv run pytest -q                     # 8 offline tests ($0)
```

## Why it's hard (and why the spike matters)

Bit-exact replay is the whole product, and it's the single hardest piece of
engineering: you have to capture *every* nondeterminism source and virtualize it so the
replayed run rebuilds byte-identical requests. Spike 0 proves the mechanism on a scoped
boundary (single-process, synchronous, clock + id nondeterminism) and surfaces the real
risks for the full build — chiefly that the Claude Agent SDK runs its loop in a
subprocess, so v1 targets the in-process Anthropic SDK seam. Details in
[`SPIKE0.md`](SPIKE0.md).
