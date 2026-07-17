# Session cross-tape design spike (tracefork-bge.66)

`tracefork-bge.66` ("tracefork session CLI verb family", backlog item 66,
P3/XL) is gated on this spike: work out what "cross-tape fork" and
"cross-tape blame" would need to MEAN over `store.py`'s
`sessions`/`spawn_edges` schema, before shipping any CLI surface for them.
This doc is that gate. It concludes that genuine cross-tape fork/blame is
blocked on a real prerequisite refactor, names that refactor precisely, and
specifies the honest, narrower verb family actually shipped in this slice
instead.

## What "cross-tape" would have to mean

A session is a spawn-lineage graph over independent tapes: a root tape plus
every tape reached by following `spawn_edges.parent_run_id -> child_run_id`
hops (`TapeStore.session_tapes`, a BFS). Each tape is its own agent's own
recorded run — its own `Tape.digest()`, its own linear exchange list. Two
genuinely NEW (not just "run the existing thing N times") operations become
imaginable once tapes are linked this way:

1. **Cross-tape counterfactual fork** — mutate a response on tape A (the
   parent/delegating run) and ask what a SPAWNED sibling or child tape B
   would then have said, had it received A's mutated output as its own
   input. This is not "fork A" and, separately, "fork B" — it requires
   re-deriving B's own request/response sequence from a causal input that
   never actually flowed to it, i.e. propagating one tape's counterfactual
   THROUGH the spawn edge into another tape's re-execution.
2. **True cross-tape causal attribution** — instead of asking "which step of
   THIS tape caused THIS tape's outcome to flip" (what `blame.py` already
   answers, rigorously, with Wilson/Jeffreys/CP/AC CIs and BH-FDR
   selection), ask "which step of a DELEGATED sub-agent's tape caused the
   PARENT's eventual outcome to flip" — attributing a parent's bad result to
   a specific call inside a child it spawned, with the same statistical
   rigor blame.py already gives single-tape attribution.

Both of these are real, valuable, and squarely what a "session" CLI verb
family should eventually deliver. Neither is what ships in this slice — see
below for why.

## The blocking prerequisite

Both operations above require the intervention primitive to understand a
causal ordering that spans MORE than one tape's own linear exchange list.
`fork.py`'s `CoalitionForkTransport` — the engine underneath every fork and
every blame trial today — hardcodes exactly one causal-ordering assumption:
every step's predecessor set is its own tape's full linear prefix, and
`first_step = min(interventions)` is the sole request-matched divergence
point; every later intervention within the SAME tape is forced
unconditionally because "the agent's requests have already diverged by
then" (see `fork.py`'s `CoalitionSpec`/`CoalitionForkTransport` docstrings).
There is no notion, anywhere in `fork.py` or `blame.py`, of a step's
predecessor set reaching INTO a different tape's exchange list. Forcing a
response on tape A today has zero mechanism for propagating that forced
value across a `spawn_edges` boundary into tape B's own replay/re-execution
— tape B's `ReplayVerifier`/`ForkTransport` only ever know about tape B's
own recorded bytes.

This is not a new observation specific to this bead: it is the SAME gap
already named in `docs/shepherd-gap-analysis.md`'s completeness-critic
item 5 ("Sequencing gap / hidden prerequisite"), which calls out precisely
this dependency —

> several P1-P2 items (Orchestration session model, Multi-agent session
> replay/fork-board UI, Multi-agent ground-truth fixture, Cross-tape causal
> blame) are listed as additive features, but the inventory documents that
> CoalitionForkTransport's whole model assumes exactly one causal ordering
> (every step's predecessor set is its full linear prefix) and a
> single-process determinism boundary. Multi-agent/orchestration support
> requires relaxing that foundational assumption first — the backlog
> doesn't call out this prerequisite refactor or sequence it before the
> features that depend on it.

Backlog item 58 ("Cross-tape / multi-agent causal blame (DAG across
sub-agent tapes)", P3/XL) and the "Generalize CoalitionForkTransport from a
flat index-set assumption to a genuine DAG-shaped intervention model"
leapfrog opportunity (theme: Causal attribution & blame) name the actual
fix: relaxing `CoalitionForkTransport`'s single-linear-ordering assumption
to an explicit, per-step, possibly-cross-tape parent-set — the same
generalization `shapley_rank`'s temporal-Shapley walk would need to become
full multi-permutation Shapley over a non-linear (here: multi-tape) DAG.
That refactor touches `fork.py`'s core intervention primitive and
`blame.py`'s Shapley walk — real engine-module surgery, not a CLI
convenience — and is explicitly OUT OF SCOPE for this bead.

## What ships in this slice instead

Given that prerequisite is not done, this bead ships five additive
`session` verbs that are honest about operating PER-TAPE over a session's
existing spawn manifest, not across the spawn edges themselves:

- **`session record`** — a batch convenience: `TapeStore.create_session`
  followed by looped `add_spawn_edge` calls in one CLI invocation, instead
  of N+1 separate `session create`/`session spawn` calls. No new engine
  logic.
- **`session replay`** — a rollup: `ReplayVerifier(tape, agent_fn).verify()`
  run independently once per `session_tapes()`-reachable run_id, printed as
  one table. Each tape's replay is completely unaware of any other tape in
  the session; this is N independent single-tape verifications, not a
  cross-tape replay primitive. (This incidentally answers backlog item 65's
  "session-level divergence rollup" as a side effect of the same rollup
  loop.)
- **`session fork`** — session-membership-guarded delegation: verifies
  RUN_ID is reachable within SESSION_ID via `session_tapes`, then calls the
  existing, UNCHANGED top-level `fork` command function directly (Typer's
  `@app.command()` decorator returns the callback unmodified, so this is
  in-process composition, not a second fork implementation). The fork
  itself is still single-tape — RUN_ID's own tape, forked exactly as
  `tracefork fork` already does.
- **`session blame`** — the same guard-then-delegate pattern over the
  existing, UNCHANGED top-level `blame` command function. Still single-tape
  attribution within one run; not the cross-tape attribution described
  above.
- **`session serve`** — validates SESSION_ID exists, prints its
  `/api/session/{id}` deep link, then starts the SAME web UI server
  `tracefork serve` already runs (no new routes — `server.py` already
  exposes `GET /api/session/{session_id}`).

Zero engine-module (`fork.py`/`blame.py`/`replay.py`/`store.py`) changes.
100% of the new logic is CLI-level looping and session-membership
validation over calls that already exist and are already tested.

## Explicit non-goals (this slice)

- **Live/streaming multi-process record capture** — every `session record`
  call operates on ALREADY-STORED tapes; there is no live multi-agent
  recording session, no streaming ingestion, no notion of "attach to a
  running orchestration."
- **Joint cross-tape counterfactual propagation** — mutating tape A's
  response and re-deriving what a spawned tape B would then say. Blocked on
  the `CoalitionForkTransport` generalization above.
- **True cross-tape causal attribution** — attributing a parent's outcome
  to a specific step inside a delegated child tape, with the same
  Wilson/Jeffreys/BH-FDR rigor `blame.py` already gives single-tape
  attribution. Same blocker.
- **A fork-board / multi-lane UI** — `session serve`'s deep link is a
  same-origin JSON endpoint (`/api/session/{id}`), already documented as
  out of scope for `report.html`'s UI itself; no new visualization ships
  here (see backlog item 33, "Multi-agent session replay and fork-board
  UI", P2, separately scoped).

## Sequencing note for future work

The genuine next step toward closing this gap is the refactor named above
— generalizing `CoalitionForkTransport`'s intervention model to an
explicit per-step parent-set that can name a step in a DIFFERENT tape as a
predecessor — sequenced BEFORE backlog item 58 (cross-tape causal blame) or
any joint cross-tape fork primitive is attempted. This spike exists so that
sequencing is written down rather than silently assumed, per the
completeness critic's own recommendation.
