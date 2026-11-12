# tracefork — product plan: built, to build, problems, competitors

*Status as of 2026-11-12. A working document for anyone deciding what to build next. Claims about
tracefork are checked against this repo; claims about other products are sourced and dated, and
the market moves fast, so re-verify them before relying on them externally.*

**Walkthrough video:** [`docs/video/tracefork-walkthrough.mp4`](video/tracefork-walkthrough.mp4)
(7:14, subtitles in [`tracefork-walkthrough.srt`](video/tracefork-walkthrough.srt)). Part 1 records
the real `tracefork serve` UI and real CLI output; Part 2 is clearly labelled mock UI for the
platform described below.

---

## 1. The goal

Grow tracefork from a local developer tool into a **"Sentry for AI agents"**: a product that
tracks agents running in production, lets teams set up and run evals on the platform, and closes
the loop from a production failure to a verified fix.

The differentiator is the existing engine. Most agent-observability tools record a trace and ask
you to read it. tracefork treats a run as a **replayable, forkable artifact**: replay it bit-exact
for $0, fork any step, and rank steps by causal blame with confidence intervals. The product
question is not "what broke?" but "what do I change to make it stop breaking, and can I prove the
fix before shipping it?"

Three pillars, built in this order:

| Pillar | What it means | tracefork's starting point |
|---|---|---|
| **Observe** | Capture production runs; cost, latency, quality; grouped issues; alerts | Capture seam, tape format, OTel `gen_ai.*` export, local report UI |
| **Evaluate** | Datasets, scorers, LLM judges, CI gates, online scoring | LLM-judge oracle with calibration and debiasing, Wilson CIs, regression gates |
| **Improve** | Failure → blame → fork-to-test → eval case → ship | Record/replay/fork/blame engine, working offline today |

---

## 2. What's built today

Everything below works offline, with no API key, and is covered by the test suite. See
`CLAUDE.md` for the per-module detail.

**Core engine**
- **Record:** captures every request/response at the Anthropic SDK's httpx transport (streaming SSE
  and JSON) into a content-addressed tape (sha256 blobs + an ordered, hash-chained event log,
  SQLite-backed, versioned zstd envelope, never pickle).
- **Nondeterminism capture:** `NondetSource` virtualizes clock, uuid, random, env and file reads so
  replays are reproducible.
- **Replay:** no network behind the transport; every request body is hash-checked against the tape.
  A negative control (`DriftingNondet`) must keep failing, so the proof isn't vacuous. Async fan-out
  completion order is recorded and re-imposed.
- **Fork:** three phases (prefix replay at $0 → injected response → fresh tail). Branches carry a
  Merkle-style `branch_digest` and form a DAG in the store.
- **Blame:** per-step flip-rate with Wilson-score CIs, a budget governor, temporal Shapley
  (necessary vs. sufficient), and coalition forks for competing causes.

**Self-validation**
- `tracefork validate`: five injected fault classes, 1.00 top-1, enforced negative control.
- `tracefork bench`: competing causes on one longer tape, 10/11 resolved, with the one miss reported
  as a `[LIMITATION]`, not hidden.
- A Monte Carlo CI-calibration harness and an LLM-judge calibration (Cohen's kappa, Rogan–Gladen
  debiasing).

**Surfaces**
- A Typer CLI with about 30 commands (`replay`, `verify`, `fork`, `blame`, `tournament`, `diff`,
  `receipt`, `export`/`ingest`, `proxy`, `bundle-*`, the `session` sub-app, and more).
- A single-file HTML report and a local FastAPI server (`serve`, bound to 127.0.0.1) with a
  timeline, exchange detail, blame/forks/cost rail, and allowlisted click-to-fork.
- Interop and integrations: OTel / OpenInference export and ingest; OpenAI, Gemini and Bedrock
  provider seams; a localhost record/replay proxy for non-Python clients; adapters for LangChain,
  LangGraph, OpenAI Agents, CrewAI, AutoGen and Google ADK; redaction; trust and release receipts.

---

## 3. What needs to be built

Ordered by the build sequence in §5. Each item names what exists and what's missing.

### Observe
- **Hosted ingest + storage.** Missing entirely. Needs an OTLP-compatible ingest API, a durable
  queue, and three stores: metadata (Postgres), analytics (columnar, e.g. ClickHouse), and tape
  blobs (object storage keyed by sha256). The tape, not the flattened span, stays the unit of
  storage, or replay/fork become impossible.
- **Production SDK.** Today's `Recorder` is a library a developer wires in by hand. A production SDK
  must be fail-open, export asynchronously in batches, sample at the edge, and redact before
  anything leaves the process. Python first; TypeScript next (the proxy covers other languages in
  the meantime).
- **Grouped issues.** A Sentry-style issue model for agents: fingerprint by failing step, tool and
  error class; lifecycle (new / regressed / resolved); frequency and impact ranking. "Failures"
  include semantic ones: low eval score, guardrail block, user thumbs-down.
- **Multi-run dashboards.** Cost per run, user and outcome; latency percentiles; tool success rates.
  `cost_profile.py` covers a single tape today.
- **Alerting and drift.** SLOs on eval pass rates, rate-based alerts, detection of silent model
  updates. `corpus.py`'s z-score regression detection is a starting point.

### Improve (causal debugging in the UI)
- From any issue, one click to: replay free → see the blame ranking → fork the top step with an edited
  response. The engine and a local click-to-fork exist; the hosted, multi-tenant version doesn't.
- A **sandboxed** compute tier for replay and fork. Forking re-executes customer agent code, so it's
  remote code execution by design. `ConfinementSpec` (declared writable roots and allowed hosts)
  is the beginning, but it is not an OS sandbox. This needs gVisor, Firecracker or Landlock-grade
  isolation before any untrusted code runs.

### Evaluate
- **Datasets:** versioned eval cases, with a one-click "turn this failing run into a case" (input +
  expected behavior + scorer). This one hop is what connects Observe to Evaluate.
- **Scorers:** heuristic, schema, tool-call assertions, LLM judge (exists as `judge.py`), human review.
- **CI gate:** run datasets on every prompt or model change; report results with CIs and multiple-
  comparison correction so noise on small sets isn't flagged as regression. The stats exist
  (`proportion_ci`, `benjamini_hochberg`); the gate product doesn't.
- **Online evals:** score a sample of live traffic continuously (cheap heuristics on all traffic, a
  judge on a sample, user feedback).

### Platform
- Auth, organizations and projects, RBAC, SSO; per-tenant isolation, quotas and retention tiers.
- A self-hosted / VPC deployment for customers who can't send prompts to a third party.

---

## 4. Problems and risks

**Technical**
1. **The determinism boundary limits production use.** Bit-exact replay requires one process with
   time and randomness read through `NondetSource`. Real production agents read `datetime.now()`,
   spawn threads, and call tools over the network. Replay *detects* drift rather than hiding it,
   which is correct, but a production run that can't replay bit-exact loses the main
   differentiator. **Open question:** what fraction of real agent runs fit inside the boundary, and can
   tool I/O be captured at the same seam?
2. **Redaction conflicts with replay.** Content redaction breaks bit-exact replay (already flagged
   as forensic-only). Teams that can't store raw prompts get structure-only traces, with no
   replay or fork for those runs.
3. **Blame on real runs costs money.** Free replay covers the prefix; every perturbed tail is a live
   model call. Blame at k=20 across every step of a long run multiplies quickly. The budget governor
   caps it, but product pricing has to absorb it.
4. **Validation uses synthetic fixtures only.** 1.00 top-1 and 10/11 are on planted faults, not real
   traces. Nothing has been run against a public benchmark such as Who&When. That's the right
   honest framing, but customers will ask.
5. **The capture seam is centred on Anthropic.** OpenAI, Gemini and Bedrock seams exist, but
   framework adapters depend on internal attribute names that change between releases.
6. **Storage scale.** Agent traces are tens of KB to MB per run, so retention has to be tiered
   (full tapes for N days, metadata longer, sampled or failing tapes indefinitely).
7. **Test environment.** On uv 0.8.4, five `test_pyproject_packaging.py` tests fail because they
   pass `--clear` to `uv build`, a flag that version lacks (1629 passed, 7 skipped locally). It's
   environmental, but worth pinning a minimum uv version.

**Market**
- **Replay and fork are no longer unique** (see §6). Several open-source projects shipped
  record-and-fork debuggers in 2026. The defensible part is the combination: *hash-verified*
  faithfulness, *statistical* blame (CIs, Shapley, self-validation), and the closed loop into evals.
- **Incumbents are moving into the same space.** Sentry ships Agent Tracing; LangSmith clusters
  production failures into issues and proposes fixes. "Grouped issues for agents" on its own is not
  a moat.

---

## 5. Plan — staged roadmap

Ship one complete turn of the loop, then widen it.

| Phase | Deliverable | Why this order |
|---|---|---|
| **0 · Foundation** | The offline engine: record, replay, fork, blame, validation, local UI. **Done.** | The hard part. |
| **1 · Observe** | Python SDK + hosted ingest/storage + trace viewer, cost dashboards, grouped issues | The familiar install that gets tracefork adopted |
| **2 · Causal debugging** | Replay, blame and fork from any issue, in a sandboxed compute tier | The part log-collection tools can't add without changing what they store |
| **3 · Evaluate** | Datasets, judges, CI gates; failing run → eval case in one click | Becomes part of the development workflow, not just incident response |
| **4 · Close the loop** | Online evals, drift alerts, prompt/version A/B, fork-verified fixes | The dataset grows with every pass |
| **5 · Enterprise** | Self-host/VPC, hardened sandbox, SSO/RBAC, provenance receipts | Moving up-market |

**Suggested next three engineering tasks**
1. Measure the determinism boundary on real agents: instrument 3–5 representative open-source agents
   and report what fraction of runs replay bit-exact, and why the rest don't (directly addresses risk 1).
2. Define the hosted data model: tape-as-unit-of-storage plus the metadata/analytics split, with OTLP
   `gen_ai.*` ingest as the wire format.
3. Specify the production SDK contract (fail-open, async batched export, edge sampling,
   pre-send redaction), reusing `Recorder` and the transport seam.

---

## 6. Competitors (as of 2026-11)

| Product | What it is | Relevance to tracefork |
|---|---|---|
| **LangSmith** (LangChain) | Tracing, evals, prompt tooling; default backend for LangChain/LangGraph 1.0. "LangSmith Engine" clusters production failures into prioritized issues and proposes fixes. Self-hosting is Enterprise-only. | Closest to the "issues + improve" vision; no hash-verified replay. |
| **Langfuse** (acquired by ClickHouse, Jan 2026) | MIT-licensed, self-hostable tracing, evals and prompt management; often described as the lowest-regret default. | Strong open-source incumbent for Observe; trace logging, not replay. |
| **Braintrust** | Eval-first: datasets, scorers, experiments, an AI assistant; OTel and gateway support. | Main competitor for the Evaluate pillar. |
| **Arize Phoenix / AX** | Phoenix is source-available (ELv2) with many built-in eval metrics; online scoring and alerting are in paid AX. | Evals and observability; no fork or blame. |
| **Sentry Agent Tracing** | `gen_ai.*` spans inside existing app traces; conversation replay *view*, dashboards, cost, MCP monitoring; content capture opt-in. | The literal "Sentry for agents" incumbent. Its replay is a view of a recorded conversation, not a re-execution. |
| **Datadog LLM Observability, Opik, Laminar, Galileo, Latitude, MLflow** | Observability and eval platforms with varying focus. | The wider field. |
| **Helicone, AgentOps** | Gateway / agent monitoring; reported to be in maintenance mode or slowed in 2026. | Leaves room for newcomers. |
| **Open-source replay/fork debuggers** — agent-replay, Retrace, Rewind (on Burr), AgentReplay, react-agent-debugger | Record agent runs, replay them, and fork steps to test fixes, mostly local and SQLite-backed. | **Direct overlap with the core engine.** tracefork's edge is hash-verified fidelity with an enforced negative control, plus causal blame with calibrated CIs and self-validation. |
| **Durable execution** (Temporal; LangGraph "time travel") | Temporal replays event history at zero cost; LangGraph re-executes from checkpoints (resumable, not deterministic). | Adjacent. A possible integration target, not just a competitor. |
| **Research:** CausalFlow, Causal Agent Replay (2026) | Interventional, counterfactual blame on agent traces. | Same thesis as tracefork's blame engine; confirms the approach, raises the bar for rigor. |

**Positioning in one sentence:** tracing tools show you what happened; replay debuggers let you
re-run it; tracefork *proves* the replay is faithful, *measures* which step caused the failure with
calibrated uncertainty, and turns that failure into a regression test.

---

## 7. Ideas backlog (unprioritized)

- **Fork-verified fixes in PRs:** a CI check that re-runs the exact production failures a PR claims
  to fix, by forking their tapes, and reports before/after eval results.
- **Blame-guided dataset curation:** automatically propose eval cases from the top-blamed step of
  each high-impact issue.
- **Model-upgrade simulator:** replay a corpus of production tapes, forking every LLM step onto a
  candidate model, and diff the outcomes before switching providers.
- **Chaos mode for agents:** reorder concurrent tool completions (`chaos_release_order` already
  exists) to surface race and ordering bugs.
- **Provenance for regulated industries:** signed, hash-chained trust receipts as an audit trail
  of what an agent did.
- **Temporal / durable-execution integration:** use event history as a capture source for
  multi-process agents that fall outside today's determinism boundary.
- **Multi-agent session view:** cross-tape blame and session fork boards (partly built in
  `cross_tape_blame.py` / `report_session.py`).

---

## Sources (competitor and market claims, retrieved 2026-11-12)

- [Top LLM Observability and Evaluation Platforms in 2026 — MarkTechPost](https://www.marktechpost.com/2026/08/09/top-llm-observability-and-evaluation-platforms-in-2026-langfuse-langsmith-braintrust-arize-and-more-compared/)
- [Best LLM Observability Tools 2026 — AgentsCamp](https://agentscamp.com/guides/comparisons/best-llm-observability-tools-2026)
- [Best AI agent observability tools in 2026 — Latitude](https://latitude.so/blog/best-ai-agent-observability-tools-2026-comparison)
- [Top 5 LLM and Agent Observability Tools in 2026 — MLflow](https://mlflow.org/top-5-agent-observability-tools/)
- [Best AI Observability Tools for Agents (2026) — Arize](https://arize.com/blog/best-ai-observability-tools-for-autonomous-agents-in-2026/)
- [ClickHouse welcomes Langfuse — ClickHouse](https://clickhouse.com/blog/clickhouse-acquires-langfuse-open-source-llm-observability)
- [Langfuse joins ClickHouse — Langfuse](https://langfuse.com/blog/joining-clickhouse)
- [Agents — Sentry Docs](https://docs.sentry.io/ai/monitoring/)
- [Introducing Sentry's Updated Agent Tracing — Sentry Blog](https://blog.sentry.io/sentrys-updated-agent-monitoring/)
- [Agentic Workflow Replay and Time-Travel Debugging — Zylos Research](https://zylos.ai/research/2026-06-20-agentic-workflow-replay-time-travel-debugging/)
- [agent-replay — GitHub](https://github.com/clay-good/agent-replay)
- [agent-time-travel-debugger (Retrace) — GitHub](https://github.com/Amritha902/agent-time-travel-debugger)
- [react-agent-debugger — GitHub](https://github.com/cookiespiggy/react-agent-debugger)
- [Time-Travel Debugging for LLM Agents: Burr's Counterfactual Replay — DEV](https://dev.to/mech_app_ai/time-travel-debugging-for-llm-agents-burrs-counterfactual-replay-architecture-fno)
- [AgentReplay: Deterministic Replay and Counterfactual Debugging — Victor Ndunda](https://victorndunda.com/blog/articles/agentreplay-deterministic-replay.html)
