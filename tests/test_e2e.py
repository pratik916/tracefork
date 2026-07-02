"""End-to-end, cross-module integration tests — the capstone bead.

Every other test file in this suite proves ONE module (or one seam between two
modules) in isolation. This file proves the *whole product* composes: a tape
recorded by one seam is the tape saved by the store, loaded by the store,
replayed by the verifier, forked by the fork engine, ranked by the blame
engine, and fed to the self-validation runner — not fresh fixtures re-built at
every stage. Where a real cross-module chain is possible it is used; where the
codebase's own architecture draws a documented boundary (e.g. OpenAI/Gemini
have no SDK client or transport pipeline here; an ingested OTel/OpenInference
tape is blame-only, not bit-exact-replayable; Bedrock's botocore seam is not
wired into ForkEngine/BlameEngine), that boundary is asserted explicitly
rather than papered over — see each section's docstring.

All offline, $0 — no ANTHROPIC_API_KEY, no network, everywhere in this file.
"""

from __future__ import annotations

import asyncio
import json
import threading

import anthropic
import httpx
import pytest
from fastapi import FastAPI

from tests.fakes import (
    AsyncScriptedFakeLLM,
    ScriptedFakeLLM,
    make_text_response,
)
from tracefork.bedrock_transport import BedrockTransport
from tracefork.bench import KNOWN_LIMITATION_CASES, run_bench
from tracefork.blame import BlameEngine, StringMatchOracle, get_oracle, registered_oracles
from tracefork.boundary_guard import BoundaryViolationError
from tracefork.constants import OTEL_INGESTED_BOUNDARY, PROXY_BOUNDARY, SONNET
from tracefork.divergence import diagnose
from tracefork.faults import FAULT_MARKER_BYTES, FaultClass, FaultInjector
from tracefork.fork import BranchSpec, ForkEngine
from tracefork.interop import (
    build_openinference_dataset,
    build_otel_trace,
    ingest_openinference_dataset,
    ingest_otel_trace,
)
from tracefork.matcher import get_matcher, registered_matchers
from tracefork.mcp_client import RecordingMCPSession
from tracefork.nondet import (
    DivergenceError,
    DriftingNondet,
    RecordingNondet,
    ReplayNondet,
    find_divergence,
)
from tracefork.providers import get_adapter, registered_providers
from tracefork.proxy import build_record_app, build_replay_app
from tracefork.recorder import AsyncRecorder, Recorder
from tracefork.redact import safe_defaults
from tracefork.replay import ReplayVerifier
from tracefork.store import TapeStore
from tracefork.synthetic import (
    FakeAWSPreparedRequest,
    FakeEventEmitter,
    ScriptedBedrockSender,
    first_non_none_response,
)
from tracefork.tape import Tape, get_serializer, registered_serializers
from tracefork.tools import ToolTransport, decode_result, make_result_frame, make_tool_call_frame
from tracefork.transport import AsyncTraceforkTransport, TraceforkTransport
from tracefork.validate import ValidationRunner

# ── shared fixture: the pipeline's own two-turn agent + tape ────────────────
#
# Reused across the record/replay/fork/blame/divergence/interop sections below
# so those sections consume a REAL upstream tape, not an independently-built
# fixture per section.

NEUTRAL_RESP = make_text_response("Checking availability")
SUCCESS_RESP = make_text_response("SUCCESS — booking confirmed")
FAIL_RESP = make_text_response("FAIL — no flights available")


def _e2e_agent(client: anthropic.Anthropic) -> str:
    """Two-turn agent; turn2's request embeds turn1's reply text, so mutating
    turn1 changes turn2's request — the property both ForkEngine and
    BlameEngine rely on to produce a genuine counterfactual."""
    r1 = client.messages.create(
        model=SONNET,
        max_tokens=100,
        messages=[{"role": "user", "content": "book a flight to Kyoto"}],
    )
    first = r1.content[0].text
    r2 = client.messages.create(
        model=SONNET,
        max_tokens=100,
        messages=[
            {"role": "user", "content": "book a flight to Kyoto"},
            {"role": "assistant", "content": first},
            {"role": "user", "content": "confirm"},
        ],
    )
    return r2.content[0].text


def _record_e2e_tape() -> Tape:
    fake = ScriptedFakeLLM([NEUTRAL_RESP, SUCCESS_RESP])
    tape = Tape(agent_name="e2e-booking-agent")
    transport = TraceforkTransport("record", tape, fake)
    client = anthropic.Anthropic(
        api_key="sk-ant-fake", http_client=httpx.Client(transport=transport), max_retries=0
    )
    _e2e_agent(client)
    return tape


# ══════════════════════════════════════════════════════════════════════════
# 1. Full pipeline: record → save → load → replay → fork → blame → validate
# ══════════════════════════════════════════════════════════════════════════


def test_anthropic_full_pipeline_record_to_validate(tmp_path):
    """The whole product spine, chained: every stage consumes the PREVIOUS
    stage's real output (not a fresh fixture) — record, persist to SQLite via
    TapeStore, reload, prove the digest is stable across that round trip
    (and across to_bytes/from_bytes too), replay bit-exact, fork a step,
    persist the resulting branch, rank every step by causal blame, and run
    the offline fault-injection self-validation suite."""
    tape = _record_e2e_tape()
    assert len(tape.exchanges) == 2
    recorded_digest = tape.digest()

    store = TapeStore(str(tmp_path / "store.db"))
    try:
        run_id = store.save_tape(tape, run_id="e2e-run")
        loaded = store.load_tape(run_id)
        assert loaded.exchanges == tape.exchanges
        assert loaded.digest() == recorded_digest  # digest stable across save/load
        assert Tape.from_bytes(loaded.to_bytes()).digest() == recorded_digest

        # ── replay: bit-exact, proven (byte-identical + digest match) ──────
        result = ReplayVerifier(loaded, _e2e_agent).verify()
        assert result.bit_exact is True
        assert result.matched == result.total == 2
        assert result.fingerprints_match is True
        assert result.divergence is None

        # ── fork: swap turn1's response, let the SAME agent run forward ────
        spec = BranchSpec(divergence_step=0, mutated_response=FAIL_RESP)
        branch = ForkEngine.fork(
            loaded, spec, _e2e_agent, post_fork_transport=ScriptedFakeLLM([SUCCESS_RESP])
        )
        assert branch.prefix_replayed == 0
        assert branch.tail_recorded == 1
        assert branch.delta_tape.exchanges[0][1] == FAIL_RESP
        assert b"FAIL" in branch.delta_tape.exchanges[1][0]  # tail embedded the mutation

        branch_id = store.save_branch(
            parent_run_id=run_id,
            divergence_step=0,
            delta_tape=branch.delta_tape,
            mutation_desc="turn1 forced to FAIL",
        )
        loaded_branch = store.load_branch(branch_id)
        assert loaded_branch["parent_run_id"] == run_id
        assert loaded_branch["divergence_step"] == 0

        # ── blame: rank every step by causal flip-rate, offline, with CIs ──
        oracle = StringMatchOracle(success_re=r"SUCCESS", failure_re=r"FAIL")

        def perturb_factory(step_idx: int) -> tuple[bytes, object]:
            return FAIL_RESP, ScriptedFakeLLM([SUCCESS_RESP])

        report = BlameEngine.rank(
            loaded, _e2e_agent, oracle, perturb_factory=perturb_factory, k=3, budget_usd=100.0
        )
        assert report.parent_outcome is True
        top = report.top()
        assert top is not None and top.step_index == 1  # the decisive final step
        assert 0.0 <= top.ci_lo <= top.flip_rate <= top.ci_hi <= 1.0
    finally:
        store.close()

    # ── validate: the instrument fingers a genuinely-injected fault, offline
    vreport = ValidationRunner(FaultClass.CORRUPTED_TOOL_OUTPUT, k=2, n_runs=2).run()
    assert vreport.top1_precision == 1.0
    assert vreport.negative_control_max_flip < 0.30


# ══════════════════════════════════════════════════════════════════════════
# 2. Negative control: proof-not-assert (DriftingNondet must still diverge)
# ══════════════════════════════════════════════════════════════════════════


def _nondet_agent(client: anthropic.Anthropic, nondet) -> str:
    v = nondet.random_float()
    resp = client.messages.create(
        model=SONNET, max_tokens=100, messages=[{"role": "user", "content": f"roll: {v.hex()}"}]
    )
    return resp.content[0].text


def _anthropic_client(transport: httpx.BaseTransport) -> anthropic.Anthropic:
    return anthropic.Anthropic(
        api_key="sk-ant-fake", http_client=httpx.Client(transport=transport), max_retries=0
    )


def test_negative_control_drifting_nondet_forces_divergence_through_the_store(tmp_path):
    """The proof-not-assert invariant, exercised through a REAL store round
    trip: replaying with the recorded draws is bit-exact (positive control),
    but replaying with `DriftingNondet` (fresh draws every call) MUST still be
    caught as a divergence — if this ever silently passed, the hash-verified
    replay claim would be vacuous."""
    tape = Tape()
    rec_nondet = RecordingNondet()
    rec_fake = ScriptedFakeLLM([make_text_response("Done.")])
    rec_transport = TraceforkTransport("record", tape, rec_fake)
    _nondet_agent(_anthropic_client(rec_transport), rec_nondet)
    tape.draws = rec_nondet.draws

    store = TapeStore(str(tmp_path / "store.db"))
    try:
        run_id = store.save_tape(tape, run_id="nondet-run")
        loaded = store.load_tape(run_id)
    finally:
        store.close()
    assert loaded.draws == tape.draws

    # positive control: the recorded draws replay bit-exact.
    good_transport = TraceforkTransport("replay", loaded)
    out = _nondet_agent(_anthropic_client(good_transport), ReplayNondet(loaded.draws))
    assert out == "Done."
    assert good_transport.fully_consumed()

    # negative control: fresh (drifting) draws are still caught.
    with pytest.raises(anthropic.APIConnectionError) as exc_info:
        _nondet_agent(_anthropic_client(TraceforkTransport("replay", loaded)), DriftingNondet())
    divergence = find_divergence(exc_info.value)
    assert divergence is not None
    assert isinstance(divergence, DivergenceError)


# ══════════════════════════════════════════════════════════════════════════
# 3. Concurrency determinism: asyncio fan-out records + replays bit-exact
# ══════════════════════════════════════════════════════════════════════════


class _DelayedAnthropicInner(httpx.AsyncBaseTransport):
    """Serves valid Anthropic wire responses with per-request delays, so
    completion order is driven by the delays — the fan-out nondeterminism
    `AsyncTraceforkTransport` must capture and re-impose on replay."""

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if b"alpha" in request.content:
            await asyncio.sleep(0.03)
            body = make_text_response("respA")
        else:
            await asyncio.sleep(0.01)
            body = make_text_response("respB")
        return httpx.Response(200, headers={"content-type": "application/json"}, content=body)


async def _gather_agent(client: anthropic.AsyncAnthropic) -> list[str]:
    async def call(prompt: str) -> str:
        resp = await client.messages.create(
            model=SONNET, max_tokens=100, messages=[{"role": "user", "content": prompt}]
        )
        return resp.content[0].text

    return await asyncio.gather(call("alpha"), call("beta"))


async def test_concurrency_determinism_gather_replays_bit_exact_through_the_store(tmp_path):
    """`asyncio.gather` fan-out through the real SDK, `async_batches` recorded,
    persisted through `TapeStore` (not just to_bytes/from_bytes), reloaded,
    and replayed bit-exact in the recorded completion order."""
    rec_client = anthropic.AsyncAnthropic(
        api_key="sk-ant-fake",
        http_client=httpx.AsyncClient(transport=_DelayedAnthropicInner()),
        max_retries=0,
    )
    async with AsyncRecorder(rec_client, agent_name="e2e-gather") as rec:
        recorded = await _gather_agent(rec.client)
        tape = rec.tape
    assert recorded == ["respA", "respB"]
    assert tape.async_batches == [[0, 1]]

    store = TapeStore(str(tmp_path / "store.db"))
    try:
        run_id = store.save_tape(tape, run_id="gather-run")
        loaded = store.load_tape(run_id)
    finally:
        store.close()
    assert loaded.async_batches == [[0, 1]]
    assert loaded.digest() == tape.digest()  # the batch log is metadata, not content

    replay_transport = AsyncTraceforkTransport("replay", loaded)
    replay_client = anthropic.AsyncAnthropic(
        api_key="sk-ant-replay",
        http_client=httpx.AsyncClient(transport=replay_transport),
        max_retries=0,
    )
    replayed = await _gather_agent(replay_client)
    assert replayed == recorded
    assert replay_transport.fully_consumed()


# ══════════════════════════════════════════════════════════════════════════
# 4. Cross-feature paths
# ══════════════════════════════════════════════════════════════════════════

# ── 4a. MCP / tool record-replay shares a tape with the LLM channel ────────


def test_llm_and_tool_exchanges_share_one_tape_and_both_replay_bit_exact(tmp_path):
    """A single `Tape` genuinely carries two independent channels — LLM
    exchanges (`TraceforkTransport`) and tool-call exchanges (`ToolTransport`,
    the seam MCP and native tools both sit on) — and both persist through
    `TapeStore` together and replay bit-exact independently."""
    tape = Tape(agent_name="mixed-channel-e2e")

    llm_transport = TraceforkTransport(
        "record", tape, ScriptedFakeLLM([make_text_response("looked it up")])
    )
    llm_client = anthropic.Anthropic(
        api_key="sk-ant-fake", http_client=httpx.Client(transport=llm_transport), max_retries=0
    )
    llm_client.messages.create(
        model=SONNET, max_tokens=100, messages=[{"role": "user", "content": "weather?"}]
    )

    call_frame = make_tool_call_frame(1, "get_weather", {"city": "NYC"})
    result_frame = make_result_frame(1, {"content": [{"type": "text", "text": "72F"}]})
    tool_rec = ToolTransport("record", tape, inner=lambda _frame: result_frame)
    served = tool_rec.handle_frame(call_frame)
    assert decode_result(served) == {"content": [{"type": "text", "text": "72F"}]}

    assert len(tape.exchanges) == 1
    assert len(tape.tool_exchanges) == 1

    store = TapeStore(str(tmp_path / "store.db"))
    try:
        run_id = store.save_tape(tape, run_id="mixed-run")
        loaded = store.load_tape(run_id)
    finally:
        store.close()
    assert loaded.digest() == tape.digest()
    assert len(loaded.tool_exchanges) == 1

    def _agent(client: anthropic.Anthropic) -> None:
        client.messages.create(
            model=SONNET, max_tokens=100, messages=[{"role": "user", "content": "weather?"}]
        )

    llm_result = ReplayVerifier(loaded, _agent).verify()
    assert llm_result.bit_exact is True

    tool_rep = ToolTransport("replay", loaded)
    # a rotated JSON-RPC id must not diverge — canonical_frame ignores it.
    replayed = tool_rep.handle_frame(make_tool_call_frame(99, "get_weather", {"city": "NYC"}))
    assert decode_result(replayed) == {"content": [{"type": "text", "text": "72F"}]}
    assert tool_rep.fully_consumed()


async def test_mcp_tool_replay_via_recording_mcp_session():
    """The MCP-specific replay seam (`RecordingMCPSession`) needs NO `mcp`
    install for replay mode — offline, synthetic JSON-RPC frames only."""
    tape = Tape()
    tape.append_tool_exchange(
        make_tool_call_frame(None, "get_weather", {"city": "NYC"}),
        make_result_frame(None, {"content": [{"type": "text", "text": "72F"}]}),
    )
    session = RecordingMCPSession(tape, "replay")
    result = await session.call_tool("get_weather", {"city": "NYC"})
    assert result == {"content": [{"type": "text", "text": "72F"}]}

    with pytest.raises(DivergenceError):
        await session.call_tool("get_weather", {"city": "Paris"})  # mismatched args


# ── 4b. Redaction end-to-end ─────────────────────────────────────────────


def test_redaction_end_to_end_scrubbed_tape_persists_and_replays(tmp_path, monkeypatch):
    """Recorder + a redactor scrub a live secret out of the tape; the SCRUBBED
    (not live) bytes are what `TapeStore` persists, and replay against the
    matcher's redacted fingerprint still verifies — proving redaction,
    recording, storage, and replay compose end-to-end."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-e2e-secret-value")
    fake = ScriptedFakeLLM([make_text_response("your key sk-ant-e2e-secret-value is invalid")])
    client = anthropic.Anthropic(
        api_key="sk-ant-fake", http_client=httpx.Client(transport=fake), max_retries=0
    )
    redactor = safe_defaults()

    with Recorder(client, redactor=redactor) as rec:
        rec.client.messages.create(
            model=SONNET,
            max_tokens=100,
            messages=[{"role": "user", "content": "my key is sk-ant-e2e-secret-value"}],
        )
    tape = rec.tape
    req, resp = tape.exchanges[0]
    assert b"sk-ant-e2e-secret-value" not in req
    assert b"sk-ant-e2e-secret-value" not in resp
    assert tape.content_redacted is False  # metadata-only redaction, not forensic

    store = TapeStore(str(tmp_path / "store.db"))
    try:
        run_id = store.save_tape(tape, run_id="redact-run")
        loaded = store.load_tape(run_id)
    finally:
        store.close()
    assert loaded.exchanges == tape.exchanges  # the scrubbed bytes, not the live secret
    assert loaded.digest() == tape.digest()

    def _agent(client: anthropic.Anthropic) -> str:
        r = client.messages.create(
            model=SONNET,
            max_tokens=100,
            messages=[{"role": "user", "content": "my key is sk-ant-e2e-secret-value"}],
        )
        return r.content[0].text

    result = ReplayVerifier(loaded, _agent, matcher=redactor.matcher()).verify()
    assert result.bit_exact is True


# ── 4c. OTel / OpenInference export → ingest round trip ────────────────────


def test_otel_and_openinference_export_ingest_round_trip_preserves_step_structure():
    """Both interop directions preserve step STRUCTURE (exchange count,
    per-step model), and OpenInference additionally preserves completion TEXT
    (its `output.value` carries it; OTel gen_ai.* spans don't). The documented
    boundary — an ingested tape is blame-by-re-execution only, NOT bit-exact
    replayable — is proven, not just asserted: replaying it against the REAL
    agent diverges immediately."""
    tape = _record_e2e_tape()

    otel_export = build_otel_trace(tape)
    ingested_otel = ingest_otel_trace(otel_export)
    assert ingested_otel.boundary == OTEL_INGESTED_BOUNDARY
    assert len(ingested_otel.exchanges) == len(tape.exchanges) == 2

    oi_export = build_openinference_dataset(tape)
    ingested_oi = ingest_openinference_dataset(oi_export)
    assert len(ingested_oi.exchanges) == 2
    adapter = get_adapter("anthropic")
    texts = [adapter.parse_response(resp).first_text() for _req, resp in ingested_oi.exchanges]
    assert texts == ["Checking availability", "SUCCESS — booking confirmed"]

    result = ReplayVerifier(ingested_otel, _e2e_agent).verify()
    assert result.bit_exact is False
    assert result.divergence is not None


# ── 4d. Plugin registries resolve every built-in by name ───────────────────


def test_plugin_registries_resolve_every_builtin_by_name():
    """provider / matcher / oracle / serializer registries are populated by
    `import tracefork` alone — never entry points by default (see
    `plugins.py`'s security guarantee, tested directly in test_plugins.py).
    The LLM-judge oracle is the one OPT-IN registration: importing
    `tracefork.judge` is itself what adds `\"llm_judge\"`."""
    for name in ("anthropic", "openai", "gemini", "bedrock"):
        assert name in registered_providers()
        assert get_adapter(name) is not None

    for name in ("identity", "gemini", "bedrock", "redacting"):
        assert name in registered_matchers()
        assert get_matcher(name) is not None

    assert "string_match" in registered_oracles()
    assert get_oracle("string_match") is StringMatchOracle

    assert "binary" in registered_serializers()
    assert get_serializer("binary") is not None

    # opt-in: importing tracefork.judge is itself what registers "llm_judge"
    # (may already be registered if another test module imported judge.py first
    # in this process — either way, the postcondition below must hold).
    import tracefork.judge as judge_module

    assert "llm_judge" in registered_oracles()
    assert get_oracle("llm_judge") is judge_module.LLMJudgeOracle


# ── 4e. Nondet + BoundaryGuard: opt-in guard catches a seeded violation ────


def test_boundary_guard_catches_seeded_violation_and_clean_runs_still_replay():
    """The opt-in guard: a thread-spawning agent under `boundary_guard=True`
    hard-errors at RECORD time (loud, not a mysterious later replay failure);
    the same guard, active during a clean recording session, does not
    false-positive — that session records and replays bit-exact."""

    def _violating_agent(client: anthropic.Anthropic) -> None:
        threading.Thread(target=lambda: None).start()

    fake = ScriptedFakeLLM([make_text_response("hi")])
    client = anthropic.Anthropic(
        api_key="sk-ant-fake", http_client=httpx.Client(transport=fake), max_retries=0
    )
    with pytest.raises(BoundaryViolationError), Recorder(client, boundary_guard=True) as rec:
        _violating_agent(rec.client)

    def _clean_agent(client: anthropic.Anthropic) -> str:
        resp = client.messages.create(
            model=SONNET, max_tokens=100, messages=[{"role": "user", "content": "hi"}]
        )
        return resp.content[0].text

    fake2 = ScriptedFakeLLM([make_text_response("hi")])
    client2 = anthropic.Anthropic(
        api_key="sk-ant-fake", http_client=httpx.Client(transport=fake2), max_retries=0
    )
    with Recorder(client2, boundary_guard=True) as rec2:
        out = _clean_agent(rec2.client)
    assert out == "hi"

    result = ReplayVerifier(rec2.tape, _clean_agent).verify()
    assert result.bit_exact is True


# ── 4f. Divergence diagnostics on a genuinely-recorded tape ────────────────


def test_divergence_diagnostics_identify_the_changed_field_on_a_recorded_tape():
    """`diagnose()` against a tape produced by the SAME recorder used
    throughout this file (not a hand-built fixture), pinpointing exactly
    which field a live request changed."""
    tape = _record_e2e_tape()
    recorded_body = tape.exchanges[0][0]
    live_body = recorded_body.replace(b"book a flight to Kyoto", b"cancel my flight")
    assert live_body != recorded_body
    live = httpx.Request("POST", "https://api.anthropic.com/v1/messages", content=live_body)

    diag = diagnose(tape, 0, live)
    assert diag is not None
    assert diag.is_real_divergence is True
    assert any("content" in d.path for d in diag.field_diffs)


# ── 4g. Base-URL record/replay proxy via an ASGI test client ───────────────


def _asgi_client(app: FastAPI) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://proxy-under-test")


async def test_base_url_proxy_record_replay_round_trip_is_bit_exact():
    """The localhost MITM-style proxy (for non-Python clients) driven
    entirely in-process via `httpx.ASGITransport` — no real socket, no real
    upstream (a fake `AsyncScriptedFakeLLM` stands in for it)."""
    tape = Tape()
    upstream = AsyncScriptedFakeLLM([b'{"id":"resp-1"}'])
    record_app = build_record_app(tape, "https://upstream.example", transport=upstream)

    req = b'{"model":"m","messages":[{"role":"user","content":"hi"}]}'
    async with _asgi_client(record_app) as client:
        rec_resp = await client.post("/v1/messages", content=req)
    await record_app.state.proxy.aclose()

    assert rec_resp.status_code == 200
    assert tape.exchanges == [(req, b'{"id":"resp-1"}')]
    # a proxy-recorded tape sits outside the full in-process determinism
    # boundary (no NondetSource behind a non-Python client) — flagged, not hidden.
    assert tape.boundary == PROXY_BOUNDARY

    replay_app = build_replay_app(tape)
    async with _asgi_client(replay_app) as client:
        replay_resp = await client.post("/v1/messages", content=req)
    assert replay_resp.status_code == 200
    assert replay_resp.content == rec_resp.content
    assert replay_app.state.proxy.fully_consumed()


# ── 4h. bench: competing-fault discrimination, documented limitation ───────


def test_bench_competing_fault_discrimination_matches_documented_scope():
    """8/9 planted, causally-distinct faults on one longer tape resolve
    exactly as planted; the one that doesn't (`gate_half_of_conjunction`) is
    a NAMED, explained limitation — surfaced, never hidden as an
    `unexpected_failure`."""
    report = run_bench(k=2, m_samples=1)
    unresolved = [c.name for c in report.cases if not c.resolved]
    assert unresolved == ["gate_half_of_conjunction"]
    assert report.unexpected_failures() == []
    limitation = next(c for c in report.cases if c.name in KNOWN_LIMITATION_CASES)
    assert "LIMITATION" in limitation.note


# ══════════════════════════════════════════════════════════════════════════
# 5. Full pipeline per OTHER provider (OpenAI, Gemini, Bedrock) — honest scope
# ══════════════════════════════════════════════════════════════════════════

# ── OpenAI / Gemini: wire-format + fault-injection only (documented scope) ─


@pytest.mark.parametrize("provider", ["openai", "gemini"])
def test_openai_gemini_wire_level_fault_pipeline_documented_scope(provider):
    """SCOPE, documented not hidden: this repo has no OpenAI/Gemini SDK client
    or transport pipeline (`grep -r 'openai\\.OpenAI(\\|genai\\.'` over `src/`
    turns up nothing) — `Recorder`/`ForkEngine`/`BlameEngine`/`ReplayVerifier`
    all hardcode `anthropic.Anthropic`/`AsyncAnthropic`. What genuinely IS a
    full pipeline for these providers is the wire-format seam: build a
    provider-shaped response via the registered adapter, place it on a tape,
    fault-inject across all 5 classes, and re-parse it with the SAME
    provider's adapter — proving the provider-generic fault seam and the
    provider registry compose end-to-end for non-Anthropic wire shapes too."""
    adapter = get_adapter(provider)
    resp = adapter.build_tool_use_response(
        "check_availability", {"seats": 2, "destination": "Kyoto"}
    )
    tape = Tape()
    tape.append_exchange(b'{"model": "x"}', resp)

    for fc in FaultClass:
        mutated = FaultInjector.inject(tape, 0, fc, provider=provider)
        json.loads(mutated)  # still valid wire JSON
        assert FAULT_MARKER_BYTES in mutated, f"{provider}/{fc} dropped the marker"
        norm = adapter.parse_response(mutated)
        assert norm is not None


# ── Bedrock: record → persist → replay bit-exact (separate, non-httpx seam) ─

_BEDROCK_INVOKE_URL = (
    "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-sonnet-4-6/invoke"
)
_BEDROCK_EVENT_NAME = "before-send.bedrock-runtime.InvokeModel"
_BEDROCK_REQ_BODY = (
    b'{"anthropic_version": "bedrock-2023-05-31", "max_tokens": 100, '
    b'"messages": [{"role": "user", "content": "hi"}]}'
)
_BEDROCK_RESP_BODY = (
    b'{"id": "msg_1", "type": "message", "role": "assistant", '
    b'"model": "anthropic.claude-sonnet-4-6", "content": [{"type": "text", "text": "hello"}], '
    b'"stop_reason": "end_turn", "usage": {"input_tokens": 10, "output_tokens": 5}}'
)


def _bedrock_prepared(
    body: bytes, *, date: str = "20260101T000000Z", token: str = "tok-A"
) -> FakeAWSPreparedRequest:
    return FakeAWSPreparedRequest(
        method="POST",
        url=_BEDROCK_INVOKE_URL,
        headers={
            "content-type": "application/json",
            "authorization": (
                f"AWS4-HMAC-SHA256 Credential=AKIAEXAMPLE/{date[:8]}/us-east-1/"
                f"bedrock/aws4_request, SignedHeaders=content-type;host, Signature=deadbeef"
            ),
            "x-amz-date": date,
            "x-amz-security-token": token,
        },
        body=body,
    )


def test_bedrock_record_replay_pipeline_persists_and_bit_exact_replays(tmp_path):
    """Bedrock's InvokeModel seam is a SEPARATE, non-httpx transport
    (`BedrockTransport`, botocore `before-send`-hook shaped), but it reuses
    `tape.py` completely unchanged: record → to_bytes/from_bytes (AND
    `TapeStore`) → replay bit-exact, tolerating a freshly re-signed SigV4
    request.

    SCOPE, documented not hidden: unlike the Anthropic pipeline in section 1,
    `ForkEngine`/`BlameEngine` build their own `httpx.Client`/
    `anthropic.Anthropic` and never drive Bedrock's botocore `before-send`
    seam — this bead does not claim fork/blame against a Bedrock tape (see
    `bedrock_transport.py`'s module docstring; no existing test in this repo
    forks or blames a Bedrock tape either)."""
    tape = Tape()
    sender = ScriptedBedrockSender([_BEDROCK_RESP_BODY])
    recorder = BedrockTransport("record", tape, sender=sender)
    emitter = FakeEventEmitter()
    recorder.register(emitter)
    result = first_non_none_response(
        emitter.emit(_BEDROCK_EVENT_NAME, request=_bedrock_prepared(_BEDROCK_REQ_BODY))
    )
    assert result is not None
    assert result.content == _BEDROCK_RESP_BODY

    store = TapeStore(str(tmp_path / "store.db"))
    try:
        run_id = store.save_tape(tape, run_id="bedrock-run")
        loaded = store.load_tape(run_id)
    finally:
        store.close()
    assert loaded.digest() == tape.digest()

    replayer = BedrockTransport("replay", loaded)
    replay_emitter = FakeEventEmitter()
    replayer.register(replay_emitter)
    rotated = _bedrock_prepared(_BEDROCK_REQ_BODY, date="20260702T121212Z", token="tok-ROTATED")
    replay_result = first_non_none_response(
        replay_emitter.emit(_BEDROCK_EVENT_NAME, request=rotated)
    )
    assert replay_result is not None
    assert replay_result.content == _BEDROCK_RESP_BODY
    assert replayer.fully_consumed()
