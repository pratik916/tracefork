"""OTel GenAI / OpenInference interop tests — export, ingest, and the
proven (not just documented) blame-only/not-bit-exact-replay boundary of an
ingested tape. All offline, no API keys, no `opentelemetry-sdk`/`structlog`
install required for the export/ingest path (see `interop.py`'s docstring).
"""

import json

import anthropic
import httpx
import pytest

from tests.fakes import ScriptedFakeLLM, make_text_response
from tracefork.blame import BlameReport, CIMethod, FlipRateResult
from tracefork.constants import GENAI_SEMCONV_VERSION, OTEL_INGESTED_BOUNDARY, SONNET
from tracefork.fork import BranchSpec, ForkEngine
from tracefork.interop import (
    ATTR_BLAME_FLIP_RATE,
    ATTR_CONVERSATION_ID,
    ATTR_OPERATION_NAME,
    ATTR_PROVIDER_NAME,
    ATTR_REQUEST_MODEL,
    ATTR_STEP_INDEX,
    ATTR_TOOL_NAME,
    ATTR_TOOL_STEP_INDEX,
    ATTR_USAGE_INPUT_TOKENS,
    OI_INPUT_VALUE,
    OI_MODEL_NAME,
    OI_OUTPUT_VALUE,
    OI_SPAN_KIND,
    OtlpExportError,
    _flatten_attrs,
    blame_report_from_json,
    build_openinference_dataset,
    build_otel_trace,
    ingest_openinference_dataset,
    ingest_otel_trace,
    normalized_to_genai_attributes,
    push_otlp_trace,
)
from tracefork.nondet import find_divergence
from tracefork.providers.base import ContentPart, NormalizedResponse
from tracefork.replay import ReplayVerifier
from tracefork.tape import Tape
from tracefork.tools import make_result_frame, make_tool_call_frame
from tracefork.transport import TraceforkTransport

RESP_A = make_text_response("Response A")
RESP_B = make_text_response("Response B")


def _build_two_turn_tape() -> Tape:
    fake = ScriptedFakeLLM([RESP_A, RESP_B])
    tape = Tape(agent_name="interop-test-agent")
    transport = TraceforkTransport("record", tape, fake)
    client = anthropic.Anthropic(
        api_key="sk-ant-fake", http_client=httpx.Client(transport=transport), max_retries=0
    )
    client.messages.create(
        model=SONNET, max_tokens=100, messages=[{"role": "user", "content": "turn1"}]
    )
    client.messages.create(
        model=SONNET, max_tokens=100, messages=[{"role": "user", "content": "turn2"}]
    )
    return tape


def _spans(export: dict) -> list[dict]:
    return export["resourceSpans"][0]["scopeSpans"][0]["spans"]


def _attr(span: dict, key: str):
    return _flatten_attrs(span["attributes"])[key]


# ── gen_ai.* attribute mapping ──────────────────────────────────────────────


def test_normalized_to_genai_attributes_maps_core_fields():
    normalized = NormalizedResponse(
        model=SONNET,
        content=(ContentPart(type="text", text="hi"),),
        input_tokens=12,
        output_tokens=34,
        finish_reason="end_turn",
        message_id="msg_abc",
    )
    attrs = normalized_to_genai_attributes(normalized, provider="anthropic")
    assert attrs[ATTR_PROVIDER_NAME] == "anthropic"
    assert attrs[ATTR_OPERATION_NAME] == "chat"
    assert attrs[ATTR_REQUEST_MODEL] == SONNET
    assert attrs["gen_ai.response.model"] == SONNET
    assert attrs["gen_ai.response.id"] == "msg_abc"
    assert attrs["gen_ai.response.finish_reasons"] == ["end_turn"]
    assert attrs[ATTR_USAGE_INPUT_TOKENS] == 12
    assert attrs["gen_ai.usage.output_tokens"] == 34


def test_normalized_to_genai_attributes_request_model_overrides_when_response_has_none():
    normalized = NormalizedResponse()
    attrs = normalized_to_genai_attributes(normalized, provider="anthropic", request_model=SONNET)
    assert attrs[ATTR_REQUEST_MODEL] == SONNET
    assert "gen_ai.response.model" not in attrs


# ── export: OTel GenAI trace ─────────────────────────────────────────────────


def test_build_otel_trace_shape_and_gen_ai_attributes():
    tape = _build_two_turn_tape()
    export = build_otel_trace(tape)

    spans = _spans(export)
    assert len(spans) == 3  # 1 root + 2 exchanges
    root, s0, s1 = spans
    assert root["parentSpanId"] == ""
    assert s0["parentSpanId"] == root["spanId"]
    assert s1["parentSpanId"] == root["spanId"]

    assert _attr(s0, ATTR_PROVIDER_NAME) == "anthropic"
    assert _attr(s0, ATTR_OPERATION_NAME) == "chat"
    assert _attr(s0, ATTR_REQUEST_MODEL) == SONNET
    assert _attr(s0, ATTR_STEP_INDEX) == 0
    assert _attr(s1, ATTR_STEP_INDEX) == 1

    resource_attrs = _flatten_attrs(export["resourceSpans"][0]["resource"]["attributes"])
    assert resource_attrs["gen_ai.semconv.version"] == GENAI_SEMCONV_VERSION
    assert resource_attrs["service.name"] == "tracefork"


def test_build_otel_trace_never_emits_the_deprecated_gen_ai_system_attribute():
    """tracefork-sis.50: `gen_ai.system` was replaced by `gen_ai.provider.name`
    in the OTel GenAI semantic conventions -- exported spans must use the
    current name only, never the deprecated one."""
    tape = _build_two_turn_tape()
    export = build_otel_trace(tape)
    for span in _spans(export):
        assert "gen_ai.system" not in _flatten_attrs(span["attributes"])


def test_build_otel_trace_emits_tool_exchanges_as_their_own_execute_tool_spans():
    """tracefork-sis.50: `tape.tool_exchanges` must no longer be silently
    dropped -- each gets its own span with `gen_ai.operation.name =
    execute_tool` (not the hardcoded "chat" every span got before) and
    `gen_ai.tool.name` decoded off the real `tools/call` JSON-RPC wire shape
    (`tools.py`'s own frame constructors, never a hand-rolled JSON string)."""
    tape = _build_two_turn_tape()
    tape.append_tool_exchange(
        make_tool_call_frame(1, "read_file", {"path": "/etc/hosts"}),
        make_result_frame(1, {"content": "127.0.0.1 localhost"}),
    )
    tape.append_tool_exchange(
        make_tool_call_frame(2, "write_file", {"path": "/tmp/out.txt"}),
        make_result_frame(2, {"ok": True}),
    )

    export = build_otel_trace(tape)
    spans = _spans(export)
    # 1 root + 2 LLM exchanges + 2 tool exchanges.
    assert len(spans) == 5
    root = spans[0]
    tool_spans = spans[3:]

    assert _attr(tool_spans[0], ATTR_OPERATION_NAME) == "execute_tool"
    assert _attr(tool_spans[0], ATTR_TOOL_NAME) == "read_file"
    assert _attr(tool_spans[0], ATTR_TOOL_STEP_INDEX) == 0
    assert _attr(tool_spans[1], ATTR_OPERATION_NAME) == "execute_tool"
    assert _attr(tool_spans[1], ATTR_TOOL_NAME) == "write_file"
    assert _attr(tool_spans[1], ATTR_TOOL_STEP_INDEX) == 1
    for tool_span in tool_spans:
        assert tool_span["parentSpanId"] == root["spanId"]
        assert "execute_tool" in tool_span["name"]

    # Every span id stays unique -- tool spans must not collide with LLM ones.
    span_ids = [s["spanId"] for s in spans]
    assert len(span_ids) == len(set(span_ids))


def test_build_otel_trace_tool_exchange_with_unparseable_frame_falls_back_gracefully():
    """A `tool_exchanges` entry that isn't a well-formed `tools/call` request
    must still get a span (never dropped, never a crash) -- with an honest
    'unknown' tool name rather than a fabricated one."""
    tape = _build_two_turn_tape()
    tape.append_tool_exchange(b"not json", b"also not json")

    export = build_otel_trace(tape)
    spans = _spans(export)
    assert len(spans) == 4
    assert _attr(spans[3], ATTR_OPERATION_NAME) == "execute_tool"
    assert _attr(spans[3], ATTR_TOOL_NAME) == "unknown"


def test_build_otel_trace_sets_conversation_id_when_session_present():
    """tracefork-sis.50: `gen_ai.conversation.id` must be set on every span
    (root, LLM-exchange, and tool-exchange) when a `session_id` is passed --
    and must be absent entirely when omitted (additive, opt-in default)."""
    tape = _build_two_turn_tape()
    tape.append_tool_exchange(
        make_tool_call_frame(1, "read_file", {"path": "/etc/hosts"}),
        make_result_frame(1, {"content": "127.0.0.1 localhost"}),
    )

    without_session = build_otel_trace(tape)
    for span in _spans(without_session):
        assert ATTR_CONVERSATION_ID not in _flatten_attrs(span["attributes"])

    with_session = build_otel_trace(tape, session_id="sess-abc123")
    for span in _spans(with_session):
        assert _attr(span, ATTR_CONVERSATION_ID) == "sess-abc123"


def test_build_otel_trace_is_deterministic():
    """Trace/span ids are content-derived, not random — export is exact-equality
    reproducible for the same tape (no float dust, no nondeterministic ids)."""
    tape = _build_two_turn_tape()
    first = json.dumps(build_otel_trace(tape), sort_keys=True)
    second = json.dumps(build_otel_trace(tape), sort_keys=True)
    assert first == second


def test_build_otel_trace_attaches_blame_attributes():
    tape = _build_two_turn_tape()
    result = FlipRateResult(
        step_index=0,
        flip_rate=0.8,
        ci_lo=0.4,
        ci_hi=0.95,
        flips=8,
        trials=10,
        valid_trials=10,
        responsible=True,
        q_value=0.02,
    )
    report = BlameReport(results=[result], k=10, total_forks=20, ci_method=CIMethod.WILSON)

    export = build_otel_trace(tape, blame=report)
    _, s0, s1 = _spans(export)

    assert _attr(s0, ATTR_BLAME_FLIP_RATE) == 0.8
    assert _attr(s0, "tracefork.blame.responsible") is True
    assert _attr(s0, "tracefork.blame.ci_method") == "wilson"
    with pytest.raises(KeyError):
        _attr(s1, ATTR_BLAME_FLIP_RATE)  # step 1 wasn't in the blame report


# ── push: OTel trace -> a live OTLP/HTTP collector ──────────────────────────


def test_push_otlp_trace_posts_the_exact_trace_body_to_v1_traces():
    """tracefork-sis.60: `push_otlp_trace` POSTs the trace dict verbatim (no
    re-encoding) to `{endpoint}/v1/traces` with `Content-Type:
    application/json` -- proven against a local fake collector
    (`httpx.MockTransport`), fully offline."""
    tape = _build_two_turn_tape()
    trace = build_otel_trace(tape)
    received: dict = {}

    def fake_collector(request: httpx.Request) -> httpx.Response:
        received["url"] = str(request.url)
        received["content_type"] = request.headers.get("content-type")
        received["body"] = json.loads(request.content)
        return httpx.Response(200, json={"partialSuccess": {}})

    response = push_otlp_trace(
        trace, "http://collector.example:4318", transport=httpx.MockTransport(fake_collector)
    )

    assert response.status_code == 200
    assert received["url"] == "http://collector.example:4318/v1/traces"
    assert received["content_type"] == "application/json"
    assert received["body"] == trace
    assert len(received["body"]["resourceSpans"][0]["scopeSpans"][0]["spans"]) == 3


def test_push_otlp_trace_strips_a_trailing_slash_from_endpoint():
    received: dict = {}

    def fake_collector(request: httpx.Request) -> httpx.Response:
        received["url"] = str(request.url)
        return httpx.Response(200, json={})

    push_otlp_trace(
        {"resourceSpans": []},
        "http://collector.example:4318/",
        transport=httpx.MockTransport(fake_collector),
    )
    assert received["url"] == "http://collector.example:4318/v1/traces"


def test_push_otlp_trace_raises_otlp_export_error_on_non_2xx_response():
    def fake_collector(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="collector on fire")

    with pytest.raises(OtlpExportError, match="500"):
        push_otlp_trace(
            {"resourceSpans": []},
            "http://collector.example:4318",
            transport=httpx.MockTransport(fake_collector),
        )


def test_push_otlp_trace_wraps_a_transport_level_failure_as_otlp_export_error():
    """A connection failure must never surface as a raw `httpx` exception --
    only `OtlpExportError`, so `cli.py` can catch exactly one type."""

    def fake_collector(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(OtlpExportError, match="could not reach"):
        push_otlp_trace(
            {"resourceSpans": []},
            "http://collector.example:4318",
            transport=httpx.MockTransport(fake_collector),
        )


# ── export: OpenInference dataset ───────────────────────────────────────────


def test_build_openinference_dataset_shape():
    tape = _build_two_turn_tape()
    dataset = build_openinference_dataset(tape)

    assert dataset["dataset_name"] == "interop-test-agent"
    examples = dataset["examples"]
    assert len(examples) == 2
    ex0 = examples[0]
    assert ex0["metadata"][OI_SPAN_KIND] == "LLM"
    assert ex0["metadata"][OI_MODEL_NAME] == SONNET
    assert ex0["metadata"]["llm.provider"] == "anthropic"
    assert ex0[OI_OUTPUT_VALUE] == "Response A"
    assert "turn1" in ex0[OI_INPUT_VALUE]


def test_build_openinference_dataset_attaches_blame_attributes():
    tape = _build_two_turn_tape()
    result = FlipRateResult(
        step_index=1,
        flip_rate=0.1,
        ci_lo=0.0,
        ci_hi=0.3,
        flips=1,
        trials=10,
        valid_trials=10,
        responsible=False,
    )
    report = BlameReport(results=[result], k=10, total_forks=20)

    dataset = build_openinference_dataset(tape, blame=report)
    assert dataset["examples"][1]["metadata"][ATTR_BLAME_FLIP_RATE] == 0.1
    assert ATTR_BLAME_FLIP_RATE not in dataset["examples"][0]["metadata"]


# ── blame_report_from_json ───────────────────────────────────────────────────


def test_blame_report_from_json_round_trips_cli_output_shape():
    """Mirrors the JSON `tracefork blame` writes to blame_<run_id>.json."""
    data = {
        "k": 5,
        "ci_method": "jeffreys",
        "confidence": 0.9,
        "null_flip_rate": 0.05,
        "fdr_q": 0.1,
        "responsible_set": [2],
        "results": [
            {
                "step_index": 2,
                "flip_rate": 0.75,
                "ci_lo": 0.3,
                "ci_hi": 0.95,
                "valid_trials": 8,
                "undefined": 2,
                "divergences": 1,
                "divergence_rate": 0.2,
                "trustworthy": True,
                "p_value": 0.01,
                "q_value": 0.02,
                "responsible": True,
                "interpretation": "decisive — this step caused it",
            }
        ],
    }
    report = blame_report_from_json(data)
    assert report.ci_method is CIMethod.JEFFREYS
    assert report.k == 5
    assert report.responsible_set == [2]
    assert len(report.results) == 1
    r = report.results[0]
    assert r.step_index == 2
    assert r.flip_rate == 0.75
    assert r.responsible is True


def test_blame_report_from_json_recomputes_q_value_from_p_value_not_forged_fields():
    """`blame_report_from_json` must recompute q_value/responsible/responsible_set
    via `benjamini_hochberg` on the decoded p_values, rather than trusting the
    JSON's own (potentially forged, independently of p_value) fields — the one
    boundary where those could be forged without touching p_value."""
    data = {
        "k": 5,
        "ci_method": "wilson",
        "confidence": 0.95,
        "null_flip_rate": 0.05,
        "fdr_q": 0.10,
        # Forged: claims step 5 (not step 0) is the responsible one.
        "responsible_set": [5],
        "results": [
            {
                # Genuinely significant (tiny p_value) but the JSON's own
                # q_value/responsible forge the OPPOSITE conclusion.
                "step_index": 0,
                "flip_rate": 0.9,
                "ci_lo": 0.5,
                "ci_hi": 1.0,
                "valid_trials": 10,
                "trustworthy": True,
                "p_value": 0.001,
                "q_value": 0.9,
                "responsible": False,
            },
            {
                # NOT significant (p_value == 1.0) but the JSON's own
                # q_value/responsible forge a false-positive "responsible" claim.
                "step_index": 5,
                "flip_rate": 0.05,
                "ci_lo": 0.0,
                "ci_hi": 0.3,
                "valid_trials": 10,
                "trustworthy": True,
                "p_value": 1.0,
                "q_value": 0.001,
                "responsible": True,
            },
        ],
    }
    report = blame_report_from_json(data)
    r0 = next(r for r in report.results if r.step_index == 0)
    r5 = next(r for r in report.results if r.step_index == 5)

    assert r0.responsible is True
    assert r0.q_value < 0.01
    assert r5.responsible is False
    assert r5.q_value == 1.0
    assert report.responsible_set == [0]


# ── ingest: step-structure only, blame-by-re-execution ──────────────────────


def test_ingest_otel_trace_builds_step_structure():
    tape = _build_two_turn_tape()
    export = build_otel_trace(tape)

    ingested = ingest_otel_trace(export)

    assert ingested.boundary == OTEL_INGESTED_BOUNDARY
    assert len(ingested.exchanges) == len(tape.exchanges) == 2
    for req, _resp in ingested.exchanges:
        assert json.loads(req)["model"] == SONNET
        assert json.loads(req)["messages"] == []  # original prompt is NOT recoverable


def test_ingest_openinference_dataset_preserves_completion_text():
    """Unlike the OTel gen_ai.* path, OpenInference's output.value DOES carry
    completion text, so ingest can rebuild a response with the real text."""
    tape = _build_two_turn_tape()
    dataset = build_openinference_dataset(tape)

    ingested = ingest_openinference_dataset(dataset)

    assert ingested.boundary == OTEL_INGESTED_BOUNDARY
    assert len(ingested.exchanges) == 2
    from tracefork.providers import get_adapter

    adapter = get_adapter("anthropic")
    texts = [adapter.parse_response(resp).first_text() for _req, resp in ingested.exchanges]
    assert texts == ["Response A", "Response B"]


def test_ingested_tape_is_not_bit_exact_replayable():
    """Proves the documented boundary: an ingested tape's request bytes are a
    synthesized placeholder, so replaying it against a real agent diverges on
    the very first step — this is expected, not a bug."""
    tape = _build_two_turn_tape()
    ingested = ingest_otel_trace(build_otel_trace(tape))

    def real_agent(client: anthropic.Anthropic) -> str:
        resp = client.messages.create(
            model=SONNET, max_tokens=100, messages=[{"role": "user", "content": "turn1"}]
        )
        return resp.content[0].text

    result = ReplayVerifier(ingested, real_agent).verify()
    assert result.bit_exact is False
    assert result.divergence is not None


def test_ingested_tape_prefix_replay_diverges_on_fork():
    """Same boundary, proven through ForkEngine's $0 prefix-replay assertion."""
    tape = _build_two_turn_tape()
    ingested = ingest_otel_trace(build_otel_trace(tape))

    def real_agent(client: anthropic.Anthropic) -> str:
        client.messages.create(
            model=SONNET, max_tokens=100, messages=[{"role": "user", "content": "turn1"}]
        )
        r2 = client.messages.create(
            model=SONNET, max_tokens=100, messages=[{"role": "user", "content": "turn2"}]
        )
        return r2.content[0].text

    spec = BranchSpec(divergence_step=1, mutated_response=RESP_B)
    with pytest.raises(Exception) as exc_info:
        ForkEngine.fork(ingested, spec, real_agent, post_fork_transport=ScriptedFakeLLM([]))
    # `find_divergence` unwraps a DivergenceError even if the Anthropic SDK
    # buried it inside an APIConnectionError — the same recovery `blame.py`'s
    # `_run_trial` relies on to tell a genuine divergence from a real error.
    assert find_divergence(exc_info.value) is not None
