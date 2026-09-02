"""OTel GenAI / OpenInference interop: export tape + blame data, ingest spans back.

Every incumbent observability stack speaks either the OpenTelemetry **GenAI**
semantic conventions (`gen_ai.*` span attributes) or **OpenInference**
(`llm.*` / `openinference.*` attributes, used by Arize Phoenix and friends).
This module is the seam between tracefork's tape/blame data and both, in two
directions — export and ingest — built entirely on the provider-neutral
`NormalizedResponse` seam in `providers/base.py`, never a hardcoded wire shape.

**Export** (`build_otel_trace`, `build_openinference_dataset`) turn a recorded
`Tape` — optionally with a `BlameReport` — into plain, JSON-serializable
dicts: an OTLP/JSON-shaped trace (`resourceSpans[].scopeSpans[].spans[]`) with
`gen_ai.*` attributes, or an OpenInference-shaped dataset (`examples[]` with
`llm.*` attributes). Neither needs `opentelemetry-sdk` or `openinference-*`
installed to produce or consume — they are plain dicts, not SDK objects. (Real
OTel *spans* emitted live, from tracefork's own process, are a separate
concern — see `observability.py`'s opt-in self-instrumentation.)

**Ingest** (`ingest_otel_trace`, `ingest_openinference_dataset`) go the other
way: given spans/a dataset exported by ANY system that speaks these
attributes (not just tracefork's own export), reconstruct a `Tape`'s *step
structure* — one exchange per LLM span, in recorded order, with a synthesized
request (model id only) and a synthesized response (rebuilt via the provider
adapter's `build_text_response`, carrying the completion text + token counts
the span reported).

IMPORTANT — an ingested tape supports blame-by-re-execution, NOT $0
bit-exact replay:

* `Tape.digest()`, bit-exact replay (`ReplayVerifier`), and the $0
  prefix-replay phase of `ForkEngine.fork()` all depend on the RAW, EXACT
  request/response bytes tracefork itself recorded, plus every
  `NondetSource` draw (clock/uuid/random) that produced them. A trace
  exported by another system carries neither: no raw bytes (only span
  attributes, which don't capture the original prompt) and no captured
  nondeterminism.
* `Tape.boundary` is set to `OTEL_INGESTED_BOUNDARY` (see `constants.py`) to
  mark this explicitly, distinct from `BOUNDARY_V1`. Feeding an ingested tape
  to `ReplayVerifier` or `ForkEngine.fork()` against a real agent will
  (correctly) diverge on the very first prefix request, because the
  reconstructed request bytes are a synthesized placeholder (`{"model": ...,
  "messages": []}`), never what the agent actually sent — see
  `tests/test_interop.py` for this proven, not just asserted.
* What DOES work: the *step structure* — step count, per-step model,
  approximate token usage, and completion text — is enough to identify
  candidate steps (and, if the ingested spans carry tracefork's own
  `tracefork.blame.*` vendor attributes, their flip-rate/CI) for read-only
  inspection or reporting, or to drive a *live re-execution* blame strategy
  where the agent is genuinely re-run end to end (real API calls, not $0)
  rather than replayed from recorded bytes.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import httpx

from .blame import BlameReport, CIMethod, FlipRateResult, benjamini_hochberg
from .constants import GENAI_SEMCONV_VERSION, OTEL_INGESTED_BOUNDARY
from .errors import TraceforkError
from .providers import get_adapter
from .providers.base import NormalizedResponse
from .store import TapeStore
from .tape import Tape, sha256_hex

# ── gen_ai.* attribute names (OTel GenAI semantic conventions) ─────────────
# https://github.com/open-telemetry/semantic-conventions-genai — version
# pinned as GENAI_SEMCONV_VERSION in constants.py. `gen_ai.system` (the
# original attribute name) was replaced by `gen_ai.provider.name` upstream;
# this module targets ONLY the current name -- see GENAI_SEMCONV_VERSION's
# own docstring in constants.py for the release that stabilized it.
ATTR_PROVIDER_NAME = "gen_ai.provider.name"
ATTR_OPERATION_NAME = "gen_ai.operation.name"
ATTR_REQUEST_MODEL = "gen_ai.request.model"
ATTR_RESPONSE_MODEL = "gen_ai.response.model"
ATTR_RESPONSE_ID = "gen_ai.response.id"
ATTR_RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"
ATTR_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
ATTR_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
# Tool-call span attributes (`gen_ai.execute_tool.internal` span type).
ATTR_TOOL_NAME = "gen_ai.tool.name"
# Correlates every span belonging to one conversation/session; set only when
# a caller passes `build_otel_trace(..., session_id=...)` (see below).
ATTR_CONVERSATION_ID = "gen_ai.conversation.id"

# tracefork-specific (vendor-namespaced, never collides with a semconv name).
ATTR_STEP_INDEX = "tracefork.step_index"
# `tape.tool_exchanges` has its own index space, separate from
# `tape.exchanges`'s ATTR_STEP_INDEX -- reusing one name across two
# differently-ordered lists would let a consumer wrongly correlate a tool
# span with an unrelated LLM-exchange step.
ATTR_TOOL_STEP_INDEX = "tracefork.tool_step_index"
ATTR_BLAME_FLIP_RATE = "tracefork.blame.flip_rate"
ATTR_BLAME_CI_LO = "tracefork.blame.ci_lo"
ATTR_BLAME_CI_HI = "tracefork.blame.ci_hi"
ATTR_BLAME_CI_METHOD = "tracefork.blame.ci_method"
ATTR_BLAME_Q_VALUE = "tracefork.blame.q_value"
ATTR_BLAME_RESPONSIBLE = "tracefork.blame.responsible"

# ── OpenInference attribute names (public semantic conventions used by
# Arize Phoenix and compatible tooling) — approximated here as attribute
# *names* attached to a dataset-example shape, not validated against the
# `openinference-semantic-conventions` package's exact schema (which is not a
# tracefork dependency, optional or otherwise); see the module docstring.
OI_SPAN_KIND = "openinference.span.kind"
OI_MODEL_NAME = "llm.model_name"
OI_PROVIDER = "llm.provider"
OI_TOKEN_PROMPT = "llm.token_count.prompt"
OI_TOKEN_COMPLETION = "llm.token_count.completion"
OI_INPUT_VALUE = "input.value"
OI_OUTPUT_VALUE = "output.value"


def normalized_to_genai_attributes(
    normalized: NormalizedResponse,
    *,
    provider: str,
    request_model: str | None = None,
    operation_name: str = "chat",
) -> dict[str, Any]:
    """Map a provider-neutral `NormalizedResponse` to a `gen_ai.*` attribute dict.

    This is the adopted-naming seam item 1 of this module asks for: every
    consumer that wants `gen_ai.*`-shaped data goes through here rather than
    reading one provider's JSON directly. `operation_name` defaults to
    `"chat"` (an LLM inference call, the only kind this function has ever
    described); `build_otel_trace`'s tool-exchange spans pass
    `"execute_tool"` instead, so operation.name is no longer hardcoded
    regardless of what actually happened.
    """
    attrs: dict[str, Any] = {ATTR_PROVIDER_NAME: provider, ATTR_OPERATION_NAME: operation_name}
    model = request_model or normalized.model
    if model:
        attrs[ATTR_REQUEST_MODEL] = model
    if normalized.model:
        attrs[ATTR_RESPONSE_MODEL] = normalized.model
    if normalized.message_id:
        attrs[ATTR_RESPONSE_ID] = normalized.message_id
    if normalized.finish_reason:
        attrs[ATTR_RESPONSE_FINISH_REASONS] = [normalized.finish_reason]
    if normalized.input_tokens is not None:
        attrs[ATTR_USAGE_INPUT_TOKENS] = normalized.input_tokens
    if normalized.output_tokens is not None:
        attrs[ATTR_USAGE_OUTPUT_TOKENS] = normalized.output_tokens
    return attrs


def _blame_attributes(result: FlipRateResult, report: BlameReport) -> dict[str, Any]:
    return {
        ATTR_BLAME_FLIP_RATE: result.flip_rate,
        ATTR_BLAME_CI_LO: result.ci_lo,
        ATTR_BLAME_CI_HI: result.ci_hi,
        ATTR_BLAME_CI_METHOD: report.ci_method.value,
        ATTR_BLAME_Q_VALUE: result.q_value,
        ATTR_BLAME_RESPONSIBLE: result.responsible,
    }


def _decode_tool_name(request_frame: bytes) -> str:
    """Best-effort `gen_ai.tool.name` off a `tape.tool_exchanges` request
    frame's real MCP `tools/call` JSON-RPC wire shape (mirrors
    `settlement.py`'s own `_decode_tool_frame` tool-name extraction) — a
    malformed/non-`tools/call` frame is never a crash, only an honest
    `"unknown"` fallback (never dropped, never fabricated)."""
    try:
        request = json.loads(request_frame)
    except (ValueError, UnicodeDecodeError):
        return "unknown"
    if not isinstance(request, dict) or request.get("method") != "tools/call":
        return "unknown"
    params = request.get("params")
    if not isinstance(params, dict):
        return "unknown"
    tool_name = params.get("name")
    return tool_name if isinstance(tool_name, str) and tool_name else "unknown"


def _normalize_exchange(
    provider: str,
    request_bytes: bytes,
    response_bytes: bytes,
    request_url: str | None = None,
) -> tuple[str | None, NormalizedResponse]:
    adapter = get_adapter(provider)
    request_model = adapter.detect_model(request_bytes, request_url=request_url)
    try:
        normalized = adapter.parse_response(response_bytes)
    except Exception:
        normalized = NormalizedResponse(model=request_model)
    return request_model, normalized


def blame_report_from_json(data: Mapping[str, Any]) -> BlameReport:
    """Reconstruct a `BlameReport` from the JSON `tracefork blame` writes
    (`blame_<run_id>.json`), so `tracefork export --otel` can attach
    flip-rate/CI attributes without re-running blame.

    `q_value`/`responsible`/`responsible_set` are RECOMPUTED from the decoded
    `p_value`s via `blame.benjamini_hochberg` (the same trustworthy-only-input
    rule `BlameEngine.rank()` applies) rather than trusted from the JSON's own
    fields — a round trip through this function is the one boundary where
    those fields could otherwise be forged independently of `p_value`, since
    nothing about the OTel/OpenInference wire shape ties them together.
    """
    fdr_q = data.get("fdr_q", 0.10)
    results = [
        FlipRateResult(
            step_index=r["step_index"],
            flip_rate=r["flip_rate"],
            ci_lo=r["ci_lo"],
            ci_hi=r["ci_hi"],
            flips=r.get("flips", 0),
            trials=r.get("trials", r.get("valid_trials", 0)),
            interpretation=r.get("interpretation", ""),
            valid_trials=r.get("valid_trials", 0),
            undefined=r.get("undefined", 0),
            divergences=r.get("divergences", 0),
            divergence_rate=r.get("divergence_rate", 0.0),
            trustworthy=r.get("trustworthy", True),
            p_value=r.get("p_value", 1.0),
            # q_value/responsible below are placeholders, overwritten by the
            # recompute pass immediately after this list comprehension.
            q_value=1.0,
            responsible=False,
        )
        for r in data.get("results", [])
    ]

    trustworthy_idx = [i for i, r in enumerate(results) if r.trustworthy]
    pvals = [results[i].p_value for i in trustworthy_idx]
    selected, qvals = benjamini_hochberg(pvals, fdr_q)
    for local_i, global_i in enumerate(trustworthy_idx):
        results[global_i].q_value = qvals[local_i]
        results[global_i].responsible = local_i in selected
    responsible_set = sorted(r.step_index for r in results if r.responsible)

    return BlameReport(
        results=results,
        k=data.get("k", 0),
        total_forks=data.get("total_forks", 0),
        ci_method=CIMethod(data.get("ci_method", "wilson")),
        confidence=data.get("confidence", 0.95),
        null_flip_rate=data.get("null_flip_rate", 0.05),
        fdr_q=fdr_q,
        responsible_set=responsible_set,
    )


# ── export: Tape (+ BlameReport) -> OTel GenAI trace (OTLP/JSON) ───────────


def _attr_value(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, (list, tuple)):
        return {"arrayValue": {"values": [_attr_value(v) for v in value]}}
    return {"stringValue": "" if value is None else str(value)}


def _kv_list(attrs: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [{"key": k, "value": _attr_value(v)} for k, v in attrs.items()]


def _flatten_attrs(kv_list: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Inverse of `_kv_list`: OTLP/JSON attribute list -> a plain dict."""
    out: dict[str, Any] = {}
    for item in kv_list:
        key = item.get("key")
        if key is None:
            continue
        value = item.get("value") or {}
        if "stringValue" in value:
            out[key] = value["stringValue"]
        elif "intValue" in value:
            out[key] = int(value["intValue"])
        elif "doubleValue" in value:
            out[key] = value["doubleValue"]
        elif "boolValue" in value:
            out[key] = value["boolValue"]
        elif "arrayValue" in value:
            out[key] = [
                _flatten_attrs([{"key": "_", "value": v}]).get("_")
                for v in value["arrayValue"].get("values", [])
            ]
        else:
            out[key] = None
    return out


def _hex_id(seed: str, nbytes: int) -> str:
    """A deterministic hex id of `nbytes` bytes, derived from `seed` — trace/span
    ids are content-derived (not random) so export output is exact-equality
    reproducible for a given tape, matching this repo's no-float-dust,
    proof-not-assertion ethos."""
    return sha256_hex(seed.encode())[: nbytes * 2]


def build_otel_trace(
    tape: Tape,
    *,
    provider: str = "anthropic",
    blame: BlameReport | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Build an OTLP/JSON-shaped trace (`resourceSpans[].scopeSpans[].spans[]`)
    from `tape`'s exchanges AND tool_exchanges: one `gen_ai.*` CLIENT span per
    LLM exchange (`gen_ai.operation.name = "chat"`) plus one
    `gen_ai.execute_tool.internal`-shaped span per tool exchange
    (`operation.name = "execute_tool"`, `gen_ai.tool.name` decoded off the
    real MCP wire frame) — children of a single root span for the run — plus
    tracefork's own vendor-namespaced `tracefork.blame.*` attributes on any
    LLM step present in `blame.results`.

    `session_id` (default `None`, byte-identical omission when left off) sets
    `gen_ai.conversation.id` on every span (root, LLM, and tool) when given —
    the OTel GenAI attribute correlating spans within one conversation/
    session; pass `store.py`'s orchestration `session_id` when the tape
    belongs to one.

    A plain JSON-serializable dict — no `opentelemetry-sdk` install required
    to produce or consume it (see the module docstring).
    """
    adapter_name = provider
    trace_id = _hex_id(tape.digest() or "tracefork-empty-tape", 16)
    root_span_id = _hex_id(f"{trace_id}:root", 8)
    blame_by_step = {r.step_index: r for r in (blame.results if blame else [])}
    n_exchanges = len(tape.exchanges)
    n_tool_exchanges = len(tape.tool_exchanges)

    root_attrs: dict[str, Any] = {
        "tracefork.agent_name": tape.agent_name,
        "tracefork.tape_digest": tape.digest(),
        "tracefork.exchange_count": n_exchanges,
    }
    if session_id:
        root_attrs[ATTR_CONVERSATION_ID] = session_id

    spans: list[dict[str, Any]] = [
        {
            "traceId": trace_id,
            "spanId": root_span_id,
            "parentSpanId": "",
            "name": f"tracefork.tape {tape.agent_name or '(agent)'}",
            "kind": 1,  # SPAN_KIND_INTERNAL
            "startTimeUnixNano": "0",
            "endTimeUnixNano": str(max(1, n_exchanges + n_tool_exchanges)),
            "attributes": _kv_list(root_attrs),
            "status": {"code": 1},  # STATUS_CODE_OK
        }
    ]

    for i, (req, resp) in enumerate(tape.exchanges):
        request_url = tape.request_urls[i] if i < len(tape.request_urls) else None
        request_model, normalized = _normalize_exchange(adapter_name, req, resp, request_url)
        attrs = normalized_to_genai_attributes(
            normalized, provider=provider, request_model=request_model
        )
        attrs[ATTR_STEP_INDEX] = i
        if session_id:
            attrs[ATTR_CONVERSATION_ID] = session_id
        result = blame_by_step.get(i)
        if result is not None and blame is not None:
            attrs.update(_blame_attributes(result, blame))
        span_id = _hex_id(f"{trace_id}:{i}", 8)
        spans.append(
            {
                "traceId": trace_id,
                "spanId": span_id,
                "parentSpanId": root_span_id,
                "name": f"chat {attrs.get(ATTR_REQUEST_MODEL, 'unknown')}",
                "kind": 3,  # SPAN_KIND_CLIENT
                "startTimeUnixNano": str(i),
                "endTimeUnixNano": str(i + 1),
                "attributes": _kv_list(attrs),
                "status": {"code": 1},
            }
        )

    for j, (tool_req, _tool_resp) in enumerate(tape.tool_exchanges):
        tool_name = _decode_tool_name(tool_req)
        tool_attrs: dict[str, Any] = {
            ATTR_OPERATION_NAME: "execute_tool",
            ATTR_TOOL_NAME: tool_name,
            ATTR_TOOL_STEP_INDEX: j,
        }
        if session_id:
            tool_attrs[ATTR_CONVERSATION_ID] = session_id
        span_id = _hex_id(f"{trace_id}:tool:{j}", 8)
        start = n_exchanges + j
        spans.append(
            {
                "traceId": trace_id,
                "spanId": span_id,
                "parentSpanId": root_span_id,
                "name": f"execute_tool {tool_name}",
                "kind": 1,  # SPAN_KIND_INTERNAL — matches gen_ai.execute_tool.internal
                "startTimeUnixNano": str(start),
                "endTimeUnixNano": str(start + 1),
                "attributes": _kv_list(tool_attrs),
                "status": {"code": 1},
            }
        )

    return {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": _kv_list(
                        {
                            "service.name": "tracefork",
                            "gen_ai.semconv.version": GENAI_SEMCONV_VERSION,
                        }
                    )
                },
                "scopeSpans": [
                    {
                        "scope": {"name": "tracefork", "version": GENAI_SEMCONV_VERSION},
                        "spans": spans,
                    }
                ],
            }
        ]
    }


# ── push: OTel trace -> a live OTLP/HTTP collector (tracefork-sis.60) ──────


class OtlpExportError(TraceforkError):
    """`push_otlp_trace` couldn't deliver the trace: a non-2xx response from
    the collector, or a transport-level failure (connection refused, DNS,
    timeout) — both surfaced as one exception type so `cli.py`'s `export`
    command can catch just this one and print an actionable stderr line
    instead of a raw traceback into `httpx`'s internals."""


def push_otlp_trace(
    trace: Mapping[str, Any],
    endpoint: str,
    *,
    timeout: float = 10.0,
    transport: httpx.BaseTransport | None = None,
) -> httpx.Response:
    """POST `trace` (as produced by `build_otel_trace`) to a live OTLP
    collector's HTTP+JSON trace-export endpoint.

    No `opentelemetry-sdk`/exporter package needed: `build_otel_trace`'s
    output is already shaped exactly as the OTLP/HTTP+JSON wire protocol's
    `ExportTraceServiceRequest` JSON encoding (camelCase field names,
    hex-encoded trace/span ids, a top-level `resourceSpans` array — see
    https://opentelemetry.io/docs/specs/otlp/#otlphttp), so this is a plain
    `httpx` POST of that same dict to `{endpoint}/v1/traces` (the spec's
    documented default trace path) with `Content-Type: application/json` —
    the exact request any real collector (the OTel Collector, Jaeger,
    vendor backends) already accepts. `endpoint`'s trailing slash, if any,
    is stripped before appending the path.

    `transport` (default `None`, real network) lets a caller — this
    module's own tests — inject an `httpx.MockTransport`/`ASGITransport`
    for a fully offline round trip against a local fake collector, the same
    injectable-transport idiom `transport.py`/`proxy.py` already establish
    throughout this codebase.

    Raises `OtlpExportError` (never a raw `httpx` exception) on either a
    non-2xx response or a transport-level failure — see that class's
    docstring. Returns the raw `httpx.Response` on success (status 2xx) so a
    caller can inspect a `partialSuccess` body if it wants to.
    """
    url = f"{endpoint.rstrip('/')}/v1/traces"
    try:
        with httpx.Client(transport=transport, timeout=timeout) as client:
            response = client.post(
                url, json=dict(trace), headers={"Content-Type": "application/json"}
            )
    except httpx.HTTPError as exc:
        raise OtlpExportError(f"could not reach OTLP collector at {url!r}: {exc}") from exc
    if not (200 <= response.status_code < 300):
        raise OtlpExportError(
            f"OTLP collector at {url!r} rejected the export: "
            f"HTTP {response.status_code} {response.text[:200]!r}"
        )
    return response


# ── OTel exemplar back-link (tracefork-bge.53) ─────────────────────────────
#
# `build_otel_trace`'s `trace_id`/`span_id` are content-derived, not random
# (see `_hex_id`), so the mapping is reversible offline: recompute the same
# hash for every stored run/step and find the match. Purely read-only —
# never touches `Tape.digest()`/`to_bytes()`/`from_bytes()` or any store
# schema.


def locate_trace(store: TapeStore, trace_id: str) -> str | None:
    """The run_id whose `build_otel_trace(...)`-exported trace_id equals
    `trace_id`, or `None` if no stored run matches.

    Recomputes each run's trace_id from its tape digest — `store.stored_digest`
    is an opportunistic fast path (skipped when that column doesn't exist
    yet, the same "never a hard dependency" pattern `fsck.py` already
    establishes for it), falling back to a full `load_tape(...).digest()`.
    """
    for run in store.list_runs():
        run_id = run["run_id"]
        digest = store.stored_digest(run_id)
        if digest is None:
            digest = store.load_tape(run_id).digest()
        if _hex_id(digest or "tracefork-empty-tape", 16) == trace_id:
            return run_id
    return None


def locate_span_step(tape: Tape, trace_id: str, span_id: str) -> int | None:
    """The exchange index whose `build_otel_trace(...)`-exported span_id
    (under `trace_id`) equals `span_id`, or `None` for the trace's own root
    span (or any span_id with no match)."""
    for i in range(len(tape.exchanges)):
        if _hex_id(f"{trace_id}:{i}", 8) == span_id:
            return i
    return None


# ── export: Tape (+ BlameReport) -> OpenInference-style dataset JSON ───────


def build_openinference_dataset(
    tape: Tape,
    *,
    provider: str = "anthropic",
    blame: BlameReport | None = None,
) -> dict[str, Any]:
    """Build an OpenInference-shaped dataset (`examples[]` with `llm.*`
    attributes) from `tape`'s exchanges — see the module docstring for exactly
    what "OpenInference-shaped" does and doesn't claim about schema fidelity.
    """
    blame_by_step = {r.step_index: r for r in (blame.results if blame else [])}
    examples: list[dict[str, Any]] = []

    for i, (req, resp) in enumerate(tape.exchanges):
        request_url = tape.request_urls[i] if i < len(tape.request_urls) else None
        request_model, normalized = _normalize_exchange(provider, req, resp, request_url)
        model = normalized.model or request_model
        metadata: dict[str, Any] = {OI_SPAN_KIND: "LLM", OI_PROVIDER: provider}
        if model:
            metadata[OI_MODEL_NAME] = model
        if normalized.input_tokens is not None:
            metadata[OI_TOKEN_PROMPT] = normalized.input_tokens
        if normalized.output_tokens is not None:
            metadata[OI_TOKEN_COMPLETION] = normalized.output_tokens
        metadata[ATTR_STEP_INDEX] = i
        result = blame_by_step.get(i)
        if result is not None and blame is not None:
            metadata.update(_blame_attributes(result, blame))

        examples.append(
            {
                "id": _hex_id(f"{tape.digest() or 'tracefork-empty-tape'}:{i}", 8),
                OI_INPUT_VALUE: req.decode(errors="replace"),
                OI_OUTPUT_VALUE: normalized.first_text() or resp.decode(errors="replace"),
                "metadata": metadata,
            }
        )

    return {
        "dataset_name": tape.agent_name or "tracefork-tape",
        "openinference_semconv_note": (
            "attribute names follow the public OpenInference semantic conventions "
            "(llm.model_name, llm.token_count.*, llm.provider, "
            "openinference.span.kind); not validated against the "
            "openinference-semantic-conventions package's schema — see interop.py."
        ),
        "examples": examples,
    }


# ── ingest: OTel/OpenInference -> Tape step-structure (blame-only) ─────────


def _iter_spans_sorted(export: Mapping[str, Any]) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    for rs in export.get("resourceSpans", []):
        for ss in rs.get("scopeSpans", []):
            spans.extend(ss.get("spans", []))
    spans.sort(key=lambda s: int(s.get("startTimeUnixNano", "0") or "0"))
    return spans


def _synthesize_exchange(
    provider: str,
    *,
    model: str | None,
    text: str,
    input_tokens: int,
    output_tokens: int,
    message_id: str | None = None,
) -> tuple[bytes, bytes]:
    """Build a placeholder (request, response) byte pair from span-level data.

    The request carries only the model id — the original prompt/messages are
    not recoverable from span attributes, so `messages` is deliberately empty.
    This is exactly the "not bit-exact" gap the module docstring documents.
    """
    request_bytes = json.dumps({"model": model, "messages": []}, sort_keys=True).encode()
    adapter = get_adapter(provider)
    response_bytes = adapter.build_text_response(
        text,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        message_id=message_id,
    )
    return request_bytes, response_bytes


def ingest_otel_trace(export: Mapping[str, Any], *, provider: str = "anthropic") -> Tape:
    """Build a `Tape`'s step structure from an OTLP/JSON trace export (as
    produced by `build_otel_trace`, or any collector emitting `gen_ai.*`
    spans) — blame-by-re-execution only, NOT bit-exact replayable. See the
    module docstring for the precise scope.
    """
    tape = Tape(boundary=OTEL_INGESTED_BOUNDARY, agent_name="otel-ingested")
    for span in _iter_spans_sorted(export):
        attrs = _flatten_attrs(span.get("attributes", []))
        if ATTR_REQUEST_MODEL not in attrs and ATTR_RESPONSE_MODEL not in attrs:
            continue  # not a gen_ai span (e.g. this module's own root/internal span)
        model = attrs.get(ATTR_RESPONSE_MODEL) or attrs.get(ATTR_REQUEST_MODEL)
        # The stable gen_ai.* span attributes carry no completion text (message
        # content capture is a separate, opt-in event-based mechanism in the
        # real OTel GenAI spec) — an ingested OTel trace's response is text-less
        # unless the exporter also stamped OpenInference's `output.value`.
        text = attrs.get(OI_OUTPUT_VALUE, "") or ""
        req_bytes, resp_bytes = _synthesize_exchange(
            provider,
            model=model,
            text=text,
            input_tokens=int(attrs.get(ATTR_USAGE_INPUT_TOKENS, 0) or 0),
            output_tokens=int(attrs.get(ATTR_USAGE_OUTPUT_TOKENS, 0) or 0),
            message_id=attrs.get(ATTR_RESPONSE_ID),
        )
        tape.append_exchange(req_bytes, resp_bytes)
    return tape


def ingest_openinference_dataset(
    dataset: Mapping[str, Any], *, provider: str = "anthropic"
) -> Tape:
    """Build a `Tape`'s step structure from an OpenInference-style dataset (as
    produced by `build_openinference_dataset`) — same blame-only, not
    bit-exact-replayable caveat as `ingest_otel_trace`; see the module
    docstring.
    """
    tape = Tape(
        boundary=OTEL_INGESTED_BOUNDARY,
        agent_name=dataset.get("dataset_name", "otel-ingested"),
    )
    examples = sorted(
        dataset.get("examples", []),
        key=lambda e: e.get("metadata", {}).get(ATTR_STEP_INDEX, 0),
    )
    for example in examples:
        metadata = example.get("metadata", {})
        req_bytes, resp_bytes = _synthesize_exchange(
            provider,
            model=metadata.get(OI_MODEL_NAME),
            text=example.get(OI_OUTPUT_VALUE, ""),
            input_tokens=int(metadata.get(OI_TOKEN_PROMPT, 0) or 0),
            output_tokens=int(metadata.get(OI_TOKEN_COMPLETION, 0) or 0),
        )
        tape.append_exchange(req_bytes, resp_bytes)
    return tape
