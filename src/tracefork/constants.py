"""Centralised constants — model IDs, pricing, determinism boundary, tape format."""

BOUNDARY_V1 = "single-process-asyncio-v1"

# ── Tape on-the-wire (to_bytes/from_bytes) format ───────────────────────────
# Magic marker + uint16 version prefix the serialized-tape envelope. The magic
# begins with a NUL-free ASCII tag and ends in NUL so a versioned blob can never
# be mistaken for the legacy JSON encoding (which starts with '{'). Blobs without
# this marker are treated as legacy format version 1 (JSON + base64) and still
# load — see tape.from_bytes. Bumping TAPE_FORMAT_VERSION adds a decoder + an
# upcaster entry; existing blobs keep loading via the read-time upcaster chain.
# v3 adds the JSON-RPC tool-exchange log (MCP / native tool frames); v2 and v1
# tapes upcast to an empty tool log, so their content digest is unchanged.
# v4 adds the concurrency-batch log (`async_batches`): the recorded completion
# order of genuinely-concurrent asyncio fan-out (see transport.py). It is
# envelope/metadata only — like `boundary`/`agent_name`, it is NOT fed into
# `digest()` — so every existing (and every sequential) tape's content digest
# is byte-identical, and v1/v2/v3 tapes upcast to an empty batch log.
# v5 adds the `provenance` witness block (matcher_name/boundary_guard/
# nondet_mode recorded by `Recorder`/`AsyncRecorder`). Like `async_batches`,
# it is envelope/metadata only — NOT fed into `digest()` — so every existing
# tape's content digest is unchanged, and v1-v4 tapes upcast to `provenance={}`.
TAPE_MAGIC = b"TFTAPE\x00"
TAPE_FORMAT_VERSION = 5

# Model IDs (consult claude-api skill before editing)
SONNET = "claude-sonnet-4-6"
HAIKU = "claude-haiku-4-5-20251001"
OPUS = "claude-opus-4-8"

# Pricing per token (USD), list price per 1M tokens — update PRICING_VERSION when
# changed. Source: the `claude-api` skill (current Anthropic list pricing).
#
# These per-token Anthropic anchors are kept as the byte-identity guardrail for
# the pricing registry: the pinned snapshot in `pricing.py`
# (`tracefork/data/pricing.json`) MUST reproduce them exactly, so `BudgetGovernor`
# behaviour is unchanged. The flat per-model `PRICING_TABLE` was replaced by the
# provider-generic `(provider, model) -> rates` registry in `pricing.py`.
PRICING_VERSION = "2026-06b"
SONNET_INPUT_PER_TOKEN = 3.00 / 1_000_000
SONNET_OUTPUT_PER_TOKEN = 15.00 / 1_000_000
HAIKU_INPUT_PER_TOKEN = 1.00 / 1_000_000
HAIKU_OUTPUT_PER_TOKEN = 5.00 / 1_000_000
OPUS_INPUT_PER_TOKEN = 5.00 / 1_000_000
OPUS_OUTPUT_PER_TOKEN = 25.00 / 1_000_000

# ── OTel GenAI / OpenInference interop (see interop.py) ─────────────────────
#
# Pinned OpenTelemetry semantic-conventions release the `gen_ai.*` attribute
# names in `interop.py` target — https://opentelemetry.io/docs/specs/semconv/gen-ai/.
# Bump deliberately (it is not auto-detected) if a future attribute rename
# lands upstream; nothing here is byte-hashed, so bumping never touches
# `Tape.digest()`.
GENAI_SEMCONV_VERSION = "1.29.0"

# Boundary marker for a `Tape` whose step structure was reconstructed from an
# ingested OTel/OpenInference trace (`interop.ingest_otel_trace` /
# `ingest_openinference_dataset`) rather than recorded by tracefork's own
# transport. Deliberately distinct from `BOUNDARY_V1` so such a tape is never
# mistaken for a bit-exact-replayable one: its exchange bytes are synthesized
# from span attributes, not raw recorded bytes, so `ReplayVerifier` /
# `ForkEngine`'s prefix-replay phase will correctly diverge against a real
# agent. It supports blame-by-re-execution at the step-structure level only —
# see `interop.py`'s module docstring for the precise scope.
OTEL_INGESTED_BOUNDARY = "otel-ingested-blame-only-v1"

# Boundary marker for a `Tape` recorded through `proxy.py`'s localhost base-URL
# record/replay proxy rather than tracefork's in-process httpx transport seam.
# Deliberately distinct from `BOUNDARY_V1`: a proxy-recorded tape has no
# in-process `NondetSource` behind it (the client is on the other side of a TCP
# socket), so it sits outside the full single-process determinism boundary —
# see `proxy.py`'s module docstring for the precise scope. Metadata only, like
# `OTEL_INGESTED_BOUNDARY`; never fed into `digest()`.
PROXY_BOUNDARY = "proxy-record-replay-v1"
