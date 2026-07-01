"""Centralised constants — model IDs, pricing, determinism boundary, tape format."""

BOUNDARY_V1 = "single-process-asyncio-v1"

# ── Tape on-the-wire (to_bytes/from_bytes) format ───────────────────────────
# Magic marker + uint16 version prefix the serialized-tape envelope. The magic
# begins with a NUL-free ASCII tag and ends in NUL so a versioned blob can never
# be mistaken for the legacy JSON encoding (which starts with '{'). Blobs without
# this marker are treated as legacy format version 1 (JSON + base64) and still
# load — see tape.from_bytes. Bumping TAPE_FORMAT_VERSION adds a decoder + an
# upcaster entry; existing blobs keep loading via the read-time upcaster chain.
TAPE_MAGIC = b"TFTAPE\x00"
TAPE_FORMAT_VERSION = 2

# Model IDs (consult claude-api skill before editing)
SONNET = "claude-sonnet-4-6"
HAIKU = "claude-haiku-4-5-20251001"
OPUS = "claude-opus-4-8"

# Pricing per token (USD), list price per 1M tokens — update PRICING_VERSION when
# changed. Source: the `claude-api` skill (current Anthropic list pricing).
PRICING_VERSION = "2026-06b"
SONNET_INPUT_PER_TOKEN = 3.00 / 1_000_000
SONNET_OUTPUT_PER_TOKEN = 15.00 / 1_000_000
HAIKU_INPUT_PER_TOKEN = 1.00 / 1_000_000
HAIKU_OUTPUT_PER_TOKEN = 5.00 / 1_000_000
OPUS_INPUT_PER_TOKEN = 5.00 / 1_000_000
OPUS_OUTPUT_PER_TOKEN = 25.00 / 1_000_000

PRICING_TABLE: dict[str, tuple[float, float]] = {
    SONNET: (SONNET_INPUT_PER_TOKEN, SONNET_OUTPUT_PER_TOKEN),
    HAIKU: (HAIKU_INPUT_PER_TOKEN, HAIKU_OUTPUT_PER_TOKEN),
    OPUS: (OPUS_INPUT_PER_TOKEN, OPUS_OUTPUT_PER_TOKEN),
}
