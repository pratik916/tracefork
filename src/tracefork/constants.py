"""Centralised constants — model IDs, pricing, determinism boundary."""

BOUNDARY_V1 = "single-process-asyncio-v1"

# Model IDs (consult claude-api skill before editing)
SONNET = "claude-sonnet-4-6"
HAIKU = "claude-haiku-4-5-20251001"
OPUS = "claude-opus-4-8"

# Pricing per token (USD) — update PRICING_VERSION when changed
PRICING_VERSION = "2026-06"
SONNET_INPUT_PER_TOKEN = 3.00 / 1_000_000
SONNET_OUTPUT_PER_TOKEN = 15.00 / 1_000_000
HAIKU_INPUT_PER_TOKEN = 0.80 / 1_000_000
HAIKU_OUTPUT_PER_TOKEN = 4.00 / 1_000_000
OPUS_INPUT_PER_TOKEN = 15.00 / 1_000_000
OPUS_OUTPUT_PER_TOKEN = 75.00 / 1_000_000

PRICING_TABLE: dict[str, tuple[float, float]] = {
    SONNET: (SONNET_INPUT_PER_TOKEN, SONNET_OUTPUT_PER_TOKEN),
    HAIKU: (HAIKU_INPUT_PER_TOKEN, HAIKU_OUTPUT_PER_TOKEN),
    OPUS: (OPUS_INPUT_PER_TOKEN, OPUS_OUTPUT_PER_TOKEN),
}
