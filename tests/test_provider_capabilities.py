"""``ProviderCapabilities`` manifest tests — registration, lookup, and
cross-checks against the real ``detect_model``/``parse_response`` behavior
the flags claim to describe. Offline, $0.
"""

import json

import tracefork.providers  # noqa: F401  (import for side effect: registers built-ins)
from tracefork.providers import (
    ProviderAdapter,
    get_adapter,
    get_capabilities,
    registered_capabilities,
)
from tracefork.providers.anthropic import AnthropicAdapter
from tracefork.providers.bedrock import BedrockAdapter, build_invoke_model_request
from tracefork.providers.gemini import GeminiAdapter
from tracefork.providers.openai import OpenAIAdapter

CONVERSE_BODY = json.dumps(
    {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": "hello from converse"}],
            }
        },
        "stopReason": "end_turn",
        "usage": {"inputTokens": 42, "outputTokens": 7, "totalTokens": 49},
    }
).encode()


def test_all_built_in_providers_registered():
    names = registered_capabilities()
    assert "anthropic" in names
    assert "openai" in names
    assert "gemini" in names
    assert "bedrock" in names


def test_anthropic_and_openai_model_detectable():
    assert get_capabilities("anthropic").model_detectable is True
    assert get_capabilities("openai").model_detectable is True


def test_gemini_and_bedrock_model_not_detectable_cross_checked():
    # Bedrock's InvokeModel body has no "model" field at all -- the model id
    # is a URL path segment.
    invoke_model_bytes = build_invoke_model_request([{"role": "user", "content": "hi"}])
    assert BedrockAdapter().detect_model(invoke_model_bytes) is None
    assert get_capabilities("bedrock").model_detectable is False

    # Gemini puts the model in the request URL, not a body "model" field.
    gemini_bytes = json.dumps({"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}).encode()
    assert GeminiAdapter().detect_model(gemini_bytes) is None
    assert get_capabilities("gemini").model_detectable is False


def test_only_bedrock_converse_response_cross_checked():
    caps = {name: get_capabilities(name) for name in ("anthropic", "openai", "gemini", "bedrock")}
    assert caps["bedrock"].converse_response is True
    assert caps["anthropic"].converse_response is False
    assert caps["openai"].converse_response is False
    assert caps["gemini"].converse_response is False

    # Bedrock genuinely parses the second (Converse) envelope...
    normalized = BedrockAdapter().parse_response(CONVERSE_BODY)
    assert normalized.first_text() == "hello from converse"

    # ...while the other three adapters have no equivalent second-shape path:
    # each one's own native envelope keys (content / choices / candidates)
    # are absent from the Converse body, so they silently read nothing back.
    assert AnthropicAdapter().parse_response(CONVERSE_BODY).first_text() == ""
    assert OpenAIAdapter().parse_response(CONVERSE_BODY).first_text() == ""
    assert GeminiAdapter().parse_response(CONVERSE_BODY).first_text() == ""


def test_unknown_provider_returns_conservative_default_without_raising():
    caps = get_capabilities("totally-unknown-provider")
    assert caps.name == "totally-unknown-provider"
    assert caps.model_detectable is False
    assert caps.converse_response is False


def test_adapter_satisfies_protocol_unaffected():
    # ProviderCapabilities lives OUTSIDE the ProviderAdapter Protocol -- a
    # registered adapter's isinstance check is untouched by this bead.
    assert isinstance(get_adapter("anthropic"), ProviderAdapter)
