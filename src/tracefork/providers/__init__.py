"""Provider adapters: normalize provider wire formats behind a stable seam.

Raw request/response **bytes** remain the immutable bit-exact replay+hash
contract (owned by ``transport.py`` and ``tape.py``); an adapter only derives a
normalized (gen_ai.*-style) view for the consumers that would otherwise hardcode
a single provider's JSON shape. Anthropic is the first *registered* adapter, not
a hardcoded assumption — importing this package registers the built-ins under
``"anthropic"``, ``"openai"``, and ``"gemini"``.
"""

from __future__ import annotations

from . import anthropic as _anthropic  # noqa: F401  (import for side effect: registers "anthropic")
from . import gemini as _gemini  # noqa: F401  (import for side effect: registers "gemini")
from . import openai as _openai  # noqa: F401  (import for side effect: registers "openai")
from .base import (
    ContentPart,
    NormalizedResponse,
    ProviderAdapter,
    default_adapter,
    get_adapter,
    load_provider_entry_points,
    register_adapter,
    registered_providers,
)

__all__ = [
    "ContentPart",
    "NormalizedResponse",
    "ProviderAdapter",
    "default_adapter",
    "get_adapter",
    "load_provider_entry_points",
    "register_adapter",
    "registered_providers",
]
