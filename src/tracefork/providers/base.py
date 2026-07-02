"""Provider-neutral seam: a normalized response view + an adapter registry.

The raw request/response **bytes** stay the immutable bit-exact replay+hash
contract (owned by ``transport.py`` and ``tape.py``). An adapter only derives a
*normalized* view — gen_ai.*-style neutral names (``model``, ``input_tokens``,
``output_tokens``, ``finish_reason``, content ``parts``) — for the consumers that
would otherwise hardcode one provider's JSON shape (blame, faults, report, the
wire builders). Anthropic is the first *registered* adapter, not a hardcoded
assumption; new backends register under their own name.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from ..plugins import PROVIDER_GROUP, Registry

DEFAULT_PROVIDER = "anthropic"


@dataclass(frozen=True)
class ContentPart:
    """One normalized content block of an assistant response.

    ``type`` is a neutral tag (``"text"``, ``"tool_use"``, or a provider tag we
    don't specially model). Text parts carry ``text``; tool-call parts carry
    ``tool_name`` / ``tool_id`` / ``tool_input``.
    """

    type: str
    text: str | None = None
    tool_name: str | None = None
    tool_id: str | None = None
    tool_input: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class NormalizedResponse:
    """Provider-neutral view of a single assistant response.

    Field names follow the OpenTelemetry ``gen_ai.*`` convention (model, usage
    input/output tokens, finish reason) so downstream code never reads one
    provider's JSON schema directly.
    """

    model: str | None = None
    content: tuple[ContentPart, ...] = ()
    input_tokens: int | None = None
    output_tokens: int | None = None
    finish_reason: str | None = None
    message_id: str | None = None

    def first_text(self) -> str:
        """Text of the first text part, or ``""`` if there is none."""
        for part in self.content:
            if part.type == "text":
                return part.text or ""
        return ""

    def text_parts(self) -> list[str]:
        """All text parts, in order."""
        return [p.text or "" for p in self.content if p.type == "text"]


@runtime_checkable
class ProviderAdapter(Protocol):
    """Normalizes one provider's wire format behind a stable seam.

    Adapters never mutate the byte contract; they read raw bytes and return a
    neutral view, or build fresh (counterfactual) response bytes for forks/faults.
    """

    name: str

    def canonicalize_request(self, request_bytes: bytes) -> str:
        """A hashable identity for a request. Anthropic's raw bytes are already
        canonical, so this is their sha256; ``transport.py`` still owns the actual
        replay-time matching (this seam is for the divergence-contract work)."""
        ...

    def detect_model(self, request_bytes: bytes) -> str | None:
        """Best-effort model id from a recorded request (``None`` if unknown)."""
        ...

    def parse_response(self, response_bytes: bytes) -> NormalizedResponse:
        """Parse a (non-streaming JSON) response into a ``NormalizedResponse``.

        Raises on non-JSON input so callers can apply their own raw fallback."""
        ...

    def parse_sse(self, response_bytes: bytes) -> dict[str, Any] | None:
        """Extract the first meaningful JSON object from an SSE stream, or ``None``."""
        ...

    def tool_use_inputs(
        self, response_bytes: bytes
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        """Return ``(parsed_response, [tool_call_input_dicts])`` for in-place
        mutation, or ``(None, [])`` when the response is not a mutable object."""
        ...

    def mutate_response(self, normalized: NormalizedResponse) -> bytes:
        """Serialize a normalized response back to provider wire bytes."""
        ...

    def build_text_response(
        self,
        text: str,
        *,
        model: str | None = None,
        input_tokens: int = 100,
        output_tokens: int = 20,
        message_id: str | None = None,
    ) -> bytes:
        """Build wire bytes for a final text response."""
        ...

    def build_tool_use_response(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        *,
        model: str | None = None,
        preamble: str = "",
        input_tokens: int = 100,
        output_tokens: int = 30,
        message_id: str | None = None,
    ) -> bytes:
        """Build wire bytes for a tool-use response."""
        ...


# ── registry ────────────────────────────────────────────────────────────────
#
# Backed by the generic `Registry` (see `plugins.py`) rather than a plain
# dict; `Registry` *is* a `dict[str, T]` subclass, so `_REGISTRY.pop(...)`,
# `sorted(_REGISTRY)`, and `name in _REGISTRY` keep working exactly as before
# this seam existed — only entry-point discovery (`load_provider_entry_points`,
# opt-in, see `plugins.py`) is new.

_REGISTRY: Registry[ProviderAdapter] = Registry(PROVIDER_GROUP, kind="provider adapter")


def register_adapter(adapter: ProviderAdapter, *, name: str | None = None) -> None:
    """Register ``adapter`` under ``name`` (defaults to ``adapter.name``)."""
    _REGISTRY.register(name or adapter.name, adapter)


def get_adapter(name: str = DEFAULT_PROVIDER) -> ProviderAdapter:
    """Look up a registered adapter by name (default: the Anthropic adapter)."""
    return _REGISTRY.get_or_raise(name)


def default_adapter() -> ProviderAdapter:
    """The default adapter (Anthropic) — existing tapes resolve here unchanged."""
    return get_adapter(DEFAULT_PROVIDER)


def registered_providers() -> list[str]:
    """Sorted names of all registered adapters."""
    return _REGISTRY.names()


def load_provider_entry_points(
    *, allow: frozenset[str] | set[str] | None = None, allow_all: bool = False
) -> list[str]:
    """Opt-in: discover third-party provider adapters advertised under the
    ``tracefork.providers`` entry-point group (see ``plugins.py`` for the
    security-gating contract — nothing loads unless explicitly allowlisted).
    """
    return _REGISTRY.load_entry_points(allow=allow, allow_all=allow_all)
