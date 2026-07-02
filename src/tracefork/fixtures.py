"""Tiny, deterministic agents backing the committed replay-fixture corpus
(``experiments/replay_fixtures/``) that ``tracefork replay --check`` gates
against. Kept separate from ``validate.py``'s fault-injection agent so the
corpus doesn't couple to that module's fault-testing concerns. See
``scripts/gen_replay_fixtures.py`` for how the corpus tapes are (re)built.
"""

from __future__ import annotations

from typing import Any

import anthropic


def _text_of(message: Any) -> str:
    """Flatten a message's first text block. Content blocks are a large typed
    union (tool_use/thinking/... included), so this narrows defensively rather
    than assuming ``content[0]`` is a text block."""
    block = message.content[0]
    text = getattr(block, "text", None)
    if not isinstance(text, str):
        raise TypeError(f"expected a text content block, got {block!r}")
    return text


def single_turn_agent(client: anthropic.Anthropic) -> str:
    """One-exchange fixture agent: ask a single question, return the answer."""
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=100,
        messages=[{"role": "user", "content": "What is 2 + 2?"}],
    )
    return _text_of(resp)


def two_turn_agent(client: anthropic.Anthropic) -> str:
    """Two-exchange fixture agent: ask, then follow up, echoing turn 1 back."""
    r1 = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=100,
        messages=[{"role": "user", "content": "Name a primary color."}],
    )
    answer = _text_of(r1)
    r2 = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=100,
        messages=[
            {"role": "user", "content": "Name a primary color."},
            {"role": "assistant", "content": answer},
            {"role": "user", "content": "Now name a shade of it."},
        ],
    )
    return _text_of(r2)
