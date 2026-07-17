"""Trivial importable agent used as the allowlisted `agent_fn` in
tests/test_fork_endpoint.py -- a real import path (`tests.fixtures.fork_ui_agent:run_agent`)
so `fork.ForkEngine.fork`'s "same agent that produced the tape" contract is
satisfied for real, not faked.
"""

from __future__ import annotations

import anthropic


def run_agent(client: anthropic.Anthropic) -> str:
    """One-exchange agent: ask a single question, return the answer text."""
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=100,
        messages=[{"role": "user", "content": "What is 2 + 2?"}],
    )
    block = resp.content[0]
    text = getattr(block, "text", None)
    return text if isinstance(text, str) else ""
