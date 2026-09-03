"""record_your_agent.py — the runnable twin of the README's "Record your own agent" section.

Shows the actual integration shape: wrap a real `anthropic.Anthropic` client with `Recorder`,
run your agent through it once, save the tape, and replay it later for a bit-exact, $0 receipt.

To stay offline/$0 the way the rest of this repo's examples are, `run_agent`'s client is backed
by a hand-written `httpx2.MockTransport` that returns one canned, wire-correct Anthropic Messages
API response instead of a real network call. That's a stand-in for the label on this file, not
tracefork machinery: swap `build_fake_client()` for a real `anthropic.Anthropic()` (reading
`ANTHROPIC_API_KEY` from the environment, no `http_client=` override) and everything below —
`Recorder`, `tape.save`, `Tape.load`, `ReplayVerifier` — is exactly what pointing tracefork at
your own agent looks like. No `tracefork.synthetic`/`tracefork.wire` test-scaffolding imports
here on purpose, unlike `examples/demo_report.py` — this file is meant to double as copy-paste
starting material for a real integration, so every import below is either the top-level
`tracefork` package or a real dependency (`anthropic`, `httpx2`) a real integration would need
too.

Usage: `uv run python examples/record_your_agent.py` — offline, $0, no `ANTHROPIC_API_KEY`.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import anthropic
import httpx2

from tracefork import Recorder, ReplayVerifier, Tape


def run_agent(client: anthropic.Anthropic) -> str:
    """A minimal one-turn "agent": ask the model a question, return its text reply.

    This is the function whose *behavior* tracefork proves is reproducible — record it once
    against a real (or, here, fake-for-offline-demo) client, then replay the exact same
    function against the recorded tape and get back the identical trajectory, verified by hash.
    """
    reply = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=32,
        messages=[{"role": "user", "content": "In one sentence, what is a time-travel debugger?"}],
    )
    block = reply.content[0]
    assert block.type == "text"
    return block.text


def _fake_response(_request: httpx2.Request) -> httpx2.Response:
    """One canned, wire-correct Anthropic Messages API response body — see this module's
    docstring: this is the offline-demo stand-in, not part of the tracefork API surface."""
    return httpx2.Response(
        200,
        json={
            "id": "msg_demo",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-6",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "A time-travel debugger lets you rewind a program's execution and "
                        "replay or fork it from any earlier point."
                    ),
                }
            ],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 24, "output_tokens": 22},
        },
    )


def build_fake_client() -> anthropic.Anthropic:
    """Offline stand-in for a real `anthropic.Anthropic()` client — see this module's
    docstring for what to swap in instead for a real recording."""
    return anthropic.Anthropic(
        api_key="sk-demo-not-a-real-key",
        http_client=httpx2.Client(transport=httpx2.MockTransport(_fake_response)),
    )


def main() -> None:
    client = build_fake_client()  # <-- swap for a real anthropic.Anthropic() client

    with Recorder(client, agent_name="record-your-agent-example") as rec:
        answer = run_agent(rec.client)
    print(f"agent said: {answer!r}")

    with tempfile.TemporaryDirectory() as tmp:
        tape_path = str(Path(tmp) / "recorded_run.db")
        rec.tape.save(tape_path)
        print(f"tape saved: {tape_path} (fingerprint {rec.tape.digest()[:16]}…)")

        # Reload from disk to prove this isn't just verifying the in-memory tape object —
        # the save/load round trip is part of what's being checked here.
        reloaded = Tape.load(tape_path)
        result = ReplayVerifier(reloaded, run_agent).verify()

    print(
        f"replay: bit_exact={result.bit_exact}  "
        f"({result.matched}/{result.total} requests matched, $0, no network)"
    )
    assert result.bit_exact, "replay was not bit-exact — this should never happen offline"


if __name__ == "__main__":
    main()
