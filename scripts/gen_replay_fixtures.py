"""Regenerate the committed replay-fixture corpus under
``experiments/replay_fixtures/`` that ``tracefork replay --check`` gates
against.

Fully offline/$0 — records each fixture agent (``tracefork.fixtures``)
against a `ScriptedFakeLLM`, saves the tape, and writes `manifest.json`
pinning each tape's agent import path and expected `digest()`.

Run this only when the fixture agents or the tape format intentionally
change — a `replay --check` regression means investigate first, don't just
regenerate to make it pass.

    uv run python scripts/gen_replay_fixtures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import anthropic
import httpx

from tracefork.fixtures import single_turn_agent, two_turn_agent
from tracefork.synthetic import ScriptedFakeLLM
from tracefork.tape import Tape
from tracefork.transport import TraceforkTransport
from tracefork.wire import make_text_response

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "experiments" / "replay_fixtures"


def _record(agent_fn, responses: list[bytes], agent_name: str) -> Tape:
    fake = ScriptedFakeLLM(responses)
    tape = Tape(agent_name=agent_name)
    transport = TraceforkTransport("record", tape, fake)
    client = anthropic.Anthropic(
        api_key="sk-ant-fake",
        http_client=httpx.Client(transport=transport),
        max_retries=0,
    )
    agent_fn(client)
    return tape


def main() -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    fixtures = [
        {
            "name": "single_turn",
            "tape_file": "single_turn.tape.sqlite",
            "agent": "tracefork.fixtures:single_turn_agent",
            "agent_fn": single_turn_agent,
            "responses": [make_text_response("4")],
        },
        {
            "name": "two_turn",
            "tape_file": "two_turn.tape.sqlite",
            "agent": "tracefork.fixtures:two_turn_agent",
            "agent_fn": two_turn_agent,
            "responses": [make_text_response("Red"), make_text_response("Crimson")],
        },
    ]

    manifest = []
    for fx in fixtures:
        tape = _record(fx["agent_fn"], fx["responses"], fx["name"])
        tape_path = FIXTURES_DIR / fx["tape_file"]
        tape.save(str(tape_path))
        # Round-trip through Tape.load to pin the digest of what replay --check
        # will actually load (SQLite save/load is already proven byte-stable
        # elsewhere; this just avoids pinning an in-memory-only digest).
        reloaded = Tape.load(str(tape_path))
        manifest.append(
            {
                "name": fx["name"],
                "tape": fx["tape_file"],
                "agent": fx["agent"],
                "digest": reloaded.digest(),
            }
        )
        print(f"  wrote {tape_path.name}  digest={reloaded.digest()[:12]}…")

    manifest_path = FIXTURES_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"  wrote {manifest_path}")


if __name__ == "__main__":
    main()
