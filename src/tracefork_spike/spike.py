"""Spike 0 orchestration: record -> persist -> load -> replay -> verify.

Answers exactly one question: can we record a tool-using Anthropic-SDK agent run and
replay it bit-exact, with proof, for $0 and no network — within a declared
determinism boundary? The boundary here: a single-process, synchronous agent whose
only nondeterminism sources are clock and id generation, both routed through the
NondetSource seam.
"""

from __future__ import annotations

import os
import tempfile

from .agent import make_client, run_agent
from .fake_llm import FakeAnthropicTransport
from .nondet import DivergenceError, DriftingNondet, RecordingNondet, ReplayNondet
from .tape import Tape
from .transport import TraceforkTransport


def record_replay_verify(tape_path: str | None = None) -> dict:
    """Run the full spike and return a structured result dict (used by the CLI and tests)."""
    cleanup = False
    if tape_path is None:
        fd, tape_path = tempfile.mkstemp(suffix=".tape.sqlite")
        os.close(fd)
        cleanup = True

    try:
        # 1. RECORD — real agent + SDK, fake (offline) endpoint, genuine nondeterminism.
        rec_tape = Tape()
        rec_nondet = RecordingNondet()
        rec_transport = TraceforkTransport("record", rec_tape, inner=FakeAnthropicTransport())
        rec_result = run_agent(make_client(rec_transport), rec_nondet)
        rec_tape.draws = rec_nondet.draws
        record_fingerprint = rec_tape.digest()

        # 2. PERSIST + RELOAD — prove the content-addressed tape round-trips through disk.
        rec_tape.save(tape_path)
        loaded = Tape.load(tape_path)
        assert loaded.digest() == record_fingerprint, "tape changed across save/load"

        # 3. REPLAY — no network: replay transport has no inner; nondeterminism virtualized.
        rep_nondet = ReplayNondet(loaded.draws)
        rep_transport = TraceforkTransport("replay", loaded)  # inner=None -> any real call errors
        rep_result = run_agent(make_client(rep_transport), rep_nondet)
        replay_fingerprint = loaded.digest()

        # 4. VERIFY — observable output identical, every request hash matched,
        #    all recorded draws consumed, no leftover exchanges.
        checks = {
            "output_identical": rep_result == rec_result,
            "fingerprint_match": replay_fingerprint == record_fingerprint,
            "all_request_hashes_matched": rep_transport.matched == len(loaded.exchanges),
            "all_exchanges_consumed": rep_transport.fully_consumed(),
            "all_draws_consumed": rep_nondet.fully_consumed(),
        }

        # 5. NEGATIVE CONTROL — replay with drifting (fresh) nondeterminism MUST diverge.
        drift_detected = False
        drift_at: str | None = None
        try:
            run_agent(make_client(TraceforkTransport("replay", loaded)), DriftingNondet())
        except DivergenceError as e:
            drift_detected = True
            drift_at = str(e)
        checks["negative_control_detected_drift"] = drift_detected

        return {
            "exchanges": len(loaded.exchanges),
            "draws": len(loaded.draws),
            "request_hashes_matched": rep_transport.matched,
            "record_fingerprint": record_fingerprint,
            "replay_fingerprint": replay_fingerprint,
            "final_text": rec_result["final_text"],
            "checks": checks,
            "drift_at": drift_at,
            "passed": all(checks.values()),
        }
    finally:
        if cleanup and os.path.exists(tape_path):
            os.remove(tape_path)


def _fmt(result: dict) -> str:
    c = result["checks"]
    ok = "PASS" if result["passed"] else "FAIL"
    lines = [
        "",
        "  tracefork — Spike 0: bit-exact record/replay",
        "  " + "-" * 52,
        f"  recorded exchanges ........ {result['exchanges']}",
        f"  nondeterminism draws ...... {result['draws']}  (clock + id, virtualized)",
        f"  request hashes matched .... {result['request_hashes_matched']}/{result['exchanges']}",
        f"  tape fingerprint .......... {result['record_fingerprint'][:24]}…",
        f"  replay fingerprint ........ {result['replay_fingerprint'][:24]}…",
        "  network calls / spend ..... 0 / $0.00",
        f"  agent final answer ........ {result['final_text']!r}",
        "",
        f"    [{'x' if c['output_identical'] else ' '}] "
        "replayed trajectory byte-identical to recorded",
        f"    [{'x' if c['fingerprint_match'] else ' '}] "
        "tape fingerprint matches after save/load round-trip",
        f"    [{'x' if c['all_request_hashes_matched'] else ' '}] "
        "every replayed request hash matched the tape",
        f"    [{'x' if c['all_draws_consumed'] else ' '}] "
        "every recorded nondeterminism draw consumed",
        f"    [{'x' if c['negative_control_detected_drift'] else ' '}] "
        "negative control: drift was DETECTED, not silently passed",
        "",
        f"  RESULT: {ok}",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    result = record_replay_verify()
    print(_fmt(result))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
