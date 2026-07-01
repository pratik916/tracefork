"""Generate the demo tracefork report (examples/demo_report.html).

Records a synthetic two-turn booking run, runs the causal **blame** engine over
it (fully offline, $0), and renders the three-panel report with the blame
overlay. Open the resulting HTML in any browser — no server required.

    uv run python examples/demo_report.py
    open examples/demo_report.html
"""

from __future__ import annotations

import json
from pathlib import Path

import anthropic
import httpx

from tracefork.blame import BlameEngine, StringMatchOracle
from tracefork.faults import FAULT_MARKER_BYTES, FaultClass, FaultInjector
from tracefork.report import generate_report
from tracefork.synthetic import FaultAwareFakeLLM, ScriptedFakeLLM
from tracefork.tape import Tape
from tracefork.transport import TraceforkTransport
from tracefork.validate import FAIL_RESP, SUCCESS_RESP, TOOL_RESP, synthetic_agent


def record_clean_run() -> Tape:
    """Record the parent run: tool call → confirmation (2 exchanges)."""
    fake = ScriptedFakeLLM([TOOL_RESP, SUCCESS_RESP])
    tape = Tape(agent_name="booking-agent")
    transport = TraceforkTransport("record", tape, fake)
    client = anthropic.Anthropic(
        api_key="sk-ant-demo",
        http_client=httpx.Client(transport=transport),
        max_retries=0,
    )
    synthetic_agent(client)
    return tape


def main() -> None:
    tape = record_clean_run()
    oracle = StringMatchOracle(success_re=r"SUCCESS", failure_re=r"FAIL")

    # Perturb step 0 (the tool call) with a corrupted output; the fault-aware
    # tail flips the run to failure. Other steps get a benign perturbation.
    mutated = FaultInjector.inject(tape, 0, FaultClass.CORRUPTED_TOOL_OUTPUT)

    def perturb_factory(step_idx: int, _m=mutated):
        if step_idx == 0:
            return _m, FaultAwareFakeLLM(
                normal_responses=[SUCCESS_RESP] * 30,
                fault_responses=[FAIL_RESP] * 30,
                fault_marker=FAULT_MARKER_BYTES,
            )
        return SUCCESS_RESP, ScriptedFakeLLM([SUCCESS_RESP] * 30)

    report = BlameEngine.rank(
        tape,
        synthetic_agent,
        oracle,
        perturb_factory=perturb_factory,
        k=20,
        budget_usd=100.0,
    )

    blame = {
        r.step_index: {"flip_rate": r.flip_rate, "ci_lo": r.ci_lo, "ci_hi": r.ci_hi}
        for r in report.results
    }

    out = Path(__file__).parent / "demo_report.html"
    generate_report(tape, out, blame=blame)
    print(f"parent outcome: {'SUCCESS' if report.parent_outcome else 'FAIL'}")
    print("blame:", json.dumps(blame, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
