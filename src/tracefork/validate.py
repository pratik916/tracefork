"""Self-validation: run the blame engine on fault-injected runs with known
ground truth and measure how often it fingers the right step.

Fully offline and $0. Each run:
  1. record a clean two-step tape with a synthetic agent;
  2. inject a known fault into step 0 (the "root cause");
  3. run the blame engine — forking re-runs the synthetic agent, which echoes
     each response into its next request, so the fault marker reaches the
     fault-aware tail and flips the outcome;
  4. score a hit when blame ranks the fault step #1 (top-1 precision).

A negative control runs blame with a no-op perturbation and asserts the
flip-rate stays near zero — otherwise a high "precision" would be meaningless.

The synthetic agent is the same callable during recording and every fork, so
the fork prefix replays bit-for-bit (the determinism contract blame relies on).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import anthropic
import httpx

from .blame import BlameEngine, StringMatchOracle
from .faults import FAULT_MARKER_BYTES, FaultClass, FaultInjector
from .synthetic import FaultAwareFakeLLM, ScriptedFakeLLM
from .tape import Tape
from .transport import TraceforkTransport
from .wire import make_text_response, make_tool_use_response

SUCCESS_RESP = make_text_response("SUCCESS — confirmed")
FAIL_RESP = make_text_response("FAIL — cancelled")
TOOL_RESP = make_tool_use_response("check_availability", {"seats": 3, "destination": "Tokyo"})


def _serialize_response(msg) -> str:
    """Flatten an Anthropic message's content to a deterministic string, so the
    agent can echo it (markers and all) into its next request."""
    parts: list[str] = []
    for block in msg.content:
        t = getattr(block, "type", None)
        if t == "text":
            parts.append(block.text)
        elif t == "tool_use":
            parts.append(f"{block.name} {json.dumps(block.input, sort_keys=True)}")
    return " | ".join(parts) or "(empty)"


def synthetic_agent(client: anthropic.Anthropic) -> str:
    """Two-turn agent: ask, then confirm — echoing turn 1's response into the
    turn-2 request so an injected fault propagates to the outcome."""
    r1 = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=100,
        messages=[{"role": "user", "content": "book a flight to Tokyo"}],
    )
    echoed = _serialize_response(r1)
    r2 = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=100,
        messages=[
            {"role": "user", "content": "book a flight to Tokyo"},
            {"role": "assistant", "content": echoed},
            {"role": "user", "content": "confirm"},
        ],
    )
    return _serialize_response(r2)


def _record_clean_tape() -> Tape:
    fake = ScriptedFakeLLM([TOOL_RESP, SUCCESS_RESP])
    tape = Tape(agent_name="synthetic_booking_agent")
    transport = TraceforkTransport("record", tape, fake)
    client = anthropic.Anthropic(
        api_key="sk-ant-fake",
        http_client=httpx.Client(transport=transport),
        max_retries=0,
    )
    synthetic_agent(client)
    return tape


@dataclass
class ValidationReport:
    fault_class: FaultClass
    n_runs: int
    top1_correct: int
    top1_precision: float
    negative_control_max_flip: float = 0.0


class ValidationRunner:
    """Runs offline fault-injection validation for a single fault class."""

    def __init__(self, fault_class: FaultClass, *, k: int = 3, n_runs: int = 5) -> None:
        self._fault_class = fault_class
        self._k = k
        self._n_runs = n_runs

    def run(self) -> ValidationReport:
        oracle = StringMatchOracle(success_re=r"SUCCESS", failure_re=r"FAIL")
        fault_step = 0
        top1_correct = 0
        max_flip_control = 0.0

        for _run in range(self._n_runs):
            tape = _record_clean_tape()
            mutated_resp = FaultInjector.inject(tape, fault_step, self._fault_class)

            # Scope note: this is a positive-vs-inert control — the faulted step gets a
            # flip-capable tail, every other step an inert one. It proves the engine ranks
            # a genuinely outcome-flipping step first (test_blame.py injects the flip at the
            # *final* step to show it isn't hardwired to step 0), not that it discriminates
            # among multiple competing causes on a long tape. See README → Validation scope.
            def perturb_factory(step_idx: int, _mutated=mutated_resp, _fault=fault_step):
                if step_idx == _fault:
                    # Inject the fault; the tail flips when it sees the marker.
                    return _mutated, FaultAwareFakeLLM(
                        normal_responses=[SUCCESS_RESP] * 10,
                        fault_responses=[FAIL_RESP] * 10,
                        fault_marker=FAULT_MARKER_BYTES,
                    )
                # Other steps: a benign perturbation that should not flip.
                return SUCCESS_RESP, ScriptedFakeLLM([SUCCESS_RESP] * 10)

            report = BlameEngine.rank(
                tape,
                synthetic_agent,
                oracle,
                perturb_factory=perturb_factory,
                k=self._k,
                budget_usd=100.0,
            )
            top = report.top()
            if top is not None and top.step_index == fault_step:
                top1_correct += 1

            # Negative control: no real perturbation anywhere → expect no flips.
            def null_perturb_factory(step_idx: int):
                return SUCCESS_RESP, ScriptedFakeLLM([SUCCESS_RESP] * 10)

            ctrl = BlameEngine.rank(
                tape,
                synthetic_agent,
                oracle,
                perturb_factory=null_perturb_factory,
                k=self._k,
                budget_usd=100.0,
            )
            for r in ctrl.results:
                max_flip_control = max(max_flip_control, r.flip_rate)

        precision = top1_correct / self._n_runs if self._n_runs > 0 else 0.0
        return ValidationReport(
            fault_class=self._fault_class,
            n_runs=self._n_runs,
            top1_correct=top1_correct,
            top1_precision=precision,
            negative_control_max_flip=max_flip_control,
        )


def run_all_fault_classes(k: int = 3, n_runs: int = 5) -> dict:
    """Run validation for all five fault classes; return a report dict."""
    results = {}
    for fc in FaultClass:
        report = ValidationRunner(fc, k=k, n_runs=n_runs).run()
        results[fc.value] = {
            "top1_precision": report.top1_precision,
            "top1_correct": report.top1_correct,
            "n_runs": report.n_runs,
            "negative_control_max_flip": report.negative_control_max_flip,
        }
    return results
