"""Correlational (non-interventional) attribution baseline -- a keyed benchmark.

Every other attribution number this project reports (`blame.py`'s flip-rate
ranking, `competing_faults.py`/`bench.py`'s coalition Shapley) is
**interventional**: it forks the recorded run, re-executes the agent with a
perturbed step, and measures whether the *outcome* actually changes. This
module answers a different, deliberately weaker question: given only the
FROZEN transcript of a run that already failed -- no forking, no
re-execution, no access to the live agent -- can a judge that just *reads
the log* name the responsible step? That is exactly the "post-hoc log
judging" task the README's Related work and scope section contrasts
tracefork's own interventional attribution against, and structurally the
same task the field's public benchmark, Who&When (Zhang et al., ICML 2025
Spotlight), scores over real annotated multi-agent failure logs.

Scored on the SAME fixture `validate.py`'s `ValidationRunner` already scores
interventionally -- `validate._record_clean_tape`/`synthetic_agent` plus
`faults.FaultInjector`'s five fault classes, ground truth `FAULT_STEP` -- so
the two numbers are directly comparable, not tuned against different data.
For each run this module forks the SAME known fault step ONCE (not `blame`'s
k-trial sweep: there is exactly one observed failing run to hand a post-hoc
judge, not many counterfactual ones) via `fork.ForkEngine.fork`, renders the
resulting transcript, and asks `attribution_fn` to name the root-cause step.

**Fairness note, read before trusting any number this produces:** the fault
fixture embeds `faults.FAULT_MARKER` as a literal debug string inside the
faulty response, purely so `synthetic.FaultAwareFakeLLM` can mechanically
decide which script to serve next. A real transcript never contains a
"the fault is right here" flag; a judge that could just grep for it would
trivially solve every fixture without any actual reasoning about the fault's
semantic content (0 seats returned, a misleading "no flights available"
claim, an overridden system prompt, ...), which would defeat the entire
point of a correlational baseline. `render_transcript` strips every literal
occurrence of the marker before a judge ever sees it.

**Keyed path, mirroring `blame.py`'s existing live-API precedent.** Every
function here is offline-testable via an injected `attribution_fn(prompt) ->
raw response text` (the same injection shape `judge.py`'s `LLMJudgeOracle`
already established for its own judge call) -- no test in this suite ever
constructs a real `anthropic.Anthropic()` client or leaves the network
socket guard's allowed hosts. `live_attribution_fn()` wires that seam to a
REAL API call and needs a resolvable credential (`ANTHROPIC_API_KEY` or
another `anthropic.Anthropic()` picks up) the moment its returned callable
is actually invoked; nothing at import time, and nothing anywhere else in
this module, touches the network. `main()` runs the full keyed benchmark
across all five fault classes and prints a report shaped exactly like
`validate.run_all_fault_classes()`'s, for a human to read side by side with
`tracefork validate`'s already-committed interventional numbers:

    uv run python -m tracefork.correlational_baseline

**Honest scope.** This is a small, single-question benchmark on a
self-injected, two-exchange fixture -- not a general-purpose "judge any
transcript" tool, and not run against Who&When's own annotated logs (no
external dataset is ever downloaded; the offline/$0 invariant applies to
everything in this module except the one live call `main()`/
`live_attribution_fn()` make). No CLI wiring (`cli.py` is untouched); a
human runs the module directly.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import anthropic

from .faults import FAULT_MARKER, FAULT_MARKER_BYTES, FaultClass, FaultInjector
from .fork import BranchSpec, ForkEngine
from .synthetic import FaultAwareFakeLLM
from .validate import FAIL_RESP, SUCCESS_RESP, _record_clean_tape, synthetic_agent

__all__ = [
    "FAULT_STEP",
    "render_transcript",
    "correlational_attribute",
    "CorrelationalBaselineReport",
    "run_correlational_baseline",
    "run_all_fault_classes_correlational",
    "live_attribution_fn",
    "main",
]


#: The root-cause step every fixture plants the fault at -- the same ground
#: truth `validate.ValidationRunner.run()` scores its own top-1 precision
#: against (its `fault_step = 0` local, not exported; documented here so the
#: shared convention is explicit rather than an unstated coincidence).
FAULT_STEP = 0

_JSON_STEP_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)
_INT_RE = re.compile(r"-?\d+")


# ── transcript rendering ─────────────────────────────────────────────────────


def _redact_marker(text: str) -> str:
    """Strip tracefork's own internal fault-injection debug flag. See this
    module's docstring's Fairness note -- a real transcript never carries it."""
    return text.replace(FAULT_MARKER, "")


def render_transcript(exchanges: Sequence[tuple[bytes, bytes]]) -> str:
    """Render (request, response) byte pairs as a numbered, human/LLM-readable
    transcript for a post-hoc judge -- exactly the information a frozen log
    would show, with `faults.FAULT_MARKER` redacted (see Fairness note)."""
    blocks = []
    for i, (req, resp) in enumerate(exchanges):
        req_text = _redact_marker(req.decode("utf-8", "replace"))
        resp_text = _redact_marker(resp.decode("utf-8", "replace"))
        blocks.append(f"--- step {i} ---\nREQUEST: {req_text}\nRESPONSE: {resp_text}")
    return "\n".join(blocks)


def _build_prompt(transcript: str, n_steps: int) -> str:
    return (
        "You are auditing a frozen transcript of an AI agent run that FAILED "
        "to complete its task. You cannot re-run the agent, intervene on any "
        "step, or query anything beyond what is shown below -- exactly like a "
        "post-hoc review of a real incident log.\n\n"
        f"The transcript has {n_steps} steps (LLM exchanges), numbered 0 to "
        f"{n_steps - 1}.\n\n"
        f"{transcript}\n\n"
        "Which step is most likely the ROOT CAUSE of the failure? Respond "
        'with a JSON object: {"step": <0-indexed integer>, "rationale": '
        '"<one sentence>"}.'
    )


# ── judge-response parsing ───────────────────────────────────────────────────


def _parse_step(raw: str, n_steps: int) -> int | None:
    """Parse a judge's raw response into a 0-indexed step guess.

    Prefers a JSON object with an integer `step` key (the format the prompt
    asks for, mirroring `judge.py`'s own JSON-first-then-fallback parsing);
    falls back to the first bare integer anywhere in the text. Returns
    `None` (abstain) when nothing parses, or when the parsed value falls
    outside `[0, n_steps)` -- an out-of-range guess can never equal the
    ground-truth step, so silently clamping or discarding it in favor of a
    stray in-range number elsewhere in the same text would inflate precision
    on a method that's actually confused.
    """
    guess: int | None = None
    for match in _JSON_STEP_RE.finditer(raw):
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        step_raw = obj.get("step")
        if isinstance(step_raw, int) and not isinstance(step_raw, bool):
            guess = step_raw
            break

    if guess is None:
        m = _INT_RE.search(raw)
        if m is not None:
            guess = int(m.group(0))

    if guess is None or not (0 <= guess < n_steps):
        return None
    return guess


def correlational_attribute(
    exchanges: Sequence[tuple[bytes, bytes]],
    *,
    attribution_fn: Callable[[str], str],
) -> int | None:
    """Ask `attribution_fn` which step of `exchanges` is most likely
    responsible for the run's outcome, given ONLY the frozen transcript --
    no forking, no re-execution. See this module's docstring.

    Returns the guessed 0-indexed step, or `None` on an unparseable or
    out-of-range response (an honest abstention, never scored as a hit).
    """
    prompt = _build_prompt(render_transcript(exchanges), len(exchanges))
    raw = attribution_fn(prompt)
    return _parse_step(raw, len(exchanges))


# ── the benchmark itself ─────────────────────────────────────────────────────


@dataclass
class CorrelationalBaselineReport:
    """Mirrors `validate.ValidationReport`'s shape (`fault_class`/`n_runs`/
    `top1_*`) so the two can be read side by side, plus `n_abstain` -- a
    judge response the parser couldn't map to a step index has no
    equivalent on the interventional side, where every trial is either a
    counted flip or a counted non-flip."""

    fault_class: FaultClass
    n_runs: int
    top1_correct: int
    top1_precision: float
    n_abstain: int


def run_correlational_baseline(
    fault_class: FaultClass,
    attribution_fn: Callable[[str], str],
    *,
    n_runs: int = 5,
) -> CorrelationalBaselineReport:
    """Score a correlational (post-hoc, non-interventional) attribution
    baseline on the SAME fault-injection fixture `validate.ValidationRunner`
    scores interventionally.

    For each run: record a clean tape (`validate._record_clean_tape`), inject
    `fault_class`'s marked fault at `FAULT_STEP`, and materialize the ONE
    observed failing run by forking there with a fault-aware tail
    (`fork.ForkEngine.fork` -- a single, non-counterfactual re-execution,
    unlike `blame.py`'s k-trial sweep: there is exactly one failing run to
    show a post-hoc judge). The resulting transcript (parent prefix +
    `Branch.delta_tape`, see `fork.py`'s "delta_tape holds only the exchanges
    from the divergence step onward") is handed to `attribution_fn` via
    `correlational_attribute`, scored top-1 against `FAULT_STEP`.
    """
    top1_correct = 0
    n_abstain = 0

    for _run in range(n_runs):
        tape = _record_clean_tape()
        mutated_resp = FaultInjector.inject(tape, FAULT_STEP, fault_class)
        fault_tail = FaultAwareFakeLLM(
            normal_responses=[SUCCESS_RESP] * 10,
            fault_responses=[FAIL_RESP] * 10,
            fault_marker=FAULT_MARKER_BYTES,
        )
        branch = ForkEngine.fork(
            tape,
            BranchSpec(
                divergence_step=FAULT_STEP,
                mutated_response=mutated_resp,
                mutation_desc=f"correlational baseline fixture: {fault_class.value}",
            ),
            synthetic_agent,
            post_fork_transport=fault_tail,
        )
        observed_exchanges = tape.exchanges[:FAULT_STEP] + branch.delta_tape.exchanges
        guess = correlational_attribute(observed_exchanges, attribution_fn=attribution_fn)

        if guess is None:
            n_abstain += 1
        elif guess == FAULT_STEP:
            top1_correct += 1

    precision = top1_correct / n_runs if n_runs > 0 else 0.0
    return CorrelationalBaselineReport(
        fault_class=fault_class,
        n_runs=n_runs,
        top1_correct=top1_correct,
        top1_precision=precision,
        n_abstain=n_abstain,
    )


def run_all_fault_classes_correlational(
    attribution_fn: Callable[[str], str], *, n_runs: int = 5
) -> dict:
    """Run the correlational baseline for all five fault classes; return a
    report dict shaped exactly like `validate.run_all_fault_classes()`'s
    (minus `negative_control_max_flip`, which has no correlational
    equivalent -- there is no perturbation to hold a negative control
    against here, only a judge reading a log), plus `n_abstain` per class."""
    results = {}
    for fc in FaultClass:
        report = run_correlational_baseline(fc, attribution_fn, n_runs=n_runs)
        results[fc.value] = {
            "top1_precision": report.top1_precision,
            "top1_correct": report.top1_correct,
            "n_runs": report.n_runs,
            "n_abstain": report.n_abstain,
        }
    return results


# ── the keyed (real-API) path ────────────────────────────────────────────────


def live_attribution_fn(
    client: anthropic.Anthropic | None = None,
    *,
    model: str = "claude-opus-5",
) -> Callable[[str], str]:
    """Wire the correlational judge to a REAL Anthropic API call.

    Never imported/exercised by any test in this suite -- mirrors
    `blame.py`'s own budget-capped live-API precedent (`tail_transport=None`
    routes a fork's tail through the real API; nothing in the offline test
    suite ever leaves it unset). `client` is injectable for offline testing
    of this wiring itself (a duck-typed stub with a `.messages.create(...)`
    matching the real SDK's shape); the default `None` builds a real
    `anthropic.Anthropic()` -- which needs a resolvable credential
    (`ANTHROPIC_API_KEY`/`ANTHROPIC_AUTH_TOKEN`/an `ant auth login` profile)
    the moment the RETURNED callable is invoked, not at this function's own
    call time. `max_tokens=256` and `effort: "low"` because this is a short
    classification-style answer (one step index plus a one-sentence
    rationale), not an open-ended generation.
    """
    live_client = client if client is not None else anthropic.Anthropic()

    def _call(prompt: str) -> str:
        response = live_client.messages.create(
            model=model,
            max_tokens=256,
            output_config={"effort": "low"},
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in response.content if block.type == "text")

    return _call


def main() -> None:
    """Run the keyed correlational baseline across all five fault classes and
    print a report dict -- the counterpart to `tracefork validate`'s
    already-committed interventional numbers. Requires a resolvable
    Anthropic credential; never invoked by the test suite or any other
    module in this package. Run explicitly:

        uv run python -m tracefork.correlational_baseline
    """
    results = run_all_fault_classes_correlational(live_attribution_fn())
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
