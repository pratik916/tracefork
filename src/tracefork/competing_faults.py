"""Long-tape competing-fault fixture: several causally-DISTINCT, SIMULTANEOUSLY
plausible faults on one longer tape, used to MEASURE (not merely assert) whether
`blame.py`'s coalition/temporal-Shapley engine (`BlameEngine.shapley_rank`)
discriminates among competing causes -- the thing `faults.py`'s short
positive-vs-inert control fixture explicitly does NOT prove (see its module
docstring and README -> Validation scope).

A single 7-exchange tape (`build_competing_fault_tape`) carries six named
step ROLES (`StepRole`); `make_perturb_factory` "lights up" a chosen subset of
them per SCENARIO so the same tape/agent can host three independent
experiments without cross-scenario masking:

  SCENARIO_ROOT_ECHO   -- step0 ROOT (necessary AND sufficient) + step1 ECHO,
                          a downstream step that merely re-expresses step0's
                          fault: independently "sufficient" (ties step0 under
                          naive single-step flip-rate, exactly like
                          `faults.py`'s fixture) but NOT "necessary" once
                          step0's fault is already in the coalition -- must
                          NOT be blamed as the root. This re-demonstrates
                          `tests/test_blame.py::test_temporal_shapley_discriminates_root_from_echo`
                          on a LONGER, noisier tape (with unrelated decoy
                          steps around it), not just a trivial 2-step one.

  SCENARIO_GATE_PAYLOAD -- step3 GATE and step4 PAYLOAD are two halves of a
                          genuine AND-conjunction (see `_fails`): neither
                          alone is sufficient, but together they are, and
                          BOTH are genuinely causally necessary (remove
                          either one and the failure reverts). This is the
                          fixture's necessary-not-sufficient case.

                          It also surfaces a HONEST, documented LIMITATION:
                          `shapley_rank`'s necessity check is a
                          TEMPORAL-ORDER-RESTRICTED Shapley walk with exactly
                          one valid permutation (see its docstring) rather
                          than an average over permutations, so it can only
                          detect the marginal contribution of the LATER-
                          joining half of a symmetric conjunction. step4
                          (PAYLOAD, joins the coalition second) is correctly
                          flagged `necessity=True`; step3 (GATE, joins the
                          coalition first) is NOT -- its own marginal is
                          measured before step4 completes the AND, so it
                          reads `necessity=False` despite being genuinely
                          necessary. This is exercised, not hidden, by
                          `tests/test_competing_faults.py::test_temporal_order_undercredits_the_earlier_half_of_a_conjunction`.

  SCENARIO_ALL          -- all four fault roles active at once: ROOT alone
                          already determines failure (an over-determined
                          run), so GATE and PAYLOAD correctly read
                          `necessity=False` here -- they are no longer what
                          is *responsible* for this specific run's failure,
                          since removing either one leaves ROOT's fault
                          intact and the run still fails. This is CORRECT
                          behaviour (not a limitation): with several REAL,
                          simultaneously-present causes on one trace, the
                          engine still isolates the one that actually
                          determined the outcome rather than crediting every
                          technically-present-but-overridden fault.

Every "inert" (not-lit-up-this-scenario) candidate step, including the two
NEUTRAL decoys (steps 2 and 5) and the terminal FINAL step (step6), carries no
marker at all -- a true-negative control embedded in the SAME long tape, so
discrimination is tested against realistic noise, not just the planted faults
in isolation.

Step6 (FINAL) is deliberately never asserted on: because it is the tape's
LAST exchange, forcing it alone (or including it as the top member of the
full 7-step coalition) makes its own raw injected bytes literally BE the
graded text, bypassing `RuleBasedTail`'s rule-based adjudication entirely. Its
filler is intentionally the same ambiguous (non-SUCCESS/FAIL-matching) text
used for every other inert step, so a trial that reaches it either way falls
back cleanly (`blame.py`: "an all-UNDEFINED coalition trial ... hold the walk
at its prior value") instead of an artificial override -- but that means its
OWN single-step trial is itself uninformative (all-UNDEFINED), so this module
makes no necessity/sufficiency claim about it.

Zero-diff over the engines: this module only calls the existing public
`blame.py` API (`BlameEngine.shapley_rank`, `StringMatchOracle`) and builds
tapes through the existing `transport.py`/`tape.py` seam -- nothing here
patches or extends those files.
"""

from __future__ import annotations

import enum
from collections.abc import Callable
from typing import Any, cast

import anthropic
import httpx

from .blame import BlameEngine, ShapleyReport, StringMatchOracle
from .synthetic import ScriptedFakeLLM
from .tape import Tape
from .transport import TraceforkTransport
from .wire import make_text_response

N_TURNS = 7

ROOT_MARKER = b"CFX_ROOT_CAUSE"
GATE_MARKER = b"CFX_GATE_HALF"
PAYLOAD_MARKER = b"CFX_PAYLOAD_HALF"

SUCCESS_TEXT = "SUCCESS - competing-fault run complete"
FAIL_TEXT = "FAIL - competing fault triggered"
NEUTRAL_TEXT = "ok, continuing"  # deliberately matches neither success_re nor failure_re

SUCCESS_RESP = make_text_response(SUCCESS_TEXT)
FAIL_RESP = make_text_response(FAIL_TEXT)
NEUTRAL_RESP = make_text_response(NEUTRAL_TEXT)


class StepRole(enum.Enum):
    ROOT = "root"  # necessary AND sufficient
    ECHO = "echo"  # sufficient, NOT necessary (downstream echo of ROOT)
    NEUTRAL = "neutral"  # neither -- true-negative decoy
    GATE = "gate"  # necessary-not-sufficient, earlier half of an AND (see module docstring)
    PAYLOAD = "payload"  # necessary-not-sufficient, later half of the same AND
    FINAL = "final"  # terminal position -- never asserted on (see module docstring)


STEP_ROLES: dict[int, StepRole] = {
    0: StepRole.ROOT,
    1: StepRole.ECHO,
    2: StepRole.NEUTRAL,
    3: StepRole.GATE,
    4: StepRole.PAYLOAD,
    5: StepRole.NEUTRAL,
    6: StepRole.FINAL,
}

# Roles whose marker text `make_perturb_factory` can "light up" for a scenario.
# NEUTRAL and FINAL are always inert -- they exist purely as decoys.
ACTIVATABLE_ROLES = frozenset({StepRole.ROOT, StepRole.ECHO, StepRole.GATE, StepRole.PAYLOAD})

_MARKER_TEXT: dict[StepRole, str] = {
    StepRole.ROOT: f"root cause triggered {ROOT_MARKER.decode()}",
    StepRole.ECHO: f"downstream echo of the root cause {ROOT_MARKER.decode()}",
    StepRole.GATE: f"gate half of the conjunction set {GATE_MARKER.decode()}",
    StepRole.PAYLOAD: f"payload half of the conjunction delivered {PAYLOAD_MARKER.decode()}",
}

SCENARIO_ROOT_ECHO: frozenset[StepRole] = frozenset({StepRole.ROOT, StepRole.ECHO})
SCENARIO_GATE_PAYLOAD: frozenset[StepRole] = frozenset({StepRole.GATE, StepRole.PAYLOAD})
SCENARIO_ALL: frozenset[StepRole] = frozenset(
    {StepRole.ROOT, StepRole.ECHO, StepRole.GATE, StepRole.PAYLOAD}
)


def _fails(accumulated: bytes) -> bool:
    """The ONE failure rule every tail transport in this fixture enforces: an
    independent root-marker fault, OR the two-part AND conjunction (gate AND
    payload) -- never anything else. `accumulated` is the full echoed request
    body a given turn actually sees (see `competing_fault_agent`), so a marker
    introduced at any earlier turn remains visible to every later turn."""
    conjunction = GATE_MARKER in accumulated and PAYLOAD_MARKER in accumulated
    return ROOT_MARKER in accumulated or conjunction


def mutated_response_for(role: StepRole) -> bytes:
    """The marker-carrying response bytes for an ACTIVATABLE role."""
    return make_text_response(_MARKER_TEXT[role])


class RuleBasedTail(httpx.BaseTransport):
    """Serves the rest of `competing_fault_agent`'s turns by adjudicating
    FAIL-vs-benign from `_fails` applied to each incoming request's own
    (already-cumulative) body -- so the SAME one failure rule governs every
    trial type (a single-step fork or a joint coalition fork) without needing
    to know which upstream steps were perturbed. Returns an explicit
    SUCCESS/FAIL text on the final call it expects to see (`remaining_turns`),
    so every trial this backs grades unambiguously."""

    def __init__(self, remaining_turns: int) -> None:
        self._remaining = remaining_turns
        self._seen = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self._seen += 1
        is_last = self._seen >= self._remaining
        body = FAIL_RESP if _fails(request.content) else (SUCCESS_RESP if is_last else NEUTRAL_RESP)
        return httpx.Response(200, headers={"content-type": "application/json"}, content=body)


def make_perturb_factory(active: frozenset[StepRole]) -> Callable[[int], tuple[bytes, Any]]:
    """Build a `perturb_factory` for `blame.py`'s `rank()`/`shapley_rank()`
    that only "lights up" the given roles' markers; every candidate step whose
    role is not in `active` (including a role that in ANOTHER scenario would
    be live) gets the same inert, marker-free filler -- so the one shared
    7-step tape can be reused across scenarios without cross-scenario masking.
    """
    if not active <= ACTIVATABLE_ROLES:
        raise ValueError(f"active roles must be a subset of {ACTIVATABLE_ROLES}")

    def factory(step_idx: int) -> tuple[bytes, Any]:
        role = STEP_ROLES[step_idx]
        mutated = mutated_response_for(role) if role in active else NEUTRAL_RESP
        remaining = N_TURNS - (step_idx + 1)
        return mutated, RuleBasedTail(remaining)

    return factory


def _echo_text(msg: Any) -> str:
    parts: list[str] = []
    for block in msg.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return " | ".join(parts) or "(empty)"


def competing_fault_agent(client: anthropic.Anthropic) -> str:
    """`N_TURNS`-turn linear agent: every turn's request carries the FULL
    prior transcript (all previous user asks + echoed assistant replies), so
    a marker introduced at any turn stays visible to every later turn -- the
    propagation mechanism every fault in this fixture relies on."""
    messages: list[dict[str, Any]] = [{"role": "user", "content": "start"}]
    last_text = ""
    for turn in range(N_TURNS):
        resp = client.messages.create(
            model="claude-sonnet-4-6", max_tokens=100, messages=cast(Any, list(messages))
        )
        last_text = _echo_text(resp)
        messages.append({"role": "assistant", "content": last_text})
        if turn < N_TURNS - 1:
            messages.append({"role": "user", "content": "continue"})
    return last_text


def build_competing_fault_tape() -> Tape:
    """Record the clean (unperturbed) 7-exchange parent tape: no markers
    anywhere, so `_fails` never trips and the run grades SUCCESS."""
    scripted = [NEUTRAL_RESP] * (N_TURNS - 1) + [SUCCESS_RESP]
    fake = ScriptedFakeLLM(scripted)
    tape = Tape(agent_name="competing_fault_agent")
    transport = TraceforkTransport("record", tape, fake)
    client = anthropic.Anthropic(
        api_key="sk-ant-fake",
        http_client=httpx.Client(transport=transport),
        max_retries=0,
    )
    competing_fault_agent(client)
    return tape


def run_shapley(active: frozenset[StepRole], *, k: int = 3, m_samples: int = 2) -> ShapleyReport:
    """Record a fresh clean tape and run `BlameEngine.shapley_rank` over it
    with only `active`'s roles lit up. `budget_usd` is generous and fixed:
    this fixture is offline/$0 regardless (every tail is a fake transport)."""
    tape = build_competing_fault_tape()
    oracle = StringMatchOracle(success_re=r"SUCCESS", failure_re=r"FAIL")
    factory = make_perturb_factory(active)
    return BlameEngine.shapley_rank(
        tape,
        competing_fault_agent,
        oracle,
        perturb_factory=factory,
        k=k,
        m_samples=m_samples,
        budget_usd=1_000_000.0,
    )
