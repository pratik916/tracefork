"""Adapter fork/blame parity — the regression guard for the pass-through claim.

Every built-in framework adapter's ``bind()`` routes its target through the
SAME shared ``build_http_clients()`` helper (see ``adapters/base.py``) that
wraps a ``TraceforkTransport`` — the identical byte seam every other tracefork
consumer already uses (confirmed by reading each adapter's ``bind()`` body).
Record-mode injection is ``# pragma: no cover - needs real SDK`` in every
adapter's own docstring/implementation, so this test drives the offline-safe
half of the claim: ``mode="replay"`` only, with a trivial ``object()`` target
(``BindResult`` always carries a working ``http_client``/``transport``
regardless of injection outcome — see each adapter's own
``test_bind_unknown_target_reports_notes``).

Two things are proven, per adapter:

  (a) ``bind(object(), tape, mode="replay").http_client`` serves byte-identical
      responses to a bare ``TraceforkTransport("replay", tape)`` replaying the
      SAME tape, for every recorded exchange, and the bind/replay/teardown
      cycle never mutates ``Tape.digest()``;
  (b) ``ForkEngine.fork()``/``BlameEngine.rank()`` computed on the tape BEFORE
      any adapter has touched it are bit-identical to the SAME computation run
      AFTER the adapter's bind/replay/teardown cycle on that SAME tape object —
      proving fork/blame are provably unaffected by any adapter's involvement.

Offline, $0, no framework installed — reuses ``ScriptedFakeLLM``/
``make_text_response`` exactly as ``test_fork.py``/``test_blame.py`` already do.
"""

from __future__ import annotations

import anthropic
import httpx
import pytest

from tests.fakes import ScriptedFakeLLM, make_text_response
from tracefork.adapters.adk import AdkAdapter
from tracefork.adapters.autogen import AutoGenAdapter
from tracefork.adapters.base import BaseFrameworkAdapter
from tracefork.adapters.crewai import CrewAIAdapter
from tracefork.adapters.langchain import LangChainAdapter
from tracefork.adapters.openai_agents import OpenAIAgentsAdapter
from tracefork.blame import BlameEngine, BlameReport, StringMatchOracle
from tracefork.fork import Branch, BranchSpec, ForkEngine
from tracefork.tape import Tape
from tracefork.transport import TraceforkTransport

RESP_TURN1 = make_text_response("Checking availability")
RESP_TURN2 = make_text_response("SUCCESS — booking confirmed")
RESP_FAIL = make_text_response("FAIL — no flights available")

REPLAY_URL = "https://api.anthropic.com/v1/messages"

ADAPTER_CLASSES: list[type[BaseFrameworkAdapter]] = [
    LangChainAdapter,
    OpenAIAgentsAdapter,
    CrewAIAdapter,
    AutoGenAdapter,
    AdkAdapter,
]


def _agent(client: anthropic.Anthropic) -> str:
    """Two-turn agent; turn2's history embeds turn1's reply text — same shape
    as test_fork.py's/test_blame.py's own fixture agents, so a mutation at
    turn1 changes turn2's request bytes (the counterfactual behaviour fork and
    blame both depend on)."""
    r1 = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=100,
        messages=[{"role": "user", "content": "book a flight"}],
    )
    first = r1.content[0].text
    r2 = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=100,
        messages=[
            {"role": "user", "content": "book a flight"},
            {"role": "assistant", "content": first},
            {"role": "user", "content": "confirm"},
        ],
    )
    return r2.content[0].text


def _build_two_turn_tape() -> Tape:
    """Parent run recorded via the raw-SDK path — zero adapter involvement,
    identical in spirit to test_fork.py's ``_build_two_turn_tape()``."""
    fake = ScriptedFakeLLM([RESP_TURN1, RESP_TURN2])
    tape = Tape()
    transport = TraceforkTransport("record", tape, fake)
    client = anthropic.Anthropic(
        api_key="sk-ant-fake",
        http_client=httpx.Client(transport=transport),
        max_retries=0,
    )
    _agent(client)
    return tape


def _perturb_factory(step_idx: int) -> tuple[bytes, object]:
    """Perturb every step with FAIL; the tail (reached only when the perturbed
    step is not final) re-succeeds, so only the final step's flip-rate is
    high — same shape as test_blame.py's own fixture."""
    return RESP_FAIL, ScriptedFakeLLM([RESP_TURN2])


def _run_fork_and_blame(tape: Tape) -> tuple[Branch, BlameReport]:
    """One ForkEngine.fork + one BlameEngine.rank snapshot computed on `tape`."""
    spec = BranchSpec(divergence_step=0, mutated_response=RESP_FAIL)
    branch = ForkEngine.fork(tape, spec, _agent, post_fork_transport=ScriptedFakeLLM([RESP_TURN2]))
    oracle = StringMatchOracle(success_re=r"SUCCESS", failure_re=r"FAIL")
    report = BlameEngine.rank(
        tape,
        _agent,
        oracle,
        perturb_factory=_perturb_factory,
        k=3,
        budget_usd=100.0,
    )
    return branch, report


@pytest.mark.parametrize(
    "adapter_cls",
    ADAPTER_CLASSES,
    ids=[cls.__name__ for cls in ADAPTER_CLASSES],
)
def test_adapter_replay_pass_through_and_fork_blame_parity(
    adapter_cls: type[BaseFrameworkAdapter],
) -> None:
    """(a) replay pass-through byte parity + digest invariance; (b) fork/blame
    results computed before vs. after the adapter's bind/replay/teardown cycle
    on the SAME tape object are bit-identical."""
    tape = _build_two_turn_tape()
    assert len(tape.exchanges) == 2

    branch_before, report_before = _run_fork_and_blame(tape)
    digest_before = tape.digest()

    # (a) the adapter's bind(mode="replay") serves byte-identical responses to
    # a bare TraceforkTransport replay of the SAME tape, for every exchange.
    bare_client = httpx.Client(transport=TraceforkTransport("replay", tape))
    adapter = adapter_cls()
    bind_result = adapter.bind(object(), tape, mode="replay", patch_uuid=False)
    try:
        assert bind_result.http_client is not None
        for request_body, response_body in tape.exchanges:
            bare_resp = bare_client.post(REPLAY_URL, content=request_body)
            adapter_resp = bind_result.http_client.post(REPLAY_URL, content=request_body)
            assert bare_resp.content == response_body
            assert adapter_resp.content == response_body
            assert bare_resp.content == adapter_resp.content
    finally:
        adapter.teardown()

    # Touching the tape through an adapter's replay bind never mutates it.
    assert tape.digest() == digest_before

    # (b) fork/blame parity: recomputed AFTER the adapter touch, on the SAME
    # tape object, must be bit-identical to the BEFORE snapshot.
    branch_after, report_after = _run_fork_and_blame(tape)

    assert branch_after.branch_digest != ""
    assert branch_after.branch_digest == branch_before.branch_digest
    assert branch_after.delta_tape.exchanges == branch_before.delta_tape.exchanges
    assert report_after.results == report_before.results
    assert report_after.responsible_set == report_before.responsible_set
