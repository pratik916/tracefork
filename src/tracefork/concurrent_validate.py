"""N-way concurrent-sibling true-negative-discrimination fixture and validator.

`competing_faults.py`'s `build_concurrent_gate_payload_tape`/`run_shapley_concurrent`
(`tracefork-bge.10`) proved that forwarding a REAL, recorded `tape.async_batches`
entry into `BlameEngine.shapley_rank` closes the single-ordering temporal-Shapley
blind spot for a 2-member GATE/PAYLOAD AND-conjunction: BOTH halves correctly read
`necessity=True`. But that fixture is symmetric -- both members are guilty by
design (`SCENARIO_GATE_PAYLOAD` lights up both markers) -- so it can prove the
engine credits a genuinely-necessary member, but it can NEVER prove the engine
correctly reads `necessity=False` on an INNOCENT sibling sitting in the SAME
unordered batch as a guilty one, because it has no innocent sibling to test
against.

This module answers that gap directly: `build_multi_branch_tape` records a
clean parent tape through `AsyncTraceforkTransport` where one setup turn is
followed by `n_branches` (default 3) sibling calls dispatched via a REAL
`asyncio.gather` -- none depending on another's not-yet-returned reply, so
`tape.async_batches` carries a genuine, non-hand-constructed batch entry for
all of them -- and a merge+final turn closes out the tape (mirrors
`competing_faults.py`'s `_concurrent_record_agent`/
`build_concurrent_gate_payload_tape` shape, generalized from 2 members to N).
`make_single_branch_perturb_factory(faulty_step)` lights up a fault marker
(`.faults.FAULT_MARKER_BYTES`, in a content field -- the same convention
`faults.py`'s five fault classes use) on EXACTLY ONE sibling step; every other
sibling (plus the setup/merge/final steps) gets the same inert, marker-free
filler. `_SingleCauseTail` is a single-cause rule -- FAIL iff the accumulated
request contains the marker -- simpler than `competing_faults.py`'s
GATE-AND-PAYLOAD conjunction, since exactly one cause is ever planted here.

Forwarding the fixture's real `tape.async_batches` into `shapley_rank` makes
`_batch_blocks`/`_block_orderings` (already in `blame.py`, `tracefork-bge.10`)
average the guilty sibling's marginal contribution over EVERY one of the
batch's internal join orders, for every possible guilty-branch position --
proving the engine both credits the actual guilty sibling (`necessity=True`)
AND correctly discriminates against its innocent siblings in the very same
batch (`necessity=False`), the true-negative proof the symmetric 2-way
fixture structurally cannot give.

`run_concurrent_branch_validation` sweeps every branch position as the
ground-truth-guilty one, plus one negative-control pass with no marker
anywhere, mirroring `validate.py`'s `ValidationRunner.run()`/
`run_all_fault_classes()`'s top-1-precision-plus-negative-control shape.

Zero-diff over the engines: this module only calls the existing public
`blame.py`/`transport.py`/`tape.py` API -- nothing here patches or extends
those files.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import anthropic
import httpx

from .blame import BlameEngine, ShapleyReport, StringMatchOracle
from .faults import FAULT_MARKER_BYTES
from .tape import Tape
from .transport import AsyncTraceforkTransport
from .wire import make_text_response

# ── tape shape ───────────────────────────────────────────────────────────────
#
# One setup turn, `n_branches` concurrent sibling turns, one merge turn, one
# final turn: `n_branches + 3` exchanges total. Branch siblings occupy
# positions `1..n_branches`; the merge sits at `n_branches + 1`, final at
# `n_branches + 2` (== the last position).

_FIXED_TURNS = 3  # setup + merge + final


def _total_turns(n_branches: int) -> int:
    return n_branches + _FIXED_TURNS


_MODEL_ID = "claude-sonnet-4-6"

NEUTRAL_TEXT = "ok, continuing"  # deliberately matches neither success_re nor failure_re
SUCCESS_TEXT = "SUCCESS - concurrent branch run complete"
FAIL_TEXT = "FAIL - concurrent branch fault triggered"

NEUTRAL_RESP = make_text_response(NEUTRAL_TEXT)
SUCCESS_RESP = make_text_response(SUCCESS_TEXT)
FAIL_RESP = make_text_response(FAIL_TEXT)
FAULT_RESP = make_text_response(f"branch fault triggered {FAULT_MARKER_BYTES.decode()}")


# ── record-time concurrency (a real asyncio.gather, never hand-constructed) ─
#
# Ascending per-branch delays (see `competing_faults.py`'s `_CONCURRENT_DELAYS`
# for the identical rationale): all `n_branches` calls are genuinely in-flight
# together (a real overlap `AsyncTraceforkTransport` can log as one batch), but
# an ascending delay by dispatch order means completion order always matches
# dispatch order too -- a fixed, non-racy mapping from branch index `i` to tape
# position `i + 1`, which the SYNC replay agent's own (sequential, i=0..n-1)
# call order must reproduce bit-exact during a fork's prefix-replay.
_BRANCH_BASE_DELAY = 0.02
_BRANCH_DELAY_STEP = 0.01


class _ConcurrentBranchFiller(httpx.AsyncBaseTransport):
    """Serves the clean NEUTRAL/SUCCESS filler for `build_multi_branch_tape`'s
    recording, keyed by call-ENTRY order (0..n_total-1) rather than a fixed
    list, since `n_branches` calls are in flight at once and only entry order
    -- not response order -- is well-defined for them (mirrors
    `competing_faults.py`'s `_ConcurrentNeutralFiller`). Every branch position
    sleeps an ascending amount (see module note above) so all `n_branches`
    calls are genuinely in-flight together and complete in call-entry order;
    every other position completes immediately."""

    def __init__(self, n_branches: int) -> None:
        self._n_branches = n_branches
        self._n_total = _total_turns(n_branches)
        self._n = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        pos = self._n
        self._n += 1
        if 1 <= pos <= self._n_branches:
            await asyncio.sleep(_BRANCH_BASE_DELAY + (pos - 1) * _BRANCH_DELAY_STEP)
        body = SUCCESS_RESP if pos == self._n_total - 1 else NEUTRAL_RESP
        return httpx.Response(200, headers={"content-type": "application/json"}, content=body)


def _echo_text(msg: Any) -> str:
    parts: list[str] = []
    for block in msg.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return " | ".join(parts) or "(empty)"


def _initial_messages() -> list[dict[str, Any]]:
    return [{"role": "user", "content": "start"}]


def _branch_request_messages(
    branch_base: list[dict[str, Any]], slot_text: str
) -> list[dict[str, Any]]:
    """A sibling branch's own request: built directly from the shared
    post-setup state, WITHOUT chaining through any other branch's reply
    (there isn't one yet at record time) -- the thing that makes every
    sibling genuinely, not just nominally, independent."""
    return [*branch_base, {"role": "user", "content": slot_text}]


def _merge_messages(
    branch_base: list[dict[str, Any]], branch_replies: list[str]
) -> list[dict[str, Any]]:
    """The merge turn: every sibling's reply folded into ONE assistant turn
    appended to the shared post-setup state -- the one place a marker planted
    in ANY sibling becomes visible to the tail's single-cause check."""
    merged = " | ".join(branch_replies)
    return [
        *branch_base,
        {"role": "assistant", "content": merged},
        {"role": "user", "content": "continue"},
    ]


async def _record_multi_branch_agent(client: anthropic.AsyncAnthropic, n_branches: int) -> str:
    """Async recording agent for `build_multi_branch_tape`: one setup turn,
    then `n_branches` sibling turns dispatched via a REAL `asyncio.gather` --
    all built from the SAME post-setup state, none aware of the others -- then
    one merge turn and one final turn. `_replay_multi_branch_agent` is the
    sync replay of this exact conversation shape, reused for every fork trial."""
    messages = _initial_messages()
    resp0 = await client.messages.create(
        model=_MODEL_ID, max_tokens=100, messages=cast(Any, list(messages))
    )
    messages.append({"role": "assistant", "content": _echo_text(resp0)})
    messages.append({"role": "user", "content": "continue"})
    branch_base = list(messages)

    async def _branch(i: int) -> Any:
        return await client.messages.create(
            model=_MODEL_ID,
            max_tokens=100,
            messages=cast(Any, _branch_request_messages(branch_base, f"branch-{i}-check")),
        )

    branch_resps = await asyncio.gather(*(_branch(i) for i in range(n_branches)))

    messages = _merge_messages(branch_base, [_echo_text(r) for r in branch_resps])
    resp_merge = await client.messages.create(
        model=_MODEL_ID, max_tokens=100, messages=cast(Any, messages)
    )
    messages.append({"role": "assistant", "content": _echo_text(resp_merge)})
    messages.append({"role": "user", "content": "continue"})
    resp_final = await client.messages.create(
        model=_MODEL_ID, max_tokens=100, messages=cast(Any, messages)
    )
    return _echo_text(resp_final)


def _replay_multi_branch_agent(client: anthropic.Anthropic, n_branches: int) -> str:
    """Sync replay of `_record_multi_branch_agent`'s exact conversation shape,
    used as `agent_fn` (via `_make_replay_agent`) for every fork/coalition
    trial over the concurrently-recorded tape (`ForkEngine`'s replay transport
    is sync-only -- see `fork.py` -- so this never needs to actually race; it
    only needs to reproduce the SAME request bytes turn-by-turn that the async
    recording produced)."""
    messages = _initial_messages()
    resp0 = client.messages.create(
        model=_MODEL_ID, max_tokens=100, messages=cast(Any, list(messages))
    )
    messages.append({"role": "assistant", "content": _echo_text(resp0)})
    messages.append({"role": "user", "content": "continue"})
    branch_base = list(messages)

    branch_resps = [
        client.messages.create(
            model=_MODEL_ID,
            max_tokens=100,
            messages=cast(Any, _branch_request_messages(branch_base, f"branch-{i}-check")),
        )
        for i in range(n_branches)
    ]

    messages = _merge_messages(branch_base, [_echo_text(r) for r in branch_resps])
    resp_merge = client.messages.create(
        model=_MODEL_ID, max_tokens=100, messages=cast(Any, messages)
    )
    messages.append({"role": "assistant", "content": _echo_text(resp_merge)})
    messages.append({"role": "user", "content": "continue"})
    resp_final = client.messages.create(
        model=_MODEL_ID, max_tokens=100, messages=cast(Any, messages)
    )
    return _echo_text(resp_final)


def _make_replay_agent(n_branches: int) -> Callable[[anthropic.Anthropic], str]:
    """Bind `n_branches` into a single-arg `agent_fn` -- `ForkEngine.fork()`/
    `fork_coalition()` always call it as `agent_fn(client)`."""

    def agent(client: anthropic.Anthropic) -> str:
        return _replay_multi_branch_agent(client, n_branches)

    return agent


async def _build_multi_branch_tape_async(n_branches: int) -> Tape:
    tape = Tape(agent_name="concurrent_validate_multi_branch_agent")
    transport = AsyncTraceforkTransport("record", tape, _ConcurrentBranchFiller(n_branches))
    client = anthropic.AsyncAnthropic(
        api_key="sk-ant-fake",
        http_client=httpx.AsyncClient(transport=transport),
        max_retries=0,
    )
    await _record_multi_branch_agent(client, n_branches)
    return tape


def build_multi_branch_tape(n_branches: int = 3) -> Tape:
    """Record the clean (unperturbed) parent tape THROUGH THE ASYNC
    TRANSPORT, with `n_branches` sibling calls dispatched via a real
    `asyncio.gather` -- so `tape.async_batches` carries a GENUINE
    concurrency-batch entry `[[1, ..., n_branches]]` for all of them (never
    hand-constructed)."""
    return asyncio.run(_build_multi_branch_tape_async(n_branches))


# ── perturbation: exactly one guilty sibling, everyone else inert ──────────


class _SingleCauseTail(httpx.BaseTransport):
    """Serves the rest of the multi-branch agent's turns, adjudicating
    FAIL-vs-benign from a SINGLE-CAUSE rule -- FAIL iff the incoming request's
    own (already-cumulative) body contains `FAULT_MARKER_BYTES` -- simpler
    than `competing_faults.py`'s AND-conjunction `_fails`, since exactly one
    marker is ever planted here. Returns an explicit SUCCESS on the final call
    it expects to see (`remaining_turns`) when no marker is present, so every
    trial this backs grades unambiguously."""

    def __init__(self, remaining_turns: int) -> None:
        self._remaining = remaining_turns
        self._seen = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self._seen += 1
        is_last = self._seen >= self._remaining
        if FAULT_MARKER_BYTES in request.content:
            body = FAIL_RESP
        elif is_last:
            body = SUCCESS_RESP
        else:
            body = NEUTRAL_RESP
        return httpx.Response(200, headers={"content-type": "application/json"}, content=body)


def make_single_branch_perturb_factory(
    faulty_step: int, *, n_branches: int = 3
) -> Callable[[int], tuple[bytes, Any]]:
    """Build a `perturb_factory` for `blame.py`'s `shapley_rank()` that lights
    up `FAULT_MARKER_BYTES` on EXACTLY ONE sibling step (`faulty_step`, one of
    the `n_branches` positions `1..n_branches`); every other candidate step
    (the other siblings, plus setup/merge/final) gets the same inert,
    marker-free filler, so the one shared tape can host every guilty-position
    scenario without cross-scenario masking."""
    if not 1 <= faulty_step <= n_branches:
        raise ValueError(
            f"faulty_step must be one of the {n_branches} branch positions "
            f"(1..{n_branches}), got {faulty_step}"
        )
    n_total = _total_turns(n_branches)

    def factory(step_idx: int) -> tuple[bytes, Any]:
        mutated = FAULT_RESP if step_idx == faulty_step else NEUTRAL_RESP
        remaining = n_total - (step_idx + 1)
        return mutated, _SingleCauseTail(remaining)

    return factory


def _make_null_perturb_factory(n_branches: int) -> Callable[[int], tuple[bytes, Any]]:
    """The negative-control `perturb_factory`: no marker anywhere, so a high
    precision score from `run_concurrent_branch_validation` isn't vacuous."""
    n_total = _total_turns(n_branches)

    def factory(step_idx: int) -> tuple[bytes, Any]:
        remaining = n_total - (step_idx + 1)
        return NEUTRAL_RESP, _SingleCauseTail(remaining)

    return factory


# ── scoring ──────────────────────────────────────────────────────────────────

_BUDGET_USD = 1_000_000.0  # generous and fixed: this fixture is offline/$0 regardless


def run_shapley_multi_branch(
    faulty_step: int, *, n_branches: int = 3, k: int = 3, m_samples: int = 2
) -> ShapleyReport:
    """Record a fresh clean multi-branch tape and run `BlameEngine.
    shapley_rank` over it with `faulty_step`'s marker lit up, forwarding the
    tape's REAL `tape.async_batches` (see `build_multi_branch_tape`) so the
    coalition walk exact-averages every internal join order of the
    `n_branches`-way fan-out -- the N-way generalization of
    `competing_faults.py`'s 2-member GATE/PAYLOAD batch."""
    tape = build_multi_branch_tape(n_branches)
    oracle = StringMatchOracle(success_re=r"SUCCESS", failure_re=r"FAIL")
    factory = make_single_branch_perturb_factory(faulty_step, n_branches=n_branches)
    agent_fn = _make_replay_agent(n_branches)
    return BlameEngine.shapley_rank(
        tape,
        agent_fn,
        oracle,
        perturb_factory=factory,
        k=k,
        m_samples=m_samples,
        budget_usd=_BUDGET_USD,
        async_batches=tape.async_batches,
    )


def run_shapley_negative_control(
    *, n_branches: int = 3, k: int = 3, m_samples: int = 2
) -> ShapleyReport:
    """Like `run_shapley_multi_branch`, but with no marker planted anywhere --
    every step, including every batch member, must read `necessity=False` and
    a near-zero `shapley_value`, or a high top-1 precision score would be
    meaningless."""
    tape = build_multi_branch_tape(n_branches)
    oracle = StringMatchOracle(success_re=r"SUCCESS", failure_re=r"FAIL")
    factory = _make_null_perturb_factory(n_branches)
    agent_fn = _make_replay_agent(n_branches)
    return BlameEngine.shapley_rank(
        tape,
        agent_fn,
        oracle,
        perturb_factory=factory,
        k=k,
        m_samples=m_samples,
        budget_usd=_BUDGET_USD,
        async_batches=tape.async_batches,
    )


@dataclass
class ConcurrentValidationReport:
    """Mirrors `validate.py`'s `ValidationReport` field shape, generalized to
    an N-way concurrent-sibling batch instead of a single fault class."""

    n_branches: int
    n_runs: int
    top1_correct: int
    top1_precision: float
    negative_control_max_shapley: float = 0.0


def run_concurrent_branch_validation(
    n_branches: int = 3, *, k: int = 3, m_samples: int = 2
) -> ConcurrentValidationReport:
    """Sweep every branch position as the ground-truth-guilty one (mirrors
    `validate.py`'s `ValidationRunner.run()` loop), plus one negative-control
    pass with no marker anywhere -- exactly mirroring `run_all_fault_classes()`'s
    top-1-precision-plus-negative-control shape."""
    top1_correct = 0
    for faulty_step in range(1, n_branches + 1):
        report = run_shapley_multi_branch(
            faulty_step, n_branches=n_branches, k=k, m_samples=m_samples
        )
        top = report.top()
        if top is not None and top.step_index == faulty_step:
            top1_correct += 1

    control = run_shapley_negative_control(n_branches=n_branches, k=k, m_samples=m_samples)
    negative_control_max_shapley = max((r.shapley_value for r in control.results), default=0.0)

    n_runs = n_branches
    precision = top1_correct / n_runs if n_runs > 0 else 0.0
    return ConcurrentValidationReport(
        n_branches=n_branches,
        n_runs=n_runs,
        top1_correct=top1_correct,
        top1_precision=precision,
        negative_control_max_shapley=negative_control_max_shapley,
    )
