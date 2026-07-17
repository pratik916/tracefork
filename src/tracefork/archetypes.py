"""Declarative fault-scenario GENERATOR: generalizes `competing_faults.py`'s
fixed 7-step fixture (`StepRole`/`make_perturb_factory`/`RuleBasedTail`/
`run_shapley` over ONE hardcoded tape) into a PARAMETERIZED one -- caller-chosen
chain length and role placement instead of six hardcoded step indices -- so new
hand-derivable causal shapes can be added without hand-building a new tape and
tail transport each time (`tracefork-bge.34`).

Three shared, generic primitives do the work `competing_faults.py` hardcodes
per-fixture:

  - `_ArchetypeTail`      -- like `competing_faults.RuleBasedTail`, but takes an
                             INJECTED `fails: Callable[[bytes], bool]` instead
                             of a module-fixed rule, so every archetype below
                             supplies its own failure predicate over the same
                             tail-serving mechanics.
  - `make_linear_agent`   -- generalizes `competing_fault_agent`'s hardcoded
                             `N_TURNS`-turn echo-chain loop to an arbitrary
                             chain length: every turn's request still carries
                             the FULL prior transcript, so a marker introduced
                             at any turn stays visible to every later turn.
  - `_make_perturb_factory` -- generalizes `make_perturb_factory`: `role_positions`
                             maps a caller-chosen SUBSET of step indices to
                             role-name strings (rather than a fixed `StepRole`
                             enum keyed 0..N_TURNS-1), so archetypes can place
                             roles at any position and chain length. It also
                             guards `competing_faults.py`'s documented "never
                             grade the tape's last exchange" invariant: a role
                             at the tape's FINAL slot raises `ValueError` at
                             factory-build time, before any tape is recorded.

Three NEW, hand-derivable archetypes are built on these primitives, each
verified against `BlameEngine.shapley_rank`'s exact documented semantics
(`phi_i = v({0..i}) - v({0..i-1})` for necessity; independent single-step
flip-rate, reused from `rank()`, for sufficiency):

  `run_or_redundancy(pos_a, pos_b, n_turns, active=...)` -- two INDEPENDENTLY-
      sufficient OR-causes with NO shared marker text (unlike ROOT/ECHO's
      embedded-string trick, where ECHO literally re-embeds ROOT_MARKER).
      With both lit up: the earlier position reads `necessity=True,
      sufficiency=True` (nothing masks it -- it's the first fault the
      temporal walk sees) and the later position reads `necessity=False`
      (its marginal is measured AFTER the earlier fault has already flipped
      the coalition) but `sufficiency=True` (forcing it ALONE, via `rank()`'s
      independent single-step trial, still flips the run) -- the OR-mirror of
      `competing_faults.py`'s ROOT/ECHO case. Run with only ONE cause lit up
      (`active=OR_CAUSE_A` / `OR_CAUSE_B`), that ONE position reads BOTH
      `necessity=True` and `sufficiency=True` (it is now the sole cause, so
      nothing masks its necessity) -- proving each cause is independently
      sufficient on its own, not merely because the other happens to be
      present in the tape.

  `run_n_way_conjunction(arity)` -- generalizes the 2-part GATE/PAYLOAD AND to
      a parameterized k-part AND (`arity` markers, all required for
      `fails`). Only the LAST-joining part (the highest step index) ever
      reads `necessity=True`; every earlier part reads `necessity=False`
      (its own marginal is measured before the AND is complete -- the same
      documented, temporal-order limitation `competing_faults.py`'s
      SCENARIO_GATE_PAYLOAD exercises for arity=2, here proven to scale
      across arity=2..5) and EVERY part reads `sufficiency=False` (no single
      part alone ever completes a >=2-part AND).

  `run_long_relay(n_relay)` -- a ROOT marker at position 0, propagated through
      a parameterized-length chain of `n_relay` inert NEUTRAL relay steps
      before the tail. Proves necessity/sufficiency attribution for a lone
      root cause is INVARIANT to how long the propagation chain is
      (`n_relay=1,5,10` all resolve identically) -- a scaling proof point the
      fixed 7-step fixture cannot make on its own.

Rescoped, per the backlog's own note, to exactly these 3 declarative
archetypes with concrete parameters (position/arity/n_relay) -- not an
open-ended scenario DSL or config-driven spec language. No CLI/`bench.py`
wiring in this slice; a follow-up could add a `tracefork archetypes`
subcommand analogous to `tracefork bench`.

Zero-diff over the engines: this module only calls the existing public
`blame.py` API (`BlameEngine.shapley_rank`, `StringMatchOracle`) and builds
tapes through the existing `transport.py`/`tape.py`/`synthetic.py`/`wire.py`
seam -- exactly like `competing_faults.py` does. Offline/$0: every tail is a
synthetic transport, never a real API key or network call.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

import anthropic
import httpx

from .blame import BlameEngine, ShapleyReport, StringMatchOracle
from .synthetic import ScriptedFakeLLM
from .tape import Tape
from .transport import TraceforkTransport
from .wire import make_text_response

SUCCESS_TEXT = "SUCCESS - archetype run complete"
FAIL_TEXT = "FAIL - archetype fault triggered"
NEUTRAL_TEXT = "ok, continuing"  # deliberately matches neither success_re nor failure_re

SUCCESS_RESP = make_text_response(SUCCESS_TEXT)
FAIL_RESP = make_text_response(FAIL_TEXT)
NEUTRAL_RESP = make_text_response(NEUTRAL_TEXT)

_MODEL_ID = "claude-sonnet-4-6"


# ── generic primitives ───────────────────────────────────────────────────────


def _marker_bytes(role: str) -> bytes:
    """The unique marker bytes for a role name. Distinct role names never
    collide (Python string equality), which is what lets `run_or_redundancy`
    plant two INDEPENDENT causes with no shared marker text -- unlike
    `competing_faults.py`'s ECHO, which deliberately re-embeds ROOT_MARKER."""
    return f"ARCHETYPE_MARKER::{role}".encode()


def _mutated_response_for(role: str) -> bytes:
    """The marker-carrying response bytes for an active role."""
    return make_text_response(f"{role} fault marker fired {_marker_bytes(role).decode()}")


class _ArchetypeTail(httpx.BaseTransport):
    """Generalizes `competing_faults.RuleBasedTail`: serves the rest of a
    `make_linear_agent` agent's turns by adjudicating FAIL-vs-benign from an
    INJECTED `fails` predicate applied to each incoming request's own
    (already-cumulative) body, instead of one module-fixed rule -- so the same
    tail-serving mechanics back every archetype below, each with its own
    failure predicate. Returns an explicit SUCCESS/FAIL text on the final call
    it expects to see (`remaining_turns`), so every trial this backs grades
    unambiguously."""

    def __init__(self, remaining_turns: int, fails: Callable[[bytes], bool]) -> None:
        self._remaining = remaining_turns
        self._fails = fails
        self._seen = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self._seen += 1
        is_last = self._seen >= self._remaining
        body = (
            FAIL_RESP
            if self._fails(request.content)
            else (SUCCESS_RESP if is_last else NEUTRAL_RESP)
        )
        return httpx.Response(200, headers={"content-type": "application/json"}, content=body)


def _echo_text(msg: Any) -> str:
    parts: list[str] = []
    for block in msg.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return " | ".join(parts) or "(empty)"


def make_linear_agent(n_turns: int) -> Callable[[anthropic.Anthropic], str]:
    """Generalizes `competing_faults.competing_fault_agent`'s hardcoded
    `N_TURNS`-turn loop to an arbitrary chain length: every turn's request
    carries the FULL prior transcript (all previous user asks + echoed
    assistant replies), so a marker introduced at any turn stays visible to
    every later turn -- the propagation mechanism every archetype below relies
    on."""
    if n_turns < 1:
        raise ValueError(f"n_turns must be >= 1, got {n_turns}")

    def agent(client: anthropic.Anthropic) -> str:
        messages: list[dict[str, Any]] = [{"role": "user", "content": "start"}]
        last_text = ""
        for turn in range(n_turns):
            resp = client.messages.create(
                model=_MODEL_ID, max_tokens=100, messages=cast(Any, list(messages))
            )
            last_text = _echo_text(resp)
            messages.append({"role": "assistant", "content": last_text})
            if turn < n_turns - 1:
                messages.append({"role": "user", "content": "continue"})
        return last_text

    return agent


def _make_perturb_factory(
    role_positions: dict[int, str],
    active: frozenset[str],
    fails: Callable[[bytes], bool],
    n_turns: int,
) -> Callable[[int], tuple[bytes, Any]]:
    """Generalizes `competing_faults.make_perturb_factory`: `role_positions`
    maps a caller-chosen SUBSET of step indices to role-name strings (rather
    than a fixed `StepRole` enum keyed 0..N_TURNS-1). Every step index not a
    key of `role_positions`, and every role not in `active`, gets the same
    inert, marker-free filler -- so a scenario can light up any subset of an
    archetype's roles without cross-role masking.

    Guards `competing_faults.py`'s documented "never grade the tape's last
    exchange" invariant: a role placed at the tape's FINAL slot (index
    `n_turns - 1`) raises `ValueError` here, at factory-build time, before any
    tape is even recorded -- forcing that slot (or including it as the top
    member of the full coalition) would make its own raw injected bytes
    literally BE the graded text, bypassing `_ArchetypeTail`'s rule-based
    adjudication entirely (see `competing_faults.py`'s module docstring).
    """
    if n_turns < 1:
        raise ValueError(f"n_turns must be >= 1, got {n_turns}")
    for pos in role_positions:
        if not 0 <= pos < n_turns:
            raise ValueError(f"role position {pos} is out of range for n_turns={n_turns}")
        if pos == n_turns - 1:
            raise ValueError(
                f"role position {pos} is the tape's FINAL slot (index {n_turns - 1}) -- "
                "never grade the tape's last exchange (see competing_faults.py's module "
                "docstring for why)"
            )

    def factory(step_idx: int) -> tuple[bytes, Any]:
        role = role_positions.get(step_idx)
        mutated = NEUTRAL_RESP
        if role is not None and role in active:
            mutated = _mutated_response_for(role)
        remaining = n_turns - (step_idx + 1)
        return mutated, _ArchetypeTail(remaining, fails)

    return factory


def _build_clean_tape(n_turns: int, agent_name: str) -> Tape:
    """Record the clean (unperturbed) `n_turns`-exchange parent tape: no
    markers anywhere, so no archetype's `fails` rule ever trips and the run
    grades SUCCESS -- generalizes
    `competing_faults.build_competing_fault_tape`'s pattern to a
    parameterized chain length."""
    scripted = [NEUTRAL_RESP] * (n_turns - 1) + [SUCCESS_RESP]
    fake = ScriptedFakeLLM(scripted)
    tape = Tape(agent_name=agent_name)
    transport = TraceforkTransport("record", tape, fake)
    client = anthropic.Anthropic(
        api_key="sk-ant-fake",
        http_client=httpx.Client(transport=transport),
        max_retries=0,
    )
    make_linear_agent(n_turns)(client)
    return tape


def _run_archetype_shapley(
    n_turns: int,
    agent_name: str,
    role_positions: dict[int, str],
    active: frozenset[str],
    fails: Callable[[bytes], bool],
    *,
    k: int,
    m_samples: int,
) -> ShapleyReport:
    """Shared plumbing every `run_*` archetype below delegates to: build the
    perturb factory (validating role placement FIRST, before recording
    anything), record a fresh clean tape, then run `BlameEngine.shapley_rank`
    over it. `budget_usd` is generous and fixed: every archetype here is
    offline/$0 regardless (every tail is a fake transport)."""
    factory = _make_perturb_factory(role_positions, active, fails, n_turns)
    tape = _build_clean_tape(n_turns, agent_name)
    oracle = StringMatchOracle(success_re=r"SUCCESS", failure_re=r"FAIL")
    agent_fn = make_linear_agent(n_turns)
    return BlameEngine.shapley_rank(
        tape,
        agent_fn,
        oracle,
        perturb_factory=factory,
        k=k,
        m_samples=m_samples,
        budget_usd=1_000_000.0,
    )


@dataclass
class ExpectedCase:
    """One step's hand-derived (necessity, sufficiency) ground truth for an
    archetype run -- generalizes `bench.py`'s `CaseResult` fields to a
    parameterized generator instead of one fixed fixture's named cases."""

    step_index: int
    role: str
    expected_necessity: bool
    expected_sufficiency: bool


@dataclass
class ArchetypeResult:
    """A parameterized archetype's `ShapleyReport` plus its hand-derived
    per-step ground truth, so a caller (or test) can check the engine's
    actual reading against the archetype's expected causal shape without
    re-deriving it inline."""

    report: ShapleyReport
    expected: list[ExpectedCase] = field(default_factory=list)

    def matches_expected(self) -> bool:
        """True iff every expected case's (necessity, sufficiency) matches
        the engine's actual reading for that step -- mirrors `bench.py`'s
        `CaseResult.resolved` check, generalized across all of this
        archetype's expected cases at once."""
        by_step = {r.step_index: r for r in self.report.results}
        return all(
            by_step[e.step_index].necessity == e.expected_necessity
            and by_step[e.step_index].sufficiency == e.expected_sufficiency
            for e in self.expected
        )


# ── archetype 1: OR-redundancy (two independently-sufficient OR-causes) ────

_CAUSE_A = "cause_a"
_CAUSE_B = "cause_b"
OR_CAUSE_A: frozenset[str] = frozenset({_CAUSE_A})
OR_CAUSE_B: frozenset[str] = frozenset({_CAUSE_B})
OR_BOTH: frozenset[str] = frozenset({_CAUSE_A, _CAUSE_B})


def _or_fails(accumulated: bytes) -> bool:
    """Two INDEPENDENT OR-causes -- no shared marker text, unlike
    `competing_faults.py`'s ECHO (which re-embeds ROOT_MARKER verbatim):
    either `cause_a`'s or `cause_b`'s own, distinct marker alone triggers
    failure."""
    return _marker_bytes(_CAUSE_A) in accumulated or _marker_bytes(_CAUSE_B) in accumulated


def _expected_or_redundancy(pos_a: int, pos_b: int, active: frozenset[str]) -> list[ExpectedCase]:
    role_at = {pos_a: _CAUSE_A, pos_b: _CAUSE_B}
    active_positions = sorted(p for p, r in role_at.items() if r in active)
    expected: list[ExpectedCase] = []
    for pos, role in role_at.items():
        if role not in active:
            expected.append(ExpectedCase(pos, role, False, False))
            continue
        is_first_active = pos == active_positions[0]
        expected.append(ExpectedCase(pos, role, is_first_active, True))
    return expected


def run_or_redundancy(
    pos_a: int,
    pos_b: int,
    n_turns: int,
    *,
    active: frozenset[str] = OR_BOTH,
    k: int = 3,
    m_samples: int = 2,
) -> ArchetypeResult:
    """Two INDEPENDENTLY-sufficient OR-causes at `pos_a` < `pos_b`, over an
    `n_turns`-exchange tape -- the OR-mirror of `competing_faults.py`'s
    ROOT/ECHO AND case, but with no shared marker text between the two
    causes (see `_or_fails`).

    With `active=OR_BOTH` (the default): `pos_a` (the earlier, first-seen
    fault) reads `necessity=True, sufficiency=True`; `pos_b` reads
    `necessity=False` (its marginal is measured AFTER `pos_a`'s fault has
    already flipped the coalition) but `sufficiency=True` (forcing it ALONE,
    with `pos_a` clean, still flips the run -- `rank()`'s independent
    single-step trial). With only ONE of `OR_CAUSE_A`/`OR_CAUSE_B` active,
    that one position reads BOTH `necessity=True` and `sufficiency=True` --
    it is now the tape's sole cause, so nothing masks its necessity, proving
    each cause is independently sufficient on its own merit.
    """
    if not (0 <= pos_a < pos_b < n_turns):
        raise ValueError(
            f"require 0 <= pos_a < pos_b < n_turns, got pos_a={pos_a}, pos_b={pos_b}, "
            f"n_turns={n_turns}"
        )
    if not active <= OR_BOTH:
        raise ValueError(f"active roles must be a subset of {OR_BOTH}")

    role_positions = {pos_a: _CAUSE_A, pos_b: _CAUSE_B}
    report = _run_archetype_shapley(
        n_turns,
        "or_redundancy_agent",
        role_positions,
        active,
        _or_fails,
        k=k,
        m_samples=m_samples,
    )
    return ArchetypeResult(report=report, expected=_expected_or_redundancy(pos_a, pos_b, active))


# ── archetype 2: N-way conjunction (parameterized k-part AND) ───────────────


def _part_role(i: int) -> str:
    return f"part_{i}"


def _make_conjunction_fails(arity: int) -> Callable[[bytes], bool]:
    markers = [_marker_bytes(_part_role(i)) for i in range(arity)]

    def fails(accumulated: bytes) -> bool:
        return all(marker in accumulated for marker in markers)

    return fails


def _expected_n_way_conjunction(arity: int) -> list[ExpectedCase]:
    return [ExpectedCase(i, _part_role(i), i == arity - 1, False) for i in range(arity)]


def run_n_way_conjunction(
    arity: int,
    *,
    k: int = 3,
    m_samples: int = 2,
) -> ArchetypeResult:
    """Generalizes `competing_faults.py`'s 2-part GATE/PAYLOAD AND to a
    parameterized `arity`-part AND: parts occupy step indices `0..arity-1`
    (consecutively, tape length `arity + 1` so the last part is never the
    FINAL slot), and `fails` requires EVERY part's marker to be present.

    Only the LAST-joining part (index `arity - 1`) ever reads
    `necessity=True` -- this is the SAME documented, temporal-order-
    restricted-Shapley limitation `competing_faults.py`'s
    SCENARIO_GATE_PAYLOAD exercises for `arity=2` (see `blame.py`'s
    `shapley_rank` docstring), here proven to scale identically across
    `arity=2..5`: every EARLIER part's own marginal is measured before the
    AND is complete, so it reads `necessity=False` despite being genuinely
    necessary. No part ever reads `sufficiency=True` -- no single part alone
    ever completes a `>= 2`-part AND (`rank()`'s independent single-step
    trial for any one part, with every other part clean, never trips
    `fails`).
    """
    if arity < 2:
        raise ValueError(f"arity must be >= 2, got {arity}")

    n_turns = arity + 1
    role_positions = {i: _part_role(i) for i in range(arity)}
    active = frozenset(role_positions.values())
    report = _run_archetype_shapley(
        n_turns,
        "n_way_conjunction_agent",
        role_positions,
        active,
        _make_conjunction_fails(arity),
        k=k,
        m_samples=m_samples,
    )
    return ArchetypeResult(report=report, expected=_expected_n_way_conjunction(arity))


# ── archetype 3: long relay (root propagated through a parameterized chain) ─

_ROOT = "root"
LONG_RELAY_ROOT: frozenset[str] = frozenset({_ROOT})


def _root_fails(accumulated: bytes) -> bool:
    return _marker_bytes(_ROOT) in accumulated


def _expected_long_relay(n_relay: int) -> list[ExpectedCase]:
    expected = [ExpectedCase(0, _ROOT, True, True)]
    for i in range(1, n_relay + 1):
        expected.append(ExpectedCase(i, "relay_decoy", False, False))
    return expected


def run_long_relay(
    n_relay: int,
    *,
    k: int = 3,
    m_samples: int = 2,
) -> ArchetypeResult:
    """A single ROOT marker at position 0, propagated through a
    parameterized-length chain of `n_relay` inert NEUTRAL relay steps before
    the tail (tape length `n_relay + 2`: ROOT, `n_relay` relay decoys, one
    FINAL closing turn).

    ROOT reads `necessity=True, sufficiency=True, shapley_value=1.0`
    regardless of `n_relay` -- proving necessity/sufficiency attribution for
    a lone root cause is INVARIANT to how long the propagation chain
    downstream of it is. Every relay/decoy position (which carries no role
    at all -- true-negative filler, exactly like `competing_faults.py`'s
    NEUTRAL steps) reads `necessity=False, sufficiency=False` regardless of
    chain length too.
    """
    if n_relay < 0:
        raise ValueError(f"n_relay must be >= 0, got {n_relay}")

    n_turns = n_relay + 2
    role_positions = {0: _ROOT}
    report = _run_archetype_shapley(
        n_turns,
        "long_relay_agent",
        role_positions,
        LONG_RELAY_ROOT,
        _root_fails,
        k=k,
        m_samples=m_samples,
    )
    return ArchetypeResult(report=report, expected=_expected_long_relay(n_relay))
