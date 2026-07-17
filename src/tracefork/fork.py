"""Fork engine: create a counterfactual branch at any step.

`ForkEngine.fork()` re-runs the *same* agent that produced the parent tape,
but intercepts its requests in three phases:

  1. prefix   (requests 0..k-1) — replayed from the parent tape, $0, and the
     request body is sha256-asserted to match (the agent is deterministic up
     to the fork point, so this must hold or the agent code changed);
  2. mutation (request k = divergence_step) — the request still matches the
     parent (the agent hasn't seen the mutated response yet), but instead of
     the recorded response we serve `spec.mutated_response`;
  3. tail     (requests k+1..) — the agent is now in counterfactual territory;
     its requests no longer match the parent, so they are recorded fresh.

The returned `Branch.delta_tape` holds only the exchanges from the divergence
step onward (the mutation exchange + any tail). The expensive prefix lives in
the parent tape and is never re-paid for — that is the "fork for $0 up to the
divergence point" property.

`CoalitionSpec` / `ForkEngine.fork_coalition()` generalize the same three-phase
idea from a single divergence step to a SET of steps forced jointly — the
intervention primitive coalition/Shapley blame needs to compute a coalition's
flip-rate `v(S)`. Only the *first* (lowest-index) intervention point is still
request-matched against the parent tape (that is the true point of first
divergence); every later intervention in the coalition is forced unconditionally,
since by then the agent's requests already diverge from the parent because an
earlier step was perturbed. `BranchSpec`/`ForkTransport`/`ForkEngine.fork()` are
unchanged — `fork_coalition` is purely additive.

Every `Branch` also carries a content-addressed `branch_digest`
(`compute_branch_digest`) folding the parent tape's and delta tape's own
digests plus the intervened steps into one sha256 — Merkle-DAG identity, so
`store.py` can key branches by content, resolve fork-of-fork chains, and
answer inverse-citation queries as plain reachability walks. Branch/store-
level metadata only: `Tape.digest()` itself is completely untouched.

`Branch.parent_tape_digest` (the parent tape's own `digest()` at fork time)
and `Branch.divergence_exchange_digest` (`compute_divergence_exchange_digest`
— sha256 of the exact request+response bytes at the fork's first divergence
point) are the citable, write-time half of a Certificate-Transparency-style
inclusion proof: `store.py`'s `load_branch` re-verifies `parent_tape_digest`
against the parent tape's CURRENT digest on every read, hard-erroring
(`ForkPointDriftError`) if the two no longer match — the retrospective
complement to the CAS write-time guard, catching drift a write-time-only
check would leave undetected. Branch/store-level metadata only, same as
`branch_digest`.

`ForkEngine.rebase(old_branch, new_parent_tape, agent_fn, ...)` is the
version-control-rebase analogue of `fork()`/`fork_coalition()`: re-run
`agent_fn` from the start, but source the UNCHANGED prefix (before
`old_branch.divergence_step`) from `new_parent_tape` instead of
`old_branch.parent_tape` — raising `DivergenceError` if that prefix itself
moved, same contract as the classic prefix assert — then re-force every one
of `old_branch.intervened_steps` to its ORIGINAL stored response (looked up
from `old_branch.delta_tape`, whose exchange at offset `step -
old_branch.divergence_step` is exactly the exchange recorded for absolute
step `step` — true for both `fork()` and `fork_coalition()`, since every
absolute step from `divergence_step` onward triggers exactly one
`delta_tape.append_exchange` call). Anything else beyond the fork point
first tries reusing `old_branch.delta_tape`'s own tail exchange by
fingerprint match ($0, `Branch.tail_reused`) before falling through to live
re-recording (`Branch.tail_recorded`) — rr/FoundationDB's "pin nondeterminism
at exact points" principle applied to a moved base: reuse the unchanged
prefix and prior interventions verbatim, only pay for genuinely new/changed
continuation. `ForkTransport`/`CoalitionForkTransport`/`fork()`/
`fork_coalition()` are completely unchanged; `rebase` is a new, parallel
static method.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import anthropic
import httpx

from .boundary_guard import BoundaryGuard, ConfinementSpec
from .constants import (
    CONFINEMENT_TIER_DECLARED,
    CONFINEMENT_TIER_GUARDED,
    CONFINEMENT_TIER_NONE,
)
from .nondet import DivergenceError
from .observability import instrument
from .tape import Tape, sha256_hex


@dataclass
class BranchSpec:
    divergence_step: int
    mutated_response: bytes
    mutation_desc: str = ""


@dataclass
class Branch:
    parent_tape: Tape
    divergence_step: int
    delta_tape: Tape
    mutation_desc: str = ""
    prefix_replayed: int = 0  # parent exchanges replayed for $0 (the savings)
    tail_recorded: int = 0  # counterfactual exchanges recorded fresh
    # rebase()-only: tail exchanges reused verbatim from the OLD branch's own
    # delta_tape by fingerprint match ($0) instead of live re-recording.
    # Always 0 for fork()/fork_coalition() (they have no "old" delta_tape to
    # reuse from).
    tail_reused: int = 0
    # Steps that were force-set to a specified response (coalition forks set
    # more than one; a classic single-step `fork()` sets exactly `(divergence_step,)`).
    intervened_steps: tuple[int, ...] = field(default_factory=tuple)
    # Content-addressed identity of this fork (Branch/store-level metadata
    # only — `Tape.digest()` itself is completely untouched by this field).
    # See `compute_branch_digest()`.
    branch_digest: str = ""
    # The parent tape's own `digest()` at fork time — the citable fork-point
    # `store.py`'s `load_branch` re-verifies on every read. Branch/store-level
    # metadata only, same as `branch_digest`.
    parent_tape_digest: str = ""
    # sha256 of the exact (request, response) bytes pair at the fork's first
    # divergence point. See `compute_divergence_exchange_digest()`.
    divergence_exchange_digest: str = ""
    # How confined the re-executed agent was during this fork's tail-record
    # phase (an axis orthogonal to `Tape.boundary`'s tiers — see
    # `constants.py`). Branch/store-level metadata only, same discipline as
    # `branch_digest`: never fed into `Tape.digest()`. See
    # `compute_confinement_tier()`.
    confinement_tier: str = CONFINEMENT_TIER_NONE


def compute_divergence_exchange_digest(request_bytes: bytes, response_bytes: bytes) -> str:
    """sha256 of the exact ``(request, response)`` byte pair at a fork's first
    divergence point — the exchange that actually diverged from the parent
    (the same request the parent recorded, paired with the response that was
    forced/mutated instead of the parent's own). Purely additive metadata,
    same as `compute_branch_digest`: `Tape.digest()` never reads it.
    """
    return sha256_hex(request_bytes + response_bytes)


def compute_branch_digest(
    parent_tape: Tape, delta_tape: Tape, intervened_steps: tuple[int, ...]
) -> str:
    """A content-addressed fingerprint of one fork: ``sha256(parent_tape.digest()
    + delta_tape.digest() + repr(intervened_steps))``.

    Git/IPLD Merkle-DAG identity — folding a node's children's hashes into its
    own hash gives identity==integrity==addressability in one field, so two
    forks with byte-identical (parent, delta content, intervened steps) are
    the SAME branch_digest (fork-of-fork and inverse-citation queries become
    plain reachability walks), while any difference in any of the three
    inputs — including which response was mutated — changes it. Purely
    additive: `Tape.digest()` never reads this value or vice versa.
    """
    payload = (parent_tape.digest() + delta_tape.digest() + repr(intervened_steps)).encode()
    return sha256_hex(payload)


def compute_confinement_tier(boundary_guard: bool, confinement: ConfinementSpec | None) -> str:
    """The confinement tier for a fork whose re-executed agent ran under the
    given `boundary_guard`/`confinement` kwargs (see `fork()`/
    `fork_coalition()`/`rebase()`).

    `confinement` (a declared writable-roots/allowed-hosts allowlist) is the
    strongest tier this bead recognizes and wins regardless of
    `boundary_guard`'s value, since passing it FORCES the guard active
    (see `fork()`'s docstring); a bare `boundary_guard=True` with no
    `confinement` is the middle tier; neither is the unconfined default.
    Purely additive metadata, same discipline as `compute_branch_digest`:
    never fed into `Tape.digest()`.
    """
    if confinement is not None:
        return CONFINEMENT_TIER_DECLARED
    if boundary_guard:
        return CONFINEMENT_TIER_GUARDED
    return CONFINEMENT_TIER_NONE


class ForkTransport(httpx.BaseTransport):
    """Three-phase transport: prefix-replay → mutation-inject → tail-record.

    `inner` is only consulted for the tail (requests after the divergence
    step); the prefix and the mutation are served from in-memory bytes, so a
    fork costs nothing up to and including the divergence point.
    """

    def __init__(
        self,
        parent_tape: Tape,
        divergence_step: int,
        mutated_response: bytes,
        delta_tape: Tape,
        inner: httpx.BaseTransport,
    ) -> None:
        self.parent = parent_tape
        self.k = divergence_step
        self.mutated = mutated_response
        self.delta = delta_tape
        self.inner = inner
        self._i = 0
        self.prefix_replayed = 0
        self.tail_recorded = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        body = request.content
        i = self._i
        self._i += 1

        if i < self.k:
            # prefix — replay from parent, assert the agent rebuilt it exactly
            rec_req, rec_resp = self.parent.exchange(i)
            if sha256_hex(rec_req) != sha256_hex(body):
                raise DivergenceError(
                    f"fork prefix request #{i} diverged from parent tape "
                    f"(recorded {sha256_hex(rec_req)[:12]}, replay {sha256_hex(body)[:12]}); "
                    f"the agent is not deterministic up to divergence_step {self.k}"
                )
            self.prefix_replayed += 1
            return _json_response(rec_resp, request)

        if i == self.k:
            # divergence point — same request, mutated response
            rec_req, _ = self.parent.exchange(i)
            if sha256_hex(rec_req) != sha256_hex(body):
                raise DivergenceError(
                    f"fork request at divergence_step {i} diverged from parent tape "
                    f"(recorded {sha256_hex(rec_req)[:12]}, replay {sha256_hex(body)[:12]})"
                )
            self.delta.append_exchange(body, self.mutated)
            return _json_response(self.mutated, request)

        # tail — counterfactual territory, record fresh
        inner_resp = self.inner.handle_request(request)
        resp_body = inner_resp.read()
        self.delta.append_exchange(body, resp_body)
        self.tail_recorded += 1
        return httpx.Response(
            inner_resp.status_code,
            headers={"content-type": inner_resp.headers.get("content-type", "application/json")},
            content=resp_body,
            request=request,
        )


def _json_response(body: bytes, request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        headers={"content-type": "application/json"},
        content=body,
        request=request,
    )


# ── coalition forks (joint, multi-step interventions) ───────────────────────


@dataclass(frozen=True)
class StepIntervention:
    """One (step, forced response) pair inside a `CoalitionSpec`."""

    step: int
    mutated_response: bytes


@dataclass
class CoalitionSpec:
    """A SET of `StepIntervention`s applied jointly — the "do(S)" primitive
    coalition/Shapley blame forks to measure `v(S)`, the coalition's flip-rate.

    Generalizes `BranchSpec` (a single divergence step) to an arbitrary,
    non-empty set of distinct step indices, each force-set to its own response.
    """

    interventions: tuple[StepIntervention, ...]
    mutation_desc: str = ""

    def __post_init__(self) -> None:
        if not self.interventions:
            raise ValueError("CoalitionSpec requires at least one intervention")
        steps = [iv.step for iv in self.interventions]
        if len(steps) != len(set(steps)):
            raise ValueError(f"duplicate step indices in CoalitionSpec: {steps}")
        self.interventions = tuple(sorted(self.interventions, key=lambda iv: iv.step))

    @property
    def steps(self) -> tuple[int, ...]:
        return tuple(iv.step for iv in self.interventions)

    @property
    def first_step(self) -> int:
        return self.interventions[0].step

    @classmethod
    def single(cls, step: int, mutated_response: bytes, mutation_desc: str = "") -> CoalitionSpec:
        """A one-element coalition — equivalent to a classic `BranchSpec` fork."""
        return cls(
            interventions=(StepIntervention(step, mutated_response),),
            mutation_desc=mutation_desc,
        )


class CoalitionForkTransport(httpx.BaseTransport):
    """N-phase transport generalizing `ForkTransport` to a coalition of steps.

    Requests before the coalition's first (lowest-index) intervention are
    prefix-replayed from the parent tape exactly like `ForkTransport`, with the
    same sha256 divergence assertion. The first intervention point is *also*
    request-matched against the parent (it is the true first point of
    divergence). Every later request is either forced to that coalition
    member's fixed response (if its index is in the coalition — no assertion,
    since the agent's request there necessarily no longer matches the parent
    once an earlier step has been perturbed) or recorded fresh from `inner`
    (the counterfactual tail beyond the coalition).
    """

    def __init__(
        self,
        parent_tape: Tape,
        spec: CoalitionSpec,
        delta_tape: Tape,
        inner: httpx.BaseTransport,
    ) -> None:
        self.parent = parent_tape
        self.spec = spec
        self._interventions = {iv.step: iv.mutated_response for iv in spec.interventions}
        self.first_step = spec.first_step
        self.delta = delta_tape
        self.inner = inner
        self._i = 0
        self.prefix_replayed = 0
        self.tail_recorded = 0
        self.interventions_applied = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        body = request.content
        i = self._i
        self._i += 1

        if i < self.first_step:
            # prefix — replay from parent, assert the agent rebuilt it exactly
            rec_req, rec_resp = self.parent.exchange(i)
            if sha256_hex(rec_req) != sha256_hex(body):
                raise DivergenceError(
                    f"coalition fork prefix request #{i} diverged from parent tape "
                    f"(recorded {sha256_hex(rec_req)[:12]}, replay {sha256_hex(body)[:12]}); "
                    f"the agent is not deterministic up to the coalition's first "
                    f"intervention {self.first_step}"
                )
            self.prefix_replayed += 1
            return _json_response(rec_resp, request)

        if i == self.first_step:
            # first intervention — same request as the parent, forced response
            rec_req, _ = self.parent.exchange(i)
            if sha256_hex(rec_req) != sha256_hex(body):
                raise DivergenceError(
                    f"coalition fork request at first intervention {i} diverged from "
                    f"parent tape (recorded {sha256_hex(rec_req)[:12]}, replay "
                    f"{sha256_hex(body)[:12]})"
                )
            mutated = self._interventions[i]
            self.delta.append_exchange(body, mutated)
            self.interventions_applied += 1
            return _json_response(mutated, request)

        if i in self._interventions:
            # a later coalition member — force it, no assertion (already diverged)
            mutated = self._interventions[i]
            self.delta.append_exchange(body, mutated)
            self.interventions_applied += 1
            return _json_response(mutated, request)

        # tail — counterfactual territory beyond the coalition, record fresh
        inner_resp = self.inner.handle_request(request)
        resp_body = inner_resp.read()
        self.delta.append_exchange(body, resp_body)
        self.tail_recorded += 1
        return httpx.Response(
            inner_resp.status_code,
            headers={"content-type": inner_resp.headers.get("content-type", "application/json")},
            content=resp_body,
            request=request,
        )


class RebaseTransport(httpx.BaseTransport):
    """Rebases `old_branch` onto `new_parent_tape`: replay-the-unchanged-
    prefix-from-the-new-parent, re-force-the-old-interventions,
    reuse-or-re-record-the-tail.

    Requests before `old_branch.divergence_step` are prefix-replayed from
    `new_parent_tape` (not `old_branch.parent_tape`), with the same sha256
    divergence assertion `ForkTransport`/`CoalitionForkTransport` already use
    — a genuinely moved prefix is a hard `DivergenceError`, never silently
    tolerated. Every step in `old_branch.intervened_steps` is force-set to
    ITS ORIGINAL stored response (no assertion — by definition the base may
    have moved there, that's the entire point of rebasing) via `_old_by_step`,
    built once from `old_branch.delta_tape` (whose exchange at offset `step -
    old_branch.divergence_step` is exactly the exchange recorded for absolute
    step `step`, since every absolute step from `divergence_step` onward
    triggers exactly one `delta_tape.append_exchange` call in both `fork()`
    and `fork_coalition()`). Everything else beyond the fork point is tail
    territory: first try `_tail_fp_queue` (a fingerprint→response FIFO queue
    built from the OLD branch's own non-intervened tail exchanges) for a
    byte-identical request reuse ($0); on a miss, fall through to `inner` for
    a live re-record, exactly like `CoalitionForkTransport`'s own tail.
    """

    def __init__(
        self,
        old_branch: Branch,
        new_parent_tape: Tape,
        delta_tape: Tape,
        inner: httpx.BaseTransport,
    ) -> None:
        self.old_branch = old_branch
        self.new_parent = new_parent_tape
        self.delta = delta_tape
        self.inner = inner
        self.first_step = old_branch.divergence_step
        self._intervened = set(old_branch.intervened_steps)

        # Absolute step -> (request, response) the OLD branch recorded there.
        # See class docstring for why this offset arithmetic is exact.
        self._old_by_step: dict[int, tuple[bytes, bytes]] = {
            old_branch.divergence_step + j: exch
            for j, exch in enumerate(old_branch.delta_tape.exchanges)
        }
        # Fingerprint (raw sha256 of the request bytes) -> FIFO queue of
        # responses, built ONLY from the tail portion (steps that were NOT
        # force-intervened) of the old branch's delta_tape -- the reuse
        # candidates for a genuinely-new request during rebase.
        self._tail_fp_queue: dict[str, deque[bytes]] = {}
        for step, (req, resp) in self._old_by_step.items():
            if step not in self._intervened:
                self._tail_fp_queue.setdefault(sha256_hex(req), deque()).append(resp)

        self._i = 0
        self.prefix_replayed = 0
        self.interventions_applied = 0
        self.tail_reused = 0
        self.tail_recorded = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        body = request.content
        i = self._i
        self._i += 1

        if i < self.first_step:
            # unchanged prefix — replay from the NEW parent, assert it still matches
            rec_req, rec_resp = self.new_parent.exchange(i)
            if sha256_hex(rec_req) != sha256_hex(body):
                raise DivergenceError(
                    f"rebase prefix request #{i} diverged from the new parent tape "
                    f"(recorded {sha256_hex(rec_req)[:12]}, replay {sha256_hex(body)[:12]}); "
                    f"the new parent's prefix moved before the fork point {self.first_step}"
                )
            self.prefix_replayed += 1
            return _json_response(rec_resp, request)

        if i in self._intervened:
            # re-force this step to its ORIGINAL response — no assertion, the
            # base may have legitimately moved here
            _old_req, old_resp = self._old_by_step[i]
            self.delta.append_exchange(body, old_resp)
            self.interventions_applied += 1
            return _json_response(old_resp, request)

        # tail — try reusing the old branch's own tail exchange by fingerprint
        # match before paying for a live re-record
        queue = self._tail_fp_queue.get(sha256_hex(body))
        if queue:
            reused_resp = queue.popleft()
            self.delta.append_exchange(body, reused_resp)
            self.tail_reused += 1
            return _json_response(reused_resp, request)

        inner_resp = self.inner.handle_request(request)
        resp_body = inner_resp.read()
        self.delta.append_exchange(body, resp_body)
        self.tail_recorded += 1
        return httpx.Response(
            inner_resp.status_code,
            headers={"content-type": inner_resp.headers.get("content-type", "application/json")},
            content=resp_body,
            request=request,
        )


class ForkEngine:
    """Creates counterfactual branches from a recorded tape."""

    @staticmethod
    @instrument("tracefork.fork")
    def fork(
        parent_tape: Tape,
        spec: BranchSpec,
        agent_fn,  # Callable[[anthropic.Anthropic], Any] — the SAME agent
        *,
        post_fork_transport: httpx.BaseTransport | None = None,
        api_key: str = "sk-ant-fork",
        boundary_guard: bool = False,
        confinement: ConfinementSpec | None = None,
    ) -> Branch:
        """Fork `parent_tape` at `spec.divergence_step`.

        `agent_fn` must be the same agent that produced the parent tape: it is
        re-run from the start, its prefix served from the tape for free, the
        response at the divergence step swapped for `spec.mutated_response`,
        and the counterfactual tail recorded via `post_fork_transport` (or the
        real Anthropic API if None).

        `boundary_guard` (default `False`, byte-identical to before when left
        off) wraps *only* the `agent_fn(client)` call in a fresh `BoundaryGuard`
        (see `boundary_guard.py`) — confining the re-executed agent's own
        tool-call/thread/random/subprocess surface for this fork, without
        touching the prefix-replay/mutation-injection transport logic above.

        `confinement` (default `None`, byte-identical to before when left off)
        is a `ConfinementSpec` (see `boundary_guard.py`) that FORCES the guard
        active for the `agent_fn(client)` call even when `boundary_guard` is
        left `False` — declaring the writable-roots/allowed-hosts surface the
        re-executed agent may touch during this fork's tail-record phase.

        Returns a `Branch` whose `delta_tape` holds only the exchanges from the
        divergence step onward.
        """
        step = spec.divergence_step
        n = len(parent_tape.exchanges)
        if step < 0 or step >= n:
            raise ValueError(f"divergence_step {step} out of range [0, {n})")

        delta_tape = Tape(
            boundary=parent_tape.boundary,
            agent_name=parent_tape.agent_name,
        )
        inner = post_fork_transport if post_fork_transport is not None else httpx.HTTPTransport()
        fork_transport = ForkTransport(parent_tape, step, spec.mutated_response, delta_tape, inner)

        client = anthropic.Anthropic(
            api_key=api_key,
            http_client=httpx.Client(transport=fork_transport),
            max_retries=0,
        )
        if confinement is not None:
            with BoundaryGuard(confinement=confinement):
                agent_fn(client)
        elif boundary_guard:
            with BoundaryGuard():
                agent_fn(client)
        else:
            agent_fn(client)

        intervened_steps = (step,)
        divergence_request, _ = parent_tape.exchange(step)
        return Branch(
            parent_tape=parent_tape,
            divergence_step=step,
            delta_tape=delta_tape,
            mutation_desc=spec.mutation_desc,
            prefix_replayed=fork_transport.prefix_replayed,
            tail_recorded=fork_transport.tail_recorded,
            intervened_steps=intervened_steps,
            branch_digest=compute_branch_digest(parent_tape, delta_tape, intervened_steps),
            parent_tape_digest=parent_tape.digest(),
            divergence_exchange_digest=compute_divergence_exchange_digest(
                divergence_request, spec.mutated_response
            ),
            confinement_tier=compute_confinement_tier(boundary_guard, confinement),
        )

    @staticmethod
    @instrument("tracefork.fork_coalition")
    def fork_coalition(
        parent_tape: Tape,
        spec: CoalitionSpec,
        agent_fn,  # Callable[[anthropic.Anthropic], Any] — the SAME agent
        *,
        post_fork_transport: httpx.BaseTransport | None = None,
        api_key: str = "sk-ant-fork",
        boundary_guard: bool = False,
        confinement: ConfinementSpec | None = None,
    ) -> Branch:
        """Fork `parent_tape` at a coalition of steps, forcing each to its own response.

        Same contract as `fork()`, generalized to `spec.interventions`: the
        prefix below the coalition's first step is replayed for $0, that first
        step (and every later coalition member) is forced to its specified
        response, and everything else is recorded fresh via
        `post_fork_transport` (or the real Anthropic API if `None`).
        `Branch.divergence_step` is the coalition's first step;
        `Branch.intervened_steps` holds the full coalition.

        `boundary_guard` (default `False`, byte-identical to before when left
        off) wraps *only* the `agent_fn(client)` call in a fresh `BoundaryGuard`,
        same as `fork()`.

        `confinement` (default `None`, byte-identical to before when left off)
        is a `ConfinementSpec` that FORCES the guard active for the
        `agent_fn(client)` call even when `boundary_guard` is left `False`,
        same as `fork()`.
        """
        n = len(parent_tape.exchanges)
        for step in spec.steps:
            if step < 0 or step >= n:
                raise ValueError(f"coalition step {step} out of range [0, {n})")

        delta_tape = Tape(
            boundary=parent_tape.boundary,
            agent_name=parent_tape.agent_name,
        )
        inner = post_fork_transport if post_fork_transport is not None else httpx.HTTPTransport()
        fork_transport = CoalitionForkTransport(parent_tape, spec, delta_tape, inner)

        client = anthropic.Anthropic(
            api_key=api_key,
            http_client=httpx.Client(transport=fork_transport),
            max_retries=0,
        )
        if confinement is not None:
            with BoundaryGuard(confinement=confinement):
                agent_fn(client)
        elif boundary_guard:
            with BoundaryGuard():
                agent_fn(client)
        else:
            agent_fn(client)

        first_request, _ = parent_tape.exchange(spec.first_step)
        first_mutated_response = spec.interventions[0].mutated_response
        return Branch(
            parent_tape=parent_tape,
            divergence_step=spec.first_step,
            delta_tape=delta_tape,
            mutation_desc=spec.mutation_desc,
            prefix_replayed=fork_transport.prefix_replayed,
            tail_recorded=fork_transport.tail_recorded,
            intervened_steps=spec.steps,
            branch_digest=compute_branch_digest(parent_tape, delta_tape, spec.steps),
            parent_tape_digest=parent_tape.digest(),
            divergence_exchange_digest=compute_divergence_exchange_digest(
                first_request, first_mutated_response
            ),
            confinement_tier=compute_confinement_tier(boundary_guard, confinement),
        )

    @staticmethod
    @instrument("tracefork.rebase")
    def rebase(
        old_branch: Branch,
        new_parent_tape: Tape,
        agent_fn,  # Callable[[anthropic.Anthropic], Any] — the SAME agent
        *,
        post_fork_transport: httpx.BaseTransport | None = None,
        api_key: str = "sk-ant-fork",
        boundary_guard: bool = False,
    ) -> Branch:
        """Rebase `old_branch` onto `new_parent_tape` — the version-control-
        rebase analogue of `fork()`/`fork_coalition()`.

        Re-runs `agent_fn` from the start, exactly like `fork()`, but instead
        of replaying the unchanged prefix from `old_branch.parent_tape`,
        sources it from `new_parent_tape` (raising `DivergenceError` if that
        prefix itself moved before `old_branch.divergence_step` — the same
        contract as `fork()`'s own prefix assert). Every step in
        `old_branch.intervened_steps` is re-forced to its ORIGINAL stored
        response (no assertion there — the base may have legitimately moved
        exactly at that point, which is the entire reason to rebase).
        Everything else beyond the fork point first tries reusing
        `old_branch.delta_tape`'s own tail exchange by fingerprint match
        (`Branch.tail_reused`, $0) before falling through to live
        re-recording via `post_fork_transport` (or the real Anthropic API if
        `None`) into `Branch.tail_recorded` — see `RebaseTransport`.

        Returns a fresh `Branch` whose `parent_tape` is `new_parent_tape` and
        whose `delta_tape` holds the exchanges from the fork point onward,
        same shape as `fork()`/`fork_coalition()`.

        `boundary_guard` (default `False`, byte-identical to before when left
        off) wraps *only* the `agent_fn(client)` call in a fresh
        `BoundaryGuard`, same as `fork()`/`fork_coalition()`.
        """
        step = old_branch.divergence_step
        if step < 0 or step > len(new_parent_tape.exchanges):
            raise ValueError(
                f"rebase divergence_step {step} out of range "
                f"[0, {len(new_parent_tape.exchanges)}] for the new parent tape"
            )

        delta_tape = Tape(
            boundary=new_parent_tape.boundary,
            agent_name=new_parent_tape.agent_name,
        )
        inner = post_fork_transport if post_fork_transport is not None else httpx.HTTPTransport()
        rebase_transport = RebaseTransport(old_branch, new_parent_tape, delta_tape, inner)

        client = anthropic.Anthropic(
            api_key=api_key,
            http_client=httpx.Client(transport=rebase_transport),
            max_retries=0,
        )
        if boundary_guard:
            with BoundaryGuard():
                agent_fn(client)
        else:
            agent_fn(client)

        # The fork point's own (request, forced-response) pair — same shape
        # as fork()/fork_coalition()'s divergence_exchange_digest input — is
        # the delta_tape's own first entry (see RebaseTransport's class
        # docstring for why this is exact).
        first_request, first_response = delta_tape.exchange(0)
        return Branch(
            parent_tape=new_parent_tape,
            divergence_step=step,
            delta_tape=delta_tape,
            mutation_desc=old_branch.mutation_desc,
            prefix_replayed=rebase_transport.prefix_replayed,
            tail_recorded=rebase_transport.tail_recorded,
            tail_reused=rebase_transport.tail_reused,
            intervened_steps=old_branch.intervened_steps,
            branch_digest=compute_branch_digest(
                new_parent_tape, delta_tape, old_branch.intervened_steps
            ),
            parent_tape_digest=new_parent_tape.digest(),
            divergence_exchange_digest=compute_divergence_exchange_digest(
                first_request, first_response
            ),
            confinement_tier=compute_confinement_tier(boundary_guard, None),
        )
