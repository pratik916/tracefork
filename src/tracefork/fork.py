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
"""

from __future__ import annotations

from dataclasses import dataclass

import anthropic
import httpx

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
    ) -> Branch:
        """Fork `parent_tape` at `spec.divergence_step`.

        `agent_fn` must be the same agent that produced the parent tape: it is
        re-run from the start, its prefix served from the tape for free, the
        response at the divergence step swapped for `spec.mutated_response`,
        and the counterfactual tail recorded via `post_fork_transport` (or the
        real Anthropic API if None).

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
        agent_fn(client)

        return Branch(
            parent_tape=parent_tape,
            divergence_step=step,
            delta_tape=delta_tape,
            mutation_desc=spec.mutation_desc,
            prefix_replayed=fork_transport.prefix_replayed,
            tail_recorded=fork_transport.tail_recorded,
        )
