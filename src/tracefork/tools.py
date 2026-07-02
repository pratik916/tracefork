"""JSON-RPC tool-call record/replay seam — the tool-I/O analog of transport.py.

An agent's divergence often comes from a TOOL result, not the model, yet non-LLM
tool I/O (MCP calls, retrieval) never crosses the httpx seam `transport.py`
records: MCP is JSON-RPC 2.0 over stdio or Streamable-HTTP, not httpx. This
module records that traffic as JSON-RPC *frames* into the same content-addressed
tape (`Tape.tool_exchanges`), mirroring the LLM seam exactly:

  * `ToolTransport` — record mode tees each request+response frame into the tape;
    replay mode fingerprint-asserts the incoming request frame against the tape
    (the divergence detector) and serves the recorded response back. This is the
    validator half of the split (replay = hash-assert the request).
  * `ToolForkTransport` — the mutator half: a three-phase fork (prefix-replay →
    mutation-inject → tail-record) over tool frames, mirroring `fork.ForkTransport`
    so blame can perturb a specific tool output and record the counterfactual tail.
  * `NativeToolSeam` — a minimal `@seam.tool(...)`-style wrapper that routes a
    plain (non-MCP) Python callable's calls through the same frame seam.

This module is provider-independent and imports nothing from the optional `mcp`
package — the `mcp` adapter lives in `mcp_client.py`. Tests drive it with
synthetic frames, exactly like the httpx fakes in `synthetic.py`.

**JSON-RPC id is volatile.** A frame's top-level `id` only correlates a request
with its response; the client session assigns it (often a per-session counter),
so it can differ between record and replay. It is therefore dropped from the
divergence fingerprint (like `CanonicalizingMatcher`'s volatile body fields), and
a served response's `id` is retargeted to the live request's `id` so JSON-RPC
correlation still holds on replay.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Callable
from typing import Any

from .nondet import DivergenceError
from .redact import Redactor
from .tape import Tape, sha256_hex

#: A tool call: takes a request frame (bytes), returns a response frame (bytes).
ToolCall = Callable[[bytes], bytes]

#: Top-level JSON-RPC frame fields dropped from the divergence fingerprint. The
#: correlation `id` is client-assigned and non-semantic, so it must not force a
#: spurious divergence when it rotates between record and replay.
DEFAULT_VOLATILE_FRAME_FIELDS: frozenset[str] = frozenset({"id"})


# ── frame utilities ──────────────────────────────────────────────────────────


def canonical_frame(
    frame: bytes, volatile: frozenset[str] = DEFAULT_VOLATILE_FRAME_FIELDS
) -> bytes:
    """Deterministic identity of a JSON-RPC frame: drop `volatile` top-level
    fields (the correlation `id`) and re-emit with sorted keys. A frame that is
    not a JSON object is returned verbatim (lossless), so equal bytes always
    canonicalize equal and unequal bytes never collide."""
    try:
        obj = json.loads(frame)
    except (ValueError, UnicodeDecodeError):
        return frame
    if isinstance(obj, dict):
        obj = {k: v for k, v in obj.items() if k not in volatile}
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def frame_fingerprint(
    frame: bytes, volatile: frozenset[str] = DEFAULT_VOLATILE_FRAME_FIELDS
) -> str:
    """`sha256` of the canonical frame — the tool-frame divergence fingerprint."""
    return sha256_hex(canonical_frame(frame, volatile))


def frame_id(frame: bytes) -> Any:
    """The JSON-RPC `id` of a frame, or ``None`` if absent / not a JSON object."""
    try:
        obj = json.loads(frame)
    except (ValueError, UnicodeDecodeError):
        return None
    return obj.get("id") if isinstance(obj, dict) else None


def retarget_frame_id(frame: bytes, new_id: Any) -> bytes:
    """Return `frame` with its JSON-RPC `id` set to `new_id`, so a recorded
    response correlates with the live request that asked for it on replay.
    Unchanged when `new_id` is ``None`` or `frame` carries no `id`."""
    if new_id is None:
        return frame
    try:
        obj = json.loads(frame)
    except (ValueError, UnicodeDecodeError):
        return frame
    if not isinstance(obj, dict) or "id" not in obj:
        return frame
    obj["id"] = new_id
    return json.dumps(obj, separators=(",", ":")).encode()


# ── frame builders (synthetic-frame constructors, cf. wire.py) ───────────────


def make_request_frame(frame_id: Any, method: str, params: Any = None) -> bytes:
    """Build a JSON-RPC 2.0 request frame."""
    frame: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        frame["params"] = params
    if frame_id is not None:
        frame["id"] = frame_id
    return json.dumps(frame, separators=(",", ":")).encode()


def make_result_frame(frame_id: Any, result: Any) -> bytes:
    """Build a JSON-RPC 2.0 success (result) frame."""
    frame: dict[str, Any] = {"jsonrpc": "2.0", "result": result}
    if frame_id is not None:
        frame["id"] = frame_id
    return json.dumps(frame, separators=(",", ":")).encode()


def make_tool_call_frame(
    frame_id: Any, name: str, arguments: dict[str, Any] | None = None
) -> bytes:
    """Build an MCP ``tools/call`` request frame for tool `name`."""
    return make_request_frame(frame_id, "tools/call", {"name": name, "arguments": arguments or {}})


def decode_result(frame: bytes) -> Any:
    """The ``result`` payload of a JSON-RPC success frame (``None`` if absent)."""
    obj = json.loads(frame)
    return obj.get("result") if isinstance(obj, dict) else None


# ── record / replay transport ────────────────────────────────────────────────


class ToolTransport:
    """Record/replay seam for JSON-RPC tool frames — mirrors `TraceforkTransport`.

    Record mode tees each request+response frame into `tape.tool_exchanges`
    (request-side redaction via `Redactor.apply_request`, response-side via
    `Redactor.apply_response`, exactly like the LLM seam). Replay mode has no
    live tool call: it fingerprint-asserts the incoming request frame against the
    tape and serves the recorded response (with its `id` retargeted for
    correlation); any mismatch or unrecorded call is a `DivergenceError`.
    """

    def __init__(
        self,
        mode: str,
        tape: Tape,
        inner: ToolCall | None = None,
        *,
        redactor: Redactor | None = None,
        volatile_fields: frozenset[str] = DEFAULT_VOLATILE_FRAME_FIELDS,
    ) -> None:
        assert mode in ("record", "replay")
        self.mode = mode
        self.tape = tape
        self.inner = inner
        self.redactor = redactor
        self.volatile = frozenset(volatile_fields)
        self._i = 0
        self.matched = 0

    def _prepare_request(self, frame: bytes) -> bytes:
        return self.redactor.apply_request(frame) if self.redactor else frame

    def handle_frame(self, request_frame: bytes, inner: ToolCall | None = None) -> bytes:
        """Record: call `inner` (or the construction-time inner), tee, and return
        the live response frame. Replay: serve the recorded response frame."""
        if self.mode == "record":
            call = inner if inner is not None else self.inner
            if call is None:
                raise ValueError("record mode requires an inner tool call")
            response_frame = call(request_frame)
            self.record_exchange(request_frame, response_frame)
            return response_frame
        return self.replay_exchange(request_frame)

    def record_exchange(self, request_frame: bytes, response_frame: bytes) -> None:
        """Tee one request/response frame pair into the tape (redacted if a
        `Redactor` is set). The unredacted `response_frame` is what the caller
        keeps — only the stored copy is scrubbed, matching the LLM seam."""
        stored_req = self._prepare_request(request_frame)
        stored_resp = (
            self.redactor.apply_response(response_frame) if self.redactor else response_frame
        )
        self.tape.append_tool_exchange(stored_req, stored_resp)

    def replay_exchange(self, request_frame: bytes) -> bytes:
        """Assert the request frame matches the next recorded tool exchange and
        return its recorded response (id retargeted to the live request)."""
        if self._i >= len(self.tape.tool_exchanges):
            raise DivergenceError(
                f"replay made unrecorded tool call #{self._i} "
                f"(tape has {len(self.tape.tool_exchanges)} tool exchanges)"
            )
        rec_req, rec_resp = self.tape.tool_exchange(self._i)
        rec_fp = frame_fingerprint(rec_req, self.volatile)
        live_fp = frame_fingerprint(self._prepare_request(request_frame), self.volatile)
        if rec_fp != live_fp:
            raise DivergenceError(
                f"tool call #{self._i} diverged from tape "
                f"(recorded {rec_fp[:12]}, replay {live_fp[:12]})"
            )
        self._i += 1
        self.matched += 1
        return retarget_frame_id(rec_resp, frame_id(request_frame))

    def fully_consumed(self) -> bool:
        return self.mode == "replay" and self._i == len(self.tape.tool_exchanges)


# ── fork transport (three-phase) ─────────────────────────────────────────────


class ToolForkTransport:
    """Three-phase tool-frame fork — the tool analog of `fork.ForkTransport`.

    prefix (i < k): replay the recorded tool response, asserting the request
    matches; mutation (i == k): assert the request matches, then serve
    `mutated_response` and record it into `delta`; tail (i > k): call `inner`
    (fresh, counterfactual) and record it. `inner` is only consulted for the
    tail, so a tool fork costs nothing up to and including the divergence point.
    """

    def __init__(
        self,
        parent_tape: Tape,
        divergence_step: int,
        mutated_response: bytes,
        delta_tape: Tape,
        inner: ToolCall | None = None,
        *,
        volatile_fields: frozenset[str] = DEFAULT_VOLATILE_FRAME_FIELDS,
    ) -> None:
        self.parent = parent_tape
        self.k = divergence_step
        self.mutated = mutated_response
        self.delta = delta_tape
        self.inner = inner
        self.volatile = frozenset(volatile_fields)
        self._i = 0
        self.prefix_replayed = 0
        self.tail_recorded = 0

    def _assert_match(self, i: int, rec_req: bytes, live_frame: bytes, label: str) -> None:
        rec_fp = frame_fingerprint(rec_req, self.volatile)
        live_fp = frame_fingerprint(live_frame, self.volatile)
        if rec_fp != live_fp:
            raise DivergenceError(
                f"{label} #{i} diverged from parent tape "
                f"(recorded {rec_fp[:12]}, replay {live_fp[:12]})"
            )

    def handle_frame(self, request_frame: bytes, inner: ToolCall | None = None) -> bytes:
        i = self._i
        self._i += 1

        if i < self.k:
            rec_req, rec_resp = self.parent.tool_exchange(i)
            self._assert_match(i, rec_req, request_frame, "fork prefix tool call")
            self.prefix_replayed += 1
            return retarget_frame_id(rec_resp, frame_id(request_frame))

        if i == self.k:
            rec_req, _ = self.parent.tool_exchange(i)
            self._assert_match(i, rec_req, request_frame, "fork tool call at divergence_step")
            self.delta.append_tool_exchange(request_frame, self.mutated)
            return retarget_frame_id(self.mutated, frame_id(request_frame))

        call = inner if inner is not None else self.inner
        if call is None:
            raise ValueError("fork tail requires an inner tool call")
        response_frame = call(request_frame)
        self.delta.append_tool_exchange(request_frame, response_frame)
        self.tail_recorded += 1
        return response_frame


# ── native (non-MCP) tool seam ───────────────────────────────────────────────


class NativeToolSeam:
    """Minimal native tool record/replay: route plain Python callables through
    the same JSON-RPC frame seam as MCP. A wrapped tool's positional args,
    keyword args and return value must be JSON-serializable.

    Record mode calls the real function and tees the call; replay mode serves the
    recorded return value and asserts the call (tool name + args) matches the
    tape — so a non-deterministic tool loop is caught, not silently accepted.

        seam = NativeToolSeam(tape, "record")

        @seam.tool("add")
        def add(a, b):
            return a + b
    """

    def __init__(self, tape: Tape, mode: str, *, redactor: Redactor | None = None) -> None:
        self._transport = ToolTransport(mode, tape, redactor=redactor)

    @property
    def mode(self) -> str:
        return self._transport.mode

    def tool(self, name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator: wrap `fn` so its calls are recorded/replayed under `name`."""

        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                request_frame = make_tool_call_frame(
                    None, name, {"args": list(args), "kwargs": kwargs}
                )

                def _inner(_frame: bytes) -> bytes:
                    return make_result_frame(None, fn(*args, **kwargs))

                response_frame = self._transport.handle_frame(request_frame, inner=_inner)
                return decode_result(response_frame)

            return wrapper

        return decorator
