# Plugin / extension API

tracefork has five pluggable seams — provider adapters, blame oracles, tape
serializers, request matchers, and framework adapters — and every one of them
is backed by the same generic mechanism: `tracefork.plugins.Registry`, a
`dict[str, T]` plus an opt-in `importlib.metadata` entry-point loader. This
document is the public contract for that mechanism: what's stable, what a
third-party package needs to implement, and — most importantly — the security
model that keeps "installed" and "loaded" two separate, explicit steps.

This has worked and been tested since the seam first landed (see
`tests/test_plugins.py`); this document exists to make it a real, citable
public API rather than something discoverable only by reading `plugins.py`'s
module docstring.

## The five groups

| Group constant (`tracefork.plugins`) | Entry-point group string | Protocol | Registered in |
|---|---|---|---|
| `PROVIDER_GROUP` | `tracefork.providers` | `ProviderAdapter` (`tracefork.providers.base`) | `providers/base.py` |
| `ORACLE_GROUP` | `tracefork.oracles` | `Oracle` (`tracefork.blame`) | `blame.py` |
| `SERIALIZER_GROUP` | `tracefork.serializers` | `TapeSerializer` (`tracefork.tape`) | `tape.py` |
| `MATCHER_GROUP` | `tracefork.matchers` | `RequestMatcher` (`tracefork.matcher`) | `matcher.py` |
| `ADAPTER_GROUP` | `tracefork.adapters` | `FrameworkAdapter` (`tracefork.adapters.base`) | `adapters/base.py` |

Each group has the same shape at its call site:

* a module-level `Registry[...]` instance (e.g. `matcher.MATCHER_REGISTRY`);
* `register_<thing>(name, obj)` — register directly, no entry points involved;
* `get_<thing>(name)` — look up by name, raising a `KeyError` that lists what
  IS registered (`Registry.get_or_raise`);
* `registered_<things>()` — sorted list of every currently-registered name;
* `load_<thing>_entry_points(*, allow=None, allow_all=False)` — the opt-in
  entry-point loader, thinly wrapping `Registry.load_entry_points` (see
  **Security model** below).

Two groups store slightly different things: `ORACLE_REGISTRY` stores
*classes* (`Oracle` implementations generally need constructor arguments —
success/failure regexes — so there's no sensible zero-arg default instance),
every other registry stores ready *instances*. `Registry.load_entry_points`
handles both: `obj() if isinstance(obj, type) else obj`.

## The protocols

Implement the relevant `Protocol` (all are `@runtime_checkable`, but
tracefork never does an `isinstance` gate at registration time — a plugin
just needs to structurally satisfy the methods below) and expose a `name`
attribute matching your registered key.

```python
# tracefork.matcher.RequestMatcher — the seam the example package below targets
class RequestMatcher(Protocol):
    name: str
    def stored_request(self, request: httpx.Request) -> bytes: ...
    def live_fingerprint(self, request: httpx.Request) -> str: ...
    def stored_fingerprint(self, stored: bytes) -> str: ...
```

The one invariant every `RequestMatcher` MUST uphold, for every request `R`:

```
stored_fingerprint(stored_request(R)) == live_fingerprint(R)
```

i.e. the fingerprint the recorder persists for `R` equals the fingerprint
recomputed from the replayed request. Get this wrong and every replay looks
like a divergence. See `examples/plugin_example/` below for a worked
implementation, and `tracefork.matcher`'s module docstring for the full
rationale (Gemini `?key=`, Bedrock SigV4 signing headers, rotating auth,
per-call idempotency keys — the class of problem a custom matcher exists to
solve).

`ProviderAdapter`, `Oracle`, `TapeSerializer`, and `FrameworkAdapter` each have
their own protocol (see `providers/base.py`, `blame.py`, `tape.py`,
`adapters/base.py` respectively) with their own contract documented in that
module's docstrings — the registration/discovery mechanics below are
identical across all five.

## Security model

**Nothing loads automatically.** This is quoted near-verbatim from
`plugins.py`'s own module docstring because it is the load-bearing guarantee
of this entire API surface, and a doc that drifted from it would be worse
than no doc:

A package on `sys.path` that advertises a `tracefork.providers` (or
`.oracles` / `.serializers` / `.matchers` / `.adapters`) entry point does
**nothing** until a caller (or operator) explicitly allowlists it:

* in code, via `Registry.load_entry_points(allow={"name", ...})` or
  `allow_all=True` (equivalently, the group-specific
  `load_<thing>_entry_points(allow=..., allow_all=...)` wrappers);
* or for operators, via the `TRACEFORK_ALLOW_PLUGINS` environment variable (a
  comma-separated list of entry-point names, or `*` for "load everything").

Merely installing a dependency must never be enough, by itself, to inject
code into tracefork's record/replay/fork/blame path — that would let any
transitive dependency silently take over a security-relevant seam. Every
built-in implementation (the Anthropic provider adapter, `IdentityMatcher`,
`StringMatchOracle`, the binary tape codec, the LangChain/OpenAI-Agents/
CrewAI/AutoGen/ADK framework adapters) is registered directly by its owning
module at import time, never through this entry-point path, so default
behavior never depends on `importlib.metadata` at all.

**Do not write a plugin, README, or wrapper script that pre-populates
`TRACEFORK_ALLOW_PLUGINS=*` "for convenience" or otherwise suggests bypassing
this gate by default.** The allowlist is the security boundary; a plugin
package should document what name to add to it, not try to widen it for the
caller.

## Stability policy

Treat this the way a `pluggy`/`hookspec`-style plugin host treats its public
hook surface — the registry mechanism is a versioned contract, independent of
what's registered through it:

* **Public, SemVer-covered API:** `tracefork.plugins.Registry` (the class and
  all its public methods: `register`, `get_or_raise`, `names`,
  `load_entry_points`), the five group constants
  (`PROVIDER_GROUP`/`ORACLE_GROUP`/`SERIALIZER_GROUP`/`MATCHER_GROUP`/
  `ADAPTER_GROUP`), the `TRACEFORK_ALLOW_PLUGINS` environment variable name
  and its comma-separated/`*` syntax, and the five protocols
  (`ProviderAdapter`/`Oracle`/`TapeSerializer`/`RequestMatcher`/
  `FrameworkAdapter`). A breaking change to any of these is a major-version
  bump with a deprecation note in `CHANGELOG.md`, same as any other public
  API in this project.
* **Internal, not covered:** the concrete built-in registrations themselves
  (which providers/oracles/matchers/serializers/adapters ship, their exact
  names, and their implementation details) may change between minor versions
  as tracefork adds/refines built-ins — a plugin should only ever depend on
  the registry mechanism plus the protocol it implements, never on a
  built-in's internals.
* **Backward compatibility:** an entry point advertising a name that
  collides with a built-in overwrites it once loaded (`Registry.register`'s
  documented overwrite semantics) — this is intentional (it's how you'd
  override a built-in with your own implementation) but means a third-party
  package should pick a distinctive name unless override is exactly the
  intent.

## Worked example: `examples/plugin_example/`

`examples/plugin_example/` is a complete, standalone package (its own
`pyproject.toml`, its own `[project.entry-points."tracefork.matchers"]`
declaration) implementing `NonceStrippingMatcher`, a `RequestMatcher` that
drops one volatile per-call header (`x-request-nonce`) before hashing —
demonstrating the same "canonicalize before matching" pattern as the built-in
`CanonicalizingMatcher`, from outside the tracefork package. See that
directory's own `README.md` for how to install and load it for real; `tests/
test_plugin_example.py` in this repo exercises it directly (round-trip
invariant + `Registry.load_entry_points` with and without an allowlist)
without requiring an actual `pip install`.

## Quick reference

```python
from tracefork.matcher import load_matcher_entry_points, get_matcher

# Load ONLY the plugins you've explicitly named:
load_matcher_entry_points(allow={"example_nonce_stripping"})

# Or load everything advertised (only ever do this in a trusted, closed
# environment — see the security model above):
load_matcher_entry_points(allow_all=True)

# Or set once, in the environment, before your process starts:
#   TRACEFORK_ALLOW_PLUGINS=example_nonce_stripping
# and call load_matcher_entry_points() with no arguments — it reads the
# env var internally.

matcher = get_matcher("example_nonce_stripping")
```
