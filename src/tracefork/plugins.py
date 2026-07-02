"""Generic, security-gated entry-point plugin registry.

Every pluggable seam in tracefork (provider adapters in ``providers/base.py``,
blame oracles in ``blame.py``, request matchers in ``matcher.py``, tape
serializers in ``tape.py``) is fundamentally a name -> implementation lookup.
This module is the one mechanism behind all four: a dict-backed ``Registry``
plus an opt-in ``importlib.metadata`` entry-point loader, so a third-party
package can ship (say) a Bedrock provider adapter or a custom Oracle without
forking tracefork.

**Security — nothing loads automatically.** A package on ``sys.path`` that
advertises a ``tracefork.providers`` (or ``.oracles`` / ``.serializers`` /
``.matchers``) entry point does *nothing* until a caller (or operator)
explicitly allowlists it:

* in code, via ``Registry.load_entry_points(allow={"name", ...})`` or
  ``allow_all=True``;
* or for operators, via the ``TRACEFORK_ALLOW_PLUGINS`` environment variable
  (a comma-separated list of entry-point names, or ``*`` for "load
  everything").

Merely installing a dependency must never be enough, by itself, to inject
code into tracefork's record/replay/fork/blame path — that would let any
transitive dependency silently take over a security-relevant seam. Every
built-in implementation (the Anthropic provider adapter, ``IdentityMatcher``,
``StringMatchOracle``, the binary tape codec) is registered directly by its
owning module at import time, never through this entry-point path, so default
behavior never depends on ``importlib.metadata`` at all.
"""

from __future__ import annotations

import os
from importlib import metadata
from typing import TypeVar

T = TypeVar("T")

#: Comma-separated allowlist of entry-point names to auto-load, or "*" for all.
#: Unset (the default) means: load nothing. Purely opt-in — see module docstring.
ALLOW_PLUGINS_ENV = "TRACEFORK_ALLOW_PLUGINS"

PROVIDER_GROUP = "tracefork.providers"
ORACLE_GROUP = "tracefork.oracles"
SERIALIZER_GROUP = "tracefork.serializers"
MATCHER_GROUP = "tracefork.matchers"


def _env_allowlist() -> tuple[set[str], bool]:
    """Parse ``TRACEFORK_ALLOW_PLUGINS``: returns ``(explicit_names, allow_all)``."""
    raw = os.environ.get(ALLOW_PLUGINS_ENV, "")
    names = {n.strip() for n in raw.split(",") if n.strip()}
    return names, "*" in names


class Registry(dict[str, T]):
    """A ``dict[str, T]`` of registered implementations, plus an opt-in
    ``importlib.metadata`` entry-point loader gated by an explicit allowlist.

    Being a genuine ``dict`` subclass keeps every pre-existing direct-dict
    idiom (``sorted(registry)``, ``name in registry``, ``registry.pop(name,
    None)``) working unchanged for registries that were plain module-level
    dicts before this seam existed (e.g. the provider adapter registry).
    """

    def __init__(self, group: str, *, kind: str) -> None:
        super().__init__()
        self.group = group
        self.kind = kind
        self.loaded_entry_points: set[str] = set()

    def register(self, name: str, item: T) -> None:
        """Register ``item`` under ``name``, overwriting any prior entry."""
        self[name] = item

    def get_or_raise(self, name: str) -> T:
        """Look up ``name``, raising a ``KeyError`` that lists what IS registered."""
        try:
            return self[name]
        except KeyError:
            raise KeyError(
                f"no {self.kind} registered for {name!r}; registered: {sorted(self)}"
            ) from None

    def names(self) -> list[str]:
        """Sorted names of everything currently registered."""
        return sorted(self)

    def load_entry_points(
        self,
        *,
        allow: frozenset[str] | set[str] | None = None,
        allow_all: bool = False,
    ) -> list[str]:
        """Discover ``self.group`` entry points and register the allowlisted ones.

        Nothing is loaded unless the caller opts in via ``allow``/``allow_all``
        or the ``TRACEFORK_ALLOW_PLUGINS`` environment variable — see the
        module docstring. An entry point may point at a zero-argument class
        (instantiated on load) or a ready-made instance/callable; either way
        the loaded object is registered under the entry point's advertised
        name. Returns the names actually loaded (already-registered names are
        overwritten, matching ``register``'s semantics).
        """
        env_allow, env_all = _env_allowlist()
        allowed = set(allow or ()) | env_allow
        allow_all = allow_all or env_all
        if not allow_all and not allowed:
            return []
        loaded: list[str] = []
        for ep in metadata.entry_points(group=self.group):
            if not allow_all and ep.name not in allowed:
                continue
            obj = ep.load()
            instance = obj() if isinstance(obj, type) else obj
            self.register(ep.name, instance)
            self.loaded_entry_points.add(ep.name)
            loaded.append(ep.name)
        return loaded
