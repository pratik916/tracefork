"""Interactive shell wrapper around `query.dispatch` -- carries no query
logic of its own; the full state/diff/causes/tree grammar lives entirely in
`query.py`, kept importable/testable standalone with no `cmd`/readline
dependency. This module is the thin, stdlib-only (`cmd.Cmd`) interactive
loop on top of it.
"""

from __future__ import annotations

import cmd

from .query import QueryError, dispatch
from .store import TapeStore

__all__ = ["QueryShell", "run_repl"]


class QueryShell(cmd.Cmd):
    """A `cmd.Cmd` shell over `query.dispatch`.

    `default()` is the only method carrying any real behavior: every input
    line -- whatever `cmd.Cmd` would otherwise try to route to a `do_<word>`
    method -- is instead handed straight to `dispatch()`, and its result (or
    `error: ...` on a `QueryError`) is printed. No verb-specific `do_*`
    method exists, so the grammar itself stays defined in exactly one place.
    """

    intro = "tracefork query -- state/diff/causes/tree, or 'exit'/'quit'/Ctrl-D to leave."
    prompt = "tracefork> "

    def __init__(self, store: TapeStore) -> None:
        super().__init__()
        self._store = store

    def default(self, line: str) -> None:
        try:
            print(dispatch(self._store, line))
        except QueryError as exc:
            print(f"error: {exc}")

    def do_exit(self, _arg: str) -> bool:
        return True

    def do_quit(self, _arg: str) -> bool:
        return True

    def do_EOF(self, _arg: str) -> bool:
        print()
        return True

    def emptyline(self) -> bool:
        # cmd.Cmd's default re-runs the last command on a blank line; a
        # no-op here is the less surprising behavior for this shell.
        return False


def run_repl(store: TapeStore) -> None:
    """Construct a `QueryShell` over `store` and run its interactive loop."""
    QueryShell(store).cmdloop()
