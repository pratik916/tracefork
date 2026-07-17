"""Multi-OS-process concurrent-write stress test for `TapeStore`/`open_sqlite`'s
WAL + busy_timeout pragma bundle. `tests/test_storage.py`'s
`test_concurrent_writers_separate_connections_no_lock` proves the same pragma
bundle survives concurrent *threads*, but threads share one GIL/interpreter/
process — they can't exercise genuine OS-level file locking. This module
spawns real `multiprocessing.Process` workers (the `spawn` context, so each
worker imports `tracefork.store` fresh in its own interpreter) to prove the
pragma bundle holds under actual multi-process contention. All offline and $0."""

from __future__ import annotations

import multiprocessing
import pathlib

from tracefork.store import TapeStore
from tracefork.tape import Tape, open_sqlite


def _mp_tape(tag: str) -> Tape:
    t = Tape(agent_name="w")
    t.append_exchange(f"req-{tag}".encode(), f"resp-{tag}".encode())
    return t


def _mp_worker(db_path: str, w: int, n_writes: int, q: multiprocessing.Queue) -> None:
    """Runs in its own spawned process: opens a fresh `TapeStore` against the
    shared `db_path` and performs `n_writes` `save_tape` calls with unique
    run_ids. Never lets an exception cross the process boundary unreported —
    the parent learns of failure only through the queue and `p.exitcode`."""
    try:
        s = TapeStore(db_path)
        for j in range(n_writes):
            s.save_tape(_mp_tape(f"{w}-{j}"), run_id=f"r{w}_{j}")
        s.close()
        q.put(("ok", w))
    except BaseException as exc:  # noqa: BLE001
        q.put(("err", w, repr(exc)))


def _run_workers(db: str, n_workers: int, n_writes: int) -> list[tuple]:
    ctx = multiprocessing.get_context("spawn")
    q: multiprocessing.Queue = ctx.Queue()
    procs = [ctx.Process(target=_mp_worker, args=(db, w, n_writes, q)) for w in range(n_workers)]
    for p in procs:
        p.start()
    results = [q.get(timeout=30) for _ in procs]
    for p in procs:
        p.join(timeout=30)
    return procs, results  # type: ignore[return-value]


def test_multiprocess_writers_no_database_locked_error(tmp_path: pathlib.Path) -> None:
    """Six real OS processes, each with its own connection, hammering one
    db file: WAL + busy_timeout must let writers queue instead of raising
    `database is locked` — the guarantee threading alone cannot prove."""
    db = str(tmp_path / "store.db")
    TapeStore(db).close()  # create schema up front

    n_workers, n_writes = 6, 5
    procs, results = _run_workers(db, n_workers, n_writes)

    errors = [r for r in results if r[0] == "err"]
    assert not errors, errors
    for p in procs:
        assert p.exitcode == 0, f"worker process exited with code {p.exitcode}"
    assert sorted(r[1] for r in results) == list(range(n_workers))

    store = TapeStore(db)
    try:
        assert len(store.list_runs()) == n_workers * n_writes
    finally:
        store.close()


def test_multiprocess_pragmas_survive_concurrent_process_opens(tmp_path: pathlib.Path) -> None:
    """After a multi-process write burst deliberately sized to maximize
    open/close churn (10 workers x 3 writes) on the pragma bundle, a fresh
    `open_sqlite()` from the parent must still see WAL + busy_timeout=5000 +
    foreign_keys=ON — repeated cross-process opens never downgrade it."""
    db = str(tmp_path / "store.db")
    TapeStore(db).close()  # create schema up front

    n_workers, n_writes = 10, 3
    procs, results = _run_workers(db, n_workers, n_writes)

    errors = [r for r in results if r[0] == "err"]
    assert not errors, errors
    for p in procs:
        assert p.exitcode == 0, f"worker process exited with code {p.exitcode}"

    con = open_sqlite(db)
    try:
        assert con.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert con.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
        assert con.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    finally:
        con.close()
