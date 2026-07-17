"""Tests for `confinement_diagnostics.py` — the confinement-violation
analogue of `test_divergence.py`'s coverage of `divergence.py`'s
`DivergenceDiagnostic`/`diagnose`/`diagnostic_to_dict` triad.

Two guarantees are load-bearing here:

1. `diagnose_confinement` reads a `ConfinementViolationError`'s own
   structured attributes (`violation_kind`/`attempted`/
   `declared_writable_roots`/`declared_allowed_hosts`, set at both
   `boundary_guard.py` raise sites since tracefork-bge.72) — never
   `str(error)` — for both the write and connect violation shapes.
2. `confinement_diagnostic_to_dict` is a JSON-safe view that round-trips.

The two CLI-level tests exercise `tracefork fork`'s `--writable-root`/
`--allowed-host` flags, which wire `ConfinementSpec`/the diagnostic echo
into `cli.py` (owned by a separate integration step for this bead — see
`test_cli_fork_confinement_violation_prints_diagnostic_and_exits_nonzero`'s
skip reason). `test_cli_fork_no_confinement_flags_preserves_existing_behavior`
needs no cli.py change at all and runs for real: it is the regression proof
that omitting the new flags is byte-identical to today.
"""

from __future__ import annotations

import json
import socket
from pathlib import Path

import anthropic
import pytest
from typer.testing import CliRunner

from tests.fakes import make_text_response
from tracefork.boundary_guard import BoundaryGuard, ConfinementSpec, ConfinementViolationError
from tracefork.cli import app
from tracefork.confinement_diagnostics import (
    ConfinementDiagnostic,
    confinement_diagnostic_to_dict,
    diagnose_confinement,
)
from tracefork.store import TapeStore
from tracefork.validate import _record_clean_tape

runner = CliRunner()


def _seeded_store(tmp_path: Path) -> tuple[Path, str]:
    """Mirrors `test_cli_smoke.py`'s helper of the same name (kept local so
    this test file doesn't couple to another test module's internals)."""
    db = tmp_path / "store.db"
    store = TapeStore(str(db))
    run_id = store.save_tape(_record_clean_tape(), run_id="confinement-diag-run")
    store.close()
    return db, run_id


# ── diagnose_confinement / confinement_diagnostic_to_dict ──────────────────


def test_diagnose_confinement_write_violation_captures_structured_fields(tmp_path):
    writable_root = tmp_path / "allowed"
    writable_root.mkdir()
    outside_file = tmp_path / "leak.txt"

    with (
        BoundaryGuard(confinement=ConfinementSpec(writable_roots=(str(writable_root),))),
        pytest.raises(ConfinementViolationError) as excinfo,
        open(outside_file, "w"),
    ):
        pass

    diag = diagnose_confinement(excinfo.value)
    assert diag.violation_kind == "write"
    assert diag.attempted == str(outside_file.resolve())
    assert diag.declared_writable_roots == (str(writable_root),)
    assert diag.declared_allowed_hosts is None


def test_diagnose_confinement_connect_violation_captures_structured_fields():
    with (
        BoundaryGuard(confinement=ConfinementSpec(allowed_hosts=("good.example.com",))),
        pytest.raises(ConfinementViolationError) as excinfo,
    ):
        # `.connect` is patched before any DNS/TCP attempt runs -- no real
        # syscall, offline/$0 even for this rejection path.
        socket.socket().connect(("evil.example.com", 443))

    diag = diagnose_confinement(excinfo.value)
    assert diag.violation_kind == "connect"
    assert diag.attempted == "evil.example.com"
    assert diag.declared_allowed_hosts == ("good.example.com",)
    assert diag.declared_writable_roots is None


def test_confinement_diagnostic_to_dict_is_json_safe():
    diag = ConfinementDiagnostic(
        violation_kind="write",
        attempted="/tmp/leak.txt",
        declared_writable_roots=("/tmp/allowed",),
        declared_allowed_hosts=None,
        message="denied",
    )
    encoded = json.dumps(confinement_diagnostic_to_dict(diag))
    round_tripped = json.loads(encoded)
    assert round_tripped == {
        "violation_kind": "write",
        "attempted": "/tmp/leak.txt",
        "declared_writable_roots": ["/tmp/allowed"],
        "declared_allowed_hosts": None,
        "message": "denied",
    }


# ── CLI: `tracefork fork --writable-root/--allowed-host` ───────────────────

# Set by the violation test just before invoking the CLI (CliRunner runs
# in-process, so a module global set right before `runner.invoke` is visible
# to the agent function the CLI resolves and calls by import path).
_leak_target: Path | None = None


def _cli_write_attempting_agent(client: anthropic.Anthropic) -> str:
    """Same two-turn shape as `tracefork.validate.synthetic_agent` (so it
    matches `_record_clean_tape()`'s recorded prefix), but attempts an
    out-of-bounds write between turns -- inside `ForkEngine.fork`'s
    confinement-guarded `agent_fn(client)` window regardless of which step
    is the divergence step."""
    r1 = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=100,
        messages=[{"role": "user", "content": "book a flight to Tokyo"}],
    )
    assert _leak_target is not None
    with open(_leak_target, "w") as f:
        f.write("leak")
    r2 = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=100,
        messages=[
            {"role": "user", "content": "book a flight to Tokyo"},
            {"role": "assistant", "content": r1.content[0].text},
            {"role": "user", "content": "confirm"},
        ],
    )
    return r2.content[0].text


def test_cli_fork_confinement_violation_prints_diagnostic_and_exits_nonzero(tmp_path):
    global _leak_target
    db, run_id = _seeded_store(tmp_path)
    writable_root = tmp_path / "allowed"
    writable_root.mkdir()
    _leak_target = tmp_path / "leak.txt"

    resp_path = tmp_path / "mutated.bytes"
    resp_path.write_bytes(make_text_response("FAIL — cancelled"))

    result = runner.invoke(
        app,
        [
            "fork",
            run_id,
            "--step",
            "1",
            "--response",
            str(resp_path),
            "--agent",
            "tests.test_confinement_diagnostics:_cli_write_attempting_agent",
            "--store",
            str(db),
            "--writable-root",
            str(writable_root),
        ],
    )
    assert result.exit_code == 1, result.output
    assert "Confinement violation" in result.output
    assert "writable_roots" in result.output
    assert not _leak_target.exists()


def test_cli_fork_no_confinement_flags_preserves_existing_behavior(tmp_path):
    """Omitting `--writable-root`/`--allowed-host` entirely must leave
    `fork`'s behavior byte-identical to before this bead: `confinement=None`,
    exit 0, `Fork created`. This needs no cli.py change to run -- it is the
    regression proof for today's baseline."""
    db, run_id = _seeded_store(tmp_path)
    resp_path = tmp_path / "mutated.bytes"
    resp_path.write_bytes(make_text_response("FAIL — cancelled"))

    result = runner.invoke(
        app,
        [
            "fork",
            run_id,
            "--step",
            "1",
            "--response",
            str(resp_path),
            "--agent",
            "tracefork.validate:synthetic_agent",
            "--store",
            str(db),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Fork created" in result.output
