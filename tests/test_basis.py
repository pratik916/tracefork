"""basis.py tests: the RecordBasis build witness, and Recorder's opt-in
record_basis= wiring (see recorder.py and basis.py's module docstrings).
"""

from __future__ import annotations

import subprocess
from importlib import metadata

import httpx
import pytest

from tests.fakes import ScriptedFakeLLM, make_text_response
from tracefork import Recorder
from tracefork.basis import (
    RecordBasis,
    basis_from_provenance,
    basis_to_provenance_keys,
    current_basis,
    diff_basis,
    format_basis_drift_warning,
)

TEXT_RESP = make_text_response("hi")


def test_current_basis_version_matches_installed_package():
    basis = current_basis()
    assert basis.tracefork_version == metadata.version("tracefork")


def test_current_basis_explicit_git_sha_used_verbatim_no_subprocess(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("subprocess.run should not be called when git_sha is given")

    monkeypatch.setattr(subprocess, "run", _boom)
    basis = current_basis(git_sha="deadbeef")
    assert basis.git_sha == "deadbeef"


@pytest.mark.parametrize(
    "exc",
    [
        FileNotFoundError("no git binary"),
        subprocess.CalledProcessError(1, ["git", "rev-parse", "HEAD"]),
        subprocess.TimeoutExpired(["git", "rev-parse", "HEAD"], 5),
    ],
)
def test_current_basis_auto_detect_swallows_subprocess_failures(monkeypatch, exc):
    def _raise(*args, **kwargs):
        raise exc

    monkeypatch.setattr(subprocess, "run", _raise)
    basis = current_basis()
    assert basis.git_sha == ""


def test_basis_from_provenance_empty_is_none():
    assert basis_from_provenance({}) is None


def test_basis_from_provenance_round_trips_defaulting_git_sha():
    basis = basis_from_provenance({"tracefork_version": "x"})
    assert basis == RecordBasis(tracefork_version="x", git_sha="")


def test_basis_to_provenance_keys_round_trip():
    basis = RecordBasis(tracefork_version="1.2.3", git_sha="cafebabe")
    keys = basis_to_provenance_keys(basis)
    assert keys == {"tracefork_version": "1.2.3", "git_sha": "cafebabe"}
    assert basis_from_provenance(keys) == basis


def test_diff_basis_identical_is_empty():
    basis = RecordBasis(tracefork_version="1.0.0", git_sha="abc123")
    assert diff_basis(basis, basis) == []


def test_diff_basis_version_only_mismatch_one_line():
    recorded = RecordBasis(tracefork_version="1.0.0", git_sha="")
    current = RecordBasis(tracefork_version="2.0.0", git_sha="")
    lines = diff_basis(recorded, current)
    assert len(lines) == 1
    assert "tracefork_version" in lines[0]


def test_diff_basis_git_sha_mismatch_only_when_both_non_empty():
    recorded = RecordBasis(tracefork_version="1.0.0", git_sha="abc123")
    current = RecordBasis(tracefork_version="1.0.0", git_sha="def456")
    lines = diff_basis(recorded, current)
    assert len(lines) == 1
    assert "git_sha" in lines[0]


def test_diff_basis_missing_sha_on_either_side_never_flagged():
    # Recorded has no sha (installed wheel / no git checkout at record time).
    recorded = RecordBasis(tracefork_version="1.0.0", git_sha="")
    current = RecordBasis(tracefork_version="1.0.0", git_sha="def456")
    assert diff_basis(recorded, current) == []

    # Current has no sha (replaying from an installed wheel / no checkout).
    recorded2 = RecordBasis(tracefork_version="1.0.0", git_sha="abc123")
    current2 = RecordBasis(tracefork_version="1.0.0", git_sha="")
    assert diff_basis(recorded2, current2) == []


def test_diff_basis_two_lines_when_both_differ():
    recorded = RecordBasis(tracefork_version="1.0.0", git_sha="abc123")
    current = RecordBasis(tracefork_version="2.0.0", git_sha="def456")
    lines = diff_basis(recorded, current)
    assert len(lines) == 2


def test_format_basis_drift_warning_none_on_no_drift():
    basis = RecordBasis(tracefork_version="1.0.0", git_sha="abc123")
    assert format_basis_drift_warning(basis, basis) is None


def test_format_basis_drift_warning_multiline_string_on_drift():
    recorded = RecordBasis(tracefork_version="1.0.0", git_sha="abc123")
    current = RecordBasis(tracefork_version="2.0.0", git_sha="def456")
    warning = format_basis_drift_warning(recorded, current)
    assert warning is not None
    assert len(warning.splitlines()) > 1
    assert "tracefork_version" in warning
    assert "git_sha" in warning


def _sync_client(fake: ScriptedFakeLLM) -> httpx.Client:
    import anthropic

    return anthropic.Anthropic(
        api_key="sk-ant-fake",
        http_client=httpx.Client(transport=fake),
        max_retries=0,
    )


def test_recorder_record_basis_true_adds_provenance_keys():
    fake = ScriptedFakeLLM([TEXT_RESP])
    client = _sync_client(fake)
    with Recorder(client, record_basis=True, basis_git_sha="cafebabe") as rec:
        rec.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=100,
            messages=[{"role": "user", "content": "hi"}],
        )
    assert rec.tape.provenance["tracefork_version"] == metadata.version("tracefork")
    assert rec.tape.provenance["git_sha"] == "cafebabe"


def test_recorder_default_record_basis_leaves_provenance_unchanged():
    fake = ScriptedFakeLLM([TEXT_RESP])
    client = _sync_client(fake)
    with Recorder(client) as rec:
        rec.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=100,
            messages=[{"role": "user", "content": "hi"}],
        )
    assert "tracefork_version" not in rec.tape.provenance
    assert "git_sha" not in rec.tape.provenance
    assert rec.tape.provenance == {
        "matcher_name": "identity",
        "boundary_guard": "false",
        "nondet_mode": "recording",
    }
