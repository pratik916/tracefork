"""``RecordBasis`` -- the "what build recorded this tape?" witness.

Layered on top of ``Tape.provenance`` (see ``tape.py``) exactly the way
``matcher_name``/``boundary_guard``/``nondet_mode`` already are: two more
optional string keys, ``tracefork_version``/``git_sha``, never fed into
``digest()``. This module is pure and offline -- no ``Tape``/CLI imports, so
it's safely importable from anywhere (including ``tape.py`` itself, should
that ever be useful, without a cycle).

``current_basis()`` captures the running ``tracefork`` package version (via
``importlib.metadata.version("tracefork")``) and a best-effort git commit sha
(``git rev-parse HEAD``, swallowing every failure -- missing git binary,
non-repo checkout, or a hung/erroring subprocess -- to ``""``; a caller
running from an installed wheel with no ``.git`` around is not an error).
``Recorder``/``AsyncRecorder``'s opt-in ``record_basis=True`` (see
``recorder.py``) writes it into ``tape.provenance`` via
``basis_to_provenance_keys``; ``cli.py``'s ``replay``/``fork``/
``coalition_fork`` commands read it back via ``basis_from_provenance`` and
print a non-fatal drift WARNING (``format_basis_drift_warning``) when the
replaying build differs from the one that recorded the tape. This is
distinct in *kind* from ``replay.py``'s ``ProvenanceMismatchError`` (a hard
error on a ``matcher_name`` mismatch, since replaying under a different
matcher genuinely breaks fingerprint comparison): a version/sha drift is
diagnostic context about *what changed*, never a correctness violation, so
it never raises and never changes a command's exit code.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from importlib import metadata


@dataclass(frozen=True)
class RecordBasis:
    """The tracefork build that recorded (or is about to replay) a tape.

    ``git_sha`` defaults to ``""`` -- a build outside a git checkout (an
    installed wheel, a shallow clone with no ``.git``) has no sha to report,
    and that absence is not itself a drift signal (see ``diff_basis``).
    """

    tracefork_version: str
    git_sha: str = ""


def _auto_git_sha() -> str:
    """Best-effort ``git rev-parse HEAD``, swallowing every failure to
    ``""`` -- no git binary, not a git checkout, a non-zero exit, or a hung
    subprocess must never raise here."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except Exception:
        return ""
    return result.stdout.strip()


def current_basis(*, git_sha: str | None = None) -> RecordBasis:
    """The running build's ``RecordBasis``.

    ``git_sha``, if given, is used verbatim -- no subprocess call at all.
    Otherwise it's auto-detected via ``git rev-parse HEAD``, swallowing every
    failure to ``""`` (see ``_auto_git_sha``).
    """
    resolved_sha = git_sha if git_sha is not None else _auto_git_sha()
    return RecordBasis(tracefork_version=metadata.version("tracefork"), git_sha=resolved_sha)


def basis_to_provenance_keys(basis: RecordBasis) -> dict[str, str]:
    """Serialize a ``RecordBasis`` into the two ``Tape.provenance`` string
    keys ``Recorder``/``AsyncRecorder`` additively ``.update()`` in when
    ``record_basis=True`` (see ``recorder.py``)."""
    return {"tracefork_version": basis.tracefork_version, "git_sha": basis.git_sha}


def basis_from_provenance(provenance: dict[str, str]) -> RecordBasis | None:
    """Deserialize a tape's recorded basis from its ``provenance`` dict, or
    ``None`` if the tape never recorded one -- every pre-basis tape, or one
    recorded by ``Recorder(record_basis=False)`` (today's default)."""
    version = provenance.get("tracefork_version")
    if version is None:
        return None
    return RecordBasis(tracefork_version=version, git_sha=provenance.get("git_sha", ""))


def diff_basis(recorded: RecordBasis, current: RecordBasis) -> list[str]:
    """Human-readable drift lines between a tape's recorded basis and the
    current build, or ``[]`` if they match on every axis this checks.

    A ``git_sha`` mismatch is reported ONLY when both sides are non-empty --
    a tape (or a build) with no sha to compare is never flagged as drifted
    on that axis alone.
    """
    lines = []
    if recorded.tracefork_version != current.tracefork_version:
        lines.append(
            f"tracefork_version: recorded {recorded.tracefork_version!r}, "
            f"current {current.tracefork_version!r}"
        )
    if recorded.git_sha and current.git_sha and recorded.git_sha != current.git_sha:
        lines.append(f"git_sha: recorded {recorded.git_sha!r}, current {current.git_sha!r}")
    return lines


def format_basis_drift_warning(recorded: RecordBasis, current: RecordBasis) -> str | None:
    """A non-fatal, multi-line WARNING block for ``cli.py`` to
    ``typer.echo``, or ``None`` when there's no drift to report (see
    ``diff_basis``). Never raises, never implies an exit-code change --
    distinct in kind from ``replay.py``'s ``ProvenanceMismatchError``.
    """
    lines = diff_basis(recorded, current)
    if not lines:
        return None
    body = "\n".join(f"    {line}" for line in lines)
    return f"  WARNING: this tape was recorded under a different tracefork build\n{body}"
