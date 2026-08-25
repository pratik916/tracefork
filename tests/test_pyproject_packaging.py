"""Packaging-config regression tests over pyproject.toml's build targets.

Runs a REAL `uv build` (not a mock) and inspects the resulting wheel/sdist
contents directly -- these are config-only changes (no importable behavior),
so the build artifact itself is the thing under test. Session-scoped: one
build is shared by every test in this module, since `uv build` is the
expensive part and none of these tests mutate the artifacts they inspect.

See pyproject.toml's `[tool.hatch.build.targets.wheel]`/`[tool.hatch.build.
targets.sdist]` and CLAUDE.md's packaging invariants.
"""

from __future__ import annotations

import subprocess
import tarfile
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def built_dist(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build both the wheel and the sdist into a scratch --out-dir."""
    out_dir = tmp_path_factory.mktemp("dist")
    subprocess.run(
        ["uv", "build", "--out-dir", str(out_dir), "--clear"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return out_dir


def _wheel_path(dist_dir: Path) -> Path:
    matches = sorted(dist_dir.glob("tracefork-*.whl"))
    assert matches, f"no wheel found in {dist_dir}"
    return matches[0]


def _sdist_path(dist_dir: Path) -> Path:
    matches = sorted(dist_dir.glob("tracefork-*.tar.gz"))
    assert matches, f"no sdist found in {dist_dir}"
    return matches[0]


def test_wheel_does_not_ship_tracefork_spike(built_dist: Path):
    """src/tracefork_spike is the retired Spike 0 -- it must never install as
    a second top-level package (name-collision risk: its own Tape class is
    an incompatible type from tracefork.Tape). It stays runnable from a
    source checkout via pytest's `pythonpath = ["src", "."]` and uv's
    editable .pth, neither of which route through the wheel's packages list.
    """
    wheel = _wheel_path(built_dist)
    with zipfile.ZipFile(wheel) as zf:
        names = zf.namelist()
    spike_entries = [n for n in names if "tracefork_spike" in n]
    assert spike_entries == [], f"wheel must not contain tracefork_spike, found: {spike_entries}"


def test_wheel_still_ships_tracefork_package(built_dist: Path):
    """Sanity check the fix didn't also drop the real package."""
    wheel = _wheel_path(built_dist)
    with zipfile.ZipFile(wheel) as zf:
        names = zf.namelist()
    assert any(n.startswith("tracefork/__init__.py") for n in names)
    assert any(n.endswith("tracefork/web/report.html") for n in names)


def test_sdist_excludes_untracked_and_local_dev_scaffolding(built_dist: Path):
    """docs/shepherd-gap-analysis.md is deliberately untracked (the project's
    naming policy relies on it never landing in a shipped artifact); the
    default hatchling sdist sweeps by .gitignore pattern, not git-tracking
    status, and matched none before this fix. .hypothesis/ (pytest-cache
    noise) and .claude/ (private dev-tool hooks) must not ship either.
    """
    sdist = _sdist_path(built_dist)
    with tarfile.open(sdist, mode="r:gz") as tf:
        names = tf.getnames()
    offenders = [
        n for n in names if "shepherd-gap-analysis" in n or "/.hypothesis/" in n or "/.claude/" in n
    ]
    assert offenders == [], f"sdist must not contain these paths, found: {offenders[:10]}"


def test_sdist_still_contains_source_and_readme(built_dist: Path):
    """Sanity check the exclude allowlist didn't also swallow real content."""
    sdist = _sdist_path(built_dist)
    with tarfile.open(sdist, mode="r:gz") as tf:
        names = tf.getnames()
    assert any(n.endswith("src/tracefork/__init__.py") for n in names)
    assert any(n.endswith("README.md") for n in names)
    assert any(n.endswith("pyproject.toml") for n in names)
