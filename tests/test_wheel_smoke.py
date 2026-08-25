"""Tests for scripts/wheel_smoke.py — the packaging gate `twine check` doesn't
cover: installing the built wheel into a clean throwaway venv (no source tree
on the path), importing it, asserting its force-included web/*.html assets
shipped, and running its console script. All offline, $0 (uv installs from the
local cache via `--offline`; no network, no API key).
"""

from __future__ import annotations

import subprocess

import pytest
from scripts.wheel_smoke import (
    REPO_ROOT,
    REQUIRED_WEB_ASSETS,
    _assert_outside_repo,
    _assert_web_assets,
    _venv_console_script_path,
    _venv_python_path,
    main,
    run_smoke,
)

# --- _assert_web_assets: unit-level, no venv/build required ---------------


def test_assert_web_assets_raises_when_one_is_missing(tmp_path):
    web_dir = tmp_path / "web"
    web_dir.mkdir()
    for name in REQUIRED_WEB_ASSETS[:-1]:
        (web_dir / name).write_text("x", encoding="utf-8")
    # last one intentionally absent

    with pytest.raises(SystemExit) as excinfo:
        _assert_web_assets(tmp_path)

    assert REQUIRED_WEB_ASSETS[-1] in str(excinfo.value)


def test_assert_web_assets_raises_when_web_dir_absent(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        _assert_web_assets(tmp_path)

    assert "web" in str(excinfo.value)


def test_assert_web_assets_passes_when_all_three_present(tmp_path):
    web_dir = tmp_path / "web"
    web_dir.mkdir()
    for name in REQUIRED_WEB_ASSETS:
        (web_dir / name).write_text("x", encoding="utf-8")

    _assert_web_assets(tmp_path)  # must not raise


# --- _assert_outside_repo: unit-level ---------------------------------------


def test_assert_outside_repo_raises_for_repo_root():
    with pytest.raises(SystemExit) as excinfo:
        _assert_outside_repo(REPO_ROOT)

    assert "repo checkout" in str(excinfo.value)


def test_assert_outside_repo_raises_for_path_under_repo_root():
    with pytest.raises(SystemExit):
        _assert_outside_repo(REPO_ROOT / "src" / "tracefork")


def test_assert_outside_repo_passes_for_unrelated_path(tmp_path):
    _assert_outside_repo(tmp_path)  # must not raise


# --- venv path helpers: unit-level ------------------------------------------


def test_venv_python_path_finds_posix_layout(tmp_path):
    (tmp_path / "bin").mkdir()
    python = tmp_path / "bin" / "python"
    python.write_text("", encoding="utf-8")

    assert _venv_python_path(tmp_path) == python


def test_venv_console_script_path_raises_when_absent(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        _venv_console_script_path(tmp_path, "tracefork")

    assert "tracefork" in str(excinfo.value)


# --- main(): argument handling, no build required for the failure path -----


def test_main_fails_loud_when_wheel_path_does_not_exist(tmp_path, capsys):
    missing = tmp_path / "does-not-exist.whl"

    exit_code = main(["--wheel", str(missing)])

    assert exit_code == 1
    err = capsys.readouterr().err
    assert "does-not-exist.whl" in err


# --- full end-to-end smoke: the actual point of this script ----------------


@pytest.mark.skipif(
    subprocess.run(["which", "uv"], capture_output=True).returncode != 0,
    reason="uv not on PATH",
)
def test_run_smoke_end_to_end_against_a_freshly_built_wheel(tmp_path):
    """Builds a real wheel, installs it into a real clean venv, and asserts
    the whole pipeline (import outside the repo + web assets present +
    console script runs) passes. This is the actual regression test for the
    gap the item describes: nothing previously ever installed the wheel."""
    dist_dir = tmp_path / "dist"
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(dist_dir)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    wheels = sorted(dist_dir.glob("tracefork-*.whl"))
    assert wheels, "uv build produced no wheel"

    run_smoke(wheels[-1])  # must not raise
