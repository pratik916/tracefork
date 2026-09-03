#!/usr/bin/env python3
"""Wheel smoke test — the packaging gate `twine check` doesn't cover.

`twine check` validates wheel *metadata* only (README renders, classifiers are
well-formed). Before this script, nothing in CI or scripts/e2e.sh ever
installed the wheel it built, so a wheel that is metadata-valid but broken at
import time, missing a force-included asset, or missing its console-script
entry point could still ship. This closes that structural blind spot:

    1. build (or reuse) the wheel,
    2. create a CLEAN throwaway venv with no source tree on its path,
    3. install ONLY the wheel into it (no editable install, no `-e .`),
    4. import tracefork from the installed site-packages copy and assert it
       did NOT resolve back to the repo checkout,
    5. assert the three web/*.html assets that report.py force-includes for
       the installed-wheel case are actually present. This matters because
       report.py's `_template_path`/`_runs_template_path` silently fall back
       to the repo-root `web/*.html` copies when run from a source checkout
       (see src/tracefork/report.py) — running only from source would mask
       a wheel that shipped without them.
    6. run the installed `tracefork` console script.

Usage:
    uv run python scripts/wheel_smoke.py                # builds its own wheel
    uv run python scripts/wheel_smoke.py --wheel dist/tracefork-*.whl

$0, no API key: this is a real `pip install` of the built wheel, so it DOES
resolve/download dependencies from PyPI like a genuine install would — that is
the thing being tested. `--offline` was tried and reverted: uv's local cache
stores some entries as revalidatable HTTP responses, and `uv pip install
--offline` against a wheel *file* (as opposed to `uv sync`, which reads exact
versions straight from the lockfile) needs to enumerate available versions to
resolve a range like `anthropic>=1.0,<2` — reproducibly failing with "not
found in the cache" even when the exact wheel is sitting in that same cache,
on a cold `UV_CACHE_DIR` (verified locally, matches the CI failure mode).
Forcing this test offline bought no determinism (nothing here is a captured
tape) and cost real reliability, so it isn't.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REQUIRED_WEB_ASSETS = ("report.html", "runs.html", "session_report.html")


def _run(cmd: list[str], *, cwd: Path, capture: bool = False) -> subprocess.CompletedProcess[str]:
    print(f"    $ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=capture, text=True)


def _build_wheel(out_dir: Path) -> Path:
    print(f"==> building a fresh wheel into {out_dir}")
    _run(["uv", "build", "--wheel", "--out-dir", str(out_dir)], cwd=REPO_ROOT)
    wheels = sorted(out_dir.glob("tracefork-*.whl"))
    if not wheels:
        raise SystemExit("FAIL: `uv build --wheel` produced no wheel")
    return wheels[-1]


def _create_clean_venv(venv_dir: Path, cwd: Path) -> None:
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    print(f"==> creating a clean throwaway venv (python {py_version}, no source tree on path)")
    _run(["uv", "venv", "--python", py_version, str(venv_dir)], cwd=cwd)


def _venv_python_path(venv_dir: Path) -> Path:
    posix = venv_dir / "bin" / "python"
    if posix.exists():
        return posix
    windows = venv_dir / "Scripts" / "python.exe"
    if windows.exists():
        return windows
    raise SystemExit(f"FAIL: no python interpreter found under {venv_dir}")


def _venv_console_script_path(venv_dir: Path, name: str) -> Path:
    posix = venv_dir / "bin" / name
    if posix.exists():
        return posix
    windows = venv_dir / "Scripts" / f"{name}.exe"
    if windows.exists():
        return windows
    raise SystemExit(
        f"FAIL: console script '{name}' not installed under {venv_dir} "
        "-- check [project.scripts] in pyproject.toml"
    )


def _install_wheel(venv_python: Path, wheel: Path, cwd: Path) -> None:
    print("==> installing ONLY the built wheel (no editable/source install)")
    _run(
        ["uv", "pip", "install", "--python", str(venv_python), str(wheel)],
        cwd=cwd,
    )


def _installed_package_dir(venv_python: Path, cwd: Path) -> Path:
    print("==> importing tracefork from the installed wheel")
    result = _run(
        [
            str(venv_python),
            "-c",
            "import tracefork, pathlib; print(pathlib.Path(tracefork.__file__).resolve().parent)",
        ],
        cwd=cwd,  # never REPO_ROOT: rules out shadowing by the source tree
        capture=True,
    )
    return Path(result.stdout.strip())


def _assert_outside_repo(installed_pkg: Path) -> None:
    if installed_pkg == REPO_ROOT or REPO_ROOT in installed_pkg.parents:
        raise SystemExit(
            f"FAIL: tracefork imported from the repo checkout ({installed_pkg}), "
            "not the installed wheel -- the venv is not clean"
        )
    print(f"    installed package: {installed_pkg}")


def _assert_web_assets(installed_pkg: Path) -> None:
    print("==> asserting web/*.html assets shipped inside the installed wheel")
    web_dir = installed_pkg / "web"
    missing = [name for name in REQUIRED_WEB_ASSETS if not (web_dir / name).is_file()]
    if missing:
        raise SystemExit(
            f"FAIL: installed wheel is missing web assets {missing} in {web_dir} "
            "-- pyproject.toml's [tool.hatch.build.targets.wheel.force-include] "
            "must force-include every web/*.html report.py can fall back to"
        )
    for name in REQUIRED_WEB_ASSETS:
        print(f"    {name}: {(web_dir / name).stat().st_size} bytes")


def _run_console_script(venv_dir: Path, cwd: Path) -> None:
    print("==> running the installed console script")
    console_script = _venv_console_script_path(venv_dir, "tracefork")
    _run([str(console_script), "--help"], cwd=cwd, capture=True)


def run_smoke(wheel: Path | None) -> None:
    """Run the full install-into-clean-venv smoke test.

    Raises ``SystemExit`` with a ``FAIL: ...`` message on any failure;
    ``subprocess.CalledProcessError`` propagates uncaught from any failed
    subprocess stage (never swallowed).
    """
    with tempfile.TemporaryDirectory(prefix="tracefork-wheel-smoke-") as tmp:
        tmp_path = Path(tmp)

        if wheel is None:
            wheel = _build_wheel(tmp_path / "dist")
        wheel = wheel.resolve()
        if not wheel.is_file():
            raise SystemExit(f"FAIL: wheel not found: {wheel}")
        print(f"    using wheel: {wheel.name}")

        venv_dir = tmp_path / "venv"
        _create_clean_venv(venv_dir, cwd=tmp_path)
        venv_python = _venv_python_path(venv_dir)

        _install_wheel(venv_python, wheel, cwd=tmp_path)

        installed_pkg = _installed_package_dir(venv_python, cwd=tmp_path)
        _assert_outside_repo(installed_pkg)
        _assert_web_assets(installed_pkg)
        _run_console_script(venv_dir, cwd=tmp_path)

    print()
    print(
        "PASS -- wheel installs clean, imports outside the repo, ships "
        "web/*.html, and its console script runs."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wheel",
        type=Path,
        default=None,
        help="path to an already-built wheel to smoke-test (default: build a fresh one)",
    )
    args = parser.parse_args(argv)
    try:
        run_smoke(args.wheel)
    except SystemExit as exc:
        message = exc.code if isinstance(exc.code, str) else str(exc)
        print(message, file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as exc:
        print(
            f"FAIL: command exited {exc.returncode}: {' '.join(str(c) for c in exc.cmd)}",
            file=sys.stderr,
        )
        if exc.stdout:
            print(exc.stdout, file=sys.stderr)
        if exc.stderr:
            print(exc.stderr, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
