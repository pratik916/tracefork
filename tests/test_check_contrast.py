"""Tests for scripts/check_contrast.py, the WCAG AA contrast auditor for
web/*.html's shared design-token blocks (part of the v1.0.0 workbench-shell
readiness item: prove AA contrast in both themes rather than just claim it).

Imports the script by file path (it's a standalone scripts/ entry point, not
a package module -- same reason check_executed_evidence.py has no direct
import-based test today; this one adds one via importlib since the contrast
math itself is worth unit-testing directly, not just via the CLI exit code).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_contrast.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("check_contrast", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_contrast"] = module
    spec.loader.exec_module(module)
    return module


cc = _load_module()


def test_relative_luminance_black_and_white():
    assert cc.relative_luminance("#000000") == 0.0
    assert abs(cc.relative_luminance("#ffffff") - 1.0) < 1e-9


def test_contrast_ratio_black_on_white_is_21():
    assert abs(cc.contrast_ratio("#000000", "#ffffff") - 21.0) < 1e-6


def test_contrast_ratio_is_symmetric():
    a, b = cc.contrast_ratio("#123456", "#abcdef"), cc.contrast_ratio("#abcdef", "#123456")
    assert abs(a - b) < 1e-9


def test_contrast_ratio_same_color_is_one():
    assert abs(cc.contrast_ratio("#336699", "#336699") - 1.0) < 1e-9


def test_extract_css_variables_parses_root_block():
    css = """
    :root {
      --bg: #0d1117; --text: #c9d1d9;
      --muted: #838b95;
    }
    """
    tokens = cc.extract_css_variables(css, ":root")
    assert tokens["bg"] == "#0d1117"
    assert tokens["text"] == "#c9d1d9"
    assert tokens["muted"] == "#838b95"


def test_extract_css_variables_parses_light_theme_block():
    css = """
    :root { --bg: #000000; }
    [data-theme="light"] {
      --bg: #ffffff;
      --text: #1f2328;
    }
    """
    tokens = cc.extract_css_variables(css, '[data-theme="light"]')
    assert tokens["bg"] == "#ffffff"
    assert tokens["text"] == "#1f2328"


def test_resolve_theme_tokens_light_inherits_and_overrides_dark():
    css = """
    :root { --bg: #0d1117; --text: #c9d1d9; --border: #30363d; }
    [data-theme="light"] { --bg: #ffffff; --text: #1f2328; }
    """
    dark, light = cc.resolve_theme_tokens(css)
    assert dark["bg"] == "#0d1117"
    assert light["bg"] == "#ffffff"
    # light theme inherits any token it doesn't itself override
    assert light["border"] == "#30363d"


def test_check_file_reports_failure_for_low_contrast_pair(tmp_path):
    # muted-on-bg deliberately too low-contrast to pass AA (ratio ~1.5)
    html = tmp_path / "bad.html"
    html.write_text("""
    <style>
    :root { --bg: #0d1117; --muted: #1a1e24; --text: #c9d1d9; --surface: #161b22;
            --green: #3fb950; --red: #f85149; --blue: #58a6ff; --orange: #f0883e;
            --purple: #d2a8ff; --yellow: #e3b341;
            --badge-pass-bg: #1a3a1a; --badge-fail-bg: #3a1a1a; --badge-info-bg: #1a2a3a;
            --badge-warn-bg: #3a3010; --badge-redacted-bg: #3a2a10; }
    [data-theme="light"] { --bg: #ffffff; --muted: #59636e; --text: #1f2328; --surface: #f6f8fa;
            --green: #1a7f37; --red: #d1242f; --blue: #0969da; --orange: #bc4c00;
            --purple: #8250df; --yellow: #9a6700;
            --badge-pass-bg: #dafbe1; --badge-fail-bg: #ffebe9; --badge-info-bg: #ddf4ff;
            --badge-warn-bg: #fff8c5; --badge-redacted-bg: #fff1e5; }
    </style>
    """)
    failures = cc.check_file(html)
    assert any("muted" in f and "dark" in f.lower() for f in failures)


def test_check_file_passes_for_fixed_report_html_tokens():
    failures = cc.check_file(REPO_ROOT / "web" / "report.html")
    assert failures == []


def test_check_file_passes_for_fixed_runs_html_tokens():
    failures = cc.check_file(REPO_ROOT / "web" / "runs.html")
    assert failures == []


def test_check_file_passes_for_fixed_session_report_html_tokens():
    failures = cc.check_file(REPO_ROOT / "web" / "session_report.html")
    assert failures == []


def test_check_color_scheme_flags_missing_declaration(tmp_path):
    html = tmp_path / "no_scheme.html"
    html.write_text('<style>:root { --bg: #000; } [data-theme="light"] { --bg: #fff; }</style>')
    problems = cc.check_color_scheme(html.read_text())
    assert problems  # missing color-scheme in both blocks


def test_check_color_scheme_passes_for_fixed_templates():
    for name in ("report.html", "runs.html", "session_report.html"):
        content = (REPO_ROOT / "web" / name).read_text()
        problems = cc.check_color_scheme(content)
        assert problems == [], f"{name}: {problems}"


def test_main_exits_zero_for_the_three_shipped_templates(capsys):
    rc = cc.main(
        [
            str(REPO_ROOT / "web" / "report.html"),
            str(REPO_ROOT / "web" / "runs.html"),
            str(REPO_ROOT / "web" / "session_report.html"),
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "PASS" in out


def test_main_exits_nonzero_on_a_failing_file(tmp_path, capsys):
    html = tmp_path / "bad.html"
    html.write_text("""
    <style>
    :root { --bg: #0d1117; --muted: #1a1e24; --text: #c9d1d9; --surface: #161b22;
            --green: #3fb950; --red: #f85149; --blue: #58a6ff; --orange: #f0883e;
            --purple: #d2a8ff; --yellow: #e3b341;
            --badge-pass-bg: #1a3a1a; --badge-fail-bg: #3a1a1a; --badge-info-bg: #1a2a3a;
            --badge-warn-bg: #3a3010; --badge-redacted-bg: #3a2a10; }
    [data-theme="light"] { --bg: #ffffff; --muted: #59636e; --text: #1f2328; --surface: #f6f8fa;
            --green: #1a7f37; --red: #d1242f; --blue: #0969da; --orange: #bc4c00;
            --purple: #8250df; --yellow: #9a6700;
            --badge-pass-bg: #dafbe1; --badge-fail-bg: #ffebe9; --badge-info-bg: #ddf4ff;
            --badge-warn-bg: #fff8c5; --badge-redacted-bg: #fff1e5; }
    </style>
    """)
    rc = cc.main([str(html)])
    err = capsys.readouterr().err
    assert rc == 1
    assert "FAIL" in err
