"""v1.0.0 readiness item 21: extend commit 64c6480's design-token/dual-theme
workbench-shell foundation with the pieces it started but never finished --
responsive breakpoints, WCAG AA contrast in both themes (see
scripts/check_contrast.py + tests/test_check_contrast.py for the contrast
proof itself), correct `color-scheme`, and consistent hand-copied token
blocks across web/report.html / web/runs.html / web/session_report.html.

Offline, string-assertion tests reading the templates directly (no JS
runtime/headless browser -- matches test_runs_page.py's
`test_runs_page_html_references_api_runs_and_run_query_param_link`, which
already reads `web/runs.html` straight off disk for structural assertions
rather than through the generator, since none of this is data-dependent).
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATES = ("report.html", "runs.html", "session_report.html")


def _read(name: str) -> str:
    return (REPO_ROOT / "web" / name).read_text()


def _root_block(content: str) -> str:
    match = re.search(r":root\s*\{(.*?)\}", content, re.DOTALL)
    assert match, "no :root block found"
    return match.group(1)


def _light_block(content: str) -> str:
    match = re.search(r'\[data-theme="light"\]\s*\{(.*?)\}', content, re.DOTALL)
    assert match, 'no [data-theme="light"] block found'
    return match.group(1)


def _var_map(block: str) -> dict[str, str]:
    return dict(re.findall(r"--([\w-]+)\s*:\s*([^;]+);", block))


# ── responsive breakpoints ──────────────────────────────────────────────────


def test_every_template_declares_at_least_one_media_query():
    for name in TEMPLATES:
        content = _read(name)
        assert "@media" in content, f"{name} has no @media rule"


def test_report_html_workbench_shell_reflows_to_single_column_below_tablet_width():
    content = _read("report.html")
    media = re.search(r"@media \(max-width: 860px\)\s*\{(.*?)\n  \}\n", content, re.DOTALL)
    assert media, "expected an 860px tablet breakpoint in report.html"
    body = media.group(1)
    assert "grid-template-columns: 1fr" in body


def test_breakpoint_values_are_shared_across_all_three_templates():
    """The three templates hand-copy their responsive rules the same way
    they hand-copy their token blocks -- keep the actual breakpoint widths
    identical so they reflow at consistent points, even though each
    template's rule bodies differ (they have different layouts)."""
    breakpoint_re = re.compile(r"@media \(max-width: (\d+)px\)")
    breakpoints_seen = {name: set(breakpoint_re.findall(_read(name))) for name in TEMPLATES}
    all_breakpoints = set().union(*breakpoints_seen.values())
    for name, seen in breakpoints_seen.items():
        assert seen == all_breakpoints, (
            f"{name} uses breakpoints {seen}, expected {all_breakpoints}"
        )


def test_runs_html_table_scrolls_in_its_own_container_not_the_page():
    content = _read("runs.html")
    assert "table-wrap" in content
    assert re.search(r"\.table-wrap\s*\{[^}]*overflow-x:\s*auto", content)
    # the render function must actually wrap the table in that container
    assert '<div class="table-wrap"><table>' in content


# ── color-scheme ────────────────────────────────────────────────────────────


def test_every_template_declares_dark_color_scheme_on_root():
    for name in TEMPLATES:
        root = _root_block(_read(name))
        assert re.search(r"color-scheme\s*:\s*dark\s*;", root), (
            f"{name}: :root missing color-scheme: dark"
        )


def test_every_template_declares_light_color_scheme_override():
    for name in TEMPLATES:
        light = _light_block(_read(name))
        assert re.search(r"color-scheme\s*:\s*light\s*;", light), (
            f'{name}: [data-theme="light"] missing color-scheme: light'
        )


# ── token consistency across the three hand-copied blocks ──────────────────


def test_root_color_tokens_are_identical_across_all_three_templates():
    color_vars = (
        "bg",
        "surface",
        "border",
        "text",
        "muted",
        "green",
        "blue",
        "orange",
        "purple",
        "red",
        "yellow",
    )
    maps = {name: _var_map(_root_block(_read(name))) for name in TEMPLATES}
    reference = maps["report.html"]
    for name, tokens in maps.items():
        for var in color_vars:
            assert var in tokens, f"{name}: :root missing --{var}"
            assert tokens[var] == reference[var], (
                f"{name}: --{var} = {tokens[var]!r} drifted from report.html's {reference[var]!r}"
            )


def test_light_theme_color_tokens_are_identical_across_all_three_templates():
    color_vars = (
        "bg",
        "surface",
        "border",
        "text",
        "muted",
        "green",
        "blue",
        "orange",
        "purple",
        "red",
        "yellow",
    )
    maps = {name: _var_map(_light_block(_read(name))) for name in TEMPLATES}
    reference = maps["report.html"]
    for name, tokens in maps.items():
        for var in color_vars:
            assert var in tokens, f'{name}: [data-theme="light"] missing --{var}'
            assert tokens[var] == reference[var], (
                f"{name}: light --{var} = {tokens[var]!r} drifted from "
                f"report.html's {reference[var]!r}"
            )


def test_dark_muted_token_meets_aa_contrast_value():
    # Regression pin for the specific fix: #6e7681 (4.12:1 on --bg, 3.77:1
    # on --surface) failed AA normal-text 4.5:1 in the dark theme.
    for name in TEMPLATES:
        tokens = _var_map(_root_block(_read(name)))
        assert tokens["muted"] != "#6e7681", (
            f"{name}: still using the pre-fix low-contrast dark --muted"
        )


# ── no near-invisible text left over from a border-color-as-text mistake ──


def test_session_report_missing_lane_cell_does_not_use_border_as_text_color():
    content = _read("session_report.html")
    assert "color: var(--border)" not in content
