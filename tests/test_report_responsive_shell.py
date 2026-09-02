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


# ── scripts/build_tokens.py: byte-identical token/theme-boot blocks ────────
# (v1.0.0 readiness item 46: the token-value assertions above already caught
# a hand-copy divergence in the individual `--var` values -- this section
# proves the FULL block (comments included) is byte-identical across all
# three templates, generated from one source rather than three manually
# kept-in-sync copies.)

import importlib.util  # noqa: E402
import sys as _sys  # noqa: E402

_BT_PATH = REPO_ROOT / "scripts" / "build_tokens.py"


def _load_build_tokens():
    spec = importlib.util.spec_from_file_location("build_tokens", _BT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    _sys.modules["build_tokens"] = module
    spec.loader.exec_module(module)
    return module


bt = _load_build_tokens()


def _anchored_span(content: str, start: str, end: str) -> str:
    match = re.search(re.escape(start) + r".*?" + re.escape(end), content, re.DOTALL)
    assert match, f"anchor span not found (start={start!r}, end={end!r})"
    return match.group(0)


def test_token_block_span_is_byte_identical_across_all_three_templates():
    spans = {
        name: _anchored_span(_read(name), bt.TOKENS_START, bt.TOKENS_END) for name in TEMPLATES
    }
    reference = spans["report.html"]
    for name, span in spans.items():
        assert span == reference, (
            f"{name}: token block text (incl. comments) drifted from report.html"
        )


def test_theme_boot_block_span_is_byte_identical_across_all_three_templates():
    spans = {name: _anchored_span(_read(name), bt.BOOT_START, bt.BOOT_END) for name in TEMPLATES}
    reference = spans["report.html"]
    for name, span in spans.items():
        assert span == reference, f"{name}: theme-boot block text drifted from report.html"


def test_shipped_templates_already_match_the_canonical_generated_blocks():
    """A future edit that updates only one file's token block (even a
    comment-only edit) must make this fail -- i.e. the shipped files are
    exactly what `build_tokens.render()` would (re)generate, byte for byte."""
    for name in TEMPLATES:
        assert bt.check_file(REPO_ROOT / "web" / name), (
            f"{name} has drifted from scripts/build_tokens.py's canonical blocks "
            "-- run `uv run python scripts/build_tokens.py` to regenerate it"
        )


def test_render_is_idempotent():
    content = _read("report.html")
    once = bt.render(content)
    twice = bt.render(once)
    assert once == twice


def test_check_file_detects_a_single_character_drift(tmp_path):
    original = _read("report.html")
    mutated = original.replace("--green: #3fb950;", "--green: #3fb951;", 1)
    assert mutated != original  # sanity: the replace actually landed
    bad = tmp_path / "report.html"
    bad.write_text(mutated, encoding="utf-8")
    assert bt.check_file(bad) is False


def test_main_check_mode_exits_nonzero_on_drift(tmp_path, capsys):
    original = _read("report.html")
    mutated = original.replace("--green: #3fb950;", "--green: #3fb951;", 1)
    bad = tmp_path / "report.html"
    bad.write_text(mutated, encoding="utf-8")
    rc = bt.main(["--check", str(bad)])
    err = capsys.readouterr().err
    assert rc == 1
    assert "FAIL" in err


# ── document semantics: landmarks, heading hierarchy, roving tabindex,
#    skip link (v1.0.0 readiness item 55) ───────────────────────────────────
# Regression tests for the specific defects the item names: both rails were
# `role="navigation"` while holding secondary/primary content (never a
# navigation landmark); there was no `<main>`; the heading levels jumped
# H1 -> H3 with nothing at H2; and the rail-collapse button was a non-tab
# child of `role="tablist"`. Offline/string-based, matching this suite's
# established convention -- no JS runtime/headless browser exists here (see
# this file's own module docstring).

_SKIP_LINK_HREF_RE = re.compile(r'class="skip-link" href="#([\w-]+)"')
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)


def _strip_html_comments(content: str) -> str:
    """Drop `<!-- ... -->` blocks -- this section's own explanatory comments
    quote things like `role="navigation"`/`<main>`/`role="tablist"` in
    prose, which would otherwise false-positive-match the very regexes
    checking that the REAL markup no longer/does contain them."""
    return _HTML_COMMENT_RE.sub("", content)


def test_no_template_uses_role_navigation_for_a_content_panel():
    for name in TEMPLATES:
        content = _strip_html_comments(_read(name))
        assert re.search(r'<[a-zA-Z][^>]*\brole="navigation"', content) is None, (
            f'{name}: role="navigation" is for a set of navigational links, '
            "not a content panel -- use <nav> only for real navigation, "
            '<aside>/role="complementary" for secondary content panels'
        )


def test_every_template_has_exactly_one_main_landmark():
    for name in TEMPLATES:
        content = _strip_html_comments(_read(name))
        # (?<!`) excludes this file's own explanatory comments, which quote
        # the tag in prose as `` `<main>` `` -- a real tag opening is never
        # preceded by a backtick.
        assert len(re.findall(r"(?<!`)<main[ >]", content)) == 1, (
            f"{name}: expected exactly one <main>"
        )


def test_every_template_ships_a_skip_link_targeting_a_real_id_in_the_same_file():
    for name in TEMPLATES:
        content = _read(name)
        match = _SKIP_LINK_HREF_RE.search(content)
        assert match, (
            f'{name}: missing a <a class="skip-link" href="#..."> as the first body element'
        )
        target_id = match.group(1)
        assert f'id="{target_id}"' in content, (
            f"{name}: skip link targets #{target_id}, but no element declares that id"
        )
        # the skip link must be the FIRST thing in <body>, before <header>
        body_start = content.index("<body>")
        header_start = content.index("<header>")
        assert body_start < match.start() < header_start, f"{name}: skip link must precede <header>"


def test_report_html_heading_hierarchy_has_h2_between_h1_and_h3():
    """Regression pin for the specific bug: H1 -> H3 with nothing at H2."""
    content = _read("report.html")
    h1 = content.index("<h1>")
    h2 = content.index("<h2")
    h3 = content.index("<h3")
    assert h1 < h2 < h3, "expected document order h1 -> h2 -> h3, not a skipped level"


def test_report_html_panel_headers_are_h2_not_div():
    content = _read("report.html")
    for label in ("Timeline", "Exchange Detail", "Blame", "Fork Tree", "Cost Profile"):
        assert re.search(rf'<h2 class="panel-header">{re.escape(label)}', content), (
            f'report.html: expected an <h2 class="panel-header"> for {label!r}'
        )
    assert content.count('<h2 class="panel-header"') == 5


def test_report_html_tablist_contains_only_tab_children_not_the_rail_toggle():
    content = _strip_html_comments(_read("report.html"))
    tablist_re = re.compile(
        r'<div[^>]*\brole="tablist"[^>]*>(.*?)</div>\s*<button[^>]*id="rail-right-toggle"',
        re.DOTALL,
    )
    match = tablist_re.search(content)
    assert match, (
        "expected the tablist's own </div> to close immediately before the rail-toggle button"
    )
    tablist_body = match.group(1)
    assert tablist_body.count('role="tab"') == 3
    assert "rail-toggle" not in tablist_body, (
        'the collapse button must not be a child of role="tablist"'
    )


def test_report_html_tablist_has_static_roving_tabindex_markup():
    content = _read("report.html")
    assert 'id="rail-tab-blame"' in content and 'tabindex="0"' in content
    for tab_id in ("rail-tab-forks", "rail-tab-cost"):
        start = content.index(f'id="{tab_id}"')
        tag = content[start : content.index(">", start)]
        assert 'tabindex="-1"' in tag, (
            f"{tab_id}: expected the initially-inactive tab to start at tabindex=-1"
        )


def test_report_html_select_rail_tab_updates_roving_tabindex_in_js():
    content = _read("report.html")
    assert "tab.tabIndex = active ? 0 : -1;" in content


def test_report_html_tablist_keydown_handler_wired_at_boot():
    content = _read("report.html")
    assert "function _onTablistKeydown(e)" in content
    assert "tablist.addEventListener('keydown', _onTablistKeydown)" in content
    for key in ("ArrowRight", "ArrowLeft", "Home", "End"):
        assert f"'{key}'" in content or f'"{key}"' in content


def test_report_html_timeline_is_a_listbox_of_options():
    content = _read("report.html")
    assert 'id="timeline-content" role="listbox"' in content
    assert 'role="option" aria-selected="false" tabindex="-1"' in content


def test_report_html_select_exchange_updates_roving_tabindex_and_aria_selected():
    content = _read("report.html")
    start = content.index("function selectExchange(i)")
    body = content[start : start + 1500]
    assert "setAttribute('aria-selected', 'false')" in body
    assert "setAttribute('aria-selected', 'true')" in body
    assert "row.tabIndex = 0" in body


def test_report_html_timeline_keydown_handler_wired_at_boot():
    content = _read("report.html")
    assert "function _onTimelineKeydown(e)" in content
    assert "timeline.addEventListener('keydown', _onTimelineKeydown)" in content


# ── Content-Security-Policy (v1.0.0 readiness item 42) ──────────────────────
# All three templates ship a restrictive CSP meta tag (server.py additionally
# sends the byte-identical policy as a response header for the two it
# serves live -- see tests/test_server.py). Not build_tokens.py-managed
# (out of that item's stated scope: the 29-line token block + theme-boot
# IIFE) -- hand-copied like the responsive breakpoints above, so this
# section proves the three copies stay identical the same way
# test_breakpoint_values_are_shared_across_all_three_templates does.

_CSP_META_RE = re.compile(r'<meta http-equiv="Content-Security-Policy" content="([^"]*)">')


def _csp_content(name: str) -> str:
    match = _CSP_META_RE.search(_read(name))
    assert match, f'{name}: missing <meta http-equiv="Content-Security-Policy">'
    return match.group(1)


def test_every_template_ships_a_csp_meta_tag_in_head():
    for name in TEMPLATES:
        content = _read(name)
        head_end = content.index("</head>")
        match = _CSP_META_RE.search(content)
        assert match, f"{name}: missing CSP meta tag"
        assert match.start() < head_end, f"{name}: CSP meta tag must be inside <head>"


def test_csp_meta_content_is_byte_identical_across_all_three_templates():
    contents = {name: _csp_content(name) for name in TEMPLATES}
    reference = contents["report.html"]
    for name, content in contents.items():
        assert content == reference, f"{name}: CSP content drifted from report.html"


def test_csp_disallows_every_external_origin():
    content = _csp_content("report.html")
    for directive in ("script-src", "style-src", "img-src", "connect-src"):
        assert directive in content, f"CSP is missing a {directive} directive"
    assert "http://" not in content
    assert "https://" not in content
    assert " *" not in content and content.strip() != "*"
    # only 'self'/'unsafe-inline'/data: keyword sources -- no bare hostname
    for directive_body in re.findall(r"(?:script|style|img|connect)-src ([^;]+)", content):
        for source in directive_body.split():
            assert source in ("'self'", "'unsafe-inline'", "data:"), (
                f"unexpected external-looking CSP source: {source!r}"
            )


def test_report_html_csp_meta_content_matches_server_py_constant_minus_frame_ancestors():
    """The meta tag and server.py's CONTENT_SECURITY_POLICY header must stay
    in lockstep for every directive EXCEPT frame-ancestors, which browsers
    ignore when delivered via <meta> (a documented CSP spec limitation, not
    a gap in this fix) -- so the header carries it and the meta tag
    deliberately omits it rather than shipping an inert, confusing
    directive (and a real Chromium console error) for no protection."""
    from tracefork.server import CONTENT_SECURITY_POLICY

    meta_content = _csp_content("report.html")
    assert "frame-ancestors" not in meta_content
    assert "frame-ancestors 'none'" in CONTENT_SECURITY_POLICY
    header_minus_frame_ancestors = CONTENT_SECURITY_POLICY.replace("; frame-ancestors 'none'", "")
    assert meta_content == header_minus_frame_ancestors


def test_main_check_mode_exits_zero_for_the_three_shipped_templates(capsys):
    rc = bt.main(
        [
            "--check",
            str(REPO_ROOT / "web" / "report.html"),
            str(REPO_ROOT / "web" / "runs.html"),
            str(REPO_ROOT / "web" / "session_report.html"),
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "PASS" in out
