#!/usr/bin/env python3
"""scripts/check_contrast.py — WCAG AA contrast auditor for web/*.html's
shared design-token blocks (v1.0.0 workbench-shell readiness).

`web/report.html` / `web/runs.html` / `web/session_report.html` each define
a dark-theme `:root { --token: #hex; ... }` block plus a
`[data-theme="light"] { ... }` override block (see each file's `TF:tokens`
comment). This script parses those two blocks straight out of the HTML
source (no CSS engine, matching this repo's existing offline/no-JS-runtime
test discipline — see tests/test_report_scrubber.py's module docstring),
computes the real WCAG 2.x relative-luminance contrast ratio for every
foreground/background token pair the templates actually render text with,
and fails loudly (exit 1) if any pair is below the 4.5:1 "AA, normal text"
threshold in either theme — a claim of AA compliance is checked here, never
just asserted in a comment.

It also checks that both theme blocks declare a `color-scheme` value (dark
for `:root`, light for `[data-theme="light"]`), so native form controls
(range sliders, scrollbars) render in the matching theme's UA chrome
instead of a mismatched default.

Usage:
    uv run python scripts/check_contrast.py [FILE ...]

With no arguments, checks the three shipped templates under `web/`.
Offline/$0 — pure local file parsing and arithmetic, no network, no
external dependency (stdlib `re` + `pathlib` only, mirroring
check_executed_evidence.py's "no new dependency" approach).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FILES = (
    REPO_ROOT / "web" / "report.html",
    REPO_ROOT / "web" / "runs.html",
    REPO_ROOT / "web" / "session_report.html",
)

WCAG_AA_NORMAL_TEXT = 4.5

# (foreground token, background token, human label) — every fg-on-bg text
# combination the three templates actually render, derived from their CSS
# (`color: var(--X)` / `fill: var(--X)` paired with the element's or its
# container's `background`). Kept as one explicit list (not "every possible
# pair") so a failure names a real rendered combination, not a spurious
# pairing nothing ever uses.
CHECK_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("text", "bg", "body text on page background"),
    ("text", "surface", "body text on panel/header surface"),
    ("muted", "bg", "muted/secondary text on page background"),
    ("muted", "surface", "muted/secondary text on panel surface"),
    ("green", "badge-pass-bg", "pass badge text"),
    ("red", "badge-fail-bg", "fail badge text"),
    ("blue", "badge-info-bg", "info badge text"),
    ("yellow", "badge-warn-bg", "warn badge text"),
    ("orange", "badge-redacted-bg", "redacted badge text"),
    ("blue", "bg", "link/accent text on page background"),
    ("purple", "bg", "assistant-role text on page background"),
    ("orange", "bg", "tool-role text on page background"),
    ("red", "bg", "error/red text on page background"),
    ("green", "bg", "success/green text on page background"),
    ("yellow", "bg", "warning/yellow text on page background"),
)


def _srgb_to_linear(channel_0_255: int) -> float:
    c = channel_0_255 / 255.0
    return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4


def relative_luminance(hex_color: str) -> float:
    """WCAG relative luminance of a `#rrggbb` color, in [0, 1]."""
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(ch * 2 for ch in h)
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return 0.2126 * _srgb_to_linear(r) + 0.7152 * _srgb_to_linear(g) + 0.0722 * _srgb_to_linear(b)


def contrast_ratio(hex_a: str, hex_b: str) -> float:
    """WCAG contrast ratio between two `#rrggbb` colors (>= 1.0, symmetric)."""
    la, lb = relative_luminance(hex_a), relative_luminance(hex_b)
    lighter, darker = max(la, lb), min(la, lb)
    return (lighter + 0.05) / (darker + 0.05)


_VAR_RE = re.compile(r"--([\w-]+)\s*:\s*([^;]+);")


def _extract_block(css_text: str, selector: str) -> str:
    """Return the raw `{ ... }` body text following a literal `selector`.

    Non-greedy up to the first `}` — correct for this repo's token blocks,
    which are flat `--name: value;` declarations with no nested braces.
    Returns `""` if `selector` isn't found (an empty/missing block, not an
    error — callers treat a missing block as "no tokens/no color-scheme
    declared here").
    """
    match = re.search(re.escape(selector) + r"\s*\{(.*?)\}", css_text, re.DOTALL)
    return match.group(1) if match else ""


def extract_css_variables(css_text: str, selector: str) -> dict[str, str]:
    """Parse every `--name: value;` custom property out of `selector`'s block."""
    block = _extract_block(css_text, selector)
    return {name: value.strip() for name, value in _VAR_RE.findall(block)}


def resolve_theme_tokens(css_text: str) -> tuple[dict[str, str], dict[str, str]]:
    """Return `(dark_tokens, light_tokens)`.

    `dark_tokens` is exactly `:root`'s own declarations (dark is this
    project's default theme, per every template's own `TF:tokens` comment).
    `light_tokens` layers `[data-theme="light"]`'s overrides on top of a
    copy of `dark_tokens`, mirroring the real cascade: a light-theme token
    the override block doesn't redeclare stays inherited from `:root`.
    """
    dark_tokens = extract_css_variables(css_text, ":root")
    light_overrides = extract_css_variables(css_text, '[data-theme="light"]')
    light_tokens = {**dark_tokens, **light_overrides}
    return dark_tokens, light_tokens


def _is_hex_color(value: str) -> bool:
    return bool(re.fullmatch(r"#[0-9a-fA-F]{3}|#[0-9a-fA-F]{6}", value))


def check_tokens(tokens: dict[str, str], theme_label: str, source_label: str) -> list[str]:
    """Check every `CHECK_PAIRS` entry against `tokens`; return failure lines."""
    failures: list[str] = []
    for fg_name, bg_name, description in CHECK_PAIRS:
        fg, bg = tokens.get(fg_name), tokens.get(bg_name)
        if fg is None or bg is None or not _is_hex_color(fg) or not _is_hex_color(bg):
            # A token this pair needs isn't declared (or isn't a plain hex
            # color) in this file/theme -- nothing to check here, not a
            # contrast failure (e.g. runs.html has no --purple usage today).
            continue
        ratio = contrast_ratio(fg, bg)
        if ratio < WCAG_AA_NORMAL_TEXT:
            failures.append(
                f"{source_label} [{theme_label}]: {description} "
                f"(--{fg_name} {fg} on --{bg_name} {bg}) = {ratio:.2f}:1, "
                f"below AA {WCAG_AA_NORMAL_TEXT}:1"
            )
    return failures


_COLOR_SCHEME_RE = re.compile(r"color-scheme\s*:\s*([a-z][a-z \t]*[a-z])\s*;?", re.IGNORECASE)


def check_color_scheme(css_text: str) -> list[str]:
    """Verify `:root` declares a dark `color-scheme` and the light override
    block declares a light one — so native form controls (e.g. the
    scrubber's `<input type="range">`) render in the matching theme's UA
    chrome rather than defaulting to light regardless of the active theme.
    Returns a list of human-readable problems (empty means OK).
    """
    problems: list[str] = []
    root_block = _extract_block(css_text, ":root")
    light_block = _extract_block(css_text, '[data-theme="light"]')

    root_match = _COLOR_SCHEME_RE.search(root_block)
    if root_match is None:
        problems.append(":root does not declare color-scheme (expected to name 'dark')")
    elif "dark" not in root_match.group(1).lower():
        problems.append(
            f":root color-scheme is '{root_match.group(1).strip()}', expected to include 'dark'"
        )

    light_match = _COLOR_SCHEME_RE.search(light_block)
    if light_match is None:
        problems.append(
            "[data-theme=\"light\"] does not declare color-scheme (expected to name 'light')"
        )
    elif "light" not in light_match.group(1).lower():
        problems.append(
            f"[data-theme=\"light\"] color-scheme is '{light_match.group(1).strip()}', "
            "expected to include 'light'"
        )
    return problems


def check_file(path: Path) -> list[str]:
    """Run the full contrast audit against one HTML file. Returns failure
    lines (empty means every checked pair clears AA in both themes)."""
    css_text = path.read_text(encoding="utf-8")
    dark_tokens, light_tokens = resolve_theme_tokens(css_text)
    failures = check_tokens(dark_tokens, "dark", path.name)
    failures += check_tokens(light_tokens, "light", path.name)
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="HTML files to audit (default: the three shipped web/ templates)",
    )
    args = parser.parse_args(argv)
    files: list[Path] = args.files or list(DEFAULT_FILES)

    all_failures: list[str] = []
    for path in files:
        if not path.is_file():
            all_failures.append(f"{path}: file not found")
            continue
        all_failures += check_file(path)
        scheme_problems = check_color_scheme(path.read_text(encoding="utf-8"))
        all_failures += [f"{path.name}: {p}" for p in scheme_problems]

    if all_failures:
        print("check_contrast: FAIL", file=sys.stderr)
        for failure in all_failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    n_pairs = len(CHECK_PAIRS)
    print(f"check_contrast: PASS ({len(files)} file(s), {n_pairs} pair(s) x 2 themes each)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
