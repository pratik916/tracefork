#!/usr/bin/env python3
"""scripts/build_tokens.py — single source of truth for the design-token CSS
block + theme-boot IIFE shared by web/report.html / web/runs.html /
web/session_report.html.

Before this script existed, both blocks were hand-copied verbatim into all
three templates with no build step and no test guarding drift -- a token
edit in one file silently diverged from the other two unless someone
remembered to copy it three times by hand (and in practice already had:
report.html/runs.html/session_report.html each carried slightly different
prose in the SAME comments, even though the `--token: value;` declarations
themselves matched). `TOKENS_BLOCK` (the box-model reset + `:root`'s dark
tokens + `[data-theme="light"]`'s overrides + `:focus-visible`) and
`THEME_BOOT_BLOCK` (the zero-FOUC theme-boot `<script>`) below are now the
ONE place either block's text lives; `render()` stamps both into a
template's existing HTML, replacing whatever currently sits between the
same two anchor lines every template already ships.

Usage:
    uv run python scripts/build_tokens.py            # regenerate the three
                                                       # shipped templates in place
    uv run python scripts/build_tokens.py --check     # verify only (CI-safe):
                                                       # exit 1 if any file has drifted
    uv run python scripts/build_tokens.py --check FILE [FILE ...]   # check specific files

Offline/$0 -- pure local file text substitution, no network, no external
dependency (stdlib `re` + `pathlib` only, mirroring check_contrast.py's
"no new dependency" approach).
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

# ── canonical token block ────────────────────────────────────────────────
# Anchored by two exact lines every template already has once, right after
# its own `<style>` tag (the box-model reset) and right after the
# `[data-theme="light"]` override block (`:focus-visible`) -- so the whole
# span between them (dark `:root` tokens + light-theme overrides) gets
# replaced wholesale with this text, comments included.
TOKENS_START = "  * { box-sizing: border-box; margin: 0; padding: 0; }"
TOKENS_END = "  :focus-visible { outline: var(--focus-ring); outline-offset: 1px; }"

TOKENS_BLOCK = """\
  * { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    /* color tokens — dark theme (default, GitHub-dark) */
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #c9d1d9; --muted: #838b95; --green: #3fb950;
    --blue: #58a6ff; --orange: #f0883e; --purple: #d2a8ff;
    --red: #f85149; --yellow: #e3b341;
    /* semantic badge backgrounds (tags, blame/shapley/evidence badges, notes) */
    --badge-pass-bg: #1a3a1a; --badge-fail-bg: #3a1a1a;
    --badge-info-bg: #1a2a3a; --badge-warn-bg: #3a3010;
    --badge-redacted-bg: #3a2a10;
    /* type tokens — --font-ui for chrome, --font-mono for code/data */
    --font-ui: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans", Helvetica, Arial,
      sans-serif;
    --font-mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
    --text-xs: 11px; --text-sm: 12px; --text-md: 13px; --text-lg: 15px;
    /* spacing scale, radius, focus ring */
    --sp-1: 4px; --sp-2: 8px; --sp-3: 12px; --sp-4: 16px; --sp-5: 20px; --sp-6: 24px;
    --radius: 6px;
    --focus-ring: 2px solid var(--blue);
    /* tells the UA to render native controls (range sliders, scrollbars,
       form-field chrome) in dark UI, matching this theme -- overridden to
       'light' below when [data-theme="light"] is active */
    color-scheme: dark;
  }
  [data-theme="light"] {
    /* light theme (GitHub-light-inspired) — stamped on <html> by the theme boot */
    --bg: #ffffff; --surface: #f6f8fa; --border: #d0d7de;
    --text: #1f2328; --muted: #59636e; --green: #1a7f37;
    --blue: #0969da; --orange: #bc4c00; --purple: #8250df;
    --red: #d1242f; --yellow: #9a6700;
    --badge-pass-bg: #dafbe1; --badge-fail-bg: #ffebe9;
    --badge-info-bg: #ddf4ff; --badge-warn-bg: #fff8c5;
    --badge-redacted-bg: #fff1e5;
    color-scheme: light;
  }
  :focus-visible { outline: var(--focus-ring); outline-offset: 1px; }"""

# ── canonical theme-boot IIFE ────────────────────────────────────────────
# Already byte-identical across all three templates today -- included here
# too so future drift is caught the same way, not just token drift.
BOOT_START = "<!-- TF:theme-boot -->"
BOOT_END = "</script>"

THEME_BOOT_BLOCK = """\
<!-- TF:theme-boot -->
<script>
// Zero-FOUC theme boot: resolve the persisted theme (localStorage
// "tracefork-theme", JSON-encoded by the toggle) — or, on first visit,
// prefers-color-scheme — and stamp it on <html data-theme> before first
// paint. Default dark. Never throws (file:// / privacy modes can make
// localStorage or matchMedia unavailable).
(function () {
  var theme = 'dark';
  try {
    var raw = localStorage.getItem('tracefork-theme');
    if (raw !== null) {
      try { theme = JSON.parse(raw); } catch (e) { theme = raw; }
    } else if (window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches) {
      theme = 'light';
    }
  } catch (e) {}
  if (theme !== 'light' && theme !== 'dark') theme = 'dark';
  document.documentElement.dataset.theme = theme;
})();
</script>"""


def _replace_span(html: str, start: str, end: str, block: str, label: str) -> str:
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.DOTALL)
    if not pattern.search(html):
        raise ValueError(f"{label}: anchor span not found (start={start!r}, end={end!r})")
    # A function replacement (not a string one) so `block`'s own content is
    # inserted literally -- `re.sub` never applies backreference/escape
    # processing to a callable's return value.
    return pattern.sub(lambda _m: block, html, count=1)


def apply_tokens(html: str) -> str:
    """Replace whatever sits between the reset rule and `:focus-visible`
    with the canonical `TOKENS_BLOCK`."""
    return _replace_span(html, TOKENS_START, TOKENS_END, TOKENS_BLOCK, "tokens block")


def apply_theme_boot(html: str) -> str:
    """Replace the `<!-- TF:theme-boot --> ... </script>` span with the
    canonical `THEME_BOOT_BLOCK`."""
    return _replace_span(html, BOOT_START, BOOT_END, THEME_BOOT_BLOCK, "theme-boot block")


def render(html: str) -> str:
    """Stamp both canonical blocks into `html`. Idempotent: `render(render(x)) == render(x)`."""
    return apply_theme_boot(apply_tokens(html))


def check_file(path: Path) -> bool:
    """True if `path` already carries both canonical blocks byte-for-byte."""
    current = path.read_text(encoding="utf-8")
    return render(current) == current


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="HTML files to process (default: the three shipped web/ templates)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify only, write nothing; exit 1 if any file has drifted from the canonical blocks",
    )
    args = parser.parse_args(argv)
    files: list[Path] = args.files or list(DEFAULT_FILES)

    if args.check:
        drifted = [f for f in files if not check_file(f)]
        if drifted:
            print(
                "build_tokens: FAIL (drifted from the canonical token/theme-boot blocks)",
                file=sys.stderr,
            )
            for f in drifted:
                print(f"  - {f}", file=sys.stderr)
            return 1
        print(f"build_tokens: PASS ({len(files)} file(s) match the canonical blocks)")
        return 0

    changed: list[Path] = []
    for f in files:
        current = f.read_text(encoding="utf-8")
        updated = render(current)
        if updated != current:
            f.write_text(updated, encoding="utf-8")
            changed.append(f)
    if changed:
        print(f"build_tokens: wrote {len(changed)} file(s): " + ", ".join(str(c) for c in changed))
    else:
        print(f"build_tokens: {len(files)} file(s) already up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
