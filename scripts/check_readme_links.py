"""Fail if README.md contains a markdown link/image target PyPI can't render.

README.md is used verbatim as the PyPI package `long_description`
(pyproject.toml's `readme = "README.md"`), and PyPI does not rewrite
repo-relative URLs the way GitHub does -- a link like `](docs/demo.png)` or
`](CONTRIBUTING.md)` is a dead image / 404 link on the rendered PyPI page.
Every markdown link/image target in README.md must therefore be either an
absolute URL (`http://` or `https://`) or an in-page anchor (`#section`).

Usage: `uv run python scripts/check_readme_links.py` (wired into
`scripts/e2e.sh`). Exits 1 and prints every offending target if any relative
link/image is found; exits 0 silently otherwise.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Matches both `[text](target)` links and `![alt](target)` images -- the `!`
# prefix is optional and not part of the captured target either way.
_LINK_RE = re.compile(r"\]\(([^)]*)\)")

# A link/image target is fine if it's an absolute http(s) URL or an in-page
# anchor. Everything else (a bare relative path, or an absolute filesystem
# path) is what PyPI can't resolve.
_ALLOWED_PREFIXES = ("http://", "https://", "#")


def find_bad_links(text: str) -> list[str]:
    bad = []
    for target in _LINK_RE.findall(text):
        target = target.strip()
        if not target:
            continue
        # A markdown link target may carry a trailing `"title"` -- only the
        # leading whitespace-delimited URL part matters for this check.
        url = target.split(" ", 1)[0]
        if not url.startswith(_ALLOWED_PREFIXES):
            bad.append(target)
    return bad


def main() -> int:
    readme_path = Path(__file__).resolve().parent.parent / "README.md"
    text = readme_path.read_text(encoding="utf-8")
    bad = find_bad_links(text)
    if bad:
        print(
            f"{readme_path}: found {len(bad)} relative markdown link/image target(s):",
            file=sys.stderr,
        )
        for target in bad:
            print(f"  ]({target})", file=sys.stderr)
        print(
            "\nPyPI renders README.md verbatim as the package long_description and does not "
            "rewrite relative URLs -- every link/image target must be an absolute http(s) URL "
            "or an in-page '#anchor'.",
            file=sys.stderr,
        )
        return 1
    print(f"{readme_path}: all link/image targets are absolute or in-page anchors.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
