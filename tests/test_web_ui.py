"""v1.0.0 readiness item 30: ~45,000 characters of inline JavaScript across
the three report templates, and not one test parsed or executed any of it
before this file existed -- the only checks were string/JSON matches
against the injected data payload. A real commit once changed 454 lines
across exactly these three files, introducing the Timeline data-i selector
collision (tracefork-bge.10/tracefork-sis.10), and the full suite stayed
green. This file adds the concrete checks that would have caught it:

1. `web/report.html` is well-formed HTML (a Python `html.parser` tag-balance
   check -- this suite has no headless browser, matching every other web/*
   test's established convention) AND every literal id it looks up via
   `getElementById`/`querySelector(All)` resolves to a real `id="..."` in
   the same file.
2. A data-contract check: every top-level key the JS dereferences off the
   injected tape data (`DATA.x`/`data.x`) is present in BOTH
   `report._tape_to_data`'s real output and `server.py`'s real
   `GET /api/run/{id}` response -- so a renamed/dropped key on either side
   is caught immediately, not discovered by a blank panel in production.
3. A `node --check` syntax gate over the extracted `<script>` blocks --
   CI-only when `node` is present (skipped, not failed, otherwise; see the
   item's own proposed design), demonstrated to actually catch a stray
   brace by mutating a copy of the file and asserting `node --check` then
   fails.
4. `escape(0)`/`escape(null)` must return strings, not throw -- proven by
   actually EXECUTING the extracted `escape()` function in real Node (not
   just a structural string check) when `node` is available.
5. All three templates share ONE `escape()` implementation (byte-identical
   function bodies), and report.html's one previously-unescaped
   `${e.message}` sink (unlike its three sibling `escape(e.message)` sinks)
   is fixed.
"""

from __future__ import annotations

import html.parser
import json
import re
import shutil
import subprocess
from pathlib import Path

import anthropic
import httpx
import pytest
from fastapi.testclient import TestClient

from tests.fakes import ScriptedFakeLLM, make_text_response
from tracefork.report import _tape_to_data
from tracefork.server import app as fastapi_app
from tracefork.server import init_store
from tracefork.store import TapeStore
from tracefork.tape import Tape
from tracefork.transport import TraceforkTransport

REPO_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = REPO_ROOT / "web"
TEMPLATES = ("report.html", "runs.html", "session_report.html")

_NODE = shutil.which("node")


def _read(name: str) -> str:
    return (WEB_DIR / name).read_text(encoding="utf-8")


def _last_script_block(content: str) -> str:
    """The main app `<script>` block -- the last of the two `<script>` tags
    every template ships (the first is the zero-FOUC theme-boot IIFE)."""
    blocks = re.findall(r"<script>(.*?)</script>", content, re.DOTALL)
    assert blocks, "no <script> block found"
    return blocks[-1]


# ── 1a. well-formedness ─────────────────────────────────────────────────────

_VOID_TAGS = {
    "meta",
    "link",
    "img",
    "br",
    "input",
    "hr",
    "area",
    "base",
    "col",
    "embed",
    "source",
    "track",
    "wbr",
}


class _TagBalanceChecker(html.parser.HTMLParser):
    """Minimal well-formedness check: every non-void start tag has a
    matching end tag in the right place. `html.parser` already treats
    `<script>`/`<style>` content as opaque CDATA (never parses JS/CSS text
    as markup), matching real browser behaviour."""

    def __init__(self) -> None:
        super().__init__()
        self.stack: list[str] = []
        self.errors: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001
        if tag in _VOID_TAGS:
            return
        self.stack.append(tag)

    def handle_endtag(self, tag: str) -> None:
        if not self.stack:
            self.errors.append(f"unexpected closing </{tag}> with nothing open")
            return
        if self.stack[-1] != tag:
            self.errors.append(f"mismatch: expected </{self.stack[-1]}>, got </{tag}>")
            # recover so one mismatch doesn't cascade into every later tag
            if tag in self.stack:
                while self.stack and self.stack[-1] != tag:
                    self.stack.pop()
                if self.stack:
                    self.stack.pop()
        else:
            self.stack.pop()


def _well_formedness_errors(content: str) -> list[str]:
    checker = _TagBalanceChecker()
    checker.feed(content)
    errors = list(checker.errors)
    if checker.stack:
        errors.append(f"unclosed tag(s) at end of document: {checker.stack}")
    return errors


def test_all_three_templates_are_well_formed_html():
    for name in TEMPLATES:
        errors = _well_formedness_errors(_read(name))
        assert errors == [], f"{name}: {errors}"


# ── 1b. every literal id lookup resolves to a real id ───────────────────────

_GET_ELEMENT_BY_ID_RE = re.compile(r"getElementById\((['\"`])((?:(?!\1).)*)\1\)")
_QUERY_SELECTOR_RE = re.compile(r"querySelectorAll?\((['\"`])((?:(?!\1).)*)\1\)")
_HASH_TOKEN_RE = re.compile(r"#([\w-]+(?:\$\{[^}]*\})?)")
_ID_ATTR_RE = re.compile(r"""id=(["'])((?:(?!\1).)*)\1""")


def _declared_ids(content: str) -> set[str]:
    """Every literal `id="..."`/`id='...'` attribute value -- including a
    JS-template-literal one like `id="fork-edit-${i}"`, captured as its
    exact raw text, since the matching lookup (`` `fork-edit-${i}` ``) uses
    the byte-identical string."""
    return {m.group(2) for m in _ID_ATTR_RE.finditer(content)}


def _referenced_ids(content: str) -> list[str]:
    """Every id this file's JS looks up via a LITERAL string/template-literal
    argument: the whole `getElementById(...)` argument, or a bare `#token`
    inside a `querySelector(All)` argument (so a compound selector like
    `#rail-right .rail-tab` or `` `#timeline-content [data-i="${i}"]` ``
    still yields its `#rail-right`/`#timeline-content` id reference).
    String-CONCATENATION lookups (e.g. `'rail-' + side + '-toggle'`) don't
    match either regex (there's no single quote/backtick spanning the whole
    argument) and are intentionally out of scope -- there's no static
    string to check them against."""
    referenced = [m.group(2) for m in _GET_ELEMENT_BY_ID_RE.finditer(content)]
    for m in _QUERY_SELECTOR_RE.finditer(content):
        referenced.extend(_HASH_TOKEN_RE.findall(m.group(2)))
    return referenced


def test_report_html_every_literal_id_lookup_resolves_to_a_real_id():
    content = _read("report.html")
    declared = _declared_ids(content)
    referenced = _referenced_ids(content)
    assert referenced, "expected to find at least one getElementById/querySelector call"
    missing = sorted({r for r in referenced if r not in declared})
    assert missing == [], f"report.html: these ids are looked up but never declared: {missing}"


def test_renaming_a_referenced_id_in_report_html_is_caught(tmp_path):
    """Proves the checker above isn't vacuous: renaming a REAL, referenced
    id (leaving its lookup untouched) must make the check fail."""
    content = _read("report.html")
    assert 'id="timeline-content"' in content
    mutated = content.replace('id="timeline-content"', 'id="timeline-content-RENAMED"', 1)
    declared = _declared_ids(mutated)
    referenced = _referenced_ids(mutated)
    missing = {r for r in referenced if r not in declared}
    assert "timeline-content" in missing


# ── 3. node --check syntax gate (CI-only when node is present) ─────────────


def _node_check(
    js_text: str, tmp_path: Path, name: str = "script.js"
) -> subprocess.CompletedProcess:
    path = tmp_path / name
    path.write_text(js_text, encoding="utf-8")
    return subprocess.run(
        [_NODE, "--check", str(path)],
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_extracted_script_blocks_have_no_syntax_errors(tmp_path):
    if _NODE is None:
        pytest.skip("node not installed -- this check is CI-only when node is present")
    for name in TEMPLATES:
        result = _node_check(_last_script_block(_read(name)), tmp_path, f"{name}.js")
        assert result.returncode == 0, f"{name}: node --check failed:\n{result.stderr}"


def test_a_stray_brace_in_report_html_is_caught_by_node_check(tmp_path):
    """Proves the syntax gate isn't vacuous: a stray `{` in the inline
    script must make `node --check` fail (this is exactly the class of bug
    that mutated report.html's own scrubber-tick collision fix once slipped
    through with the full suite green -- see this module's docstring)."""
    if _NODE is None:
        pytest.skip("node not installed -- this check is CI-only when node is present")
    script = _last_script_block(_read("report.html"))
    mutated = script.replace("function escape(s)", "function escape(s) {{{", 1)
    result = _node_check(mutated, tmp_path, "mutated.js")
    assert result.returncode != 0


# ── 4. escape(0)/escape(null) return strings, never throw ──────────────────

_ESCAPE_FN_RE = re.compile(r"function escape\(s\) \{.*?\}", re.DOTALL)


def _escape_fn_source(content: str) -> str:
    match = _ESCAPE_FN_RE.search(content)
    assert match, "no `function escape(s) { ... }` found"
    return match.group(0)


def test_escape_handles_zero_and_null_without_throwing_real_node_execution(tmp_path):
    """Executes the ACTUAL extracted escape() in real Node -- not just a
    structural string check -- since only real execution can prove a
    runtime claim like 'does not throw'."""
    if _NODE is None:
        pytest.skip("node not installed -- this check is CI-only when node is present")
    fn_source = _escape_fn_source(_read("report.html"))
    harness = (
        fn_source
        + "\nconsole.log(JSON.stringify(["
        + "typeof escape(0), escape(0), typeof escape(null), escape(null),"
        + "typeof escape(undefined), escape(undefined)"
        + "]));\n"
    )
    path = tmp_path / "escape_harness.js"
    path.write_text(harness, encoding="utf-8")
    result = subprocess.run([_NODE, str(path)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, f"escape() threw in Node:\n{result.stderr}"
    typeof_zero, escaped_zero, typeof_null, escaped_null, typeof_undef, escaped_undef = json.loads(
        result.stdout
    )
    assert typeof_zero == "string" and escaped_zero == "0"
    assert typeof_null == "string" and escaped_null == "null"
    assert typeof_undef == "string" and escaped_undef == "undefined"


def test_escape_source_coerces_with_string_before_replace():
    """A lighter-weight, ALWAYS-runs (no node dependency) regression pin for
    the same fix: `String(s)` must run before the first `.replace` call, in
    every one of the three templates."""
    for name in TEMPLATES:
        fn_source = _escape_fn_source(_read(name))
        assert "String(s)" in fn_source, f"{name}: escape() is missing the String(s) coercion"
        # coercion must happen BEFORE the first .replace, not after
        assert fn_source.index("String(s)") < fn_source.index(".replace")


# ── 5. one escape() implementation; the e.message sink is fixed ────────────


def test_all_three_templates_share_one_escape_implementation():
    sources = {name: _escape_fn_source(_read(name)) for name in TEMPLATES}
    reference = sources["report.html"]
    for name, source in sources.items():
        assert source == reference, f"{name}: escape() drifted from report.html's implementation"


def test_report_html_error_message_sinks_are_all_escaped():
    content = _read("report.html")
    assert "${e.message}</div>`;" not in content
    assert "${e.message}</span>`;" not in content
    # every `.innerHTML = ...e.message...` sink goes through escape()
    for match in re.finditer(r"\.innerHTML = [^;]*\$\{e\.message\}[^;]*;", content):
        assert "escape(e.message)" in match.group(0), match.group(0)


# ── 2. data contract: JS-dereferenced keys vs. the real Python producers ───

_DATA_KEY_RE = re.compile(r"\b(?:DATA|data)\.([a-zA-Z_][a-zA-Z0-9_]*)")
# `data.est_usd` (in getForkEstimate) is a DIFFERENT, shadowing local
# variable -- the fork-cost-estimate POST response
# (`{agent_name, step, est_usd}`, see server.py's `estimate_fork`), not the
# tape data object every other `data.x`/`DATA.x` reference in this file is.
_SHADOWED_LOCAL_KEYS = {"est_usd"}


def _js_dereferenced_top_level_keys(content: str) -> set[str]:
    keys = {m.group(1) for m in _DATA_KEY_RE.finditer(content)}
    return keys - _SHADOWED_LOCAL_KEYS


def test_report_html_dereferences_at_least_the_documented_data_keys():
    """Sanity check on the extraction regex itself: it must find the well-known
    keys report.html's render functions are known to use."""
    keys = _js_dereferenced_top_level_keys(_read("report.html"))
    for expected in ("exchanges", "blame", "branches", "causal_edges", "branch_details", "shapley"):
        assert expected in keys, f"extraction regex missed a known key: {expected!r}"


def test_every_js_dereferenced_key_is_present_in_tape_to_data_output():
    fake = ScriptedFakeLLM([make_text_response("hi")])
    tape = Tape(agent_name="a")
    transport = TraceforkTransport("record", tape, fake)
    client = anthropic.Anthropic(
        api_key="sk-ant-fake", http_client=httpx.Client(transport=transport), max_retries=0
    )
    client.messages.create(
        model="claude-sonnet-4-6", max_tokens=10, messages=[{"role": "user", "content": "hi"}]
    )

    real_keys = set(_tape_to_data(tape).keys())
    referenced_keys = _js_dereferenced_top_level_keys(_read("report.html"))
    missing = sorted(referenced_keys - real_keys)
    assert missing == [], (
        f"report.html's JS dereferences these top-level keys, but "
        f"report._tape_to_data's real output doesn't have them: {missing}"
    )


def test_every_js_dereferenced_key_is_present_in_get_run_response(tmp_path):
    db = tmp_path / "store.db"
    store = TapeStore(str(db))
    tape = Tape(agent_name="a")
    tape.append_exchange(b"req", b"resp")
    run_id = store.save_tape(tape, run_id="run1")
    store.close()

    init_store(str(db))
    client = TestClient(fastapi_app)
    resp = client.get(f"/api/run/{run_id}")
    assert resp.status_code == 200
    real_keys = set(resp.json().keys())
    referenced_keys = _js_dereferenced_top_level_keys(_read("report.html"))
    missing = sorted(referenced_keys - real_keys)
    assert missing == [], (
        f"report.html's JS dereferences these top-level keys, but "
        f"GET /api/run/{{id}}'s real response doesn't have them: {missing}"
    )
