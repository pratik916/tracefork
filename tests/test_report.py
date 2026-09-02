"""Report generation smoke-tests — offline, no API keys."""

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import anthropic
import httpx

from tests.fakes import ScriptedFakeLLM, make_text_response
from tracefork.constants import BOUNDARY_V1, OTEL_INGESTED_BOUNDARY, PROXY_BOUNDARY
from tracefork.report import _tape_to_data, generate_report
from tracefork.tape import Tape
from tracefork.transport import TraceforkTransport

_REPO_ROOT = Path(__file__).resolve().parent.parent

TEXT_RESP = make_text_response("Hello world")


def _make_tape() -> Tape:
    fake = ScriptedFakeLLM([TEXT_RESP])
    tape = Tape(agent_name="test_agent")
    transport = TraceforkTransport("record", tape, fake)
    client = anthropic.Anthropic(
        api_key="sk-ant-fake",
        http_client=httpx.Client(transport=transport),
        max_retries=0,
    )
    client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=100,
        messages=[{"role": "user", "content": "Hello"}],
    )
    return tape


def _extract_data(content: str) -> dict:
    marker = "window.__TRACEFORK_DATA__ = "
    start = content.find(marker) + len(marker)
    end = content.find(";\n", start)
    return json.loads(content[start:end])


def test_generate_report_creates_html_file():
    tape = _make_tape()
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "report.html"
        generate_report(tape, out)
        assert out.exists()
        content = out.read_text()
        assert "tracefork" in content
        assert "__TRACEFORK_DATA__" in content


def test_report_embeds_tape_data():
    tape = _make_tape()
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "report.html"
        generate_report(tape, out)
        data = _extract_data(out.read_text())
        assert data["agent_name"] == "test_agent"
        assert len(data["exchanges"]) == 1


def test_report_has_valid_exchange_structure():
    tape = _make_tape()
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "report.html"
        generate_report(tape, out)
        data = _extract_data(out.read_text())
        ex = data["exchanges"][0]
        assert "role" in ex
        assert "preview" in ex
        assert "request" in ex
        assert ex["preview"] == "Hello world"


def test_report_escapes_script_breakout():
    """A tape whose content contains </script> must not break out of the inline script."""
    evil = make_text_response("</script><img src=x onerror=alert(1)>")
    fake = ScriptedFakeLLM([evil])
    tape = Tape(agent_name="evil")
    transport = TraceforkTransport("record", tape, fake)
    client = anthropic.Anthropic(
        api_key="sk-ant-fake",
        http_client=httpx.Client(transport=transport),
        max_retries=0,
    )
    client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=100,
        messages=[{"role": "user", "content": "hi"}],
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "report.html"
        generate_report(tape, out)
        content = out.read_text()
        # The injected data block must not contain a raw closing script tag.
        marker = "window.__TRACEFORK_DATA__ = "
        start = content.find(marker)
        end = content.find(";\n", start)
        injected = content[start:end]
        assert "</script" not in injected
        assert "\\u003c/script" in injected
        # And the escaped payload still parses back to the original text.
        data = _extract_data(content)
        assert data["exchanges"][0]["preview"] == "</script><img src=x onerror=alert(1)>"


def test_report_includes_blame_when_provided():
    tape = _make_tape()
    blame = {0: {"flip_rate": 0.8, "ci_lo": 0.6, "ci_hi": 0.95}}
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "report.html"
        generate_report(tape, out, blame=blame)
        data = _extract_data(out.read_text())
        # JSON object keys are strings after round-trip
        assert data["blame"]["0"]["flip_rate"] == 0.8


def test_report_blame_includes_trust_flags():
    """Per-step divergence rate / UNDEFINED counts (FlipRateResult's trust
    flags) must round-trip into the embedded report data."""
    tape = _make_tape()
    blame = {
        0: {
            "flip_rate": 0.8,
            "ci_lo": 0.6,
            "ci_hi": 0.95,
            "divergence_rate": 0.3,
            "undefined": 3,
            "trials": 10,
            "valid_trials": 7,
            "trustworthy": False,
        }
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "report.html"
        generate_report(tape, out, blame=blame)
        data = _extract_data(out.read_text())
        step0 = data["blame"]["0"]
        assert step0["divergence_rate"] == 0.3
        assert step0["undefined"] == 3
        assert step0["trustworthy"] is False


# ── Wilson CI as the primary visual mark (v1.0.0 readiness item 52) ────────
# Before this, the 80px bar encoded only the flip_rate point estimate and
# the Wilson interval was 10px muted text; trials/q_value/p_value were in
# the payload and rendered nowhere.


def test_blame_ci_range_bar_replaces_the_point_estimate_only_bar():
    tape = _make_tape()
    blame = {0: {"flip_rate": 0.8, "ci_lo": 0.6, "ci_hi": 0.95}}
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "report.html"
        generate_report(tape, out, blame=blame)
        content = out.read_text()
        assert "function blameCiBarHtml(info, isDecisive)" in content
        assert "blame-ci-range" in content
        assert "blame-ci-point" in content
        # the old point-estimate-only bar (width sized to flip_rate alone,
        # nothing marking the interval) must be gone, not just supplemented
        # -- distinct from .blame-bar-wrap, the (still-present) outer
        # wrapper class this new bar renders inside of.
        assert 'class="blame-bar${' not in content
        assert ".blame-bar {" not in content
        assert ".blame-bar.decisive {" not in content


def test_blame_ci_bar_encodes_lo_hi_as_left_and_width_not_just_the_point():
    tape = _make_tape()
    blame = {0: {"flip_rate": 0.8, "ci_lo": 0.6, "ci_hi": 0.95}}
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "report.html"
        generate_report(tape, out, blame=blame)
        content = out.read_text()
        start = content.index("function blameCiBarHtml(info, isDecisive)")
        body = content[start : start + 900]
        assert "clamp01(info.ci_lo)" in body
        assert "clamp01(info.ci_hi)" in body
        assert "clamp01(info.flip_rate)" in body
        # the range's CSS left/width come from lo/hi, the point's left from
        # flip_rate -- both distinct visual encodings on the same mark
        assert 'style="left:${lo}%;width:' in body
        assert 'style="left:${point}%"' in body


def test_blame_panel_surfaces_trials_p_value_q_value_when_present():
    tape = _make_tape()
    blame = {
        0: {
            "flip_rate": 0.8,
            "ci_lo": 0.6,
            "ci_hi": 0.95,
            "trials": 12,
            "p_value": 0.0123,
            "q_value": 0.0456,
        }
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "report.html"
        generate_report(tape, out, blame=blame)
        content = out.read_text()
        assert "function blameStatsHtml(info)" in content
        assert "n=${info.trials}" in content
        assert "p=${info.p_value.toFixed(3)}" in content
        assert "q=${info.q_value.toFixed(3)}" in content
        # data round-trips into the payload the function above reads
        data = _extract_data(content)
        assert data["blame"]["0"]["trials"] == 12
        assert data["blame"]["0"]["p_value"] == 0.0123
        assert data["blame"]["0"]["q_value"] == 0.0456


def test_blame_stats_html_degrades_gracefully_when_fields_are_absent():
    """A legacy/partial blame dict without trials/p_value/q_value must not
    render blank/NaN gaps -- each field is independently optional."""
    tape = _make_tape()
    blame = {0: {"flip_rate": 0.8, "ci_lo": 0.6, "ci_hi": 0.95}}
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "report.html"
        generate_report(tape, out, blame=blame)
        content = out.read_text()
        start = content.index("function blameStatsHtml(info)")
        body = content[start : start + 500]
        assert "typeof info.trials === 'number'" in body
        assert "typeof info.p_value === 'number'" in body
        assert "typeof info.q_value === 'number'" in body


def test_render_blame_wires_the_new_bar_and_stats_helpers():
    content = (_REPO_ROOT / "web" / "report.html").read_text()
    start = content.index("function renderBlame(data)")
    body = content[start : start + 1200]
    assert "blameCiBarHtml(info, isDecisive)" in body
    assert "blameStatsHtml(info)" in body


def test_report_defaults_replay_to_empty_dict_when_not_provided():
    tape = _make_tape()
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "report.html"
        generate_report(tape, out)
        data = _extract_data(out.read_text())
        assert data["replay"] == {}


def test_report_includes_replay_diagnostics_when_provided():
    """`replay=` (from `verification_result_to_dict`) must embed the
    bit-exactness receipt and, on divergence, the structured field diff."""
    tape = _make_tape()
    replay = {
        "bit_exact": False,
        "matched": 0,
        "total": 1,
        "fingerprints_match": False,
        "divergence": {
            "step_index": 0,
            "cause": "code_change",
            "message": "request #0 diverged from tape (recorded abc123, replay def456)",
            "diag": {
                "step_index": 0,
                "recorded_fingerprint": "abc123",
                "live_fingerprint": "def456",
                "matcher_name": "identity",
                "normalized_fields": [],
                "is_real_divergence": True,
                "message": "1 field(s) differ from the recorded request",
                "field_diffs": [
                    {"path": "$.messages[0].content", "recorded": "Hello", "live": "Goodbye"}
                ],
            },
        },
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "report.html"
        generate_report(tape, out, replay=replay)
        data = _extract_data(out.read_text())
        assert data["replay"]["bit_exact"] is False
        diag = data["replay"]["divergence"]["diag"]
        assert diag["is_real_divergence"] is True
        assert diag["field_diffs"][0]["path"] == "$.messages[0].content"
        assert diag["field_diffs"][0]["recorded"] == "Hello"
        assert diag["field_diffs"][0]["live"] == "Goodbye"


def test_report_escapes_script_breakout_in_divergence_diff():
    """A divergence diff whose recorded/live values contain </script> must not
    break out of the inline script either — the same escaping that protects
    tape content (`test_report_escapes_script_breakout`) covers the whole
    injected data blob, diagnostics included."""
    tape = _make_tape()
    replay = {
        "bit_exact": False,
        "matched": 0,
        "total": 1,
        "fingerprints_match": False,
        "divergence": {
            "step_index": 0,
            "cause": "code_change",
            "message": "diverged",
            "diag": {
                "step_index": 0,
                "recorded_fingerprint": "abc",
                "live_fingerprint": "def",
                "matcher_name": "identity",
                "normalized_fields": [],
                "is_real_divergence": True,
                "message": "diverged",
                "field_diffs": [
                    {
                        "path": "$.messages[0].content",
                        "recorded": "hi",
                        "live": "</script><img src=x onerror=alert(1)>",
                    }
                ],
            },
        },
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "report.html"
        generate_report(tape, out, replay=replay)
        content = out.read_text()
        marker = "window.__TRACEFORK_DATA__ = "
        start = content.find(marker)
        end = content.find(";\n", start)
        injected = content[start:end]
        assert "</script" not in injected
        data = _extract_data(content)
        live_value = data["replay"]["divergence"]["diag"]["field_diffs"][0]["live"]
        assert live_value == "</script><img src=x onerror=alert(1)>"


# ── boundary / provenance / redaction badge (tracefork-bge.20) ─────────────


def test_tape_to_data_emits_correct_boundary_for_all_three_boundary_constants():
    """`_tape_to_data` must surface `tape.boundary` verbatim so the report UI can
    render a trust badge — a forensic-only tape must not look identical to a
    verified one (see `constants.py`'s boundary markers)."""
    for boundary in (BOUNDARY_V1, OTEL_INGESTED_BOUNDARY, PROXY_BOUNDARY):
        tape = _make_tape()
        tape.boundary = boundary
        data = _tape_to_data(tape)
        assert data["boundary"] == boundary


def test_report_html_content_redacted_true_drives_the_redaction_badge():
    """A `content_redacted=True` tape must embed that flag in the injected data
    AND ship the client-side badge wiring (element + renderer) that turns it
    into a visible warning — content_redacted stays forensic-only (never fed
    into `digest()`), so the report is the only place a viewer learns about it."""
    tape = _make_tape()
    tape.content_redacted = True
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "report.html"
        generate_report(tape, out)
        content = out.read_text()
        data = _extract_data(content)
        assert data["content_redacted"] is True
        # Structural wiring: the badge element and its renderer must exist in
        # the single-file template so the injected flag actually reaches the UI.
        assert 'id="redacted-tag"' in content
        assert 'id="boundary-tag"' in content
        assert "renderProvenanceBadges" in content


def test_report_html_boundary_badge_wiring_present_for_forensic_boundary():
    """A forensic-only boundary (OTel-ingested / proxy-recorded) must not be
    silently indistinguishable from a verified `BOUNDARY_V1` tape in the report."""
    tape = _make_tape()
    tape.boundary = PROXY_BOUNDARY
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "report.html"
        generate_report(tape, out)
        data = _extract_data(content := out.read_text())
        assert data["boundary"] == PROXY_BOUNDARY
        assert "renderProvenanceBadges" in content


# ── fork-tree panel data (tracefork-bge.15) ────────────────────────────────


def test_tape_to_data_defaults_branches_to_empty_list():
    """No `branches=` passed must still yield a falsy `[]`, the same neutral
    empty-state pattern `replay={}` already establishes."""
    tape = _make_tape()
    data = _tape_to_data(tape)
    assert data["branches"] == []


def test_tape_to_data_includes_populated_branches_list():
    """A populated `branches=` list (the shape `TapeStore.list_branches`
    returns) round-trips into the data dict unchanged."""
    tape = _make_tape()
    branches = [
        {
            "branch_id": "b1",
            "divergence_step": 3,
            "mutation_desc": "swapped tool result",
            "created_at": "2026-01-01T00:00:00",
            "branch_digest": "abc123def456",
        },
        {
            "branch_id": "b2",
            "divergence_step": 0,
            "mutation_desc": "swapped assistant text",
            "created_at": "2026-01-02T00:00:00",
            "branch_digest": "def456abc123",
        },
    ]
    data = _tape_to_data(tape, branches=branches)
    assert data["branches"] == branches


def test_report_embeds_populated_branches_list():
    """`generate_report`'s injected data blob carries the branches list end to
    end, and the single-file template ships the fork-tree render wiring."""
    tape = _make_tape()
    branches = [
        {
            "branch_id": "b1",
            "divergence_step": 0,
            "mutation_desc": "mutated response",
            "created_at": "2026-01-01T00:00:00",
            "branch_digest": "abc123",
        }
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "report.html"
        generate_report(tape, out, branches=branches)
        content = out.read_text()
        data = _extract_data(content)
        assert data["branches"] == branches
        assert "renderForkTree" in content


def test_report_escapes_script_breakout_in_branch_mutation_desc():
    """A branch's `mutation_desc` containing `</script>` must not break out of
    the inline script either — the same escaping that protects tape content
    and replay diagnostics covers branch metadata too."""
    tape = _make_tape()
    branches = [
        {
            "branch_id": "b1",
            "divergence_step": 0,
            "mutation_desc": "</script><img src=x onerror=alert(1)>",
            "created_at": "",
            "branch_digest": "abc123",
        }
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "report.html"
        generate_report(tape, out, branches=branches)
        content = out.read_text()
        marker = "window.__TRACEFORK_DATA__ = "
        start = content.find(marker)
        end = content.find(";\n", start)
        injected = content[start:end]
        assert "</script" not in injected
        data = _extract_data(content)
        assert data["branches"][0]["mutation_desc"] == "</script><img src=x onerror=alert(1)>"


def test_report_survives_non_utf8_locale():
    """Regression for tracefork-sis.31: `report.py`'s `_template_path().read_text()`
    and `output_path.write_text()` must pass `encoding="utf-8"` explicitly.

    `web/report.html` (this project's headline single-file deliverable) itself
    contains non-ASCII bytes (a UTF-8 em-dash in its own `<title>`). Without an
    explicit `encoding=`, both calls fall back to the interpreter's LOCALE
    encoding (`locale.getpreferredencoding`) — fine on a UTF-8-locale POSIX box,
    but on a genuinely non-UTF-8 locale (stock `LC_ALL=C`, the common
    non-UTF-8-locale-platform case this project's `ubuntu-latest`-only CI would
    never catch) `read_text()` raises `UnicodeDecodeError` outright and, even if
    it didn't, `write_text()` would silently mojibake the report's own title/UI
    text on the way back out.

    The locale-driven default text encoding is fixed by the C locale at
    INTERPRETER STARTUP (PEP 538/540) — monkeypatching `locale
    .getpreferredencoding` inside an already-running process has no effect on
    it — so this spawns two genuinely separate interpreters (one UTF-8-locale,
    one forced non-UTF-8-locale) and asserts they produce byte-identical
    report HTML, with the `<title>` tag's em-dash round-tripping correctly in
    both.
    """
    gen_script = (
        "import sys\n"
        "from pathlib import Path\n"
        "from tracefork.report import generate_report\n"
        "from tracefork.tape import Tape\n"
        "tape = Tape(agent_name='locale_test')\n"
        "generate_report(tape, Path(sys.argv[1]))\n"
    )

    def _generate_report_html(tmpdir: Path, name: str, env_overrides: dict[str, str]) -> bytes:
        script_path = tmpdir / "gen_report.py"
        script_path.write_text(gen_script, encoding="utf-8")
        out_path = tmpdir / name
        env = dict(os.environ)
        env.update(env_overrides)
        subprocess.run(
            [sys.executable, str(script_path), str(out_path)],
            check=True,
            cwd=str(_REPO_ROOT),
            env=env,
            capture_output=True,
        )
        return out_path.read_bytes()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        utf8_html = _generate_report_html(
            tmp_path,
            "report_utf8_locale.html",
            {"LC_ALL": "en_US.UTF-8", "LANG": "en_US.UTF-8", "PYTHONUTF8": "1"},
        )
        non_utf8_html = _generate_report_html(
            tmp_path,
            "report_non_utf8_locale.html",
            {
                "LC_ALL": "C",
                "LANG": "C",
                "PYTHONCOERCECLOCALE": "0",
                "PYTHONUTF8": "0",
            },
        )

    assert non_utf8_html == utf8_html, (
        "report HTML generated under a forced non-UTF-8 locale must be "
        "byte-identical to one generated under a UTF-8 locale"
    )

    title_match = re.search(rb"<title>(.*?)</title>", non_utf8_html)
    assert title_match is not None
    assert title_match.group(1).decode("utf-8") == "tracefork — time-travel debugger"


# ── branch_details cap (tracefork-sis.56) ───────────────────────────────────


def test_tape_to_data_branch_details_under_cap_is_embedded_whole():
    """A small branch_details dict, well under the default cap, must round
    -trip unchanged with no truncation marker -- the byte-identical-when-
    -unaffected discipline every other optional field in this module follows."""
    tape = _make_tape()
    branch_details = {"b1": {"agent_name": "synthetic", "exchanges": []}}
    data = _tape_to_data(tape, branch_details=branch_details)
    assert data["branch_details"] == branch_details
    assert data["branch_details_truncated"] is None


def test_tape_to_data_defaults_branch_details_truncated_to_none_when_absent():
    tape = _make_tape()
    data = _tape_to_data(tape)
    assert data["branch_details"] == {}
    assert data["branch_details_truncated"] is None


def test_generate_report_caps_branch_details_and_reports_truncation():
    """tracefork-sis.56: reproduces the measured defect (a run with 100
    branches producing a payload dominated by branch_details, no cap, no
    opt-out) at a scale a test can run fast -- exceeding the cap must
    truncate the embedded payload, not silently balloon it, and the output
    must carry a structured truncation notice a consumer can act on."""
    tape = _make_tape()
    big_note = "x" * 20_000
    branch_details = {
        f"branch-{i}": {"agent_name": "synthetic", "exchanges": [], "note": big_note}
        for i in range(100)
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "report.html"
        generate_report(tape, out, branch_details=branch_details)
        payload_bytes = out.stat().st_size
        data = _extract_data(out.read_text())

    kept = len(data["branch_details"])
    assert 0 < kept < 100, "must be a real, non-trivial, strict subset -- not all-or-nothing"
    truncated = data["branch_details_truncated"]
    assert truncated is not None
    assert truncated["total_branches"] == 100
    assert truncated["included"] == kept
    assert truncated["omitted"] == 100 - kept
    # 100 uncapped 20KB branches would be ~2MB; the cap must actually bound
    # the file, not just add a cosmetic marker beside an unbounded blob.
    assert payload_bytes < 700_000, f"payload was {payload_bytes} bytes -- cap did not bound it"


def test_tape_to_data_branch_details_cap_bytes_is_a_tunable_parameter():
    """A caller can pass a smaller/larger `branch_details_cap_bytes` -- proves
    the cap is a real, tunable parameter (the cli.py flag's target), not a
    hardcoded constant baked into this function."""
    tape = _make_tape()
    branch_details = {f"b{i}": {"note": "x" * 1000} for i in range(10)}

    tight = _tape_to_data(tape, branch_details=branch_details, branch_details_cap_bytes=500)
    assert len(tight["branch_details"]) < 10
    assert tight["branch_details_truncated"] is not None

    loose = _tape_to_data(tape, branch_details=branch_details, branch_details_cap_bytes=1_000_000)
    assert loose["branch_details"] == branch_details
    assert loose["branch_details_truncated"] is None


def test_generate_report_branch_details_cap_bytes_param_threads_through():
    """`generate_report` itself (not just `_tape_to_data`) must accept and
    apply `branch_details_cap_bytes`."""
    tape = _make_tape()
    branch_details = {f"b{i}": {"note": "x" * 1000} for i in range(10)}
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "report.html"
        generate_report(tape, out, branch_details=branch_details, branch_details_cap_bytes=500)
        data = _extract_data(out.read_text())
    assert len(data["branch_details"]) < 10
    assert data["branch_details_truncated"] is not None


# ── compressed payload for large runs (v1.0.0 readiness item 34) ───────────
# Measured: report.py's _safe_json used indent=2 and no compression, so a
# 400-step run's HTML report ran into the hundreds of MB. gzip+base64 (no
# new dependency; web/report.html decodes it with the standard
# DecompressionStream Web API) is dramatically smaller. A run under the
# threshold must round-trip byte-for-byte identical to before this feature
# existed (all the OTHER tests in this file already pin that -- this
# section is additive, not a replacement).


def _make_tape_n(n: int):
    responses = [make_text_response(f"reply {i}") for i in range(n)]
    fake = ScriptedFakeLLM(responses)
    tape = Tape(agent_name="test_agent")
    transport = TraceforkTransport("record", tape, fake)
    client = anthropic.Anthropic(
        api_key="sk-ant-fake", http_client=httpx.Client(transport=transport), max_retries=0
    )
    messages: list[dict] = []
    for i in range(n):
        messages.append({"role": "user", "content": f"question {i}"})
        resp = client.messages.create(model="claude-sonnet-4-6", max_tokens=100, messages=messages)
        messages.append({"role": "assistant", "content": resp.content[0].text})
    return tape


def _extract_gzip_b64_data(content: str) -> dict:
    import base64
    import gzip

    marker = "window.__TRACEFORK_DATA_GZIP_B64__ = "
    start = content.find(marker) + len(marker)
    end = content.find(";\n", start)
    b64 = json.loads(content[start:end])  # the payload is a JSON-quoted base64 string
    return json.loads(gzip.decompress(base64.b64decode(b64)))


def test_report_under_threshold_still_uses_the_plain_uncompressed_path():
    tape = _make_tape_n(5)
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "report.html"
        generate_report(tape, out, compression_step_threshold=50)
        content = out.read_text()
        assert "window.__TRACEFORK_DATA__ = " in content
        assert "window.__TRACEFORK_DATA_GZIP_B64__ = " not in content
        data = _extract_data(content)
        assert len(data["exchanges"]) == 5


def test_report_at_or_above_threshold_uses_the_gzip_b64_path():
    tape = _make_tape_n(5)
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "report.html"
        # a low threshold exercises the compressed path without a slow,
        # huge test fixture
        generate_report(tape, out, compression_step_threshold=5)
        content = out.read_text()
        assert "window.__TRACEFORK_DATA_GZIP_B64__ = " in content
        assert "window.__TRACEFORK_DATA__ = " not in content
        data = _extract_gzip_b64_data(content)
        assert len(data["exchanges"]) == 5


def test_gzip_b64_payload_decompresses_to_functionally_identical_data():
    """The compressed report must decode to the EXACT same `_tape_to_data`
    shape the uncompressed path embeds -- functionally identical, per this
    item's own acceptance bar, not merely "close enough"."""
    tape = _make_tape_n(6)
    with tempfile.TemporaryDirectory() as tmpdir:
        uncompressed = Path(tmpdir) / "plain.html"
        compressed = Path(tmpdir) / "gz.html"
        generate_report(tape, uncompressed, compression_step_threshold=100)
        generate_report(tape, compressed, compression_step_threshold=1)
        plain_data = _extract_data(uncompressed.read_text())
        gzip_data = _extract_gzip_b64_data(compressed.read_text())
        assert gzip_data == plain_data


def test_gzip_b64_payload_is_html_safe_with_no_escaping_needed():
    """A base64 alphabet (A-Za-z0-9+/=) contains no `< > &` -- a
    `</script>`-breakout payload in the recorded content must survive
    compression with no special-casing (unlike _safe_json's escaping)."""
    tape = _make_tape_n(3)
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "report.html"
        generate_report(tape, out, compression_step_threshold=1)
        content = out.read_text()
        marker = "window.__TRACEFORK_DATA_GZIP_B64__ = "
        start = content.find(marker) + len(marker)
        end = content.find(";\n", start)
        payload = content[start:end]
        assert "</script" not in payload
        assert "<" not in payload and ">" not in payload and "&" not in payload
        # and it's a real, non-trivial base64 string, not an empty stub
        assert len(payload) > 20


def test_gzip_b64_payload_shrinks_a_large_report_dramatically():
    """Real measured evidence, not just a structural assertion: the
    compressed report file must actually be smaller than the uncompressed
    one for the SAME data, by a wide margin."""
    tape = _make_tape_n(60)
    with tempfile.TemporaryDirectory() as tmpdir:
        uncompressed = Path(tmpdir) / "plain.html"
        compressed = Path(tmpdir) / "gz.html"
        generate_report(tape, uncompressed, compression_step_threshold=1_000_000)
        generate_report(tape, compressed, compression_step_threshold=50)
        uncompressed_size = uncompressed.stat().st_size
        compressed_size = compressed.stat().st_size
        assert compressed_size < uncompressed_size
        # a conservative floor, real ratio measured far higher
        assert uncompressed_size / compressed_size > 5


def test_report_under_threshold_is_byte_for_byte_identical_to_default_behavior():
    """The default `compression_step_threshold` must never change output for
    small tapes -- every OTHER test in this file (written before this
    feature existed) generates a report with NO threshold override at all,
    so this pins that the new parameter's default keeps them passing."""
    tape = _make_tape_n(3)
    with tempfile.TemporaryDirectory() as tmpdir:
        default_call = Path(tmpdir) / "default.html"
        explicit_high_threshold = Path(tmpdir) / "explicit.html"
        generate_report(tape, default_call)
        generate_report(tape, explicit_high_threshold, compression_step_threshold=10_000)
        assert default_call.read_text() == explicit_high_threshold.read_text()


def test_cli_report_defaults_to_the_plain_path_for_a_small_tape(tmp_path):
    """The CLI's `report` command never passes `compression_step_threshold`
    explicitly -- it must inherit generate_report's own default and use the
    plain path for ordinary small demo/quickstart tapes."""
    from typer.testing import CliRunner

    from tracefork.cli import app

    runner = CliRunner()
    tape = _make_tape_n(3)
    tape_path = tmp_path / "run.tape.sqlite"
    tape.save(str(tape_path))
    out = tmp_path / "report.html"
    result = runner.invoke(app, ["report", "--tape", str(tape_path), "-o", str(out)])
    assert result.exit_code == 0, result.output
    content = out.read_text()
    assert "window.__TRACEFORK_DATA__ = " in content
    assert "window.__TRACEFORK_DATA_GZIP_B64__ = " not in content
