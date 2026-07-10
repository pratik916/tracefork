"""Report generation smoke-tests — offline, no API keys."""

import json
import tempfile
from pathlib import Path

import anthropic
import httpx

from tests.fakes import ScriptedFakeLLM, make_text_response
from tracefork.constants import BOUNDARY_V1, OTEL_INGESTED_BOUNDARY, PROXY_BOUNDARY
from tracefork.report import _tape_to_data, generate_report
from tracefork.tape import Tape
from tracefork.transport import TraceforkTransport

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
