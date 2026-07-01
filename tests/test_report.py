"""Report generation smoke-tests — offline, no API keys."""

import json
import tempfile
from pathlib import Path

import anthropic
import httpx

from tests.fakes import ScriptedFakeLLM, make_text_response
from tracefork.report import generate_report
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
