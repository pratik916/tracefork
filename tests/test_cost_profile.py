"""Offline tests for tracefork-bge.52: the cost/profile dashboard panel.

`compute_cost_profile`/`cost_profile_to_dict` (src/tracefork/cost_profile.py)
are built entirely on the existing `providers.get_adapter`/`pricing.get_rates`
seams `blame.BudgetGovernor.estimate` already uses -- these tests price
exchanges the same way `test_blame.py::test_budget_governor_estimates` does.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from typer.testing import CliRunner

from tracefork.cli import app
from tracefork.constants import SONNET
from tracefork.cost_profile import compute_cost_profile, cost_profile_to_dict
from tracefork.pricing import get_rates
from tracefork.providers.anthropic import AnthropicAdapter
from tracefork.report import generate_report
from tracefork.store import TapeStore
from tracefork.tape import Tape

runner = CliRunner()


def _extract_data(content: str) -> dict:
    marker = "window.__TRACEFORK_DATA__ = "
    start = content.find(marker) + len(marker)
    end = content.find(";\n", start)
    return json.loads(content[start:end])


def test_compute_cost_profile_groups_by_model_and_prices_correctly():
    adapter = AnthropicAdapter()
    tape = Tape(agent_name="t")
    tape.append_exchange(
        b"{}", adapter.build_text_response("hi", model=SONNET, input_tokens=100, output_tokens=20)
    )
    tape.append_exchange(
        b"{}", adapter.build_text_response("bye", model=SONNET, input_tokens=50, output_tokens=10)
    )

    profile = compute_cost_profile(tape)
    assert len(profile.by_model) == 1
    model_cost = profile.by_model[0]
    assert model_cost.model == SONNET
    assert model_cost.n_exchanges == 2
    assert model_cost.input_tokens == 150
    assert model_cost.output_tokens == 30

    in_rate, out_rate = get_rates(SONNET)
    expected = 150 * in_rate + 30 * out_rate
    assert abs(model_cost.total_cost_usd - expected) < 1e-9
    assert abs(profile.total_cost_usd - expected) < 1e-9


def test_compute_cost_profile_groups_by_tool_name():
    adapter = AnthropicAdapter()
    tape = Tape(agent_name="t")
    tape.append_exchange(
        b"{}",
        adapter.build_tool_use_response(
            "search", {"q": "x"}, model=SONNET, input_tokens=80, output_tokens=15
        ),
    )

    profile = compute_cost_profile(tape)
    assert len(profile.by_tool) == 1
    tool_cost = profile.by_tool[0]
    assert tool_cost.tool_name == "search"
    assert tool_cost.n_calls == 1

    in_rate, out_rate = get_rates(SONNET)
    expected = 80 * in_rate + 15 * out_rate
    assert abs(tool_cost.total_cost_usd - expected) < 1e-9


def test_compute_cost_profile_multi_tool_exchange_attributes_full_cost_to_each_tool():
    """A single exchange invoking >1 distinct tool name attributes its FULL
    modeled cost to each (documented over-count, see the module docstring)."""
    adapter = AnthropicAdapter()
    resp = adapter.build_tool_use_response(
        "search", {"q": "x"}, model=SONNET, input_tokens=80, output_tokens=15
    )
    # Craft a response with two distinct tool_use blocks by hand -- the
    # adapter builder only emits one, so patch content directly.
    data = json.loads(resp)
    data["content"].append({"type": "tool_use", "id": "toolu_2", "name": "browse", "input": {}})
    resp2 = json.dumps(data).encode()

    tape = Tape(agent_name="t")
    tape.append_exchange(b"{}", resp2)
    profile = compute_cost_profile(tape)
    by_tool = {t.tool_name: t for t in profile.by_tool}
    assert set(by_tool) == {"search", "browse"}
    assert by_tool["search"].total_cost_usd == by_tool["browse"].total_cost_usd
    assert by_tool["search"].total_cost_usd == profile.total_cost_usd


def test_compute_cost_profile_empty_tape_returns_zeroed_profile():
    tape = Tape(agent_name="empty")
    profile = compute_cost_profile(tape)
    assert profile.by_model == ()
    assert profile.by_tool == ()
    assert profile.total_cost_usd == 0.0


def test_cost_profile_to_dict_round_trips_through_json():
    adapter = AnthropicAdapter()
    tape = Tape(agent_name="t")
    tape.append_exchange(b"{}", adapter.build_text_response("hi", model=SONNET))
    profile = compute_cost_profile(tape)
    d = cost_profile_to_dict(profile)
    parsed = json.loads(json.dumps(d))
    assert parsed["total_cost_usd"] == d["total_cost_usd"]
    assert parsed["by_model"][0]["model"] == d["by_model"][0]["model"] == SONNET
    assert parsed["by_tool"] == d["by_tool"] == []


def test_report_embeds_cost_profile_and_defaults_to_empty_dict():
    adapter = AnthropicAdapter()
    tape = Tape(agent_name="t")
    tape.append_exchange(b"{}", adapter.build_text_response("hi", model=SONNET))
    profile_dict = cost_profile_to_dict(compute_cost_profile(tape))

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "report.html"
        generate_report(tape, out, cost_profile=profile_dict)
        content = out.read_text()
        data = _extract_data(content)
        assert data["cost_profile"]["by_model"][0]["model"] == SONNET
        assert "renderCostProfile" in content

        out2 = Path(tmpdir) / "report2.html"
        generate_report(tape, out2)
        data2 = _extract_data(out2.read_text())
        assert data2["cost_profile"] == {}


def test_cli_report_auto_embeds_cost_profile(tmp_path):
    db = tmp_path / "store.db"
    store = TapeStore(str(db))
    adapter = AnthropicAdapter()
    tape = Tape(agent_name="t")
    resp = adapter.build_text_response("hi", model=SONNET, input_tokens=10, output_tokens=5)
    tape.append_exchange(b"{}", resp)
    run_id = store.save_tape(tape, run_id="run1")
    store.close()

    out = tmp_path / "report.html"
    result = runner.invoke(app, ["report", run_id, "--store", str(db), "-o", str(out)])
    assert result.exit_code == 0, result.output
    data = _extract_data(out.read_text())
    assert data["cost_profile"]["by_model"][0]["n_exchanges"] == 1


def test_server_get_run_includes_cost_profile(tmp_path):
    from fastapi.testclient import TestClient

    from tracefork.server import app as fastapi_app
    from tracefork.server import init_store

    db = tmp_path / "store.db"
    store = TapeStore(str(db))
    adapter = AnthropicAdapter()
    tape = Tape(agent_name="t")
    tape.append_exchange(b"{}", adapter.build_text_response("hi", model=SONNET))
    run_id = store.save_tape(tape, run_id="run1")
    store.close()

    init_store(str(db))
    client = TestClient(fastapi_app)
    resp = client.get(f"/api/run/{run_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["cost_profile"]["total_cost_usd"] > 0
