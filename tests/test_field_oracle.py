"""FieldDiffOracle tests — all offline, zero API spend."""

import json

import anthropic
import httpx

from tests.fakes import ScriptedFakeLLM, make_text_response
from tracefork import blame
from tracefork.blame import BlameEngine, StringMatchOracle
from tracefork.field_oracle import FieldDiffOracle
from tracefork.tape import Tape
from tracefork.transport import TraceforkTransport

# ── registration ─────────────────────────────────────────────────────────────


def test_field_diff_oracle_registered_by_import():
    assert "field_diff" in blame.registered_oracles()
    assert blame.get_oracle("field_diff") is FieldDiffOracle


# ── field resolution + grading ───────────────────────────────────────────────


def test_grades_true_on_nested_dict_field():
    oracle = FieldDiffOracle(field_path="$.result.status", success_re="SUCCESS", failure_re="FAIL")
    output = json.dumps({"result": {"status": "SUCCESS"}, "other": "noise"})
    assert oracle.grade(output) is True


def test_grades_false_on_list_index_field():
    oracle = FieldDiffOracle(
        field_path="$.items[0].status", success_re="SUCCESS", failure_re="FAIL"
    )
    output = json.dumps({"items": [{"status": "FAIL"}]})
    assert oracle.grade(output) is False


def test_returns_none_when_path_does_not_resolve():
    oracle = FieldDiffOracle(field_path="$.result.status", success_re="SUCCESS", failure_re="FAIL")
    # Missing key.
    assert oracle.grade(json.dumps({"other": "noise"})) is None

    indexed = FieldDiffOracle(
        field_path="$.items[5].status", success_re="SUCCESS", failure_re="FAIL"
    )
    # Out-of-range index.
    assert indexed.grade(json.dumps({"items": [{"status": "SUCCESS"}]})) is None


def test_returns_none_on_non_json_output():
    oracle = FieldDiffOracle(field_path="$.status", success_re="SUCCESS", failure_re="FAIL")
    assert oracle.grade("this is not json at all") is None


# ── provenance-of-value proof ────────────────────────────────────────────────


def test_field_scoped_grading_is_immune_to_unrelated_field_noise():
    """Two outputs differing only in an UNRELATED field, but sharing the same
    value at the graded ``field_path``, must grade identically — unlike a
    whole-text ``StringMatchOracle``, which reacts to the noise."""
    field_oracle = FieldDiffOracle(
        field_path="$.result.status", success_re="SUCCESS", failure_re="FAIL"
    )
    string_oracle = StringMatchOracle(success_re="SUCCESS", failure_re="FAIL")

    quiet = json.dumps({"result": {"status": "FAIL"}, "note": "nothing unusual"})
    noisy = json.dumps({"result": {"status": "FAIL"}, "note": "previous attempt reported SUCCESS"})

    # Same graded field value ("FAIL") in both -> identical verdict.
    assert field_oracle.grade(quiet) is False
    assert field_oracle.grade(noisy) is False

    # Contrast: a whole-text oracle DOES react to the unrelated "SUCCESS" noise.
    assert string_oracle.grade(quiet) is False
    assert string_oracle.grade(noisy) is True


# ── real BlameEngine wiring (integration) ────────────────────────────────────


def _field_agent(client: anthropic.Anthropic) -> str:
    """Two-turn agent; turn2's history embeds turn1's reply text, so a
    mutation at turn1 changes what turn2 asks (and thus the counterfactual
    tail) — mirrors ``test_blame.py``'s ``_booking_agent`` pattern."""
    r1 = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=100,
        messages=[{"role": "user", "content": "run the job"}],
    )
    first = r1.content[0].text
    r2 = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=100,
        messages=[
            {"role": "user", "content": "run the job"},
            {"role": "assistant", "content": first},
            {"role": "user", "content": "report the result as JSON"},
        ],
    )
    return r2.content[0].text


def _record_field_run(resp1: bytes, resp2: bytes) -> Tape:
    fake = ScriptedFakeLLM([resp1, resp2])
    tape = Tape()
    transport = TraceforkTransport("record", tape, fake)
    client = anthropic.Anthropic(
        api_key="sk-ant-fake",
        http_client=httpx.Client(transport=transport),
        max_retries=0,
    )
    _field_agent(client)
    return tape


def test_field_diff_oracle_wired_into_blame_engine():
    """Real end-to-end ``BlameEngine.rank()`` wiring, zero ``BlameEngine``
    changes: perturbing the non-final step whose re-recorded tail changes
    only an UNRELATED field yields NO_FLIP; perturbing the step whose
    response flips the GRADED field yields FLIP."""
    turn1 = make_text_response("neutral turn1 text")
    turn2 = make_text_response(json.dumps({"result": {"status": "SUCCESS"}, "note": "original"}))
    tape = _record_field_run(turn1, turn2)

    oracle = FieldDiffOracle(field_path="$.result.status", success_re="SUCCESS", failure_re="FAIL")

    unrelated_noise_tail = make_text_response(
        json.dumps({"result": {"status": "SUCCESS"}, "note": "a totally different note"})
    )
    graded_flip_resp = make_text_response(
        json.dumps({"result": {"status": "FAIL"}, "note": "original"})
    )

    def perturb_factory(step_idx: int) -> tuple[bytes, object]:
        if step_idx == 0:
            # Non-final step: mutate turn1; the re-recorded tail (turn2) keeps
            # the SAME graded field but a DIFFERENT unrelated field.
            return make_text_response("mutated turn1 text"), ScriptedFakeLLM([unrelated_noise_tail])
        # Final step: mutate turn2 directly, flipping the GRADED field.
        return graded_flip_resp, None

    report = BlameEngine.rank(
        tape,
        _field_agent,
        oracle,
        perturb_factory=perturb_factory,
        k=3,
        budget_usd=100.0,
    )

    assert report is not None
    assert report.parent_outcome is True
    step0 = next(r for r in report.results if r.step_index == 0)
    step1 = next(r for r in report.results if r.step_index == 1)
    assert step0.flip_rate == 0.0
    assert step1.flip_rate == 1.0
