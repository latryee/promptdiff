"""Deep unit tests for JsonValidityEvaluator, DatasetMutator, and SLABudgetSimulator."""

from __future__ import annotations

from pathlib import Path

import pytest

from promptdiff.core.models import PromptVersion, RunResult, TestCase
from promptdiff.evaluators.json_validity import (
    JsonValidityEvaluator,
    extract_json,
    validate_schema,
)
from promptdiff.generators.mutator import (
    DatasetMutator,
    inject_slang_and_abbreviations,
    inject_typos,
)
from promptdiff.production.sla import (
    SLABudgetReport,
    SLABudgetSimulator,
)

# ============================================================================
# json_validity.py tests
# ============================================================================


def test_extract_json_markdown_and_substring() -> None:
    # 1. Direct JSON
    d1, err1 = extract_json('{"key": "value"}')
    assert d1 == {"key": "value"}
    assert err1 is None

    # 2. Markdown block
    md = 'Here is the response:\n```json\n{"status": "ok", "code": 200}\n```\nHope it helps!'
    d2, err2 = extract_json(md)
    assert d2 == {"status": "ok", "code": 200}
    assert err2 is None

    # 3. Substring in text
    sub = 'Prefix notes: {"user_id": 42} suffix notes.'
    d3, err3 = extract_json(sub)
    assert d3 == {"user_id": 42}
    assert err3 is None

    # 4. Completely invalid
    d4, err4 = extract_json("Not json at all")
    assert d4 is None
    assert err4 is not None


def test_validate_schema_rules() -> None:
    schema = {
        "type": "object",
        "required": ["name", "age"],
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
    }
    # Valid
    ok, err = validate_schema({"name": "Alice", "age": 30}, schema)
    assert ok is True
    assert err is None

    # Missing required field
    missing_ok, missing_err = validate_schema({"name": "Alice"}, schema)
    assert missing_ok is False
    assert "Missing required field" in (missing_err or "")

    # Wrong type
    type_ok, type_err = validate_schema({"name": "Alice", "age": "thirty"}, schema)
    assert type_ok is False
    assert "expected integer" in (type_err or "")

    # Non-dict data
    non_dict_ok, non_dict_err = validate_schema(["item1"], schema)
    assert non_dict_ok is False
    assert "Expected object" in (non_dict_err or "")


def test_json_validity_evaluator_with_schema() -> None:
    ev = JsonValidityEvaluator()
    schema = {
        "required": ["name"],
        "properties": {"name": {"type": "string"}},
    }
    tc = TestCase(id="tc_schema", schema_definition=schema, vars={})

    r1 = RunResult(
        prompt_name="v1",
        test_case_id="tc_schema",
        rendered_prompt="p",
        output='{"name": "test"}',
        latency_ms=10.0,
        prompt_tokens=5,
        completion_tokens=5,
        total_tokens=10,
        cost_usd=0.0,
        model="mock",
    )
    # v2 has missing required field
    r2 = RunResult(
        prompt_name="v2",
        test_case_id="tc_schema",
        rendered_prompt="p",
        output='{"other": "field"}',
        latency_ms=10.0,
        prompt_tokens=5,
        completion_tokens=5,
        total_tokens=10,
        cost_usd=0.0,
        model="mock",
    )
    score = ev.evaluate(r1, r2, tc)
    assert score.name == "json_validity"
    assert score.v1_score == 1.0
    assert score.v2_score == 0.5
    assert "Schema mismatch" in score.message


# ============================================================================
# mutator.py tests
# ============================================================================


def test_mutator_helpers() -> None:
    typo_text = inject_typos("please check my account password", typo_rate=0.5)
    assert len(typo_text) > 0

    slang_text = inject_slang_and_abbreviations("Please thank you as soon as possible password account")
    assert "pls" in slang_text
    assert "thx" in slang_text
    assert "asap" in slang_text
    assert "pwd" in slang_text
    assert "acct" in slang_text


def test_dataset_mutator_modes(tmp_path: Path) -> None:
    seed = [TestCase(id="seed_1", vars={"query": "Please refund order 123", "amount": "50"})]
    mutator = DatasetMutator(seed_testcases=seed, multiplier=6)

    mutated_cases = mutator.generate_mutations()
    assert len(mutated_cases) >= 6

    # Test saving
    out_file = tmp_path / "mutated.jsonl"
    mutator.save_to_jsonl(mutated_cases, str(out_file))
    assert out_file.exists()
    assert len(out_file.read_text(encoding="utf-8").strip().splitlines()) >= 6


# ============================================================================
# sla.py tests
# ============================================================================


@pytest.mark.asyncio
async def test_sla_budget_simulator() -> None:
    pv = PromptVersion(name="sla_prompt", template="Answer: {{query}}")
    cases = [
        TestCase(id="sla_1", vars={"query": "Query A"}),
        TestCase(id="sla_2", vars={"query": "Query B"}),
    ]

    # Passing test with generous ceilings
    sim_pass = SLABudgetSimulator(
        prompt_version=pv,
        test_cases=cases,
        max_p99_latency_ms=5000.0,
        max_cost_per_request_usd=1.0,
        force_mock=True,
    )
    rep_pass = await sim_pass.run_stress_test()
    assert isinstance(rep_pass, SLABudgetReport)
    assert rep_pass.sla_passed is True
    assert len(rep_pass.breaches) == 0

    # Failing test with strict ceilings
    sim_fail = SLABudgetSimulator(
        prompt_version=pv,
        test_cases=cases,
        max_p99_latency_ms=0.001,  # Guaranteed breach
        max_cost_per_request_usd=0.00000001,
        force_mock=True,
    )
    rep_fail = await sim_fail.run_stress_test()
    assert rep_fail.sla_passed is False
    assert len(rep_fail.breaches) > 0
