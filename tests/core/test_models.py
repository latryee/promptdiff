"""Tests migrated to feature-focused test suite."""

from __future__ import annotations

from promptdiff.evaluators.trajectory import extract_tool_calls


def test_extract_tool_calls() -> None:
    """Test extraction of tool calls from agent outputs."""
    output_xml = '<tool_call>{"name": "search_db", "arguments": {"query": "user_123"}}</tool_call>'
    calls = extract_tool_calls(output_xml)
    assert len(calls) == 1
    assert calls[0].get("name") == "search_db"

    output_json_block = '```json\n{"action": "send_email", "to": "test@example.com"}\n```'
    calls_json = extract_tool_calls(output_json_block)
    assert len(calls_json) == 1
    assert calls_json[0].get("action") == "send_email"


def test_diff_report_schema_sync_with_examples() -> None:
    """Ensure examples/schema.json is strictly in sync with DiffReport.model_json_schema()."""
    import json
    from pathlib import Path

    from promptdiff.core.models import DiffReport

    schema_file = Path(__file__).parents[2] / "examples" / "schema.json"
    assert schema_file.exists(), "examples/schema.json does not exist"

    disk_schema = json.loads(schema_file.read_text(encoding="utf-8"))
    model_schema = DiffReport.model_json_schema()

    assert disk_schema == model_schema, (
        "examples/schema.json has drifted from DiffReport.model_json_schema()! "
        "Update examples/schema.json to match the updated Pydantic model."
    )


def test_result_models_frozen_immutability() -> None:
    """Ensure DiffReport and RunResult cannot be mutated accidentally after instantiation."""
    import pytest
    from pydantic import ValidationError

    from promptdiff.core.models import DiffReport, RegressionVerdict, RunResult

    res = RunResult(
        prompt_name="v1",
        test_case_id="tc1",
        rendered_prompt="test",
        output="test output",
        latency_ms=10.0,
        prompt_tokens=5,
        completion_tokens=5,
        total_tokens=10,
        cost_usd=0.001,
        model="mock",
    )

    with pytest.raises(ValidationError):
        res.cost_usd = 999.0  # type: ignore[misc]

    report = DiffReport(
        v1_name="v1",
        v2_name="v2",
        model_v1="mock",
        model_v2="mock",
        comparisons=[],
        verdict=RegressionVerdict(passed=True),
        evaluators=["latency"],
        total_cases=0,
    )

    with pytest.raises(ValidationError):
        report.v1_name = "mutated_v1"  # type: ignore[misc]
