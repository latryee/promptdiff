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

