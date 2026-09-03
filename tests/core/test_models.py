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
