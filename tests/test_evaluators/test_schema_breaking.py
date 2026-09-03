"""Unit tests for JSON Schema Breaking Change Detector and Evaluator."""

from __future__ import annotations

import json

import pytest

from promptdiff.core.models import RunResult, TestCase
from promptdiff.evaluators.schema_breaking import (
    SchemaBreakingChangeEvaluator,
    detect_schema_breaking_changes,
)


def test_schema_breaking_identical_compatible() -> None:
    """Identical structured outputs must be fully compatible."""
    v1 = json.dumps({"order_id": "ORD-123", "items_count": 3, "total_price": 49.99})
    v2 = json.dumps({"order_id": "ORD-123", "items_count": 3, "total_price": 49.99})

    report = detect_schema_breaking_changes(v1, v2)
    assert report.is_compatible is True
    assert report.has_breaking_changes is False
    assert report.breaking_count == 0


def test_schema_breaking_removed_field() -> None:
    """Omitting an existing field in candidate output is a breaking change."""
    v1 = json.dumps({"order_id": "ORD-123", "status": "shipped", "customer_email": "alice@example.com"})
    v2 = json.dumps({"order_id": "ORD-123", "status": "shipped"})  # customer_email missing

    report = detect_schema_breaking_changes(v1, v2)
    assert report.is_compatible is False
    assert report.has_breaking_changes is True
    assert report.breaking_count == 1
    diff = report.differences[0]
    assert diff.path == "$.customer_email"
    assert diff.change_type == "REMOVED_FIELD"


def test_schema_breaking_type_mutation() -> None:
    """Mutating a field type (e.g. integer to string) is a breaking change."""
    v1 = json.dumps({"user_id": 9942, "active": True})
    v2 = json.dumps({"user_id": "9942", "active": True})  # integer -> string

    report = detect_schema_breaking_changes(v1, v2)
    assert report.is_compatible is False
    assert report.has_breaking_changes is True
    diff = report.differences[0]
    assert diff.path == "$.user_id"
    assert diff.change_type == "TYPE_MUTATION"
    assert diff.v1_value_or_type == "integer"
    assert diff.v2_value_or_type == "string"


def test_schema_breaking_nullability_violation() -> None:
    """Field becoming null in v2 is a nullability violation."""
    v1 = json.dumps({"token": "secret_token_abc"})
    v2 = json.dumps({"token": None})

    report = detect_schema_breaking_changes(v1, v2)
    assert report.is_compatible is False
    assert report.has_breaking_changes is True
    diff = report.differences[0]
    assert diff.change_type == "NULLABILITY_VIOLATION"


def test_schema_breaking_additive_extension_is_compatible() -> None:
    """Adding new fields in v2 is non-breaking (additive extension)."""
    v1 = json.dumps({"id": 1, "name": "PromptDiff"})
    v2 = json.dumps({"id": 1, "name": "PromptDiff", "version": "3.4.0"})

    report = detect_schema_breaking_changes(v1, v2)
    assert report.is_compatible is True
    assert report.has_breaking_changes is False
    assert report.warning_count == 1
    assert report.differences[0].change_type == "ADDED_FIELD"


@pytest.mark.asyncio
async def test_schema_breaking_evaluator_integration() -> None:
    """Test SchemaBreakingChangeEvaluator running inside runner / test case."""
    evaluator = SchemaBreakingChangeEvaluator()
    tc = TestCase(id="tc1", vars={"query": "get invoice"})

    def make_result(prompt_name: str, output: str) -> RunResult:
        return RunResult(
            prompt_name=prompt_name,
            test_case_id="tc1",
            rendered_prompt="get invoice",
            output=output,
            latency_ms=100.0,
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            cost_usd=0.001,
            model="mock-gpt-4o",
        )

    # Case 1: Pass
    v1_res = make_result("v1", json.dumps({"invoice_id": 101, "amount": 250.0}))
    v2_res = make_result("v2", json.dumps({"invoice_id": 102, "amount": 300.0}))
    res_pass = await evaluator.async_evaluate(v1_res, v2_res, tc)
    assert res_pass.passed is True
    assert res_pass.v2_score == 1.0

    # Case 2: Fail due to removed field
    v2_broken = make_result("v2", json.dumps({"amount": 300.0}))  # invoice_id dropped
    res_fail = await evaluator.async_evaluate(v1_res, v2_broken, tc)
    assert res_fail.passed is False
    assert res_fail.v2_score < 1.0
    assert res_fail.details["breaking_count"] == 1
