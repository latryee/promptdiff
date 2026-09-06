"""Tests for strict assertion parsing, exit codes, and diagnostic explanations."""

from __future__ import annotations

import pytest

from promptdiff.core.exceptions import ConfigurationError
from promptdiff.core.exit_codes import ExitCode
from promptdiff.core.models import ComparisonResult, EvaluatorScore, RunResult, TestCase
from promptdiff.evaluators.assertions import evaluate_assertions, parse_assertion_list, parse_assertion_string


def test_parse_assertion_strict_invalid_raises() -> None:
    with pytest.raises(ConfigurationError, match="Invalid assertion expression"):
        parse_assertion_string("cost_delta invalid 10%", strict=True)


def test_parse_assertion_strict_valid() -> None:
    rule = parse_assertion_string("cost_delta <= 15%", strict=True)
    assert rule is not None
    assert rule.metric == "cost_delta"
    assert rule.operator == "<="
    assert rule.threshold == 15.0
    assert rule.is_pct is True


def test_parse_assertion_list_strict() -> None:
    rules = parse_assertion_list(["cost_delta <= 10%", "latency_delta <= 20ms"], strict=True)
    assert len(rules) == 2

    with pytest.raises(ConfigurationError):
        parse_assertion_list(["broken_rule_without_operator"], strict=True)


def test_exit_code_values() -> None:
    assert ExitCode.SUCCESS == 0
    assert ExitCode.REGRESSION_DETECTED == 1
    assert ExitCode.CONFIGURATION_ERROR == 2
    assert ExitCode.PROVIDER_ERROR == 3
    assert ExitCode.INTERNAL_ERROR == 4


def test_assertion_diagnostic_explanation() -> None:
    tc = TestCase(id="case_regression")
    v1 = RunResult(
        prompt_name="v1",
        test_case_id="case_regression",
        rendered_prompt="test",
        output="output",
        latency_ms=100.0,
        prompt_tokens=10,
        completion_tokens=10,
        total_tokens=20,
        cost_usd=0.001,
        model="mock",
    )
    v2 = RunResult(
        prompt_name="v2",
        test_case_id="case_regression",
        rendered_prompt="test",
        output="output",
        latency_ms=250.0,
        prompt_tokens=10,
        completion_tokens=10,
        total_tokens=20,
        cost_usd=0.003,
        model="mock",
    )
    comp = ComparisonResult(
        test_case=tc,
        v1_result=v1,
        v2_result=v2,
        scores={
            "json_validity": EvaluatorScore(
                name="json_validity",
                v1_score=1.0,
                v2_score=0.0,
                delta=-1.0,
                passed=False,
            )
        },
    )

    rules = parse_assertion_list(["cost_delta <= 50%", "json_validity >= 1.0"])
    verdict = evaluate_assertions([comp], rules)
    assert verdict.passed is False
    assert len(verdict.failed_assertions) == 2
    assert any("Cost delta" in msg for msg in verdict.failed_assertions)
    assert any("failed on test case 'case_regression'" in msg for msg in verdict.failed_assertions)
