"""Unit tests for CI/CD Regression Assertion Engine."""

from promptdiff.core.models import (
    ComparisonResult,
    EvaluatorScore,
    RunResult,
    TestCase,
)
from promptdiff.evaluators.assertions import (
    evaluate_assertions,
    parse_assertion_list,
    parse_assertion_string,
)


def test_parse_assertion_string():
    rule1 = parse_assertion_string("cost_delta <= 10%")
    assert rule1 is not None
    assert rule1.metric == "cost_delta"
    assert rule1.operator == "<="
    assert rule1.threshold == 10.0
    assert rule1.is_pct is True

    rule2 = parse_assertion_string("latency_delta <= 50ms")
    assert rule2 is not None
    assert rule2.metric == "latency_delta"
    assert rule2.threshold == 50.0

    rule3 = parse_assertion_string("json_validity == 1.0")
    assert rule3 is not None
    assert rule3.operator == "=="
    assert rule3.threshold == 1.0


def test_evaluate_assertions_passing():
    r1 = RunResult(
        prompt_name="v1",
        test_case_id="tc1",
        rendered_prompt="p",
        output="out",
        latency_ms=200.0,
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        cost_usd=0.0010,
        model="gpt-4o",
    )
    r2 = RunResult(
        prompt_name="v2",
        test_case_id="tc1",
        rendered_prompt="p",
        output="out",
        latency_ms=180.0,  # -10%
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        cost_usd=0.0009,  # -10%
        model="gpt-4o",
    )
    comp = ComparisonResult(
        test_case=TestCase(id="tc1"),
        v1_result=r1,
        v2_result=r2,
        scores={"json_validity": EvaluatorScore(name="json_validity", v1_score=1.0, v2_score=1.0, passed=True)},
    )

    rules = parse_assertion_list(["cost_delta <= 10%", "latency_delta <= 15%", "json_validity == 1.0"])
    verdict = evaluate_assertions([comp], rules)

    assert verdict.passed is True
    assert verdict.status == "PASSED"
    assert len(verdict.failed_assertions) == 0


def test_evaluate_assertions_failing():
    r1 = RunResult(
        prompt_name="v1",
        test_case_id="tc1",
        rendered_prompt="p",
        output="out",
        latency_ms=100.0,
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        cost_usd=0.0010,
        model="gpt-4o",
    )
    r2 = RunResult(
        prompt_name="v2",
        test_case_id="tc1",
        rendered_prompt="p",
        output="out",
        latency_ms=250.0,  # +150% regression!
        prompt_tokens=10,
        completion_tokens=50,
        total_tokens=60,
        cost_usd=0.0025,  # +150% cost spike!
        model="gpt-4o",
    )
    comp = ComparisonResult(
        test_case=TestCase(id="tc1"),
        v1_result=r1,
        v2_result=r2,
        scores={"json_validity": EvaluatorScore(name="json_validity", v1_score=1.0, v2_score=0.0, passed=False)},
    )

    rules = parse_assertion_list(["cost_delta <= 10%", "latency_delta <= 20%", "json_validity == 1.0"])
    verdict = evaluate_assertions([comp], rules)

    assert verdict.passed is False
    assert verdict.status == "REGRESSION_DETECTED"
    assert len(verdict.failed_assertions) == 3
