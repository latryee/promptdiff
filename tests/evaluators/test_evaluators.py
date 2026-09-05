"""Unit tests for Evaluators subsystem."""

from promptdiff.core.models import RunResult, TestCase
from promptdiff.evaluators.cost import CostEvaluator
from promptdiff.evaluators.json_validity import JsonValidityEvaluator, validate_schema
from promptdiff.evaluators.latency import LatencyEvaluator
from promptdiff.evaluators.regex_match import RegexMatchEvaluator
from promptdiff.evaluators.similarity import (
    SimilarityEvaluator,
)


def create_run(output: str, latency_ms: float, cost_usd: float, tokens: int = 50) -> RunResult:
    return RunResult(
        prompt_name="v",
        test_case_id="tc",
        rendered_prompt="test",
        output=output,
        latency_ms=latency_ms,
        prompt_tokens=10,
        completion_tokens=tokens,
        total_tokens=10 + tokens,
        cost_usd=cost_usd,
        model="gpt-4o",
    )


def test_json_validity_evaluator():
    evaluator = JsonValidityEvaluator()
    tc = TestCase(id="tc1")

    # Valid JSON in markdown block
    r1 = create_run('```json\n{"status": "ok", "code": 200}\n```', 100.0, 0.001)
    r2 = create_run('{"status": "ok", "code": 200}', 90.0, 0.001)

    score = evaluator.evaluate(r1, r2, tc)
    assert score.v1_score == 1.0
    assert score.v2_score == 1.0
    assert score.passed is True

    # Invalid JSON
    r_invalid = create_run("This is not JSON { bad syntax", 100.0, 0.001)
    score_bad = evaluator.evaluate(r1, r_invalid, tc)
    assert score_bad.v2_score == 0.0
    assert score_bad.passed is False

    # Plain text without JSON syntax indicators
    r_text1 = create_run("Hello! How can I help you today?", 100.0, 0.001)
    r_text2 = create_run("Hi there! Feel free to ask any question.", 90.0, 0.001)
    score_na = evaluator.evaluate(r_text1, r_text2, tc)
    assert score_na.v1_score == "N/A"
    assert score_na.v2_score == "N/A"
    assert score_na.passed is True
    assert "N/A" in score_na.message


def test_json_schema_validation():
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
        "required": ["name", "age"],
    }
    valid_data = {"name": "John", "age": 30}
    invalid_data = {"name": "John"}  # missing age

    ok1, _ = validate_schema(valid_data, schema)
    ok2, err2 = validate_schema(invalid_data, schema)

    assert ok1 is True
    assert ok2 is False
    assert "Missing required field: 'age'" in str(err2)


def test_latency_evaluator():
    evaluator = LatencyEvaluator()
    tc = TestCase(id="tc1")
    r1 = create_run("out", latency_ms=200.0, cost_usd=0.001)
    r2 = create_run("out", latency_ms=150.0, cost_usd=0.001)

    score = evaluator.evaluate(r1, r2, tc)
    assert score.v1_score == 200.0
    assert score.v2_score == 150.0
    assert score.delta == -50.0
    assert score.delta_pct == -25.0


def test_cost_evaluator():
    evaluator = CostEvaluator()
    tc = TestCase(id="tc1")
    r1 = create_run("out", latency_ms=100.0, cost_usd=0.002)
    r2 = create_run("out", latency_ms=100.0, cost_usd=0.001)

    score = evaluator.evaluate(r1, r2, tc)
    assert score.delta == -0.001
    assert score.delta_pct == -50.0


def test_similarity_evaluator():
    evaluator = SimilarityEvaluator()
    tc = TestCase(id="tc1")
    r1 = create_run("Reset your password by visiting the settings page.", 100.0, 0.001)
    r2 = create_run("Reset your password by opening the settings tab.", 100.0, 0.001)

    score = evaluator.evaluate(r1, r2, tc)
    assert score.v2_score > 0.6
    assert score.passed is True


def test_regex_match_evaluator():
    evaluator = RegexMatchEvaluator(pattern=r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
    tc = TestCase(id="tc1")
    r1 = create_run("Contact support at support@example.com for help.", 100.0, 0.001)
    r2 = create_run("Contact support via phone.", 100.0, 0.001)

    score = evaluator.evaluate(r1, r2, tc)
    assert score.v1_score == 1.0
    assert score.v2_score == 0.0
    assert score.passed is False
