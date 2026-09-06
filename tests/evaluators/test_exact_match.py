"""Unit tests for ExactMatchEvaluator and custom evaluator registry."""

from __future__ import annotations

from promptdiff.core.models import EvaluatorScore, RunResult, TestCase
from promptdiff.evaluators.base import BaseEvaluator
from promptdiff.evaluators.exact_match import ExactMatchEvaluator
from promptdiff.evaluators.registry import get_evaluators, register_evaluator


def test_exact_match_against_expected_output() -> None:
    evaluator = ExactMatchEvaluator()
    v1 = RunResult(
        prompt_name="v1",
        test_case_id="tc1",
        rendered_prompt="Hello",
        output="Hello world!",
        latency_ms=10.0,
        prompt_tokens=2,
        completion_tokens=3,
        total_tokens=5,
        cost_usd=0.0001,
        model="mock",
    )
    v2 = RunResult(
        prompt_name="v2",
        test_case_id="tc1",
        rendered_prompt="Hello",
        output="Hello world!",
        latency_ms=10.0,
        prompt_tokens=2,
        completion_tokens=3,
        total_tokens=5,
        cost_usd=0.0001,
        model="mock",
    )
    tc = TestCase(id="tc1", expected_output="Hello world!")

    score = evaluator.evaluate(v1, v2, tc)
    assert score.passed is True
    assert score.v1_score == 1.0
    assert score.v2_score == 1.0
    assert score.delta == 0.0


def test_exact_match_mismatch_and_diff() -> None:
    evaluator = ExactMatchEvaluator()
    v1 = RunResult(
        prompt_name="v1",
        test_case_id="tc1",
        rendered_prompt="Hello",
        output="Hello world!",
        latency_ms=10.0,
        prompt_tokens=2,
        completion_tokens=3,
        total_tokens=5,
        cost_usd=0.0001,
        model="mock",
    )
    v2 = RunResult(
        prompt_name="v2",
        test_case_id="tc1",
        rendered_prompt="Hello",
        output="Hello brave new world!",
        latency_ms=10.0,
        prompt_tokens=2,
        completion_tokens=3,
        total_tokens=5,
        cost_usd=0.0001,
        model="mock",
    )
    tc = TestCase(id="tc1", expected_output="Hello world!")

    score = evaluator.evaluate(v1, v2, tc)
    assert score.passed is False
    assert score.v1_score == 1.0
    assert score.v2_score == 0.0
    assert score.delta == -1.0
    assert "Mismatch with expected target" in score.message


def test_exact_match_whitespace_and_case_options() -> None:
    evaluator = ExactMatchEvaluator(ignore_case=True, strip_whitespace=True)
    v1 = RunResult(
        prompt_name="v1",
        test_case_id="tc1",
        rendered_prompt="Hello",
        output="HELLO WORLD\n",
        latency_ms=10.0,
        prompt_tokens=2,
        completion_tokens=3,
        total_tokens=5,
        cost_usd=0.0001,
        model="mock",
    )
    v2 = RunResult(
        prompt_name="v2",
        test_case_id="tc1",
        rendered_prompt="Hello",
        output="  hello world  ",
        latency_ms=10.0,
        prompt_tokens=2,
        completion_tokens=3,
        total_tokens=5,
        cost_usd=0.0001,
        model="mock",
    )
    tc = TestCase(id="tc1", expected_output="hello world")

    score = evaluator.evaluate(v1, v2, tc)
    assert score.passed is True
    assert score.v1_score == 1.0
    assert score.v2_score == 1.0


def test_registry_resolve_exact_match() -> None:
    evals1 = get_evaluators(["exact_match"])
    assert len(evals1) == 1
    assert isinstance(evals1[0], ExactMatchEvaluator)

    evals2 = get_evaluators(["exact"])
    assert len(evals2) == 1
    assert isinstance(evals2[0], ExactMatchEvaluator)


def test_register_custom_evaluator() -> None:
    class CustomDummyEvaluator(BaseEvaluator):
        name = "dummy_custom"

        def evaluate(self, v1: RunResult, v2: RunResult, tc: TestCase) -> EvaluatorScore:
            return EvaluatorScore(name=self.name, v1_score=1.0, v2_score=1.0, passed=True)

    register_evaluator("dummy_custom", CustomDummyEvaluator)
    evals = get_evaluators(["dummy_custom"])
    assert len(evals) == 1
    assert isinstance(evals[0], CustomDummyEvaluator)
