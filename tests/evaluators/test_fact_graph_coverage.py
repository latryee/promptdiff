"""Comprehensive coverage tests for fact_graph.py."""

from __future__ import annotations

from promptdiff.core.models import RunResult, TestCase
from promptdiff.evaluators.fact_graph import (
    FactGraphEvaluator,
    KnowledgeTriplet,
    extract_triplets_heuristic,
)


def test_knowledge_triplet_to_string() -> None:
    t = KnowledgeTriplet("France", "capital is", "Paris")
    assert t.to_string() == "(France -> capital is -> Paris)"


def test_extract_triplets_heuristic() -> None:
    text = "Python was created by Guido van Rossum. It supports functional programming."
    triplets = extract_triplets_heuristic(text)
    assert len(triplets) > 0
    assert any("python" in t.subject for t in triplets)


def test_extract_triplets_fallback() -> None:
    text = "Apple banana orange."
    triplets = extract_triplets_heuristic(text)
    assert len(triplets) == 1
    assert triplets[0].predicate == "relates_to"


def test_fact_graph_evaluator_with_context() -> None:
    evaluator = FactGraphEvaluator(match_threshold=0.50)
    tc = TestCase(
        id="tc1",
        vars={"context": "Python is created by Guido. It supports dynamic typing."},
    )
    r1 = RunResult(
        prompt_name="v1",
        test_case_id="tc1",
        rendered_prompt="p",
        output="Python is created by Guido.",
        latency_ms=10.0,
        prompt_tokens=5,
        completion_tokens=5,
        total_tokens=10,
        cost_usd=0.0,
        model="mock",
    )
    r2 = RunResult(
        prompt_name="v2",
        test_case_id="tc1",
        rendered_prompt="p",
        output="Python is created by Guido and supports dynamic typing.",
        latency_ms=10.0,
        prompt_tokens=5,
        completion_tokens=5,
        total_tokens=10,
        cost_usd=0.0,
        model="mock",
    )
    score = evaluator.evaluate(r1, r2, tc)
    assert score.name == "fact_graph"
    assert score.passed is True
    assert score.v2_score >= 0.5


def test_fact_graph_evaluator_missing_context() -> None:
    evaluator = FactGraphEvaluator()
    tc = TestCase(id="tc_empty", vars={})
    r1 = RunResult(
        prompt_name="v1",
        test_case_id="tc_empty",
        rendered_prompt="p",
        output="Hello",
        latency_ms=10.0,
        prompt_tokens=1,
        completion_tokens=1,
        total_tokens=2,
        cost_usd=0.0,
        model="m",
    )
    r2 = RunResult(
        prompt_name="v2",
        test_case_id="tc_empty",
        rendered_prompt="p",
        output="Hello",
        latency_ms=10.0,
        prompt_tokens=1,
        completion_tokens=1,
        total_tokens=2,
        cost_usd=0.0,
        model="m",
    )
    score = evaluator.evaluate(r1, r2, tc)
    assert score.passed is True
    assert "No ground truth" in score.message
