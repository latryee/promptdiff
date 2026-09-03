"""Deep unit tests for answer_relevance, hallucination_graph, trajectory, and council evaluators."""

from __future__ import annotations

import pytest

from promptdiff.core.models import RunResult, TestCase
from promptdiff.evaluators.answer_relevance import AnswerRelevanceEvaluator
from promptdiff.evaluators.council import CouncilOfJudgesEvaluator
from promptdiff.evaluators.hallucination_graph import (
    HallucinationGraphResult,
    TokenAttributionEvaluator,
    _compute_span_similarity,
    _extract_claim_spans,
)
from promptdiff.evaluators.trajectory import (
    TrajectoryEvaluator,
    extract_tool_calls,
)

# ============================================================================
# answer_relevance.py tests
# ============================================================================


def test_answer_relevance_extract_query() -> None:
    ev = AnswerRelevanceEvaluator(force_mock=True)
    tc1 = TestCase(id="1", vars={"query": "How to reset password?"})
    assert ev._extract_query(tc1) == "How to reset password?"

    tc2 = TestCase(id="2", vars={"input": "Billing question"})
    assert ev._extract_query(tc2) == "Billing question"

    tc3 = TestCase(id="3", description="Fallback desc", vars={})
    assert ev._extract_query(tc3) == "Fallback desc"


def test_answer_relevance_parse_output() -> None:
    ev = AnswerRelevanceEvaluator(force_mock=True)

    # Standard format
    text1 = "[REASONING] Directly and clearly answers the user.\n[SCORE] 0.95"
    score1, reason1 = ev._parse_relevance_output(text1)
    assert score1 == 0.95
    assert "Directly" in reason1

    # Alternative format
    text2 = "relevance: 0.70\nSome other notes"
    score2, _ = ev._parse_relevance_output(text2)
    assert score2 == 0.70


@pytest.mark.asyncio
async def test_answer_relevance_evaluate() -> None:
    ev = AnswerRelevanceEvaluator(threshold=0.60, force_mock=True)
    tc = TestCase(id="q1", vars={"query": "What is Python?"})
    r1 = RunResult(
        prompt_name="v1",
        test_case_id="q1",
        rendered_prompt="p",
        output="Python is a high-level programming language.",
        latency_ms=10.0,
        prompt_tokens=5,
        completion_tokens=5,
        total_tokens=10,
        cost_usd=0.0,
        model="mock",
    )
    r2 = RunResult(
        prompt_name="v2",
        test_case_id="q1",
        rendered_prompt="p",
        output="Python is an interpreted language for general programming.",
        latency_ms=10.0,
        prompt_tokens=5,
        completion_tokens=5,
        total_tokens=10,
        cost_usd=0.0,
        model="mock",
    )
    score = await ev.async_evaluate(r1, r2, tc)
    assert score.name == "answer_relevance"
    assert score.passed is True
    assert score.v2_score > 0.0

    # Empty output case
    empty_r = RunResult(
        prompt_name="v2",
        test_case_id="q1",
        rendered_prompt="p",
        output="",
        latency_ms=10.0,
        prompt_tokens=1,
        completion_tokens=0,
        total_tokens=1,
        cost_usd=0.0,
        model="mock",
    )
    empty_score = await ev.async_evaluate(r1, empty_r, tc)
    assert empty_score.v2_score == 0.0


# ============================================================================
# hallucination_graph.py tests
# ============================================================================


def test_hallucination_spans_and_similarity() -> None:
    text = "Paris is the capital of France. It has the Eiffel Tower."
    spans = _extract_claim_spans(text)
    assert len(spans) >= 2

    ctx = "France is a country. Its capital is Paris with the Eiffel Tower."
    sim, match = _compute_span_similarity("Paris is the capital of France", ctx)
    assert sim > 0.5
    assert len(match) > 0

    # Empty context
    sim_empty, _ = _compute_span_similarity("Some text", "")
    assert sim_empty == 0.0


def test_token_attribution_evaluator_analyze() -> None:
    ev = TokenAttributionEvaluator()
    output = "Paris is the capital of France. Mars has oceans of liquid water."
    context = "Paris is the capital of France. Earth is a blue planet."
    res = ev.analyze(output, context)

    assert isinstance(res, HallucinationGraphResult)
    assert res.total_tokens > 0
    assert any(s.is_grounded for s in res.spans)
    assert any(s.status == "HALLUCINATED" for s in res.spans)
    assert len(res.highlighted_terminal_markup) > 0


def test_token_attribution_evaluator_evaluate() -> None:
    ev = TokenAttributionEvaluator()
    tc = TestCase(id="rag1", vars={"context": "Alice is a software engineer living in Berlin."})
    r1 = RunResult(
        prompt_name="v1",
        test_case_id="rag1",
        rendered_prompt="p",
        output="Alice lives in Berlin and works in software engineering.",
        latency_ms=10.0,
        prompt_tokens=5,
        completion_tokens=5,
        total_tokens=10,
        cost_usd=0.0,
        model="mock",
    )
    r2 = RunResult(
        prompt_name="v2",
        test_case_id="rag1",
        rendered_prompt="p",
        output="Alice lives in Tokyo and is an astronaut.",
        latency_ms=10.0,
        prompt_tokens=5,
        completion_tokens=5,
        total_tokens=10,
        cost_usd=0.0,
        model="mock",
    )
    score = ev.evaluate(r1, r2, tc)
    assert score.name == "token_attribution"
    assert score.v1_score > score.v2_score

    # Missing context
    tc_no_ctx = TestCase(id="no_ctx", vars={})
    score_no_ctx = ev.evaluate(r1, r2, tc_no_ctx)
    assert score_no_ctx.passed is True


# ============================================================================
# trajectory.py tests
# ============================================================================


def test_trajectory_parse_judge_score() -> None:
    ev = TrajectoryEvaluator(force_mock=True)

    text1 = "[TRAJECTORY_ANALYSIS] Good path\n[TOOL_ERRORS]\n- Redundant call to weather\n[SCORE] 0.85"
    score, errors = ev._parse_judge_score(text1)
    assert score == 0.85
    assert len(errors) == 1
    assert "weather" in errors[0]

    # No errors
    text2 = "[SCORE] 1.0\n[TOOL_ERRORS] None"
    score2, errors2 = ev._parse_judge_score(text2)
    assert score2 == 1.0
    assert len(errors2) == 0


def test_trajectory_extract_tool_calls_variations() -> None:
    text = (
        '<function_call>{"name": "get_user", "args": {"id": 1}}</function_call>\n'
        "<tool_call>invalid json raw text</tool_call>"
    )
    calls = extract_tool_calls(text)
    assert len(calls) == 2
    assert calls[0]["name"] == "get_user"
    assert "raw" in calls[1]


# ============================================================================
# council.py tests
# ============================================================================


@pytest.mark.asyncio
async def test_council_judge_vote_aggregations() -> None:
    ev = CouncilOfJudgesEvaluator(force_mock=True)
    tc = TestCase(id="1", vars={})
    r1 = RunResult(
        prompt_name="v1",
        test_case_id="1",
        rendered_prompt="p",
        output="out1",
        latency_ms=10.0,
        prompt_tokens=5,
        completion_tokens=5,
        total_tokens=10,
        cost_usd=0.0,
        model="mock",
    )
    r2 = RunResult(
        prompt_name="v2",
        test_case_id="1",
        rendered_prompt="p",
        output="out2",
        latency_ms=10.0,
        prompt_tokens=5,
        completion_tokens=5,
        total_tokens=10,
        cost_usd=0.0,
        model="mock",
    )
    score = await ev.async_evaluate(r1, r2, tc)
    assert score.passed is True
    assert "votes" in score.details
    assert len(score.details["votes"]) == 3
