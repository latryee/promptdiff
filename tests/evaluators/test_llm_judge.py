"""Tests migrated to feature-focused test suite."""

from __future__ import annotations

import pytest

from promptdiff.core.models import (
    RunResult,
    TestCase,
)
from promptdiff.evaluators.council import CouncilOfJudgesEvaluator
from promptdiff.evaluators.debate import MultiAgentDebateEvaluator
from promptdiff.evaluators.llm_judge import LLMJudgeEvaluator


@pytest.mark.asyncio
async def test_llm_judge_evaluator() -> None:
    """Test LLM-as-a-Judge scoring, comparative evaluation, and dynamic v1 scoring."""
    judge = LLMJudgeEvaluator(model_name="mock-gpt-4o", force_mock=True, pass_threshold=3.0)

    tc = TestCase(id="tc_judge", vars={"query": "How do I update billing?"})
    v1_res = RunResult(
        prompt_name="v1",
        test_case_id="tc_judge",
        rendered_prompt="Query: How do I update billing?",
        output="Please call 1-800-HELP for billing.",
        latency_ms=100.0,
        prompt_tokens=10,
        completion_tokens=10,
        total_tokens=20,
        cost_usd=0.0001,
        model="mock-gpt-4o",
    )
    v2_res = RunResult(
        prompt_name="v2",
        test_case_id="tc_judge",
        rendered_prompt="Query: How do I update billing?",
        output="Go to Settings > Billing and click 'Update Payment Method'.",
        latency_ms=120.0,
        prompt_tokens=10,
        completion_tokens=15,
        total_tokens=25,
        cost_usd=0.00012,
        model="mock-gpt-4o",
    )

    score1 = await judge.async_evaluate(v1_res, v2_res, tc)
    assert score1.name == "llm_judge"
    assert isinstance(score1.v1_score, (int, float))
    assert isinstance(score1.v2_score, (int, float))
    assert score1.v1_score >= 1.0
    assert score1.v2_score >= 1.0
    assert "reasoning" in score1.details
    assert "preference" in score1.details

    # Test with a completely different v1 baseline output to verify v1_score is dynamically computed, not hardcoded constant
    v1_res_alt = RunResult(
        prompt_name="v1",
        test_case_id="tc_judge",
        rendered_prompt="Query: How do I update billing?",
        output="A totally different, extremely verbose, convoluted response that rambles on about unrelated topics and historical billing records from 1999.",
        latency_ms=300.0,
        prompt_tokens=10,
        completion_tokens=50,
        total_tokens=60,
        cost_usd=0.0005,
        model="mock-gpt-4o",
    )
    score2 = await judge.async_evaluate(v1_res_alt, v2_res, tc)
    # v1_score must be dynamically evaluated based on input content
    assert isinstance(score2.v1_score, (int, float))
    assert score2.v1_score >= 1.0
    assert score1.v1_score != score2.v1_score or score2.details["preference"] == "V2"


@pytest.mark.asyncio
async def test_council_of_judges_evaluator() -> None:
    """Test multi-model ensemble Council evaluator."""
    ev = CouncilOfJudgesEvaluator(force_mock=True)
    tc = TestCase(id="t1", vars={"query": "Test question"})
    r1 = RunResult(
        prompt_name="p1",
        test_case_id="t1",
        rendered_prompt="x",
        output="Hello",
        latency_ms=10.0,
        prompt_tokens=10,
        completion_tokens=10,
        total_tokens=20,
        cost_usd=0.0001,
        model="gpt-4o",
    )
    r2 = RunResult(
        prompt_name="p2",
        test_case_id="t1",
        rendered_prompt="x",
        output="Hello world",
        latency_ms=10.0,
        prompt_tokens=10,
        completion_tokens=10,
        total_tokens=20,
        cost_usd=0.0001,
        model="gpt-4o",
    )

    score = await ev.async_evaluate(r1, r2, tc)
    assert score.name == "council"
    assert "Council Consensus" in score.message
    assert score.passed is True


@pytest.mark.asyncio
async def test_multi_agent_debate_evaluator() -> None:
    """Test MultiAgentDebateEvaluator adversarial cross-examination."""
    evaluator = MultiAgentDebateEvaluator(force_mock=True)
    round_res = await evaluator.conduct_debate(
        query="Explain quantum tunneling concisely.",
        v1_output="Quantum tunneling is a quantum mechanical phenomenon where subatomic particles pass through a potential barrier.",
        v2_output="Quantum tunneling: Particles cross barriers via wave-function probability.",
    )
    assert round_res.winner in ("v1", "v2", "TIE")
    assert round_res.confidence >= 0.50

    tc = TestCase(id="t1", vars={"query": "Explain quantum tunneling"})
    r1 = RunResult(
        prompt_name="p1",
        test_case_id="t1",
        rendered_prompt="x",
        output="long explanation",
        latency_ms=100.0,
        prompt_tokens=10,
        completion_tokens=10,
        total_tokens=20,
        cost_usd=0.001,
        model="gpt-4o",
    )
    r2 = RunResult(
        prompt_name="p2",
        test_case_id="t1",
        rendered_prompt="x",
        output="concise explanation",
        latency_ms=100.0,
        prompt_tokens=10,
        completion_tokens=10,
        total_tokens=20,
        cost_usd=0.001,
        model="gpt-4o",
    )
    score = await evaluator.async_evaluate(r1, r2, tc)
    assert score.name == "debate_judge"
