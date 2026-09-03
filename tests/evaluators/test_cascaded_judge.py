"""Unit tests for Cost-Aware Cascaded LLM Judge Evaluator."""

from __future__ import annotations

import pytest

from promptdiff.core.models import RunResult, TestCase
from promptdiff.evaluators.cascaded_judge import (
    CascadedLLMJudge,
    CascadedLLMJudgeEvaluator,
    cascaded_judge,
)


def test_cascaded_judge_tier1_high_confidence() -> None:
    """Distinctly superior concise candidate should be accepted by Tier 1 judge without escalation."""
    judge = CascadedLLMJudge(confidence_threshold=0.85, force_mock=True)

    v1 = "Hello. We are very happy to help you with your inquiry regarding password reset today."
    v2 = "Here is how to reset your password:\n- Click Forgot Password\n- Verify email\n- Set new password"

    res = judge.judge(v1, v2, query="password reset")
    assert res.tier_used == 1
    assert res.escalated is False
    assert res.cost_saved_pct > 0.0
    assert res.confidence >= 0.85
    assert "Tier-1" in res.reasoning


def test_cascaded_judge_tier2_escalation_on_borderline() -> None:
    """Ambiguous/borderline difference should trigger escalation to Tier 2 frontier judge."""
    # When confidence threshold is very high (0.95), normal cases escalate to Tier 2
    judge = CascadedLLMJudge(confidence_threshold=0.95, force_mock=True)

    v1 = "The package will arrive tomorrow afternoon."
    v2 = "The shipment is scheduled for delivery tomorrow."

    res = judge.judge(v1, v2, query="When will my package arrive?")
    assert res.tier_used == 2
    assert res.escalated is True
    assert "Tier-2" in res.reasoning


def test_cascaded_judge_identical_outputs() -> None:
    """Identical outputs should immediately register high-confidence tie at Tier 1."""
    judge = CascadedLLMJudge(force_mock=True)
    text = "Exact identical response string."

    res = judge.judge(text, text)
    assert res.winner == "tie"
    assert res.tier_used == 1
    assert res.escalated is False
    assert res.confidence > 0.90


@pytest.mark.asyncio
async def test_cascaded_judge_evaluator_integration() -> None:
    """Test CascadedLLMJudgeEvaluator integration with RunResult and TestCase."""
    evaluator = CascadedLLMJudgeEvaluator(force_mock=True)
    tc = TestCase(id="tc_judge", vars={"query": "Explain quantum computing"})

    v1_res = RunResult(
        prompt_name="v1",
        test_case_id="tc_judge",
        rendered_prompt="Explain quantum computing",
        output="Quantum computing uses qubits.",
        latency_ms=150.0,
        prompt_tokens=10,
        completion_tokens=10,
        total_tokens=20,
        cost_usd=0.0005,
        model="mock-gpt-4o",
    )
    v2_res = RunResult(
        prompt_name="v2",
        test_case_id="tc_judge",
        rendered_prompt="Explain quantum computing",
        output="Quantum computing uses qubits:\n- Superposition allows parallel state processing\n- Entanglement correlates particles",
        latency_ms=160.0,
        prompt_tokens=10,
        completion_tokens=25,
        total_tokens=35,
        cost_usd=0.0008,
        model="mock-gpt-4o",
    )

    score = await evaluator.async_evaluate(v1_res, v2_res, tc)
    assert score.name == "cascaded_judge"
    assert score.v2_score >= 0.0
    assert "tier_used" in score.details
    assert "cost_saved_pct" in score.details


def test_cascaded_judge_sdk_convenience() -> None:
    """Test top-level cascaded_judge SDK function."""
    res = cascaded_judge(
        v1_output="Old verbose greeting text.",
        v2_output="Concise solution:\n- Step 1\n- Step 2",
        query="Help",
        force_mock=True,
    )
    assert res.winner in ("v1", "v2", "tie")
    assert res.confidence >= 0.0
