"""Tests migrated to feature-focused test suite."""

from __future__ import annotations

import pytest

from promptdiff.core.models import RunResult, TestCase
from promptdiff.evaluators.code_sandbox import SafeCodeSandboxEvaluator
from promptdiff.evaluators.fairness import FairnessEvaluator
from promptdiff.evaluators.multilingual import MultilingualConsistencyEvaluator
from promptdiff.evaluators.schema_repair import SchemaRepairEvaluator
from promptdiff.evaluators.trajectory import TrajectoryEvaluator
from promptdiff.evaluators.vision import VisionDiffEvaluator


@pytest.mark.asyncio
async def test_fairness_evaluator() -> None:
    """Test AI fairness & demographic perturbation evaluator."""
    ev = FairnessEvaluator(force_mock=True)
    tc = TestCase(id="t1", vars={"query": "David is applying for a senior loan"})

    r1 = RunResult(
        prompt_name="p1",
        test_case_id="t1",
        rendered_prompt="x",
        output="David loan approved.",
        latency_ms=100.0,
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
        output="David loan approved for senior tier.",
        latency_ms=100.0,
        prompt_tokens=10,
        completion_tokens=10,
        total_tokens=20,
        cost_usd=0.0001,
        model="gpt-4o",
    )

    score = await ev.async_evaluate(r1, r2, tc)
    assert score.name == "fairness"
    assert score.passed is True


@pytest.mark.asyncio
async def test_schema_repair_evaluator() -> None:
    """Test JSON schema auto-repair evaluator."""
    ev = SchemaRepairEvaluator(force_mock=True)
    tc = TestCase(id="t1", vars={"query": "Give me JSON"})

    r1 = RunResult(
        prompt_name="p1",
        test_case_id="t1",
        rendered_prompt="x",
        output='```json\n{"status": "ok",}\n```',
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
        output='{"status": "ok"}',
        latency_ms=10.0,
        prompt_tokens=10,
        completion_tokens=10,
        total_tokens=20,
        cost_usd=0.0001,
        model="gpt-4o",
    )

    score = await ev.async_evaluate(r1, r2, tc)
    assert score.passed is True


@pytest.mark.asyncio
async def test_vision_evaluator() -> None:
    """Test multi-modal vision evaluator."""
    ev = VisionDiffEvaluator(force_mock=True)
    tc = TestCase(id="t1", vars={"image": "sample.jpg"}, expected_output="Invoice total is $500")

    r1 = RunResult(
        prompt_name="p1",
        test_case_id="t1",
        rendered_prompt="x",
        output="Total: $500",
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
        output="Invoice total is $500.00",
        latency_ms=10.0,
        prompt_tokens=10,
        completion_tokens=10,
        total_tokens=20,
        cost_usd=0.0001,
        model="gpt-4o",
    )

    score = await ev.async_evaluate(r1, r2, tc)
    assert score.passed is True


def test_code_sandbox_evaluator() -> None:
    """Test safe isolated execution of generated code."""
    evaluator = SafeCodeSandboxEvaluator()
    code_valid = "def add(a, b):\n    return a + b"
    test_valid = "assert add(2, 3) == 5"

    res = evaluator.execute_snippet(code_valid, test_valid)
    assert res.passed is True

    test_failing = "assert add(2, 3) == 999"
    res_fail = evaluator.execute_snippet(code_valid, test_failing)
    assert res_fail.passed is False
    assert "AssertionError" in str(res_fail.error_message)


def test_multilingual_evaluator() -> None:
    """Test cross-lingual consistency and language parity."""
    evaluator = MultilingualConsistencyEvaluator()
    score_tr = evaluator.evaluate_language_invariance("Bu sistem prompt diff testidir ve harika çalışır.", "tr")
    assert score_tr >= 0.60

    score_en = evaluator.evaluate_language_invariance("This is a prompt diff evaluation tool and works great.", "en")
    assert score_en >= 0.60


@pytest.mark.asyncio
async def test_trajectory_evaluator() -> None:
    """Test agent multi-turn trajectory evaluator."""
    ev = TrajectoryEvaluator(force_mock=True)

    tc = TestCase(
        id="agent_case_1",
        vars={
            "trajectory": [
                {"role": "user", "content": "Fetch order #999 and refund"},
                {"role": "assistant", "content": "Let me lookup the order."},
            ]
        },
    )

    r1 = RunResult(
        prompt_name="agent_v1",
        test_case_id="agent_case_1",
        rendered_prompt="test",
        output='<tool_call>{"name": "get_order", "id": 999}</tool_call>',
        latency_ms=150.0,
        prompt_tokens=50,
        completion_tokens=20,
        total_tokens=70,
        cost_usd=0.0001,
        model="gpt-4o",
    )
    r2 = RunResult(
        prompt_name="agent_v2",
        test_case_id="agent_case_1",
        rendered_prompt="test",
        output='<tool_call>{"name": "get_order", "id": 999}</tool_call><tool_call>{"name": "issue_refund", "id": 999}</tool_call>',
        latency_ms=160.0,
        prompt_tokens=50,
        completion_tokens=30,
        total_tokens=80,
        cost_usd=0.00015,
        model="gpt-4o",
    )

    score = await ev.async_evaluate(r1, r2, tc)
    assert score.name == "trajectory"
    assert score.passed is True
    assert score.v2_score >= 0.80
