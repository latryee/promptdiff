"""Coverage tests for TrajectoryEvaluator and CouncilOfJudgesEvaluator."""

from __future__ import annotations

import pytest

from promptdiff.core.models import RunResult, TestCase
from promptdiff.evaluators.council import CouncilOfJudgesEvaluator
from promptdiff.evaluators.trajectory import (
    TrajectoryEvaluator,
    extract_tool_calls,
)


def test_extract_tool_calls() -> None:
    text = (
        "I will search for the user:\n"
        '<tool_call>{"name": "search_users", "arguments": {"id": 123}}</tool_call>\n'
        "And in markdown:\n"
        "```json\n"
        '{"tool": "calculator", "action": "add"}\n'
        "```"
    )
    calls = extract_tool_calls(text)
    assert len(calls) == 2
    assert calls[0]["name"] == "search_users"
    assert calls[1]["tool"] == "calculator"


@pytest.mark.asyncio
async def test_trajectory_evaluator_mock_mode() -> None:
    evaluator = TrajectoryEvaluator(force_mock=True)
    tc = TestCase(
        id="agent_tc",
        vars={
            "conversation": [
                {"role": "user", "content": "Book a flight to Paris"},
            ]
        },
    )
    r1 = RunResult(
        prompt_name="v1",
        test_case_id="agent_tc",
        rendered_prompt="p",
        output="I cannot book flights.",
        latency_ms=10.0,
        prompt_tokens=5,
        completion_tokens=5,
        total_tokens=10,
        cost_usd=0.0,
        model="mock",
    )
    r2 = RunResult(
        prompt_name="v2",
        test_case_id="agent_tc",
        rendered_prompt="p",
        output='<tool_call>{"name": "book_flight", "destination": "CDG"}</tool_call>',
        latency_ms=10.0,
        prompt_tokens=5,
        completion_tokens=5,
        total_tokens=10,
        cost_usd=0.0,
        model="mock",
    )

    score = await evaluator.async_evaluate(r1, r2, tc)
    assert score.name == "trajectory"
    assert score.v2_score == 0.95
    assert score.passed is True


@pytest.mark.asyncio
async def test_council_of_judges_mock_mode() -> None:
    evaluator = CouncilOfJudgesEvaluator(force_mock=True)
    tc = TestCase(id="council_tc", vars={"query": "Explain relativity"})
    r1 = RunResult(
        prompt_name="v1",
        test_case_id="council_tc",
        rendered_prompt="p",
        output="v1 answer",
        latency_ms=10.0,
        prompt_tokens=5,
        completion_tokens=5,
        total_tokens=10,
        cost_usd=0.0,
        model="mock",
    )
    r2 = RunResult(
        prompt_name="v2",
        test_case_id="council_tc",
        rendered_prompt="p",
        output="v2 answer",
        latency_ms=10.0,
        prompt_tokens=5,
        completion_tokens=5,
        total_tokens=10,
        cost_usd=0.0,
        model="mock",
    )

    score = await evaluator.async_evaluate(r1, r2, tc)
    assert score.name == "council"
    assert score.v2_score > score.v1_score
    assert score.passed is True
    assert "Council Consensus" in score.message
