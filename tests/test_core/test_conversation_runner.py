"""Unit tests for ConversationVersion multi-turn execution and trajectory evaluation."""

from __future__ import annotations

import pytest

from promptdiff.core.models import ConversationVersion, TestCase
from promptdiff.core.runner import PromptDiffRunner
from promptdiff.evaluators.trajectory import TrajectoryEvaluator
from promptdiff.providers.mock_provider import MockProvider


@pytest.mark.asyncio
async def test_conversation_version_rendering() -> None:
    """Test ConversationVersion multi-turn message rendering with template substitution."""
    conv = ConversationVersion(
        name="conv_v1",
        messages=[
            {"role": "user", "content": "I want to query {{item_id}} from the database."},
            {"role": "assistant", "content": "Understood. Querying database for item {{item_id}}."},
            {"role": "user", "content": "What is the status?"},
        ],
    )
    rendered_msgs = conv.render_messages({"item_id": "item_999"})
    assert len(rendered_msgs) == 3
    assert rendered_msgs[0]["content"] == "I want to query item_999 from the database."
    assert rendered_msgs[1]["content"] == "Understood. Querying database for item item_999."

    transcript = conv.render({"item_id": "item_999"})
    assert "User: I want to query item_999" in transcript
    assert "Assistant: Understood. Querying" in transcript


@pytest.mark.asyncio
async def test_multi_turn_runner_with_trajectory_evaluator() -> None:
    """Test PromptDiffRunner executing multi-turn ConversationVersions with TrajectoryEvaluator."""
    conv_v1 = ConversationVersion(
        name="conv_v1",
        model="mock-gpt-4o",
        messages=[
            {"role": "user", "content": "Find order {{order_id}} and cancel it."},
        ],
    )
    conv_v2 = ConversationVersion(
        name="conv_v2",
        model="mock-gpt-4o",
        messages=[
            {"role": "user", "content": "Find order {{order_id}} and cancel it if pending."},
            {"role": "assistant", "content": '<tool_call>{"tool": "lookup_order", "id": "{{order_id}}"}</tool_call>'},
            {"role": "user", "content": "Proceed with cancellation."},
        ],
    )

    provider_v1 = MockProvider("mock-gpt-4o")
    provider_v2 = MockProvider("mock-gpt-4o")
    evaluator = TrajectoryEvaluator(force_mock=True)

    runner = PromptDiffRunner(
        v1_prompt=conv_v1,
        v2_prompt=conv_v2,
        provider_v1=provider_v1,
        provider_v2=provider_v2,
        evaluators=[evaluator],
    )

    test_cases = [
        TestCase(id="tc_order_1", vars={"order_id": "ORD-12345"}),
        TestCase(id="tc_order_2", vars={"order_id": "ORD-67890"}),
    ]

    report = await runner.run(test_cases)
    assert report.total_cases == 2
    assert len(report.comparisons) == 2

    comp = report.comparisons[0]
    assert "trajectory" in comp.scores
    score = comp.scores["trajectory"]
    assert score.passed is True
    assert score.v2_score >= 0.80
    assert "ORD-12345" in comp.v1_result.rendered_prompt
    assert "ORD-12345" in comp.v2_result.rendered_prompt
