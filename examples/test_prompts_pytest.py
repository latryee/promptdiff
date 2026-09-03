"""Example Pytest Test Suite using PromptDiff Fixtures.

Demonstrates both:
1. `await prompt_diff.compare(...)` (Async pytest integration)
2. `promptdiff_eval(...)` (Synchronous pytest integration)
"""

import pytest

from promptdiff.core.models import TestCase


@pytest.mark.asyncio
async def test_support_bot_prompt_regression(prompt_diff):
    """Test regression between v1 and v2 customer support prompts using async prompt_diff fixture."""
    v1 = "You are a customer support agent. Answer politely: {{query}}"
    v2 = "You are a customer support agent. Answer concisely in bullet points: {{query}}"

    test_cases = [
        TestCase(
            id="tc_1",
            vars={"query": "How do I reset my password?", "context": "Reset in Settings > Security"},
        ),
        TestCase(
            id="tc_2",
            vars={"query": "Request a refund for order #100", "context": "Refunds within 30 days"},
        ),
    ]

    report = await prompt_diff.compare(
        v1=v1,
        v2=v2,
        test_cases=test_cases,
        model="gpt-4o",
        mock=True,
        eval_metrics="json_validity,latency,cost,similarity,faithfulness,security",
        assert_rules=["cost_delta <= 15%", "latency_delta <= 25%"],
    )

    assert report.verdict.passed is True
    assert len(report.comparisons) == 2


def test_support_bot_sync_regression(promptdiff_eval):
    """Test regression using synchronous promptdiff_eval fixture."""
    v1 = "You are a support bot: {{query}}"
    v2 = "You are an AI assistant: {{query}}"

    report = promptdiff_eval(
        v1=v1,
        v2=v2,
        dataset=[{"id": "tc1", "vars": {"query": "Hello"}}],
        mock=True,
        eval_metrics="json_validity,latency,cost",
    )

    assert report.verdict.passed is True
