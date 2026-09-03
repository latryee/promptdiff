"""Tests migrated to feature-focused test suite."""

from __future__ import annotations

import pytest

from promptdiff.core.models import (
    PromptVersion,
    TestCase,
)
from promptdiff.core.runner import PromptDiffRunner
from promptdiff.providers.mock_provider import MockProvider


@pytest.mark.asyncio
async def test_async_runner_concurrency() -> None:
    """Test concurrent batch execution with semaphore limit."""
    v1 = PromptVersion(name="v1", template="Hello {{name}}", model="mock-gpt-4o")
    v2 = PromptVersion(name="v2", template="Hi {{name}}, welcome!", model="mock-gpt-4o")
    p1 = MockProvider()
    p2 = MockProvider()

    runner = PromptDiffRunner(
        v1_prompt=v1,
        v2_prompt=v2,
        provider_v1=p1,
        provider_v2=p2,
        concurrency=4,
    )

    test_cases = [TestCase(id=f"case_{i}", vars={"name": f"User_{i}"}) for i in range(12)]

    report = await runner.run(test_cases)
    assert len(report.comparisons) == 12
    assert report.total_cases == 12
    assert report.verdict.passed is True
