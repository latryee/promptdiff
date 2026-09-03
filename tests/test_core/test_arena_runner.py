"""Tests migrated to feature-focused test suite."""

from __future__ import annotations

import pytest

from promptdiff.core.models import (
    PromptVersion,
    TestCase,
)
from promptdiff.core.runner import ArenaRunner
from promptdiff.providers.mock_provider import MockProvider


@pytest.mark.asyncio
async def test_arena_runner_multi_model() -> None:
    """Test Multi-Model Arena runner across 3 models."""
    variants = {
        "gpt-4o": PromptVersion(name="gpt-4o", template="Answer {{q}} concisely", model="mock-gpt-4o"),
        "claude-3-5": PromptVersion(
            name="claude-3-5", template="Answer {{q}} in bullets", model="mock-claude-3-5-sonnet"
        ),
        "gemini-flash": PromptVersion(
            name="gemini-flash", template="Answer {{q}} with steps", model="mock-gemini-2.0-flash"
        ),
    }
    providers = {k: MockProvider(model_name=k) for k in variants}

    arena = ArenaRunner(variants=variants, providers=providers, concurrency=6)
    test_cases = [
        TestCase(id="tc_1", vars={"q": "What is Docker?"}),
        TestCase(id="tc_2", vars={"q": "Explain Kubernetes pods"}),
    ]

    report = await arena.run(test_cases)
    assert len(report.leaderboard) == 3
    assert report.total_cases == 2
    assert report.leaderboard[0].rank == 1
    for summary in report.leaderboard:
        assert summary.p_value is not None
        assert summary.confidence_interval is not None
