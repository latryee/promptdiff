"""Tests migrated to feature-focused test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from promptdiff.core.models import PromptVersion, TestCase
from promptdiff.optimizer.auto_prompt import PromptOptimizer
from promptdiff.providers.mock_provider import MockProvider


@pytest.mark.asyncio
async def test_auto_prompt_optimizer(tmp_path: Path) -> None:
    """Test DSPy-style Auto-Prompt Optimizer."""
    provider = MockProvider(force_mock=True)

    initial_prompt = PromptVersion(
        name="customer_support",
        template="Answer the user query: {{query}}",
        model="mock-gpt-4o",
    )

    test_cases = [
        TestCase(id="opt_1", description="Refund case", vars={"query": "I want a refund."}),
        TestCase(id="opt_2", description="Password case", vars={"query": "Reset my password."}),
    ]

    optimizer = PromptOptimizer(
        prompt_version=initial_prompt,
        test_cases=test_cases,
        provider=provider,
        meta_provider=provider,
        max_iterations=2,
        force_mock=True,
    )

    result = await optimizer.optimize()

    assert result.original_prompt == initial_prompt.template
    assert len(result.optimized_prompt) > 0
    assert result.final_pass_rate >= result.initial_pass_rate

    out_file = tmp_path / "system_v3_optimized.txt"
    saved = optimizer.save_optimized_prompt(result.optimized_prompt, str(out_file))
    assert Path(saved).is_file()
    assert len(Path(saved).read_text(encoding="utf-8")) > 0
