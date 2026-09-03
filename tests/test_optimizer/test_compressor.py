"""Tests migrated to feature-focused test suite."""

from __future__ import annotations

import pytest

from promptdiff.core.models import PromptVersion, TestCase
from promptdiff.optimizer.compressor import PromptCompressor


@pytest.mark.asyncio
async def test_prompt_compressor() -> None:
    """Test prompt token compressor with quality retention."""
    pv = PromptVersion(
        name="verbose_support",
        template="You are a helpful assistant that answers queries politely. Please kindly assist: {{query}}",
        model="mock-gpt-4o",
    )
    test_cases = [
        TestCase(id="c1", vars={"query": "Reset password"}),
        TestCase(id="c2", vars={"query": "Billing help"}),
    ]

    compressor = PromptCompressor(
        prompt_version=pv,
        test_cases=test_cases,
        target_reduction=0.30,
        force_mock=True,
    )

    res = await compressor.compress()
    assert res.compressed_tokens <= res.original_tokens
    assert res.token_reduction_pct >= 0.0
    assert res.quality_retained_pct > 0.0
    assert res.compressed_prompt is not None
