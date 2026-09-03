"""Tests migrated to feature-focused test suite."""

from __future__ import annotations

import pytest

from promptdiff.core.models import PromptVersion
from promptdiff.evaluators.haystack import NeedleInAHaystackTester
from promptdiff.evaluators.needle_matrix import NeedleMatrixEvaluator


@pytest.mark.asyncio
async def test_haystack_needle_tester() -> None:
    """Test needle in a haystack context degradation."""
    pv = PromptVersion(name="haystack_target", template="Context: {{context}}\n\nQuery: {{query}}")
    tester = NeedleInAHaystackTester(
        prompt_version=pv, context_lengths=[1000], depth_percentages=[0, 100], force_mock=True
    )
    rep = await tester.run_haystack_test()
    assert rep.total_test_points == 2
    assert rep.accuracy_pct >= 50.0


def test_needle_matrix_benchmark() -> None:
    """Test 2D Needle-in-a-Haystack retrieval matrix."""
    evaluator = NeedleMatrixEvaluator()
    rep = evaluator.benchmark_mock()
    assert rep.overall_retrieval_rate_pct > 0.0
    ascii_grid = rep.render_ascii_matrix()
    assert "Needle Retrieval Matrix" in ascii_grid
