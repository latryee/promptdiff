"""Tests migrated to feature-focused test suite."""

from __future__ import annotations

import pytest

from promptdiff.core.models import PromptVersion, TestCase
from promptdiff.optimizer.mutation_tester import MutationTestingEngine


@pytest.mark.asyncio
async def test_mutation_testing_engine() -> None:
    """Test mutation testing engine."""
    pv = PromptVersion(name="orig", template="Answer the query in strict JSON format: {{query}}")
    engine = MutationTestingEngine(
        original_prompt=pv, test_cases=[TestCase(id="1", vars={"query": "test"})], force_mock=True
    )
    rep = await engine.run_mutation_analysis()
    assert rep.total_mutants_generated > 0
    assert rep.mutation_score_pct >= 50.0
