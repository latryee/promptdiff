"""Tests migrated to feature-focused test suite."""

from __future__ import annotations

import pytest

from promptdiff.core.models import PromptVersion, TestCase
from promptdiff.optimizer.tuner import (
    HyperparameterConfig,
    PromptTuner,
    TuneCandidateResult,
    compute_pareto_frontier,
)
from promptdiff.providers.mock_provider import MockProvider


def test_pareto_frontier_computation() -> None:
    """Test multi-objective Pareto dominance calculation."""
    c1 = TuneCandidateResult(
        config=HyperparameterConfig(0.0, 0.7),
        avg_judge_score=4.8,
        avg_latency_ms=150.0,
        avg_tokens=40.0,
        total_cost=0.0001,
        passed_rate=1.0,
    )
    c2 = TuneCandidateResult(
        config=HyperparameterConfig(1.0, 1.0),
        avg_judge_score=3.2,
        avg_latency_ms=300.0,
        avg_tokens=80.0,
        total_cost=0.0003,
        passed_rate=0.5,
    )

    pareto = compute_pareto_frontier([c1, c2])
    assert len(pareto) == 1
    assert pareto[0].config.temperature == 0.0
    assert c1.is_pareto_optimal is True
    assert c2.is_pareto_optimal is False
    assert c1.rank == 1


@pytest.mark.asyncio
async def test_prompt_tuner_grid_search() -> None:
    """Test hyperparameter tuning grid search with mock provider."""
    provider = MockProvider(force_mock=True)

    pv = PromptVersion(
        name="support_bot",
        template="Answer concisely: {{query}}",
        model="mock-gpt-4o",
    )

    test_cases = [
        TestCase(id="t1", vars={"query": "Hello"}),
        TestCase(id="t2", vars={"query": "Refund please"}),
    ]

    tuner = PromptTuner(
        prompt_version=pv,
        test_cases=test_cases,
        provider=provider,
        temperatures=[0.0, 0.7],
        top_ps=[0.7, 1.0],
        force_mock=True,
    )

    report = await tuner.tune()
    assert report.total_configs_tested == 4
    assert len(report.all_results) == 4
    assert len(report.pareto_candidates) >= 1
    assert report.best_config is not None
