"""Tests migrated to feature-focused test suite."""

from __future__ import annotations

import pytest

from promptdiff.core.models import TestCase
from promptdiff.optimizer.mcts import (
    MCTSNode,
    MCTSPromptOptimizer,
    ParetoMetrics,
)


@pytest.mark.asyncio
async def test_mcts_prompt_optimizer() -> None:
    """Test MCTS prompt exploration, UCB1 selection, and Pareto frontier."""
    m1 = ParetoMetrics(quality_score=0.90, latency_ms=150.0, cost_usd=0.001, token_count=50)
    m2 = ParetoMetrics(quality_score=0.80, latency_ms=200.0, cost_usd=0.002, token_count=80)
    assert m1.dominates(m2) is True
    assert m2.dominates(m1) is False

    node_root = MCTSNode(prompt_template="Answer: {{query}}", visits=4, total_reward=3.2)
    node_child = MCTSNode(
        prompt_template="Answer step by step: {{query}}", parent=node_root, visits=2, total_reward=1.8
    )
    assert node_child.ucb1() > 0.0

    # Test complete optimization cycle
    test_cases = [
        TestCase(id="t1", vars={"query": "Calculate 2+2"}),
        TestCase(id="t2", vars={"query": "Explain gravity"}),
    ]
    optimizer = MCTSPromptOptimizer(
        initial_prompt="You are a helpful assistant. Answer {{query}}",
        test_cases=test_cases,
        max_iterations=3,
        force_mock=True,
    )
    result = await optimizer.optimize()

    assert result.best_prompt is not None
    assert result.nodes_explored >= 3
    assert len(result.pareto_frontier) >= 1
    assert "MCTS Tree Root" in result.tree_ascii


def test_mcts_iteration_guard_limits() -> None:
    """Verify that MCTS rejects non-positive or excessive iterations exceeding MAX_ALLOWED_ITERATIONS."""
    from promptdiff.optimizer.mcts import MAX_ALLOWED_ITERATIONS, validate_mcts_iterations

    assert MAX_ALLOWED_ITERATIONS == 500

    # Valid iterations
    assert validate_mcts_iterations(1) == 1
    assert validate_mcts_iterations(500) == 500

    # Non-positive iterations
    with pytest.raises(ValueError, match="positive integer"):
        validate_mcts_iterations(0)

    with pytest.raises(ValueError, match="positive integer"):
        validate_mcts_iterations(-10)

    # Exceeding MAX_ALLOWED_ITERATIONS
    with pytest.raises(ValueError, match="exceeds MAX_ALLOWED_ITERATIONS"):
        validate_mcts_iterations(501)

    with pytest.raises(ValueError, match="exceeds MAX_ALLOWED_ITERATIONS"):
        validate_mcts_iterations(1000)

    # In MCTSPromptOptimizer constructor
    test_cases = [TestCase(id="t1", vars={"query": "Test"})]
    with pytest.raises(ValueError, match="exceeds MAX_ALLOWED_ITERATIONS"):
        MCTSPromptOptimizer(
            initial_prompt="Prompt: {{query}}",
            test_cases=test_cases,
            num_iterations=999,
        )

    with pytest.raises(ValueError, match="positive integer"):
        MCTSPromptOptimizer(
            initial_prompt="Prompt: {{query}}",
            test_cases=test_cases,
            max_iterations=0,
        )
