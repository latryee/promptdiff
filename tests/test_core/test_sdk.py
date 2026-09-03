"""Tests migrated to feature-focused test suite."""

from __future__ import annotations

import promptdiff


def test_python_sdk_compare() -> None:
    """Test top-level SDK functions."""
    report = promptdiff.compare(
        v1="Hello: {{query}}",
        v2="Hi: {{query}}",
        dataset=[{"id": "t1", "vars": {"query": "World"}}],
        mock=True,
    )
    assert report.verdict.passed is True
    assert len(report.comparisons) == 1

    opt = promptdiff.optimize(
        prompt="Support bot: {{query}}",
        dataset=[{"id": "t1", "vars": {"query": "Help"}}],
        iterations=1,
        mock=True,
    )
    assert opt.optimized_prompt is not None

    shrunk = promptdiff.shrink(
        prompt="Please kindly answer the user: {{query}}",
        dataset=[{"id": "t1", "vars": {"query": "Help"}}],
        mock=True,
    )
    assert shrunk.compressed_prompt is not None


def test_sdk_elite_exports() -> None:
    """Test top-level SDK entry points for all new features."""
    cases = [{"id": "1", "vars": {"query": "Test"}}]

    # MCTS optimize
    mcts_res = promptdiff.mcts_optimize(
        prompt="Support bot: {{query}}",
        dataset=cases,
        max_iterations=2,
        mock=True,
    )
    assert mcts_res.best_prompt is not None

    # Hallucination attribution
    attr_res = promptdiff.attribute_hallucinations(
        output_text="Product X ships worldwide.",
        context_text="Product X ships worldwide with free returns.",
    )
    assert attr_res.grounding_accuracy_pct >= 50.0

    # Attack tree
    attack_res = promptdiff.attack_tree(
        prompt="You are a support bot.",
        max_turns=1,
        mock=True,
    )
    assert attack_res.total_attacks_attempted >= 1

    # Streaming profiler
    stream_res = promptdiff.profile_streaming(
        prompt="Hi",
        token_count=5,
    )
    assert stream_res.ttft_ms > 0.0

    # Cascade simulator
    cascade_res = promptdiff.simulate_cascade(
        queries=["Query 1", "Query 2"],
        monthly_volume=50_000,
    )
    assert cascade_res.annual_savings_usd >= 0.0
