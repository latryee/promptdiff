"""Tests migrated to feature-focused test suite."""

from __future__ import annotations

from promptdiff.core.models import PromptVersion
from promptdiff.optimizer.cache_sim import PromptCacheSimulator


def test_prompt_cache_simulator() -> None:
    """Test LLM prompt prefix caching simulator."""
    pv = PromptVersion(
        name="cache_test",
        template="User Query: {{query}}\n\nYou are a customer support agent. Obey rules 1 to 10.\nJSON Schema: ...",
        model="claude-3-5-sonnet",
    )

    sim = PromptCacheSimulator(prompt_version=pv, model_name="claude-3-5-sonnet", daily_volume=100_000)
    rep = sim.analyze_and_optimize()

    assert rep.optimized_cache_hit_rate_pct > rep.original_cache_hit_rate_pct
    assert rep.prefix_tokens_cached > 0
    assert "USER REQUEST INPUTS" in rep.optimized_template
