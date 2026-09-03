"""Tests migrated to feature-focused test suite."""

from __future__ import annotations

from promptdiff.optimizer.prefix_warmup import PrefixCacheOptimizer


def test_prefix_cache_optimizer() -> None:
    """Test prefix cache restructuring for KV-cache reuse."""
    opt = PrefixCacheOptimizer()
    prompt = "Query: {{query}}\n\nYou are an enterprise AI assistant. Always follow policies."
    res = opt.optimize(prompt)
    assert res.estimated_cache_hit_rate_pct >= 80.0
    assert "enterprise AI assistant" in res.static_prefix
