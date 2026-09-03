"""Unit tests for KV-cache prefix divergence and financial impact analyzer."""

from promptdiff.optimizer.cache_impact import PromptCacheImpactAnalyzer, analyze_cache_impact


def test_cache_impact_below_threshold() -> None:
    """Prompts below minimum vendor threshold should report cache not active."""
    v1 = "You are a helpful customer support bot. Query: {{query}}"
    v2 = "You are a concise customer support bot. Query: {{query}}"

    analyzer = PromptCacheImpactAnalyzer(model_name="claude-3-5-sonnet")
    res = analyzer.analyze(v1, v2, daily_requests=10_000)

    assert res.v1_tokens < 1024
    assert res.v1_cache_eligible is False
    assert res.v2_cache_eligible is False
    assert res.cache_preserved is False
    assert res.lost_cached_tokens == 0
    assert "below" in res.recommendation.lower()


def test_cache_impact_above_threshold_preserved() -> None:
    """Shared prefix >= 1024 tokens should report cache preserved with zero lost tokens."""
    # Build a 1200-token static prefix
    long_prefix = "Directive instruction context rule requirement specification item. " * 150
    v1 = long_prefix + "\n\nOriginal dynamic query: {{query}}"
    v2 = long_prefix + "\n\nUpdated candidate dynamic query: {{query}}"

    res = analyze_cache_impact(v1, v2, model="claude-3-5-sonnet", daily_volume=50_000)

    assert res.v1_tokens >= 1024
    assert res.common_prefix_tokens >= 1024
    assert res.cache_preserved is True
    assert res.lost_cached_tokens == 0
    assert res.monthly_financial_impact_usd == 0.0
    assert "optimal" in res.recommendation.lower()


def test_cache_impact_invalidation_and_cost() -> None:
    """When divergence occurs before 1024 tokens on an eligible prompt, report invalidation and cash loss."""
    static_filler = "Standard enterprise governance compliance guideline note. " * 160
    # v1 has static filler first (1200 tokens)
    v1 = static_filler + "\n\nQuery: {{query}}"
    # v2 accidentally puts dynamic greeting at the very beginning, breaking prefix cache
    v2 = "Hello user! " + static_filler + "\n\nQuery: {{query}}"

    res = analyze_cache_impact(v1, v2, model="claude-3-5-sonnet", daily_volume=10_000)

    assert res.v1_cache_eligible is True
    assert res.cache_preserved is False
    assert res.lost_cached_tokens > 0
    assert res.cost_delta_per_request_usd > 0.0
    assert res.monthly_financial_impact_usd > 0.0
    assert res.breakpoint_token_idx < 5  # Broke immediately


def test_cache_impact_vendor_thresholds() -> None:
    """Verify vendor-specific thresholds (DeepSeek 64 tokens, Claude 1024 tokens)."""
    deepseek_analyzer = PromptCacheImpactAnalyzer(model_name="deepseek-chat")
    assert deepseek_analyzer.min_cache_tokens == 64

    claude_analyzer = PromptCacheImpactAnalyzer(model_name="claude-3-5-sonnet")
    assert claude_analyzer.min_cache_tokens == 1024


def test_cache_impact_cli() -> None:
    """Test CLI command 'promptdiff cache-impact'."""
    from typer.testing import CliRunner

    from promptdiff.cli.app import app

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "cache-impact",
            "System prompt v1: {{query}}",
            "System prompt v2: {{query}}",
            "--model",
            "claude-3-5-sonnet",
            "--volume",
            "5000",
        ],
    )
    assert res.exit_code == 0
    assert "KV-Cache Prefix Caching Impact" in res.output
    assert "claude-3-5-sonnet" in res.output
