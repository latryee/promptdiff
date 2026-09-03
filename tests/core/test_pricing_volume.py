"""Tests migrated to feature-focused test suite."""

from __future__ import annotations

from promptdiff.optimizer.compressor import estimate_tokens
from promptdiff.pricing import calculate_forecast, parse_volume_string


def test_volume_parsing() -> None:
    """Test string volume parsing."""
    assert parse_volume_string("1M") == 1_000_000
    assert parse_volume_string("500k") == 500_000
    assert parse_volume_string("2.5M") == 2_500_000
    assert parse_volume_string("100,000") == 100_000
    assert parse_volume_string(10_000) == 10_000


def test_cost_forecasting() -> None:
    """Test cost projection over daily, monthly, and annual volume."""
    fc = calculate_forecast(
        total_cost_v1=0.0004,
        total_cost_v2=0.0002,
        total_cases=4,
        daily_volume="1M",
    )

    assert fc.daily_volume == 1_000_000
    assert fc.monthly_volume == 30_000_000
    assert fc.v1_monthly_cost == 3000.0
    assert fc.v2_monthly_cost == 1500.0
    assert fc.monthly_savings_usd == 1500.0
    assert fc.annual_savings_usd == 18000.0
    assert fc.cost_delta_pct == -50.0
    assert "Projected Savings: $1,500.00/mo" in fc.summary_text


def test_token_estimation() -> None:
    """Test heuristic token estimation."""
    text = "You are a helpful customer support agent."
    tokens = estimate_tokens(text)
    assert tokens > 0
    assert tokens >= 5
