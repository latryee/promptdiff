"""Comprehensive coverage tests for pricing.py."""

from __future__ import annotations

from promptdiff.pricing import (
    DEFAULT_PRICE,
    CostForecast,
    ModelPrice,
    calculate_cost,
    calculate_forecast,
    get_model_pricing,
    normalize_model_name,
    parse_volume_string,
)


def test_model_price_properties() -> None:
    p = ModelPrice(10.0, 30.0, "Test model")
    assert p.input_per_million == 10.0
    assert p.output_per_million == 30.0
    assert p.input_per_token == 0.00001
    assert p.output_per_token == 0.00003


def test_normalize_model_name() -> None:
    assert normalize_model_name("  GPT-4o  ") == "gpt-4o"
    assert normalize_model_name("Claude-3-5-Sonnet") == "claude-3-5-sonnet"


def test_get_model_pricing_variations() -> None:
    # Direct match
    assert get_model_pricing("gpt-4o").input_per_million == 2.50

    # Vendor prefix stripping (e.g. openai/gpt-4o)
    assert get_model_pricing("openai/gpt-4o").input_per_million == 2.50

    # Substring match
    assert get_model_pricing("my-custom-gpt-4o-service").input_per_million == 2.50

    # Ollama / local match
    assert get_model_pricing("ollama/llama3").input_per_million == 0.0
    assert get_model_pricing("local-model-v1").input_per_million == 0.0

    # Completely unknown model -> DEFAULT_PRICE
    unknown = get_model_pricing("totally-unknown-xyz-999")
    assert unknown == DEFAULT_PRICE


def test_calculate_cost() -> None:
    # gpt-4o: $2.50 / 1M in, $10.00 / 1M out
    cost = calculate_cost("gpt-4o", prompt_tokens=1000, completion_tokens=500)
    expected = (1000 * 2.50 / 1_000_000) + (500 * 10.00 / 1_000_000)
    assert abs(cost - expected) < 1e-5


def test_parse_volume_string() -> None:
    assert parse_volume_string(1000) == 1000
    assert parse_volume_string("500k") == 500_000
    assert parse_volume_string("2.5M") == 2_500_000
    assert parse_volume_string("1B") == 1_000_000_000
    assert parse_volume_string("1T") == 1_000_000_000_000
    assert parse_volume_string("10,000") == 10_000
    assert parse_volume_string("invalid_vol") == 100_000


def test_calculate_forecast_savings() -> None:
    # v1 costs $0.01 per case, v2 costs $0.005 per case -> 50% savings
    fc = calculate_forecast(
        total_cost_v1=0.10,
        total_cost_v2=0.05,
        total_cases=10,
        daily_volume="10k",
    )
    assert isinstance(fc, CostForecast)
    assert fc.daily_volume == 10_000
    assert fc.monthly_savings_usd > 0
    assert fc.annual_savings_usd > 0
    assert "Projected Savings" in fc.summary_text


def test_calculate_forecast_inflation() -> None:
    # v1 costs $0.005, v2 costs $0.01 -> inflation
    fc = calculate_forecast(
        total_cost_v1=0.05,
        total_cost_v2=0.10,
        total_cases=10,
        daily_volume=5000,
    )
    assert fc.monthly_delta_cost > 0
    assert fc.monthly_savings_usd == 0.0
    assert "Cost Increase" in fc.summary_text
