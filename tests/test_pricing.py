"""Unit tests for Model Pricing & Token Cost Calculations."""

from promptdiff.pricing import calculate_cost, get_model_pricing, normalize_model_name


def test_normalize_model_name():
    assert normalize_model_name("  GPT-4o ") == "gpt-4o"
    assert normalize_model_name("Claude-3-5-Sonnet") == "claude-3-5-sonnet"


def test_get_model_pricing_exact_and_fuzzy():
    # Exact match
    p_gpt4o = get_model_pricing("gpt-4o")
    assert p_gpt4o.input_per_million == 2.50
    assert p_gpt4o.output_per_million == 10.00

    # Prefix/vendor stripped match
    p_vendor = get_model_pricing("openai/gpt-4o")
    assert p_vendor.input_per_million == 2.50

    # Claude 3.5 Sonnet
    p_claude = get_model_pricing("claude-3-5-sonnet-latest")
    assert p_claude.input_per_million == 3.00
    assert p_claude.output_per_million == 15.00

    # Gemini 2.0 Flash
    p_gemini = get_model_pricing("gemini-2.0-flash")
    assert p_gemini.input_per_million == 0.10

    # Unknown fallback
    p_unknown = get_model_pricing("some-custom-fine-tuned-model")
    assert p_unknown.input_per_million > 0


def test_calculate_cost():
    # 1000 input tokens and 500 output tokens on gpt-4o
    # 1000 * ($2.50 / 1M) = 0.0025
    # 500 * ($10.00 / 1M) = 0.0050
    # Total = 0.0075
    cost = calculate_cost("gpt-4o", prompt_tokens=1000, completion_tokens=500)
    assert cost == 0.0075

    # Zero tokens = 0.0
    assert calculate_cost("gpt-4o", 0, 0) == 0.0
