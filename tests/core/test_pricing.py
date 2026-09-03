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


def test_pricing_sync_script_diff_and_formatting() -> None:
    """Test pricing sync diff computation and table rendering."""
    from promptdiff.pricing import ModelPrice
    from scripts.sync_pricing import compute_pricing_diff, format_diff_table

    local_table = {
        "gpt-4o": ModelPrice(2.50, 10.00, "Local GPT-4o"),
        "gemini-2.0-flash": ModelPrice(0.10, 0.40, "Local Gemini"),
    }
    remote_data = {
        "gpt-4o": {"input_cost_per_token": 0.000003, "output_cost_per_token": 0.000012},  # $3.00 / $12.00
        "gemini-2.0-flash": {"input_cost_per_token": 0.0000001, "output_cost_per_token": 0.0000004},  # $0.10 / $0.40
    }

    diffs = compute_pricing_diff(local_table, remote_data)
    assert len(diffs) == 1
    assert diffs[0].model == "gpt-4o"
    assert diffs[0].remote_input == 3.00
    assert diffs[0].status == "MODIFIED"

    formatted = format_diff_table(diffs)
    assert "gpt-4o" in formatted
    assert "MODIFIED" in formatted

    # Test empty diff message
    empty_msg = format_diff_table([])
    assert "in sync" in empty_msg


def test_estimate_tokens_regex_fallback(caplog) -> None:
    """Test token estimation falling back to regex word count when tiktoken is unavailable."""
    import logging
    import sys
    from unittest.mock import patch

    import promptdiff.pricing as pricing

    # Reset warned flag to test warning output
    pricing._TIKTOKEN_WARNED = False

    with patch.dict(sys.modules, {"tiktoken": None}):
        with caplog.at_level(logging.WARNING):
            tok_count = pricing.estimate_tokens("Hello world this is an estimation test.")
            assert tok_count > 0
            assert any("tiktoken is not installed" in rec.message for rec in caplog.records)


def test_estimate_tokens_with_tiktoken() -> None:
    """Test token estimation when tiktoken is available."""
    import sys
    from unittest.mock import MagicMock, patch

    import promptdiff.pricing as pricing

    mock_enc = MagicMock()
    mock_enc.encode.return_value = [100, 200, 300, 400]
    mock_tiktoken = MagicMock()
    mock_tiktoken.encoding_for_model.return_value = mock_enc

    with patch.dict(sys.modules, {"tiktoken": mock_tiktoken}):
        tok_count = pricing.estimate_tokens("Mocked token text", model_name="gpt-4o")
        assert tok_count == 4
        mock_tiktoken.encoding_for_model.assert_called_with("gpt-4o")


def test_calculate_text_cost() -> None:
    """Test calculate_text_cost using prompt and completion strings."""
    from promptdiff.pricing import calculate_text_cost

    cost = calculate_text_cost("gpt-4o", prompt_text="Short prompt query", completion_text="Short response output")
    assert cost > 0.0
