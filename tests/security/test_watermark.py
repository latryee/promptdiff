"""Tests migrated to feature-focused test suite."""

from __future__ import annotations

import pytest

from promptdiff.security.stego_detector import StatisticalWatermarkDetector
from promptdiff.security.watermark import PromptWatermarker


def test_prompt_watermark() -> None:
    """Test cryptographic prompt watermarking with explicit secret key and grounded confidence."""
    # Must raise ValueError if secret_key is not provided
    with pytest.raises(ValueError, match="secret_key must be explicitly provided"):
        PromptWatermarker(secret_key=None)

    wm = PromptWatermarker(secret_key="my-secure-eval-key", organization="Acme AI Corp")
    orig_prompt = "You are an enterprise AI assistant. Always format responses in Markdown."
    watermarked = wm.inject_watermark(orig_prompt)

    assert watermarked != orig_prompt  # Contains zero-width invisible characters

    inspection = wm.inspect_text_for_watermark(watermarked)
    assert inspection.is_watermarked is True
    assert inspection.matched_organization == "Acme AI Corp"
    assert inspection.confidence_pct == 100.0

    # Non-watermarked check
    unmarked_inspection = wm.inspect_text_for_watermark("Regular plain text without watermarks.")
    assert unmarked_inspection.is_watermarked is False
    assert unmarked_inspection.confidence_pct == 0.0


def test_watermark_stego_detector() -> None:
    """Test Kirchenbauer watermark statistical z-score detector."""
    detector = StatisticalWatermarkDetector()
    text = "This is a clean natural language paragraph evaluated for statistical green token distribution testing."
    rep = detector.test_text(text)
    assert rep.text_length_tokens >= 10
    assert rep.p_value >= 0.0
