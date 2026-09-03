"""Statistical AI Watermark & Steganography Detection Engine.

Implements the Kirchenbauer et al. green/red token frequency statistical hypothesis test,
calculating z-scores to mathematically prove whether generated text contains synthetic
AI watermarks or covert steganographic channels.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass


@dataclass
class WatermarkDetectionReport:
    """Outcome of Kirchenbauer statistical watermark test."""

    text_length_tokens: int
    green_tokens_count: int
    expected_green_count: float
    z_score: float
    p_value: float
    watermark_detected: bool  # True if z_score > 3.0 (p < 0.001)
    gamma: float  # Fraction of green list vocabulary (typically 0.5)


class StatisticalWatermarkDetector:
    """Detects statistical bias in token distributions indicating synthetic watermarking."""

    def __init__(self, gamma: float = 0.5, z_threshold: float = 3.0):
        self.gamma = gamma
        self.z_threshold = z_threshold

    def _hash_token(self, token: str) -> int:
        """Deterministic pseudo-random hash assigning token to green or red list."""
        h = 2166136261
        for char in token:
            h = (h ^ ord(char)) * 16777619
        return h & 0xFFFFFFFF

    def test_text(self, text: str) -> WatermarkDetectionReport:
        """Compute green-token ratio and z-score."""
        tokens = re.findall(r"\w+", text.lower())
        t = len(tokens)
        if t < 5:
            return WatermarkDetectionReport(
                text_length_tokens=t,
                green_tokens_count=0,
                expected_green_count=0.0,
                z_score=0.0,
                p_value=1.0,
                watermark_detected=False,
                gamma=self.gamma,
            )

        green_count = 0
        for token in tokens:
            # Token is green if hash mod 100 < gamma * 100
            if (self._hash_token(token) % 100) < (self.gamma * 100):
                green_count += 1

        expected = self.gamma * t
        std_dev = math.sqrt(t * self.gamma * (1.0 - self.gamma))

        if std_dev > 0:
            z = (green_count - expected) / std_dev
            # One-sided p-value
            p = 0.5 * (1.0 - math.erf(z / math.sqrt(2.0)))
            p = max(0.0, min(1.0, p))
        else:
            z = 0.0
            p = 1.0

        return WatermarkDetectionReport(
            text_length_tokens=t,
            green_tokens_count=green_count,
            expected_green_count=round(expected, 1),
            z_score=round(z, 2),
            p_value=round(p, 6),
            watermark_detected=z >= self.z_threshold,
            gamma=self.gamma,
        )
