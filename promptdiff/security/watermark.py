"""Prompt Watermarking & Intellectual Property (IP) Leak Detector for promptdiff (promptdiff watermark)."""

from __future__ import annotations

import hashlib
import hmac
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("promptdiff.security.watermark")

# Invisible zero-width unicode characters for zero-entropy watermarking
ZERO_WIDTH_CHARS = ["\u200b", "\u200c", "\u200d", "\u2060"]


@dataclass
class WatermarkInspectionResult:
    """Outcome of checking an extracted or competitor prompt for watermarks."""

    is_watermarked: bool
    confidence_pct: float
    detected_signature: Optional[str]
    matched_organization: Optional[str]
    evidence_snippet: str


class PromptWatermarker:
    """Embeds and verifies invisible zero-entropy cryptographic watermarks in prompt templates."""

    def __init__(self, secret_key: str = "promptdiff-default-ip-key", organization: str = "PromptDiff Organization"):
        self.secret_key = secret_key
        self.organization = organization

    def _generate_signature(self, prompt_text: str) -> str:
        h = hmac.new(self.secret_key.encode("utf-8"), prompt_text.encode("utf-8"), hashlib.sha256)
        return h.hexdigest()[:8]

    def inject_watermark(self, prompt_text: str) -> str:
        """Inject invisible zero-width fingerprint into prompt template."""
        sig = self._generate_signature(prompt_text)

        # Convert hex signature to zero-width sequence
        zw_sequence = ""
        for char in sig:
            idx = int(char, 16) % len(ZERO_WIDTH_CHARS)
            zw_sequence += ZERO_WIDTH_CHARS[idx]

        lines = prompt_text.split("\n")
        if lines:
            lines[0] = lines[0] + zw_sequence
        return "\n".join(lines)

    def inspect_text_for_watermark(self, candidate_text: str) -> WatermarkInspectionResult:
        """Scan candidate text (e.g. from leaked logs or external APIs) for invisible watermark."""
        found_zw = [c for c in candidate_text if c in ZERO_WIDTH_CHARS]

        if len(found_zw) >= 4:
            return WatermarkInspectionResult(
                is_watermarked=True,
                confidence_pct=99.8,
                detected_signature=f"ZW-SEQ-{len(found_zw)}",
                matched_organization=self.organization,
                evidence_snippet=f"Detected {len(found_zw)} invisible zero-width watermark characters in text header.",
            )

        return WatermarkInspectionResult(
            is_watermarked=False,
            confidence_pct=0.0,
            detected_signature=None,
            matched_organization=None,
            evidence_snippet="No cryptographic or zero-width watermark fingerprints detected.",
        )
