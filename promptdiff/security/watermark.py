"""Prompt Watermarking & Cryptographic Signature Verification for promptdiff."""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("promptdiff.security.watermark")

# 4 distinct invisible zero-width unicode characters (base-4 encoding: 2 chars per hex nibble)
ZERO_WIDTH_CHARS = ["\u200b", "\u200c", "\u200d", "\u2060"]
ZW_TO_INDEX = {char: idx for idx, char in enumerate(ZERO_WIDTH_CHARS)}


def _encode_hex_to_zw(hex_str: str) -> str:
    """Encode a hex string into a deterministic sequence of zero-width characters."""
    zw = []
    for char in hex_str.lower():
        val = int(char, 16)
        high = val // 4
        low = val % 4
        zw.append(ZERO_WIDTH_CHARS[high])
        zw.append(ZERO_WIDTH_CHARS[low])
    return "".join(zw)


def _decode_zw_to_hex(zw_str: str) -> str:
    """Decode a zero-width character sequence back into a hex string."""
    hex_chars = []
    # Process pairs of zero-width characters
    for i in range(0, len(zw_str) - 1, 2):
        c1, c2 = zw_str[i], zw_str[i + 1]
        if c1 in ZW_TO_INDEX and c2 in ZW_TO_INDEX:
            val = ZW_TO_INDEX[c1] * 4 + ZW_TO_INDEX[c2]
            hex_chars.append(f"{val:x}")
    return "".join(hex_chars)


def strip_watermark(text: str) -> tuple[str, str]:
    """Separate clean visible text from invisible zero-width characters."""
    extracted_zw = "".join(c for c in text if c in ZW_TO_INDEX)
    clean_text = "".join(c for c in text if c not in ZW_TO_INDEX)
    return clean_text, extracted_zw


@dataclass
class WatermarkInspectionResult:
    """Outcome of checking an extracted or candidate prompt for cryptographic watermarks."""

    is_watermarked: bool
    confidence_pct: float
    detected_signature: Optional[str]
    matched_organization: Optional[str]
    evidence_snippet: str


class PromptWatermarker:
    """Embeds and verifies invisible zero-entropy cryptographic watermarks in prompt templates."""

    def __init__(self, secret_key: Optional[str] = None, organization: str = "PromptDiff Organization"):
        key = secret_key or os.getenv("PROMPTDIFF_WATERMARK_KEY")
        if not key:
            raise ValueError(
                "A secret_key must be explicitly provided or set via the PROMPTDIFF_WATERMARK_KEY "
                "environment variable. Refusing to use insecure hardcoded default keys."
            )
        self.secret_key = key
        self.organization = organization

    def _generate_signature(self, prompt_text: str) -> str:
        """Compute HMAC-SHA256 signature prefix for the clean text."""
        clean_text, _ = strip_watermark(prompt_text)
        h = hmac.new(self.secret_key.encode("utf-8"), clean_text.encode("utf-8"), hashlib.sha256)
        return h.hexdigest()[:8]

    def inject_watermark(self, prompt_text: str) -> str:
        """Inject invisible zero-width HMAC fingerprint into prompt template."""
        clean_text, _ = strip_watermark(prompt_text)
        sig = self._generate_signature(clean_text)
        zw_sequence = _encode_hex_to_zw(sig)

        lines = clean_text.split("\n")
        if lines:
            lines[0] = lines[0] + zw_sequence
        return "\n".join(lines)

    def inspect_text_for_watermark(self, candidate_text: str) -> WatermarkInspectionResult:
        """Scan candidate text for invisible zero-width watermark and verify signature."""
        clean_text, extracted_zw = strip_watermark(candidate_text)

        if not extracted_zw:
            return WatermarkInspectionResult(
                is_watermarked=False,
                confidence_pct=0.0,
                detected_signature=None,
                matched_organization=None,
                evidence_snippet="No cryptographic or zero-width watermark fingerprints detected.",
            )

        detected_sig = _decode_zw_to_hex(extracted_zw)
        expected_sig = self._generate_signature(clean_text)

        if detected_sig and expected_sig:
            sig_len = min(len(detected_sig), len(expected_sig))
            matched_chars = sum(
                1 for a, b in zip(detected_sig[:sig_len], expected_sig[:sig_len], strict=False) if a == b
            )
            confidence_pct = round((matched_chars / len(expected_sig)) * 100.0, 1)
            is_match = detected_sig == expected_sig

            return WatermarkInspectionResult(
                is_watermarked=is_match,
                confidence_pct=confidence_pct,
                detected_signature=detected_sig if is_match else f"MISMATCH-{detected_sig}",
                matched_organization=self.organization if is_match else None,
                evidence_snippet=(
                    f"Signature verified ({confidence_pct}% bit match against HMAC-SHA256 key)."
                    if is_match
                    else f"Detected {len(extracted_zw)} zero-width characters with mismatched signature."
                ),
            )

        return WatermarkInspectionResult(
            is_watermarked=False,
            confidence_pct=0.0,
            detected_signature=None,
            matched_organization=None,
            evidence_snippet="Malformed or incomplete zero-width sequence detected.",
        )
