"""Real-Time Pre-Execution Input Guardrail & Sanitizer Shield.

Inspects incoming prompt variables and user queries prior to LLM dispatch:
- Strips zero-width unicode steganography characters (\u200b, \u200c, \u200d, \ufeff).
- Decodes hidden base64 smuggling blocks to uncover obfuscated injection payloads.
- Defangs malicious Markdown image exfiltration URLs.
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass


@dataclass
class SanitizationResult:
    """Outcome of input security defense screening."""

    sanitized_text: str
    threats_neutralized: list[str]
    was_modified: bool
    risk_blocked: bool


class InputDefenseShield:
    """Pre-inference security firewall for LLM applications."""

    def sanitize(self, input_text: str) -> SanitizationResult:
        """Strip invisible steganography and defang injection smuggling."""
        threats = []
        cleaned = input_text
        blocked = False

        # 1. Zero-width unicode characters
        zw_pattern = re.compile(r"[\u200b\u200c\u200d\u200e\u200f\ufeff]")
        if zw_pattern.search(cleaned):
            cleaned = zw_pattern.sub("", cleaned)
            threats.append("Neutralized Zero-Width Unicode Steganography payload.")

        # 2. Base64 payload smuggling
        b64_pattern = re.compile(r"`([A-Za-z0-9+/]{20,}={0,2})`")
        for match in b64_pattern.finditer(cleaned):
            raw_b64 = match.group(1)
            try:
                decoded = base64.b64decode(raw_b64).decode("utf-8", errors="ignore")
                if any(kw in decoded.lower() for kw in ["ignore all", "system prompt", "override"]):
                    threats.append(f"Blocked Base64 Smuggled Prompt Injection: '{decoded[:40]}...'")
                    blocked = True
            except Exception:
                pass

        # 3. Markdown exfiltration defanging
        exfil_pattern = re.compile(r"!\[.*?\]\((https?://[^\)]+)\)")
        if exfil_pattern.search(cleaned):
            cleaned = exfil_pattern.sub("[Image URL Removed by Defense Shield]", cleaned)
            threats.append("Defanged Markdown External Image Exfiltration URL.")

        return SanitizationResult(
            sanitized_text=cleaned,
            threats_neutralized=threats,
            was_modified=cleaned != input_text,
            risk_blocked=blocked,
        )
