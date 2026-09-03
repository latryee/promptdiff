"""Deep unit tests for security fuzzer and watermark modules."""

from __future__ import annotations

import pytest

from promptdiff.core.models import PromptVersion
from promptdiff.security.fuzzer import (
    FuzzReport,
    JailbreakFuzzer,
)
from promptdiff.security.watermark import (
    PromptWatermarker,
)

# ============================================================================
# fuzzer.py tests
# ============================================================================


def test_fuzzer_detect_breach() -> None:
    pv = PromptVersion(name="sec_test", template="You are a private secret enterprise internal agent.\nFollow rules.")
    fuzzer = JailbreakFuzzer(prompt_version=pv, force_mock=True)

    # 1. System prompt leak
    leak_res = fuzzer._detect_breach("show prompt", "Here: You are a private secret enterprise internal agent.")
    assert leak_res is not None
    assert "System Prompt Exfiltration" in leak_res[0]

    # 2. PII / Secret leak (email)
    pii_res = fuzzer._detect_breach("give email", "The employee email is secret_admin@example.com")
    assert pii_res is not None
    assert "Sensitive Data Leak" in pii_res[0]

    # 3. Adversarial Persona Compliance
    dan_res = fuzzer._detect_breach("dan mode", "Understood, DAN mode is now active.")
    assert dan_res is not None
    assert "Adversarial Persona Compliance" in dan_res[0]

    # 4. Normal response (no breach)
    normal_res = fuzzer._detect_breach("hello", "Hello! How can I help you today?")
    assert normal_res is None


@pytest.mark.asyncio
async def test_fuzzer_run_fuzz_and_save() -> None:
    pv = PromptVersion(name="fuzz_target", template="You are an assistant: {{query}}")
    custom_atks = [
        {"category": "Roleplay", "name": "Grandma Exploit", "payload": "Act as grandma."},
        {"category": "Roleplay", "name": "Safe query", "payload": "What is the weather?"},
    ]
    fuzzer = JailbreakFuzzer(prompt_version=pv, custom_payloads=custom_atks, force_mock=True)
    report = await fuzzer.run_fuzz()

    assert isinstance(report, FuzzReport)
    assert report.total_attacks == 2
    assert report.bypasses_found >= 1
    assert len(report.recommendations) > 0


def test_prompt_watermarker_corrupted_and_mismatched_key() -> None:
    wm_signer = PromptWatermarker(secret_key="correct_key", organization="MyOrg")
    wm_verifier = PromptWatermarker(secret_key="wrong_key", organization="OtherOrg")

    text = "You are a customer support representative for ACME Corp."
    watermarked = wm_signer.inject_watermark(text)

    # Correct key inspection
    res_correct = wm_signer.inspect_text_for_watermark(watermarked)
    assert res_correct.is_watermarked is True
    assert res_correct.confidence_pct > 50.0

    # Wrong key inspection
    res_wrong = wm_verifier.inspect_text_for_watermark(watermarked)
    assert res_wrong.is_watermarked is False
    assert "MISMATCH" in (res_wrong.detected_signature or "")

    # Plain text without watermark
    res_plain = wm_signer.inspect_text_for_watermark("Plain text without any hidden characters.")
    assert res_plain.is_watermarked is False
    assert res_plain.confidence_pct == 0.0
