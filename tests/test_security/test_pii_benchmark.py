"""Comprehensive realistic benchmark test suite for PII regex false positive/negative rates."""

from __future__ import annotations

from typing import Any

import pytest

from promptdiff.evaluators.security import detect_pii, evaluate_pii_detector_accuracy

PII_BENCHMARK_CORPUS: list[dict[str, Any]] = [
    # --- POSITIVE SAMPLES (Actual PII / Secrets) ---
    {"text": "Contact our support team at support@company.com for help.", "has_pii": True, "type": "EMAIL"},
    {"text": "Employee personal email: john.doe+billing@subdomain.example.co.uk", "has_pii": True, "type": "EMAIL"},
    {"text": "Send credentials to admin-alert_12@service.org immediately.", "has_pii": True, "type": "EMAIL"},
    {"text": "Call our hotline at +1 (555) 234-5678 during office hours.", "has_pii": True, "type": "PHONE"},
    {"text": "Customer mobile line is 555-876-5432.", "has_pii": True, "type": "PHONE"},
    {"text": "Emergency contact: +1-202-555-0199.", "has_pii": True, "type": "PHONE"},
    {"text": "Applicant verified Social Security Number: 123-45-6789.", "has_pii": True, "type": "SSN"},
    {"text": "Taxpayer SSN registered under 456-78-1234.", "has_pii": True, "type": "SSN"},
    {"text": "Exported secret sk-abc1234567890defghijklmnop to production environment.", "has_pii": True, "type": "API_KEY"},
    {"text": "GitHub personal access token: ghp_1234567890abcdefghijklmnopqrstuvwxyz.", "has_pii": True, "type": "API_KEY"},
    {"text": "AWS root access key ID AKIAIOSFODNN7EXAMPLE detected in config file.", "has_pii": True, "type": "API_KEY"},
    {"text": "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.e30.secret", "has_pii": True, "type": "API_KEY"},
    {"text": "Processed payment card: 4532 0151 1283 0366 via gateway.", "has_pii": True, "type": "CREDIT_CARD"},
    {"text": "Card on file 4532-0151-1283-0366 expires 12/28.", "has_pii": True, "type": "CREDIT_CARD"},

    # --- NEGATIVE SAMPLES (Clean technical text, dates, versions, UUIDs, hashes) ---
    {"text": "Server deployed on 2024-05-12 at 14:30:00 UTC.", "has_pii": False, "type": "DATE"},
    {"text": "Upgraded dependencies to promptdiff v3.4.0 and pytest 8.1.1.", "has_pii": False, "type": "VERSION"},
    {"text": "Database cluster node IP address 192.168.1.100 listening on port 5432.", "has_pii": False, "type": "IP"},
    {"text": "IPv6 loopback configured as ::1 or 2001:0db8:85a3:0000:0000:8a2e:0370:7334.", "has_pii": False, "type": "IPV6"},
    {"text": "Transaction session UUID 123e4567-e89b-12d3-a456-426614174000 created.", "has_pii": False, "type": "UUID"},
    {"text": "Git commit SHA 4b65fb7cdf694586deb2817f0612e5c7068cbbb1 merged to main.", "has_pii": False, "type": "GIT_HASH"},
    {"text": "Product order SKU ORD-99283-X1 with discount code SAVE20.", "has_pii": False, "type": "SKU"},
    {"text": "Style guide background color hex code #FFFFFF with border #334455.", "has_pii": False, "type": "HEX_COLOR"},
    {"text": "Calculated model token latency: 120ms with 4096 context tokens.", "has_pii": False, "type": "METRICS"},
    {"text": "Markdown table row: | Customer_ID | Status | Tier | 100 | Active | Gold |", "has_pii": False, "type": "TABLE"},
    {"text": "Mathematical formula: f(x) = 2x^2 + 5x - 10 where x in [0, 100].", "has_pii": False, "type": "MATH"},
    {"text": "Invalid credit card test numbers 1234 5678 9012 3456 rejected by Luhn.", "has_pii": False, "type": "INVALID_CARD"},
    {"text": "Documentation URL: https://docs.promptdiff.dev/guide/quickstart.html", "has_pii": False, "type": "URL"},
    {"text": "Standard error code ERR_CONNECTION_TIMED_OUT occurred in worker-4.", "has_pii": False, "type": "ERROR_CODE"},
]


def test_pii_detector_benchmark_metrics() -> None:
    """Benchmark false positive rate (FPR) and false negative rate (FNR)."""
    metrics = evaluate_pii_detector_accuracy(PII_BENCHMARK_CORPUS)

    # Statistical assertions
    assert metrics["false_positive_rate"] <= 0.05, f"FPR too high: {metrics['false_positive_rate']}"
    assert metrics["false_negative_rate"] <= 0.05, f"FNR too high: {metrics['false_negative_rate']}"
    assert metrics["precision"] >= 0.95, f"Precision too low: {metrics['precision']}"
    assert metrics["recall"] >= 0.95, f"Recall too low: {metrics['recall']}"
    assert metrics["f1"] >= 0.95, f"F1 score too low: {metrics['f1']}"
    assert metrics["fp"] == 0, f"Unexpected false positives: {metrics['fp']}"
    assert metrics["fn"] == 0, f"Unexpected false negatives: {metrics['fn']}"


@pytest.mark.parametrize(
    ("text", "expected_type"),
    [
        ("Reach out at info@domain.com", "EMAIL"),
        ("Call (555) 123-4567 today", "PHONE_NUMBER"),
        ("Government ID: 123-45-6789", "SSN"),
        ("sk-proj1234567890abcdef123456", "SECRET_OR_API_KEY"),
        ("Visa payment 4532 0151 1283 0366", "CREDIT_CARD"),
    ],
)
def test_pii_detector_individual_categories(text: str, expected_type: str) -> None:
    """Verify exact categorization for each supported PII entity type."""
    findings = detect_pii(text)
    assert len(findings) >= 1
    assert any(f["type"] == expected_type for f in findings)
