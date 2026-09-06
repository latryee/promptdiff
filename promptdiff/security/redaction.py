"""Automated Redaction Engine for Secrets, Tokens, and Sensitive PII.

Protects sensitive credentials, API keys, Bearer tokens, passwords, and PII
from leaking into logs, artifacts, diff reports, and CI/CD outputs.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from promptdiff.core.models import (
    ComparisonResult,
    DiffChunk,
    DiffReport,
)

# Compiled regex patterns for high-throughput redaction
_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # API Keys
    (re.compile(r"\bsk-ant-[a-zA-Z0-9_-]{20,}\b"), "[ANTHROPIC_KEY_REDACTED]"),
    (re.compile(r"\bsk-[a-zA-Z0-9_-]{20,}\b"), "[API_KEY_REDACTED]"),
    (re.compile(r"\bAIza[0-9A-Za-z-_]{35}\b"), "[GOOGLE_KEY_REDACTED]"),
    (re.compile(r"\bhf_[a-zA-Z0-9]{34,}\b"), "[HF_TOKEN_REDACTED]"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[AWS_KEY_REDACTED]"),
    # Bearer & JWT tokens
    (re.compile(r"(?i)\bBearer\s+[a-zA-Z0-9_\-\.]{20,}\b"), "Bearer [TOKEN_REDACTED]"),
    (re.compile(r"\beyJ[a-zA-Z0-9_-]+\.eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\b"), "[JWT_REDACTED]"),
    # PII
    (re.compile(r"\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b"), "[EMAIL_REDACTED]"),
    (re.compile(r"\b(?:\d{4}[ -]?){3}\d{4}\b"), "[CREDIT_CARD_REDACTED]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN_REDACTED]"),
    # Key-value secret assignments in text: api_key = "..." or password: '...'
    (
        re.compile(
            r"(?i)(api[_-]?key|client[_-]?secret|access[_-]?token|auth[_-]?token|password|passwd|private[_-]?key)\s*[:=]\s*([\"']?)([^\"'\s,;]+)\2"
        ),
        r"\1=\2[SECRET_REDACTED]\2",
    ),
]

_SENSITIVE_KEYS: set[str] = {
    "api_key",
    "apikey",
    "secret",
    "client_secret",
    "token",
    "access_token",
    "auth_token",
    "authorization",
    "auth",
    "password",
    "passwd",
    "private_key",
    "secret_key",
    "access_key",
}


def redact_text(text: str | None) -> str:
    """Redact known secrets, tokens, and PII from a text string.

    Args:
        text: Input string.

    Returns:
        Redacted string with sensitive patterns replaced by placeholders.
    """
    if not text:
        return text or ""

    redacted = text
    for pattern, replacement in _PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def redact_data(data: Any) -> Any:
    """Recursively redact dictionary, list, or primitive data structures.

    Args:
        data: Arbitrary nested Python structure (dict, list, str, primitive).

    Returns:
        New structure with sensitive fields and text redacted.
    """
    if isinstance(data, str):
        return redact_text(data)
    elif isinstance(data, dict):
        result: dict[str, Any] = {}
        for k, v in data.items():
            if str(k).lower() in _SENSITIVE_KEYS:
                result[k] = "[SECRET_REDACTED]"
            else:
                result[k] = redact_data(v)
        return result
    elif isinstance(data, list):
        return [redact_data(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(redact_data(item) for item in data)
    elif isinstance(data, set):
        return {redact_data(item) for item in data}
    return data


def redact_diff_report(report: DiffReport) -> DiffReport:
    """Create a redacted copy of a DiffReport safe for public sharing or PR comments.

    Retains all scores, verdicts, latencies, tokens, and costs while masking
    any sensitive credentials, tokens, or PII in inputs, outputs, and diff chunks.
    """
    redacted_comparisons: list[ComparisonResult] = []

    for comp in report.comparisons:
        # Redact test case variables
        redacted_tc = comp.test_case.model_copy(
            update={
                "description": redact_text(comp.test_case.description),
                "vars": redact_data(comp.test_case.vars),
                "expected_output": (
                    redact_text(comp.test_case.expected_output) if comp.test_case.expected_output else None
                ),
            }
        )

        # Redact run results
        v1_res = comp.v1_result.model_copy(
            update={
                "rendered_prompt": redact_text(comp.v1_result.rendered_prompt),
                "output": redact_text(comp.v1_result.output),
                "error": redact_text(comp.v1_result.error) if comp.v1_result.error else None,
            }
        )
        v2_res = comp.v2_result.model_copy(
            update={
                "rendered_prompt": redact_text(comp.v2_result.rendered_prompt),
                "output": redact_text(comp.v2_result.output),
                "error": redact_text(comp.v2_result.error) if comp.v2_result.error else None,
            }
        )

        # Redact text diff chunks
        redacted_chunks = [
            DiffChunk(
                kind=chunk.kind,
                v1_text=redact_text(chunk.v1_text),
                v2_text=redact_text(chunk.v2_text),
            )
            for chunk in comp.text_diff
        ]

        redacted_comp = ComparisonResult(
            test_case=redacted_tc,
            v1_result=v1_res,
            v2_result=v2_res,
            scores=comp.scores,
            text_diff=redacted_chunks,
            json_diff=comp.json_diff,
            is_json=comp.is_json,
        )
        redacted_comparisons.append(redacted_comp)

    # Redact config snapshot if present
    redacted_snapshot = redact_data(report.config_snapshot) if report.config_snapshot else None

    return report.model_copy(
        update={
            "comparisons": redacted_comparisons,
            "config_snapshot": redacted_snapshot,
        }
    )


class SecretRedactingFilter(logging.Filter):
    """Logging filter that automatically redacts API keys, credentials, and PII from log messages."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.msg and isinstance(record.msg, str):
            record.msg = redact_text(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = redact_data(record.args)
            elif isinstance(record.args, (list, tuple)):
                record.args = tuple(redact_text(arg) if isinstance(arg, str) else arg for arg in record.args)
        return True
