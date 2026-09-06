"""Unit tests for secret, token, and PII redaction engine."""

import logging
from io import StringIO

from promptdiff.core.models import (
    ComparisonResult,
    DiffChunk,
    DiffReport,
    EvaluatorScore,
    RegressionVerdict,
    RunResult,
    TestCase,
)
from promptdiff.security.redaction import (
    SecretRedactingFilter,
    redact_data,
    redact_diff_report,
    redact_text,
)


def test_redact_text_api_keys():
    # OpenAI format
    text = "Use key sk-1234567890abcdef1234567890 to authenticate."
    redacted = redact_text(text)
    assert "sk-1234567890abcdef1234567890" not in redacted
    assert "[API_KEY_REDACTED]" in redacted

    # Anthropic format
    text_ant = "Auth header: sk-ant-api03-abcdefghijklmnopqrstuvwxyz"
    assert "[ANTHROPIC_KEY_REDACTED]" in redact_text(text_ant)

    # Google format
    text_goog = "Key is AIzaSyD12345678901234567890123456789012"
    assert "[GOOGLE_KEY_REDACTED]" in redact_text(text_goog)

    # Hugging Face format
    text_hf = "hf_0123456789012345678901234567890123"
    assert "[HF_TOKEN_REDACTED]" in redact_text(text_hf)

    # AWS Access Key
    text_aws = "AWS id AKIAIOSFODNN7EXAMPLE"
    assert "[AWS_KEY_REDACTED]" in redact_text(text_aws)


def test_redact_text_tokens_and_pii():
    # Bearer token
    text_bearer = "Authorization: Bearer mySecretToken1234567890abcdef"
    assert "Bearer [TOKEN_REDACTED]" in redact_text(text_bearer)

    # Email
    text_email = "Contact user at john.doe@example.com for info"
    assert "[EMAIL_REDACTED]" in redact_text(text_email)
    assert "john.doe@example.com" not in redact_text(text_email)

    # SSN
    text_ssn = "Customer SSN is 123-45-6789"
    assert "[SSN_REDACTED]" in redact_text(text_ssn)

    # Credit card
    text_cc = "Payment card 4111 2222 3333 4444 approved"
    assert "[CREDIT_CARD_REDACTED]" in redact_text(text_cc)


def test_redact_data_nested():
    data = {
        "user": "alice",
        "api_key": "raw_secret_value_123",
        "nested": {
            "password": "super_secret_pw",
            "email": "test@domain.org",
            "items": ["safe_string", "sk-1234567890abcdef1234567890"],
        },
    }
    redacted = redact_data(data)
    assert redacted["api_key"] == "[SECRET_REDACTED]"
    assert redacted["nested"]["password"] == "[SECRET_REDACTED]"
    assert redacted["nested"]["email"] == "[EMAIL_REDACTED]"
    assert redacted["nested"]["items"][0] == "safe_string"
    assert "[API_KEY_REDACTED]" in redacted["nested"]["items"][1]


def test_redact_diff_report():
    tc = TestCase(id="tc_1", vars={"secret": "sk-1234567890abcdef1234567890", "user": "bob"})
    v1 = RunResult(
        prompt_name="v1",
        test_case_id="tc_1",
        rendered_prompt="Hello secret: sk-1234567890abcdef1234567890",
        output="Result with sk-ant-api03-abcdefghijklmnopqrstuvwxyz",
        prompt_tokens=10,
        completion_tokens=15,
        total_tokens=25,
        model="gpt-4o",
        cost_usd=0.001,
        latency_ms=100.0,
    )
    v2 = RunResult(
        prompt_name="v2",
        test_case_id="tc_1",
        rendered_prompt="Hello clean",
        output="Safe output without keys",
        prompt_tokens=10,
        completion_tokens=12,
        total_tokens=22,
        model="gpt-4o",
        cost_usd=0.0005,
        latency_ms=80.0,
    )
    comp = ComparisonResult(
        test_case=tc,
        v1_result=v1,
        v2_result=v2,
        scores={"similarity": EvaluatorScore(name="similarity", v1_score=1.0, v2_score=0.9, delta=-0.1, passed=True)},
        text_diff=[DiffChunk(kind="delete", v1_text="sk-1234567890abcdef1234567890", v2_text="")],
    )
    report = DiffReport(
        timestamp="2026-01-01T00:00:00Z",
        v1_name="v1",
        v2_name="v2",
        model_v1="gpt-4o",
        model_v2="gpt-4o",
        comparisons=[comp],
        verdict=RegressionVerdict(passed=True),
        evaluators=["similarity"],
        total_cases=1,
    )

    clean_report = redact_diff_report(report)

    # Scores and verdicts are preserved intact
    assert clean_report.verdict.passed is True
    assert clean_report.comparisons[0].scores["similarity"].delta == -0.1

    # Secrets are masked
    assert "sk-1234567890abcdef1234567890" not in str(clean_report.comparisons[0].test_case.vars)
    assert "[ANTHROPIC_KEY_REDACTED]" in clean_report.comparisons[0].v1_result.output
    assert "[API_KEY_REDACTED]" in clean_report.comparisons[0].text_diff[0].v1_text


def test_secret_redacting_filter():
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.addFilter(SecretRedactingFilter())
    handler.setFormatter(logging.Formatter("%(message)s"))

    test_logger = logging.getLogger("test_redact_logger")
    test_logger.setLevel(logging.INFO)
    test_logger.addHandler(handler)

    test_logger.info("Connecting using key %s", "sk-1234567890abcdef1234567890")
    log_output = stream.getvalue()
    assert "sk-1234567890abcdef1234567890" not in log_output
    assert "[API_KEY_REDACTED]" in log_output
