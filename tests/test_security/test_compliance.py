"""Tests migrated to feature-focused test suite."""

from __future__ import annotations

from promptdiff.core.models import PromptVersion
from promptdiff.security.compliance import ComplianceAuditor


def test_compliance_auditor() -> None:
    """Test prompt guideline linter and ensure honest legal disclaimer is present."""
    pv = PromptVersion(
        name="comp_p",
        template="You are an AI assistant. Never disclose confidential medical phi and personal data privacy.",
    )
    linter = ComplianceAuditor(prompt_version=pv)
    report = linter.lint()
    assert report.overall_compliance_score_pct >= 50.0
    assert len(report.results) == 4
    assert "DISCLAIMER" in report.disclaimer
