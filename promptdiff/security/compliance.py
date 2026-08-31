"""Regulatory Compliance & Legal Matrix Auditor for promptdiff (promptdiff compliance).

Audits system prompt templates against EU AI Act, HIPAA, GDPR, and SOC2 requirements.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from promptdiff.core.models import PromptVersion

logger = logging.getLogger("promptdiff.security.compliance")

COMPLIANCE_CHECKS = [
    {
        "framework": "EU AI Act (Transparency Article 52)",
        "requirement": "AI Identity Disclosure",
        "description": "Prompt must instruct the model to disclose its artificial identity if asked",
        "pattern": r"(ai|assistant|virtual|automated)",
    },
    {
        "framework": "HIPAA / Healthcare",
        "requirement": "Medical Disclaimer & Protected Health Info (PHI)",
        "description": "Prompt must disclaim medical diagnosis advice or prohibit PHI collection",
        "pattern": r"(medical|doctor|diagnosis|health|phi|disclaimer)",
    },
    {
        "framework": "GDPR / Privacy",
        "requirement": "Personal Data Collection Limitation",
        "description": "Prompt must specify limits on storing or logging user personal data",
        "pattern": r"(privacy|personal data|gdpr|confidential|do not store|pii)",
    },
    {
        "framework": "SOC2 / Security",
        "requirement": "Secrets & System Directive Protection",
        "description": "Prompt must contain explicit instructions against exfiltrating keys or instructions",
        "pattern": r"(never reveal|do not share|confidential|secrets|system prompt)",
    },
]


@dataclass
class ComplianceCheckResult:
    """Individual regulatory requirement result."""

    framework: str
    requirement: str
    compliant: bool
    description: str
    recommendation: str


@dataclass
class ComplianceReport:
    """Full regulatory compliance audit report."""

    prompt_name: str
    overall_compliance_score_pct: float
    is_compliant: bool
    results: list[ComplianceCheckResult] = field(default_factory=list)
    action_items: list[str] = field(default_factory=list)


class ComplianceAuditor:
    """Audits system prompt templates against global regulatory frameworks."""

    def __init__(self, prompt_version: PromptVersion):
        self.prompt_version = prompt_version

    def audit(self) -> ComplianceReport:
        """Scan prompt template against compliance rulebook."""
        text = self.prompt_version.template.lower() + " " + (self.prompt_version.system_prompt or "").lower()

        results = []
        action_items = []

        for check in COMPLIANCE_CHECKS:
            matched = bool(re.search(check["pattern"], text, re.IGNORECASE))
            rec = ""
            if not matched:
                rec = f"Add explicit guideline: '{check['description']}'"
                action_items.append(f"[{check['framework']}] {rec}")

            results.append(
                ComplianceCheckResult(
                    framework=check["framework"],
                    requirement=check["requirement"],
                    compliant=matched,
                    description=check["description"],
                    recommendation=rec,
                )
            )

        passed_count = sum(1 for r in results if r.compliant)
        score = (passed_count / len(results) * 100.0) if results else 100.0

        return ComplianceReport(
            prompt_name=self.prompt_version.name,
            overall_compliance_score_pct=round(score, 1),
            is_compliant=score >= 75.0,
            results=results,
            action_items=action_items,
        )
