"""Prompt Guideline Keyword Linter (Heuristic Keyword Pattern Checks).

DISCLAIMER: This module performs basic heuristic keyword pattern matching against
common prompt engineering guidelines and conventions. It is NOT legal, regulatory,
or formal compliance advice, and does NOT constitute an audit or guarantee of compliance
with the EU AI Act, HIPAA, GDPR, SOC2, or any other statutory or legal framework.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from promptdiff.core.models import PromptVersion

logger = logging.getLogger("promptdiff.security.compliance")

DISCLAIMER_NOTICE = (
    "DISCLAIMER: This report is a heuristic keyword lint, NOT legal advice or a regulatory compliance audit."
)

GUIDELINE_KEYWORD_RULES = [
    {
        "category": "Transparency Guideline (e.g. AI Identity)",
        "guideline": "AI Identity Disclosure",
        "description": "Prompt instructs the model to acknowledge its artificial identity if asked",
        "pattern": r"(ai|assistant|virtual|automated)",
    },
    {
        "category": "Healthcare Guideline (e.g. Medical Disclaimer)",
        "guideline": "Medical Advice Disclaimer",
        "description": "Prompt disclaims medical diagnosis advice or prohibits PHI collection",
        "pattern": r"(medical|doctor|diagnosis|health|phi|disclaimer)",
    },
    {
        "category": "Privacy Guideline (e.g. Data Minimization)",
        "guideline": "Personal Data Collection Limit",
        "description": "Prompt specifies limits on storing or logging user personal data",
        "pattern": r"(privacy|personal data|gdpr|confidential|do not store|pii)",
    },
    {
        "category": "Security Guideline (e.g. Secret Protection)",
        "guideline": "Secrets & System Directive Protection",
        "description": "Prompt contains explicit instructions against exfiltrating keys or instructions",
        "pattern": r"(never reveal|do not share|confidential|secrets|system prompt)",
    },
]

# Backwards-compatibility alias
COMPLIANCE_CHECKS = GUIDELINE_KEYWORD_RULES


@dataclass
class GuidelineCheckResult:
    """Individual heuristic guideline rule result."""

    category: str
    guideline: str
    compliant: bool
    description: str
    recommendation: str

    @property
    def framework(self) -> str:
        """Backwards-compatibility accessor."""
        return self.category

    @property
    def requirement(self) -> str:
        """Backwards-compatibility accessor."""
        return self.guideline


# Backwards-compatibility alias
ComplianceCheckResult = GuidelineCheckResult


@dataclass
class GuidelineLintReport:
    """Heuristic prompt guideline lint report."""

    prompt_name: str
    overall_compliance_score_pct: float
    is_compliant: bool
    results: list[GuidelineCheckResult] = field(default_factory=list)
    action_items: list[str] = field(default_factory=list)
    disclaimer: str = DISCLAIMER_NOTICE


# Backwards-compatibility alias
ComplianceReport = GuidelineLintReport


class PromptGuidelineLinter:
    """Heuristic regex keyword linter for common prompt guidelines."""

    def __init__(self, prompt_version: PromptVersion):
        self.prompt_version = prompt_version

    def lint(self) -> GuidelineLintReport:
        """Scan prompt template against guideline keyword rules."""
        text = self.prompt_version.template.lower() + " " + (self.prompt_version.system_prompt or "").lower()

        results = []
        action_items = []

        for rule in GUIDELINE_KEYWORD_RULES:
            matched = bool(re.search(rule["pattern"], text, re.IGNORECASE))
            rec = ""
            if not matched:
                rec = f"Consider adding guideline: '{rule['description']}'"
                action_items.append(f"[{rule['category']}] {rec}")

            results.append(
                GuidelineCheckResult(
                    category=rule["category"],
                    guideline=rule["guideline"],
                    compliant=matched,
                    description=rule["description"],
                    recommendation=rec,
                )
            )

        passed_count = sum(1 for r in results if r.compliant)
        score = (passed_count / len(results) * 100.0) if results else 100.0

        return GuidelineLintReport(
            prompt_name=self.prompt_version.name,
            overall_compliance_score_pct=round(score, 1),
            is_compliant=score >= 75.0,
            results=results,
            action_items=action_items,
            disclaimer=DISCLAIMER_NOTICE,
        )

    def audit(self) -> GuidelineLintReport:
        """Backwards-compatibility alias for lint()."""
        return self.lint()


# Backwards-compatibility alias
ComplianceAuditor = PromptGuidelineLinter
