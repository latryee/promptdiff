"""Regex Match Evaluator.

Checks whether model outputs match specific structural regex patterns or expected tokens.
"""

from __future__ import annotations

import re
from typing import Optional
from promptdiff.core.models import EvaluatorScore, RunResult, TestCase
from promptdiff.evaluators.base import BaseEvaluator


class RegexMatchEvaluator(BaseEvaluator):
    """Evaluates regex pattern compliance across prompt outputs."""

    name: str = "regex_match"
    description: str = "Checks output structure compliance against regex patterns"

    def __init__(self, pattern: Optional[str] = None):
        self.pattern = pattern

    def evaluate(
        self,
        v1_result: RunResult,
        v2_result: RunResult,
        test_case: TestCase,
    ) -> EvaluatorScore:
        # Pattern can come from constructor or test_case vars
        pat = self.pattern or test_case.vars.get("regex_pattern") or test_case.vars.get("pattern")
        if not pat:
            # Default check: non-empty output
            pat = r"\S+"

        try:
            compiled = re.compile(pat, re.IGNORECASE)
            v1_match = bool(compiled.search(v1_result.output))
            v2_match = bool(compiled.search(v2_result.output))
        except Exception as err:
            return EvaluatorScore(
                name=self.name,
                v1_score=0.0,
                v2_score=0.0,
                passed=False,
                message=f"Regex compile error: {err}",
            )

        v1_score = 1.0 if v1_match else 0.0
        v2_score = 1.0 if v2_match else 0.0
        delta = v2_score - v1_score

        msg = "Pattern matched" if v2_match else "Pattern not found"

        return EvaluatorScore(
            name=self.name,
            v1_score=v1_score,
            v2_score=v2_score,
            delta=delta,
            passed=v2_match,
            message=msg,
            details={
                "pattern": pat,
                "v1_matched": v1_match,
                "v2_matched": v2_match,
            },
        )
