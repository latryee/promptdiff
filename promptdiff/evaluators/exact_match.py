"""Deterministic Exact Match Evaluator.

Checks whether candidate model outputs exactly match expected target ground truth
or baseline outputs, with configurable whitespace and case sensitivity normalization.
"""

from __future__ import annotations

import difflib
from typing import Any

from promptdiff.core.models import EvaluatorScore, RunResult, TestCase
from promptdiff.evaluators.base import BaseEvaluator


class ExactMatchEvaluator(BaseEvaluator):
    """Deterministic exact match comparison against expected ground truth or baseline."""

    name: str = "exact_match"
    description: str = "Evaluates exact string match against expected output or baseline (1.0 = match, 0.0 = mismatch)"
    version: str = "1.0.0"
    is_heuristic: bool = False
    is_llm_judge: bool = False
    scoring_range: tuple[float, float] = (0.0, 1.0)

    def __init__(
        self,
        ignore_case: bool = False,
        strip_whitespace: bool = True,
        normalize_newlines: bool = True,
        compare_to_expected: bool = True,
    ):
        self.ignore_case = ignore_case
        self.strip_whitespace = strip_whitespace
        self.normalize_newlines = normalize_newlines
        self.compare_to_expected = compare_to_expected

    def _normalize(self, text: str) -> str:
        """Apply configured normalization to input text."""
        result = text
        if self.normalize_newlines:
            result = result.replace("\r\n", "\n").replace("\r", "\n")
        if self.strip_whitespace:
            result = result.strip()
        if self.ignore_case:
            result = result.lower()
        return result

    def evaluate(
        self,
        v1_result: RunResult,
        v2_result: RunResult,
        test_case: TestCase,
    ) -> EvaluatorScore:
        norm_v1 = self._normalize(v1_result.output)
        norm_v2 = self._normalize(v2_result.output)

        target = (
            test_case.expected_output if (self.compare_to_expected and test_case.expected_output is not None) else None
        )

        if target is not None:
            norm_target = self._normalize(target)
            v1_match = norm_v1 == norm_target
            v2_match = norm_v2 == norm_target
            v1_score = 1.0 if v1_match else 0.0
            v2_score = 1.0 if v2_match else 0.0
            target_desc = "expected target"
        else:
            v1_score = 1.0
            v2_match = norm_v1 == norm_v2
            v2_score = 1.0 if v2_match else 0.0
            target_desc = "v1 baseline"

        delta = v2_score - v1_score
        passed = v2_score >= v1_score

        if v2_score == 1.0:
            msg = f"Exact match with {target_desc}"
        else:
            diff_lines = list(
                difflib.unified_diff(
                    norm_target.splitlines(keepends=True) if target is not None else norm_v1.splitlines(keepends=True),
                    norm_v2.splitlines(keepends=True),
                    fromfile="expected",
                    tofile="actual",
                    n=1,
                )
            )
            diff_summary = "".join(diff_lines[:5]).strip()
            msg = f"Mismatch with {target_desc}" + (f": {diff_summary}" if diff_summary else "")

        details: dict[str, Any] = {
            "v1_match": v1_score == 1.0,
            "v2_match": v2_score == 1.0,
            "has_expected_target": target is not None,
            "ignore_case": self.ignore_case,
            "strip_whitespace": self.strip_whitespace,
        }

        return EvaluatorScore(
            name=self.name,
            v1_score=v1_score,
            v2_score=v2_score,
            delta=round(delta, 2),
            delta_pct=round(delta * 100.0, 1),
            passed=passed,
            message=msg,
            details=details,
        )
