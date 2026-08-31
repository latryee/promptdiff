"""Sentence-Level Citation & Hallucination Attribution Evaluator for promptdiff."""

from __future__ import annotations

import asyncio
import logging
import re

from promptdiff.core.models import EvaluatorScore, RunResult, TestCase
from promptdiff.evaluators.base import BaseEvaluator

logger = logging.getLogger("promptdiff.evaluators.citation")


class CitationEvaluator(BaseEvaluator):
    """Pinpoints exact sentences unsupported by the reference context (hallucination attribution)."""

    name: str = "citation"
    description: str = "Pinpoints sentence-level hallucination and source context attribution"

    def __init__(
        self,
        model_name: str = "gpt-4o",
        force_mock: bool = False,
    ):
        self.model_name = model_name
        self.force_mock = force_mock

    def evaluate(
        self,
        v1_result: RunResult,
        v2_result: RunResult,
        test_case: TestCase,
    ) -> EvaluatorScore:
        return asyncio.run(self.async_evaluate(v1_result, v2_result, test_case))

    def _split_sentences(self, text: str) -> list[str]:
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip() and len(s) > 10]

    def _attribute_sentences(self, output: str, context: str) -> tuple[int, int, list[str]]:
        sentences = self._split_sentences(output)
        if not sentences or not context:
            return 0, 0, []

        c_lower = context.lower()
        supported = 0
        unsupported = []

        for s in sentences:
            s_words = set(re.findall(r"\w+", s.lower()))
            overlap = sum(1 for w in s_words if w in c_lower)
            overlap_ratio = overlap / max(1, len(s_words))

            if overlap_ratio >= 0.50:
                supported += 1
            else:
                unsupported.append(s)

        return supported, len(sentences), unsupported

    async def async_evaluate(
        self,
        v1_result: RunResult,
        v2_result: RunResult,
        test_case: TestCase,
    ) -> EvaluatorScore:
        context = str(test_case.vars.get("context") or test_case.vars.get("reference") or "")

        if not context:
            return EvaluatorScore(
                name=self.name,
                v1_score=1.0,
                v2_score=1.0,
                passed=True,
                message="No reference context provided for citation attribution.",
            )

        v1_supp, v1_tot, _ = self._attribute_sentences(v1_result.output, context)
        v2_supp, v2_tot, v2_unsupp = self._attribute_sentences(v2_result.output, context)

        v1_score = round((v1_supp / v1_tot) if v1_tot else 1.0, 3)
        v2_score = round((v2_supp / v2_tot) if v2_tot else 1.0, 3)

        passed = v2_score >= 0.70
        if v2_unsupp:
            reason = f"Citation attribution score {v2_score:.2f}. Found {len(v2_unsupp)} ungrounded sentence(s): '{v2_unsupp[0][:60]}...'"
        else:
            reason = f"100% Citation attribution: all {v2_tot} sentences grounded in source context."

        return EvaluatorScore(
            name=self.name,
            v1_score=v1_score,
            v2_score=v2_score,
            passed=passed,
            message=reason,
        )
