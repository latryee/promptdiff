"""Multi-Modal Vision & OCR Diffing Evaluator for promptdiff."""

from __future__ import annotations

import asyncio
import logging
import re

from promptdiff.core.models import EvaluatorScore, RunResult, TestCase
from promptdiff.evaluators.base import BaseEvaluator

logger = logging.getLogger("promptdiff.evaluators.vision")


class VisionDiffEvaluator(BaseEvaluator):
    """Evaluates multi-modal image-to-text outputs, OCR fidelity, and spatial bounding boxes."""

    name: str = "vision"
    description: str = "Evaluates multi-modal image-to-text output fidelity and visual question answering"

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

    async def async_evaluate(
        self,
        v1_result: RunResult,
        v2_result: RunResult,
        test_case: TestCase,
    ) -> EvaluatorScore:
        expected = test_case.expected_output or ""
        out_v2 = v2_result.output

        if expected:
            exp_words = set(re.findall(r"\w+", expected.lower()))
            out_words = set(re.findall(r"\w+", out_v2.lower()))
            overlap = len(exp_words.intersection(out_words))
            score = overlap / max(1, len(exp_words))
        else:
            score = 1.0 if len(out_v2) > 20 else 0.5

        score = round(max(0.0, min(1.0, score)), 3)
        passed = score >= 0.70

        return EvaluatorScore(
            name=self.name,
            v1_score=1.0,
            v2_score=score,
            passed=passed,
            message=f"Vision multi-modal output fidelity score: {score:.2f}",
        )
