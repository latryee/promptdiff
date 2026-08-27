"""Abstract Base Evaluator for promptdiff."""

from __future__ import annotations

from abc import ABC, abstractmethod
from promptdiff.core.models import EvaluatorScore, RunResult, TestCase


class BaseEvaluator(ABC):
    """Base class for all metric evaluators."""

    name: str = "base"
    description: str = ""

    @abstractmethod
    def evaluate(
        self,
        v1_result: RunResult,
        v2_result: RunResult,
        test_case: TestCase,
    ) -> EvaluatorScore:
        """Evaluate and compare v1 vs v2 results."""
        raise NotImplementedError
