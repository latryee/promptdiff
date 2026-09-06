"""Abstract Base Evaluator for promptdiff."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from promptdiff.core.models import EvaluatorScore, RunResult, TestCase


class BaseEvaluator(ABC):
    """Base class for all metric evaluators."""

    name: str = "base"
    description: str = ""
    version: str = "1.0.0"
    is_heuristic: bool = False
    is_llm_judge: bool = False
    scoring_range: tuple[float, float] = (0.0, 1.0)

    def get_metadata(self) -> dict[str, Any]:
        """Return structured evaluator metadata and operational properties."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "is_heuristic": self.is_heuristic,
            "is_llm_judge": self.is_llm_judge,
            "scoring_range": self.scoring_range,
        }

    @abstractmethod
    def evaluate(
        self,
        v1_result: RunResult,
        v2_result: RunResult,
        test_case: TestCase,
    ) -> EvaluatorScore:
        """Evaluate and compare v1 vs v2 results synchronously."""
        raise NotImplementedError

    async def async_evaluate(
        self,
        v1_result: RunResult,
        v2_result: RunResult,
        test_case: TestCase,
    ) -> EvaluatorScore:
        """Evaluate and compare v1 vs v2 results asynchronously."""
        return self.evaluate(v1_result, v2_result, test_case)
