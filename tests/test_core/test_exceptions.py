"""Unit tests for promptdiff exception hierarchy and structured error handling."""

from __future__ import annotations

import pytest

from promptdiff.core.exceptions import (
    CacheError,
    CacheReadError,
    CacheWriteError,
    ConfigurationError,
    DatasetLoadError,
    EvaluatorExecutionError,
    PromptDiffError,
    ProviderExecutionError,
    RunnerError,
)
from promptdiff.core.models import PromptVersion, TestCase
from promptdiff.core.runner import PromptDiffRunner
from promptdiff.evaluators.base import BaseEvaluator
from promptdiff.providers.base import BaseLLMProvider, ProviderResponse


def test_exception_hierarchy_inheritance() -> None:
    """Verify all custom exceptions inherit from PromptDiffError."""
    assert issubclass(CacheError, PromptDiffError)
    assert issubclass(CacheReadError, CacheError)
    assert issubclass(CacheWriteError, CacheError)
    assert issubclass(RunnerError, PromptDiffError)
    assert issubclass(ProviderExecutionError, RunnerError)
    assert issubclass(EvaluatorExecutionError, RunnerError)
    assert issubclass(DatasetLoadError, PromptDiffError)
    assert issubclass(ConfigurationError, PromptDiffError)


class FailingProvider(BaseLLMProvider):
    """Provider that deliberately raises an exception to test error handling."""

    async def generate(self, prompt: str, **kwargs) -> ProviderResponse:
        raise ConnectionResetError("Simulated provider outage")


class FailingEvaluator(BaseEvaluator):
    """Evaluator that deliberately raises an exception in evaluate."""

    name: str = "failing_eval"
    description: str = "Test evaluator that fails"

    def evaluate(self, v1_result, v2_result, test_case):
        raise ValueError("Simulated evaluator bug")


@pytest.mark.asyncio
async def test_runner_structured_error_resilience() -> None:
    """Verify PromptDiffRunner catches provider and evaluator errors and logs them without crashing."""
    p_fail = FailingProvider("failing-model")
    v1 = PromptVersion(name="v1", template="Hello {{name}}", model="failing-model")
    v2 = PromptVersion(name="v2", template="Hi {{name}}", model="failing-model")

    evaluator = FailingEvaluator()
    runner = PromptDiffRunner(
        v1_prompt=v1,
        v2_prompt=v2,
        provider_v1=p_fail,
        provider_v2=p_fail,
        evaluators=[evaluator],
    )

    tc = TestCase(id="tc_error_test", vars={"name": "Alice"})
    comp = await runner.compare_case(tc)

    # Provider failure should be captured in RunResult.error without crashing
    assert comp.v1_result.error is not None
    assert "Simulated provider outage" in comp.v1_result.error
    assert comp.v2_result.error is not None

    # Evaluator failure should produce a failed score with error message
    assert "failing_eval" in comp.scores
    score = comp.scores["failing_eval"]
    assert score.passed is False
    assert "Evaluation failed" in score.message
