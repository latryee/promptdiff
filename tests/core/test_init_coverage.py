"""Coverage boost tests for package __init__.py imports and evaluators/base.py."""

from __future__ import annotations

import pytest

from promptdiff.core.models import EvaluatorScore, RunResult, TestCase
from promptdiff.evaluators.base import BaseEvaluator


def test_generators_init_import() -> None:
    """Cover generators/__init__.py by importing SyntheticTestGenerator."""
    from promptdiff.generators import SyntheticTestGenerator

    assert SyntheticTestGenerator is not None


def test_optimizer_init_import() -> None:
    """Cover optimizer/__init__.py by importing all exported names."""
    from promptdiff.optimizer import (
        PromptOptimizer,
        PromptTuner,
        TuningReport,
    )

    assert PromptOptimizer is not None
    assert PromptTuner is not None
    assert TuningReport is not None


def test_production_init_import() -> None:
    """Cover production/__init__.py by importing all exported names."""
    from promptdiff.production import (
        CanaryConfigGenerator,
        ModelCascadeRouter,
        ShadowTrafficReplayer,
    )

    assert CanaryConfigGenerator is not None
    assert ModelCascadeRouter is not None
    assert ShadowTrafficReplayer is not None


def test_base_evaluator_subclass_evaluate() -> None:
    """Cover BaseEvaluator abstract interface and async_evaluate default impl."""

    class DummyEvaluator(BaseEvaluator):
        name = "dummy"

        def evaluate(self, v1_result: RunResult, v2_result: RunResult, test_case: TestCase) -> EvaluatorScore:
            return EvaluatorScore(
                name=self.name,
                v1_score=1.0,
                v2_score=0.9,
                passed=True,
                message="test",
            )

    ev = DummyEvaluator()
    assert ev.name == "dummy"
    assert ev.description == ""

    tc = TestCase(id="tc1", vars={"query": "test"})
    v1 = RunResult(
        prompt_name="v1",
        test_case_id="tc1",
        rendered_prompt="p",
        output="out1",
        latency_ms=10.0,
        prompt_tokens=5,
        completion_tokens=5,
        total_tokens=10,
        cost_usd=0.0,
        model="mock",
    )
    v2 = RunResult(
        prompt_name="v2",
        test_case_id="tc1",
        rendered_prompt="p",
        output="out2",
        latency_ms=12.0,
        prompt_tokens=5,
        completion_tokens=5,
        total_tokens=10,
        cost_usd=0.0,
        model="mock",
    )

    score = ev.evaluate(v1, v2, tc)
    assert score.v1_score == 1.0
    assert score.v2_score == 0.9


@pytest.mark.asyncio
async def test_base_evaluator_async_default() -> None:
    """async_evaluate default implementation delegates to evaluate."""

    class DummyEval(BaseEvaluator):
        name = "dummy_async"

        def evaluate(self, v1_result: RunResult, v2_result: RunResult, test_case: TestCase) -> EvaluatorScore:
            return EvaluatorScore(name=self.name, v1_score=1.0, v2_score=0.8, passed=True)

    ev = DummyEval()
    tc = TestCase(id="tc", vars={})
    v1 = RunResult(
        prompt_name="v1",
        test_case_id="tc",
        rendered_prompt="p",
        output="o",
        latency_ms=10.0,
        prompt_tokens=1,
        completion_tokens=1,
        total_tokens=2,
        cost_usd=0.0,
        model="m",
    )
    v2 = RunResult(
        prompt_name="v2",
        test_case_id="tc",
        rendered_prompt="p",
        output="o",
        latency_ms=10.0,
        prompt_tokens=1,
        completion_tokens=1,
        total_tokens=2,
        cost_usd=0.0,
        model="m",
    )
    score = await ev.async_evaluate(v1, v2, tc)
    assert score.v2_score == 0.8
