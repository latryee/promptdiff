"""Tests migrated to feature-focused test suite."""

from __future__ import annotations

import pytest

from promptdiff.core.models import PromptVersion, TestCase
from promptdiff.optimizer.reflex import SelfCorrectionBenchmark
from promptdiff.optimizer.reflexion_bench import ReflexionConvergenceBenchmark


@pytest.mark.asyncio
async def test_self_correction_benchmark() -> None:
    """Test self-correction reflection loop benchmark."""
    pv = PromptVersion(name="reflex_p", template="Answer concisely: {{query}}")
    bench = SelfCorrectionBenchmark(
        prompt_version=pv, test_cases=[TestCase(id="1", vars={"query": "Hi"})], force_mock=True
    )
    report = await bench.benchmark_reflection()
    assert report.reflection_judge_score >= report.direct_judge_score
    assert len(report.roi_verdict) > 0


def test_reflexion_convergence_benchmark() -> None:
    """Test self-correction reflexion trajectory benchmark."""
    bench = ReflexionConvergenceBenchmark()
    rep = bench.evaluate_trajectory([0.5, 0.75, 0.90, 0.90])
    assert rep.optimal_stopping_step == 3
    assert rep.diminishing_returns_reached is True
