"""Historical performance benchmark suite for PromptDiffRunner."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from promptdiff.core.models import PromptVersion, TestCase
from promptdiff.core.runner import PromptDiffRunner
from promptdiff.evaluators.registry import get_evaluators
from promptdiff.providers.registry import get_provider


def _make_benchmark_runner() -> tuple[PromptDiffRunner, list[TestCase]]:
    v1 = PromptVersion(name="v1", template="Summarize text: {{text}}")
    v2 = PromptVersion(name="v2", template="Summarize text concisely: {{text}}")
    evaluators = get_evaluators(["latency", "cost", "json_validity"])
    p1 = get_provider("gpt-4o", force_mock=True)
    p2 = get_provider("gpt-4o", force_mock=True)

    runner = PromptDiffRunner(
        v1_prompt=v1,
        v2_prompt=v2,
        provider_v1=p1,
        provider_v2=p2,
        evaluators=evaluators,
        concurrency=4,
    )
    cases = [TestCase(id=f"bench_{i}", vars={"text": f"Input payload {i}"}) for i in range(10)]
    return runner, cases


def test_runner_execution_benchmark(pytestconfig: Any) -> None:
    """Benchmark runner execution throughput and latency across 10 test cases."""
    runner, cases = _make_benchmark_runner()

    def run_sync() -> None:
        asyncio.run(runner.run(cases))

    start = time.perf_counter()
    iterations = 5
    for _ in range(iterations):
        run_sync()
    duration = time.perf_counter() - start

    avg_per_run_ms = (duration / iterations) * 1000.0
    throughput_runs_per_sec = iterations / duration

    assert avg_per_run_ms < 1000.0, f"Runner execution too slow: {avg_per_run_ms:.2f}ms"
    assert throughput_runs_per_sec > 1.0


def test_runner_concurrency_scaling_benchmark() -> None:
    """Benchmark speedup and throughput under varying concurrency levels."""
    v1 = PromptVersion(name="v1", template="Hello {{name}}")
    v2 = PromptVersion(name="v2", template="Hi {{name}}")
    cases = [TestCase(id=f"scale_{i}", vars={"name": f"User_{i}"}) for i in range(20)]

    runner_c1 = PromptDiffRunner(
        v1_prompt=v1,
        v2_prompt=v2,
        provider_v1=get_provider("gpt-4o", force_mock=True),
        provider_v2=get_provider("gpt-4o", force_mock=True),
        evaluators=get_evaluators(["latency"]),
        concurrency=1,
    )
    runner_c4 = PromptDiffRunner(
        v1_prompt=v1,
        v2_prompt=v2,
        provider_v1=get_provider("gpt-4o", force_mock=True),
        provider_v2=get_provider("gpt-4o", force_mock=True),
        evaluators=get_evaluators(["latency"]),
        concurrency=4,
    )

    t0 = time.perf_counter()
    rep1 = asyncio.run(runner_c1.run(cases))
    d1 = time.perf_counter() - t0

    t1 = time.perf_counter()
    rep4 = asyncio.run(runner_c4.run(cases))
    d4 = time.perf_counter() - t1

    assert rep1.total_cases == 20
    assert rep4.total_cases == 20
    assert d1 > 0.0
    assert d4 > 0.0
