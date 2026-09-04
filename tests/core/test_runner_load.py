"""Load and concurrency stress test for PromptDiffRunner."""

from __future__ import annotations

import statistics
import time
import tracemalloc

import pytest

from promptdiff.core.cache import DiskCache
from promptdiff.core.models import PromptVersion, TestCase
from promptdiff.core.runner import PromptDiffRunner
from promptdiff.evaluators.registry import get_evaluators
from promptdiff.providers.mock_provider import MockProvider


@pytest.mark.asyncio
async def test_runner_high_concurrency_load() -> None:
    """Run 250+ concurrent test cases through PromptDiffRunner and verify integrity and resource usage."""
    num_cases = 250
    concurrency_limit = 16

    v1 = PromptVersion(
        name="v1",
        template="Hello {{name}}, order {{query}} is ready.",
        model="mock-gpt-4o",
    )
    v2 = PromptVersion(
        name="v2",
        template="Hi {{name}}, your order {{query}} is ready to ship!",
        model="mock-gpt-4o",
    )

    provider_v1 = MockProvider(latency_range=(0.001, 0.003))
    provider_v2 = MockProvider(latency_range=(0.001, 0.003))

    cache = DiskCache(enabled=False)
    # Use standard fast evaluators (json_validity, latency, cost) for load benchmarking
    runner = PromptDiffRunner(
        v1_prompt=v1,
        v2_prompt=v2,
        provider_v1=provider_v1,
        provider_v2=provider_v2,
        evaluators=get_evaluators(["json_validity", "latency", "cost"]),
        cache=cache,
        concurrency=concurrency_limit,
    )

    test_cases = [
        TestCase(
            id=f"case_{i:04d}",
            vars={"name": f"Customer_{i}", "query": f"ORD-{1000 + i}"},
        )
        for i in range(num_cases)
    ]

    tracemalloc.start()
    start_time = time.perf_counter()

    report = await runner.run(test_cases)

    total_duration = time.perf_counter() - start_time
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # 1. Assert no data loss or dropped cases
    assert report.total_cases == num_cases
    assert len(report.comparisons) == num_cases

    # 2. Assert no race condition / case ID interleaving
    expected_ids = {f"case_{i:04d}" for i in range(num_cases)}
    actual_ids = {c.test_case.id for c in report.comparisons}
    assert actual_ids == expected_ids

    for comp in report.comparisons:
        assert comp.v1_result.test_case_id == comp.test_case.id
        assert comp.v2_result.test_case_id == comp.test_case.id
        assert comp.v1_result.output != ""
        assert comp.v2_result.output != ""

    # 3. Calculate latency percentiles (p50, p95, p99)
    v1_latencies = sorted(c.v1_result.latency_ms for c in report.comparisons)
    v2_latencies = sorted(c.v2_result.latency_ms for c in report.comparisons)

    def calc_percentile(data: list[float], pct: float) -> float:
        idx = int(len(data) * pct / 100)
        return data[min(idx, len(data) - 1)]

    p50_v1 = calc_percentile(v1_latencies, 50)
    p95_v1 = calc_percentile(v1_latencies, 95)
    p99_v1 = calc_percentile(v1_latencies, 99)

    assert p50_v1 >= 0.0
    assert p95_v1 >= p50_v1
    assert p99_v1 >= p95_v1
    assert statistics.mean(v1_latencies) >= 0.0
    assert statistics.mean(v2_latencies) >= 0.0

    # 4. Assert memory efficiency under load (< 25 MB peak)
    peak_mb = peak_memory / (1024 * 1024)
    assert peak_mb < 25.0, f"Peak memory {peak_mb:.2f}MB exceeded limit"
    assert total_duration > 0.0
