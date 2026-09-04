"""Tests for runner concurrency hard upper bounds and queue backpressure."""

from __future__ import annotations

import pytest

from promptdiff.core.cache import DiskCache
from promptdiff.core.models import PromptVersion, TestCase
from promptdiff.core.runner import (
    MAX_RUNNER_CONCURRENCY,
    ArenaRunner,
    MultiVariantRunner,
    PromptDiffRunner,
    resolve_concurrency,
)
from promptdiff.evaluators.registry import get_evaluators
from promptdiff.providers.mock_provider import MockProvider


def test_resolve_concurrency_bounds() -> None:
    """Validate concurrency clamping, limits, and strict exceptions."""
    # Valid within bounds
    assert resolve_concurrency(8) == 8
    assert resolve_concurrency(MAX_RUNNER_CONCURRENCY) == MAX_RUNNER_CONCURRENCY

    # Clamping behavior when exceeding limit
    assert resolve_concurrency(100) == MAX_RUNNER_CONCURRENCY
    assert resolve_concurrency(500, max_limit=32) == 32

    # Strict mode error when exceeding limit
    with pytest.raises(ValueError, match="exceeds maximum allowed concurrency limit"):
        resolve_concurrency(100, strict=True)

    with pytest.raises(ValueError, match="exceeds maximum allowed concurrency limit"):
        resolve_concurrency(65, max_limit=64, strict=True)

    # Invalid non-positive concurrency
    with pytest.raises(ValueError, match="Concurrency must be a positive integer"):
        resolve_concurrency(0)

    with pytest.raises(ValueError, match="Concurrency must be a positive integer"):
        resolve_concurrency(-5)


def test_runner_initialization_concurrency_clamps() -> None:
    """Verify PromptDiffRunner and MultiVariantRunner enforce concurrency boundaries."""
    v1 = PromptVersion(name="v1", template="Query: {{q}}", model="mock-model")
    v2 = PromptVersion(name="v2", template="Question: {{q}}", model="mock-model")
    provider = MockProvider()

    # PromptDiffRunner default clamp
    runner = PromptDiffRunner(
        v1_prompt=v1,
        v2_prompt=v2,
        provider_v1=provider,
        provider_v2=provider,
        concurrency=128,
    )
    assert runner.concurrency == MAX_RUNNER_CONCURRENCY

    # PromptDiffRunner strict validation
    with pytest.raises(ValueError, match="exceeds maximum allowed concurrency limit"):
        PromptDiffRunner(
            v1_prompt=v1,
            v2_prompt=v2,
            provider_v1=provider,
            provider_v2=provider,
            concurrency=128,
            strict_concurrency=True,
        )

    # MultiVariantRunner / ArenaRunner alias check
    assert MultiVariantRunner is ArenaRunner
    arena = MultiVariantRunner(
        variants={"v1": v1, "v2": v2},
        providers={"v1": provider, "v2": provider},
        concurrency=256,
    )
    assert arena.concurrency == MAX_RUNNER_CONCURRENCY

    with pytest.raises(ValueError, match="exceeds maximum allowed concurrency limit"):
        MultiVariantRunner(
            variants={"v1": v1, "v2": v2},
            providers={"v1": provider, "v2": provider},
            concurrency=256,
            strict_concurrency=True,
        )


@pytest.mark.asyncio
async def test_runner_backpressure_bounded_queue() -> None:
    """Verify that bounded queue processes large batches under tight worker/queue capacity."""
    v1 = PromptVersion(name="v1", template="Hello {{name}}", model="mock-gpt-4o")
    v2 = PromptVersion(name="v2", template="Hi {{name}}!", model="mock-gpt-4o")
    p1 = MockProvider(latency_range=(0.001, 0.002))
    p2 = MockProvider(latency_range=(0.001, 0.002))

    # Concurrency 4 with tight queue buffer (queue_size=6)
    runner = PromptDiffRunner(
        v1_prompt=v1,
        v2_prompt=v2,
        provider_v1=p1,
        provider_v2=p2,
        evaluators=get_evaluators(["json_validity", "latency", "cost"]),
        cache=DiskCache(enabled=False),
        concurrency=4,
        queue_size=6,
    )

    num_cases = 60
    test_cases = [TestCase(id=f"case_{i:03d}", vars={"name": f"User_{i}"}) for i in range(num_cases)]

    progress_events: list[tuple[int, int]] = []

    def on_progress(completed: int, total: int) -> None:
        progress_events.append((completed, total))

    report = await runner.run(test_cases, progress_cb=on_progress)

    # Assert all items completed in ordered fashion
    assert report.total_cases == num_cases
    assert len(report.comparisons) == num_cases
    assert [c.test_case.id for c in report.comparisons] == [f"case_{i:03d}" for i in range(num_cases)]
    assert len(progress_events) == num_cases
    assert progress_events[-1] == (num_cases, num_cases)


@pytest.mark.asyncio
async def test_runner_empty_test_cases() -> None:
    """Verify runner handles empty case list gracefully with queue worker pipeline."""
    v1 = PromptVersion(name="v1", template="Hello {{name}}", model="mock-gpt-4o")
    v2 = PromptVersion(name="v2", template="Hi {{name}}!", model="mock-gpt-4o")
    runner = PromptDiffRunner(
        v1_prompt=v1,
        v2_prompt=v2,
        provider_v1=MockProvider(),
        provider_v2=MockProvider(),
        cache=DiskCache(enabled=False),
    )

    report = await runner.run([])
    assert report.total_cases == 0
    assert len(report.comparisons) == 0

    arena = MultiVariantRunner(
        variants={"v1": v1, "v2": v2},
        providers={"v1": MockProvider(), "v2": MockProvider()},
    )
    arena_report = await arena.run([])
    assert arena_report.total_cases == 0
    assert len(arena_report.comparisons) == 0
