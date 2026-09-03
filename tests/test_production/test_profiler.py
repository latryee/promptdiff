"""Tests migrated to feature-focused test suite."""

from __future__ import annotations

import pytest

from promptdiff.core.models import PromptVersion
from promptdiff.production.profiler import StreamingProfiler
from promptdiff.production.streaming_profiler import (
    AsyncStreamingProfiler,
    _build_ascii_sparkline,
)


@pytest.mark.asyncio
async def test_streaming_profiler() -> None:
    """Test streaming TTFT and inter-token latency profiler."""
    pv = PromptVersion(name="stream_p", template="Answer: {{query}}")
    profiler = StreamingProfiler(prompt_version=pv, force_mock=True)
    res = await profiler.profile_stream("Tell me a story")
    assert res.time_to_first_token_ms > 0
    assert res.tokens_per_second > 0
    assert len(res.full_output) > 0


@pytest.mark.asyncio
async def test_streaming_profiler_and_sparklines() -> None:
    """Test microsecond TTFT profiler and sparkline rendering."""
    spark = _build_ascii_sparkline([10.0, 20.0, 50.0, 30.0, 10.0])
    assert len(spark) > 0

    profiler = AsyncStreamingProfiler(target_ttft_sla_ms=500.0)
    report = await profiler.simulate_streaming_profiling(
        prompt="Test prompt",
        token_count=10,
        base_ttft_ms=50.0,
        base_itl_ms=10.0,
    )

    assert report.total_tokens >= 5
    assert report.ttft_ms > 0.0
    assert report.mean_itl_ms >= 0.0
    assert report.tokens_per_second > 0.0
    assert len(report.velocity_sparkline) > 0
