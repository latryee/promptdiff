"""Microsecond-Precision Asynchronous Streaming TTFT & Inter-Token Latency (ITL) Profiler.

Profiles real-time Server-Sent Events (SSE) token generation streams, calculating
Time-To-First-Token (TTFT), Inter-Token Arrival Latency (ITL), token generation velocity,
jitter standard deviation, and percentile distributions (P50, P90, P95, P99).
"""

from __future__ import annotations

import asyncio
import math
import statistics
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Optional

from promptdiff.providers.base import BaseLLMProvider


@dataclass
class TokenArrival:
    """Telemetry data point for a streamed token chunk."""

    chunk_text: str
    arrival_time_s: float
    latency_delta_ms: float
    token_count: int


@dataclass
class StreamingProfileReport:
    """Comprehensive streaming performance profile."""

    model_name: str
    total_tokens: int
    total_stream_duration_ms: float
    ttft_ms: float  # Time-to-first-token in ms
    mean_itl_ms: float  # Mean inter-token latency
    median_itl_ms: float  # P50
    p90_itl_ms: float
    p95_itl_ms: float
    p99_itl_ms: float
    jitter_std_dev_ms: float  # Latency jitter
    tokens_per_second: float
    velocity_sparkline: str
    sla_target_ttft_ms: float
    sla_passed: bool


def _build_ascii_sparkline(values: list[float], bins: int = 16) -> str:
    """Render compact ASCII sparkline bar visualization."""
    if not values:
        return " "
    min_val = min(values)
    max_val = max(values)
    val_range = max(1e-5, max_val - min_val)

    bars = [" ", "▂", "▃", "▄", "▅", "▆", "▇", "█"]

    # Bucket values into bins
    chunk_size = max(1, len(values) // bins)
    bucketed = []
    for i in range(0, len(values), chunk_size):
        chunk = values[i : i + chunk_size]
        bucketed.append(sum(chunk) / len(chunk))

    sparkline = []
    for v in bucketed:
        norm = (v - min_val) / val_range
        idx = min(len(bars) - 1, max(0, int(norm * (len(bars) - 1))))
        sparkline.append(bars[idx])
    return "".join(sparkline)


class AsyncStreamingProfiler:
    """Asynchronous Profiler for LLM token streaming streams."""

    def __init__(self, target_ttft_sla_ms: float = 400.0, target_p99_itl_ms: float = 60.0):
        self.target_ttft_sla_ms = target_ttft_sla_ms
        self.target_p99_itl_ms = target_p99_itl_ms

    async def profile_stream(
        self,
        stream_iter: AsyncIterator[str],
        model_name: str = "gpt-4o",
    ) -> StreamingProfileReport:
        """Profile an active asynchronous token stream."""
        start_time = time.perf_counter()
        first_token_time: Optional[float] = None
        prev_time = start_time

        arrivals: list[TokenArrival] = []
        itl_deltas: list[float] = []
        total_tokens = 0

        async for chunk in stream_iter:
            curr_time = time.perf_counter()
            delta_ms = (curr_time - prev_time) * 1000.0

            if first_token_time is None:
                first_token_time = curr_time
            else:
                itl_deltas.append(delta_ms)

            words = max(1, len(chunk.split()))
            total_tokens += words
            arrivals.append(
                TokenArrival(
                    chunk_text=chunk,
                    arrival_time_s=curr_time,
                    latency_delta_ms=delta_ms,
                    token_count=words,
                )
            )
            prev_time = curr_time

        end_time = time.perf_counter()
        total_duration_ms = (end_time - start_time) * 1000.0
        ttft_ms = ((first_token_time - start_time) * 1000.0) if first_token_time else total_duration_ms

        # Compute ITL statistics
        if itl_deltas:
            sorted_itls = sorted(itl_deltas)
            mean_itl = statistics.mean(itl_deltas)
            median_itl = statistics.median(itl_deltas)
            jitter_std = statistics.stdev(itl_deltas) if len(itl_deltas) > 1 else 0.0

            def _pct(p: float) -> float:
                k = (len(sorted_itls) - 1) * p
                f = math.floor(k)
                c = math.ceil(k)
                if f == c:
                    return sorted_itls[int(k)]
                return sorted_itls[f] * (c - k) + sorted_itls[c] * (k - f)

            p90 = _pct(0.90)
            p95 = _pct(0.95)
            p99 = _pct(0.99)
        else:
            mean_itl = 0.0
            median_itl = 0.0
            jitter_std = 0.0
            p90 = 0.0
            p95 = 0.0
            p99 = 0.0

        tps = (total_tokens / (total_duration_ms / 1000.0)) if total_duration_ms > 0 else 0.0
        sparkline = _build_ascii_sparkline(itl_deltas if itl_deltas else [0.0])

        sla_passed = (ttft_ms <= self.target_ttft_sla_ms) and (p99 <= self.target_p99_itl_ms or not itl_deltas)

        return StreamingProfileReport(
            model_name=model_name,
            total_tokens=total_tokens,
            total_stream_duration_ms=round(total_duration_ms, 2),
            ttft_ms=round(ttft_ms, 2),
            mean_itl_ms=round(mean_itl, 2),
            median_itl_ms=round(median_itl, 2),
            p90_itl_ms=round(p90, 2),
            p95_itl_ms=round(p95, 2),
            p99_itl_ms=round(p99, 2),
            jitter_std_dev_ms=round(jitter_std, 2),
            tokens_per_second=round(tps, 1),
            velocity_sparkline=sparkline,
            sla_target_ttft_ms=self.target_ttft_sla_ms,
            sla_passed=sla_passed,
        )

    async def simulate_streaming_profiling(
        self,
        prompt: str,
        model_name: str = "gpt-4o",
        token_count: int = 30,
        base_ttft_ms: float = 180.0,
        base_itl_ms: float = 25.0,
    ) -> StreamingProfileReport:
        """Deterministic simulation of token streaming with jitter for offline profiling."""

        async def _generator() -> AsyncIterator[str]:
            # Simulate initial prefill TTFT flight time
            await asyncio.sleep(base_ttft_ms / 1000.0)
            yield "Hello"

            # Simulate streamed token generation with synthetic network jitter
            import random

            rng = random.Random(hash(prompt) % 100000)
            words = [
                "world",
                "this",
                "is",
                "a",
                "high-performance",
                "low-latency",
                "inference",
                "stream",
                "profiled",
                "by",
                "promptdiff",
                "engine",
                "with",
                "microsecond",
                "precision",
                "jitter",
                "telemetry",
            ]
            for i in range(max(1, token_count - 1)):
                w = words[i % len(words)]
                # Add synthetic jitter (e.g. 15ms - 45ms)
                jitter = (rng.random() - 0.5) * 15.0
                step_ms = max(5.0, base_itl_ms + jitter)
                await asyncio.sleep(step_ms / 1000.0)
                yield f" {w}"

        return await self.profile_stream(_generator(), model_name=model_name)

    async def profile_provider_stream(
        self,
        provider: BaseLLMProvider,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = 2048,
    ) -> StreamingProfileReport:
        """Profile live token stream directly from a BaseLLMProvider instance."""
        stream_iter = provider.generate_stream(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return await self.profile_stream(stream_iter, model_name=provider.model_name)
