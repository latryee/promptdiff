"""Memory leak and long-running stability tests for PromptHealthDaemon."""

from __future__ import annotations

import tracemalloc

import pytest

from promptdiff.core.models import PromptVersion
from promptdiff.production.watch_daemon import PromptHealthDaemon


@pytest.mark.asyncio
async def test_watch_daemon_bounded_sliding_window_memory() -> None:
    """Simulate 10,000 live evaluation cycles and assert strict memory bounding via tracemalloc."""
    pv = PromptVersion(name="prod_bot", template="Support question: {{q}}")
    golden_refs = [
        "Welcome to our customer service portal. How can I help you today?",
        "Thank you for contacting billing. Your balance is zero dollars.",
    ]

    daemon = PromptHealthDaemon(
        prompt_version=pv,
        golden_reference_outputs=golden_refs,
        drift_threshold=0.60,
        max_history=1000,
    )

    tracemalloc.start()
    _, initial_memory = tracemalloc.get_traced_memory()

    # Simulate 10,000 live evaluation cycles
    num_cycles = 10_000
    for i in range(num_cycles):
        if i % 3 == 0:
            sample = f"Welcome to our service portal. Request number {i} is processed."
        elif i % 3 == 1:
            sample = f"Completely unrelated text about astrophysics and stars {i}."
        else:
            sample = f"Thank you for contacting billing team about account {i}."

        await daemon.evaluate_live_call(sample)

    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # 1. Verify sliding window deques are strictly clamped to max_history (1000)
    assert len(daemon.recent_scores) == 1000
    assert len(daemon.recent_alerts) <= 1000
    assert daemon.total_monitored == num_cycles
    assert daemon.alerts_count > 0

    # 2. Assert bounded memory growth (< 15 MB peak during 10,000 iterations)
    peak_mb = peak_memory / (1024 * 1024)
    assert peak_mb < 15.0, f"Peak memory usage was {peak_mb:.2f} MB, which exceeds 15 MB limit"

    # 3. Verify health status computes over bounded sliding window
    status = daemon.get_health_status()
    assert status.total_calls_monitored == num_cycles
    assert 0.0 <= status.avg_health_score_pct <= 100.0

    # 4. Verify explicit purge_history flushes deque memory
    daemon.purge_history()
    assert len(daemon.recent_scores) == 0
    assert len(daemon.recent_alerts) == 0
