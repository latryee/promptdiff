"""Tests migrated to feature-focused test suite."""

from __future__ import annotations

import promptdiff


def test_sla_simulator() -> None:
    """Test SLA budget stress simulator."""
    rep = promptdiff.sla_stress(
        prompt="Answer: {{query}}",
        dataset=[{"id": "1", "vars": {"query": "hi"}}],
        mock=True,
    )
    assert rep.total_requests == 1
    assert rep.p50_latency_ms >= 0.0
