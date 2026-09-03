"""Tests migrated to feature-focused test suite."""

from __future__ import annotations

import promptdiff
from promptdiff.production.routing import ConfidenceCascadeRouter


def test_cascade_router() -> None:
    """Test model cascading router."""
    rep = promptdiff.cascade(
        prompt="Answer: {{query}}",
        dataset=[{"id": "1", "vars": {"query": "hi"}}],
        mock=True,
    )
    assert rep.tier1_route_pct >= 0.0
    assert rep.cost_savings_pct >= 0.0


def test_confidence_cascade_router() -> None:
    """Test query complexity classification, model cascade routing, and ROI forecast."""
    router = ConfidenceCascadeRouter()

    # Simple query should route to Tier 1
    d_simple = router.route_query("Hello there")
    assert "Tier 1" in d_simple.selected_tier

    # Complex reasoning query should escalate
    d_complex = router.route_query(
        "Explain step by step the mathematical proof of why P does not equal NP with formal logic."
    )
    assert d_complex.complexity_score > d_simple.complexity_score
    assert d_complex.escalated is True

    # ROI forecast
    forecast = router.forecast_roi(
        queries=["What is Python?", "Explain quantum entanglement in depth."],
        monthly_volume=100_000,
    )
    assert forecast.monthly_request_volume == 100_000
    assert forecast.baseline_monthly_cost_usd > 0.0
    assert forecast.cascade_monthly_cost_usd > 0.0
    assert forecast.savings_percentage >= 0.0
