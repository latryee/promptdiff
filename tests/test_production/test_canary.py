"""Tests migrated to feature-focused test suite."""

from __future__ import annotations

import promptdiff
from promptdiff.production.canary import CanaryConfigGenerator


def test_canary_config_generator() -> None:
    """Test Canary rollout generator."""
    report = promptdiff.compare(
        v1="a: {{query}}", v2="b: {{query}}", dataset=[{"id": "1", "vars": {"query": "x"}}], mock=True
    )
    cfg = CanaryConfigGenerator(report=report).generate()
    assert cfg.v1_weight_pct + cfg.v2_weight_pct == 100
    assert "rollout" in cfg.launchdarkly_json["fallthrough"]
