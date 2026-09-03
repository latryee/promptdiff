"""Tests migrated to feature-focused test suite."""

from __future__ import annotations

from promptdiff.production.drift_detector import CUSUMDriftDetector


def test_cusum_drift_detector() -> None:
    """Test sequential CUSUM change-point drift detector."""
    detector = CUSUMDriftDetector(target_mean=100.0, slack_k=10.0, threshold_h=40.0)
    # Series with abrupt latency spike
    latencies = [100.0, 102.0, 98.0, 101.0, 150.0, 160.0, 175.0, 190.0]
    report = detector.analyze_series(latencies, metric_name="latency_ms")
    assert report.drift_detected is True
    assert report.change_point_index is not None
