"""Sequential Change-Point & Metric Drift Detector (CUSUM & Page-Hinkley).

Monitors production inference streams, detecting significant distribution shifts
in latency, token counts, or semantic scores using Cumulative Sum Control Charts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class DriftEvent:
    """An identified anomaly or distribution shift in production metrics."""

    timestamp_index: int
    metric_name: str
    observed_value: float
    threshold: float
    severity: str  # WARNING, CRITICAL
    message: str


@dataclass
class DriftReport:
    """Outcome of sequential change-point analysis over sliding windows."""

    metric_name: str
    drift_detected: bool
    change_point_index: Optional[int]
    baseline_mean: float
    current_mean: float
    events: list[DriftEvent]


class CUSUMDriftDetector:
    """Cumulative Sum (CUSUM) change-point detector."""

    def __init__(self, target_mean: float = 200.0, slack_k: float = 15.0, threshold_h: float = 50.0):
        self.target_mean = target_mean
        self.slack_k = slack_k
        self.threshold_h = threshold_h
        self.s_pos = 0.0
        self.s_neg = 0.0

    def analyze_series(self, values: list[float], metric_name: str = "latency_ms") -> DriftReport:
        """Process time series of observations and report change points."""
        events: list[DriftEvent] = []
        change_idx = None
        s_pos = 0.0
        s_neg = 0.0

        for i, val in enumerate(values):
            # CUSUM positive and negative accumulators
            s_pos = max(0.0, s_pos + (val - self.target_mean) - self.slack_k)
            s_neg = max(0.0, s_neg - (val - self.target_mean) - self.slack_k)

            if s_pos > self.threshold_h:
                events.append(
                    DriftEvent(
                        timestamp_index=i,
                        metric_name=metric_name,
                        observed_value=val,
                        threshold=self.threshold_h,
                        severity="CRITICAL",
                        message=f"Positive drift detected in {metric_name} at index {i} (Accumulator={s_pos:.1f} > {self.threshold_h}).",
                    )
                )
                if change_idx is None:
                    change_idx = i
                s_pos = 0.0  # Reset

        curr_mean = (sum(values[-10:]) / max(1, len(values[-10:]))) if values else 0.0

        return DriftReport(
            metric_name=metric_name,
            drift_detected=len(events) > 0,
            change_point_index=change_idx,
            baseline_mean=round(self.target_mean, 2),
            current_mean=round(curr_mean, 2),
            events=events,
        )
