"""Real-Time Prompt Health & Semantic Drift Watch Daemon for promptdiff (promptdiff watch / daemon)."""

from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import dataclass
from typing import Optional

import httpx

from promptdiff.core.models import PromptVersion

logger = logging.getLogger("promptdiff.production.watch")


@dataclass
class DriftAlert:
    """Drift alert payload sent to monitoring webhooks."""

    timestamp: str
    prompt_name: str
    observed_similarity: float
    drift_threshold: float
    sample_output: str
    alert_level: str  # WARNING, CRITICAL


@dataclass
class HealthStatus:
    """Live health status of monitored prompt."""

    prompt_name: str
    total_calls_monitored: int
    drift_alerts_triggered: int
    avg_health_score_pct: float
    status: str  # HEALTHY, DRIFT_DETECTED, DEGRADED


class PromptHealthDaemon:
    """Monitors live LLM production output streams and alerts on concept/semantic drift."""

    def __init__(
        self,
        prompt_version: PromptVersion,
        golden_reference_outputs: list[str],
        drift_threshold: float = 0.75,
        webhook_url: Optional[str] = None,
    ):
        self.prompt_version = prompt_version
        self.golden_references = golden_reference_outputs or ["Default reference response"]
        self.drift_threshold = drift_threshold
        self.webhook_url = webhook_url
        self.total_monitored = 0
        self.alerts_count = 0
        self.recent_scores: list[float] = []

    def _compute_similarity(self, text1: str, text2: str) -> float:
        w1 = set(re.findall(r"\w+", text1.lower()))
        w2 = set(re.findall(r"\w+", text2.lower()))
        if not w1 or not w2:
            return 0.0
        return len(w1.intersection(w2)) / math.sqrt(len(w1) * len(w2))

    def evaluate_live_call(self, live_output: str) -> Optional[DriftAlert]:
        """Check a single live production output against golden reference vectors."""
        self.total_monitored += 1

        # Max similarity to any golden reference
        sims = [self._compute_similarity(live_output, ref) for ref in self.golden_references]
        max_sim = max(sims) if sims else 1.0
        self.recent_scores.append(max_sim)

        if max_sim < self.drift_threshold:
            self.alerts_count += 1
            alert = DriftAlert(
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                prompt_name=self.prompt_version.name,
                observed_similarity=round(max_sim, 3),
                drift_threshold=self.drift_threshold,
                sample_output=live_output[:200],
                alert_level="CRITICAL" if max_sim < (self.drift_threshold - 0.20) else "WARNING",
            )
            self._dispatch_webhook(alert)
            return alert

        return None

    def _dispatch_webhook(self, alert: DriftAlert) -> bool:
        if not self.webhook_url:
            return False

        payload = {
            "text": f"🚨 [PromptDiff Alert] Semantic Drift Detected on '{alert.prompt_name}'! Similarity: {alert.observed_similarity:.2f} < {alert.drift_threshold:.2f}",
            "attachments": [
                {
                    "color": "danger" if alert.alert_level == "CRITICAL" else "warning",
                    "fields": [
                        {"title": "Level", "value": alert.alert_level, "short": True},
                        {"title": "Observed Similarity", "value": str(alert.observed_similarity), "short": True},
                        {"title": "Snippet", "value": alert.sample_output, "short": False},
                    ],
                }
            ],
        }
        try:
            with httpx.Client(timeout=4.0) as client:
                client.post(self.webhook_url, json=payload)
                return True
        except Exception as e:
            logger.warning(f"Could not dispatch drift webhook: {e}")
            return False

    def get_health_status(self) -> HealthStatus:
        """Get live health summary."""
        avg_score = (sum(self.recent_scores) / len(self.recent_scores)) if self.recent_scores else 1.0
        if self.alerts_count > 0:
            stat = "DRIFT_DETECTED"
        elif avg_score < 0.80:
            stat = "DEGRADED"
        else:
            stat = "HEALTHY"

        return HealthStatus(
            prompt_name=self.prompt_version.name,
            total_calls_monitored=self.total_monitored,
            drift_alerts_triggered=self.alerts_count,
            avg_health_score_pct=round(avg_score * 100.0, 1),
            status=stat,
        )
