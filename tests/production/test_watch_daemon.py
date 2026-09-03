"""Tests migrated to feature-focused test suite."""

from __future__ import annotations

import httpx
import pytest

from promptdiff.core.models import PromptVersion
from promptdiff.production.watch_daemon import PromptHealthDaemon


@pytest.mark.asyncio
async def test_watch_health_daemon() -> None:
    """Test semantic drift and real-time health daemon."""
    pv = PromptVersion(name="daemon_p", template="Support: {{query}}")
    daemon = PromptHealthDaemon(
        prompt_version=pv, golden_reference_outputs=["Hello customer, how can I help you today?"], drift_threshold=0.50
    )

    # 1. Healthy call
    alert1 = await daemon.evaluate_live_call("Hello customer, how can I assist you today?")
    assert alert1 is None

    # 2. Drifted call
    alert2 = await daemon.evaluate_live_call("Quantum mechanics is the study of matter and radiation.")
    assert alert2 is not None
    assert alert2.alert_level in ("WARNING", "CRITICAL")

    status = daemon.get_health_status()
    assert status.total_calls_monitored == 2
    assert status.drift_alerts_triggered == 1


@pytest.mark.asyncio
async def test_watch_health_daemon_webhook_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test async webhook dispatching on drift alert."""
    pv = PromptVersion(name="daemon_p", template="Support: {{query}}")
    daemon = PromptHealthDaemon(
        prompt_version=pv,
        golden_reference_outputs=["Hello customer, how can I help you today?"],
        drift_threshold=0.50,
        webhook_url="https://webhook.example.com/alerts",
    )

    posted_payloads: list[dict] = []

    async def mock_post(self, url: str, json: dict) -> None:  # noqa: ARG001
        posted_payloads.append(json)

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    alert = await daemon.evaluate_live_call("Quantum mechanics is the study of matter.")
    assert alert is not None
    assert len(posted_payloads) == 1
    assert "Semantic Drift Detected" in posted_payloads[0]["text"]
