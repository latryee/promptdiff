"""Deep unit tests for production cascade router, replay, and circuit breaker."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from promptdiff.core.models import PromptVersion, TestCase
from promptdiff.production.cascade import CascadeRouteReport, ModelCascadeRouter
from promptdiff.production.circuit_breaker import CircuitBreakerStatus, LLMCircuitBreaker
from promptdiff.production.replay import ReplayReport, ShadowTrafficReplayer

# ============================================================================
# cascade.py tests
# ============================================================================


@pytest.mark.asyncio
async def test_model_cascade_router_optimize() -> None:
    cases = [
        TestCase(id="c1", vars={"query": "Simple greeting"}),
        TestCase(id="c2", vars={"query": "Complex code refactoring request"}),
    ]
    router = ModelCascadeRouter(
        prompt_template="Answer: {{query}}",
        test_cases=cases,
        tier1_model="gpt-4o-mini",
        tier2_model="gpt-4o",
        quality_threshold=4.0,
        force_mock=True,
    )
    report = await router.optimize()

    assert isinstance(report, CascadeRouteReport)
    assert report.total_test_cases == 2
    assert report.tier1_model == "gpt-4o-mini"
    assert report.tier2_model == "gpt-4o"
    assert report.blended_quality_score >= 0.0
    assert report.routing_rules_json["strategy"] == "cascade_fallback"


# ============================================================================
# replay.py tests
# ============================================================================


def test_shadow_traffic_replayer_sanitize_pii() -> None:
    pv = PromptVersion(name="cand", template="Help: {{query}}")
    replayer = ShadowTrafficReplayer(candidate_prompt=pv, force_mock=True)

    raw_text = "Contact support at john.doe@example.com or phone (555) 123-4567. My card is 4111 2222 3333 4444."
    cleaned, count = replayer.sanitize_pii(raw_text)
    assert count >= 3
    assert "[EMAIL_REDACTED]" in cleaned
    assert "[PHONE_REDACTED]" in cleaned
    assert "[CARD_REDACTED]" in cleaned


def test_shadow_traffic_replayer_load_logs(tmp_path: Path) -> None:
    pv = PromptVersion(name="cand", template="Help: {{query}}")
    replayer = ShadowTrafficReplayer(candidate_prompt=pv, force_mock=True)

    log_file = tmp_path / "traffic.jsonl"
    records = [
        {"query": "Help me reset password for alice@test.com"},
        {"input": "What are your hours? Call 555-123-4567"},
    ]
    log_file.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")

    cases, pii_count = replayer.load_and_sanitize_logs(str(log_file))
    assert len(cases) == 2
    assert pii_count >= 2


@pytest.mark.asyncio
async def test_shadow_traffic_replayer_replay(tmp_path: Path) -> None:
    pv = PromptVersion(name="cand", template="Answer: {{query}}")
    replayer = ShadowTrafficReplayer(candidate_prompt=pv, force_mock=True)

    log_file = tmp_path / "prod.jsonl"
    log_file.write_text(json.dumps({"query": "hello world"}) + "\n", encoding="utf-8")

    rep = await replayer.replay(str(log_file))
    assert isinstance(rep, ReplayReport)
    assert rep.total_logs_processed == 1
    assert rep.pass_rate_pct >= 0.0


# ============================================================================
# circuit_breaker.py tests
# ============================================================================


def test_circuit_breaker_lifecycle() -> None:
    cb = LLMCircuitBreaker(
        max_tokens_per_window=1000,
        max_cost_usd_per_window=0.05,
        failure_threshold=2,
        cooldown_sec=0.1,
    )
    assert cb.state == "CLOSED"
    assert cb.can_execute(estimated_tokens=100, estimated_cost=0.001) is True

    # Record success
    cb.record_success(tokens_used=100, cost_usd=0.001)
    assert cb.tokens_consumed == 100
    assert cb.consecutive_failures == 0

    # Failures trip to OPEN
    cb.record_failure()
    assert cb.state == "CLOSED"
    cb.record_failure()
    assert cb.state == "OPEN"

    # Execution rejected while OPEN
    assert cb.can_execute() is False
    assert cb.rejections > 0

    # Cooldown transition to HALF_OPEN
    time.sleep(0.12)
    assert cb.can_execute() is True
    assert cb.state == "HALF_OPEN"

    # Success closes circuit again
    cb.record_success(tokens_used=50, cost_usd=0.0005)
    assert cb.state == "CLOSED"

    # Exceed quota trips circuit
    assert cb.can_execute(estimated_tokens=5000, estimated_cost=1.0) is False
    assert cb.state == "OPEN"

    status = cb.get_status()
    assert isinstance(status, CircuitBreakerStatus)
    assert status.state == "OPEN"
