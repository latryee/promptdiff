"""Tests migrated to feature-focused test suite."""

from __future__ import annotations

from promptdiff.production.circuit_breaker import LLMCircuitBreaker


def test_llm_circuit_breaker() -> None:
    """Test token budget enforcer and circuit breaker state transitions."""
    breaker = LLMCircuitBreaker(max_tokens_per_window=1000, max_cost_usd_per_window=0.05, failure_threshold=2)
    assert breaker.can_execute(estimated_tokens=200, estimated_cost=0.01) is True

    # Record failures to trip breaker
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state == "OPEN"
    assert breaker.can_execute() is False
