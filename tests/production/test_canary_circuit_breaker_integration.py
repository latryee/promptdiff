"""End-to-end integration test combining Canary Rollout and Circuit Breaker."""

from __future__ import annotations

import time

from promptdiff.core.models import DiffReport, RegressionVerdict
from promptdiff.production.canary import CanaryConfigGenerator
from promptdiff.production.circuit_breaker import LLMCircuitBreaker


class ResilientCanaryRouter:
    """Production router that directs traffic using canary weights and guards candidate with a circuit breaker."""

    def __init__(
        self,
        report: DiffReport,
        circuit_breaker: LLMCircuitBreaker,
    ) -> None:
        self.config = CanaryConfigGenerator(report).generate()
        self.breaker = circuit_breaker
        self.v1_dispatches = 0
        self.v2_dispatches = 0
        self.fallback_dispatches = 0

    def route_request(self, user_id: int) -> str:
        """Route request to v1 or v2 based on canary weights, with automatic fail-safe circuit breaker."""
        # Check if hash/modulo falls within v2 canary bucket
        in_canary_bucket = (user_id % 100) < self.config.v2_weight_pct

        if in_canary_bucket:
            # Check circuit breaker before routing to candidate
            if self.breaker.can_execute(estimated_tokens=200, estimated_cost=0.002):
                self.v2_dispatches += 1
                return "v2_candidate"
            else:
                # Breaker tripped: Fallback to baseline
                self.fallback_dispatches += 1
                self.v1_dispatches += 1
                return "v1_baseline_fallback"

        self.v1_dispatches += 1
        return "v1_baseline"


def test_canary_circuit_breaker_end_to_end() -> None:
    """Verify canary candidate failure cascade triggers circuit breaker and instant fallback to baseline."""
    # 1. Generate canary config from a passing DiffReport (recommends 10% canary)
    passed_report = DiffReport(
        run_id="run_pass_01",
        v1_name="v1_baseline",
        v2_name="v2_candidate",
        model_v1="gpt-4o",
        model_v2="gpt-4o",
        comparisons=[],
        verdict=RegressionVerdict(passed=True, cost_delta_pct=2.0, latency_delta_pct=-5.0),
        evaluators=["latency"],
        total_cases=10,
    )

    breaker = LLMCircuitBreaker(
        max_tokens_per_window=50_000,
        max_cost_usd_per_window=5.0,
        failure_threshold=3,
        cooldown_sec=0.2,
    )

    router = ResilientCanaryRouter(report=passed_report, circuit_breaker=breaker)
    assert router.config.v2_weight_pct == 10
    assert router.config.v1_weight_pct == 90

    # 2. Normal routing: users 0-9 hit canary v2 (10%), users 10-99 hit baseline v1 (90%)
    target = router.route_request(user_id=5)
    assert target == "v2_candidate"
    breaker.record_success(tokens_used=150, cost_usd=0.001)

    target_base = router.route_request(user_id=45)
    assert target_base == "v1_baseline"

    # 3. Simulate candidate errors causing breaker to trip
    breaker.record_failure()
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state == "OPEN"

    # 4. Canary bucket traffic should now be safely redirected to baseline fallback
    fallback_target = router.route_request(user_id=5)
    assert fallback_target == "v1_baseline_fallback"
    assert router.fallback_dispatches == 1

    # 5. After cooldown, breaker moves to HALF_OPEN probe
    time.sleep(0.25)
    assert breaker.can_execute() is True
    assert breaker.state == "HALF_OPEN"

    # 6. Success resets breaker back to CLOSED
    breaker.record_success(tokens_used=100, cost_usd=0.001)
    assert breaker.state == "CLOSED"
    assert router.route_request(user_id=5) == "v2_candidate"
