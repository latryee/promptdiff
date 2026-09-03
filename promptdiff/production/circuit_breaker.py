"""Token Quota & Runaway Cost Circuit Breaker.

Provides stateful circuit-breaking (CLOSED, OPEN, HALF_OPEN) to safeguard
production LLM gateways against runaway loops, token quota breaches, and provider error cascades.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class CircuitBreakerStatus:
    """Current operating state of the LLM gateway circuit breaker."""

    state: str  # CLOSED, OPEN, HALF_OPEN
    total_tokens_consumed: int
    total_cost_usd: float
    failure_count: int
    rejection_count: int
    time_until_reset_sec: float


class LLMCircuitBreaker:
    """Stateful circuit breaker enforcing token and financial limits."""

    def __init__(
        self,
        max_tokens_per_window: int = 100_000,
        max_cost_usd_per_window: float = 10.0,
        failure_threshold: int = 5,
        cooldown_sec: float = 30.0,
    ):
        self.max_tokens = max_tokens_per_window
        self.max_cost = max_cost_usd_per_window
        self.failure_threshold = failure_threshold
        self.cooldown_sec = cooldown_sec

        self.state = "CLOSED"
        self.tokens_consumed = 0
        self.cost_accumulated = 0.0
        self.consecutive_failures = 0
        self.rejections = 0
        self.last_state_change = time.time()

    def can_execute(self, estimated_tokens: int = 500, estimated_cost: float = 0.01) -> bool:
        """Check if request is permitted to proceed."""
        now = time.time()

        if self.state == "OPEN":
            if now - self.last_state_change >= self.cooldown_sec:
                self.state = "HALF_OPEN"
                self.last_state_change = now
                return True
            self.rejections += 1
            return False

        if (self.tokens_consumed + estimated_tokens > self.max_tokens) or (
            self.cost_accumulated + estimated_cost > self.max_cost
        ):
            self.trip("Budget/Token quota window exceeded.")
            self.rejections += 1
            return False

        return True

    def record_success(self, tokens_used: int, cost_usd: float) -> None:
        """Record successful completion."""
        self.tokens_consumed += tokens_used
        self.cost_accumulated += cost_usd
        self.consecutive_failures = 0
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
            self.last_state_change = time.time()

    def record_failure(self) -> None:
        """Record provider error or timeout."""
        self.consecutive_failures += 1
        if self.consecutive_failures >= self.failure_threshold:
            self.trip(f"Exceeded {self.failure_threshold} consecutive provider failures.")

    def trip(self, reason: str = "") -> None:
        """Trip circuit breaker to OPEN."""
        self.state = "OPEN"
        self.last_state_change = time.time()

    def get_status(self) -> CircuitBreakerStatus:
        """Query circuit breaker telemetry."""
        now = time.time()
        elapsed = now - self.last_state_change
        remaining = max(0.0, self.cooldown_sec - elapsed) if self.state == "OPEN" else 0.0

        return CircuitBreakerStatus(
            state=self.state,
            total_tokens_consumed=self.tokens_consumed,
            total_cost_usd=round(self.cost_accumulated, 4),
            failure_count=self.consecutive_failures,
            rejection_count=self.rejections,
            time_until_reset_sec=round(remaining, 1),
        )
