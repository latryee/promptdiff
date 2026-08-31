"""Production SLA & Concurrency Budget Breaker Simulator for promptdiff (promptdiff budget / sla)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Optional

from promptdiff.core.models import PromptVersion, TestCase
from promptdiff.core.runner import PromptDiffRunner
from promptdiff.evaluators.registry import get_evaluators
from promptdiff.providers.registry import get_provider

logger = logging.getLogger("promptdiff.production.sla")


@dataclass
class SLABreachItem:
    """Individual SLA threshold breach."""

    test_case_id: str
    breach_type: str  # LATENCY, COST, FORMAT
    threshold_value: float
    observed_value: float
    message: str


@dataclass
class SLABudgetReport:
    """Complete SLA stress test report."""

    prompt_name: str
    model_name: str
    total_requests: int
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    max_latency_ms: float
    avg_cost_per_req_usd: float
    max_cost_per_req_usd: float
    sla_passed: bool
    breaches: list[SLABreachItem] = field(default_factory=list)


class SLABudgetSimulator:
    """Simulates high load and verifies p95/p99 latency ceilings and per-request cost budgets."""

    def __init__(
        self,
        prompt_version: PromptVersion,
        test_cases: list[TestCase],
        max_p99_latency_ms: float = 1500.0,
        max_cost_per_request_usd: float = 0.005,
        model_name: str = "gpt-4o",
        concurrency: int = 10,
        force_mock: bool = False,
    ):
        self.prompt_version = prompt_version
        self.test_cases = test_cases
        self.max_p99_latency_ms = max_p99_latency_ms
        self.max_cost_per_request_usd = max_cost_per_request_usd
        self.model_name = model_name
        self.concurrency = concurrency
        self.force_mock = force_mock

    async def run_stress_test(
        self,
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> SLABudgetReport:
        """Execute concurrent stress run and compute percentile SLA compliance."""
        runner = PromptDiffRunner(
            v1_prompt=self.prompt_version,
            v2_prompt=self.prompt_version,
            provider_v1=get_provider(model_name=self.model_name, force_mock=self.force_mock),
            provider_v2=get_provider(model_name=self.model_name, force_mock=self.force_mock),
            evaluators=get_evaluators(["latency", "cost"]),
            concurrency=self.concurrency,
        )

        diff_report = await runner.run(self.test_cases, progress_cb=progress_cb)

        latencies = sorted([float(c.v2_result.latency_ms) for c in diff_report.comparisons])
        costs = [float(c.v2_result.cost_usd) for c in diff_report.comparisons]
        n = len(latencies)

        p50 = latencies[int(n * 0.50)] if n else 0.0
        p95 = latencies[min(n - 1, int(n * 0.95))] if n else 0.0
        p99 = latencies[min(n - 1, int(n * 0.99))] if n else 0.0
        max_lat = max(latencies) if n else 0.0

        avg_cost = sum(costs) / n if n else 0.0
        max_cost = max(costs) if n else 0.0

        breaches: list[SLABreachItem] = []

        # Check p99 latency SLA
        if p99 > self.max_p99_latency_ms:
            breaches.append(
                SLABreachItem(
                    test_case_id="AGGREGATE_P99",
                    breach_type="LATENCY_P99_EXCEEDED",
                    threshold_value=self.max_p99_latency_ms,
                    observed_value=p99,
                    message=f"P99 latency ({p99:.1f}ms) exceeds SLA ceiling ({self.max_p99_latency_ms:.1f}ms)",
                )
            )

        # Check per-case cost SLA
        for comp in diff_report.comparisons:
            c_val = comp.v2_result.cost_usd
            if c_val > self.max_cost_per_request_usd:
                breaches.append(
                    SLABreachItem(
                        test_case_id=comp.test_case.id,
                        breach_type="COST_BUDGET_EXCEEDED",
                        threshold_value=self.max_cost_per_request_usd,
                        observed_value=c_val,
                        message=f"Request cost (${c_val:.6f}) exceeds per-request budget ceiling (${self.max_cost_per_request_usd:.6f})",
                    )
                )

        return SLABudgetReport(
            prompt_name=self.prompt_version.name,
            model_name=self.model_name,
            total_requests=n,
            p50_latency_ms=round(p50, 1),
            p95_latency_ms=round(p95, 1),
            p99_latency_ms=round(p99, 1),
            max_latency_ms=round(max_lat, 1),
            avg_cost_per_req_usd=round(avg_cost, 6),
            max_cost_per_req_usd=round(max_cost, 6),
            sla_passed=len(breaches) == 0,
            breaches=breaches,
        )
