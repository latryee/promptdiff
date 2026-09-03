"""Production Model Cascade Router & Enterprise Scale ROI Forecaster.

Simulates and evaluates tiered LLM routing cascades (Tier 1: Fast/Affordable,
Tier 2: Flagship, Tier 3: Deep Reasoning) based on uncertainty and complexity scoring,
calculating financial savings ($) and latency gains at scale (1M+ reqs/month).
"""

from __future__ import annotations

from dataclasses import dataclass

from promptdiff.pricing import calculate_cost


@dataclass
class CascadeTier:
    """A configured model execution tier in the routing cascade."""

    tier_name: str
    model_name: str
    confidence_threshold: float  # minimum confidence required to terminate at this tier
    avg_latency_ms: float
    description: str


@dataclass
class RouteDecision:
    """Outcome of routing a specific query through the cascade."""

    query: str
    complexity_score: float  # 0.0 (simple) to 1.0 (complex)
    selected_tier: str
    selected_model: str
    confidence_score: float
    escalated: bool
    estimated_cost_usd: float
    estimated_latency_ms: float
    routing_rationale: str


@dataclass
class CascadeROIForecast:
    """Projected business and infrastructure impact of model cascading."""

    monthly_request_volume: int
    baseline_model: str
    baseline_monthly_cost_usd: float
    cascade_monthly_cost_usd: float
    monthly_savings_usd: float
    savings_percentage: float
    tier_distribution_pct: dict[str, float]
    avg_latency_reduction_pct: float
    annual_savings_usd: float


class ConfidenceCascadeRouter:
    """Uncertainty-Aware Model Cascade Engine."""

    def __init__(
        self,
        tier1_model: str = "gpt-4o-mini",
        tier2_model: str = "gpt-4o",
        tier3_model: str = "claude-3-7-sonnet",
    ):
        self.tier1 = CascadeTier(
            tier_name="Tier 1 (Fast / Edge)",
            model_name=tier1_model,
            confidence_threshold=0.75,
            avg_latency_ms=120.0,
            description="Ultra-fast, low-cost model for straightforward classification and retrieval.",
        )
        self.tier2 = CascadeTier(
            tier_name="Tier 2 (Standard Flagship)",
            model_name=tier2_model,
            confidence_threshold=0.50,
            avg_latency_ms=280.0,
            description="Balanced multimodal model for structured extraction and standard synthesis.",
        )
        self.tier3 = CascadeTier(
            tier_name="Tier 3 (Deep Reasoning)",
            model_name=tier3_model,
            confidence_threshold=0.00,
            avg_latency_ms=750.0,
            description="Frontier reasoning engine for complex logic, multi-step math, and ambiguous edge cases.",
        )

    def evaluate_complexity(self, query: str) -> float:
        """Estimate query complexity and reasoning requirement (0.0 to 1.0)."""
        score = 0.1  # base

        # Length factor
        words = len(query.split())
        if words > 100:
            score += 0.3
        elif words > 40:
            score += 0.15

        # Reasoning & constraint cues
        reasoning_keywords = [
            "why",
            "explain step by step",
            "proof",
            "calculate",
            "compare and contrast",
            "edge case",
            "algorithm",
            "root cause",
            "mathematical",
            "contradiction",
        ]
        for kw in reasoning_keywords:
            if kw in query.lower():
                score += 0.15

        # Schema constraints
        if "{" in query and "}" in query or "json" in query.lower() or "schema" in query.lower():
            score += 0.15

        return min(1.0, round(score, 2))

    def route_query(self, query: str, prompt_tokens: int = 150, completion_tokens: int = 80) -> RouteDecision:
        """Simulate cascade routing decision for a query."""
        complexity = self.evaluate_complexity(query)
        confidence = 1.0 - (complexity * 0.8)

        if confidence >= self.tier1.confidence_threshold:
            chosen = self.tier1
            escalated = False
            rationale = "Low complexity; handled efficiently by Tier 1 small model."
        elif confidence >= self.tier2.confidence_threshold:
            chosen = self.tier2
            escalated = True
            rationale = "Moderate complexity / structural requirements; escalated to Tier 2 flagship."
        else:
            chosen = self.tier3
            escalated = True
            rationale = "High complexity / multi-step reasoning; escalated to Tier 3 frontier model."

        cost = calculate_cost(chosen.model_name, prompt_tokens, completion_tokens)

        return RouteDecision(
            query=query,
            complexity_score=complexity,
            selected_tier=chosen.tier_name,
            selected_model=chosen.model_name,
            confidence_score=round(confidence, 2),
            escalated=escalated,
            estimated_cost_usd=cost,
            estimated_latency_ms=chosen.avg_latency_ms,
            routing_rationale=rationale,
        )

    def forecast_roi(
        self,
        queries: list[str],
        monthly_volume: int = 1_000_000,
        baseline_model: str = "gpt-4o",
        avg_prompt_tokens: int = 200,
        avg_completion_tokens: int = 100,
    ) -> CascadeROIForecast:
        """Forecast production ROI and SLA impact over monthly scale."""
        decisions = [self.route_query(q, avg_prompt_tokens, avg_completion_tokens) for q in queries]
        if not decisions:
            decisions = [self.route_query("Sample enterprise query", avg_prompt_tokens, avg_completion_tokens)]

        total_sample_cost = sum(d.estimated_cost_usd for d in decisions)
        avg_cascade_cost_per_req = total_sample_cost / len(decisions)

        baseline_cost_per_req = calculate_cost(baseline_model, avg_prompt_tokens, avg_completion_tokens)

        baseline_monthly_cost = baseline_cost_per_req * monthly_volume
        cascade_monthly_cost = avg_cascade_cost_per_req * monthly_volume
        monthly_savings = max(0.0, baseline_monthly_cost - cascade_monthly_cost)
        savings_pct = (monthly_savings / baseline_monthly_cost * 100.0) if baseline_monthly_cost > 0 else 0.0

        tier_counts = {
            "tier1": sum(1 for d in decisions if "Tier 1" in d.selected_tier),
            "tier2": sum(1 for d in decisions if "Tier 2" in d.selected_tier),
            "tier3": sum(1 for d in decisions if "Tier 3" in d.selected_tier),
        }
        total_dec = max(1, len(decisions))
        tier_dist = {
            self.tier1.model_name: round((tier_counts["tier1"] / total_dec) * 100.0, 1),
            self.tier2.model_name: round((tier_counts["tier2"] / total_dec) * 100.0, 1),
            self.tier3.model_name: round((tier_counts["tier3"] / total_dec) * 100.0, 1),
        }

        # Average latency comparison
        base_latency = 280.0
        avg_cascade_latency = sum(d.estimated_latency_ms for d in decisions) / total_dec
        latency_reduction = ((base_latency - avg_cascade_latency) / base_latency) * 100.0

        return CascadeROIForecast(
            monthly_request_volume=monthly_volume,
            baseline_model=baseline_model,
            baseline_monthly_cost_usd=round(baseline_monthly_cost, 2),
            cascade_monthly_cost_usd=round(cascade_monthly_cost, 2),
            monthly_savings_usd=round(monthly_savings, 2),
            savings_percentage=round(savings_pct, 1),
            tier_distribution_pct=tier_dist,
            avg_latency_reduction_pct=round(max(0.0, latency_reduction), 1),
            annual_savings_usd=round(monthly_savings * 12, 2),
        )
