"""Cost Evaluator for calculating token spending deltas."""

from __future__ import annotations

from promptdiff.core.models import EvaluatorScore, RunResult, TestCase
from promptdiff.evaluators.base import BaseEvaluator


class CostEvaluator(BaseEvaluator):
    """Calculates LLM token cost differences based on model pricing tables."""

    name: str = "cost"
    description: str = "Calculates LLM token cost delta in USD"

    def evaluate(
        self,
        v1_result: RunResult,
        v2_result: RunResult,
        test_case: TestCase,
    ) -> EvaluatorScore:
        v1_cost = v1_result.cost_usd
        v2_cost = v2_result.cost_usd

        delta = v2_cost - v1_cost
        delta_pct = (delta / v1_cost * 100.0) if v1_cost > 0 else 0.0

        sign = "+" if delta > 0 else ""
        pct_sign = "+" if delta_pct > 0 else ""

        message = f"${v1_cost:.6f} → ${v2_cost:.6f} ({pct_sign}{delta_pct:.1f}%)"

        return EvaluatorScore(
            name=self.name,
            v1_score=round(v1_cost, 6),
            v2_score=round(v2_cost, 6),
            delta=round(delta, 6),
            delta_pct=round(delta_pct, 2),
            passed=delta_pct <= 20.0,
            message=message,
            details={
                "v1_cost_usd": v1_cost,
                "v2_cost_usd": v2_cost,
                "v1_tokens": v1_result.total_tokens,
                "v2_tokens": v2_result.total_tokens,
                "delta_tokens": v2_result.total_tokens - v1_result.total_tokens,
            },
        )
