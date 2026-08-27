"""Latency Evaluator for measuring response duration regressions."""

from __future__ import annotations

from promptdiff.core.models import EvaluatorScore, RunResult, TestCase
from promptdiff.evaluators.base import BaseEvaluator


class LatencyEvaluator(BaseEvaluator):
    """Measures execution latency difference in milliseconds."""

    name: str = "latency"
    description: str = "Measures execution latency delta in milliseconds"

    def evaluate(
        self,
        v1_result: RunResult,
        v2_result: RunResult,
        test_case: TestCase,
    ) -> EvaluatorScore:
        v1_ms = v1_result.latency_ms
        v2_ms = v2_result.latency_ms

        delta_ms = v2_ms - v1_ms
        delta_pct = (delta_ms / v1_ms * 100.0) if v1_ms > 0 else 0.0

        # Sign formatting
        sign = "+" if delta_ms > 0 else ""
        pct_sign = "+" if delta_pct > 0 else ""

        message = f"{v1_ms:.1f}ms → {v2_ms:.1f}ms ({pct_sign}{delta_pct:.1f}%)"

        return EvaluatorScore(
            name=self.name,
            v1_score=round(v1_ms, 2),
            v2_score=round(v2_ms, 2),
            delta=round(delta_ms, 2),
            delta_pct=round(delta_pct, 2),
            passed=delta_pct <= 25.0,  # Default heuristic threshold
            message=message,
            details={
                "v1_latency_ms": v1_ms,
                "v2_latency_ms": v2_ms,
                "delta_ms": delta_ms,
                "delta_pct": delta_pct,
            },
        )
