"""Length & Verbosity Drift Evaluator."""

from __future__ import annotations

from promptdiff.core.models import EvaluatorScore, RunResult, TestCase
from promptdiff.evaluators.base import BaseEvaluator


class LengthDriftEvaluator(BaseEvaluator):
    """Tracks token and character output length changes."""

    name: str = "length_drift"
    description: str = "Monitors output verbosity and token count inflation/deflation"

    def evaluate(
        self,
        v1_result: RunResult,
        v2_result: RunResult,
        test_case: TestCase,
    ) -> EvaluatorScore:
        v1_len = len(v1_result.output)
        v2_len = len(v2_result.output)

        delta_chars = v2_len - v1_len
        delta_pct = (delta_chars / v1_len * 100.0) if v1_len > 0 else 0.0

        v1_tok = v1_result.completion_tokens
        v2_tok = v2_result.completion_tokens
        delta_tok = v2_tok - v1_tok

        sign = "+" if delta_chars > 0 else ""
        pct_sign = "+" if delta_pct > 0 else ""

        message = f"{v1_tok} tok ({v1_len} char) → {v2_tok} tok ({v2_len} char) [{pct_sign}{delta_pct:.1f}%]"

        return EvaluatorScore(
            name=self.name,
            v1_score=v1_tok,
            v2_score=v2_tok,
            delta=float(delta_tok),
            delta_pct=round(delta_pct, 2),
            passed=True,
            message=message,
            details={
                "v1_chars": v1_len,
                "v2_chars": v2_len,
                "v1_tokens": v1_tok,
                "v2_tokens": v2_tok,
            },
        )
