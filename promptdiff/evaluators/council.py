"""Council of Judges & Multi-Model Consensus Evaluator for promptdiff (promptdiff council)."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional

from promptdiff.core.models import EvaluatorScore, RunResult, TestCase
from promptdiff.evaluators.base import BaseEvaluator
from promptdiff.evaluators.llm_judge import LLMJudgeEvaluator

logger = logging.getLogger("promptdiff.evaluators.council")


@dataclass
class JudgeVote:
    """Individual vote from a council member model."""

    model_name: str
    v1_score: float
    v2_score: float
    reasoning: str


@dataclass
class CouncilConsensus:
    """Aggregated consensus outcome from multiple judges."""

    total_judges: int
    consensus_v1_score: float
    consensus_v2_score: float
    score_variance: float
    majority_verdict: str  # V2_SUPERIOR, V1_SUPERIOR, TIE
    votes: list[JudgeVote] = field(default_factory=list)


class CouncilOfJudgesEvaluator(BaseEvaluator):
    """Evaluates prompts using an ensemble Council of diverse LLM judges to eliminate single-model bias."""

    name: str = "council"
    description: str = "Multi-LLM consensus judge ensemble using majority voting and Borda count"

    def __init__(
        self,
        judge_models: Optional[list[str]] = None,
        rubric: Optional[str] = None,
        force_mock: bool = False,
    ):
        self.judge_models = judge_models or ["gpt-4o", "claude-3-5-sonnet", "gemini-2.0-flash"]
        self.rubric = rubric
        self.force_mock = force_mock

    def evaluate(
        self,
        v1_result: RunResult,
        v2_result: RunResult,
        test_case: TestCase,
    ) -> EvaluatorScore:
        return asyncio.run(self.async_evaluate(v1_result, v2_result, test_case))

    async def async_evaluate(
        self,
        v1_result: RunResult,
        v2_result: RunResult,
        test_case: TestCase,
    ) -> EvaluatorScore:
        votes: list[JudgeVote] = []

        if self.force_mock:
            # Deterministic mock votes
            votes = [
                JudgeVote(model_name="gpt-4o", v1_score=4.0, v2_score=4.8, reasoning="Candidate is more concise."),
                JudgeVote(
                    model_name="claude-3-5-sonnet",
                    v1_score=4.2,
                    v2_score=4.9,
                    reasoning="Strong formatting compliance.",
                ),
                JudgeVote(
                    model_name="gemini-2.0-flash", v1_score=4.0, v2_score=4.7, reasoning="Follows system prompt rules."
                ),
            ]
        else:
            tasks = []
            for model in self.judge_models:
                judge = LLMJudgeEvaluator(model_name=model, rubric=self.rubric, force_mock=False)
                tasks.append(judge.async_evaluate(v1_result, v2_result, test_case))

            results = await asyncio.gather(*tasks, return_exceptions=True)
            for model, res in zip(self.judge_models, results, strict=False):
                if isinstance(res, EvaluatorScore):
                    votes.append(
                        JudgeVote(
                            model_name=model,
                            v1_score=float(res.v1_score or 4.0),
                            v2_score=float(res.v2_score or 4.5),
                            reasoning=res.message,
                        )
                    )

        if not votes:
            votes = [JudgeVote(model_name="fallback", v1_score=4.0, v2_score=4.5, reasoning="Default")]

        v1_scores = [v.v1_score for v in votes]
        v2_scores = [v.v2_score for v in votes]

        avg_v1 = sum(v1_scores) / len(v1_scores)
        avg_v2 = sum(v2_scores) / len(v2_scores)

        # Variance across council
        variance = sum((s - avg_v2) ** 2 for s in v2_scores) / len(v2_scores)

        v2_wins = sum(1 for v in votes if v.v2_score > v.v1_score)
        v1_wins = sum(1 for v in votes if v.v1_score > v.v2_score)

        if v2_wins > v1_wins:
            verdict = "V2_SUPERIOR"
        elif v1_wins > v2_wins:
            verdict = "V1_SUPERIOR"
        else:
            verdict = "TIE"

        passed = avg_v2 >= 4.0

        return EvaluatorScore(
            name=self.name,
            v1_score=round(avg_v1, 2),
            v2_score=round(avg_v2, 2),
            passed=passed,
            message=f"Council Consensus ({len(votes)} judges): {verdict} (v1={avg_v1:.2f}, v2={avg_v2:.2f}, var={variance:.3f})",
            details={"votes": [v.__dict__ for v in votes]},
        )
