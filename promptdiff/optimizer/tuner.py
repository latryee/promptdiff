"""Hyperparameter Grid Search & Pareto Optimization Engine for LLM Prompts."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from promptdiff.core.models import PromptVersion, TestCase
from promptdiff.evaluators.base import BaseEvaluator
from promptdiff.evaluators.llm_judge import LLMJudgeEvaluator
from promptdiff.evaluators.registry import get_evaluators
from promptdiff.pricing import calculate_cost
from promptdiff.providers.base import BaseLLMProvider
from promptdiff.providers.registry import get_provider

logger = logging.getLogger("promptdiff.optimizer.tuner")


@dataclass
class HyperparameterConfig:
    """Single hyperparameter candidate point."""

    temperature: float
    top_p: float
    max_tokens: int = 2048

    def to_dict(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }


@dataclass
class TuneCandidateResult:
    """Evaluation metrics for a hyperparameter configuration."""

    config: HyperparameterConfig
    avg_judge_score: float
    avg_latency_ms: float
    avg_tokens: float
    total_cost: float
    passed_rate: float
    is_pareto_optimal: bool = False
    utility_score: float = 0.0
    rank: int = 0


@dataclass
class TuningReport:
    """Full hyperparameter grid search report."""

    prompt_name: str
    model_name: str
    total_configs_tested: int
    best_config: HyperparameterConfig
    pareto_candidates: list[TuneCandidateResult]
    all_results: list[TuneCandidateResult] = field(default_factory=list)


def compute_pareto_frontier(results: list[TuneCandidateResult]) -> list[TuneCandidateResult]:
    """Identify non-dominated Pareto-optimal configurations.

    Maximizes: avg_judge_score, passed_rate
    Minimizes: avg_tokens, avg_latency_ms, total_cost
    """
    for cand in results:
        cand.is_pareto_optimal = True

    for i, a in enumerate(results):
        for j, b in enumerate(results):
            if i == j:
                continue
            # Check if b strictly dominates a
            b_better_or_equal = (
                b.avg_judge_score >= a.avg_judge_score
                and b.avg_tokens <= a.avg_tokens
                and b.avg_latency_ms <= a.avg_latency_ms
                and b.total_cost <= a.total_cost
            )
            b_strictly_better = (
                b.avg_judge_score > a.avg_judge_score
                or b.avg_tokens < a.avg_tokens
                or b.avg_latency_ms < a.avg_latency_ms
                or b.total_cost < a.total_cost
            )
            if b_better_or_equal and b_strictly_better:
                a.is_pareto_optimal = False
                break

    # Calculate multi-objective utility score
    max_score = max((r.avg_judge_score for r in results), default=5.0) or 1.0
    max_tok = max((r.avg_tokens for r in results), default=100.0) or 1.0
    max_lat = max((r.avg_latency_ms for r in results), default=500.0) or 1.0

    for r in results:
        norm_score = r.avg_judge_score / max_score
        norm_tok = 1.0 - (r.avg_tokens / max_tok)
        norm_lat = 1.0 - (r.avg_latency_ms / max_lat)
        # Weights: 50% Quality (Judge), 25% Token Efficiency, 25% Latency
        r.utility_score = round(0.50 * norm_score + 0.25 * norm_tok + 0.25 * norm_lat, 4)

    # Sort results by utility score descending
    results.sort(key=lambda r: (r.is_pareto_optimal, r.utility_score), reverse=True)
    for idx, r in enumerate(results, start=1):
        r.rank = idx

    return [r for r in results if r.is_pareto_optimal]


class PromptTuner:
    """Orchestrates Hyperparameter Grid Search across temperature, top_p, and token constraints."""

    def __init__(
        self,
        prompt_version: PromptVersion,
        test_cases: list[TestCase],
        provider: BaseLLMProvider | None = None,
        model_name: str = "gpt-4o",
        temperatures: list[float] | None = None,
        top_ps: list[float] | None = None,
        evaluators: list[BaseEvaluator] | None = None,
        force_mock: bool = False,
        concurrency: int = 6,
    ):
        self.prompt_version = prompt_version
        self.test_cases = test_cases
        self.provider = provider or get_provider(model_name=model_name, force_mock=force_mock)
        self.model_name = model_name
        self.temperatures = temperatures or [0.0, 0.3, 0.7, 1.0]
        self.top_ps = top_ps or [0.7, 0.9, 1.0]
        self.evaluators = evaluators or get_evaluators(["llm_judge", "json_validity"])
        self.force_mock = force_mock
        self.concurrency = concurrency
        self.semaphore = asyncio.Semaphore(concurrency)

    async def _evaluate_config(
        self,
        config: HyperparameterConfig,
        judge_evaluator: LLMJudgeEvaluator,
    ) -> TuneCandidateResult:
        """Evaluate a single hyperparameter configuration on all test cases."""
        latencies = []
        tokens_list = []
        costs = []
        judge_scores = []
        passed_count = 0

        async def _run_single_case(tc: TestCase) -> None:
            rendered = self.prompt_version.render(tc.vars)
            async with self.semaphore:
                try:
                    resp = await self.provider.generate(
                        prompt=rendered,
                        system_prompt=self.prompt_version.system_prompt,
                        temperature=config.temperature,
                        max_tokens=config.max_tokens,
                    )
                    cost = calculate_cost(self.model_name, resp.prompt_tokens, resp.completion_tokens)
                    latencies.append(resp.latency_ms)
                    tokens_list.append(resp.total_tokens)
                    costs.append(cost)

                    # Mock LLM Judge evaluation for configuration
                    if self.force_mock:
                        # Higher temp in mock gives slightly lower determinism score
                        mock_score = max(3.0, 4.8 - (config.temperature * 0.4) - (config.top_p * 0.2))
                        judge_scores.append(round(mock_score, 2))
                    else:
                        from promptdiff.core.models import RunResult

                        res_obj = RunResult(
                            prompt_name="candidate",
                            test_case_id=tc.id,
                            rendered_prompt=rendered,
                            output=resp.output,
                            latency_ms=resp.latency_ms,
                            prompt_tokens=resp.prompt_tokens,
                            completion_tokens=resp.completion_tokens,
                            total_tokens=resp.total_tokens,
                            cost_usd=cost,
                            model=self.model_name,
                        )
                        score = await judge_evaluator.async_evaluate(res_obj, res_obj, tc)
                        judge_scores.append(float(score.v2_score))

                except Exception as e:
                    logger.warning(f"Error evaluating config {config}: {e}")
                    judge_scores.append(1.0)
                    latencies.append(500.0)
                    tokens_list.append(50)
                    costs.append(0.0)

        tasks = [_run_single_case(tc) for tc in self.test_cases]
        await asyncio.gather(*tasks)

        avg_lat = sum(latencies) / max(1, len(latencies))
        avg_tok = sum(tokens_list) / max(1, len(tokens_list))
        tot_cost = sum(costs)
        avg_judge = sum(judge_scores) / max(1, len(judge_scores))
        passed_count = sum(1 for s in judge_scores if s >= 3.5)
        passed_rate = passed_count / max(1, len(judge_scores))

        return TuneCandidateResult(
            config=config,
            avg_judge_score=round(avg_judge, 2),
            avg_latency_ms=round(avg_lat, 1),
            avg_tokens=round(avg_tok, 1),
            total_cost=round(tot_cost, 6),
            passed_rate=round(passed_rate, 2),
        )

    async def tune(
        self,
        progress_cb: Callable[[int, int, str], None] | None = None,
    ) -> TuningReport:
        """Run hyperparameter search across the defined grid."""
        judge_ev = None
        for ev in self.evaluators:
            if isinstance(ev, LLMJudgeEvaluator):
                judge_ev = ev
                break
        if not judge_ev:
            judge_ev = LLMJudgeEvaluator(model_name=self.model_name, force_mock=self.force_mock)

        # Generate grid
        grid = [HyperparameterConfig(temperature=t, top_p=p) for t in self.temperatures for p in self.top_ps]
        total_points = len(grid)
        results: list[TuneCandidateResult] = []

        for idx, config in enumerate(grid, start=1):
            if progress_cb:
                progress_cb(idx, total_points, f"Testing config: temp={config.temperature}, top_p={config.top_p}")
            res = await self._evaluate_config(config, judge_ev)
            results.append(res)

        pareto = compute_pareto_frontier(results)
        best = results[0].config if results else grid[0]

        return TuningReport(
            prompt_name=self.prompt_version.name,
            model_name=self.model_name,
            total_configs_tested=total_points,
            best_config=best,
            pareto_candidates=pareto,
            all_results=results,
        )
