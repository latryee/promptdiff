"""Async Execution Engine for Prompt Regression Testing & Multi-Model Arena."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import datetime, timezone

from promptdiff.core.cache import DiskCache
from promptdiff.core.models import (
    ArenaModelSummary,
    ArenaReport,
    ComparisonResult,
    DiffChunk,
    DiffReport,
    EvaluatorScore,
    MultiComparisonResult,
    PromptVersion,
    RunResult,
    TestCase,
)
from promptdiff.diff.json_diff import compute_json_diff
from promptdiff.diff.text_diff import compute_word_diff
from promptdiff.evaluators.assertions import evaluate_assertions, parse_assertion_list
from promptdiff.evaluators.base import BaseEvaluator
from promptdiff.evaluators.json_validity import extract_json
from promptdiff.evaluators.registry import get_evaluators
from promptdiff.pricing import calculate_cost
from promptdiff.providers.base import BaseLLMProvider

logger = logging.getLogger("promptdiff.core.runner")


class PromptDiffRunner:
    """Orchestrates async LLM runs, bounded concurrency, caching, and evaluator scoring."""

    def __init__(
        self,
        v1_prompt: PromptVersion,
        v2_prompt: PromptVersion,
        provider_v1: BaseLLMProvider,
        provider_v2: BaseLLMProvider,
        evaluators: list[BaseEvaluator] | None = None,
        assertions: list[str] | None = None,
        cache: DiskCache | None = None,
        concurrency: int = 4,
    ):
        self.v1_prompt = v1_prompt
        self.v2_prompt = v2_prompt
        self.provider_v1 = provider_v1
        self.provider_v2 = provider_v2
        self.evaluators = evaluators or get_evaluators(["json_validity", "latency", "cost", "similarity"])
        self.assertion_rules = parse_assertion_list(assertions or [])
        self.cache = cache or DiskCache(enabled=True)
        self.concurrency = max(1, concurrency)
        self.semaphore = asyncio.Semaphore(self.concurrency)

    async def _execute_single(
        self,
        prompt_version: PromptVersion,
        provider: BaseLLMProvider,
        test_case: TestCase,
    ) -> RunResult:
        """Execute a single prompt version on a testcase with caching and bounded concurrency."""
        rendered = prompt_version.render(test_case.vars)
        cache_key = DiskCache.compute_key(
            prompt_text=rendered,
            system_prompt=prompt_version.system_prompt,
            model=prompt_version.model,
            temperature=prompt_version.temperature,
            max_tokens=prompt_version.max_tokens,
        )

        # 1. Check disk cache
        if self.cache.enabled:
            cached_result = self.cache.get(cache_key)
            if cached_result:
                cached_result.test_case_id = test_case.id
                cached_result.prompt_name = prompt_version.name
                return cached_result

        # 2. Execute via Provider under Semaphore limit
        async with self.semaphore:
            try:
                response = await provider.generate(
                    prompt=rendered,
                    system_prompt=prompt_version.system_prompt,
                    temperature=prompt_version.temperature,
                    max_tokens=prompt_version.max_tokens,
                )
                cost = calculate_cost(
                    model_name=prompt_version.model,
                    prompt_tokens=response.prompt_tokens,
                    completion_tokens=response.completion_tokens,
                )
                run_result = RunResult(
                    prompt_name=prompt_version.name,
                    test_case_id=test_case.id,
                    rendered_prompt=rendered,
                    output=response.output,
                    latency_ms=response.latency_ms,
                    prompt_tokens=response.prompt_tokens,
                    completion_tokens=response.completion_tokens,
                    total_tokens=response.total_tokens,
                    cost_usd=cost,
                    model=prompt_version.model,
                    cached=False,
                )
            except Exception as err:
                logger.warning(f"Error executing provider for case {test_case.id}: {err}")
                run_result = RunResult(
                    prompt_name=prompt_version.name,
                    test_case_id=test_case.id,
                    rendered_prompt=rendered,
                    output="",
                    latency_ms=0.0,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    cost_usd=0.0,
                    model=prompt_version.model,
                    cached=False,
                    error=str(err),
                )

        # 3. Store in cache if successful
        if self.cache.enabled and not run_result.error:
            self.cache.set(cache_key, run_result)

        return run_result

    async def compare_case(
        self,
        test_case: TestCase,
        progress_cb: Callable[[str], None] | None = None,
    ) -> ComparisonResult:
        """Run v1 and v2 on a test case concurrently, diff outputs, and compute evaluator metrics."""
        if progress_cb:
            progress_cb(test_case.id)

        v1_result, v2_result = await asyncio.gather(
            self._execute_single(self.v1_prompt, self.provider_v1, test_case),
            self._execute_single(self.v2_prompt, self.provider_v2, test_case),
        )

        # Word-level text diff
        text_diff = compute_word_diff(v1_result.output, v2_result.output)

        # JSON structural diff check
        v1_json, _ = extract_json(v1_result.output)
        v2_json, _ = extract_json(v2_result.output)
        is_json = v1_json is not None and v2_json is not None
        json_diff_obj = compute_json_diff(v1_json, v2_json) if is_json else None

        # Evaluator scores (support async evaluation if available)
        scores: dict[str, EvaluatorScore] = {}
        for evaluator in self.evaluators:
            try:
                score = await evaluator.async_evaluate(v1_result, v2_result, test_case)
            except Exception:
                score = evaluator.evaluate(v1_result, v2_result, test_case)
            scores[evaluator.name] = score

        return ComparisonResult(
            test_case=test_case,
            v1_result=v1_result,
            v2_result=v2_result,
            scores=scores,
            text_diff=text_diff,
            json_diff=json_diff_obj,
            is_json=is_json,
        )

    async def run(
        self,
        test_cases: list[TestCase],
        progress_cb: Callable[[int, int], None] | None = None,
    ) -> DiffReport:
        """Run batch regression comparison concurrently across all test cases with bounded concurrency."""
        total = len(test_cases)
        completed_count = 0
        lock = asyncio.Lock()

        async def _run_single_case(tc: TestCase) -> ComparisonResult:
            nonlocal completed_count
            comp = await self.compare_case(tc)
            async with lock:
                completed_count += 1
                if progress_cb:
                    progress_cb(completed_count, total)
            return comp

        # Execute all test cases concurrently using asyncio.gather bounded by semaphore
        comparisons = await asyncio.gather(*[_run_single_case(tc) for tc in test_cases])

        # Run CI/CD assertion checks
        verdict = evaluate_assertions(list(comparisons), self.assertion_rules)

        # Build aggregate statistics
        evaluator_names = [e.name for e in self.evaluators]
        report = DiffReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            v1_name=self.v1_prompt.name,
            v2_name=self.v2_prompt.name,
            model_v1=self.v1_prompt.model,
            model_v2=self.v2_prompt.model,
            comparisons=list(comparisons),
            verdict=verdict,
            evaluators=evaluator_names,
            total_cases=len(test_cases),
            aggregate_stats={
                "total_cost_v1": verdict.total_cost_v1,
                "total_cost_v2": verdict.total_cost_v2,
                "cost_delta_pct": verdict.cost_delta_pct,
                "avg_latency_v1": verdict.avg_latency_v1,
                "avg_latency_v2": verdict.avg_latency_v2,
                "latency_delta_pct": verdict.latency_delta_pct,
                "passed_cases": sum(1 for c in comparisons if all(s.passed for s in c.scores.values())),
            },
        )

        return report


class ArenaRunner:
    """Multi-Model / Multi-Prompt Arena Runner for evaluating N variants simultaneously."""

    def __init__(
        self,
        variants: dict[str, PromptVersion],
        providers: dict[str, BaseLLMProvider],
        baseline_name: str = "v1",
        evaluators: list[BaseEvaluator] | None = None,
        cache: DiskCache | None = None,
        concurrency: int = 6,
    ):
        self.variants = variants
        self.providers = providers
        self.baseline_name = baseline_name if baseline_name in variants else list(variants.keys())[0]
        self.evaluators = evaluators or get_evaluators(["json_validity", "latency", "cost", "similarity"])
        self.cache = cache or DiskCache(enabled=True)
        self.concurrency = max(1, concurrency)
        self.semaphore = asyncio.Semaphore(self.concurrency)

    async def _execute_variant(
        self,
        variant_name: str,
        test_case: TestCase,
    ) -> RunResult:
        """Execute single variant on test case."""
        pv = self.variants[variant_name]
        provider = self.providers[variant_name]
        rendered = pv.render(test_case.vars)
        cache_key = DiskCache.compute_key(
            prompt_text=rendered,
            system_prompt=pv.system_prompt,
            model=pv.model,
            temperature=pv.temperature,
            max_tokens=pv.max_tokens,
        )

        if self.cache.enabled:
            cached = self.cache.get(cache_key)
            if cached:
                cached.test_case_id = test_case.id
                cached.prompt_name = variant_name
                return cached

        async with self.semaphore:
            try:
                response = await provider.generate(
                    prompt=rendered,
                    system_prompt=pv.system_prompt,
                    temperature=pv.temperature,
                    max_tokens=pv.max_tokens,
                )
                cost = calculate_cost(
                    model_name=pv.model,
                    prompt_tokens=response.prompt_tokens,
                    completion_tokens=response.completion_tokens,
                )
                run_res = RunResult(
                    prompt_name=variant_name,
                    test_case_id=test_case.id,
                    rendered_prompt=rendered,
                    output=response.output,
                    latency_ms=response.latency_ms,
                    prompt_tokens=response.prompt_tokens,
                    completion_tokens=response.completion_tokens,
                    total_tokens=response.total_tokens,
                    cost_usd=cost,
                    model=pv.model,
                )
            except Exception as e:
                run_res = RunResult(
                    prompt_name=variant_name,
                    test_case_id=test_case.id,
                    rendered_prompt=rendered,
                    output="",
                    latency_ms=0.0,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    cost_usd=0.0,
                    model=pv.model,
                    error=str(e),
                )

        if self.cache.enabled and not run_res.error:
            self.cache.set(cache_key, run_res)

        return run_res

    async def evaluate_multi_case(self, test_case: TestCase) -> MultiComparisonResult:
        """Run all variants on a testcase and evaluate pairwise against baseline."""
        variant_names = list(self.variants.keys())
        tasks = [self._execute_variant(name, test_case) for name in variant_names]
        results_list = await asyncio.gather(*tasks)
        results: dict[str, RunResult] = dict(zip(variant_names, results_list, strict=False))

        baseline_res = results.get(self.baseline_name, results_list[0])
        scores: dict[str, dict[str, EvaluatorScore]] = {}
        pairwise_diffs: dict[str, list[DiffChunk]] = {}

        for name, cand_res in results.items():
            if name == self.baseline_name:
                continue
            scores[name] = {}
            for ev in self.evaluators:
                try:
                    score_obj = await ev.async_evaluate(baseline_res, cand_res, test_case)
                except Exception:
                    score_obj = ev.evaluate(baseline_res, cand_res, test_case)
                scores[name][ev.name] = score_obj
            pairwise_diffs[name] = compute_word_diff(baseline_res.output, cand_res.output)

        return MultiComparisonResult(
            test_case=test_case,
            results=results,
            scores=scores,
            pairwise_diffs=pairwise_diffs,
        )

    async def run(
        self,
        test_cases: list[TestCase],
        progress_cb: Callable[[int, int], None] | None = None,
    ) -> ArenaReport:
        """Run full multi-model arena evaluation across all testcases."""
        total = len(test_cases)
        completed = 0
        lock = asyncio.Lock()

        async def _eval_case(tc: TestCase) -> MultiComparisonResult:
            nonlocal completed
            res = await self.evaluate_multi_case(tc)
            async with lock:
                completed += 1
                if progress_cb:
                    progress_cb(completed, total)
            return res

        comparisons = await asyncio.gather(*[_eval_case(tc) for tc in test_cases])

        # Compute leaderboard aggregates per variant
        variant_names = list(self.variants.keys())
        summaries: list[ArenaModelSummary] = []

        for name in variant_names:
            pv = self.variants[name]
            runs = [comp.results[name] for comp in comparisons if name in comp.results]
            tot_cost = sum(r.cost_usd for r in runs)
            avg_lat = sum(r.latency_ms for r in runs) / max(1, len(runs))
            avg_tok = sum(r.total_tokens for r in runs) / max(1, len(runs))

            # Average eval scores when compared to baseline
            avg_eval: dict[str, float] = {}
            if name != self.baseline_name:
                for ev in self.evaluators:
                    scores_list = [comp.scores.get(name, {}).get(ev.name) for comp in comparisons]
                    valid_scores = [
                        float(s.v2_score) for s in scores_list if s is not None and isinstance(s.v2_score, (int, float))
                    ]
                    if valid_scores:
                        avg_eval[ev.name] = round(sum(valid_scores) / len(valid_scores), 3)

            summaries.append(
                ArenaModelSummary(
                    name=name,
                    model=pv.model,
                    total_cost=round(tot_cost, 6),
                    avg_latency_ms=round(avg_lat, 2),
                    avg_tokens=round(avg_tok, 1),
                    avg_eval_scores=avg_eval,
                )
            )

        # Sort leaderboard by lowest cost, then lowest latency
        summaries.sort(key=lambda s: (s.total_cost, s.avg_latency_ms))
        for idx, s in enumerate(summaries):
            s.rank = idx + 1

        return ArenaReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            variants=variant_names,
            models={name: pv.model for name, pv in self.variants.items()},
            total_cases=len(test_cases),
            leaderboard=summaries,
            comparisons=list(comparisons),
        )
