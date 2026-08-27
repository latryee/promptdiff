"""Async Execution Engine for Prompt Regression Testing."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
from typing import Callable, List, Optional
from promptdiff.core.cache import DiskCache
from promptdiff.core.models import (
    ComparisonResult,
    DiffReport,
    PromptVersion,
    RegressionVerdict,
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


class PromptDiffRunner:
    """Orchestrates async LLM runs, caching, diff calculations, and evaluations."""

    def __init__(
        self,
        v1_prompt: PromptVersion,
        v2_prompt: PromptVersion,
        provider_v1: BaseLLMProvider,
        provider_v2: BaseLLMProvider,
        evaluators: Optional[List[BaseEvaluator]] = None,
        assertions: Optional[List[str]] = None,
        cache: Optional[DiskCache] = None,
        concurrency: int = 4,
    ):
        self.v1_prompt = v1_prompt
        self.v2_prompt = v2_prompt
        self.provider_v1 = provider_v1
        self.provider_v2 = provider_v2
        self.evaluators = evaluators or get_evaluators(["json_validity", "latency", "cost", "similarity"])
        self.assertion_rules = parse_assertion_list(assertions or [])
        self.cache = cache or DiskCache(enabled=True)
        self.semaphore = asyncio.Semaphore(concurrency)

    async def _execute_single(
        self,
        prompt_version: PromptVersion,
        provider: BaseLLMProvider,
        test_case: TestCase,
    ) -> RunResult:
        """Execute a single prompt version on a testcase with caching."""
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

        # 2. Execute via Provider
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
        progress_cb: Optional[Callable[[str], None]] = None,
    ) -> ComparisonResult:
        """Run v1 and v2 on a test case, diff outputs, and compute evaluator metrics."""
        if progress_cb:
            progress_cb(f"Running test case '{test_case.id}'...")

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

        # Evaluator scores
        scores = {}
        for evaluator in self.evaluators:
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
        test_cases: List[TestCase],
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> DiffReport:
        """Run batch regression comparison across all test cases."""
        comparisons: List[ComparisonResult] = []

        total = len(test_cases)
        for i, tc in enumerate(test_cases):
            comp = await self.compare_case(tc)
            comparisons.append(comp)
            if progress_cb:
                progress_cb(i + 1, total)

        # Run CI/CD assertion checks
        verdict = evaluate_assertions(comparisons, self.assertion_rules)

        # Build aggregate statistics
        evaluator_names = [e.name for e in self.evaluators]
        report = DiffReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            v1_name=self.v1_prompt.name,
            v2_name=self.v2_prompt.name,
            model_v1=self.v1_prompt.model,
            model_v2=self.v2_prompt.model,
            comparisons=comparisons,
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
