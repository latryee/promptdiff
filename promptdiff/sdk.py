"""High-level Python SDK and programmatic API for promptdiff."""

from __future__ import annotations

import asyncio
from typing import Optional, Union

from promptdiff.cli.history import GitHistoryReport, track_git_history
from promptdiff.core.config import load_dataset, load_prompt_file
from promptdiff.core.models import DiffReport, TestCase
from promptdiff.core.runner import PromptDiffRunner
from promptdiff.evaluators.registry import get_evaluators
from promptdiff.generators.mutator import DatasetMutator
from promptdiff.optimizer.auto_prompt import OptimizationResult, PromptOptimizer
from promptdiff.optimizer.cache_sim import CacheSimReport, PromptCacheSimulator
from promptdiff.optimizer.compressor import CompressionResult, PromptCompressor
from promptdiff.optimizer.tuner import PromptTuner, TuningReport
from promptdiff.providers.registry import get_provider
from promptdiff.reporters.bundle_html import generate_interactive_bundle_html
from promptdiff.security.fuzzer import FuzzReport, JailbreakFuzzer


def _resolve_testcases(dataset: Optional[Union[str, list[TestCase], list[dict]]]) -> list[TestCase]:
    if dataset is None or isinstance(dataset, str):
        return load_dataset(dataset)
    elif isinstance(dataset, list):
        cases = []
        for item in dataset:
            if isinstance(item, TestCase):
                cases.append(item)
            elif isinstance(item, dict):
                cases.append(TestCase(
                    id=str(item.get("id", f"tc_{len(cases)+1}")),
                    description=str(item.get("description", "")),
                    vars=item.get("vars", {}),
                    expected_output=item.get("expected_output"),
                    tags=item.get("tags", []),
                ))
        return cases
    return []


async def async_compare(
    v1: str,
    v2: str,
    dataset: Optional[Union[str, list[TestCase], list[dict]]] = None,
    model: str = "gpt-4o",
    model_v1: Optional[str] = None,
    model_v2: Optional[str] = None,
    eval_metrics: str = "json_validity,latency,cost,similarity,llm_judge,faithfulness,security",
    assertions: Optional[list[str]] = None,
    mock: bool = False,
    concurrency: int = 4,
) -> DiffReport:
    """Asynchronously compare two prompt versions across test cases."""
    m1 = model_v1 or model
    m2 = model_v2 or model

    p1 = load_prompt_file(v1, version_name="v1", model=m1)
    p2 = load_prompt_file(v2, version_name="v2", model=m2)
    test_cases = _resolve_testcases(dataset)

    prov1 = get_provider(model_name=m1, force_mock=mock)
    prov2 = get_provider(model_name=m2, force_mock=mock)
    eval_list = get_evaluators([eval_metrics])

    runner = PromptDiffRunner(
        v1_prompt=p1,
        v2_prompt=p2,
        provider_v1=prov1,
        provider_v2=prov2,
        evaluators=eval_list,
        assertions=assertions,
        concurrency=concurrency,
    )

    return await runner.run(test_cases)


def compare(
    v1: str,
    v2: str,
    dataset: Optional[Union[str, list[TestCase], list[dict]]] = None,
    model: str = "gpt-4o",
    model_v1: Optional[str] = None,
    model_v2: Optional[str] = None,
    eval_metrics: str = "json_validity,latency,cost,similarity,llm_judge,faithfulness,security",
    assertions: Optional[list[str]] = None,
    mock: bool = False,
    concurrency: int = 4,
) -> DiffReport:
    """Synchronously compare two prompt versions across test cases."""
    return asyncio.run(async_compare(
        v1=v1,
        v2=v2,
        dataset=dataset,
        model=model,
        model_v1=model_v1,
        model_v2=model_v2,
        eval_metrics=eval_metrics,
        assertions=assertions,
        mock=mock,
        concurrency=concurrency,
    ))


def optimize(
    prompt: str,
    dataset: Optional[Union[str, list[TestCase], list[dict]]] = None,
    model: str = "gpt-4o",
    meta_model: str = "gpt-4o",
    iterations: int = 3,
    mock: bool = False,
) -> OptimizationResult:
    """Optimize prompt template automatically using failed test cases and Meta-LLM reflection."""
    p = load_prompt_file(prompt, version_name="initial", model=model)
    test_cases = _resolve_testcases(dataset)

    optimizer = PromptOptimizer(
        prompt_version=p,
        test_cases=test_cases,
        model_name=model,
        meta_model_name=meta_model,
        max_iterations=iterations,
        force_mock=mock,
    )
    return asyncio.run(optimizer.optimize())


def tune(
    prompt: str,
    dataset: Optional[Union[str, list[TestCase], list[dict]]] = None,
    model: str = "gpt-4o",
    temperatures: Optional[list[float]] = None,
    top_ps: Optional[list[float]] = None,
    mock: bool = False,
) -> TuningReport:
    """Run hyperparameter grid search and return Pareto-optimal configuration."""
    p = load_prompt_file(prompt, version_name="tune_target", model=model)
    test_cases = _resolve_testcases(dataset)

    tuner = PromptTuner(
        prompt_version=p,
        test_cases=test_cases,
        model_name=model,
        temperatures=temperatures or [0.0, 0.3, 0.7, 1.0],
        top_ps=top_ps or [0.7, 0.9, 1.0],
        force_mock=mock,
    )
    return asyncio.run(tuner.tune())


def shrink(
    prompt: str,
    dataset: Optional[Union[str, list[TestCase], list[dict]]] = None,
    model: str = "gpt-4o",
    target_reduction: float = 0.30,
    mock: bool = False,
) -> CompressionResult:
    """Compress and prune prompt tokens while ensuring zero quality loss."""
    p = load_prompt_file(prompt, version_name="shrink_target", model=model)
    test_cases = _resolve_testcases(dataset)

    compressor = PromptCompressor(
        prompt_version=p,
        test_cases=test_cases,
        model_name=model,
        target_reduction=target_reduction,
        force_mock=mock,
    )
    return asyncio.run(compressor.compress())


def fuzz(
    prompt: str,
    model: str = "gpt-4o",
    attacks_count: int = 15,
    mock: bool = False,
) -> FuzzReport:
    """Run autonomous adversarial red-teaming and jailbreak fuzzing."""
    p = load_prompt_file(prompt, version_name="fuzz_target", model=model)
    fuzzer = JailbreakFuzzer(
        prompt_version=p,
        model_name=model,
        force_mock=mock,
    )
    return asyncio.run(fuzzer.run_fuzz())


def cache_sim(
    prompt: str,
    model: str = "claude-3-5-sonnet",
    daily_volume: int = 1_000_000,
) -> CacheSimReport:
    """Analyze and optimize prompt template for prefix caching."""
    p = load_prompt_file(prompt, version_name="cache_target", model=model)
    sim = PromptCacheSimulator(
        prompt_version=p,
        model_name=model,
        daily_volume=daily_volume,
    )
    return sim.analyze_and_optimize()


def mutate(
    dataset: Union[str, list[TestCase]],
    output: Optional[str] = None,
    multiplier: int = 5,
) -> list[TestCase]:
    """Mutate and expand seed test cases into diverse high-entropy stress test cases."""
    seed_cases = _resolve_testcases(dataset)
    mutator = DatasetMutator(seed_testcases=seed_cases, multiplier=multiplier)
    mutated = mutator.generate_mutations()
    if output:
        mutator.save_to_jsonl(mutated, output)
    return mutated


def history(
    prompt_file: str,
    dataset: Optional[str] = None,
    commits: int = 4,
    model: str = "gpt-4o",
    mock: bool = True,
) -> GitHistoryReport:
    """Benchmark prompt evolution across Git commit history."""
    return asyncio.run(track_git_history(
        prompt_file=prompt_file,
        dataset_path=dataset,
        commits_count=commits,
        model_name=model,
        force_mock=mock,
    ))


def export_bundle(report: DiffReport, output_path: str = "promptdiff-bundle.html") -> str:
    """Export single-file zero-dependency interactive HTML bundle."""
    return generate_interactive_bundle_html(report, output_path)
