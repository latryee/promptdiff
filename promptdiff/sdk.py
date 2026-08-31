"""High-level Python SDK and programmatic API for promptdiff."""

from __future__ import annotations

import asyncio
from typing import Optional, Union

from promptdiff.cli.history import GitHistoryReport, track_git_history
from promptdiff.core.config import load_dataset, load_prompt_file
from promptdiff.core.models import DiffReport, EvaluatorScore, TestCase
from promptdiff.core.runner import PromptDiffRunner
from promptdiff.evaluators.council import CouncilOfJudgesEvaluator
from promptdiff.evaluators.registry import get_evaluators
from promptdiff.generators.distiller import FineTuningDistiller
from promptdiff.generators.mutator import DatasetMutator
from promptdiff.generators.personas import PersonaStressTester
from promptdiff.generators.property_tester import PropertyBasedTester, PropertyTestReport
from promptdiff.lsp.server import PromptDiagnostic, PromptLanguageServer
from promptdiff.optimizer.auto_prompt import OptimizationResult, PromptOptimizer
from promptdiff.optimizer.cache_sim import CacheSimReport, PromptCacheSimulator
from promptdiff.optimizer.compiler import CompilationResult, PromptJITCompiler
from promptdiff.optimizer.compressor import CompressionResult, PromptCompressor
from promptdiff.optimizer.mutation_tester import MutationScoreReport, MutationTestingEngine
from promptdiff.optimizer.reflex import ReflectionLoopReport, SelfCorrectionBenchmark
from promptdiff.optimizer.saliency import PromptSaliencyMapper, SaliencyReport
from promptdiff.optimizer.tuner import PromptTuner, TuningReport
from promptdiff.production.canary import CanaryConfigGenerator, CanaryRolloutConfig
from promptdiff.production.cascade import CascadeRouteReport, ModelCascadeRouter
from promptdiff.production.edge_quant import EdgeQuantizationBenchmark, EdgeQuantReport
from promptdiff.production.profiler import StreamingProfiler, StreamingProfileResult
from promptdiff.production.replay import ReplayReport, ShadowTrafficReplayer
from promptdiff.production.sla import SLABudgetReport, SLABudgetSimulator
from promptdiff.providers.registry import get_provider
from promptdiff.reporters.bundle_html import generate_interactive_bundle_html
from promptdiff.reporters.notebook import JupyterNotebookExporter
from promptdiff.security.compliance import ComplianceAuditor, ComplianceReport
from promptdiff.security.fuzzer import FuzzReport, JailbreakFuzzer
from promptdiff.security.watermark import PromptWatermarker, WatermarkInspectionResult


def _resolve_testcases(dataset: Optional[Union[str, list[TestCase], list[dict]]] = None) -> list[TestCase]:
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
    mock: bool = False,
) -> FuzzReport:
    """Run autonomous adversarial red-teaming and jailbreak fuzzing."""
    p = load_prompt_file(prompt, version_name="fuzz_target", model=model)
    fuzzer = JailbreakFuzzer(prompt_version=p, model_name=model, force_mock=mock)
    return asyncio.run(fuzzer.run_fuzz())


def cache_sim(
    prompt: str,
    model: str = "claude-3-5-sonnet",
    daily_volume: int = 1_000_000,
) -> CacheSimReport:
    """Analyze and optimize prompt template for prefix caching."""
    p = load_prompt_file(prompt, version_name="cache_target", model=model)
    sim = PromptCacheSimulator(prompt_version=p, model_name=model, daily_volume=daily_volume)
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


def shadow_replay(
    candidate_prompt: str,
    log_path: str,
    model: str = "gpt-4o",
    mock: bool = True,
) -> ReplayReport:
    """Replay production logs against candidate prompt with automated PII redaction."""
    pv = load_prompt_file(candidate_prompt, version_name="candidate", model=model)
    replayer = ShadowTrafficReplayer(candidate_prompt=pv, model_name=model, force_mock=mock)
    return asyncio.run(replayer.replay(log_path))


def cascade(
    prompt: str,
    dataset: Union[str, list[TestCase]],
    tier1_model: str = "gpt-4o-mini",
    tier2_model: str = "gpt-4o",
    mock: bool = True,
) -> CascadeRouteReport:
    """Evaluate multi-tier model cascading router policies."""
    test_cases = _resolve_testcases(dataset)
    router = ModelCascadeRouter(
        prompt_template=prompt,
        test_cases=test_cases,
        tier1_model=tier1_model,
        tier2_model=tier2_model,
        force_mock=mock,
    )
    return asyncio.run(router.optimize())


def canary(report: DiffReport, flag_name: str = "prompt_rollout") -> CanaryRolloutConfig:
    """Generate production A/B/n canary feature flag configs."""
    generator = CanaryConfigGenerator(report=report, flag_name=flag_name)
    return generator.generate()


def sla_stress(
    prompt: str,
    dataset: Union[str, list[TestCase]],
    max_p99_latency_ms: float = 1500.0,
    model: str = "gpt-4o",
    mock: bool = True,
) -> SLABudgetReport:
    """Simulate load and verify SLA p99 latency ceilings."""
    pv = load_prompt_file(prompt, version_name="sla_test", model=model)
    test_cases = _resolve_testcases(dataset)
    sim = SLABudgetSimulator(
        prompt_version=pv,
        test_cases=test_cases,
        max_p99_latency_ms=max_p99_latency_ms,
        model_name=model,
        force_mock=mock,
    )
    return asyncio.run(sim.run_stress_test())


def personas(dataset: Union[str, list[TestCase]], output: Optional[str] = None) -> list[TestCase]:
    """Generate test cases across diverse human personas."""
    seed_cases = _resolve_testcases(dataset)
    tester = PersonaStressTester(seed_testcases=seed_cases)
    cases = tester.generate_persona_testcases()
    if output:
        tester.save_to_jsonl(cases, output)
    return cases


def saliency(prompt: str, sample_outputs: list[str]) -> SaliencyReport:
    """Map output influence and identify dead-weight tokens in prompt."""
    pv = load_prompt_file(prompt, version_name="saliency_target")
    mapper = PromptSaliencyMapper(prompt_version=pv)
    return mapper.analyze(sample_outputs)


def distill(report: DiffReport, output: str = "distilled_train.jsonl") -> tuple[str, int]:
    """Extract fine-tuning dataset pairs from evaluation runs."""
    distiller = FineTuningDistiller(report=report)
    return distiller.export_jsonl(output)


def mutation_score(
    prompt: str,
    dataset: Union[str, list[TestCase]],
    model: str = "gpt-4o",
    mock: bool = True,
) -> MutationScoreReport:
    """Evaluate test suite quality by injecting faults into prompt."""
    pv = load_prompt_file(prompt, version_name="original", model=model)
    test_cases = _resolve_testcases(dataset)
    engine = MutationTestingEngine(original_prompt=pv, test_cases=test_cases, model_name=model, force_mock=mock)
    return asyncio.run(engine.run_mutation_analysis())


def lsp_diagnostics(file_path: str, model: str = "gpt-4o") -> list[PromptDiagnostic]:
    """Analyze prompt file for LSP diagnostics."""
    server = PromptLanguageServer(model_name=model)
    return server.analyze_file(file_path)


def export_bundle(report: DiffReport, output_path: str = "promptdiff-bundle.html") -> str:
    """Export single-file zero-dependency interactive HTML bundle."""
    return generate_interactive_bundle_html(report, output_path)


def council(
    v1: str,
    v2: str,
    test_case: TestCase,
    judges: Optional[list[str]] = None,
    mock: bool = True,
) -> EvaluatorScore:
    """Evaluate using Council of Judges ensemble."""
    evaluator = CouncilOfJudgesEvaluator(judge_models=judges, force_mock=mock)
    p1 = load_prompt_file(v1, version_name="v1")
    p2 = load_prompt_file(v2, version_name="v2")
    r1 = p1.render(test_case.vars)
    r2 = p2.render(test_case.vars)
    from promptdiff.core.models import RunResult
    res1 = RunResult(prompt_name="v1", test_case_id=test_case.id, rendered_prompt=r1, output=r1, latency_ms=10.0, prompt_tokens=10, completion_tokens=10, total_tokens=20, cost_usd=0.0001, model="gpt-4o")
    res2 = RunResult(prompt_name="v2", test_case_id=test_case.id, rendered_prompt=r2, output=r2, latency_ms=10.0, prompt_tokens=10, completion_tokens=10, total_tokens=20, cost_usd=0.0001, model="gpt-4o")
    return evaluator.evaluate(res1, res2, test_case)


def profile_stream(prompt: str, query: str, model: str = "gpt-4o", mock: bool = True) -> StreamingProfileResult:
    """Profile Time-To-First-Token (TTFT) and token streaming speed."""
    pv = load_prompt_file(prompt, version_name="profile_target", model=model)
    profiler = StreamingProfiler(prompt_version=pv, model_name=model, force_mock=mock)
    return asyncio.run(profiler.profile_stream(query))


def watermark(prompt: str, secret_key: str = "default-key") -> str:
    """Inject invisible zero-entropy cryptographic watermark into prompt template."""
    wm = PromptWatermarker(secret_key=secret_key)
    return wm.inject_watermark(prompt)


def inspect_watermark(text: str) -> WatermarkInspectionResult:
    """Scan candidate text for watermark signature."""
    wm = PromptWatermarker()
    return wm.inspect_text_for_watermark(text)


def edge_quant(prompt: str, testcases: Union[str, list[TestCase]], mock: bool = True) -> EdgeQuantReport:
    """Benchmark local edge quantization degradation."""
    pv = load_prompt_file(prompt, version_name="edge_target")
    cases = _resolve_testcases(testcases)
    bench = EdgeQuantizationBenchmark(prompt_version=pv, test_cases=cases, force_mock=mock)
    return asyncio.run(bench.benchmark_quant_levels())


def property_test(prompt: str, iterations: int = 10, mock: bool = True) -> PropertyTestReport:
    """Run property-based invariant fuzzing on prompt."""
    pv = load_prompt_file(prompt, version_name="property_target")
    tester = PropertyBasedTester(prompt_version=pv, num_iterations=iterations, force_mock=mock)
    return asyncio.run(tester.run_property_tests())


def compliance_audit(prompt: str) -> ComplianceReport:
    """Audit prompt against EU AI Act, HIPAA, GDPR, and SOC2."""
    pv = load_prompt_file(prompt, version_name="compliance_target")
    auditor = ComplianceAuditor(prompt_version=pv)
    return auditor.audit()


def reflex_benchmark(prompt: str, testcases: Union[str, list[TestCase]], mock: bool = True) -> ReflectionLoopReport:
    """Benchmark self-correction reflection loop vs direct generation."""
    pv = load_prompt_file(prompt, version_name="reflex_target")
    cases = _resolve_testcases(testcases)
    bench = SelfCorrectionBenchmark(prompt_version=pv, test_cases=cases, force_mock=mock)
    return asyncio.run(bench.benchmark_reflection())


def export_notebook(report: DiffReport, output_path: str = "report.ipynb") -> str:
    """Export experiment report to interactive Jupyter Notebook."""
    exporter = JupyterNotebookExporter(report=report)
    return exporter.save_notebook(output_path)


def compile_prompt(prompt: str) -> CompilationResult:
    """JIT compile and AST minify prompt template."""
    pv = load_prompt_file(prompt, version_name="compile_target")
    compiler = PromptJITCompiler(prompt_version=pv)
    return compiler.compile()
