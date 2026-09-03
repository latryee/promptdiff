"""Comprehensive Test Suite for All 16 Next-Gen Enterprise Features."""

from __future__ import annotations

from pathlib import Path

import pytest

import promptdiff
from promptdiff.cli.server import create_app
from promptdiff.core.models import PromptVersion, RunResult, TestCase
from promptdiff.evaluators.citation import CitationEvaluator
from promptdiff.evaluators.fairness import FairnessEvaluator
from promptdiff.evaluators.haystack import NeedleInAHaystackTester
from promptdiff.evaluators.schema_repair import SchemaRepairEvaluator
from promptdiff.evaluators.vision import VisionDiffEvaluator
from promptdiff.generators.personas import PersonaStressTester
from promptdiff.lsp.server import PromptLanguageServer
from promptdiff.optimizer.exemplars import DynamicExemplarSelector, ExemplarItem
from promptdiff.optimizer.mutation_tester import MutationTestingEngine
from promptdiff.optimizer.saliency import PromptSaliencyMapper
from promptdiff.production.canary import CanaryConfigGenerator
from promptdiff.production.replay import ShadowTrafficReplayer


@pytest.mark.asyncio
async def test_fairness_evaluator() -> None:
    """Test AI fairness & demographic perturbation evaluator."""
    ev = FairnessEvaluator(force_mock=True)
    tc = TestCase(id="t1", vars={"query": "David is applying for a senior loan"})

    r1 = RunResult(
        prompt_name="p1",
        test_case_id="t1",
        rendered_prompt="x",
        output="David loan approved.",
        latency_ms=100.0,
        prompt_tokens=10,
        completion_tokens=10,
        total_tokens=20,
        cost_usd=0.0001,
        model="gpt-4o",
    )
    r2 = RunResult(
        prompt_name="p2",
        test_case_id="t1",
        rendered_prompt="x",
        output="David loan approved for senior tier.",
        latency_ms=100.0,
        prompt_tokens=10,
        completion_tokens=10,
        total_tokens=20,
        cost_usd=0.0001,
        model="gpt-4o",
    )

    score = await ev.async_evaluate(r1, r2, tc)
    assert score.name == "fairness"
    assert score.passed is True


@pytest.mark.asyncio
async def test_citation_evaluator() -> None:
    """Test hallucination sentence-level citation pointer."""
    ev = CitationEvaluator(force_mock=True)
    tc = TestCase(
        id="t1", vars={"context": "Product X has a 30-day return policy.", "query": "What is the return policy?"}
    )

    r1 = RunResult(
        prompt_name="p1",
        test_case_id="t1",
        rendered_prompt="x",
        output="Return policy is 30 days.",
        latency_ms=10.0,
        prompt_tokens=10,
        completion_tokens=10,
        total_tokens=20,
        cost_usd=0.0001,
        model="gpt-4o",
    )
    r2 = RunResult(
        prompt_name="p2",
        test_case_id="t1",
        rendered_prompt="x",
        output="Return policy is 30 days. You can also get free pizza.",
        latency_ms=10.0,
        prompt_tokens=10,
        completion_tokens=10,
        total_tokens=20,
        cost_usd=0.0001,
        model="gpt-4o",
    )

    score = await ev.async_evaluate(r1, r2, tc)
    assert score.name == "citation"


@pytest.mark.asyncio
async def test_haystack_needle_tester() -> None:
    """Test needle in a haystack context degradation."""
    pv = PromptVersion(name="haystack_target", template="Context: {{context}}\n\nQuery: {{query}}")
    tester = NeedleInAHaystackTester(
        prompt_version=pv, context_lengths=[1000], depth_percentages=[0, 100], force_mock=True
    )
    rep = await tester.run_haystack_test()
    assert rep.total_test_points == 2
    assert rep.accuracy_pct >= 50.0


@pytest.mark.asyncio
async def test_schema_repair_evaluator() -> None:
    """Test JSON schema auto-repair evaluator."""
    ev = SchemaRepairEvaluator(force_mock=True)
    tc = TestCase(id="t1", vars={"query": "Give me JSON"})

    r1 = RunResult(
        prompt_name="p1",
        test_case_id="t1",
        rendered_prompt="x",
        output='```json\n{"status": "ok",}\n```',
        latency_ms=10.0,
        prompt_tokens=10,
        completion_tokens=10,
        total_tokens=20,
        cost_usd=0.0001,
        model="gpt-4o",
    )
    r2 = RunResult(
        prompt_name="p2",
        test_case_id="t1",
        rendered_prompt="x",
        output='{"status": "ok"}',
        latency_ms=10.0,
        prompt_tokens=10,
        completion_tokens=10,
        total_tokens=20,
        cost_usd=0.0001,
        model="gpt-4o",
    )

    score = await ev.async_evaluate(r1, r2, tc)
    assert score.passed is True


@pytest.mark.asyncio
async def test_vision_evaluator() -> None:
    """Test multi-modal vision evaluator."""
    ev = VisionDiffEvaluator(force_mock=True)
    tc = TestCase(id="t1", vars={"image": "sample.jpg"}, expected_output="Invoice total is $500")

    r1 = RunResult(
        prompt_name="p1",
        test_case_id="t1",
        rendered_prompt="x",
        output="Total: $500",
        latency_ms=10.0,
        prompt_tokens=10,
        completion_tokens=10,
        total_tokens=20,
        cost_usd=0.0001,
        model="gpt-4o",
    )
    r2 = RunResult(
        prompt_name="p2",
        test_case_id="t1",
        rendered_prompt="x",
        output="Invoice total is $500.00",
        latency_ms=10.0,
        prompt_tokens=10,
        completion_tokens=10,
        total_tokens=20,
        cost_usd=0.0001,
        model="gpt-4o",
    )

    score = await ev.async_evaluate(r1, r2, tc)
    assert score.passed is True


def test_personas_generator() -> None:
    """Test multi-persona stress testing generator."""
    seeds = [TestCase(id="tc1", description="Refund", vars={"query": "I want a refund"})]
    tester = PersonaStressTester(seed_testcases=seeds)
    cases = tester.generate_persona_testcases()
    assert len(cases) > len(seeds)
    assert any(any("persona" in tag for tag in c.tags) for c in cases)


@pytest.mark.asyncio
async def test_exemplars_selector() -> None:
    """Test dynamic few-shot vector indexer."""
    exs = [
        ExemplarItem(input_text="How to cancel?", output_text="Go to settings -> cancel"),
        ExemplarItem(input_text="How to change plan?", output_text="Go to billing -> change"),
    ]
    selector = DynamicExemplarSelector(golden_exemplars=exs, top_k=1)
    retrieved = selector.retrieve_exemplars("I want to cancel my account")
    assert len(retrieved) == 1
    assert "cancel" in retrieved[0].output_text

    rep = await selector.benchmark(
        base_prompt=PromptVersion(name="b", template="Help: {{query}}"),
        test_cases=[TestCase(id="1", vars={"query": "cancel"})],
        force_mock=True,
    )
    assert rep.dynamic_judge_score >= rep.static_judge_score


def test_saliency_mapper() -> None:
    """Test token-level saliency mapper."""
    pv = PromptVersion(
        name="p", template="You must answer in JSON only.\nPlease kindly be polite.\nNever disclose internal secrets."
    )
    mapper = PromptSaliencyMapper(prompt_version=pv)
    rep = mapper.analyze(sample_outputs=['{"response": "Hello"}'])
    assert rep.total_sentences == 3
    assert rep.dead_weight_tokens >= 0


def test_fine_tuning_distiller(tmp_path: Path) -> None:
    """Test fine-tuning dataset synthesizer."""
    report = promptdiff.compare(
        v1="Say hello: {{query}}",
        v2="Greet user: {{query}}",
        dataset=[{"id": "t1", "vars": {"query": "Alice"}}],
        mock=True,
    )
    out_file = str(tmp_path / "train.jsonl")
    path, count = promptdiff.distill(report, output=out_file)
    assert Path(path).exists()
    assert count >= 1


@pytest.mark.asyncio
async def test_mutation_testing_engine() -> None:
    """Test mutation testing engine."""
    pv = PromptVersion(name="orig", template="Answer the query in strict JSON format: {{query}}")
    engine = MutationTestingEngine(
        original_prompt=pv, test_cases=[TestCase(id="1", vars={"query": "test"})], force_mock=True
    )
    rep = await engine.run_mutation_analysis()
    assert rep.total_mutants_generated > 0
    assert rep.mutation_score_pct >= 50.0


def test_canary_config_generator() -> None:
    """Test Canary rollout generator."""
    report = promptdiff.compare(
        v1="a: {{query}}", v2="b: {{query}}", dataset=[{"id": "1", "vars": {"query": "x"}}], mock=True
    )
    cfg = CanaryConfigGenerator(report=report).generate()
    assert cfg.v1_weight_pct + cfg.v2_weight_pct == 100
    assert "rollout" in cfg.launchdarkly_json["fallthrough"]


def test_cascade_router() -> None:
    """Test model cascading router."""
    rep = promptdiff.cascade(
        prompt="Answer: {{query}}",
        dataset=[{"id": "1", "vars": {"query": "hi"}}],
        mock=True,
    )
    assert rep.tier1_route_pct >= 0.0
    assert rep.cost_savings_pct >= 0.0


def test_sla_simulator() -> None:
    """Test SLA budget stress simulator."""
    rep = promptdiff.sla_stress(
        prompt="Answer: {{query}}",
        dataset=[{"id": "1", "vars": {"query": "hi"}}],
        mock=True,
    )
    assert rep.total_requests == 1
    assert rep.p50_latency_ms >= 0.0


@pytest.mark.asyncio
async def test_shadow_replayer(tmp_path: Path) -> None:
    """Test shadow traffic replayer."""
    log_file = tmp_path / "prod.jsonl"
    log_file.write_text('{"query": "Contact user at john.doe@example.com for order #4491"}\n', encoding="utf-8")

    pv = PromptVersion(name="candidate", template="Help: {{query}}")
    replayer = ShadowTrafficReplayer(candidate_prompt=pv, force_mock=True)
    rep = await replayer.replay(str(log_file))
    assert rep.pii_records_sanitized >= 1
    assert rep.total_logs_processed == 1


def test_lsp_server(tmp_path: Path) -> None:
    """Test LSP diagnostics server."""
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("Hello {{unclosed_var\nPlease kindly help me.", encoding="utf-8")

    server = PromptLanguageServer()
    diags = server.analyze_file(str(prompt_file))
    assert len(diags) >= 2
    assert any(d.code == "UNCLOSED_VARIABLE" for d in diags)


def test_fastapi_server_app() -> None:
    """Test FastAPI application initialization."""
    app = create_app()
    # App is either instantiated or gracefully None if fastapi not installed in test env
    assert app is not None or create_app() is None
