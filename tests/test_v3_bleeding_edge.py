"""Comprehensive Tests for the 10 Bleeding-Edge Enterprise Features."""

from __future__ import annotations

from pathlib import Path

import pytest

import promptdiff
from promptdiff.core.models import PromptVersion, RunResult, TestCase
from promptdiff.evaluators.council import CouncilOfJudgesEvaluator
from promptdiff.generators.property_tester import PropertyBasedTester
from promptdiff.optimizer.compiler import PromptJITCompiler
from promptdiff.optimizer.reflex import SelfCorrectionBenchmark
from promptdiff.production.edge_quant import EdgeQuantizationBenchmark
from promptdiff.production.profiler import StreamingProfiler
from promptdiff.production.watch_daemon import PromptHealthDaemon
from promptdiff.security.compliance import ComplianceAuditor
from promptdiff.security.watermark import PromptWatermarker


@pytest.mark.asyncio
async def test_council_of_judges_evaluator() -> None:
    """Test multi-model ensemble Council evaluator."""
    ev = CouncilOfJudgesEvaluator(force_mock=True)
    tc = TestCase(id="t1", vars={"query": "Test question"})
    r1 = RunResult(
        prompt_name="p1",
        test_case_id="t1",
        rendered_prompt="x",
        output="Hello",
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
        output="Hello world",
        latency_ms=10.0,
        prompt_tokens=10,
        completion_tokens=10,
        total_tokens=20,
        cost_usd=0.0001,
        model="gpt-4o",
    )

    score = await ev.async_evaluate(r1, r2, tc)
    assert score.name == "council"
    assert "Council Consensus" in score.message
    assert score.passed is True


@pytest.mark.asyncio
async def test_streaming_profiler() -> None:
    """Test streaming TTFT and inter-token latency profiler."""
    pv = PromptVersion(name="stream_p", template="Answer: {{query}}")
    profiler = StreamingProfiler(prompt_version=pv, force_mock=True)
    res = await profiler.profile_stream("Tell me a story")
    assert res.time_to_first_token_ms > 0
    assert res.tokens_per_second > 0
    assert len(res.full_output) > 0


def test_watch_health_daemon() -> None:
    """Test semantic drift and real-time health daemon."""
    pv = PromptVersion(name="daemon_p", template="Support: {{query}}")
    daemon = PromptHealthDaemon(
        prompt_version=pv, golden_reference_outputs=["Hello customer, how can I help you today?"], drift_threshold=0.50
    )

    # 1. Healthy call
    alert1 = daemon.evaluate_live_call("Hello customer, how can I assist you today?")
    assert alert1 is None

    # 2. Drifted call
    alert2 = daemon.evaluate_live_call("Quantum mechanics is the study of matter and radiation.")
    assert alert2 is not None
    assert alert2.alert_level in ("WARNING", "CRITICAL")

    status = daemon.get_health_status()
    assert status.total_calls_monitored == 2
    assert status.drift_alerts_triggered == 1


def test_prompt_watermark() -> None:
    """Test cryptographic prompt watermarking with explicit secret key and grounded confidence."""
    # Must raise ValueError if secret_key is not provided
    with pytest.raises(ValueError, match="secret_key must be explicitly provided"):
        PromptWatermarker(secret_key=None)

    wm = PromptWatermarker(secret_key="my-secure-eval-key", organization="Acme AI Corp")
    orig_prompt = "You are an enterprise AI assistant. Always format responses in Markdown."
    watermarked = wm.inject_watermark(orig_prompt)

    assert watermarked != orig_prompt  # Contains zero-width invisible characters

    inspection = wm.inspect_text_for_watermark(watermarked)
    assert inspection.is_watermarked is True
    assert inspection.matched_organization == "Acme AI Corp"
    assert inspection.confidence_pct == 100.0

    # Non-watermarked check
    unmarked_inspection = wm.inspect_text_for_watermark("Regular plain text without watermarks.")
    assert unmarked_inspection.is_watermarked is False
    assert unmarked_inspection.confidence_pct == 0.0


@pytest.mark.asyncio
async def test_edge_quantization_benchmark() -> None:
    """Test local model quantization degradation benchmark."""
    pv = PromptVersion(name="edge_p", template="Explain: {{query}}")
    bench = EdgeQuantizationBenchmark(
        prompt_version=pv, test_cases=[TestCase(id="1", vars={"query": "AI"})], force_mock=True
    )
    report = await bench.benchmark_quant_levels()
    assert len(report.levels) == 5
    assert any(lvl.quant_level.startswith("Q4_K_M") for lvl in report.levels)
    assert "Q4_K_M" in report.optimal_edge_quant


@pytest.mark.asyncio
async def test_property_based_tester() -> None:
    """Test property-based invariant fuzzing."""
    pv = PromptVersion(name="prop_p", template="Process user {{name}}: {{query}}")
    tester = PropertyBasedTester(prompt_version=pv, num_iterations=5, force_mock=True)
    rep = await tester.run_property_tests()
    assert rep.total_permutations_tested == 5
    assert rep.all_invariants_hold is True


def test_compliance_auditor() -> None:
    """Test prompt guideline linter and ensure honest legal disclaimer is present."""
    pv = PromptVersion(
        name="comp_p",
        template="You are an AI assistant. Never disclose confidential medical phi and personal data privacy.",
    )
    linter = ComplianceAuditor(prompt_version=pv)
    report = linter.lint()
    assert report.overall_compliance_score_pct >= 50.0
    assert len(report.results) == 4
    assert "DISCLAIMER" in report.disclaimer


@pytest.mark.asyncio
async def test_self_correction_benchmark() -> None:
    """Test self-correction reflection loop benchmark."""
    pv = PromptVersion(name="reflex_p", template="Answer concisely: {{query}}")
    bench = SelfCorrectionBenchmark(
        prompt_version=pv, test_cases=[TestCase(id="1", vars={"query": "Hi"})], force_mock=True
    )
    report = await bench.benchmark_reflection()
    assert report.reflection_judge_score >= report.direct_judge_score
    assert len(report.roi_verdict) > 0


def test_jupyter_notebook_exporter(tmp_path: Path) -> None:
    """Test Jupyter Notebook .ipynb export."""
    report = promptdiff.compare(
        v1="Say hello: {{query}}",
        v2="Greet user: {{query}}",
        dataset=[{"id": "t1", "vars": {"query": "Alice"}}],
        mock=True,
    )
    nb_path = str(tmp_path / "experiment.ipynb")
    saved_path = promptdiff.export_notebook(report, output_path=nb_path)
    assert Path(saved_path).exists()
    assert Path(saved_path).stat().st_size > 100


def test_prompt_jit_compiler() -> None:
    """Test prompt JIT compiler and AST minifier."""
    raw_template = (
        "{# Internal developer note #}\n\nYou are an AI assistant.\n\n\n\nPlease answer query: {{ user_query }}."
    )
    pv = PromptVersion(name="compiler_p", template=raw_template)
    compiler = PromptJITCompiler(prompt_version=pv)
    res = compiler.compile()
    assert "{#" not in res.compiled_template
    assert "{{user_query}}" in res.compiled_template
    assert res.tokens_saved >= 0
