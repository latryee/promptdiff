"""Coverage tests for SLA Budget Simulator, Edge Quantization, and Persona Stress Testing."""

from __future__ import annotations

from pathlib import Path

import pytest

from promptdiff.core.models import PromptVersion, TestCase
from promptdiff.generators.personas import PersonaStressTester
from promptdiff.production.edge_quant import EdgeQuantizationBenchmark, EdgeQuantReport
from promptdiff.production.sla import SLABudgetReport, SLABudgetSimulator


def test_persona_stress_tester_generation(tmp_path: Path) -> None:
    seeds = [
        TestCase(id="q1", description="Account balance", vars={"query": "What is my account balance?"}),
    ]
    tester = PersonaStressTester(seed_testcases=seeds)
    expanded = tester.generate_persona_testcases()

    # Original + 5 personas = 6 test cases
    assert len(expanded) == 6
    assert expanded[0].id == "q1"
    assert any("angry" in tc.id for tc in expanded)

    out_file = tmp_path / "personas.jsonl"
    saved_path = tester.save_to_jsonl(expanded, str(out_file))
    assert Path(saved_path).exists()


@pytest.mark.asyncio
async def test_edge_quantization_benchmark() -> None:
    pv = PromptVersion(name="v1", template="Summarize: {{query}}")
    cases = [TestCase(id="c1", vars={"query": "hello world"})]
    bench = EdgeQuantizationBenchmark(
        prompt_version=pv,
        test_cases=cases,
        force_mock=True,
    )
    report = await bench.benchmark_quant_levels()

    assert isinstance(report, EdgeQuantReport)
    assert len(report.levels) == 5
    assert report.levels[0].quant_level.startswith("FP16")
    assert "Q4_K_M" in report.optimal_edge_quant
    assert any(lvl.status == "RECOMMENDED_FOR_EDGE" for lvl in report.levels)


@pytest.mark.asyncio
async def test_sla_budget_simulator_passed() -> None:
    pv = PromptVersion(name="v1", template="Answer: {{query}}")
    cases = [TestCase(id=f"tc_{i}", vars={"query": f"question {i}"}) for i in range(5)]
    sim = SLABudgetSimulator(
        prompt_version=pv,
        test_cases=cases,
        max_p99_latency_ms=5000.0,
        max_cost_per_request_usd=1.0,
        force_mock=True,
    )
    report = await sim.run_stress_test()

    assert isinstance(report, SLABudgetReport)
    assert report.total_requests == 5
    assert report.sla_passed is True
    assert len(report.breaches) == 0


@pytest.mark.asyncio
async def test_sla_budget_simulator_breach() -> None:
    pv = PromptVersion(name="v1", template="Answer: {{query}}")
    cases = [TestCase(id="c1", vars={"query": "test"})]
    # Set impossible tight budget: 0.1ms latency and $0.00000001 cost
    sim = SLABudgetSimulator(
        prompt_version=pv,
        test_cases=cases,
        max_p99_latency_ms=0.01,
        max_cost_per_request_usd=0.00000001,
        force_mock=True,
    )
    report = await sim.run_stress_test()

    assert report.sla_passed is False
    assert len(report.breaches) > 0
    assert any("LATENCY" in b.breach_type for b in report.breaches)
