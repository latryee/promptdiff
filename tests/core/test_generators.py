"""Tests migrated to feature-focused test suite."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

import promptdiff
from promptdiff.core.models import (
    DiffReport,
    EvaluatorScore,
    PromptVersion,
    RegressionVerdict,
    RunResult,
    TestCase,
)
from promptdiff.generators.dpo_synthesizer import DPOSynthesizer
from promptdiff.generators.hard_negatives import HardNegativeGenerator
from promptdiff.generators.mutator import DatasetMutator, inject_slang_and_abbreviations, inject_typos
from promptdiff.generators.personas import PersonaStressTester
from promptdiff.generators.property_tester import PropertyBasedTester
from promptdiff.generators.synthetic import SyntheticTestGenerator


@pytest.mark.asyncio
async def test_synthetic_test_generator() -> None:
    """Test synthetic test data generation."""
    generator = SyntheticTestGenerator(
        prompt_template="You are a support agent for {{service}}. User says: {{query}}",
        description="Customer support helpdesk routing",
        force_mock=True,
    )

    cases = await generator.generate(count=10)
    assert len(cases) == 10
    assert all(isinstance(c, TestCase) for c in cases)
    assert any("synthetic" in c.tags for c in cases)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = str(Path(tmpdir) / "testcases.jsonl")
        generator.save_to_jsonl(cases, out_file)
        assert Path(out_file).is_file()
        assert Path(out_file).stat().st_size > 0


@pytest.mark.asyncio
async def test_property_based_tester() -> None:
    """Test property-based invariant fuzzing."""
    pv = PromptVersion(name="prop_p", template="Process user {{name}}: {{query}}")
    tester = PropertyBasedTester(prompt_version=pv, num_iterations=5, force_mock=True)
    rep = await tester.run_property_tests()
    assert rep.total_permutations_tested == 5
    assert rep.all_invariants_hold is True


def test_personas_generator() -> None:
    """Test multi-persona stress testing generator."""
    seeds = [TestCase(id="tc1", description="Refund", vars={"query": "I want a refund"})]
    tester = PersonaStressTester(seed_testcases=seeds)
    cases = tester.generate_persona_testcases()
    assert len(cases) > len(seeds)
    assert any(any("persona" in tag for tag in c.tags) for c in cases)


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


def test_dataset_mutator() -> None:
    """Test synthetic test case mutation engine."""
    seed_cases = [
        TestCase(id="tc_1", description="Refund inquiry", vars={"query": "Please cancel my subscription and refund."}),
    ]

    typo_text = inject_typos("customer", typo_rate=0.5)
    assert len(typo_text) == 8

    slang_text = inject_slang_and_abbreviations("Please help me as soon as possible thank you")
    assert "pls" in slang_text
    assert "asap" in slang_text
    assert "thx" in slang_text

    mutator = DatasetMutator(seed_testcases=seed_cases, multiplier=4)
    mutated = mutator.generate_mutations()

    assert len(mutated) == 4
    assert any("mutated" in tc.tags for tc in mutated)


@pytest.mark.asyncio
async def test_synthetic_adversarial_edge_cases_and_mutator_integration() -> None:
    """Test synthetic adversarial edge-case generation (empty, oversized, multilingual) and mutator integration."""
    from promptdiff.generators.synthetic import SyntheticTestGenerator

    gen = SyntheticTestGenerator(
        prompt_template="Answer question: {{query}} for customer {{customer_id}}",
        force_mock=True,
        mode="adversarial",
    )
    edge_cases = gen.generate_adversarial_edge_cases()
    assert len(edge_cases) >= 8

    # Verify empty string case
    empty_case = next(c for c in edge_cases if c.id == "edge_case_empty")
    assert empty_case.vars["query"] == ""

    # Verify oversized case
    oversized_case = next(c for c in edge_cases if c.id == "edge_case_oversized")
    assert len(oversized_case.vars["query"]) > 2000

    # Verify multilingual cases
    es_case = next(c for c in edge_cases if "multilingual_es" in c.id)
    assert "¿" in es_case.vars["query"]

    # Verify generation with adversarial mode
    batch = await gen.generate(count=15)
    assert len(batch) == 15
    assert any("adversarial" in c.tags or "edge_case" in c.tags for c in batch)

    # Verify integration with mutator
    seed_cases = [TestCase(id="tc_seed", vars={"query": "Help me login"})]
    mutator = DatasetMutator(seed_testcases=seed_cases, multiplier=2)
    augmented = mutator.augment_with_synthetic_adversarial(
        prompt_template="Answer: {{query}}",
        force_mock=True,
    )
    assert len(augmented) > 2
    assert any(c.id == "edge_case_empty" for c in augmented)


def test_hard_negative_generator() -> None:
    """Test automated synthesis of boundary adversarial cases."""
    gen = HardNegativeGenerator(count_per_category=1)
    suite = gen.generate("You are an assistant. Answer: {{query}} using {{context}}")
    assert suite.total_generated >= 4
    assert len(suite.identified_vulnerabilities) >= 3


def test_dpo_synthesizer() -> None:
    """Test synthesis of DPO preference pairs from DiffReport."""
    comp_score_v1 = EvaluatorScore(
        name="sim", passed=True, v1_score=0.9, v2_score=0.4, delta=-0.5, delta_pct=-50.0, message="ok"
    )
    tc = TestCase(id="tc1", vars={"query": "Hello"})
    r1 = RunResult(
        prompt_name="v1",
        test_case_id="tc1",
        rendered_prompt="Hello",
        output="Better output",
        latency_ms=100.0,
        prompt_tokens=10,
        completion_tokens=10,
        total_tokens=20,
        cost_usd=0.001,
        model="gpt-4o",
    )
    r2 = RunResult(
        prompt_name="v2",
        test_case_id="tc1",
        rendered_prompt="Hello",
        output="Worse output",
        latency_ms=100.0,
        prompt_tokens=10,
        completion_tokens=10,
        total_tokens=20,
        cost_usd=0.001,
        model="gpt-4o",
    )
    comp = promptdiff.core.models.ComparisonResult(
        test_case=tc,
        v1_result=r1,
        v2_result=r2,
        scores={"sim": comp_score_v1},
    )
    verdict = RegressionVerdict(
        passed=True,
        status="PASSED",
        failed_assertions=[],
        total_cost_v1=0.001,
        total_cost_v2=0.001,
        cost_delta_pct=0.0,
        avg_latency_v1=100.0,
        avg_latency_v2=100.0,
        latency_delta_pct=0.0,
    )
    report = DiffReport(
        v1_name="v1",
        v2_name="v2",
        model_v1="gpt-4o",
        model_v2="gpt-4o",
        comparisons=[comp],
        verdict=verdict,
        evaluators=["sim"],
        total_cases=1,
    )

    synth = DPOSynthesizer()
    dpo_res = synth.synthesize(report)
    assert dpo_res.total_pairs >= 1
    assert "Better output" in dpo_res.to_jsonl()
