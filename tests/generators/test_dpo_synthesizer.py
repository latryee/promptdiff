"""Tests verifying DPO dataset synthesis and HuggingFace TRL schema compliance."""

from __future__ import annotations

import json

from promptdiff.core.models import (
    ComparisonResult,
    DiffReport,
    EvaluatorScore,
    RegressionVerdict,
    RunResult,
    TestCase,
)
from promptdiff.generators.dpo_synthesizer import DPOSynthesizer


def _create_mock_report() -> DiffReport:
    tc1 = TestCase(id="tc_1", vars={"query": "Explain quantum computing"})
    tc2 = TestCase(id="tc_2", vars={"query": "Draft email"})

    # tc1: v2 is superior (chosen = v2, rejected = v1)
    comp1 = ComparisonResult(
        test_case=tc1,
        v1_result=RunResult(
            prompt_name="v1",
            test_case_id="tc_1",
            rendered_prompt="Explain quantum computing simply",
            output="Quantum computing uses qubits and superposition.",
            latency_ms=100.0,
            prompt_tokens=10,
            completion_tokens=15,
            total_tokens=25,
            cost_usd=0.0001,
            model="gpt-4o",
        ),
        v2_result=RunResult(
            prompt_name="v2",
            test_case_id="tc_1",
            rendered_prompt="Explain quantum computing in detail",
            output="Quantum computing leverages quantum mechanics principles such as superposition and entanglement.",
            latency_ms=90.0,
            prompt_tokens=10,
            completion_tokens=25,
            total_tokens=35,
            cost_usd=0.0002,
            model="gpt-4o",
        ),
        scores={"quality": EvaluatorScore(name="quality", passed=True, v1_score=0.6, v2_score=0.95, delta=0.35)},
    )

    # tc2: tied scores (delta = 0.0, should be filtered out)
    comp2 = ComparisonResult(
        test_case=tc2,
        v1_result=RunResult(
            prompt_name="v1",
            test_case_id="tc_2",
            rendered_prompt="Draft email",
            output="Dear team...",
            latency_ms=50.0,
            prompt_tokens=5,
            completion_tokens=5,
            total_tokens=10,
            cost_usd=0.0001,
            model="gpt-4o",
        ),
        v2_result=RunResult(
            prompt_name="v2",
            test_case_id="tc_2",
            rendered_prompt="Draft email",
            output="Dear team...",
            latency_ms=50.0,
            prompt_tokens=5,
            completion_tokens=5,
            total_tokens=10,
            cost_usd=0.0001,
            model="gpt-4o",
        ),
        scores={"quality": EvaluatorScore(name="quality", passed=True, v1_score=0.8, v2_score=0.8, delta=0.0)},
    )

    return DiffReport(
        run_id="run_dpo_001",
        v1_name="v1",
        v2_name="v2",
        model_v1="gpt-4o",
        model_v2="gpt-4o",
        comparisons=[comp1, comp2],
        verdict=RegressionVerdict(passed=True),
        evaluators=["quality"],
        total_cases=2,
    )


def test_dpo_synthesizer_huggingface_trl_format() -> None:
    """Verify synthesized JSONL strictly matches Hugging Face TRL DPOTrainer triplet schema."""
    report = _create_mock_report()
    synthesizer = DPOSynthesizer()
    res = synthesizer.synthesize(report, min_delta_threshold=0.1)

    assert res.total_pairs == 1
    assert res.mean_margin > 0.0

    # Test JSONL serialization
    jsonl_str = res.to_jsonl()
    lines = [line for line in jsonl_str.strip().split("\n") if line.strip()]
    assert len(lines) == 1

    row = json.loads(lines[0])

    # Hugging Face TRL standard keys
    assert set(row.keys()) == {"prompt", "chosen", "rejected"}
    assert isinstance(row["prompt"], str) and len(row["prompt"]) > 0
    assert isinstance(row["chosen"], str) and len(row["chosen"]) > 0
    assert isinstance(row["rejected"], str) and len(row["rejected"]) > 0
    assert row["chosen"] != row["rejected"]
    assert "superposition and entanglement" in row["chosen"]
