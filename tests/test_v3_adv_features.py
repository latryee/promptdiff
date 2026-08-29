"""Test Suite for Next-Gen Advanced Features: TUI, Hyperparameter Tuning, Cost Forecasting, and PR Bot."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from promptdiff.core.models import PromptVersion, TestCase
from promptdiff.optimizer.tuner import (
    HyperparameterConfig,
    PromptTuner,
    TuneCandidateResult,
    compute_pareto_frontier,
)
from promptdiff.pricing import calculate_forecast, parse_volume_string
from promptdiff.providers.mock_provider import MockProvider
from promptdiff.reporters.pr_bot import (
    STICKY_HEADER_TAG,
    generate_pr_markdown_comment,
    parse_pr_number_from_event,
)


def test_volume_parsing() -> None:
    """Test string volume parsing."""
    assert parse_volume_string("1M") == 1_000_000
    assert parse_volume_string("500k") == 500_000
    assert parse_volume_string("2.5M") == 2_500_000
    assert parse_volume_string("100,000") == 100_000
    assert parse_volume_string(10_000) == 10_000


def test_cost_forecasting() -> None:
    """Test cost projection over daily, monthly, and annual volume."""
    fc = calculate_forecast(
        total_cost_v1=0.0004,
        total_cost_v2=0.0002,
        total_cases=4,
        daily_volume="1M",
    )

    assert fc.daily_volume == 1_000_000
    assert fc.monthly_volume == 30_000_000
    assert fc.v1_monthly_cost == 3000.0
    assert fc.v2_monthly_cost == 1500.0
    assert fc.monthly_savings_usd == 1500.0
    assert fc.annual_savings_usd == 18000.0
    assert fc.cost_delta_pct == -50.0
    assert "Projected Savings: $1,500.00/mo" in fc.summary_text


def test_pareto_frontier_computation() -> None:
    """Test multi-objective Pareto dominance calculation."""
    c1 = TuneCandidateResult(
        config=HyperparameterConfig(0.0, 0.7),
        avg_judge_score=4.8,
        avg_latency_ms=150.0,
        avg_tokens=40.0,
        total_cost=0.0001,
        passed_rate=1.0,
    )
    c2 = TuneCandidateResult(
        config=HyperparameterConfig(1.0, 1.0),
        avg_judge_score=3.2,
        avg_latency_ms=300.0,
        avg_tokens=80.0,
        total_cost=0.0003,
        passed_rate=0.5,
    )

    pareto = compute_pareto_frontier([c1, c2])
    assert len(pareto) == 1
    assert pareto[0].config.temperature == 0.0
    assert c1.is_pareto_optimal is True
    assert c2.is_pareto_optimal is False
    assert c1.rank == 1


@pytest.mark.asyncio
async def test_prompt_tuner_grid_search() -> None:
    """Test hyperparameter tuning grid search with mock provider."""
    provider = MockProvider(force_mock=True)

    pv = PromptVersion(
        name="support_bot",
        template="Answer concisely: {{query}}",
        model="mock-gpt-4o",
    )

    test_cases = [
        TestCase(id="t1", vars={"query": "Hello"}),
        TestCase(id="t2", vars={"query": "Refund please"}),
    ]

    tuner = PromptTuner(
        prompt_version=pv,
        test_cases=test_cases,
        provider=provider,
        temperatures=[0.0, 0.7],
        top_ps=[0.7, 1.0],
        force_mock=True,
    )

    report = await tuner.tune()
    assert report.total_configs_tested == 4
    assert len(report.all_results) == 4
    assert len(report.pareto_candidates) >= 1
    assert report.best_config is not None


def test_pr_commenter_markdown_generation(tmp_path: Path) -> None:
    """Test GitHub PR comment generation."""
    sample_report = {
        "v1_name": "system_v1.txt",
        "v2_name": "system_v2.txt",
        "model_v2": "gpt-4o",
        "total_cases": 2,
        "verdict": {
            "passed": True,
            "status": "PASSED",
            "failed_assertions": [],
            "total_cost_v1": 0.0002,
            "total_cost_v2": 0.0001,
            "cost_delta_pct": -50.0,
            "avg_latency_v1": 200.0,
            "avg_latency_v2": 150.0,
            "latency_delta_pct": -25.0,
        },
        "aggregate_stats": {"passed_cases": 2},
        "evaluators": ["similarity", "llm_judge"],
        "comparisons": [
            {
                "test_case": {"id": "c1", "description": "Greeting"},
                "v1_result": {"output": "Hello world", "latency_ms": 200.0},
                "v2_result": {"output": "Hi world", "latency_ms": 150.0},
                "scores": {
                    "similarity": {"v1_score": 1.0, "v2_score": 0.9, "passed": True},
                    "llm_judge": {"v1_score": 4.0, "v2_score": 4.8, "passed": True},
                },
            }
        ],
    }

    body = generate_pr_markdown_comment(sample_report, forecast_vol="500k")
    assert STICKY_HEADER_TAG in body
    assert "All Quality Gates Passed" in body
    assert "-50.0%" in body
    assert "Projected Monthly Impact" in body

    # Event payload parsing
    event_file = tmp_path / "event.json"
    event_file.write_text(json.dumps({"pull_request": {"number": 42}}), encoding="utf-8")
    pr_num = parse_pr_number_from_event(str(event_file))
    assert pr_num == 42
