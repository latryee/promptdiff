"""High-leverage test suite systematically targeting uncovered lines to maximize overall test coverage."""

from __future__ import annotations

import pytest

from promptdiff.core.models import (
    ArenaModelSummary,
    ArenaReport,
    ComparisonResult,
    DiffReport,
    EvaluatorScore,
    RegressionVerdict,
    RunResult,
    TestCase,
)
from promptdiff.evaluators.answer_relevance import AnswerRelevanceEvaluator
from promptdiff.evaluators.council import CouncilOfJudgesEvaluator
from promptdiff.evaluators.length_drift import LengthDriftEvaluator
from promptdiff.evaluators.trajectory import TrajectoryEvaluator, extract_tool_calls
from promptdiff.optimizer.tuner import HyperparameterConfig, TuneCandidateResult, TuningReport
from promptdiff.providers.registry import get_provider
from promptdiff.reporters.terminal import (
    render_arena_terminal_report,
    render_terminal_report,
    render_tuning_terminal_report,
)


def test_length_drift_evaluator_coverage() -> None:
    """Exercise LengthDriftEvaluator on verbosity expansion and reduction."""
    evaluator = LengthDriftEvaluator()
    tc = TestCase(id="tc1")

    # 1. Expansion (+delta)
    r1 = RunResult(
        prompt_name="v1",
        test_case_id="tc1",
        rendered_prompt="P1",
        output="Short answer.",
        latency_ms=50.0,
        prompt_tokens=5,
        completion_tokens=5,
        total_tokens=10,
        cost_usd=0.0001,
        model="gpt-4o",
    )
    r2 = RunResult(
        prompt_name="v2",
        test_case_id="tc1",
        rendered_prompt="P2",
        output="This is a substantially longer and much more elaborate answer with extra details.",
        latency_ms=100.0,
        prompt_tokens=5,
        completion_tokens=20,
        total_tokens=25,
        cost_usd=0.0002,
        model="gpt-4o",
    )

    score = evaluator.evaluate(r1, r2, tc)
    assert score.passed is True
    assert score.delta == 15.0
    assert score.delta_pct > 0.0
    assert "+" in score.message

    # 2. Reduction (-delta)
    score_red = evaluator.evaluate(r2, r1, tc)
    assert score_red.delta == -15.0
    assert score_red.delta_pct < 0.0

    # 3. Empty v1 output
    r_empty = RunResult(
        prompt_name="v1",
        test_case_id="tc1",
        rendered_prompt="P1",
        output="",
        latency_ms=10.0,
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        cost_usd=0.0,
        model="gpt-4o",
    )
    score_empty = evaluator.evaluate(r_empty, r2, tc)
    assert score_empty.delta_pct == 0.0


def test_answer_relevance_evaluator_edge_cases() -> None:
    """Exercise answer relevance edge cases: empty responses, sync evaluate, and alternate score patterns."""
    evaluator = AnswerRelevanceEvaluator(force_mock=True)
    tc = TestCase(id="tc_rel", vars={"query": "How to setup SSO?"})

    r1 = RunResult(
        prompt_name="v1",
        test_case_id="tc_rel",
        rendered_prompt="P1",
        output="Navigate to settings -> SSO -> configure SAML.",
        latency_ms=50.0,
        prompt_tokens=10,
        completion_tokens=10,
        total_tokens=20,
        cost_usd=0.0001,
        model="gpt-4o",
    )
    r_empty = RunResult(
        prompt_name="v2",
        test_case_id="tc_rel",
        rendered_prompt="P2",
        output="",
        latency_ms=10.0,
        prompt_tokens=5,
        completion_tokens=0,
        total_tokens=5,
        cost_usd=0.0001,
        model="gpt-4o",
    )

    # Empty candidate output
    score_empty = evaluator.evaluate(r1, r_empty, tc)
    assert score_empty.v2_score == 0.0
    assert "Empty response" in score_empty.message

    # Test alternate parsing patterns
    s1, reason1 = evaluator._parse_relevance_output("[SCORE] 0.95\n[REASONING] Very relevant")
    assert s1 == 0.95
    assert reason1 == "Very relevant"

    s_bad, _ = evaluator._parse_relevance_output("[SCORE] not_a_number\n[REASONING] failed")
    assert s_bad == 0.85

    s_alt_bad, _ = evaluator._parse_relevance_output("relevance: not_a_number")
    assert s_alt_bad == 0.85

    # Embedding fallback method
    emb_sim = evaluator._compute_embedding_similarity("query", "response")
    assert 0.0 <= emb_sim <= 1.0


def test_terminal_reports_deep_coverage() -> None:
    """Exercise terminal reporter: forecast panel, regression failures, tuning, and arena reports."""
    from rich.console import Console

    console = Console(record=True)

    tc = TestCase(id="tc_fail", vars={"query": "Test query"})
    r1 = RunResult(
        prompt_name="v1",
        test_case_id="tc_fail",
        rendered_prompt="p1",
        output="out1",
        latency_ms=50.0,
        prompt_tokens=5,
        completion_tokens=5,
        total_tokens=10,
        cost_usd=0.0005,
        model="gpt-4o",
    )
    r2 = RunResult(
        prompt_name="v2",
        test_case_id="tc_fail",
        rendered_prompt="p2",
        output="out2",
        latency_ms=150.0,
        prompt_tokens=5,
        completion_tokens=5,
        total_tokens=10,
        cost_usd=0.0015,
        model="gpt-4o",
    )
    comp = ComparisonResult(
        test_case=tc,
        v1_result=r1,
        v2_result=r2,
        scores={"latency": EvaluatorScore(name="latency", passed=False, v1_score=50.0, v2_score=150.0, delta=100.0)},
    )
    fail_report = DiffReport(
        run_id="run_fail_01",
        v1_name="v1",
        v2_name="v2",
        model_v1="gpt-4o",
        model_v2="gpt-4o",
        comparisons=[comp],
        verdict=RegressionVerdict(
            passed=False,
            status="REGRESSION_DETECTED",
            failed_assertions=["latency < 100ms"],
            total_cost_v1=0.0005,
            total_cost_v2=0.0015,
            cost_delta_pct=200.0,
        ),
        evaluators=["latency"],
        total_cases=1,
    )

    # 1. Render terminal report with failed assertions and forecast
    render_terminal_report(fail_report, console=console, forecast="1M")
    output = console.export_text()
    assert "REGRESSION DETECTED" in output
    assert "Production Volume Cost Impact Forecast" in output

    # 2. Render Tuning Report
    c1 = TuneCandidateResult(
        config=HyperparameterConfig(temperature=0.2, top_p=0.9),
        avg_judge_score=4.8,
        avg_latency_ms=45.0,
        avg_tokens=15.0,
        total_cost=0.0002,
        passed_rate=1.0,
        is_pareto_optimal=True,
        utility_score=0.95,
        rank=1,
    )
    c2 = TuneCandidateResult(
        config=HyperparameterConfig(temperature=0.8, top_p=1.0),
        avg_judge_score=4.1,
        avg_latency_ms=65.0,
        avg_tokens=18.0,
        total_cost=0.0003,
        passed_rate=0.85,
        is_pareto_optimal=False,
        utility_score=0.75,
        rank=2,
    )
    t_report = TuningReport(
        prompt_name="test_prompt",
        model_name="gpt-4o",
        total_configs_tested=2,
        best_config=c1.config,
        pareto_candidates=[c1],
        all_results=[c1, c2],
    )
    render_tuning_terminal_report(t_report, console=console)
    tune_output = console.export_text()
    assert "Pareto Optimal Frontier" in tune_output

    # 3. Render Arena Report
    arena_summary = [
        ArenaModelSummary(
            name="v1",
            model="gpt-4o",
            rank=1,
            total_cost=0.001,
            avg_latency_ms=45.0,
            avg_tokens=15.0,
            win_rate_pct=85.0,
        ),
        ArenaModelSummary(
            name="v2",
            model="claude-3-5-sonnet",
            rank=2,
            total_cost=0.002,
            avg_latency_ms=60.0,
            avg_tokens=20.0,
            win_rate_pct=70.0,
        ),
    ]
    arena_report = ArenaReport(
        run_id="arena_01",
        variants=["v1", "v2"],
        models={"v1": "gpt-4o", "v2": "claude-3-5-sonnet"},
        leaderboard=arena_summary,
        comparisons=[],
        total_cases=5,
    )
    render_arena_terminal_report(arena_report, console=console)
    arena_output = console.export_text()
    assert "Multi-Model Arena Leaderboard" in arena_output
    assert "Arena Winner: v1" in arena_output


def test_provider_registry_coverage() -> None:
    """Exercise provider registry for mock, local, and custom configurations."""
    p_mock = get_provider("gpt-4o", force_mock=True)
    assert type(p_mock).__name__ == "MockProvider"

    p_gemini = get_provider("gemini-2.0-flash", api_key="dummy_gemini_key")
    assert type(p_gemini).__name__ == "GeminiProvider"

    p_claude = get_provider("claude-3-5-sonnet", api_key="dummy_anthropic_key")
    assert type(p_claude).__name__ == "AnthropicProvider"

    p_ollama = get_provider("ollama/llama3")
    assert type(p_ollama).__name__ == "OllamaProvider"

    p_openai = get_provider("gpt-4o-mini", api_key="dummy_openai_key")
    assert type(p_openai).__name__ == "OpenAIProvider"


def test_main_module_entrypoint() -> None:
    """Exercise promptdiff.__main__ execution via python -m promptdiff."""
    import subprocess
    import sys

    res = subprocess.run(
        [sys.executable, "-m", "promptdiff", "--help"],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert res.returncode == 0
    assert "Usage" in (res.stdout or "") or "promptdiff" in (res.stdout or "")


@pytest.mark.asyncio
async def test_trajectory_evaluator_deep_coverage() -> None:
    """Exercise TrajectoryEvaluator tool calling step matching and validation."""
    # Test extract_tool_calls on XML and JSON code blocks
    xml_text = '<tool_call>{"tool": "calculator", "args": {"expr": "2+2"}}</tool_call>'
    json_block = '```json\n{"action": "search", "query": "PromptDiff"}\n```'
    extracted_xml = extract_tool_calls(xml_text)
    extracted_json = extract_tool_calls(json_block)
    assert len(extracted_xml) == 1
    assert extracted_xml[0]["tool"] == "calculator"
    assert len(extracted_json) == 1
    assert extracted_json[0]["action"] == "search"

    evaluator = TrajectoryEvaluator(force_mock=True)

    v1_tool_output = """
    Step 1: Thought: I need to query weather.
    Action: weather_api
    Action Input: {"city": "Paris"}
    Observation: Sunny, 22C
    Final Answer: Paris is sunny and 22C.
    """
    v2_tool_output = """
    Step 1: Thought: I will fetch weather for Paris.
    Action: weather_api
    Action Input: {"city": "Paris"}
    Observation: Sunny, 22C
    Final Answer: Paris is sunny and 22C.
    """

    tc = TestCase(id="tc_traj", vars={"city": "Paris"}, expected_output="Paris is sunny")

    r1 = RunResult(
        prompt_name="v1",
        test_case_id="tc_traj",
        rendered_prompt="Query weather",
        output=v1_tool_output,
        latency_ms=100.0,
        prompt_tokens=10,
        completion_tokens=30,
        total_tokens=40,
        cost_usd=0.0001,
        model="gpt-4o",
    )
    r2 = RunResult(
        prompt_name="v2",
        test_case_id="tc_traj",
        rendered_prompt="Query weather",
        output=v2_tool_output,
        latency_ms=90.0,
        prompt_tokens=10,
        completion_tokens=28,
        total_tokens=38,
        cost_usd=0.0001,
        model="gpt-4o",
    )

    score = await evaluator.async_evaluate(r1, r2, tc)
    assert score.passed is True
    assert score.v2_score > 0.5


@pytest.mark.asyncio
async def test_evaluation_council_coverage() -> None:
    """Exercise CouncilOfJudgesEvaluator with multi-evaluator scoring and weighted debate."""
    council = CouncilOfJudgesEvaluator(force_mock=True)
    tc = TestCase(id="tc_council")
    r1 = RunResult(
        prompt_name="v1",
        test_case_id="tc_council",
        rendered_prompt="p",
        output='{"status": "ok"}',
        latency_ms=50.0,
        prompt_tokens=5,
        completion_tokens=5,
        total_tokens=10,
        cost_usd=0.0001,
        model="gpt-4o",
    )
    r2 = RunResult(
        prompt_name="v2",
        test_case_id="tc_council",
        rendered_prompt="p",
        output='{"status": "ok", "version": 2}',
        latency_ms=45.0,
        prompt_tokens=5,
        completion_tokens=8,
        total_tokens=13,
        cost_usd=0.0001,
        model="gpt-4o",
    )

    score = await council.async_evaluate(r1, r2, tc)
    assert score.passed is True
    assert score.name == "council"
    # Also test sync evaluate
    sync_score = council.evaluate(r1, r2, tc)
    assert sync_score.passed is True
