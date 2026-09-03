"""Test Suite for Roadmap Features: Pytest Plugin, Token Compressor, Agent Trajectory, OTEL Exporter, and Python SDK."""

from __future__ import annotations

import pytest

import promptdiff
from promptdiff.core.models import PromptVersion, RunResult, TestCase
from promptdiff.evaluators.trajectory import TrajectoryEvaluator, extract_tool_calls
from promptdiff.optimizer.compressor import PromptCompressor, estimate_tokens


def test_token_estimation() -> None:
    """Test heuristic token estimation."""
    text = "You are a helpful customer support agent."
    tokens = estimate_tokens(text)
    assert tokens > 0
    assert tokens >= 5


@pytest.mark.asyncio
async def test_prompt_compressor() -> None:
    """Test prompt token compressor with quality retention."""
    pv = PromptVersion(
        name="verbose_support",
        template="You are a helpful assistant that answers queries politely. Please kindly assist: {{query}}",
        model="mock-gpt-4o",
    )
    test_cases = [
        TestCase(id="c1", vars={"query": "Reset password"}),
        TestCase(id="c2", vars={"query": "Billing help"}),
    ]

    compressor = PromptCompressor(
        prompt_version=pv,
        test_cases=test_cases,
        target_reduction=0.30,
        force_mock=True,
    )

    res = await compressor.compress()
    assert res.compressed_tokens <= res.original_tokens
    assert res.token_reduction_pct >= 0.0
    assert res.quality_retained_pct > 0.0
    assert res.compressed_prompt is not None


def test_extract_tool_calls() -> None:
    """Test extraction of tool calls from agent outputs."""
    output_xml = '<tool_call>{"name": "search_db", "arguments": {"query": "user_123"}}</tool_call>'
    calls = extract_tool_calls(output_xml)
    assert len(calls) == 1
    assert calls[0].get("name") == "search_db"

    output_json_block = '```json\n{"action": "send_email", "to": "test@example.com"}\n```'
    calls_json = extract_tool_calls(output_json_block)
    assert len(calls_json) == 1
    assert calls_json[0].get("action") == "send_email"


@pytest.mark.asyncio
async def test_trajectory_evaluator() -> None:
    """Test agent multi-turn trajectory evaluator."""
    ev = TrajectoryEvaluator(force_mock=True)

    tc = TestCase(
        id="agent_case_1",
        vars={
            "trajectory": [
                {"role": "user", "content": "Fetch order #999 and refund"},
                {"role": "assistant", "content": "Let me lookup the order."},
            ]
        },
    )

    r1 = RunResult(
        prompt_name="agent_v1",
        test_case_id="agent_case_1",
        rendered_prompt="test",
        output='<tool_call>{"name": "get_order", "id": 999}</tool_call>',
        latency_ms=150.0,
        prompt_tokens=50,
        completion_tokens=20,
        total_tokens=70,
        cost_usd=0.0001,
        model="gpt-4o",
    )
    r2 = RunResult(
        prompt_name="agent_v2",
        test_case_id="agent_case_1",
        rendered_prompt="test",
        output='<tool_call>{"name": "get_order", "id": 999}</tool_call><tool_call>{"name": "issue_refund", "id": 999}</tool_call>',
        latency_ms=160.0,
        prompt_tokens=50,
        completion_tokens=30,
        total_tokens=80,
        cost_usd=0.00015,
        model="gpt-4o",
    )

    score = await ev.async_evaluate(r1, r2, tc)
    assert score.name == "trajectory"
    assert score.passed is True
    assert score.v2_score >= 0.80


def test_python_sdk_compare() -> None:
    """Test top-level SDK functions."""
    report = promptdiff.compare(
        v1="Hello: {{query}}",
        v2="Hi: {{query}}",
        dataset=[{"id": "t1", "vars": {"query": "World"}}],
        mock=True,
    )
    assert report.verdict.passed is True
    assert len(report.comparisons) == 1

    opt = promptdiff.optimize(
        prompt="Support bot: {{query}}",
        dataset=[{"id": "t1", "vars": {"query": "Help"}}],
        iterations=1,
        mock=True,
    )
    assert opt.optimized_prompt is not None

    shrunk = promptdiff.shrink(
        prompt="Please kindly answer the user: {{query}}",
        dataset=[{"id": "t1", "vars": {"query": "Help"}}],
        mock=True,
    )
    assert shrunk.compressed_prompt is not None


def test_pytest_plugin_fixture(promptdiff_eval: pytest.FixtureRequest) -> None:
    """Test pytest fixture provided by pytest_plugin."""
    report = promptdiff_eval(
        v1="Say hello: {{name}}",
        v2="Greet user: {{name}}",
        dataset=[TestCase(id="1", vars={"name": "Alice"})],
        mock=True,
    )
    assert report.total_cases == 1
    assert report.verdict.passed is True
