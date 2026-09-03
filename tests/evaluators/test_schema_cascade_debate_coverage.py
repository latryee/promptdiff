"""Deep unit tests for schema breaking change detector, cascaded judge, and debate evaluators."""

from __future__ import annotations

import json

import pytest

from promptdiff.core.models import RunResult, TestCase
from promptdiff.evaluators.cascaded_judge import (
    CascadedJudgeResult,
    CascadedLLMJudge,
    CascadedLLMJudgeEvaluator,
)
from promptdiff.evaluators.debate import (
    DebateRound,
    MultiAgentDebateEvaluator,
)
from promptdiff.evaluators.schema_breaking import (
    SchemaBreakingChangeDetector,
    SchemaBreakingChangeEvaluator,
)

# ============================================================================
# schema_breaking.py tests
# ============================================================================


def test_schema_breaking_parse_errors() -> None:
    detector = SchemaBreakingChangeDetector()

    # Invalid v1
    rep1 = detector.evaluate("not json {", '{"a": 1}')
    assert rep1.is_compatible is False
    assert any(d.change_type == "PARSE_ERROR_V1" for d in rep1.differences)

    # Invalid v2
    rep2 = detector.evaluate('{"a": 1}', "broken json [")
    assert rep2.is_compatible is False
    assert any(d.change_type == "PARSE_ERROR_V2" for d in rep2.differences)


def test_schema_breaking_empty_array_and_structural_diffs() -> None:
    detector = SchemaBreakingChangeDetector()

    # Empty array warning
    v1_arr = json.dumps({"items": [1, 2, 3]})
    v2_arr = json.dumps({"items": []})
    rep_arr = detector.evaluate(v1_arr, v2_arr)
    assert any(d.change_type == "EMPTY_ARRAY" for d in rep_arr.differences)

    # Type change (breaking)
    v1_type = json.dumps({"id": 123})
    v2_type = json.dumps({"id": "abc"})
    rep_type = detector.evaluate(v1_type, v2_type)
    assert rep_type.has_breaking_changes is True
    assert any(d.change_type == "TYPE_MUTATION" for d in rep_type.differences)

    # Removed key (breaking)
    v1_key = json.dumps({"required_field": "val", "other": 1})
    v2_key = json.dumps({"other": 1})
    rep_key = detector.evaluate(v1_key, v2_key)
    assert rep_key.has_breaking_changes is True
    assert any(d.change_type == "REMOVED_FIELD" for d in rep_key.differences)


@pytest.mark.asyncio
async def test_schema_breaking_evaluator_evaluate() -> None:
    ev = SchemaBreakingChangeEvaluator()
    tc = TestCase(id="schema_tc", vars={})
    r1 = RunResult(
        prompt_name="v1",
        test_case_id="schema_tc",
        rendered_prompt="p",
        output='{"name": "test", "status": "active"}',
        latency_ms=10.0,
        prompt_tokens=5,
        completion_tokens=5,
        total_tokens=10,
        cost_usd=0.0,
        model="mock",
    )
    r2 = RunResult(
        prompt_name="v2",
        test_case_id="schema_tc",
        rendered_prompt="p",
        output='{"name": "test", "status": "active", "extra": true}',
        latency_ms=10.0,
        prompt_tokens=5,
        completion_tokens=5,
        total_tokens=10,
        cost_usd=0.0,
        model="mock",
    )
    score = await ev.async_evaluate(r1, r2, tc)
    assert score.passed is True
    assert score.v2_score == 1.0


# ============================================================================
# cascaded_judge.py tests
# ============================================================================


@pytest.mark.asyncio
async def test_cascaded_judge_async() -> None:
    judge = CascadedLLMJudge(force_mock=True)
    res = await judge.async_judge("Response A", "Response B", query="How are you?")
    assert isinstance(res, CascadedJudgeResult)
    assert res.winner in ("v1", "v2", "tie")
    assert res.confidence >= 0.0


@pytest.mark.asyncio
async def test_cascaded_judge_evaluator_evaluate() -> None:
    ev = CascadedLLMJudgeEvaluator(force_mock=True)
    tc = TestCase(id="casc_tc", vars={"query": "Explain quantum entanglement"})
    r1 = RunResult(
        prompt_name="v1",
        test_case_id="casc_tc",
        rendered_prompt="p",
        output="Entanglement is physics.",
        latency_ms=10.0,
        prompt_tokens=5,
        completion_tokens=5,
        total_tokens=10,
        cost_usd=0.0,
        model="mock",
    )
    r2 = RunResult(
        prompt_name="v2",
        test_case_id="casc_tc",
        rendered_prompt="p",
        output="Quantum entanglement is a phenomenon where particles share quantum states.",
        latency_ms=10.0,
        prompt_tokens=5,
        completion_tokens=5,
        total_tokens=10,
        cost_usd=0.0,
        model="mock",
    )
    score = await ev.async_evaluate(r1, r2, tc)
    assert score.name == "cascaded_judge"
    assert "winner" in score.details


# ============================================================================
# debate.py tests
# ============================================================================


@pytest.mark.asyncio
async def test_multi_agent_debate_evaluator() -> None:
    ev = MultiAgentDebateEvaluator(force_mock=True)
    tc = TestCase(id="deb_tc", vars={"query": "Summarize policy"})
    r1 = RunResult(
        prompt_name="v1",
        test_case_id="deb_tc",
        rendered_prompt="p",
        output="Here is a very long and verbose text with many repeated phrases.",
        latency_ms=10.0,
        prompt_tokens=5,
        completion_tokens=15,
        total_tokens=20,
        cost_usd=0.0,
        model="mock",
    )
    r2 = RunResult(
        prompt_name="v2",
        test_case_id="deb_tc",
        rendered_prompt="p",
        output="Concise policy summary.",
        latency_ms=10.0,
        prompt_tokens=5,
        completion_tokens=5,
        total_tokens=10,
        cost_usd=0.0,
        model="mock",
    )
    round_res = await ev.conduct_debate("Summarize policy", r1.output, r2.output)
    assert isinstance(round_res, DebateRound)
    assert round_res.winner in ("v1", "v2", "TIE")

    score = await ev.async_evaluate(r1, r2, tc)
    assert score.name == "debate_judge"
    assert score.passed is True
