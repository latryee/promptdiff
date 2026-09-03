"""Comprehensive unit tests for FaithfulnessEvaluator in evaluators/faithfulness.py."""

from __future__ import annotations

import pytest

from promptdiff.core.models import RunResult, TestCase
from promptdiff.evaluators.faithfulness import FaithfulnessEvaluator


def test_faithfulness_extract_context() -> None:
    ev = FaithfulnessEvaluator(force_mock=True)

    # Missing context
    assert ev._extract_context(TestCase(id="1", vars={})) is None

    # String context
    tc_str = TestCase(id="2", vars={"context": "Context document"})
    assert ev._extract_context(tc_str) == "Context document"

    # List of documents
    tc_list = TestCase(id="3", vars={"docs": ["Doc 1", "Doc 2"]})
    assert ev._extract_context(tc_list) == "Doc 1\n\nDoc 2"


def test_faithfulness_parse_output() -> None:
    ev = FaithfulnessEvaluator(force_mock=True)

    raw = (
        "[CLAIMS_EVALUATION]\n"
        "- Claim 1: [GROUNDED] - supported\n"
        "[HALLUCINATIONS]\n"
        "- Hallucination: False fact\n"
        "[SCORE] 0.75"
    )
    score, hallus = ev._parse_faithfulness_output(raw)
    assert score == 0.75
    assert len(hallus) == 1
    assert "False fact" in hallus[0]

    # Alternate match format
    alt_raw = "faithfulness: 0.90\n[HALLUCINATIONS] None"
    alt_score, alt_hallus = ev._parse_faithfulness_output(alt_raw)
    assert alt_score == 0.90
    assert len(alt_hallus) == 0


def test_faithfulness_heuristic_check() -> None:
    ev = FaithfulnessEvaluator(force_mock=True)
    score, hallus = ev._heuristic_check(
        context="The quick brown fox jumps over the lazy dog.",
        response="The brown fox jumps quickly.",
    )
    assert score > 0.0
    assert len(hallus) == 0

    # Empty response
    empty_score, _ = ev._heuristic_check("Context text", "")
    assert empty_score == 1.0


@pytest.mark.asyncio
async def test_faithfulness_evaluate_with_mock() -> None:
    ev = FaithfulnessEvaluator(threshold=0.70, force_mock=True)
    tc = TestCase(
        id="rag_tc",
        vars={"retrieved_context": "The server timeout is configured to 30 seconds."},
    )
    r1 = RunResult(
        prompt_name="v1",
        test_case_id="rag_tc",
        rendered_prompt="p",
        output="The server timeout is 30 seconds.",
        latency_ms=10.0,
        prompt_tokens=5,
        completion_tokens=5,
        total_tokens=10,
        cost_usd=0.0,
        model="mock",
    )
    r2 = RunResult(
        prompt_name="v2",
        test_case_id="rag_tc",
        rendered_prompt="p",
        output="The timeout is 30 seconds according to server settings.",
        latency_ms=10.0,
        prompt_tokens=5,
        completion_tokens=5,
        total_tokens=10,
        cost_usd=0.0,
        model="mock",
    )

    score = await ev.async_evaluate(r1, r2, tc)
    assert score.name == "faithfulness"
    assert score.passed is True
    assert score.v2_score > 0.5


def test_faithfulness_evaluate_sync_missing_context() -> None:
    ev = FaithfulnessEvaluator(force_mock=True)
    tc = TestCase(id="no_ctx", vars={})
    r = RunResult(
        prompt_name="v1",
        test_case_id="no_ctx",
        rendered_prompt="p",
        output="out",
        latency_ms=10.0,
        prompt_tokens=1,
        completion_tokens=1,
        total_tokens=2,
        cost_usd=0.0,
        model="m",
    )
    score = ev.evaluate(r, r, tc)
    assert score.passed is True
    assert "No reference context" in score.message
