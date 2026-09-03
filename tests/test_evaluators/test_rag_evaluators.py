"""Tests migrated to feature-focused test suite."""

from __future__ import annotations

import pytest

from promptdiff.core.models import RunResult, TestCase
from promptdiff.evaluators.answer_relevance import AnswerRelevanceEvaluator
from promptdiff.evaluators.citation import CitationEvaluator
from promptdiff.evaluators.fact_graph import FactGraphEvaluator, extract_triplets_heuristic
from promptdiff.evaluators.faithfulness import FaithfulnessEvaluator
from promptdiff.evaluators.hallucination_graph import (
    TokenAttributionEvaluator,
    _compute_span_similarity,
    _extract_claim_spans,
)
from promptdiff.providers.mock_provider import MockProvider


@pytest.mark.asyncio
async def test_faithfulness_evaluator() -> None:
    """Test RAG faithfulness & hallucination detection evaluator."""
    provider = MockProvider(force_mock=True)
    evaluator = FaithfulnessEvaluator(provider=provider, threshold=0.80)

    # 1. Test case with reference context
    tc = TestCase(
        id="rag_tc1",
        description="Password policy check",
        vars={
            "query": "What is the password requirement?",
            "context": "Passwords must contain at least 12 characters, including one uppercase letter and one symbol.",
        },
    )

    v1_res = RunResult(
        prompt_name="v1",
        test_case_id="rag_tc1",
        rendered_prompt="",
        output="Passwords must be at least 12 characters long and include an uppercase letter and a symbol.",
        latency_ms=100.0,
        prompt_tokens=20,
        completion_tokens=15,
        total_tokens=35,
        cost_usd=0.0001,
        model="gpt-4o",
    )

    v2_res = RunResult(
        prompt_name="v2",
        test_case_id="rag_tc1",
        rendered_prompt="",
        output="Requirements:\n- Min 12 chars\n- 1 uppercase\n- 1 symbol",
        latency_ms=80.0,
        prompt_tokens=20,
        completion_tokens=12,
        total_tokens=32,
        cost_usd=0.00008,
        model="gpt-4o",
    )

    score = await evaluator.async_evaluate(v1_res, v2_res, tc)
    assert score.name == "faithfulness"
    assert score.passed is True
    assert score.v2_score >= 0.8
    assert score.details["context_provided"] is True


@pytest.mark.asyncio
async def test_answer_relevance_evaluator() -> None:
    """Test RAG answer relevance evaluator."""
    provider = MockProvider(force_mock=True)
    evaluator = AnswerRelevanceEvaluator(provider=provider, threshold=0.75)

    tc = TestCase(
        id="rel_tc1",
        description="Pricing tier inquiry",
        vars={"query": "How much does Enterprise plan cost?"},
    )

    v1_res = RunResult(
        prompt_name="v1",
        test_case_id="rel_tc1",
        rendered_prompt="",
        output="Hello! Our company has been in business for 10 years. Contact sales for pricing.",
        latency_ms=150.0,
        prompt_tokens=20,
        completion_tokens=15,
        total_tokens=35,
        cost_usd=0.0001,
        model="gpt-4o",
    )

    v2_res = RunResult(
        prompt_name="v2",
        test_case_id="rel_tc1",
        rendered_prompt="",
        output="Enterprise plan starts at $499/month with custom SLAs. Contact sales to activate.",
        latency_ms=100.0,
        prompt_tokens=20,
        completion_tokens=14,
        total_tokens=34,
        cost_usd=0.00008,
        model="gpt-4o",
    )

    score = await evaluator.async_evaluate(v1_res, v2_res, tc)
    assert score.name == "answer_relevance"
    assert score.passed is True
    assert score.v2_score >= 0.75


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


def test_fact_graph_evaluator() -> None:
    """Test knowledge triplet extraction and fact verification."""
    text = "PromptDiff was created by AI Engineers in 2026. It supports Python 3.13."
    triplets = extract_triplets_heuristic(text)
    assert len(triplets) >= 1

    evaluator = FactGraphEvaluator()
    fidelity = evaluator.compute_triplet_fidelity(triplets, text)
    assert fidelity >= 0.50


def test_hallucination_graph_attribution() -> None:
    """Test claim segmentation, bipartite grounding graph, and token hallucination rate."""
    context = "PromptDiff is an enterprise prompt regression testing framework released in 2026."
    output_grounded = "PromptDiff is an enterprise prompt regression testing tool. It was launched in 2026."
    output_hallucinated = "PromptDiff was created in 1995 by an alien spaceship in Atlantis."

    spans = _extract_claim_spans(output_grounded)
    assert len(spans) >= 1

    score_good, _ = _compute_span_similarity(spans[0][0], context)
    assert score_good > 0.40

    evaluator = TokenAttributionEvaluator()
    res_grounded = evaluator.analyze(output_grounded, context)
    assert res_grounded.grounding_accuracy_pct >= 50.0

    res_hallucinated = evaluator.analyze(output_hallucinated, context)
    assert res_hallucinated.token_hallucination_rate_pct >= 60.0
    assert len(res_hallucinated.bipartite_edges) >= 0


@pytest.mark.asyncio
async def test_token_attribution_evaluator_runner() -> None:
    """Test TokenAttributionEvaluator integrated as an evaluator in runner."""
    evaluator = TokenAttributionEvaluator()
    tc = TestCase(id="t1", vars={"context": "Acme Corp refund policy gives users 30 days."})

    r1 = RunResult(
        prompt_name="p1",
        test_case_id="t1",
        rendered_prompt="x",
        output="Refunds are 30 days.",
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
        output="Refunds are 30 days at Acme Corp.",
        latency_ms=100.0,
        prompt_tokens=10,
        completion_tokens=10,
        total_tokens=20,
        cost_usd=0.0001,
        model="gpt-4o",
    )

    score = await evaluator.async_evaluate(r1, r2, tc)
    assert score.name == "token_attribution"
    assert score.passed is True
