"""Tests migrated to feature-focused test suite."""

from __future__ import annotations

import pytest

from promptdiff.core.models import (
    RunResult,
    TestCase,
)
from promptdiff.evaluators.similarity import SimilarityEvaluator


def test_sentence_transformers_similarity() -> None:
    """Test sentence-transformers semantic similarity evaluator."""
    pytest.importorskip("sentence_transformers", reason="sentence-transformers required for dense neural similarity")
    sim_eval = SimilarityEvaluator(model_name="all-MiniLM-L6-v2")

    tc = TestCase(id="tc_sim", vars={})
    v1_res = RunResult(
        prompt_name="v1",
        test_case_id="tc_sim",
        rendered_prompt="test",
        output="The quick brown fox jumps over the lazy dog.",
        latency_ms=50.0,
        prompt_tokens=10,
        completion_tokens=10,
        total_tokens=20,
        cost_usd=0.0001,
        model="mock",
    )
    v2_res = RunResult(
        prompt_name="v2",
        test_case_id="tc_sim",
        rendered_prompt="test",
        output="A fast brown fox leaped over a sleepy dog.",
        latency_ms=50.0,
        prompt_tokens=10,
        completion_tokens=10,
        total_tokens=20,
        cost_usd=0.0001,
        model="mock",
    )

    score = sim_eval.evaluate(v1_res, v2_res, tc)
    assert score.name == "similarity"
    assert score.v2_score > 0.70
    assert score.passed is True


def test_similarity_evaluator_fallback_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test graceful fallback to token overlap when sentence-transformers is not available."""
    import promptdiff.evaluators.similarity as sim_mod

    monkeypatch.setattr(sim_mod, "_get_embedding_model", lambda _model_name="": None)

    sim_eval = SimilarityEvaluator(threshold=0.40)
    tc = TestCase(id="tc_sim_fb", vars={})
    v1_res = RunResult(
        prompt_name="v1",
        test_case_id="tc_sim_fb",
        rendered_prompt="test",
        output="The quick brown fox jumps over the lazy dog.",
        latency_ms=50.0,
        prompt_tokens=10,
        completion_tokens=10,
        total_tokens=20,
        cost_usd=0.0001,
        model="mock",
    )
    v2_res = RunResult(
        prompt_name="v2",
        test_case_id="tc_sim_fb",
        rendered_prompt="test",
        output="A fast brown fox leaped over a sleepy dog.",
        latency_ms=50.0,
        prompt_tokens=10,
        completion_tokens=10,
        total_tokens=20,
        cost_usd=0.0001,
        model="mock",
    )

    score = sim_eval.evaluate(v1_res, v2_res, tc)
    assert score.name == "similarity"
    assert score.details.get("fallback") is True
    assert score.v2_score > 0.40
    assert score.passed is True
