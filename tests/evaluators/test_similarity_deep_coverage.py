"""Deep unit tests for SimilarityEvaluator and similarity math helpers."""

from __future__ import annotations

import numpy as np
import pytest

from promptdiff.core.models import RunResult, TestCase
from promptdiff.evaluators.similarity import (
    SimilarityEvaluator,
    _emit_fallback_warning,
    cosine_similarity,
    jaccard_similarity,
    sequence_similarity,
    tokenize,
)


def test_similarity_math_helpers() -> None:
    # Cosine similarity with valid vectors
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([1.0, 0.0, 0.0])
    assert cosine_similarity(v1, v2) == 1.0

    # Orthogonal vectors
    v3 = np.array([0.0, 1.0, 0.0])
    assert cosine_similarity(v1, v3) == 0.0

    # Zero vectors
    v_zero = np.array([0.0, 0.0, 0.0])
    assert cosine_similarity(v1, v_zero) == 0.0

    # Tokenize
    tokens = tokenize("Hello, World! 123")
    assert "hello" in tokens
    assert "world" in tokens
    assert "123" in tokens

    # Jaccard similarity
    assert jaccard_similarity("", "") == 1.0
    assert jaccard_similarity("hello", "") == 0.0
    assert jaccard_similarity("apple banana", "apple banana") == 1.0
    assert 0.0 < jaccard_similarity("apple orange", "apple banana") < 1.0

    # Sequence similarity
    assert sequence_similarity("", "") == 1.0
    assert sequence_similarity("abc", "abc") == 1.0
    assert sequence_similarity("abc", "xyz") == 0.0


def test_similarity_evaluator_empty_outputs() -> None:
    ev = SimilarityEvaluator()
    tc = TestCase(id="tc1", vars={})
    r_empty = RunResult(
        prompt_name="v1",
        test_case_id="tc1",
        rendered_prompt="p",
        output="",
        latency_ms=10.0,
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        cost_usd=0.0,
        model="mock",
    )
    score = ev.evaluate(r_empty, r_empty, tc)
    assert score.passed is True
    assert score.v2_score == 1.0
    assert "Empty outputs" in score.message


def test_similarity_evaluator_fallback_path(monkeypatch: pytest.MonkeyPatch) -> None:
    import promptdiff.evaluators.similarity as sim_mod

    # Force embedding model to be None so it always exercises the difflib/jaccard fallback
    monkeypatch.setattr(sim_mod, "_get_embedding_model", lambda *args, **kwargs: None)

    ev = SimilarityEvaluator(threshold=0.50)
    tc = TestCase(id="tc2", vars={})
    r1 = RunResult(
        prompt_name="v1",
        test_case_id="tc2",
        rendered_prompt="p",
        output="The quick brown fox jumps over the lazy dog.",
        latency_ms=10.0,
        prompt_tokens=5,
        completion_tokens=5,
        total_tokens=10,
        cost_usd=0.0,
        model="mock",
    )
    r2 = RunResult(
        prompt_name="v2",
        test_case_id="tc2",
        rendered_prompt="p",
        output="A quick brown dog jumps over the lazy fox.",
        latency_ms=10.0,
        prompt_tokens=5,
        completion_tokens=5,
        total_tokens=10,
        cost_usd=0.0,
        model="mock",
    )
    score = ev.evaluate(r1, r2, tc)
    assert score.name == "similarity"
    assert score.passed is True
    assert score.details["fallback"] is True
    assert "composite_score" in score.details


def test_emit_fallback_warning() -> None:
    # Multiple calls should not crash or duplicate warning
    _emit_fallback_warning()
    _emit_fallback_warning()
