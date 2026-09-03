"""Semantic & Embedding Cosine Similarity Evaluator.

Measures output semantic preservation vs drift using local sentence-transformers (e.g. all-MiniLM-L6-v2)
for zero-cost, high-speed semantic cosine similarity, with graceful token-similarity fallback.
"""

from __future__ import annotations

import difflib
import logging
import re
from typing import Any

import numpy as np

from promptdiff.core.models import EvaluatorScore, RunResult, TestCase
from promptdiff.evaluators.base import BaseEvaluator

logger = logging.getLogger("promptdiff.evaluators.similarity")

# Global singleton cache for local embedding model
_EMBEDDING_MODEL: Any | None = None
_EMBEDDING_MODEL_LOADED: bool = False
_FALLBACK_WARNING_EMITTED: bool = False


def _emit_fallback_warning() -> None:
    """Emit a single visible CLI warning when falling back to difflib."""
    global _FALLBACK_WARNING_EMITTED
    if not _FALLBACK_WARNING_EMITTED:
        _FALLBACK_WARNING_EMITTED = True
        try:
            from rich.console import Console

            Console(stderr=True).print(
                "[bold yellow]⚠️  [SimilarityEvaluator] sentence-transformers not installed. "
                "Falling back to textual token-overlap (difflib). "
                "Install promptdiff[semantic] for dense neural embeddings.[/bold yellow]"
            )
        except Exception:
            logger.warning(
                "[SimilarityEvaluator] sentence-transformers not installed. "
                "Falling back to textual token-overlap (difflib). Install promptdiff[semantic]."
            )


def _get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> Any | None:
    """Lazy-load sentence-transformers model instance."""
    global _EMBEDDING_MODEL, _EMBEDDING_MODEL_LOADED
    if _EMBEDDING_MODEL_LOADED:
        return _EMBEDDING_MODEL

    try:
        import os
        import warnings

        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from sentence_transformers import SentenceTransformer

            _EMBEDDING_MODEL = SentenceTransformer(model_name)
        _EMBEDDING_MODEL_LOADED = True
        return _EMBEDDING_MODEL
    except Exception as e:
        logger.debug(f"sentence-transformers not available or failed to load: {e}")
        _EMBEDDING_MODEL = None
        _EMBEDDING_MODEL_LOADED = True
        return None


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two numpy vectors."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def tokenize(text: str) -> set[str]:
    """Tokenize string into lowercase alphanumeric words."""
    return set(re.findall(r"\w+", text.lower()))


def jaccard_similarity(s1: str, s2: str) -> float:
    """Calculate token Jaccard similarity between two strings."""
    tokens1 = tokenize(s1)
    tokens2 = tokenize(s2)
    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0
    intersection = len(tokens1.intersection(tokens2))
    union = len(tokens1.union(tokens2))
    return intersection / union if union > 0 else 0.0


def sequence_similarity(s1: str, s2: str) -> float:
    """Calculate Levenshtein-like sequence similarity ratio."""
    if not s1 and not s2:
        return 1.0
    matcher = difflib.SequenceMatcher(None, s1, s2)
    return matcher.ratio()


class SimilarityEvaluator(BaseEvaluator):
    """Measures semantic and textual preservation between prompt versions."""

    name: str = "similarity"
    description: str = (
        "Measures semantic cosine similarity (1.0 = Identical, 0.0 = Dissimilar) using sentence-transformers"
    )

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", threshold: float = 0.50):
        self.model_name = model_name
        self.threshold = threshold

    def evaluate(
        self,
        v1_result: RunResult,
        v2_result: RunResult,
        test_case: TestCase,
    ) -> EvaluatorScore:
        out1 = v1_result.output
        out2 = v2_result.output

        if not out1 and not out2:
            return EvaluatorScore(
                name=self.name,
                v1_score=1.0,
                v2_score=1.0,
                delta=0.0,
                delta_pct=0.0,
                passed=True,
                message="100.0% Match (Empty outputs)",
                details={"method": "exact", "similarity": 1.0},
            )

        model = _get_embedding_model(self.model_name)

        if model is not None:
            try:
                embeddings = model.encode([out1, out2])
                sim = cosine_similarity(embeddings[0], embeddings[1])
                method = f"sentence-transformers ({self.model_name})"
                passed = sim >= self.threshold
                delta = sim - 1.0
                message = f"{sim * 100:.1f}% Semantic Cosine Sim ({method})"

                return EvaluatorScore(
                    name=self.name,
                    v1_score=1.0,
                    v2_score=round(sim, 3),
                    delta=round(delta, 3),
                    delta_pct=round(delta * 100, 1),
                    passed=passed,
                    message=message,
                    details={
                        "method": method,
                        "cosine_similarity": round(sim, 4),
                        "model": self.model_name,
                    },
                )
            except Exception as e:
                logger.debug(f"Embedding computation failed, falling back to difflib: {e}")

        # Fallback to composite sequence & jaccard similarity
        _emit_fallback_warning()
        seq_sim = sequence_similarity(out1, out2)
        jaccard_sim = jaccard_similarity(out1, out2)
        composite = 0.6 * seq_sim + 0.4 * jaccard_sim

        delta = composite - 1.0
        passed = composite >= self.threshold
        message = f"{composite * 100:.1f}% Textual Match [FALLBACK: difflib token-overlap] (Seq: {seq_sim * 100:.1f}%, Jaccard: {jaccard_sim * 100:.1f}%)"

        return EvaluatorScore(
            name=self.name,
            v1_score=1.0,
            v2_score=round(composite, 3),
            delta=round(delta, 3),
            delta_pct=round(delta * 100, 1),
            passed=passed,
            message=message,
            details={
                "method": "textual_difflib_fallback",
                "fallback": True,
                "fallback_warning": "sentence-transformers not installed; using difflib token overlap. Install promptdiff[semantic] for neural embeddings.",
                "sequence_similarity": round(seq_sim, 4),
                "jaccard_similarity": round(jaccard_sim, 4),
                "composite_score": round(composite, 4),
            },
        )
