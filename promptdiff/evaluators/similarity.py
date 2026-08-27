"""Semantic & Textual Similarity Evaluator.

Measures output preservation vs drift using SequenceMatcher and Token Jaccard similarity.
"""

from __future__ import annotations

import difflib
import re
from typing import Set
from promptdiff.core.models import EvaluatorScore, RunResult, TestCase
from promptdiff.evaluators.base import BaseEvaluator


def tokenize(text: str) -> Set[str]:
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
    """Measures textual and semantic preservation between prompt versions."""

    name: str = "similarity"
    description: str = "Measures output similarity ratio (1.0 = Identical, 0.0 = Completely Different)"

    def evaluate(
        self,
        v1_result: RunResult,
        v2_result: RunResult,
        test_case: TestCase,
    ) -> EvaluatorScore:
        out1 = v1_result.output
        out2 = v2_result.output

        seq_sim = sequence_similarity(out1, out2)
        jaccard_sim = jaccard_similarity(out1, out2)

        # Weighted composite score: 60% sequence, 40% jaccard
        composite = 0.6 * seq_sim + 0.4 * jaccard_sim

        delta = composite - 1.0  # Deviation from baseline (1.0)
        passed = composite >= 0.50

        message = f"{composite * 100:.1f}% Match (Seq: {seq_sim * 100:.1f}%, Jaccard: {jaccard_sim * 100:.1f}%)"

        return EvaluatorScore(
            name=self.name,
            v1_score=1.0,  # Baseline
            v2_score=round(composite, 3),
            delta=round(delta, 3),
            delta_pct=round(delta * 100, 1),
            passed=passed,
            message=message,
            details={
                "sequence_similarity": round(seq_sim, 4),
                "jaccard_similarity": round(jaccard_sim, 4),
                "composite_score": round(composite, 4),
            },
        )
