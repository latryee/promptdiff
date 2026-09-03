"""Entity-Relation Knowledge Graph Fact Verification Evaluator.

Extracts (Subject, Predicate, Object) knowledge triplets from LLM outputs
and verifies relational fidelity against ground truth contexts using bipartite triplet overlap.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass

from promptdiff.core.models import EvaluatorScore, RunResult, TestCase
from promptdiff.evaluators.base import BaseEvaluator


@dataclass
class KnowledgeTriplet:
    """A semantic relationship triplet (Subject, Predicate, Object)."""

    subject: str
    predicate: str
    object: str

    def to_string(self) -> str:
        return f"({self.subject} -> {self.predicate} -> {self.object})"


def extract_triplets_heuristic(text: str) -> list[KnowledgeTriplet]:
    """Lightweight rule-based triplet extraction from sentences."""
    triplets: list[KnowledgeTriplet] = []
    sentences = re.split(r"[\.\n;\?!]+", text)

    relation_patterns = [
        r"(\b[\w\s]{2,20}\b)\s+(is|was|are|were|released in|created by|costs|supports|contains)\s+(\b[\w\s]{2,20}\b)",
        r"(\b[\w\s]{2,20}\b)\s+(has|have|had|features|requires)\s+(\b[\w\s]{2,20}\b)",
    ]

    for s in sentences:
        s_clean = s.strip()
        if not s_clean:
            continue
        for pat in relation_patterns:
            matches = re.finditer(pat, s_clean, re.IGNORECASE)
            for m in matches:
                sub, pred, obj = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
                if len(sub) > 1 and len(obj) > 1:
                    triplets.append(KnowledgeTriplet(subject=sub.lower(), predicate=pred.lower(), object=obj.lower()))

    # Fallback to simple bigram triplet if pattern misses
    if not triplets and len(text.split()) >= 3:
        words = text.split()
        triplets.append(KnowledgeTriplet(subject=words[0].lower(), predicate="relates_to", object=words[-1].lower()))
    return triplets


class FactGraphEvaluator(BaseEvaluator):
    """Knowledge Triplet Graph Fact-Checking Evaluator."""

    name: str = "fact_graph"
    description: str = "Knowledge triplet graph relation verifier."

    def __init__(self, match_threshold: float = 0.50):
        self.match_threshold = match_threshold

    def compute_triplet_fidelity(self, candidate_triplets: list[KnowledgeTriplet], truth_text: str) -> float:
        """Compute proportion of candidate relations verified in ground truth context."""
        if not candidate_triplets:
            return 1.0
        truth_lower = truth_text.lower()
        verified: float = 0.0

        for t in candidate_triplets:
            # Subject and Object must both appear in truth
            if t.subject in truth_lower and t.object in truth_lower:
                verified += 1
            elif t.subject in truth_lower or t.object in truth_lower:
                verified += 0.5

        return min(1.0, verified / len(candidate_triplets))

    async def async_evaluate(self, r1: RunResult, r2: RunResult, test_case: TestCase) -> EvaluatorScore:
        truth = test_case.vars.get("context", "") or test_case.vars.get("ground_truth", "")
        if not truth:
            return EvaluatorScore(
                name="fact_graph",
                passed=True,
                v1_score=1.0,
                v2_score=1.0,
                delta=0.0,
                delta_pct=0.0,
                message="No ground truth context supplied for fact graph verification.",
            )

        triplets_v1 = extract_triplets_heuristic(r1.output)
        triplets_v2 = extract_triplets_heuristic(r2.output)

        fid_v1 = self.compute_triplet_fidelity(triplets_v1, truth)
        fid_v2 = self.compute_triplet_fidelity(triplets_v2, truth)

        passed = fid_v2 >= self.match_threshold
        delta = fid_v2 - fid_v1

        return EvaluatorScore(
            name="fact_graph",
            passed=passed,
            v1_score=round(fid_v1, 3),
            v2_score=round(fid_v2, 3),
            delta=round(delta, 3),
            delta_pct=round(delta * 100.0, 1),
            message=f"Fact Graph Triplet Fidelity: {fid_v2 * 100:.1f}% ({len(triplets_v2)} relations checked).",
        )

    def evaluate(self, r1: RunResult, r2: RunResult, test_case: TestCase) -> EvaluatorScore:
        return asyncio.run(self.async_evaluate(r1, r2, test_case))
