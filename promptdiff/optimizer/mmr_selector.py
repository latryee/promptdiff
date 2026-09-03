"""Dynamic Few-Shot Exemplar Selector via Maximal Marginal Relevance (MMR).

Selects optimal few-shot prompt demonstration examples from an exemplar pool by
maximizing semantic query similarity while penalizing inter-exemplar redundancy:
MMR = argmax_{d_i} [ lambda * Sim(d_i, query) - (1 - lambda) * max_{d_j} Sim(d_i, d_j) ]
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class Exemplar:
    """A prompt demonstration candidate."""

    id: str
    input_text: str
    output_text: str
    metadata: dict[str, Any] = None  # type: ignore


@dataclass
class MMRSelectionResult:
    """Outcome of MMR dynamic few-shot selection."""

    selected_exemplars: list[Exemplar]
    diversity_penalty_lambda: float
    total_pool_size: int


def _compute_jaccard_similarity(text1: str, text2: str) -> float:
    """Compute token Jaccard similarity between two texts."""
    w1 = set(re.findall(r"\w+", text1.lower()))
    w2 = set(re.findall(r"\w+", text2.lower()))
    if not w1 or not w2:
        return 0.0
    return len(w1.intersection(w2)) / len(w1.union(w2))


class MMRExemplarSelector:
    """Selects diverse, highly relevant few-shot exemplars."""

    def __init__(self, diversity_lambda: float = 0.65):
        self.diversity_lambda = diversity_lambda

    def select(self, query: str, pool: list[Exemplar], top_k: int = 3) -> MMRSelectionResult:
        """Select top_k diverse exemplars for the given query."""
        if not pool:
            return MMRSelectionResult(
                selected_exemplars=[], diversity_penalty_lambda=self.diversity_lambda, total_pool_size=0
            )

        selected: list[Exemplar] = []
        remaining = list(pool)

        while len(selected) < min(top_k, len(pool)) and remaining:
            best_score = -float("inf")
            best_candidate = remaining[0]

            for cand in remaining:
                # Similarity to user query
                sim_to_query = _compute_jaccard_similarity(cand.input_text, query)

                # Max similarity to already selected exemplars
                if selected:
                    max_sim_selected = max(_compute_jaccard_similarity(cand.input_text, s.input_text) for s in selected)
                else:
                    max_sim_selected = 0.0

                mmr_score = (self.diversity_lambda * sim_to_query) - ((1.0 - self.diversity_lambda) * max_sim_selected)

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_candidate = cand

            selected.append(best_candidate)
            remaining.remove(best_candidate)

        return MMRSelectionResult(
            selected_exemplars=selected,
            diversity_penalty_lambda=self.diversity_lambda,
            total_pool_size=len(pool),
        )
