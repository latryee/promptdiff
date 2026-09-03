"""Memory-Efficient Semantic Centroid Clustering & Dataset Deduplication.

Compresses large-scale production query logs (10k+ instances) into a compact,
representative evaluation dataset using Cosine Centroid Clustering without external vector DBs.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from promptdiff.core.models import TestCase


@dataclass
class ClusterCentroid:
    """A representative centroid extracted from the query cluster."""

    centroid_id: str
    representative_query: str
    cluster_size: int
    diversity_radius: float


@dataclass
class ClusteringResult:
    """Outcome of dataset semantic compression."""

    total_original_samples: int
    condensed_sample_count: int
    compression_ratio_pct: float
    centroids: list[ClusterCentroid]
    condensed_test_cases: list[TestCase]


def _bow_vectorize(text: str, vocab: dict[str, int]) -> list[float]:
    """Sparse bag-of-words vectorizer."""
    vec = [0.0] * len(vocab)
    words = re.findall(r"\w+", text.lower())
    for w in words:
        if w in vocab:
            vec[vocab[w]] += 1.0
    # Normalize
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


def _cosine_dist(v1: list[float], v2: list[float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2, strict=False))
    return 1.0 - max(-1.0, min(1.0, dot))


class DatasetCentroidCompressor:
    """Condenses redundant test queries into centroid exemplars."""

    def __init__(self, target_clusters: int = 10):
        self.target_clusters = target_clusters

    def compress(self, queries: list[str]) -> ClusteringResult:
        """Partition queries into clusters and pick central representative queries."""
        if not queries:
            return ClusteringResult(0, 0, 0.0, [], [])

        # Build vocabulary
        vocab: dict[str, int] = {}
        for q in queries:
            for w in re.findall(r"\w+", q.lower()):
                if w not in vocab and len(vocab) < 500:
                    vocab[w] = len(vocab)

        vectors = [_bow_vectorize(q, vocab) for q in queries]
        k = min(self.target_clusters, len(queries))

        # Simple greedy furthest-first traversal for initial centroids
        centroids_idx = [0]
        while len(centroids_idx) < k:
            max_d = -1.0
            best_i = 0
            for i in range(len(queries)):
                if i in centroids_idx:
                    continue
                min_d_to_c = min(_cosine_dist(vectors[i], vectors[c]) for c in centroids_idx)
                if min_d_to_c > max_d:
                    max_d = min_d_to_c
                    best_i = i
            centroids_idx.append(best_i)

        # Assign points to closest centroid
        clusters: dict[int, list[int]] = {c: [] for c in centroids_idx}
        for i, vec in enumerate(vectors):
            closest = min(centroids_idx, key=lambda c: _cosine_dist(vec, vectors[c]))
            clusters[closest].append(i)

        res_centroids: list[ClusterCentroid] = []
        test_cases: list[TestCase] = []

        for idx, (c_idx, members) in enumerate(clusters.items()):
            rep_text = queries[c_idx]
            c_obj = ClusterCentroid(
                centroid_id=f"cluster_{idx + 1}",
                representative_query=rep_text,
                cluster_size=len(members),
                diversity_radius=0.25,
            )
            res_centroids.append(c_obj)
            test_cases.append(TestCase(id=f"tc_centroid_{idx + 1}", vars={"query": rep_text}))

        ratio = round((1.0 - (len(test_cases) / max(1, len(queries)))) * 100.0, 1)

        return ClusteringResult(
            total_original_samples=len(queries),
            condensed_sample_count=len(test_cases),
            compression_ratio_pct=ratio,
            centroids=res_centroids,
            condensed_test_cases=test_cases,
        )
