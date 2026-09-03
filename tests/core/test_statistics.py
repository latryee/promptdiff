"""Tests migrated to feature-focused test suite."""

from __future__ import annotations

from promptdiff.core.clustering import DatasetCentroidCompressor
from promptdiff.core.hypothesis_testing import compute_paired_wilcoxon
from promptdiff.core.statistics import (
    analyze_significance,
    bootstrap_ci,
    permutation_test_p_value,
)


def test_statistics_bootstrap_and_p_value() -> None:
    """Test statistical significance bootstrap and permutation test."""
    v1_lats = [200.0, 210.0, 195.0, 205.0, 202.0, 215.0, 198.0, 204.0]
    v2_lats = [150.0, 155.0, 148.0, 152.0, 150.0, 153.0, 149.0, 151.0]

    ci_low, ci_high = bootstrap_ci([v2 - v1 for v1, v2 in zip(v1_lats, v2_lats, strict=False)])
    assert ci_low < 0.0
    assert ci_high < 0.0

    p_val = permutation_test_p_value(v1_lats, v2_lats)
    assert p_val < 0.05

    sig = analyze_significance("latency_ms", v1_lats, v2_lats)
    assert sig is not None
    assert sig.is_statistically_significant is True
    assert "Statistically Significant" in sig.verdict_text


def test_hypothesis_testing_wilcoxon() -> None:
    """Test non-parametric Paired Wilcoxon Signed-Rank Test and Bootstrap CI."""
    # Significant improvement
    s1 = [0.60, 0.62, 0.65, 0.61, 0.64, 0.63, 0.62, 0.65, 0.63, 0.61, 0.64, 0.62]
    s2 = [0.85, 0.88, 0.90, 0.87, 0.89, 0.86, 0.88, 0.91, 0.87, 0.89, 0.88, 0.86]
    rep = compute_paired_wilcoxon(s1, s2, alpha=0.05)
    assert rep.is_significant is True
    assert rep.p_value < 0.05
    assert rep.delta_mean > 0.20
    assert len(rep.confidence_interval_95) == 2

    # Identical scores
    rep_same = compute_paired_wilcoxon([0.5, 0.5], [0.5, 0.5])
    assert rep_same.p_value == 1.0
    assert rep_same.is_significant is False


def test_centroid_clustering() -> None:
    """Test in-memory semantic clustering of dataset queries."""
    compressor = DatasetCentroidCompressor(target_clusters=2)
    queries = [
        "How do I reset my password?",
        "Password reset error",
        "Where is my invoice?",
        "Billing invoice payment receipt",
    ]
    res = compressor.compress(queries)
    assert res.condensed_sample_count == 2
    assert res.compression_ratio_pct == 50.0
    assert len(res.condensed_test_cases) == 2
