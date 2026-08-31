"""Statistical Significance & Confidence Interval Engine for promptdiff.

Provides Bootstrap Resampling and Non-Parametric Hypothesis Testing (Wilcoxon/Permutation)
to distinguish genuine prompt performance improvements from stochastic LLM sampling noise.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SignificanceResult:
    """Statistical significance analysis result."""

    metric_name: str
    v1_mean: float
    v2_mean: float
    delta_mean: float
    delta_pct: float
    p_value: float
    is_statistically_significant: bool  # True if p_value < alpha (0.05)
    ci_95_lower: float
    ci_95_upper: float
    verdict_text: str


def bootstrap_ci(
    deltas: list[float],
    num_samples: int = 1000,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    """Calculate non-parametric bootstrap confidence interval for mean delta."""
    if not deltas:
        return 0.0, 0.0
    if len(deltas) == 1:
        return deltas[0], deltas[0]

    n = len(deltas)
    bootstrap_means = []

    for _ in range(num_samples):
        sample = [random.choice(deltas) for _ in range(n)]
        bootstrap_means.append(sum(sample) / n)

    bootstrap_means.sort()
    alpha = (1.0 - confidence_level) / 2.0
    lower_idx = int(alpha * num_samples)
    upper_idx = int((1.0 - alpha) * num_samples) - 1

    lower_idx = max(0, min(num_samples - 1, lower_idx))
    upper_idx = max(0, min(num_samples - 1, upper_idx))

    return round(bootstrap_means[lower_idx], 4), round(bootstrap_means[upper_idx], 4)


def permutation_test_p_value(
    v1_values: list[float],
    v2_values: list[float],
    num_permutations: int = 1000,
) -> float:
    """Calculate two-sided empirical permutation test p-value for paired differences."""
    if not v1_values or not v2_values or len(v1_values) != len(v2_values):
        return 1.0

    n = len(v1_values)
    actual_diffs = [v2_values[i] - v1_values[i] for i in range(n)]
    actual_mean_diff = abs(sum(actual_diffs) / n)

    if actual_mean_diff == 0.0:
        return 1.0

    count_greater = 0
    for _ in range(num_permutations):
        # Random sign flip for paired differences under null hypothesis
        permuted_diffs = [diff if random.random() > 0.5 else -diff for diff in actual_diffs]
        permuted_mean = abs(sum(permuted_diffs) / n)
        if permuted_mean >= actual_mean_diff:
            count_greater += 1

    return round(max(0.001, count_greater / num_permutations), 4)


def analyze_significance(
    metric_name: str,
    v1_values: list[float],
    v2_values: list[float],
    alpha: float = 0.05,
) -> Optional[SignificanceResult]:
    """Perform full significance and bootstrap analysis on paired evaluation arrays."""
    if not v1_values or not v2_values or len(v1_values) != len(v2_values):
        return None

    n = len(v1_values)
    v1_mean = sum(v1_values) / n
    v2_mean = sum(v2_values) / n
    delta = v2_mean - v1_mean
    delta_pct = (delta / v1_mean * 100.0) if v1_mean != 0 else 0.0

    deltas = [v2_values[i] - v1_values[i] for i in range(n)]
    ci_lower, ci_upper = bootstrap_ci(deltas)
    p_val = permutation_test_p_value(v1_values, v2_values)
    is_sig = p_val < alpha

    if is_sig:
        direction = "Improvement" if delta > 0 else "Regression"
        verdict = f"Statistically Significant {direction} (p={p_val:.3f} < {alpha}, 95% CI [{ci_lower:+.3f}, {ci_upper:+.3f}])"
    else:
        verdict = f"Not Statistically Significant / Sampling Noise (p={p_val:.3f} >= {alpha}, 95% CI [{ci_lower:+.3f}, {ci_upper:+.3f}])"

    return SignificanceResult(
        metric_name=metric_name,
        v1_mean=round(v1_mean, 4),
        v2_mean=round(v2_mean, 4),
        delta_mean=round(delta, 4),
        delta_pct=round(delta_pct, 2),
        p_value=p_val,
        is_statistically_significant=is_sig,
        ci_95_lower=ci_lower,
        ci_95_upper=ci_upper,
        verdict_text=verdict,
    )
