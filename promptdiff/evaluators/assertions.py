"""CI/CD Regression Assertion Engine for promptdiff.

Parses assertions like 'cost_delta <= 10%, latency_delta <= 15%, json_validity >= 1.0'
and computes pass/fail regression verdicts with exit codes.
"""

from __future__ import annotations

import re
from typing import Literal

from promptdiff.core.models import (
    AssertionRule,
    ComparisonResult,
    RegressionVerdict,
)


def parse_assertion_string(expr: str) -> AssertionRule | None:
    """Parse string expression into structured AssertionRule.

    Examples:
        - "cost_delta <= 10%" -> metric="cost_delta", op="<=", val=10.0, is_pct=True
        - "latency_delta <= 50ms" -> metric="latency_delta", op="<=", val=50.0, is_pct=False
        - "json_validity == 1.0" -> metric="json_validity", op="==", val=1.0, is_pct=False
        - "similarity >= 0.85" -> metric="similarity", op=">=", val=0.85, is_pct=False
    """
    clean = expr.strip()
    match = re.match(
        r"^([a-zA-Z0-9_\-\.]+)\s*(<=|>=|<|>|==|!=)\s*([+\-]?[0-9]+(?:\.[0-9]+)?)\s*(%|ms|s)?$",
        clean,
    )
    if not match:
        return None

    metric = match.group(1).lower()
    op = match.group(2)
    val = float(match.group(3))
    unit = match.group(4) or ""
    is_pct = "%" in unit or "pct" in metric or "delta_pct" in metric

    return AssertionRule(
        metric=metric,
        operator=op,  # type: ignore
        threshold=val,
        is_pct=is_pct,
        raw_expression=clean,
    )


def parse_assertion_list(assertion_inputs: list[str]) -> list[AssertionRule]:
    """Parse comma-separated or list of assertion strings."""
    rules: list[AssertionRule] = []
    for item in assertion_inputs:
        for sub_expr in item.split(","):
            sub_clean = sub_expr.strip()
            if sub_clean:
                rule = parse_assertion_string(sub_clean)
                if rule:
                    rules.append(rule)
    return rules


def evaluate_assertions(
    comparisons: list[ComparisonResult],
    assertion_rules: list[AssertionRule],
) -> RegressionVerdict:
    """Evaluate all comparison results against assertion rules and build verdict."""
    failed: list[str] = []

    # Calculate aggregate totals
    total_cost_v1 = sum(c.v1_result.cost_usd for c in comparisons)
    total_cost_v2 = sum(c.v2_result.cost_usd for c in comparisons)
    cost_delta_pct = (
        ((total_cost_v2 - total_cost_v1) / total_cost_v1 * 100.0)
        if total_cost_v1 > 0
        else 0.0
    )

    avg_lat_v1 = (
        sum(c.v1_result.latency_ms for c in comparisons) / len(comparisons)
        if comparisons
        else 0.0
    )
    avg_lat_v2 = (
        sum(c.v2_result.latency_ms for c in comparisons) / len(comparisons)
        if comparisons
        else 0.0
    )
    lat_delta_pct = (
        ((avg_lat_v2 - avg_lat_v1) / avg_lat_v1 * 100.0) if avg_lat_v1 > 0 else 0.0
    )

    # Check aggregate and case-by-case rules
    for rule in assertion_rules:
        metric = rule.metric

        # 1. Cost Delta Check
        if metric in ["cost_delta", "cost_delta_pct", "cost"]:
            val = cost_delta_pct if rule.is_pct else (total_cost_v2 - total_cost_v1)
            if not rule.evaluate(val):
                failed.append(
                    f"Regression: Cost delta ({val:+.2f}{'%' if rule.is_pct else '$'}) "
                    f"violated assertion '{rule.raw_expression}'"
                )

        # 2. Latency Delta Check
        elif metric in ["latency_delta", "latency_delta_pct", "latency"]:
            val = lat_delta_pct if rule.is_pct else (avg_lat_v2 - avg_lat_v1)
            if not rule.evaluate(val):
                failed.append(
                    f"Regression: Latency delta ({val:+.2f}{'%' if rule.is_pct else 'ms'}) "
                    f"violated assertion '{rule.raw_expression}'"
                )

        # 3. Per-case metric evaluations (json_validity, similarity, etc.)
        else:
            for comp in comparisons:
                score_obj = comp.scores.get(metric)
                if score_obj:
                    val = float(score_obj.v2_score)
                    if not rule.evaluate(val):
                        failed.append(
                            f"Assertion '{rule.raw_expression}' failed on test case '{comp.test_case.id}' "
                            f"(actual {metric}={val:.2f}, v1={score_obj.v1_score})"
                        )

    passed = len(failed) == 0
    verdict_status: Literal["PASSED", "REGRESSION_DETECTED", "ERROR"] = "PASSED" if passed else "REGRESSION_DETECTED"

    return RegressionVerdict(
        passed=passed,
        status=verdict_status,
        failed_assertions=failed,
        total_cost_v1=round(total_cost_v1, 6),
        total_cost_v2=round(total_cost_v2, 6),
        cost_delta_pct=round(cost_delta_pct, 2),
        avg_latency_v1=round(avg_lat_v1, 2),
        avg_latency_v2=round(avg_lat_v2, 2),
        latency_delta_pct=round(lat_delta_pct, 2),
    )
