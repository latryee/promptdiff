#!/usr/bin/env python3
"""LLM Judge Human Correlation Calibration Utility (scripts/calibrate_judge.py).

Evaluates LLM-as-a-Judge scoring against human ground-truth labels, measuring
Pearson correlation (r), Spearman rank correlation (rho), Mean Absolute Error (MAE),
and Root Mean Squared Error (RMSE) to validate judge alignment before production deployment.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from promptdiff.core.models import RunResult, TestCase
from promptdiff.evaluators.llm_judge import LLMJudgeEvaluator


@dataclass
class CalibrationMetrics:
    """Statistical alignment metrics between LLM Judge and Human Annotators."""

    sample_count: int
    pearson_r: float
    spearman_rho: float
    mae: float
    rmse: float
    alignment_verdict: str


def compute_ranks(values: Sequence[float]) -> list[float]:
    """Compute fractional ranks for tied values."""
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j < len(indexed) - 1 and indexed[j + 1][1] == indexed[j][1]:
            j += 1
        avg_rank = (i + j + 2) / 2.0  # 1-indexed rank average
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def compute_pearson_r(x: Sequence[float], y: Sequence[float]) -> float:
    """Calculate Pearson correlation coefficient."""
    n = len(x)
    if n < 2:
        return 0.0
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y, strict=True))
    den_x = sum((xi - mean_x) ** 2 for xi in x)
    den_y = sum((yi - mean_y) ** 2 for yi in y)
    denom = math.sqrt(den_x * den_y)
    return round(num / denom, 4) if denom > 1e-9 else 0.0


def compute_spearman_rho(x: Sequence[float], y: Sequence[float]) -> float:
    """Calculate Spearman rank correlation coefficient."""
    if len(x) < 2:
        return 0.0
    ranks_x = compute_ranks(x)
    ranks_y = compute_ranks(y)
    return compute_pearson_r(ranks_x, ranks_y)


def compute_calibration_metrics(judge_scores: list[float], human_scores: list[float]) -> CalibrationMetrics:
    """Compute full suite of calibration and alignment metrics."""
    n = len(judge_scores)
    if n == 0:
        raise ValueError("Cannot compute calibration on empty score lists.")

    r = compute_pearson_r(judge_scores, human_scores)
    rho = compute_spearman_rho(judge_scores, human_scores)
    mae = round(sum(abs(j - h) for j, h in zip(judge_scores, human_scores, strict=True)) / n, 3)
    rmse = round(math.sqrt(sum((j - h) ** 2 for j, h in zip(judge_scores, human_scores, strict=True)) / n), 3)

    if r >= 0.70 and rho >= 0.70:
        verdict = "STRONG ALIGNMENT (Production Ready)"
    elif r >= 0.50:
        verdict = "MODERATE ALIGNMENT (Rubric refinement recommended)"
    else:
        verdict = "POOR ALIGNMENT (Judge rubric requires significant overhaul)"

    return CalibrationMetrics(
        sample_count=n,
        pearson_r=r,
        spearman_rho=rho,
        mae=mae,
        rmse=rmse,
        alignment_verdict=verdict,
    )


async def run_calibration(
    dataset_path: str,
    model_name: str = "gpt-4o",
    force_mock: bool = False,
) -> CalibrationMetrics:
    """Evaluate LLM judge on human-labeled dataset."""
    path = Path(dataset_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    # Parse JSON or JSONL dataset
    raw = path.read_text(encoding="utf-8").strip()
    records = []
    if raw.startswith("["):
        records = json.loads(raw)
    else:
        for line in raw.split("\n"):
            if line.strip():
                records.append(json.loads(line))

    evaluator = LLMJudgeEvaluator(model_name=model_name, force_mock=force_mock)
    judge_scores: list[float] = []
    human_scores: list[float] = []

    for idx, rec in enumerate(records):
        tc = TestCase(id=f"calib_{idx}", vars=rec.get("vars", {}))
        prompt_text = rec.get("prompt", "Analyze input")
        output_text = rec.get("output", "")
        human_score = float(rec.get("human_score", 3.0))

        run_result = RunResult(
            prompt_name="eval_candidate",
            test_case_id=tc.id,
            rendered_prompt=prompt_text,
            output=output_text,
            latency_ms=100.0,
            prompt_tokens=20,
            completion_tokens=20,
            total_tokens=40,
            cost_usd=0.0001,
            model=model_name,
        )

        score = await evaluator.async_evaluate(
            test_case=tc,
            v1_result=run_result,
            v2_result=run_result,
        )
        judge_scores.append(float(score.v2_score))
        human_scores.append(human_score)

    return compute_calibration_metrics(judge_scores, human_scores)


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate LLM Judge alignment against human evaluation scores.")
    parser.add_argument("--dataset", "-d", required=True, help="Path to JSON or JSONL file with human ratings")
    parser.add_argument("--model", "-m", default="gpt-4o", help="Judge model name")
    parser.add_argument("--mock", action="store_true", help="Use deterministic mock judge")
    parser.add_argument("--output", "-o", help="Optional path to export JSON calibration report")

    args = parser.parse_args()

    try:
        metrics = asyncio.run(run_calibration(args.dataset, model_name=args.model, force_mock=args.mock))
    except Exception as e:
        print(f"Error during calibration: {e}", file=sys.stderr)
        return 1

    print("\n" + "=" * 55)
    print(" ⚖️  LLM Judge Calibration & Human Alignment Report")
    print("=" * 55)
    print(f" Samples Evaluated    : {metrics.sample_count}")
    print(f" Pearson Correlation (r)  : {metrics.pearson_r:+.4f}")
    print(f" Spearman Rank (rho)     : {metrics.spearman_rho:+.4f}")
    print(f" Mean Absolute Error (MAE): {metrics.mae:.3f}")
    print(f" Root Mean Sq Error (RMSE): {metrics.rmse:.3f}")
    print(f" Alignment Verdict    : {metrics.alignment_verdict}")
    print("=" * 55 + "\n")

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(metrics.__dict__, indent=2), encoding="utf-8")
        print(f"Calibration metrics saved to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
