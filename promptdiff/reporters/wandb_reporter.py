"""Weights & Biases (W&B) Integration Reporter for Model Evaluation & Tracking."""

from __future__ import annotations

import logging
from typing import Any, Optional

from promptdiff.core.models import DiffReport

logger = logging.getLogger("promptdiff.reporters.wandb")


def log_to_wandb(
    report: DiffReport,
    project: str = "promptdiff",
    entity: Optional[str] = None,
    run_name: Optional[str] = None,
) -> bool:
    """Log full regression evaluation telemetry, comparison tables, and metrics to Weights & Biases.

    Args:
        report: Serialized DiffReport from runner.
        project: W&B project name.
        entity: Optional W&B username or organization entity.
        run_name: Optional custom run name.

    Returns:
        True if logged successfully, False otherwise.
    """
    try:
        import wandb
    except ImportError:
        logger.warning(
            "Weights & Biases is not installed. To enable W&B logging, install with `pip install wandb`."
        )
        return False

    try:
        r_name = run_name or f"{report.v1_name}-vs-{report.v2_name}-{report.run_id}"

        wandb.init(
            project=project,
            entity=entity,
            name=r_name,
            config={
                "run_id": report.run_id,
                "v1_name": report.v1_name,
                "v2_name": report.v2_name,
                "model_v1": report.model_v1,
                "model_v2": report.model_v2,
                "total_cases": report.total_cases,
                "evaluators": report.evaluators,
            },
            reinit=True,
        )

        v = report.verdict
        # 1. Summary Metrics
        summary_metrics: dict[str, Any] = {
            "total_cost_v1": v.total_cost_v1,
            "total_cost_v2": v.total_cost_v2,
            "cost_delta_pct": v.cost_delta_pct,
            "avg_latency_v1_ms": v.avg_latency_v1,
            "avg_latency_v2_ms": v.avg_latency_v2,
            "latency_delta_pct": v.latency_delta_pct,
            "passed": v.passed,
            "verdict": v.status,
        }

        # Evaluator scores
        for ev_name in report.evaluators:
            scores = []
            for comp in report.comparisons:
                if ev_name in comp.scores:
                    val = comp.scores[ev_name].v2_score
                    if isinstance(val, (int, float)):
                        scores.append(float(val))
            if scores:
                summary_metrics[f"eval_{ev_name}_mean"] = round(sum(scores) / len(scores), 4)

        wandb.log(summary_metrics)

        # 2. Comparison Table
        table_columns = [
            "Test ID",
            "Description",
            "v1 Output",
            "v2 Output",
            "v1 Latency (ms)",
            "v2 Latency (ms)",
            "v1 Cost ($)",
            "v2 Cost ($)",
            "Scores",
            "Passed",
        ]
        table = wandb.Table(columns=table_columns)

        for comp in report.comparisons:
            scores_repr = ", ".join(
                f"{k}: {s.v2_score}" for k, s in comp.scores.items()
            )
            passed = all(s.passed for s in comp.scores.values())
            table.add_data(
                comp.test_case.id,
                comp.test_case.description,
                comp.v1_result.output[:300],
                comp.v2_result.output[:300],
                comp.v1_result.latency_ms,
                comp.v2_result.latency_ms,
                comp.v1_result.cost_usd,
                comp.v2_result.cost_usd,
                scores_repr,
                passed,
            )

        wandb.log({"eval_comparisons": table})
        wandb.finish()

        logger.info(f"Successfully logged evaluation to Weights & Biases project '{project}'")
        return True

    except Exception as e:
        logger.error(f"Failed to log run to Weights & Biases: {e}")
        return False
