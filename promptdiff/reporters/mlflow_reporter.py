"""MLflow Integration Reporter for Experiment Tracking and Telemetry."""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

from promptdiff.core.models import DiffReport

logger = logging.getLogger("promptdiff.reporters.mlflow")


def log_to_mlflow(
    report: DiffReport,
    experiment_name: str = "promptdiff-evals",
    run_name: str | None = None,
    tracking_uri: str | None = None,
) -> bool:
    """Log full regression evaluation telemetry, metrics, parameters, and artifacts to MLflow.

    Args:
        report: Serialized DiffReport from runner.
        experiment_name: MLflow experiment name.
        run_name: Custom name for the run (defaults to report.run_id).
        tracking_uri: Optional MLflow server URI.

    Returns:
        True if logged successfully, False otherwise.
    """
    try:
        import mlflow
    except ImportError:
        logger.warning(
            "MLflow is not installed. To enable MLflow logging, install with `pip install mlflow`."
        )
        return False

    try:
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        mlflow.set_experiment(experiment_name)

        r_name = run_name or f"{report.v1_name}-vs-{report.v2_name}-{report.run_id}"

        with mlflow.start_run(run_name=r_name) as run:
            # 1. Log Parameters
            mlflow.log_params(
                {
                    "run_id": report.run_id,
                    "v1_name": report.v1_name,
                    "v2_name": report.v2_name,
                    "model_v1": report.model_v1,
                    "model_v2": report.model_v2,
                    "total_cases": report.total_cases,
                    "evaluators": ",".join(report.evaluators),
                    "verdict_status": report.verdict.status,
                }
            )

            # 2. Log Summary Metrics
            v = report.verdict
            metrics: dict[str, float] = {
                "total_cost_v1": v.total_cost_v1,
                "total_cost_v2": v.total_cost_v2,
                "cost_delta_pct": v.cost_delta_pct,
                "avg_latency_v1_ms": v.avg_latency_v1,
                "avg_latency_v2_ms": v.avg_latency_v2,
                "latency_delta_pct": v.latency_delta_pct,
                "passed_ratio": 1.0 if v.passed else 0.0,
                "failed_assertions_count": len(v.failed_assertions),
            }

            # 3. Compute and log evaluator means
            for ev_name in report.evaluators:
                scores = []
                for comp in report.comparisons:
                    if ev_name in comp.scores:
                        val = comp.scores[ev_name].v2_score
                        if isinstance(val, (int, float)):
                            scores.append(float(val))
                if scores:
                    metrics[f"eval_{ev_name}_mean"] = round(sum(scores) / len(scores), 4)

            mlflow.log_metrics(metrics)

            # 4. Log Per-Case Metrics as Step Time-Series
            for step_idx, comp in enumerate(report.comparisons):
                mlflow.log_metrics(
                    {
                        "step_latency_v1_ms": comp.v1_result.latency_ms,
                        "step_latency_v2_ms": comp.v2_result.latency_ms,
                        "step_cost_v1_usd": comp.v1_result.cost_usd,
                        "step_cost_v2_usd": comp.v2_result.cost_usd,
                    },
                    step=step_idx,
                )

            # 5. Log Tags
            mlflow.set_tags(
                {
                    "framework": "promptdiff",
                    "version": "2.0.0",
                    "status": v.status,
                }
            )

            # 6. Log Artifacts (JSON report)
            with tempfile.TemporaryDirectory() as tmpdir:
                report_json_path = Path(tmpdir) / "report.json"
                report_json_path.write_text(
                    json.dumps(report.model_dump(mode="json"), indent=2),
                    encoding="utf-8",
                )
                mlflow.log_artifact(str(report_json_path), artifact_path="promptdiff_reports")

            logger.info(f"Successfully logged evaluation run {run.info.run_id} to MLflow experiment '{experiment_name}'")
            return True

    except Exception as e:
        logger.error(f"Failed to log run to MLflow: {e}")
        return False
