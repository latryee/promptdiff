"""Unit tests for LLM Judge human calibration utilities."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.calibrate_judge import (
    compute_calibration_metrics,
    compute_pearson_r,
    compute_spearman_rho,
    run_calibration,
)


def test_correlation_statistics() -> None:
    # Perfectly correlated
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [2.0, 4.0, 6.0, 8.0, 10.0]
    assert compute_pearson_r(x, y) == 1.0
    assert compute_spearman_rho(x, y) == 1.0

    # Negative correlation
    y_neg = [10.0, 8.0, 6.0, 4.0, 2.0]
    assert compute_pearson_r(x, y_neg) == -1.0
    assert compute_spearman_rho(x, y_neg) == -1.0

    metrics = compute_calibration_metrics(x, y)
    assert metrics.sample_count == 5
    assert metrics.pearson_r == 1.0
    assert "STRONG ALIGNMENT" in metrics.alignment_verdict


@pytest.mark.asyncio
async def test_run_calibration_with_dataset(tmp_path: Path) -> None:
    data = [
        {"prompt": "Summarize text", "output": "Good concise summary.", "human_score": 4.5},
        {"prompt": "Write email", "output": "Dear team, updates enclosed.", "human_score": 4.0},
    ]
    file_path = tmp_path / "calibration_data.json"
    import json

    file_path.write_text(json.dumps(data), encoding="utf-8")

    metrics = await run_calibration(str(file_path), model_name="gpt-4o", force_mock=True)
    assert metrics.sample_count == 2
    assert metrics.mae >= 0.0
