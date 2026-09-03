"""Tests migrated to feature-focused test suite."""

from __future__ import annotations

import json
from pathlib import Path

import promptdiff
from promptdiff.core.models import DiffReport, RegressionVerdict
from promptdiff.reporters.bundle_html import generate_interactive_bundle_html
from promptdiff.reporters.executive import ExecutiveReportExporter
from promptdiff.reporters.pr_bot import (
    STICKY_HEADER_TAG,
    generate_pr_markdown_comment,
    parse_pr_number_from_event,
)


def test_pr_commenter_markdown_generation(tmp_path: Path) -> None:
    """Test GitHub PR comment generation."""
    sample_report = {
        "v1_name": "system_v1.txt",
        "v2_name": "system_v2.txt",
        "model_v2": "gpt-4o",
        "total_cases": 2,
        "verdict": {
            "passed": True,
            "status": "PASSED",
            "failed_assertions": [],
            "total_cost_v1": 0.0002,
            "total_cost_v2": 0.0001,
            "cost_delta_pct": -50.0,
            "avg_latency_v1": 200.0,
            "avg_latency_v2": 150.0,
            "latency_delta_pct": -25.0,
        },
        "aggregate_stats": {"passed_cases": 2},
        "evaluators": ["similarity", "llm_judge"],
        "comparisons": [
            {
                "test_case": {"id": "c1", "description": "Greeting"},
                "v1_result": {"output": "Hello world", "latency_ms": 200.0},
                "v2_result": {"output": "Hi world", "latency_ms": 150.0},
                "scores": {
                    "similarity": {"v1_score": 1.0, "v2_score": 0.9, "passed": True},
                    "llm_judge": {"v1_score": 4.0, "v2_score": 4.8, "passed": True},
                },
            }
        ],
    }

    body = generate_pr_markdown_comment(sample_report, forecast_vol="500k")
    assert STICKY_HEADER_TAG in body
    assert "All Quality Gates Passed" in body
    assert "-50.0%" in body
    assert "Projected Monthly Impact" in body

    # Event payload parsing
    event_file = tmp_path / "event.json"
    event_file.write_text(json.dumps({"pull_request": {"number": 42}}), encoding="utf-8")
    pr_num = parse_pr_number_from_event(str(event_file))
    assert pr_num == 42


def test_jupyter_notebook_exporter(tmp_path: Path) -> None:
    """Test Jupyter Notebook .ipynb export."""
    report = promptdiff.compare(
        v1="Say hello: {{query}}",
        v2="Greet user: {{query}}",
        dataset=[{"id": "t1", "vars": {"query": "Alice"}}],
        mock=True,
    )
    nb_path = str(tmp_path / "experiment.ipynb")
    saved_path = promptdiff.export_notebook(report, output_path=nb_path)
    assert Path(saved_path).exists()
    assert Path(saved_path).stat().st_size > 100


def test_bundle_html_exporter(tmp_path: Path) -> None:
    """Test standalone interactive HTML bundle generation."""
    out_file = str(tmp_path / "bundle.html")
    report = promptdiff.compare(
        v1="Hello: {{query}}",
        v2="Hi: {{query}}",
        dataset=[{"id": "t1", "vars": {"query": "World"}}],
        mock=True,
    )
    bpath = generate_interactive_bundle_html(report, out_file)
    assert Path(bpath).exists()
    content = Path(bpath).read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in content
    assert "PromptDiff Interactive Regression Report" in content


def test_executive_report_exporter() -> None:
    """Test executive scorecard briefing exporter."""
    verdict = RegressionVerdict(
        passed=True,
        status="PASSED",
        failed_assertions=[],
        total_cost_v1=0.01,
        total_cost_v2=0.008,
        cost_delta_pct=-20.0,
        avg_latency_v1=200.0,
        avg_latency_v2=190.0,
        latency_delta_pct=-5.0,
    )
    report = DiffReport(
        v1_name="v1",
        v2_name="v2",
        model_v1="gpt-4o",
        model_v2="gpt-4o",
        comparisons=[],
        verdict=verdict,
        evaluators=["latency"],
        total_cases=0,
    )
    exporter = ExecutiveReportExporter()
    card = exporter.generate(report, project_name="Banking Bot")
    assert card.decision == "APPROVED FOR PRODUCTION"
    assert card.annualized_savings_usd > 0.0
    md = exporter.export_markdown(card)
    assert "Executive AI Telemetry" in md
