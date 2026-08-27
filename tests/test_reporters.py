"""Unit tests for HTML, Markdown, JSON, and Terminal Reporters."""

import json
from pathlib import Path
from promptdiff.core.models import (
    ComparisonResult,
    DiffChunk,
    DiffReport,
    EvaluatorScore,
    RegressionVerdict,
    RunResult,
    TestCase,
)
from promptdiff.reporters.html import generate_html_report
from promptdiff.reporters.json_reporter import generate_json_report
from promptdiff.reporters.markdown import generate_markdown_report
from promptdiff.reporters.terminal import render_terminal_report
from rich.console import Console


def sample_report() -> DiffReport:
    r1 = RunResult(
        prompt_name="v1",
        test_case_id="tc1",
        rendered_prompt="Hello",
        output="Hello world from v1",
        latency_ms=150.0,
        prompt_tokens=5,
        completion_tokens=10,
        total_tokens=15,
        cost_usd=0.0001,
        model="gpt-4o",
    )
    r2 = RunResult(
        prompt_name="v2",
        test_case_id="tc1",
        rendered_prompt="Hello",
        output="Hello universe from v2",
        latency_ms=120.0,
        prompt_tokens=5,
        completion_tokens=10,
        total_tokens=15,
        cost_usd=0.0001,
        model="gpt-4o",
    )
    comp = ComparisonResult(
        test_case=TestCase(id="tc1", description="Greeting test"),
        v1_result=r1,
        v2_result=r2,
        scores={
            "json_validity": EvaluatorScore(name="json_validity", v1_score=1.0, v2_score=1.0, message="Valid"),
            "latency": EvaluatorScore(name="latency", v1_score=150.0, v2_score=120.0, delta_pct=-20.0, message="-20%"),
        },
        text_diff=[
            DiffChunk(kind="equal", v1_text="Hello ", v2_text="Hello "),
            DiffChunk(kind="replace", v1_text="world", v2_text="universe"),
            DiffChunk(kind="equal", v1_text=" from v", v2_text=" from v"),
            DiffChunk(kind="replace", v1_text="1", v2_text="2"),
        ],
    )
    verdict = RegressionVerdict(
        passed=True,
        status="PASSED",
        total_cost_v1=0.0001,
        total_cost_v2=0.0001,
        avg_latency_v1=150.0,
        avg_latency_v2=120.0,
        latency_delta_pct=-20.0,
    )
    return DiffReport(
        timestamp="2026-08-27T12:00:00Z",
        v1_name="v1.txt",
        v2_name="v2.txt",
        model_v1="gpt-4o",
        model_v2="gpt-4o",
        comparisons=[comp],
        verdict=verdict,
        evaluators=["json_validity", "latency"],
        total_cases=1,
    )


def test_html_report_generation(tmp_path: Path):
    report = sample_report()
    out_file = tmp_path / "report.html"
    generate_html_report(report, str(out_file))

    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in content
    assert "promptdiff" in content
    assert "tc1" in content
    assert "ALL ASSERTIONS PASSED" in content


def test_markdown_report_generation(tmp_path: Path):
    report = sample_report()
    out_file = tmp_path / "report.md"
    md = generate_markdown_report(report, str(out_file))

    assert out_file.exists()
    assert "promptdiff Regression Report" in md
    assert "Total Token Cost" in md
    assert "PASSED" in md


def test_json_report_generation(tmp_path: Path):
    report = sample_report()
    out_file = tmp_path / "report.json"
    json_str = generate_json_report(report, str(out_file))

    assert out_file.exists()
    parsed = json.loads(json_str)
    assert parsed["v1_name"] == "v1.txt"
    assert parsed["verdict"]["passed"] is True


def test_terminal_report_rendering():
    report = sample_report()
    c = Console(record=True, width=120)
    render_terminal_report(report, console=c)
    out = c.export_text()
    assert "Execution & Regression Summary" in out
    assert "NO REGRESSIONS DETECTED" in out
