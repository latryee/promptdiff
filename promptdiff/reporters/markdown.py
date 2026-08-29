"""Markdown Summary Reporter for GitHub PR Comments & CI/CD Summaries."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from promptdiff.core.models import DiffReport


def generate_markdown_report(report: DiffReport, output_path: Optional[str] = None) -> str:
    """Generate clean Markdown table summary for CI/CD."""
    v = report.verdict
    status_icon = "✅ **PASSED**" if v.passed else "❌ **REGRESSION DETECTED**"

    lines = [
        "## ⚡ promptdiff Regression Report",
        "",
        f"**Comparison**: `{report.v1_name}` ➔ `{report.v2_name}`  ",
        f"**Status**: {status_icon}  ",
        f"**Timestamp**: `{report.timestamp}`  ",
        "",
        "### 📊 Key Performance Metrics",
        "",
        "| Metric | v1 Baseline | v2 Candidate | Delta / Impact |",
        "| :--- | :--- | :--- | :--- |",
        f"| **Total Token Cost** | `${v.total_cost_v1:.6f}` | `${v.total_cost_v2:.6f}` | `{v.cost_delta_pct:+.1f}%` |",
        f"| **Avg Latency** | `{v.avg_latency_v1:.1f}ms` | `{v.avg_latency_v2:.1f}ms` | `{v.latency_delta_pct:+.1f}%` |",
        f"| **Test Cases** | `{report.total_cases}` total | `{report.aggregate_stats.get('passed_cases', report.total_cases)}` passed | `{report.total_cases - report.aggregate_stats.get('passed_cases', report.total_cases)}` regressions |",
        "",
    ]

    if not v.passed:
        lines.append("### ⚠️ Failed Assertions & Regressions")
        lines.append("")
        for f in v.failed_assertions:
            lines.append(f"- ❌ `{f}`")
        lines.append("")

    lines.append("### 🔍 Test Case Breakdown")
    lines.append("")
    lines.append("| Test ID | JSON Valid | Similarity | Latency Delta | Cost Delta | Verdict |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")

    for comp in report.comparisons:
        tc = comp.test_case
        json_s = comp.scores.get("json_validity")
        json_txt = f"{json_s.v2_score:.1f}" if json_s else "-"

        sim_s = comp.scores.get("similarity")
        sim_txt = f"{sim_s.v2_score * 100:.1f}%" if sim_s else "-"

        lat_s = comp.scores.get("latency")
        lat_txt = f"{lat_s.delta_pct:+.1f}%" if lat_s and lat_s.delta_pct is not None else "-"

        cost_s = comp.scores.get("cost")
        cost_txt = f"{cost_s.delta_pct:+.1f}%" if cost_s and cost_s.delta_pct is not None else "-"

        all_pass = all(s.passed for s in comp.scores.values())
        verdict_str = "✅ PASS" if all_pass else "❌ FAIL"

        lines.append(f"| `{tc.id}` | `{json_txt}` | `{sim_txt}` | `{lat_txt}` | `{cost_txt}` | {verdict_str} |")

    lines.append("")
    lines.append("---")
    lines.append("*Generated automatically by [promptdiff](https://github.com/latryee/promptdiff)*")

    content = "\n".join(lines)
    if output_path:
        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")

    return content
