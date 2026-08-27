"""Rich Terminal Reporter for promptdiff."""

from __future__ import annotations

from typing import Optional
from rich.box import ROUNDED
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from promptdiff.core.models import DiffReport
from promptdiff.diff.visualizer import build_comparison_panel

ASCII_BANNER = r"""[bold cyan]
                               _     _ _  __  __
  _ __  _ __ ___  _ __ ___  _ __ | |_ _| (_)/ _|/ _|
 | '_ \| '__/ _ \| '_ ` _ \| '_ \| __/ _` | | |_| |_
 | |_) | | | (_) | | | | | | |_) | || (_| | |  _|  _|
 | .__/|_|  \___/|_| |_| |_| .__/ \__\__,_|_|_| |_|
 |_|                       |_|
[/bold cyan][dim] LLM Prompt & Output Regression Tester CLI v0.1.0[/dim]
"""


def render_terminal_report(report: DiffReport, console: Optional[Console] = None) -> None:
    """Render full regression test report in rich terminal UI."""
    if console is None:
        console = Console()

    console.print(ASCII_BANNER)

    # 1. Render all comparison panels
    for comp in report.comparisons:
        panel_group = build_comparison_panel(comp, console_width=console.width)
        console.print(panel_group)
        console.print("[dim]" + "─" * min(console.width, 100) + "[/dim]")

    # 2. Executive Summary Box
    v = report.verdict

    summary_table = Table(
        title="[bold yellow]Execution & Regression Summary[/bold yellow]",
        box=ROUNDED,
        show_header=True,
        header_style="bold cyan",
        expand=True,
    )
    summary_table.add_column("Metric", style="bold")
    summary_table.add_column("v1 Baseline", justify="right")
    summary_table.add_column("v2 Candidate", justify="right")
    summary_table.add_column("Delta / Impact", justify="right")

    # Cost row
    cost_sign = "+" if v.cost_delta_pct > 0 else ""
    cost_color = "red" if v.cost_delta_pct > 0 else "green"
    summary_table.add_row(
        "Total Token Cost",
        f"${v.total_cost_v1:.6f}",
        f"${v.total_cost_v2:.6f}",
        f"[{cost_color}]{cost_sign}{v.cost_delta_pct:.1f}%[/{cost_color}]",
    )

    # Latency row
    lat_sign = "+" if v.latency_delta_pct > 0 else ""
    lat_color = "red" if v.latency_delta_pct > 0 else "green"
    summary_table.add_row(
        "Avg Latency",
        f"{v.avg_latency_v1:.1f} ms",
        f"{v.avg_latency_v2:.1f} ms",
        f"[{lat_color}]{lat_sign}{v.latency_delta_pct:.1f}%[/{lat_color}]",
    )

    # Test cases count
    passed_cases = report.aggregate_stats.get("passed_cases", len(report.comparisons))
    summary_table.add_row(
        "Test Cases Passed",
        f"{report.total_cases} total",
        f"{passed_cases} passed",
        f"{report.total_cases - passed_cases} failed",
    )

    console.print()
    console.print(summary_table)

    # 3. Regression Verdict Callout
    if v.passed:
        verdict_panel = Panel(
            "[bold green][PASS] NO REGRESSIONS DETECTED[/bold green]\n"
            f"[dim]All {report.total_cases} test cases and assertion rules passed successfully.[/dim]",
            border_style="green",
            box=ROUNDED,
            padding=(1, 2),
        )
    else:
        failure_bullets = "\n".join(f" [bold red][FAIL][/bold red] {f}" for f in v.failed_assertions)
        verdict_panel = Panel(
            f"[bold red][!] REGRESSION DETECTED ({len(v.failed_assertions)} Failed Assertions)[/bold red]\n\n"
            f"{failure_bullets}",
            border_style="red",
            box=ROUNDED,
            padding=(1, 2),
        )

    console.print()
    console.print(verdict_panel)
    console.print()
