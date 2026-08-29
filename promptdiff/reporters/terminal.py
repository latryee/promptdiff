"""Rich Terminal Reporter for promptdiff v2.0."""

from __future__ import annotations

from rich.box import ROUNDED
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from promptdiff.core.models import ArenaReport, DiffReport
from promptdiff.diff.visualizer import build_comparison_panel

ASCII_BANNER = r"""[bold cyan]
                                _     _ _  __  __
  _ __  _ __ ___  _ __ ___  _ __ | |_ _| (_)/ _|/ _|
 | '_ \| '__/ _ \| '_ ` _ \| '_ \| __/ _` | | |_| |_
 | |_) | | | (_) | | | | | | |_) | || (_| | |  _|  _|
 | .__/|_|  \___/|_| |_| |_| .__/ \__\__,_|_|_| |_|
 |_|                       |_|
[/bold cyan][dim] Enterprise LLM Prompt & Model Regression Tester CLI v2.0.0[/dim]
"""


def render_terminal_report(report: DiffReport, console: Console | None = None) -> None:
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
    summary_table.add_column(f"Baseline ({report.v1_name})", justify="right", style="cyan")
    summary_table.add_column(f"Candidate ({report.v2_name})", justify="right", style="magenta")
    summary_table.add_column("Delta / Impact", justify="right")

    # Cost row (Green for reduction, Red for inflation)
    cost_sign = "+" if v.cost_delta_pct > 0 else ""
    cost_color = "red" if v.cost_delta_pct > 0 else "green"
    cost_status = f"[{cost_color}]{cost_sign}{v.cost_delta_pct:.1f}% ({v.total_cost_v2 - v.total_cost_v1:+.6f}$)[/{cost_color}]"
    summary_table.add_row(
        "Total Token Cost",
        f"${v.total_cost_v1:.6f}",
        f"${v.total_cost_v2:.6f}",
        cost_status,
    )

    # Latency row (Green for faster, Red for slower)
    lat_sign = "+" if v.latency_delta_pct > 0 else ""
    lat_color = "red" if v.latency_delta_pct > 0 else "green"
    lat_status = f"[{lat_color}]{lat_sign}{v.latency_delta_pct:.1f}% ({v.avg_latency_v2 - v.avg_latency_v1:+.1f} ms)[/{lat_color}]"
    summary_table.add_row(
        "Avg Latency",
        f"{v.avg_latency_v1:.1f} ms",
        f"{v.avg_latency_v2:.1f} ms",
        lat_status,
    )

    # Evaluator metrics summary rows
    for ev_name in report.evaluators:
        scores_v1 = []
        scores_v2 = []
        for comp in report.comparisons:
            if ev_name in comp.scores:
                s1 = comp.scores[ev_name].v1_score
                s2 = comp.scores[ev_name].v2_score
                if isinstance(s1, (int, float)):
                    scores_v1.append(float(s1))
                if isinstance(s2, (int, float)):
                    scores_v2.append(float(s2))
        if scores_v1 and scores_v2:
            m1 = sum(scores_v1) / len(scores_v1)
            m2 = sum(scores_v2) / len(scores_v2)
            d = m2 - m1
            color = "green" if d >= 0 else "red"
            sign = "+" if d > 0 else ""
            summary_table.add_row(
                f"Eval: {ev_name.replace('_', ' ').title()}",
                f"{m1:.3f}",
                f"{m2:.3f}",
                f"[{color}]{sign}{d:.3f}[/{color}]",
            )

    # Test cases count
    passed_cases = report.aggregate_stats.get("passed_cases", len(report.comparisons))
    summary_table.add_row(
        "Test Cases Passed",
        f"{report.total_cases} total",
        f"[green]{passed_cases} passed[/green]",
        f"[red]{report.total_cases - passed_cases} failed[/red]" if passed_cases < report.total_cases else "[green]0 failed[/green]",
    )

    console.print()
    console.print(summary_table)

    # 3. Regression Verdict Callout
    if v.passed:
        verdict_panel = Panel(
            "[bold green][PASS] NO REGRESSIONS DETECTED[/bold green]\n"
            f"[dim]All {report.total_cases} test cases and assertion rules passed successfully. Quality gate cleared.[/dim]",
            border_style="green",
            box=ROUNDED,
            padding=(1, 2),
        )
    else:
        failure_bullets = "\n".join(f" [bold red][FAIL][/bold red] {f}" for f in v.failed_assertions)
        verdict_panel = Panel(
            f"[bold red][!] REGRESSION DETECTED ({len(v.failed_assertions)} Failed Assertions)[/bold red]\n\n"
            f"{failure_bullets}\n\n"
            f"[dim red]Quality gate failed. Pull Request check will be blocked.[/dim red]",
            border_style="red",
            box=ROUNDED,
            padding=(1, 2),
        )

    console.print()
    console.print(verdict_panel)
    console.print()


def render_arena_terminal_report(report: ArenaReport, console: Console | None = None) -> None:
    """Render Multi-Model Arena leaderboard and ranking table."""
    if console is None:
        console = Console()

    console.print(ASCII_BANNER)

    leaderboard_table = Table(
        title="[bold yellow]🏆 Multi-Model Arena Leaderboard & Cost-Performance Analysis[/bold yellow]",
        box=ROUNDED,
        show_header=True,
        header_style="bold cyan",
        expand=True,
    )
    leaderboard_table.add_column("Rank", justify="center", style="bold")
    leaderboard_table.add_column("Variant / Prompt", style="bold white")
    leaderboard_table.add_column("Model Engine", style="cyan")
    leaderboard_table.add_column("Total Cost ($)", justify="right", style="green")
    leaderboard_table.add_column("Avg Latency (ms)", justify="right", style="yellow")
    leaderboard_table.add_column("Avg Tokens", justify="right", style="blue")

    for s in report.leaderboard:
        rank_badge = "🥇 #1" if s.rank == 1 else ("🥈 #2" if s.rank == 2 else ("🥉 #3" if s.rank == 3 else f"#{s.rank}"))
        leaderboard_table.add_row(
            rank_badge,
            s.name,
            s.model,
            f"${s.total_cost:.6f}",
            f"{s.avg_latency_ms:.1f} ms",
            f"{s.avg_tokens:.0f}",
        )

    console.print()
    console.print(leaderboard_table)

    winner = report.leaderboard[0] if report.leaderboard else None
    if winner:
        winner_panel = Panel(
            f"[bold green]🏆 Arena Winner: {winner.name} ({winner.model})[/bold green]\n"
            f"[dim]Lowest overall cost (${winner.total_cost:.6f}) and high-efficiency throughput across {report.total_cases} test cases.[/dim]",
            border_style="green",
            box=ROUNDED,
            padding=(1, 2),
        )
        console.print()
        console.print(winner_panel)
        console.print()
