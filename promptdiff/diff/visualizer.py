"""Rich Terminal Visualizer for Side-by-Side Prompt Diffing."""

from __future__ import annotations

from rich.columns import Columns
from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from promptdiff.core.models import ComparisonResult, DiffChunk


def render_diff_text(chunks: list[DiffChunk], side: str = "v2") -> Text:
    """Render highlighted Rich Text from diff chunks."""
    result = Text()
    for chunk in chunks:
        if chunk.kind == "equal":
            txt = chunk.v1_text if side == "v1" else chunk.v2_text
            result.append(txt, style="bright_white")
        elif chunk.kind == "delete":
            if side == "v1":
                result.append(chunk.v1_text, style="bold red on dark_red")
        elif chunk.kind == "insert":
            if side == "v2":
                result.append(chunk.v2_text, style="bold green on dark_green")
        elif chunk.kind == "replace":
            if side == "v1":
                result.append(chunk.v1_text, style="bold magenta on dark_red")
            else:
                result.append(chunk.v2_text, style="bold cyan on dark_green")
    return result


def build_comparison_panel(comparison: ComparisonResult, console_width: int = 120) -> RenderableType:
    """Build a side-by-side Rich comparison renderable for a test case."""
    v1 = comparison.v1_result
    v2 = comparison.v2_result
    tc = comparison.test_case

    # Subtitle / Header badges for v1
    v1_cost_str = f"${v1.cost_usd:.5f}"
    v1_title = f"[bold cyan]v1 (Baseline)[/bold cyan] | [dim]{v1.model}[/dim]"
    v1_subtitle = f"Latency: [yellow]{v1.latency_ms:.1f}ms[/yellow]  Tokens: [blue]{v1.total_tokens}[/blue]  Cost: [green]{v1_cost_str}[/green]"

    # Subtitle / Header badges for v2
    v2_title = f"[bold magenta]v2 (Candidate)[/bold magenta] | [dim]{v2.model}[/dim]"

    # Latency badge with delta
    lat_delta_ms = v2.latency_ms - v1.latency_ms
    lat_delta_pct = (lat_delta_ms / v1.latency_ms * 100.0) if v1.latency_ms > 0 else 0.0
    lat_color = "green" if lat_delta_ms <= 0 else "red"
    lat_str = f"Latency: [{lat_color}]{v2.latency_ms:.1f}ms ({lat_delta_ms:+.1f}ms, {lat_delta_pct:+.1f}%)[/{lat_color}]"

    # Cost badge with delta
    cost_delta = v2.cost_usd - v1.cost_usd
    cost_delta_pct = (cost_delta / v1.cost_usd * 100.0) if v1.cost_usd > 0 else 0.0
    cost_color = "green" if cost_delta <= 0 else "red"
    cost_str = f"Cost: [{cost_color}]${v2.cost_usd:.5f} ({cost_delta_pct:+.1f}%)[/{cost_color}]"

    # Tokens badge
    tok_delta = v2.total_tokens - v1.total_tokens
    tok_str = f"Tokens: [blue]{v2.total_tokens} ({tok_delta:+d})[/blue]"

    v2_subtitle = f"{lat_str}  {tok_str}  {cost_str}"

    # Build side-by-side highlighted text
    v1_text_render = render_diff_text(comparison.text_diff, side="v1")
    v2_text_render = render_diff_text(comparison.text_diff, side="v2")

    panel_width = max(40, (console_width - 8) // 2)

    p1 = Panel(
        v1_text_render,
        title=v1_title,
        subtitle=v1_subtitle,
        border_style="cyan",
        width=panel_width,
        padding=(1, 2),
    )

    p2 = Panel(
        v2_text_render,
        title=v2_title,
        subtitle=v2_subtitle,
        border_style="magenta",
        width=panel_width,
        padding=(1, 2),
    )

    side_by_side = Columns([p1, p2], equal=True, expand=True)

    # Score metrics summary table under the panels
    score_table = Table(
        show_header=True,
        header_style="bold white",
        box=None,
        padding=(0, 2),
    )
    score_table.add_column("Metric", style="bold yellow")
    score_table.add_column("v1 Baseline", style="cyan")
    score_table.add_column("v2 Candidate", style="magenta")
    score_table.add_column("Delta / Status", style="bold")
    score_table.add_column("Verdict", justify="center")

    for metric_name, score in comparison.scores.items():
        verdict = "[green]PASS[/green]" if score.passed else "[red]FAIL[/red]"
        score_table.add_row(
            metric_name.replace("_", " ").title(),
            str(score.v1_score),
            str(score.v2_score),
            score.message,
            verdict,
        )

    # Case Header Info
    case_desc = f"[bold yellow]Test Case: {tc.id}[/bold yellow]"
    if tc.description:
        case_desc += f" [dim]({tc.description})[/dim]"
    if tc.vars:
        vars_str = ", ".join(f"{k}={v}" for k, v in tc.vars.items())
        case_desc += f"\n[dim italic]Vars: {vars_str}[/dim italic]"

    # Wrap in master container panel
    return Group(
        Text.from_markup(f"\n{case_desc}"),
        side_by_side,
        score_table,
    )
