"""CLI Rich Formatting & UI Utilities."""

from __future__ import annotations

from typing import Dict
from rich.box import ROUNDED, SIMPLE
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from promptdiff.pricing import MODEL_PRICING_TABLE, ModelPrice

console = Console()


def print_pricing_table(filter_name: str = "") -> None:
    """Print formatted model pricing registry table."""
    table = Table(
        title="[bold cyan]LLM Model Pricing Registry (USD per 1M Tokens)[/bold cyan]",
        box=ROUNDED,
        header_style="bold yellow",
    )
    table.add_column("Model Name", style="bold white")
    table.add_column("Input / 1M", justify="right", style="green")
    table.add_column("Output / 1M", justify="right", style="cyan")
    table.add_column("Description", style="dim")

    for name, price in sorted(MODEL_PRICING_TABLE.items()):
        if filter_name and filter_name.lower() not in name.lower():
            continue
        table.add_row(
            name,
            f"${price.input_per_million:.2f}",
            f"${price.output_per_million:.2f}",
            price.description,
        )

    console.print(table)


def print_init_success(path: str) -> None:
    """Print project scaffolded message."""
    panel = Panel(
        f"[bold green][+] Initialized promptdiff starter kit in {path}[/bold green]\n\n"
        "[bold white]Created assets:[/bold white]\n"
        "  - [cyan]prompts/system_v1.txt[/cyan] (Baseline prompt)\n"
        "  - [cyan]prompts/system_v2.txt[/cyan] (Candidate prompt)\n"
        "  - [cyan]testcases.jsonl[/cyan]      (Regression test scenarios)\n"
        "  - [cyan]promptdiff.yaml[/cyan]      (Configuration file)\n\n"
        "[bold yellow]Run your first regression test:[/bold yellow]\n"
        "  [magenta]promptdiff test prompts/system_v1.txt prompts/system_v2.txt --inputs testcases.jsonl --mock[/magenta]",
        title="[bold cyan]promptdiff Setup Complete[/bold cyan]",
        box=ROUNDED,
        padding=(1, 2),
    )
    console.print(panel)
