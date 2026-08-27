"""Main CLI Application for promptdiff using Typer and Rich."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import sys
from typing import List, Optional
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# Reconfigure stdout/stderr for utf-8 on Windows
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from promptdiff.cli.formatters import console, print_init_success, print_pricing_table
from promptdiff.core.cache import DiskCache
from promptdiff.core.config import load_dataset, load_project_config, load_prompt_file
from promptdiff.core.models import TestCase
from promptdiff.core.runner import PromptDiffRunner
from promptdiff.evaluators.registry import get_evaluators
from promptdiff.providers.registry import get_provider
from promptdiff.reporters.html import generate_html_report
from promptdiff.reporters.json_reporter import generate_json_report
from promptdiff.reporters.markdown import generate_markdown_report
from promptdiff.reporters.terminal import render_terminal_report

app = typer.Typer(
    name="promptdiff",
    help="⚡ LLM Prompt & Output Regression Tester CLI with side-by-side visual diffs & CI/CD assertions.",
    add_completion=False,
    no_args_is_help=True,
)

cache_app = typer.Typer(name="cache", help="Manage deterministic prompt execution cache.")
app.add_typer(cache_app)


@app.command(name="test")
def test_cmd(
    v1: str = typer.Argument(..., help="Path to v1 prompt template file or raw text string"),
    v2: str = typer.Argument(..., help="Path to v2 prompt template file or raw text string"),
    inputs: Optional[str] = typer.Option(None, "--inputs", "-i", help="Path to test dataset (.jsonl, .yaml, .csv, .json)"),
    eval_metrics: str = typer.Option(
        "json_validity,latency,cost,similarity",
        "--eval",
        "-e",
        help="Comma-separated evaluation metrics (e.g. 'json_validity,latency,cost,similarity')",
    ),
    model: str = typer.Option("gpt-4o", "--model", "-m", help="Target LLM model (e.g. gpt-4o, claude-3-5-sonnet, gemini-2.0-flash)"),
    model_v1: Optional[str] = typer.Option(None, "--model-v1", help="Override model specifically for v1"),
    model_v2: Optional[str] = typer.Option(None, "--model-v2", help="Override model specifically for v2"),
    temperature: float = typer.Option(0.0, "--temperature", "-t", help="Sampling temperature (0.0 to 1.0)"),
    system_prompt: Optional[str] = typer.Option(None, "--system", "-s", help="Optional system prompt or path to file"),
    assertions: Optional[List[str]] = typer.Option(
        None,
        "--assert",
        "-a",
        help="Regression assertion threshold (e.g. 'cost_delta <= 10%', 'latency_delta <= 15%', 'json_validity == 1.0')",
    ),
    mock: bool = typer.Option(False, "--mock", help="Use deterministic offline MockProvider (no API keys required)"),
    cache_enabled: bool = typer.Option(True, "--cache/--no-cache", help="Enable or disable persistent disk cache"),
    export_html: Optional[str] = typer.Option(None, "--export-html", help="Path to export standalone interactive HTML report"),
    export_markdown: Optional[str] = typer.Option(None, "--export-markdown", help="Path to export Markdown report for GitHub PRs"),
    export_json: Optional[str] = typer.Option(None, "--export-json", help="Path to export structured JSON report"),
    concurrency: int = typer.Option(4, "--concurrency", "-c", help="Number of concurrent LLM requests"),
) -> None:
    """Run regression comparison between two prompt versions across test cases."""
    # 1. Load prompt templates
    m1 = model_v1 or model
    m2 = model_v2 or model

    v1_prompt = load_prompt_file(v1, version_name="v1", model=m1, temperature=temperature)
    v2_prompt = load_prompt_file(v2, version_name="v2", model=m2, temperature=temperature)

    if system_prompt:
        if Path(system_prompt).is_file():
            sys_text = Path(system_prompt).read_text(encoding="utf-8")
        else:
            sys_text = system_prompt
        v1_prompt.system_prompt = sys_text
        v2_prompt.system_prompt = sys_text

    # 2. Load test case dataset
    try:
        test_cases = load_dataset(inputs)
    except Exception as e:
        console.print(f"[bold red]Error loading dataset:[/bold red] {e}")
        raise typer.Exit(code=1)

    # 3. Resolve Providers
    p1 = get_provider(model_name=m1, force_mock=mock)
    p2 = get_provider(model_name=m2, force_mock=mock)

    # 4. Resolve Evaluators & Assertions
    eval_list = get_evaluators([eval_metrics])
    cache = DiskCache(enabled=cache_enabled)

    runner = PromptDiffRunner(
        v1_prompt=v1_prompt,
        v2_prompt=v2_prompt,
        provider_v1=p1,
        provider_v2=p2,
        evaluators=eval_list,
        assertions=assertions,
        cache=cache,
        concurrency=concurrency,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(f"Running promptdiff on {len(test_cases)} test case(s)...", total=len(test_cases))

        def on_step(current: int, total: int) -> None:
            progress.update(task, completed=current)

        # Run async loop
        report = asyncio.run(runner.run(test_cases, progress_cb=on_step))

    # 5. Render Output
    render_terminal_report(report, console=console)

    # 6. Export Reports if requested
    if export_html:
        path = generate_html_report(report, export_html)
        console.print(f"[bold green][+] HTML Report generated:[/bold green] [cyan]{path}[/cyan]")

    if export_markdown:
        generate_markdown_report(report, export_markdown)
        console.print(f"[bold green][+] Markdown Report generated:[/bold green] [cyan]{export_markdown}[/cyan]")

    if export_json:
        generate_json_report(report, export_json)
        console.print(f"[bold green][+] JSON Report generated:[/bold green] [cyan]{export_json}[/cyan]")

    # 7. CI/CD Exit Code: 0 on Pass, 1 on Regression
    if not report.verdict.passed:
        raise typer.Exit(code=1)


@app.command(name="diff")
def diff_cmd(
    v1: str = typer.Argument(..., help="Path to first prompt file"),
    v2: str = typer.Argument(..., help="Path to second prompt file"),
) -> None:
    """Quick side-by-side terminal diff of two prompt templates without LLM execution."""
    p1 = Path(v1)
    p2 = Path(v2)

    if not p1.is_file() or not p2.is_file():
        console.print("[bold red]Both arguments must be valid files.[/bold red]")
        raise typer.Exit(code=1)

    t1 = p1.read_text(encoding="utf-8")
    t2 = p2.read_text(encoding="utf-8")

    from promptdiff.diff.text_diff import compute_word_diff
    from promptdiff.diff.visualizer import render_diff_text
    from rich.columns import Columns
    from rich.panel import Panel

    chunks = compute_word_diff(t1, t2)
    r1 = render_diff_text(chunks, side="v1")
    r2 = render_diff_text(chunks, side="v2")

    panel1 = Panel(r1, title=f"[cyan]{p1.name} (v1)[/cyan]", border_style="cyan")
    panel2 = Panel(r2, title=f"[magenta]{p2.name} (v2)[/magenta]", border_style="magenta")

    console.print(Columns([panel1, panel2], equal=True))


@app.command(name="pricing")
def pricing_cmd(
    query: str = typer.Argument("", help="Filter model pricing by name"),
) -> None:
    """Display model token pricing table (cost per 1M tokens)."""
    print_pricing_table(query)


@app.command(name="init")
def init_cmd(
    directory: str = typer.Argument(".", help="Directory to scaffold promptdiff starter project"),
) -> None:
    """Scaffold sample promptdiff project with example prompts, testcases, and config."""
    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir = target_dir / "prompts"
    prompts_dir.mkdir(exist_ok=True)

    # 1. Baseline Prompt
    (prompts_dir / "system_v1.txt").write_text(
        "You are a customer support agent. Answer the user query politely and provide contact info.\nQuery: {{query}}",
        encoding="utf-8",
    )

    # 2. Candidate Prompt (Concise & Structured)
    (prompts_dir / "system_v2.txt").write_text(
        "You are a customer support agent. Answer the user query concisely using bullet points.\nQuery: {{query}}",
        encoding="utf-8",
    )

    # 3. Test Cases Dataset
    (target_dir / "testcases.jsonl").write_text(
        '{"id": "tc_1", "description": "Password reset inquiry", "vars": {"query": "How do I reset my password?"}}\n'
        '{"id": "tc_2", "description": "Refund request", "vars": {"query": "I want a refund for my order #1234."}}\n'
        '{"id": "tc_3", "description": "API rate limits", "vars": {"query": "What are the rate limits on Tier 2?"}}\n',
        encoding="utf-8",
    )

    # 4. Config File
    (target_dir / "promptdiff.yaml").write_text(
        "v1_prompt: prompts/system_v1.txt\n"
        "v2_prompt: prompts/system_v2.txt\n"
        "model: gpt-4o\n"
        "evaluators:\n"
        "  - json_validity\n"
        "  - latency\n"
        "  - cost\n"
        "  - similarity\n"
        "assertions:\n"
        '  - "cost_delta <= 15%"\n'
        '  - "latency_delta <= 20%"\n'
        "dataset: testcases.jsonl\n",
        encoding="utf-8",
    )

    print_init_success(str(target_dir.resolve()))


@cache_app.command(name="clear")
def cache_clear_cmd() -> None:
    """Clear promptdiff persistent cache."""
    cache = DiskCache(enabled=True)
    count = cache.clear()
    console.print(f"[bold green][+] Cleared {count} cached execution entries.[/bold green]")


@cache_app.command(name="stats")
def cache_stats_cmd() -> None:
    """Show cache statistics."""
    cache = DiskCache(enabled=True)
    count = cache.count()
    console.print(f"[bold cyan]Prompt Cache Entries:[/bold cyan] [bold yellow]{count}[/bold yellow]")


def main() -> int:
    """Main execution wrapper."""
    try:
        app()
        return 0
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 0


if __name__ == "__main__":
    sys.exit(main())
