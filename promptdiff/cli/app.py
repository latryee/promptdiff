"""Main CLI Application for promptdiff v3.0 using Typer and Rich."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import typer
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

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
from promptdiff.core.config import load_dataset, load_prompt_file
from promptdiff.core.models import PromptVersion
from promptdiff.core.runner import ArenaRunner, PromptDiffRunner
from promptdiff.evaluators.answer_relevance import AnswerRelevanceEvaluator
from promptdiff.evaluators.faithfulness import FaithfulnessEvaluator
from promptdiff.evaluators.llm_judge import LLMJudgeEvaluator
from promptdiff.evaluators.registry import get_evaluators
from promptdiff.generators.synthetic import SyntheticTestGenerator
from promptdiff.optimizer.auto_prompt import PromptOptimizer
from promptdiff.optimizer.tuner import PromptTuner
from promptdiff.providers.registry import get_provider
from promptdiff.reporters.html import generate_html_report
from promptdiff.reporters.json_reporter import generate_json_report
from promptdiff.reporters.markdown import generate_markdown_report
from promptdiff.reporters.mlflow_reporter import log_to_mlflow
from promptdiff.reporters.terminal import (
    render_arena_terminal_report,
    render_terminal_report,
    render_tuning_terminal_report,
)
from promptdiff.reporters.wandb_reporter import log_to_wandb

app = typer.Typer(
    name="promptdiff",
    help="⚡ Enterprise LLM Prompt & Model Regression Tester CLI with Textual TUI, Hyperparameter Tuning, Cost Forecasting & CI/CD PR Bot.",
    add_completion=False,
    no_args_is_help=True,
)

cache_app = typer.Typer(name="cache", help="Manage deterministic prompt execution cache.")
app.add_typer(cache_app)


def _run_test_suite(
    v1: str,
    v2: str,
    inputs: str | None,
    eval_metrics: str,
    model: str,
    model_v1: str | None,
    model_v2: str | None,
    temperature: float,
    system_prompt: str | None,
    assertions: list[str] | None,
    mock: bool,
    cache_enabled: bool,
    export_html: str | None,
    export_markdown: str | None,
    export_json: str | None,
    concurrency: int,
    fail_on_regression: bool,
    mlflow: bool,
    wandb: bool,
    mlflow_experiment: str,
    wandb_project: str,
    rubric: str | None,
    forecast: str | None,
) -> None:
    """Core test execution logic shared between `promptdiff test` and `promptdiff run`."""
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

    try:
        test_cases = load_dataset(inputs)
    except Exception as e:
        console.print(f"[bold red]Error loading dataset:[/bold red] {e}")
        raise typer.Exit(code=1)

    p1 = get_provider(model_name=m1, force_mock=mock)
    p2 = get_provider(model_name=m2, force_mock=mock)

    eval_list = get_evaluators([eval_metrics])
    for idx, ev in enumerate(eval_list):
        if isinstance(ev, LLMJudgeEvaluator):
            eval_list[idx] = LLMJudgeEvaluator(model_name=m2, rubric=rubric, force_mock=mock)
        elif isinstance(ev, FaithfulnessEvaluator):
            eval_list[idx] = FaithfulnessEvaluator(model_name=m2, force_mock=mock)
        elif isinstance(ev, AnswerRelevanceEvaluator):
            eval_list[idx] = AnswerRelevanceEvaluator(model_name=m2, force_mock=mock)

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
        BarColumn(bar_width=40, style="cyan", complete_style="green"),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task(f"Running promptdiff on {len(test_cases)} test case(s)...", total=len(test_cases))

        def on_step(current: int, total: int) -> None:
            progress.update(task, completed=current)

        report = asyncio.run(runner.run(test_cases, progress_cb=on_step))

    render_terminal_report(report, console=console, forecast=forecast)

    if export_html:
        path = generate_html_report(report, export_html)
        console.print(f"[bold green][+] HTML Report generated:[/bold green] [cyan]{path}[/cyan]")

    if export_markdown:
        generate_markdown_report(report, export_markdown)
        console.print(f"[bold green][+] Markdown Report generated:[/bold green] [cyan]{export_markdown}[/cyan]")

    if export_json:
        generate_json_report(report, export_json)
        console.print(f"[bold green][+] JSON Report generated:[/bold green] [cyan]{export_json}[/cyan]")

    if mlflow:
        ok = log_to_mlflow(report, experiment_name=mlflow_experiment)
        if ok:
            console.print(f"[bold green][+] Telemetry logged to MLflow experiment:[/bold green] [cyan]{mlflow_experiment}[/cyan]")

    if wandb:
        ok = log_to_wandb(report, project=wandb_project)
        if ok:
            console.print(f"[bold green][+] Telemetry logged to Weights & Biases project:[/bold green] [cyan]{wandb_project}[/cyan]")

    if fail_on_regression and not report.verdict.passed:
        console.print("[bold red][!] CI/CD Quality Gate: Regression threshold violated. Exiting with code 1.[/bold red]")
        raise typer.Exit(code=1)


@app.command(name="test")
def test_cmd(
    v1: str = typer.Argument(..., help="Path to v1 prompt template file or raw text string"),
    v2: str = typer.Argument(..., help="Path to v2 prompt template file or raw text string"),
    inputs: str | None = typer.Option(None, "--inputs", "-i", help="Path to test dataset (.jsonl, .yaml, .csv, .json)"),
    eval_metrics: str = typer.Option(
        "json_validity,latency,cost,similarity,faithfulness,answer_relevance,security",
        "--eval",
        "-e",
        help="Comma-separated evaluation metrics",
    ),
    model: str = typer.Option("gpt-4o", "--model", "-m", help="Target LLM model"),
    model_v1: str | None = typer.Option(None, "--model-v1", help="Override model specifically for v1"),
    model_v2: str | None = typer.Option(None, "--model-v2", help="Override model specifically for v2"),
    temperature: float = typer.Option(0.0, "--temperature", "-t", help="Sampling temperature (0.0 to 1.0)"),
    system_prompt: str | None = typer.Option(None, "--system", "-s", help="Optional system prompt or path to file"),
    assertions: list[str] | None = typer.Option(
        None,
        "--assert",
        "-a",
        help="Regression assertion threshold",
    ),
    mock: bool = typer.Option(False, "--mock", help="Use deterministic offline MockProvider"),
    cache_enabled: bool = typer.Option(True, "--cache/--no-cache", help="Enable or disable persistent disk cache"),
    export_html: str | None = typer.Option(None, "--export-html", help="Path to export standalone interactive HTML report"),
    export_markdown: str | None = typer.Option(None, "--export-markdown", help="Path to export Markdown report for GitHub PRs"),
    export_json: str | None = typer.Option(None, "--export-json", help="Path to export structured JSON report"),
    concurrency: int = typer.Option(4, "--concurrency", "-c", help="Number of concurrent LLM requests"),
    fail_on_regression: bool = typer.Option(True, "--fail-on-regression/--no-fail-on-regression", help="Exit with code 1 if regressions detected"),
    mlflow: bool = typer.Option(False, "--mlflow", help="Log metrics, parameters, and artifacts to MLflow"),
    wandb: bool = typer.Option(False, "--wandb", help="Log metrics and comparison tables to Weights & Biases"),
    mlflow_experiment: str = typer.Option("promptdiff-evals", "--mlflow-experiment", help="MLflow experiment name"),
    wandb_project: str = typer.Option("promptdiff", "--wandb-project", help="Weights & Biases project name"),
    rubric: str | None = typer.Option(None, "--rubric", help="Custom evaluation rubric for LLM Judge"),
    forecast: str | None = typer.Option(None, "--forecast", "-f", help="Projected daily production request volume (e.g. '1M', '500k')"),
) -> None:
    """Run regression comparison between two prompt versions across test cases."""
    _run_test_suite(
        v1=v1,
        v2=v2,
        inputs=inputs,
        eval_metrics=eval_metrics,
        model=model,
        model_v1=model_v1,
        model_v2=model_v2,
        temperature=temperature,
        system_prompt=system_prompt,
        assertions=assertions,
        mock=mock,
        cache_enabled=cache_enabled,
        export_html=export_html,
        export_markdown=export_markdown,
        export_json=export_json,
        concurrency=concurrency,
        fail_on_regression=fail_on_regression,
        mlflow=mlflow,
        wandb=wandb,
        mlflow_experiment=mlflow_experiment,
        wandb_project=wandb_project,
        rubric=rubric,
        forecast=forecast,
    )


@app.command(name="run")
def run_cmd(
    v1: str = typer.Argument(..., help="Path to v1 prompt template file or raw text string"),
    v2: str = typer.Argument(..., help="Path to v2 prompt template file or raw text string"),
    inputs: str | None = typer.Option(None, "--inputs", "-i", help="Path to test dataset (.jsonl, .yaml, .csv, .json)"),
    eval_metrics: str = typer.Option(
        "json_validity,latency,cost,similarity,faithfulness,answer_relevance,security",
        "--eval",
        "-e",
        help="Comma-separated evaluation metrics",
    ),
    model: str = typer.Option("gpt-4o", "--model", "-m", help="Target LLM model"),
    model_v1: str | None = typer.Option(None, "--model-v1", help="Override model specifically for v1"),
    model_v2: str | None = typer.Option(None, "--model-v2", help="Override model specifically for v2"),
    temperature: float = typer.Option(0.0, "--temperature", "-t", help="Sampling temperature"),
    system_prompt: str | None = typer.Option(None, "--system", "-s", help="Optional system prompt"),
    assertions: list[str] | None = typer.Option(None, "--assert", "-a", help="Regression assertions"),
    mock: bool = typer.Option(False, "--mock", help="Use deterministic offline mock"),
    cache_enabled: bool = typer.Option(True, "--cache/--no-cache", help="Enable disk cache"),
    export_html: str | None = typer.Option(None, "--export-html", help="Export HTML report"),
    export_markdown: str | None = typer.Option(None, "--export-markdown", help="Export Markdown report"),
    export_json: str | None = typer.Option(None, "--export-json", help="Export JSON report"),
    concurrency: int = typer.Option(4, "--concurrency", "-c", help="Concurrency limit"),
    fail_on_regression: bool = typer.Option(True, "--fail-on-regression/--no-fail-on-regression", help="Exit code 1 on regression"),
    mlflow: bool = typer.Option(False, "--mlflow", help="Log to MLflow"),
    wandb: bool = typer.Option(False, "--wandb", help="Log to WandB"),
    mlflow_experiment: str = typer.Option("promptdiff-evals", "--mlflow-experiment", help="MLflow experiment name"),
    wandb_project: str = typer.Option("promptdiff", "--wandb-project", help="W&B project name"),
    rubric: str | None = typer.Option(None, "--rubric", help="Custom evaluation rubric for LLM Judge"),
    forecast: str | None = typer.Option(None, "--forecast", "-f", help="Projected daily production request volume (e.g. '1M', '500k')"),
) -> None:
    """Run regression comparison between prompt versions (alias for `promptdiff test`)."""
    _run_test_suite(
        v1=v1,
        v2=v2,
        inputs=inputs,
        eval_metrics=eval_metrics,
        model=model,
        model_v1=model_v1,
        model_v2=model_v2,
        temperature=temperature,
        system_prompt=system_prompt,
        assertions=assertions,
        mock=mock,
        cache_enabled=cache_enabled,
        export_html=export_html,
        export_markdown=export_markdown,
        export_json=export_json,
        concurrency=concurrency,
        fail_on_regression=fail_on_regression,
        mlflow=mlflow,
        wandb=wandb,
        mlflow_experiment=mlflow_experiment,
        wandb_project=wandb_project,
        rubric=rubric,
        forecast=forecast,
    )


@app.command(name="tune")
def tune_cmd(
    prompt: str = typer.Argument(..., help="Path to prompt template file or raw prompt string"),
    inputs: str | None = typer.Option(None, "--inputs", "-i", help="Path to test dataset (.jsonl, .yaml, .csv, .json)"),
    model: str = typer.Option("gpt-4o", "--model", "-m", help="Target LLM model"),
    temperatures: str = typer.Option("0.0,0.3,0.7,1.0", "--temperatures", "-t", help="Comma-separated temperature grid points"),
    top_ps: str = typer.Option("0.7,0.9,1.0", "--top-ps", "-p", help="Comma-separated top_p grid points"),
    mock: bool = typer.Option(False, "--mock", help="Use deterministic mock execution"),
    concurrency: int = typer.Option(6, "--concurrency", "-c", help="Concurrency limit for evaluation"),
) -> None:
    """Hyperparameter Grid Search: Optimize temperature & top_p to identify Pareto-optimal configurations."""
    pv = load_prompt_file(prompt, version_name="tune_target", model=model)

    try:
        test_cases = load_dataset(inputs)
    except Exception as e:
        console.print(f"[bold red]Error loading dataset:[/bold red] {e}")
        raise typer.Exit(code=1)

    temp_list = [float(t.strip()) for t in temperatures.split(",") if t.strip()]
    top_p_list = [float(p.strip()) for p in top_ps.split(",") if p.strip()]

    tuner = PromptTuner(
        prompt_version=pv,
        test_cases=test_cases,
        model_name=model,
        temperatures=temp_list,
        top_ps=top_p_list,
        force_mock=mock,
        concurrency=concurrency,
    )

    total_points = len(temp_list) * len(top_p_list)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=40, style="magenta", complete_style="green"),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task(f"Running grid search over {total_points} hyperparameter points...", total=total_points)

        def on_step(current: int, total: int, msg: str) -> None:
            progress.update(task, completed=current, description=f"[bold cyan]{msg}[/bold cyan]")

        report = asyncio.run(tuner.tune(progress_cb=on_step))

    render_tuning_terminal_report(report, console=console)


@app.command(name="tui")
def tui_cmd(
    v1: str | None = typer.Argument(None, help="Optional initial path to baseline prompt (v1)"),
    v2: str | None = typer.Argument(None, help="Optional initial path to candidate prompt (v2)"),
    inputs: str | None = typer.Option(None, "--inputs", "-i", help="Path to test dataset (.jsonl)"),
    model: str = typer.Option("gpt-4o", "--model", "-m", help="Target model identifier"),
    mock: bool = typer.Option(True, "--mock/--live", help="Start in offline mock mode by default"),
) -> None:
    """Launch Interactive Split-Screen Terminal UI (TUI) Dashboard."""
    from promptdiff.cli.tui import launch_tui
    launch_tui(v1=v1, v2=v2, inputs=inputs, model=model, mock=mock)


@app.command(name="optimize")
def optimize_cmd(
    prompt: str = typer.Argument(..., help="Path to initial prompt template file or raw prompt string"),
    inputs: str | None = typer.Option(None, "--inputs", "-i", help="Path to test dataset (.jsonl, .yaml, .csv)"),
    output: str = typer.Option("system_v3_optimized.txt", "--output", "-o", help="Target output path for optimized prompt"),
    eval_metrics: str = typer.Option("json_validity,latency,cost,similarity,llm_judge", "--eval", "-e", help="Evaluation metrics"),
    model: str = typer.Option("gpt-4o", "--model", "-m", help="Target model being optimized"),
    meta_model: str = typer.Option("gpt-4o", "--meta-model", help="Meta-optimizer model (DSPy style)"),
    iterations: int = typer.Option(3, "--iterations", "-n", help="Max optimization reflection iterations"),
    mock: bool = typer.Option(False, "--mock", help="Use deterministic mock optimizer"),
) -> None:
    """Auto-Prompt Optimizer (DSPy style): Meta-LLM reflection on failed cases and judge criticism."""
    prompt_obj = load_prompt_file(prompt, version_name="initial", model=model)

    try:
        test_cases = load_dataset(inputs)
    except Exception as e:
        console.print(f"[bold red]Error loading dataset:[/bold red] {e}")
        raise typer.Exit(code=1)

    eval_list = get_evaluators([eval_metrics])

    optimizer = PromptOptimizer(
        prompt_version=prompt_obj,
        test_cases=test_cases,
        evaluators=eval_list,
        model_name=model,
        meta_model_name=meta_model,
        max_iterations=iterations,
        force_mock=mock,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=40, style="magenta", complete_style="green"),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task(f"Optimizing prompt across {len(test_cases)} test cases...", total=iterations)

        def on_step(current: int, total: int, msg: str) -> None:
            progress.update(task, completed=current, description=f"[bold cyan]{msg}[/bold cyan]")

        result = asyncio.run(optimizer.optimize(progress_cb=on_step))

    saved_path = optimizer.save_optimized_prompt(result.optimized_prompt, output)

    summary_table = Table(
        title="[bold yellow]🧠 Auto-Prompt Optimization Results (DSPy Style)[/bold yellow]",
        box=None,
        show_header=True,
        header_style="bold cyan",
    )
    summary_table.add_column("Metric", style="bold white")
    summary_table.add_column("Initial Prompt", style="cyan")
    summary_table.add_column("Optimized Prompt", style="green")
    summary_table.add_column("Improvement", style="bold green")

    init_pct = result.initial_pass_rate * 100.0
    final_pct = result.final_pass_rate * 100.0
    diff_pct = final_pct - init_pct

    summary_table.add_row("Pass Rate", f"{init_pct:.1f}%", f"{final_pct:.1f}%", f"+{diff_pct:.1f}%")
    summary_table.add_row("Iterations Run", "-", str(result.iterations), f"{result.iterations} round(s)")
    summary_table.add_row("Failures Fixed", "-", str(result.failed_cases_addressed), f"{result.failed_cases_addressed} cases")

    console.print()
    console.print(summary_table)

    opt_panel = Panel(
        f"[bold green]✨ Optimized Prompt Saved to:[/bold green] [cyan]{saved_path}[/cyan]\n\n"
        f"[bold white]Optimized Template Preview:[/bold white]\n"
        f"[dim]{result.optimized_prompt[:250]}...[/dim]",
        title="[bold cyan]Optimization Complete[/bold cyan]",
        border_style="green",
        padding=(1, 2),
    )
    console.print()
    console.print(opt_panel)
    console.print()


@app.command(name="ui")
def ui_cmd(
    report: str | None = typer.Option(None, "--report", "-r", help="Path to JSON report file to visualize"),
    port: int = typer.Option(8501, "--port", "-p", help="Streamlit web port"),
    host: str = typer.Option("localhost", "--host", help="Streamlit server host address"),
) -> None:
    """Launch Streamlit Interactive Web Dashboard."""
    from promptdiff.cli.dashboard import launch_dashboard
    launch_dashboard(port=port, host=host, report_path=report)


@app.command(name="dashboard")
def dashboard_cmd(
    report: str | None = typer.Option(None, "--report", "-r", help="Path to JSON report file to visualize"),
    port: int = typer.Option(8501, "--port", "-p", help="Streamlit web port"),
    host: str = typer.Option("localhost", "--host", help="Streamlit server host address"),
) -> None:
    """Launch Streamlit Interactive Web Dashboard (alias for `promptdiff ui`)."""
    from promptdiff.cli.dashboard import launch_dashboard
    launch_dashboard(port=port, host=host, report_path=report)


@app.command(name="arena")
def arena_cmd(
    prompts: str = typer.Option(..., "--prompts", "-p", help="Comma-separated paths to prompt templates (e.g. 'v1.txt,v2.txt,v3.txt')"),
    models: str = typer.Option("gpt-4o,claude-3-5-sonnet,gemini-2.0-flash", "--models", "-m", help="Comma-separated models to evaluate"),
    inputs: str | None = typer.Option(None, "--inputs", "-i", help="Path to test dataset (.jsonl, .yaml, .csv, .json)"),
    eval_metrics: str = typer.Option("json_validity,latency,cost,similarity,faithfulness,security", "--eval", "-e", help="Evaluation metrics"),
    mock: bool = typer.Option(False, "--mock", help="Use deterministic mock provider"),
    concurrency: int = typer.Option(6, "--concurrency", "-c", help="Concurrency limit"),
) -> None:
    """Multi-Model Arena: Evaluate N prompts and models simultaneously with leaderboard rankings."""
    prompt_list = [p.strip() for p in prompts.split(",") if p.strip()]
    model_list = [m.strip() for m in models.split(",") if m.strip()]

    if not prompt_list:
        console.print("[bold red]At least one prompt template must be specified.[/bold red]")
        raise typer.Exit(code=1)

    try:
        test_cases = load_dataset(inputs)
    except Exception as e:
        console.print(f"[bold red]Error loading dataset:[/bold red] {e}")
        raise typer.Exit(code=1)

    variants: dict[str, PromptVersion] = {}
    providers = {}

    if len(prompt_list) == 1 and len(model_list) > 1:
        base_path = prompt_list[0]
        for m in model_list:
            var_name = f"{m}"
            pv = load_prompt_file(base_path, version_name=var_name, model=m)
            variants[var_name] = pv
            providers[var_name] = get_provider(model_name=m, force_mock=mock)
    else:
        for idx, p_path in enumerate(prompt_list):
            m_name = model_list[idx % len(model_list)]
            var_name = f"var_{idx+1}_{Path(p_path).stem}" if Path(p_path).exists() else f"var_{idx+1}"
            pv = load_prompt_file(p_path, version_name=var_name, model=m_name)
            variants[var_name] = pv
            providers[var_name] = get_provider(model_name=m_name, force_mock=mock)

    eval_list = get_evaluators([eval_metrics])
    arena = ArenaRunner(
        variants=variants,
        providers=providers,
        evaluators=eval_list,
        concurrency=concurrency,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=40, style="magenta", complete_style="green"),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task(f"Running Multi-Model Arena on {len(test_cases)} test cases...", total=len(test_cases))

        def on_step(current: int, total: int) -> None:
            progress.update(task, completed=current)

        report = asyncio.run(arena.run(test_cases, progress_cb=on_step))

    render_arena_terminal_report(report, console=console)


@app.command(name="generate-tests")
def generate_tests_cmd(
    description: str | None = typer.Option(None, "--desc", "-d", help="Prompt description or natural language task spec"),
    prompt_file: str | None = typer.Option(None, "--prompt", "-p", help="Path to prompt template file to extract variables and context"),
    output: str = typer.Option("testcases.jsonl", "--output", "-o", help="Target output JSONL path"),
    count: int = typer.Option(50, "--count", "-n", help="Number of diverse test cases to generate"),
    model: str = typer.Option("gpt-4o", "--model", "-m", help="LLM to use for generation"),
    mock: bool = typer.Option(False, "--mock", help="Use deterministic mock generator without API keys"),
) -> None:
    """Generate 50+ diverse, edge-case synthetic test payloads using an LLM."""
    template_content = ""
    if prompt_file and Path(prompt_file).is_file():
        template_content = Path(prompt_file).read_text(encoding="utf-8")
        if not description:
            description = f"Task based on template in {prompt_file}"

    desc = description or "Customer support & classification prompt"

    generator = SyntheticTestGenerator(
        prompt_template=template_content,
        description=desc,
        model_name=model,
        force_mock=mock,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=40, style="yellow", complete_style="green"),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task(f"Generating {count} synthetic edge-case test payloads...", total=count)

        def on_progress(completed: int, total: int, msg: str) -> None:
            progress.update(task, completed=completed, description=f"[bold cyan]{msg}[/bold cyan]")

        cases = asyncio.run(generator.generate(count=count, progress_cb=on_progress))

    generator.save_to_jsonl(cases, output)

    table = Table(
        title=f"[bold green]✨ Generated {len(cases)} Test Cases -> {output}[/bold green]",
        box=None,
        show_header=True,
        header_style="bold yellow",
    )
    table.add_column("Case ID", style="bold cyan")
    table.add_column("Description", style="white")
    table.add_column("Tags / Category", style="magenta")
    table.add_column("Sample Variable Keys", style="dim")

    for tc in cases[:10]:
        table.add_row(
            tc.id,
            tc.description[:60] + ("..." if len(tc.description) > 60 else ""),
            ", ".join(tc.tags),
            ", ".join(tc.vars.keys()),
        )

    console.print()
    console.print(table)
    if len(cases) > 10:
        console.print(f"[dim]... and {len(cases) - 10} more test cases written to {output}[/dim]\n")


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

    from rich.columns import Columns

    from promptdiff.diff.text_diff import compute_word_diff
    from promptdiff.diff.visualizer import render_diff_text

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

    (prompts_dir / "system_v1.txt").write_text(
        "You are a customer support agent. Answer the user query politely and provide contact info.\nQuery: {{query}}",
        encoding="utf-8",
    )
    (prompts_dir / "system_v2.txt").write_text(
        "You are a customer support agent. Answer the user query concisely using bullet points.\nQuery: {{query}}",
        encoding="utf-8",
    )
    (target_dir / "testcases.jsonl").write_text(
        '{"id": "tc_1", "description": "Password reset inquiry", "vars": {"query": "How do I reset my password?", "context": "Password resets are in Settings > Security."}}\n'
        '{"id": "tc_2", "description": "Refund request", "vars": {"query": "I want a refund for my order #1234.", "context": "Refunds are processed within 30 days."}}\n'
        '{"id": "tc_3", "description": "API rate limits", "vars": {"query": "What are the rate limits on Tier 2?", "context": "Tier 2 rate limit is 1,000 RPM."}}\n',
        encoding="utf-8",
    )
    (target_dir / "promptdiff.yaml").write_text(
        "v1_prompt: prompts/system_v1.txt\n"
        "v2_prompt: prompts/system_v2.txt\n"
        "model: gpt-4o\n"
        "evaluators:\n"
        "  - json_validity\n"
        "  - latency\n"
        "  - cost\n"
        "  - similarity\n"
        "  - llm_judge\n"
        "  - faithfulness\n"
        "  - security\n"
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
