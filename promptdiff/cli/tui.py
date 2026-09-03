"""Interactive Terminal UI (TUI) Studio for promptdiff built with Textual."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from rich.panel import Panel
from rich.text import Text

try:
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical, VerticalScroll
    from textual.widgets import (
        Button,
        Checkbox,
        DataTable,
        Footer,
        Header,
        Input,
        Label,
        ProgressBar,
        RichLog,
        Static,
        TabbedContent,
        TabPane,
        TextArea,
    )

    TEXTUAL_INSTALLED = True
except ImportError:
    TEXTUAL_INSTALLED = False
    App = object  # type: ignore[misc,assignment]

from promptdiff.core.config import load_dataset
from promptdiff.core.models import PromptVersion
from promptdiff.core.runner import PromptDiffRunner
from promptdiff.diff.text_diff import compute_word_diff
from promptdiff.diff.visualizer import render_diff_text
from promptdiff.pricing import calculate_forecast
from promptdiff.providers.registry import get_provider

DEFAULT_V1 = """You are a customer support agent. Answer the user query politely and provide contact info.

Query: {{query}}
Context: {{context}}"""

DEFAULT_V2 = """You are a concise customer support agent. Answer the user query directly using bullet points. Never hallucinate outside context.

Query: {{query}}
Context: {{context}}"""


class PromptDiffTUI(App[None]):
    """Interactive split-screen Textual dashboard for prompt comparison and evaluation."""

    CSS = """
    Screen {
        background: #0b0f19;
        color: #f8fafc;
    }
    Header {
        background: #1e293b;
        color: #38bdf8;
        dock: top;
        text-style: bold;
    }
    Footer {
        background: #1e293b;
        color: #94a3b8;
    }
    .prompt-pane {
        width: 1fr;
        height: 14;
        border: round #334155;
        padding: 0 1;
        margin: 0 1;
        background: #0f172a;
    }
    .pane-title {
        text-style: bold;
        color: #38bdf8;
        padding-bottom: 1;
    }
    .candidate-title {
        color: #c084fc;
    }
    .controls-row {
        height: auto;
        padding: 1;
        background: #1e293b;
        border-top: solid #334155;
        border-bottom: solid #334155;
        align: center middle;
    }
    .control-input {
        width: 32;
        margin-right: 1;
    }
    .btn-run {
        background: #059669;
        color: white;
        text-style: bold;
        margin-right: 1;
    }
    .btn-run:hover {
        background: #10b981;
    }
    .btn-clear {
        background: #475569;
        color: white;
    }
    TabbedContent {
        height: 1fr;
        margin: 1;
    }
    DataTable {
        height: 100%;
        background: #0f172a;
    }
    RichLog {
        height: 100%;
        background: #0f172a;
        padding: 1;
    }
    .kpi-panel {
        padding: 2;
        background: #0f172a;
        border: round #334155;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit Studio", show=True),
        Binding("r", "run_diff", "Run Eval", show=True),
        Binding("c", "clear_results", "Clear", show=True),
    ]

    def __init__(
        self,
        v1_path: Optional[str] = None,
        v2_path: Optional[str] = None,
        dataset_path: Optional[str] = None,
        model: str = "gpt-4o",
        mock: bool = True,
    ):
        super().__init__()
        self.v1_initial = (
            Path(v1_path).read_text(encoding="utf-8") if v1_path and Path(v1_path).is_file() else DEFAULT_V1
        )
        self.v2_initial = (
            Path(v2_path).read_text(encoding="utf-8") if v2_path and Path(v2_path).is_file() else DEFAULT_V2
        )
        self.dataset_initial = dataset_path or "testcases.jsonl"
        self.model_name = model
        self.mock_mode = mock

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with Horizontal():
            # Left: Baseline (v1)
            with Vertical(classes="prompt-pane"):
                yield Label("🔹 Baseline Prompt (v1)", classes="pane-title")
                yield TextArea(self.v1_initial, id="txt_v1")

            # Right: Candidate (v2)
            with Vertical(classes="prompt-pane"):
                yield Label("⚡ Candidate Prompt (v2)", classes="pane-title candidate-title")
                yield TextArea(self.v2_initial, id="txt_v2")

        # Middle Toolbar
        with Horizontal(classes="controls-row"):
            yield Label("Dataset: ", classes="pane-title")
            yield Input(
                value=self.dataset_initial, placeholder="testcases.jsonl", id="inp_dataset", classes="control-input"
            )
            yield Checkbox("Offline Mock Mode", value=self.mock_mode, id="chk_mock")
            yield Button("▶ Run Evaluation (R)", id="btn_run", classes="btn-run")
            yield Button("🧹 Clear (C)", id="btn_clear", classes="btn-clear")
            yield ProgressBar(id="prog_bar", show_eta=False, total=100)

        # Bottom Tabbed Views
        with TabbedContent():
            with TabPane("🔍 Side-by-Side Diff", id="tab_diff"):
                yield RichLog(id="log_diff", highlight=True, markup=True)

            with TabPane("📊 Evaluator Score Table", id="tab_scores"):
                yield DataTable(id="table_scores")

            with TabPane("📈 KPI & Cost Forecast", id="tab_kpis"):
                yield VerticalScroll(Static(id="lbl_kpis", classes="kpi-panel"))

        yield Footer()

    def on_mount(self) -> None:
        self.title = "PromptDiff Interactive TUI Studio"
        self.sub_title = "v3.0 Realtime Regression Tester"

        # Init DataTable
        table = self.query_one("#table_scores", DataTable)
        table.cursor_type = "row"
        table.add_columns("Case ID", "Similarity", "Judge Score", "Faithfulness", "Relevance", "Security", "Status")

        # Initial static diff render
        self.action_render_static_diff()

    def action_render_static_diff(self) -> None:
        """Render word diff between current text areas in the diff log."""
        txt1 = self.query_one("#txt_v1", TextArea).text
        txt2 = self.query_one("#txt_v2", TextArea).text
        log = self.query_one("#log_diff", RichLog)
        log.clear()

        chunks = compute_word_diff(txt1, txt2)
        r1 = render_diff_text(chunks, side="v1")
        r2 = render_diff_text(chunks, side="v2")

        log.write(Panel(r1, title="[cyan]Baseline (v1)[/cyan]", border_style="cyan"))
        log.write(Panel(r2, title="[magenta]Candidate (v2)[/magenta]", border_style="magenta"))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn_run":
            self.action_run_diff()
        elif event.button.id == "btn_clear":
            self.action_clear_results()

    def action_clear_results(self) -> None:
        table = self.query_one("#table_scores", DataTable)
        table.clear()
        lbl = self.query_one("#lbl_kpis", Static)
        lbl.update("Evaluation cleared.")
        self.action_render_static_diff()

    def action_run_diff(self) -> None:
        """Trigger async evaluation worker."""
        self.run_worker(self._async_execute_eval(), exclusive=True)

    async def _async_execute_eval(self) -> None:
        btn = self.query_one("#btn_run", Button)
        p_bar = self.query_one("#prog_bar", ProgressBar)
        table = self.query_one("#table_scores", DataTable)
        kpi_lbl = self.query_one("#lbl_kpis", Static)

        btn.disabled = True
        btn.label = "Running..."
        p_bar.update(progress=10)

        t1 = self.query_one("#txt_v1", TextArea).text
        t2 = self.query_one("#txt_v2", TextArea).text
        ds_path = self.query_one("#inp_dataset", Input).value
        is_mock = self.query_one("#chk_mock", Checkbox).value

        # 1. Update text diff preview
        self.action_render_static_diff()

        # 2. Load dataset
        try:
            test_cases = load_dataset(ds_path if Path(ds_path).exists() else None)
        except Exception as e:
            kpi_lbl.update(f"[bold red]Error loading dataset: {e}[/bold red]")
            btn.disabled = False
            btn.label = "▶ Run Evaluation (R)"
            return

        p_bar.update(progress=30)

        # 3. Setup runners & evaluators
        p1 = PromptVersion(name="v1", template=t1, model=self.model_name)
        p2 = PromptVersion(name="v2", template=t2, model=self.model_name)
        prov1 = get_provider(model_name=self.model_name, force_mock=is_mock)
        prov2 = get_provider(model_name=self.model_name, force_mock=is_mock)

        from promptdiff.evaluators.answer_relevance import AnswerRelevanceEvaluator
        from promptdiff.evaluators.faithfulness import FaithfulnessEvaluator
        from promptdiff.evaluators.llm_judge import LLMJudgeEvaluator
        from promptdiff.evaluators.security import SecurityEvaluator
        from promptdiff.evaluators.similarity import SimilarityEvaluator

        eval_list = [
            SimilarityEvaluator(),
            LLMJudgeEvaluator(force_mock=is_mock),
            FaithfulnessEvaluator(force_mock=is_mock),
            AnswerRelevanceEvaluator(force_mock=is_mock),
            SecurityEvaluator(),
        ]

        runner = PromptDiffRunner(
            v1_prompt=p1,
            v2_prompt=p2,
            provider_v1=prov1,
            provider_v2=prov2,
            evaluators=eval_list,
            concurrency=4,
        )

        p_bar.update(progress=50)

        report = await runner.run(test_cases)
        p_bar.update(progress=90)

        # 4. Populate DataTable
        table.clear()
        for comp in report.comparisons:
            tc = comp.test_case
            sc = comp.scores

            sim_score = sc.get("similarity")
            sim_val = (
                f"{float(sim_score.v2_score) * 100:.0f}%"
                if sim_score and isinstance(sim_score.v2_score, (int, float))
                else "-"
            )

            judge_score = sc.get("llm_judge")
            judge_val = (
                f"{float(judge_score.v2_score):.1f}/5.0"
                if judge_score and isinstance(judge_score.v2_score, (int, float))
                else "-"
            )

            faith_score = sc.get("faithfulness")
            faith_val = (
                f"{float(faith_score.v2_score) * 100:.0f}%"
                if faith_score and isinstance(faith_score.v2_score, (int, float))
                else "-"
            )

            rel_score = sc.get("answer_relevance")
            rel_val = (
                f"{float(rel_score.v2_score) * 100:.0f}%"
                if rel_score and isinstance(rel_score.v2_score, (int, float))
                else "-"
            )

            sec_score = sc.get("security")
            sec_val = "Clean" if (sec_score and sec_score.passed) else "Risk"
            all_pass = "PASS" if all(s.passed for s in sc.values()) else "FAIL"

            table.add_row(
                tc.id,
                sim_val,
                judge_val,
                faith_val,
                rel_val,
                sec_val,
                Text(all_pass, style="bold green" if all_pass == "PASS" else "bold red"),
            )

        # 5. Populate KPI summary & Cost Forecast
        v = report.verdict
        fc = calculate_forecast(v.total_cost_v1, v.total_cost_v2, report.total_cases, 100_000)

        kpi_text = f"""
[bold cyan]⚡ PROMPTDIFF REGRESSION VERDICT:[/bold cyan] [bold {"green" if v.passed else "red"}]{v.status}[/bold {"green" if v.passed else "red"}]

[bold white]Performance Summary:[/bold white]
- [bold]Total Token Cost:[/bold] ${v.total_cost_v1:.6f} ➔ ${v.total_cost_v2:.6f} ([bold {"green" if v.cost_delta_pct <= 0 else "red"}]{v.cost_delta_pct:+.1f}%[/bold {"green" if v.cost_delta_pct <= 0 else "red"}])
- [bold]Avg Latency:[/bold] {v.avg_latency_v1:.1f}ms ➔ {v.avg_latency_v2:.1f}ms ([bold {"green" if v.latency_delta_pct <= 0 else "red"}]{v.latency_delta_pct:+.1f}%[/bold {"green" if v.latency_delta_pct <= 0 else "red"}])
- [bold]Test Cases Passed:[/bold] {report.aggregate_stats.get("passed_cases", len(test_cases))} / {len(test_cases)}

[bold yellow]💰 Projected Production Cost Impact (100,000 reqs/day):[/bold yellow]
- [bold]Baseline Monthly Spend:[/bold] ${fc.v1_monthly_cost:,.2f}
- [bold]Candidate Monthly Spend:[/bold] ${fc.v2_monthly_cost:,.2f}
- [bold green]{fc.summary_text}[/bold green]
"""
        kpi_lbl.update(kpi_text.strip())

        p_bar.update(progress=100)
        btn.disabled = False
        btn.label = "▶ Run Evaluation (R)"


def launch_tui(
    v1: Optional[str] = None,
    v2: Optional[str] = None,
    inputs: Optional[str] = None,
    model: str = "gpt-4o",
    mock: bool = True,
) -> None:
    """Entrypoint to launch Textual TUI Application."""
    if not TEXTUAL_INSTALLED:
        import sys

        print(
            "[!] Error: Textual is not installed. Install with `pip install promptdiff[tui]` to use the interactive Terminal UI.",
            file=sys.stderr,
        )
        sys.exit(1)

    app = PromptDiffTUI(
        v1_path=v1,
        v2_path=v2,
        dataset_path=inputs,
        model=model,
        mock=mock,
    )
    app.run()
