"""Jupyter Notebook (.ipynb) & Google Colab Exporter for promptdiff (promptdiff notebook)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from promptdiff.core.models import DiffReport


class JupyterNotebookExporter:
    """Exports regression test experiments into interactive Jupyter Notebooks with charts."""

    def __init__(self, report: DiffReport):
        self.report = report

    def generate_notebook_dict(self) -> dict[str, Any]:
        """Construct standard Jupyter Notebook JSON format v4."""
        v = self.report.verdict

        md_intro = (
            f"# ⚡ PromptDiff Experiment Report: {self.report.v1_name} vs {self.report.v2_name}\n\n"
            f"- **Target Model**: `{self.report.model_v2}`\n"
            f"- **Total Test Cases**: `{self.report.total_cases}`\n"
            f"- **Quality Gate Passed**: `{'✅ YES' if v.passed else '❌ NO'}`\n"
            f"- **Cost Delta**: `{v.cost_delta_pct:+.1f}%`\n"
            f"- **Latency Delta**: `{v.latency_delta_pct:+.1f}%`\n"
        )

        py_setup = (
            "# Visualizing PromptDiff Results with Plotly\n"
            "import plotly.graph_objects as go\n"
            "import pandas as pd\n\n"
            "data = {\n"
            f"    'Version': ['{self.report.v1_name}', '{self.report.v2_name}'],\n"
            f"    'Cost_USD': [{v.total_cost_v1}, {v.total_cost_v2}],\n"
            f"    'Avg_Latency_ms': [{v.avg_latency_v1}, {v.avg_latency_v2}]\n"
            "}\n"
            "df = pd.DataFrame(data)\n"
            "fig = go.Figure(data=[\n"
            "    go.Bar(name='Cost ($)', x=df['Version'], y=df['Cost_USD']),\n"
            "])\n"
            "fig.update_layout(title='Token Cost Comparison')\n"
            "fig.show()\n"
        )

        cells = [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [line + "\n" for line in md_intro.split("\n")],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + "\n" for line in py_setup.split("\n")],
            },
        ]

        return {
            "cells": cells,
            "metadata": {
                "language_info": {"name": "python", "version": "3.11"},
            },
            "nbformat": 4,
            "nbformat_minor": 2,
        }

    def save_notebook(self, output_path: str = "experiment_report.ipynb") -> str:
        """Export notebook to file."""
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        nb_dict = self.generate_notebook_dict()
        target.write_text(json.dumps(nb_dict, indent=2), encoding="utf-8")
        return str(target.resolve())
