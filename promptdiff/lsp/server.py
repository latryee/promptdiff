"""VS Code & Cursor Language Server Protocol (LSP) & Diagnostic Bridge for promptdiff."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from promptdiff.pricing import get_model_pricing

logger = logging.getLogger("promptdiff.lsp")


@dataclass
class PromptDiagnostic:
    """Diagnostic annotation for prompt file."""

    line: int
    character: int
    severity: str  # ERROR, WARNING, INFORMATION, HINT
    message: str
    code: str


class PromptLanguageServer:
    """LSP and diagnostic analyzer for .txt and .jinja2 prompt files."""

    def __init__(self, model_name: str = "gpt-4o"):
        self.model_name = model_name

    def analyze_file(self, file_path: str) -> list[PromptDiagnostic]:
        """Inspect prompt file for syntax issues, unclosed braces, and cost estimation."""
        path = Path(file_path)
        if not path.exists():
            return []

        content = path.read_text(encoding="utf-8")
        lines = content.split("\n")
        diagnostics: list[PromptDiagnostic] = []

        pricing = get_model_pricing(self.model_name)
        total_tokens = max(1, int(len(re.findall(r"\w+|[^\w\s]", content, re.UNICODE)) * 1.1))
        cost_per_call = total_tokens * pricing.input_per_token

        # File header CodeLens info
        diagnostics.append(
            PromptDiagnostic(
                line=0,
                character=0,
                severity="INFORMATION",
                message=f"PromptDiff: ~{total_tokens} tokens (~${cost_per_call:.6f}/req on {self.model_name})",
                code="PROMPT_COST_INFO",
            )
        )

        for line_idx, line in enumerate(lines):
            # Check for unclosed variable braces e.g. {{query without }}
            if "{{" in line and "}}" not in line:
                diagnostics.append(
                    PromptDiagnostic(
                        line=line_idx,
                        character=line.find("{{"),
                        severity="ERROR",
                        message="Unclosed template variable '{{'. Missing closing '}}'.",
                        code="UNCLOSED_VARIABLE",
                    )
                )

            # Warn on redundant politeness fluff
            if re.search(r"\b(please kindly|as an ai|feel free to)\b", line, flags=re.IGNORECASE):
                diagnostics.append(
                    PromptDiagnostic(
                        line=line_idx,
                        character=0,
                        severity="HINT",
                        message="Redundant boilerplate detected. Run `promptdiff shrink` to prune tokens.",
                        code="TOKEN_OPTIMIZATION_HINT",
                    )
                )

        return diagnostics

    def print_diagnostics_cli(self, file_path: str) -> None:
        """Render LSP diagnostics to stdout."""
        diags = self.analyze_file(file_path)
        print(f"Diagnostics for {file_path}:")
        for d in diags:
            print(f"[{d.severity}] Line {d.line + 1}: {d.message} ({d.code})")
