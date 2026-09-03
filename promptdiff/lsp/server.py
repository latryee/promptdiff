"""VS Code & Cursor Language Server Protocol (LSP) & Diagnostic Bridge for promptdiff."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from promptdiff.pricing import get_model_pricing

logger = logging.getLogger("promptdiff.lsp")

DEFAULT_LINT_CONFIG: dict[str, Any] = {
    "rules": {
        "ambiguous_instructions": {
            "enabled": True,
            "severity": "WARNING",
            "patterns": [
                r"do your best",
                r"try to",
                r"be creative",
                r"give a good answer",
                r"as you see fit",
                r"make it nice",
            ],
        },
        "contradictory_constraints": {
            "enabled": True,
            "severity": "WARNING",
            "pairs": [
                {"first": r"\b(concise|brief|short|terse)\b", "second": r"\b(detailed|comprehensive|exhaustive|in-depth)\b"},
                {"first": r"\b(json only|only json)\b", "second": r"\b(explain your reasoning|conversational)\b"},
            ],
        },
        "missing_few_shot": {
            "enabled": True,
            "severity": "WARNING",
            "triggers": [
                r"\bfew-shot\b",
                r"\bexamples? below\b",
                r"\blike the following examples?\b",
            ],
        },
    }
}


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

    def __init__(self, model_name: str = "gpt-4o", config_path: str | Path | None = None):
        self.model_name = model_name
        self.config = self._load_lint_config(config_path)

    def _load_lint_config(self, config_path: str | Path | None) -> dict[str, Any]:
        """Load .promptdifflintrc.yaml configuration if present."""
        path_to_try = Path(config_path) if config_path else Path(".promptdifflintrc.yaml")
        if path_to_try.is_file():
            try:
                loaded = yaml.safe_load(path_to_try.read_text(encoding="utf-8"))
                if isinstance(loaded, dict) and "rules" in loaded:
                    return loaded
            except Exception as e:
                logger.warning(f"Failed to parse lint config at {path_to_try}: {e}")
        return DEFAULT_LINT_CONFIG

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

        rules_cfg = self.config.get("rules", {})

        # Rule 1: Syntax & variables check
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

            # Rule 2: Ambiguous instruction check
            ambig_cfg = rules_cfg.get("ambiguous_instructions", {})
            if ambig_cfg.get("enabled", True):
                patterns = ambig_cfg.get("patterns", [])
                for pat in patterns:
                    m = re.search(pat, line, flags=re.IGNORECASE)
                    if m:
                        diagnostics.append(
                            PromptDiagnostic(
                                line=line_idx,
                                character=m.start(),
                                severity=ambig_cfg.get("severity", "WARNING"),
                                message=f"Ambiguous instruction detected: '{m.group(0)}'. Use precise, deterministic directives.",
                                code="AMBIGUOUS_INSTRUCTION",
                            )
                        )

        # Rule 3: Contradictory constraints check
        contra_cfg = rules_cfg.get("contradictory_constraints", {})
        if contra_cfg.get("enabled", True):
            pairs = contra_cfg.get("pairs", [])
            for pair in pairs:
                p1 = pair.get("first", "")
                p2 = pair.get("second", "")
                m1 = re.search(p1, content, flags=re.IGNORECASE)
                m2 = re.search(p2, content, flags=re.IGNORECASE)
                if m1 and m2:
                    diagnostics.append(
                        PromptDiagnostic(
                            line=0,
                            character=0,
                            severity=contra_cfg.get("severity", "WARNING"),
                            message=f"Contradictory constraints detected: '{m1.group(0)}' conflicts with '{m2.group(0)}'.",
                            code="CONTRADICTORY_CONSTRAINTS",
                        )
                    )

        # Rule 4: Missing few-shot examples check
        fewshot_cfg = rules_cfg.get("missing_few_shot", {})
        if fewshot_cfg.get("enabled", True):
            triggers = fewshot_cfg.get("triggers", [])
            has_trigger = any(re.search(t, content, flags=re.IGNORECASE) for t in triggers)
            if has_trigger:
                has_examples = bool(
                    re.search(
                        r"(example\s*[0-9#:]|input\s*:.*?output\s*:|user\s*:.*?assistant\s*:|\{\s*\"input\")",
                        content,
                        flags=re.IGNORECASE | re.DOTALL,
                    )
                )
                if not has_examples:
                    diagnostics.append(
                        PromptDiagnostic(
                            line=0,
                            character=0,
                            severity=fewshot_cfg.get("severity", "WARNING"),
                            message="Prompt references few-shot examples but contains no concrete example demonstrations.",
                            code="MISSING_FEW_SHOT",
                        )
                    )

        return diagnostics

    def print_diagnostics_cli(self, file_path: str) -> None:
        """Render LSP diagnostics to stdout."""
        diags = self.analyze_file(file_path)
        print(f"Diagnostics for {file_path}:")
        for d in diags:
            print(f"[{d.severity}] Line {d.line + 1}: {d.message} ({d.code})")
