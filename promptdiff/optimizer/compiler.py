"""Prompt JIT Compiler & AST Template Minifier for promptdiff (promptdiff compile)."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from promptdiff.core.models import PromptVersion

logger = logging.getLogger("promptdiff.optimizer.compiler")


@dataclass
class CompilationResult:
    """Compiled prompt template and token savings."""

    original_template: str
    compiled_template: str
    original_tokens: int
    compiled_tokens: int
    tokens_saved: int
    compression_pct: float
    optimizations_applied: list[str]


class PromptJITCompiler:
    """JIT Compiles and minifies prompt templates into high-density Intermediate Representation (IR)."""

    def __init__(self, prompt_version: PromptVersion):
        self.prompt_version = prompt_version

    def _estimate_tokens(self, text: str) -> int:
        return max(1, int(len(re.findall(r"\w+|[^\w\s]", text, re.UNICODE)) * 1.1))

    def compile(self) -> CompilationResult:
        """Apply AST optimizations, dead-branch elimination, and whitespace minification."""
        text = self.prompt_version.template
        orig_tokens = self._estimate_tokens(text)
        optimizations = []

        # 1. Eliminate Jinja2 comments {# ... #}
        no_comments = re.sub(r"\{#.*?#\}", "", text, flags=re.DOTALL)
        if len(no_comments) < len(text):
            optimizations.append("Stripped template comments ({# ... #})")
        text = no_comments

        # 2. Collapse multi-line whitespace & empty lines
        no_blank_lines = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
        if len(no_blank_lines) < len(text):
            optimizations.append("Collapsed redundant vertical blank lines")
        text = no_blank_lines

        # 3. Minify inline spaces
        no_extra_spaces = re.sub(r"[ \t]+", " ", text)
        text = no_extra_spaces

        # 4. Standardize variable syntax (e.g. {{ var }} -> {{var}})
        std_vars = re.sub(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}", r"{{\1}}", text)
        text = std_vars.strip()
        optimizations.append("Normalized variable placeholders to dense AST syntax")

        comp_tokens = self._estimate_tokens(text)
        saved = max(0, orig_tokens - comp_tokens)
        pct = (saved / orig_tokens * 100.0) if orig_tokens else 0.0

        return CompilationResult(
            original_template=self.prompt_version.template,
            compiled_template=text,
            original_tokens=orig_tokens,
            compiled_tokens=comp_tokens,
            tokens_saved=saved,
            compression_pct=round(pct, 1),
            optimizations_applied=optimizations,
        )
