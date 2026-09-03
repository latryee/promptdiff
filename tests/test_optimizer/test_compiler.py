"""Tests migrated to feature-focused test suite."""

from __future__ import annotations

from promptdiff.core.models import PromptVersion
from promptdiff.optimizer.compiler import PromptJITCompiler


def test_prompt_jit_compiler() -> None:
    """Test prompt JIT compiler and AST minifier."""
    raw_template = (
        "{# Internal developer note #}\n\nYou are an AI assistant.\n\n\n\nPlease answer query: {{ user_query }}."
    )
    pv = PromptVersion(name="compiler_p", template=raw_template)
    compiler = PromptJITCompiler(prompt_version=pv)
    res = compiler.compile()
    assert "{#" not in res.compiled_template
    assert "{{user_query}}" in res.compiled_template
    assert res.tokens_saved >= 0
