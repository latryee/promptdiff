"""Tests migrated to feature-focused test suite."""

from __future__ import annotations

from pathlib import Path

from promptdiff.lsp.server import PromptLanguageServer


def test_lsp_server(tmp_path: Path) -> None:
    """Test LSP diagnostics server."""
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("Hello {{unclosed_var\nPlease kindly help me.", encoding="utf-8")

    server = PromptLanguageServer()
    diags = server.analyze_file(str(prompt_file))
    assert len(diags) >= 2
    assert any(d.code == "UNCLOSED_VARIABLE" for d in diags)


def test_prompt_linter_advanced_rules(tmp_path: Path) -> None:
    """Test ambiguous instructions, contradictory constraints, and missing few-shot rules."""
    prompt_file = tmp_path / "bad_prompt.txt"
    prompt_file.write_text(
        "You are an assistant. Please do your best to answer.\n"
        "Be concise and brief, but provide an exhaustive and detailed explanation.\n"
        "Follow the few-shot examples below:\n",
        encoding="utf-8",
    )

    server = PromptLanguageServer()
    diags = server.analyze_file(str(prompt_file))
    codes = {d.code for d in diags}

    assert "AMBIGUOUS_INSTRUCTION" in codes
    assert "CONTRADICTORY_CONSTRAINTS" in codes
    assert "MISSING_FEW_SHOT" in codes


def test_lsp_hover_and_inline_diagnostics(tmp_path: Path) -> None:
    """Test LSP hover tooltips and inline diagnostic range structures."""
    prompt_file = tmp_path / "hover_prompt.txt"
    prompt_file.write_text("Line 1: System prompt\nLine 2: Answer {{query with no closing", encoding="utf-8")

    server = PromptLanguageServer(model_name="gpt-4o")
    hover = server.get_hover(str(prompt_file), line=0, character=5)

    assert hover.line == 0
    assert hover.character == 5
    assert hover.tokens > 0
    assert hover.estimated_cost_usd > 0
    assert "PromptDiff Telemetry" in hover.markdown_content

    inline = server.get_inline_diagnostics(str(prompt_file))
    assert len(inline) >= 1
    diag = next(d for d in inline if d["code"] == "UNCLOSED_VARIABLE")
    assert "range" in diag
    assert diag["range"]["start"]["line"] == 1
    assert diag["severity"] == 1  # ERROR
