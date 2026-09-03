"""Tests migrated to feature-focused test suite."""

from __future__ import annotations

from promptdiff.lsp.extension_gen import ExtensionScaffolder


def test_extension_scaffolder(tmp_path) -> None:
    """Test editor extension generator for VS Code and Cursor."""
    scaffolder = ExtensionScaffolder()
    scaffolder.scaffold(str(tmp_path))
    assert (tmp_path / "package.json").exists()
    assert (tmp_path / ".cursorrules").exists()
