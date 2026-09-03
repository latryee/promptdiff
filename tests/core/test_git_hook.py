"""Tests migrated to feature-focused test suite."""

from __future__ import annotations

from pathlib import Path

from promptdiff.cli.hooks import GitHookInstaller


def test_git_hook_installer(tmp_path: Path) -> None:
    """Test installing and verifying the pre-commit Git hook."""
    git_dir = tmp_path / ".git"
    git_dir.mkdir()

    installer = GitHookInstaller(repo_root=str(tmp_path))
    assert installer.is_installed() is False

    path = installer.install_pre_commit()
    assert Path(path).exists()
    assert installer.is_installed() is True
    content = Path(path).read_text(encoding="utf-8")
    assert "promptdiff check" in content
    assert "ruff check --fix" in content
    assert "mypy promptdiff" in content


def test_git_hook_installer_merge_existing(tmp_path: Path) -> None:
    """Verify installer merges cleanly into existing custom user pre-commit hooks."""
    hooks_dir = tmp_path / ".git" / "hooks"
    hooks_dir.mkdir(parents=True)
    existing_hook = hooks_dir / "pre-commit"
    existing_hook.write_text("#!/bin/sh\necho 'Running user custom linter'\nexit 0\n", encoding="utf-8")

    installer = GitHookInstaller(repo_root=str(tmp_path))
    installer.install_pre_commit(force=False)

    merged_content = existing_hook.read_text(encoding="utf-8")
    assert "Running user custom linter" in merged_content
    assert "⚡ Running PromptDiff Pre-Commit Gate..." in merged_content
    assert installer.is_installed() is True
