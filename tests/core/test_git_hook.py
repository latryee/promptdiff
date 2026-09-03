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
