"""Pre-Commit Git Hook Platform for PromptDiff.

Installs an automated `.git/hooks/pre-commit` script that statically analyzes
staged prompt template files with `promptdiff check`, preventing accidental commits
of broken templates, syntax errors, or unclosed variables.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

PRE_COMMIT_SCRIPT_CONTENT = """#!/bin/sh
# PromptDiff Automated Pre-Commit Hook
echo "⚡ Running PromptDiff Pre-Commit Gate..."

# Find all staged prompt files
STAGED_PROMPTS=$(git diff --cached --name-only --diff-filter=ACM | grep -E '\\.(prompt|txt|md)$' | grep -v 'README')

if [ -z "$STAGED_PROMPTS" ]; then
    exit 0
fi

echo "Scanning staged prompts: $STAGED_PROMPTS"
promptdiff check $STAGED_PROMPTS

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "❌ PromptDiff pre-commit check failed! Commit aborted."
    exit 1
fi

echo "✓ PromptDiff pre-commit check passed!"
exit 0
"""


class GitHookInstaller:
    """Installs and manages Git hooks for prompt quality enforcement."""

    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root)
        self.hooks_dir = self.repo_root / ".git" / "hooks"

    def install_pre_commit(self) -> str:
        """Install pre-commit hook script."""
        if not (self.repo_root / ".git").exists():
            raise RuntimeError(f"Directory '{self.repo_root}' is not a Git repository.")

        self.hooks_dir.mkdir(parents=True, exist_ok=True)
        hook_path = self.hooks_dir / "pre-commit"

        hook_path.write_text(PRE_COMMIT_SCRIPT_CONTENT, encoding="utf-8")

        # Make executable on Unix/macOS
        if os.name != "nt":
            curr_mode = hook_path.stat().st_mode
            hook_path.chmod(curr_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

        return str(hook_path.resolve())

    def is_installed(self) -> bool:
        """Check if PromptDiff pre-commit hook is active."""
        hook_path = self.hooks_dir / "pre-commit"
        if not hook_path.is_file():
            return False
        content = hook_path.read_text(encoding="utf-8", errors="ignore")
        return "PromptDiff" in content
