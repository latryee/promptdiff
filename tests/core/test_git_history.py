"""Tests migrated to feature-focused test suite."""

from __future__ import annotations

import pytest

from promptdiff.cli.history import track_git_history


@pytest.mark.asyncio
async def test_git_history_tracker() -> None:
    """Test Git revision regression tracker."""
    rep = await track_git_history(
        prompt_file="prompts/system_v1.txt",
        dataset_path="testcases.jsonl",
        commits_count=2,
        force_mock=True,
    )
    assert len(rep.revisions_evaluated) > 0
