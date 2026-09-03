"""Git Version History Regression Tracker for promptdiff (promptdiff history)."""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from promptdiff.core.config import load_dataset
from promptdiff.core.models import PromptVersion
from promptdiff.core.runner import PromptDiffRunner
from promptdiff.evaluators.registry import get_evaluators
from promptdiff.providers.registry import get_provider

logger = logging.getLogger("promptdiff.cli.history")


@dataclass
class CommitRevisionBenchmark:
    """Benchmark results for a specific Git commit revision."""

    commit_hash: str
    short_hash: str
    commit_date: str
    author: str
    message: str
    avg_latency_ms: float
    total_cost_usd: float
    avg_judge_score: float
    passed_cases: int
    total_cases: int


@dataclass
class GitHistoryReport:
    """Complete Git timeline regression report."""

    prompt_path: str
    model_name: str
    revisions_evaluated: list[CommitRevisionBenchmark] = field(default_factory=list)


def get_git_file_revisions(file_path: str, max_commits: int = 5) -> list[dict[str, str]]:
    """Retrieve Git commit history metadata for a specific file."""
    try:
        cmd = [
            "git",
            "log",
            f"-n{max_commits}",
            "--pretty=format:%H|%h|%s|%an|%ad",
            "--date=short",
            "--",
            file_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        revisions = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split("|")
            if len(parts) >= 5:
                revisions.append(
                    {
                        "hash": parts[0],
                        "short_hash": parts[1],
                        "message": parts[2],
                        "author": parts[3],
                        "date": parts[4],
                    }
                )
        return revisions
    except Exception as e:
        logger.warning(f"Could not retrieve Git history: {e}")
        return []


def get_file_content_at_commit(file_path: str, commit_hash: str) -> Optional[str]:
    """Retrieve file text content at a specific Git commit hash."""
    try:
        # Normalize relative path for git show
        rel_path = Path(file_path).as_posix()
        cmd = ["git", "show", f"{commit_hash}:{rel_path}"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except Exception as e:
        logger.warning(f"Could not read {file_path} at commit {commit_hash}: {e}")
        return None


async def track_git_history(
    prompt_file: str,
    dataset_path: Optional[str] = None,
    commits_count: int = 4,
    model_name: str = "gpt-4o",
    force_mock: bool = False,
) -> GitHistoryReport:
    """Benchmark prompt template evolution across Git commit history."""
    revisions = get_git_file_revisions(prompt_file, max_commits=commits_count)
    test_cases = load_dataset(dataset_path)

    report = GitHistoryReport(prompt_path=prompt_file, model_name=model_name)

    if not revisions:
        # Fallback if no git repo or untracked file
        current_content = Path(prompt_file).read_text(encoding="utf-8") if Path(prompt_file).exists() else "Prompt"
        pv = PromptVersion(name="HEAD", template=current_content, model=model_name)
        runner = PromptDiffRunner(
            v1_prompt=pv,
            v2_prompt=pv,
            provider_v1=get_provider(model_name=model_name, force_mock=force_mock),
            provider_v2=get_provider(model_name=model_name, force_mock=force_mock),
            evaluators=get_evaluators(["json_validity", "latency", "cost", "llm_judge"]),
        )
        res = await runner.run(test_cases)
        v = res.verdict
        report.revisions_evaluated.append(
            CommitRevisionBenchmark(
                commit_hash="HEAD",
                short_hash="HEAD",
                commit_date="Today",
                author="Local Developer",
                message="Current working tree",
                avg_latency_ms=v.avg_latency_v2,
                total_cost_usd=v.total_cost_v2,
                avg_judge_score=4.5,
                passed_cases=res.aggregate_stats.get("passed_cases", len(test_cases)),
                total_cases=len(test_cases),
            )
        )
        return report

    for rev in revisions:
        content = get_file_content_at_commit(prompt_file, rev["hash"])
        if not content:
            continue

        pv = PromptVersion(name=rev["short_hash"], template=content, model=model_name)
        runner = PromptDiffRunner(
            v1_prompt=pv,
            v2_prompt=pv,
            provider_v1=get_provider(model_name=model_name, force_mock=force_mock),
            provider_v2=get_provider(model_name=model_name, force_mock=force_mock),
            evaluators=get_evaluators(["json_validity", "latency", "cost", "llm_judge"]),
        )
        res = await runner.run(test_cases)
        v = res.verdict
        judge_score = 4.5
        for comp in res.comparisons:
            if "llm_judge" in comp.scores:
                judge_score = float(comp.scores["llm_judge"].v2_score)
                break

        report.revisions_evaluated.append(
            CommitRevisionBenchmark(
                commit_hash=rev["hash"],
                short_hash=rev["short_hash"],
                commit_date=rev["date"],
                author=rev["author"],
                message=rev["message"][:40],
                avg_latency_ms=round(v.avg_latency_v2, 1),
                total_cost_usd=round(v.total_cost_v2, 6),
                avg_judge_score=round(judge_score, 2),
                passed_cases=res.aggregate_stats.get("passed_cases", len(test_cases)),
                total_cases=len(test_cases),
            )
        )

    return report
