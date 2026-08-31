"""Test Suite for Ultimate Features: Fuzzer, Cache Simulator, Mutator, Statistics, Git History, and Bundle HTML."""

from __future__ import annotations

from pathlib import Path

import pytest

import promptdiff
from promptdiff.cli.history import track_git_history
from promptdiff.core.models import PromptVersion, TestCase
from promptdiff.core.statistics import (
    analyze_significance,
    bootstrap_ci,
    permutation_test_p_value,
)
from promptdiff.generators.mutator import DatasetMutator, inject_slang_and_abbreviations, inject_typos
from promptdiff.optimizer.cache_sim import PromptCacheSimulator
from promptdiff.reporters.bundle_html import generate_interactive_bundle_html
from promptdiff.security.fuzzer import JailbreakFuzzer


def test_statistics_bootstrap_and_p_value() -> None:
    """Test statistical significance bootstrap and permutation test."""
    v1_lats = [200.0, 210.0, 195.0, 205.0, 202.0, 215.0, 198.0, 204.0]
    v2_lats = [150.0, 155.0, 148.0, 152.0, 150.0, 153.0, 149.0, 151.0]

    ci_low, ci_high = bootstrap_ci([v2 - v1 for v1, v2 in zip(v1_lats, v2_lats, strict=False)])
    assert ci_low < 0.0
    assert ci_high < 0.0

    p_val = permutation_test_p_value(v1_lats, v2_lats)
    assert p_val < 0.05

    sig = analyze_significance("latency_ms", v1_lats, v2_lats)
    assert sig is not None
    assert sig.is_statistically_significant is True
    assert "Statistically Significant" in sig.verdict_text


def test_dataset_mutator() -> None:
    """Test synthetic test case mutation engine."""
    seed_cases = [
        TestCase(id="tc_1", description="Refund inquiry", vars={"query": "Please cancel my subscription and refund."}),
    ]

    typo_text = inject_typos("customer", typo_rate=0.5)
    assert len(typo_text) == 8

    slang_text = inject_slang_and_abbreviations("Please help me as soon as possible thank you")
    assert "pls" in slang_text
    assert "asap" in slang_text
    assert "thx" in slang_text

    mutator = DatasetMutator(seed_testcases=seed_cases, multiplier=4)
    mutated = mutator.generate_mutations()

    assert len(mutated) == 4
    assert any("mutated" in tc.tags for tc in mutated)


@pytest.mark.asyncio
async def test_jailbreak_fuzzer() -> None:
    """Test adversarial jailbreak red-teaming fuzzer."""
    pv = PromptVersion(
        name="secure_system_prompt",
        template="You are a secure customer support assistant. Help with query: {{query}}",
        model="gpt-4o",
    )

    fuzzer = JailbreakFuzzer(
        prompt_version=pv,
        model_name="gpt-4o",
        force_mock=True,
    )

    report = await fuzzer.run_fuzz()
    assert report.total_attacks > 0
    assert report.resilience_score_pct >= 0.0
    assert len(report.recommendations) > 0


def test_prompt_cache_simulator() -> None:
    """Test LLM prompt prefix caching simulator."""
    pv = PromptVersion(
        name="cache_test",
        template="User Query: {{query}}\n\nYou are a customer support agent. Obey rules 1 to 10.\nJSON Schema: ...",
        model="claude-3-5-sonnet",
    )

    sim = PromptCacheSimulator(prompt_version=pv, model_name="claude-3-5-sonnet", daily_volume=100_000)
    rep = sim.analyze_and_optimize()

    assert rep.optimized_cache_hit_rate_pct > rep.original_cache_hit_rate_pct
    assert rep.prefix_tokens_cached > 0
    assert "USER REQUEST INPUTS" in rep.optimized_template


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


def test_bundle_html_exporter(tmp_path: Path) -> None:
    """Test standalone interactive HTML bundle generation."""
    out_file = str(tmp_path / "bundle.html")
    report = promptdiff.compare(
        v1="Hello: {{query}}",
        v2="Hi: {{query}}",
        dataset=[{"id": "t1", "vars": {"query": "World"}}],
        mock=True,
    )
    bpath = generate_interactive_bundle_html(report, out_file)
    assert Path(bpath).exists()
    content = Path(bpath).read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in content
    assert "PromptDiff Interactive Regression Report" in content
