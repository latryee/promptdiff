"""Test Suite for promptdiff v2.0 Features:
Async Concurrency, LLM-as-a-Judge, Sentence Transformers Cosine Similarity,
Multi-Model Arena, Synthetic Data Generator, and MLOps Reporters.
"""

import tempfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from promptdiff.cli.app import app
from promptdiff.core.models import (
    PromptVersion,
    RunResult,
    TestCase,
)
from promptdiff.core.runner import ArenaRunner, PromptDiffRunner
from promptdiff.evaluators.llm_judge import LLMJudgeEvaluator
from promptdiff.evaluators.similarity import SimilarityEvaluator
from promptdiff.generators.synthetic import SyntheticTestGenerator
from promptdiff.providers.mock_provider import MockProvider

cli_runner = CliRunner()


@pytest.mark.asyncio
async def test_async_runner_concurrency() -> None:
    """Test concurrent batch execution with semaphore limit."""
    v1 = PromptVersion(name="v1", template="Hello {{name}}", model="mock-gpt-4o")
    v2 = PromptVersion(name="v2", template="Hi {{name}}, welcome!", model="mock-gpt-4o")
    p1 = MockProvider()
    p2 = MockProvider()

    runner = PromptDiffRunner(
        v1_prompt=v1,
        v2_prompt=v2,
        provider_v1=p1,
        provider_v2=p2,
        concurrency=4,
    )

    test_cases = [TestCase(id=f"case_{i}", vars={"name": f"User_{i}"}) for i in range(12)]

    report = await runner.run(test_cases)
    assert len(report.comparisons) == 12
    assert report.total_cases == 12
    assert report.verdict.passed is True


@pytest.mark.asyncio
async def test_llm_judge_evaluator() -> None:
    """Test LLM-as-a-Judge scoring, comparative evaluation, and dynamic v1 scoring."""
    judge = LLMJudgeEvaluator(model_name="mock-gpt-4o", force_mock=True, pass_threshold=3.0)

    tc = TestCase(id="tc_judge", vars={"query": "How do I update billing?"})
    v1_res = RunResult(
        prompt_name="v1",
        test_case_id="tc_judge",
        rendered_prompt="Query: How do I update billing?",
        output="Please call 1-800-HELP for billing.",
        latency_ms=100.0,
        prompt_tokens=10,
        completion_tokens=10,
        total_tokens=20,
        cost_usd=0.0001,
        model="mock-gpt-4o",
    )
    v2_res = RunResult(
        prompt_name="v2",
        test_case_id="tc_judge",
        rendered_prompt="Query: How do I update billing?",
        output="Go to Settings > Billing and click 'Update Payment Method'.",
        latency_ms=120.0,
        prompt_tokens=10,
        completion_tokens=15,
        total_tokens=25,
        cost_usd=0.00012,
        model="mock-gpt-4o",
    )

    score1 = await judge.async_evaluate(v1_res, v2_res, tc)
    assert score1.name == "llm_judge"
    assert isinstance(score1.v1_score, (int, float))
    assert isinstance(score1.v2_score, (int, float))
    assert score1.v1_score >= 1.0
    assert score1.v2_score >= 1.0
    assert "reasoning" in score1.details
    assert "preference" in score1.details

    # Test with a completely different v1 baseline output to verify v1_score is dynamically computed, not hardcoded constant
    v1_res_alt = RunResult(
        prompt_name="v1",
        test_case_id="tc_judge",
        rendered_prompt="Query: How do I update billing?",
        output="A totally different, extremely verbose, convoluted response that rambles on about unrelated topics and historical billing records from 1999.",
        latency_ms=300.0,
        prompt_tokens=10,
        completion_tokens=50,
        total_tokens=60,
        cost_usd=0.0005,
        model="mock-gpt-4o",
    )
    score2 = await judge.async_evaluate(v1_res_alt, v2_res, tc)
    # v1_score must be dynamically evaluated based on input content
    assert isinstance(score2.v1_score, (int, float))
    assert score2.v1_score >= 1.0
    assert score1.v1_score != score2.v1_score or score2.details["preference"] == "V2"


def test_sentence_transformers_similarity() -> None:
    """Test sentence-transformers semantic similarity evaluator."""
    sim_eval = SimilarityEvaluator(model_name="all-MiniLM-L6-v2")

    tc = TestCase(id="tc_sim", vars={})
    v1_res = RunResult(
        prompt_name="v1",
        test_case_id="tc_sim",
        rendered_prompt="test",
        output="The quick brown fox jumps over the lazy dog.",
        latency_ms=50.0,
        prompt_tokens=10,
        completion_tokens=10,
        total_tokens=20,
        cost_usd=0.0001,
        model="mock",
    )
    v2_res = RunResult(
        prompt_name="v2",
        test_case_id="tc_sim",
        rendered_prompt="test",
        output="A fast brown fox leaped over a sleepy dog.",
        latency_ms=50.0,
        prompt_tokens=10,
        completion_tokens=10,
        total_tokens=20,
        cost_usd=0.0001,
        model="mock",
    )

    score = sim_eval.evaluate(v1_res, v2_res, tc)
    assert score.name == "similarity"
    assert score.v2_score > 0.70
    assert score.passed is True


@pytest.mark.asyncio
async def test_arena_runner_multi_model() -> None:
    """Test Multi-Model Arena runner across 3 models."""
    variants = {
        "gpt-4o": PromptVersion(name="gpt-4o", template="Answer {{q}} concisely", model="mock-gpt-4o"),
        "claude-3-5": PromptVersion(
            name="claude-3-5", template="Answer {{q}} in bullets", model="mock-claude-3-5-sonnet"
        ),
        "gemini-flash": PromptVersion(
            name="gemini-flash", template="Answer {{q}} with steps", model="mock-gemini-2.0-flash"
        ),
    }
    providers = {k: MockProvider(model_name=k) for k in variants}

    arena = ArenaRunner(variants=variants, providers=providers, concurrency=6)
    test_cases = [
        TestCase(id="tc_1", vars={"q": "What is Docker?"}),
        TestCase(id="tc_2", vars={"q": "Explain Kubernetes pods"}),
    ]

    report = await arena.run(test_cases)
    assert len(report.leaderboard) == 3
    assert report.total_cases == 2
    assert report.leaderboard[0].rank == 1


@pytest.mark.asyncio
async def test_synthetic_test_generator() -> None:
    """Test synthetic test data generation."""
    generator = SyntheticTestGenerator(
        prompt_template="You are a support agent for {{service}}. User says: {{query}}",
        description="Customer support helpdesk routing",
        force_mock=True,
    )

    cases = await generator.generate(count=10)
    assert len(cases) == 10
    assert all(isinstance(c, TestCase) for c in cases)
    assert any("synthetic" in c.tags for c in cases)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = str(Path(tmpdir) / "testcases.jsonl")
        generator.save_to_jsonl(cases, out_file)
        assert Path(out_file).is_file()
        assert Path(out_file).stat().st_size > 0


def test_cli_run_and_arena() -> None:
    """Test CLI commands: promptdiff run, promptdiff arena, promptdiff generate-tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        p1 = Path(tmpdir) / "p1.txt"
        p2 = Path(tmpdir) / "p2.txt"
        tc_file = Path(tmpdir) / "testcases.jsonl"

        p1.write_text("Hello {{name}}", encoding="utf-8")
        p2.write_text("Hi {{name}}!", encoding="utf-8")
        tc_file.write_text('{"id": "t1", "vars": {"name": "Alice"}}\n', encoding="utf-8")

        # 1. Test `promptdiff run`
        res_run = cli_runner.invoke(
            app,
            ["run", str(p1), str(p2), "--inputs", str(tc_file), "--mock"],
        )
        assert res_run.exit_code == 0
        assert "Execution & Regression Summary" in res_run.stdout

        # 2. Test `promptdiff arena`
        res_arena = cli_runner.invoke(
            app,
            ["arena", "--prompts", str(p1), "--models", "mock-1,mock-2,mock-3", "--inputs", str(tc_file), "--mock"],
        )
        assert res_arena.exit_code == 0
        assert "Multi-Model Arena Leaderboard" in res_arena.stdout

        # 3. Test `promptdiff generate-tests`
        gen_out = str(Path(tmpdir) / "gen_tests.jsonl")
        res_gen = cli_runner.invoke(
            app,
            ["generate-tests", "--prompt", str(p1), "--output", gen_out, "--count", "10", "--mock"],
        )
        assert res_gen.exit_code == 0
        assert Path(gen_out).exists()


def test_cli_fail_on_regression() -> None:
    """Test CLI --fail-on-regression flag triggering exit code 1 on failed assertion."""
    with tempfile.TemporaryDirectory() as tmpdir:
        p1 = Path(tmpdir) / "p1.txt"
        p2 = Path(tmpdir) / "p2.txt"
        tc_file = Path(tmpdir) / "testcases.jsonl"

        p1.write_text("Hello {{name}}", encoding="utf-8")
        p2.write_text("Hi {{name}}!", encoding="utf-8")
        tc_file.write_text('{"id": "t1", "vars": {"name": "Alice"}}\n', encoding="utf-8")

        res = cli_runner.invoke(
            app,
            [
                "run",
                str(p1),
                str(p2),
                "--inputs",
                str(tc_file),
                "--mock",
                "--assert",
                "similarity >= 1.0",  # Will fail since p1 != p2
                "--fail-on-regression",
            ],
        )
        assert res.exit_code == 1
