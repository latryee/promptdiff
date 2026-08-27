"""Unit tests for promptdiff CLI interface."""

from pathlib import Path
from typer.testing import CliRunner
from promptdiff.cli.app import app

runner = CliRunner()


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "LLM Prompt & Output Regression Tester CLI" in result.output


def test_cli_pricing():
    result = runner.invoke(app, ["pricing", "gpt-4o"])
    assert result.exit_code == 0
    assert "gpt-4o" in result.output


def test_cli_init(tmp_path: Path):
    target_dir = tmp_path / "scaffold"
    result = runner.invoke(app, ["init", str(target_dir)])
    assert result.exit_code == 0
    assert (target_dir / "prompts" / "system_v1.txt").exists()
    assert (target_dir / "testcases.jsonl").exists()
    assert (target_dir / "promptdiff.yaml").exists()


def test_cli_static_diff(tmp_path: Path):
    f1 = tmp_path / "f1.txt"
    f2 = tmp_path / "f2.txt"
    f1.write_text("Hello prompt v1", encoding="utf-8")
    f2.write_text("Hello prompt v2", encoding="utf-8")

    result = runner.invoke(app, ["diff", str(f1), str(f2)])
    assert result.exit_code == 0


def test_cli_test_run_mock_passing(tmp_path: Path):
    f1 = tmp_path / "v1.txt"
    f2 = tmp_path / "v2.txt"
    f1.write_text("You are a support bot. Answer: {{query}}", encoding="utf-8")
    f2.write_text("You are a concise support bot. Answer in bullets: {{query}}", encoding="utf-8")

    dataset = tmp_path / "cases.jsonl"
    dataset.write_text('{"id": "tc1", "vars": {"query": "Hello"}}\n', encoding="utf-8")

    html_out = tmp_path / "report.html"
    md_out = tmp_path / "report.md"
    json_out = tmp_path / "report.json"

    result = runner.invoke(
        app,
        [
            "test",
            str(f1),
            str(f2),
            "--inputs",
            str(dataset),
            "--mock",
            "--eval",
            "json_validity,latency,cost,similarity",
            "--export-html",
            str(html_out),
            "--export-markdown",
            str(md_out),
            "--export-json",
            str(json_out),
        ],
    )

    assert result.exit_code == 0
    assert html_out.exists()
    assert md_out.exists()
    assert json_out.exists()


def test_cli_test_run_mock_failing_assertion(tmp_path: Path):
    f1 = tmp_path / "v1.txt"
    f2 = tmp_path / "v2.txt"
    f1.write_text("Short", encoding="utf-8")
    f2.write_text("A very long candidate response with lots of words and tokens", encoding="utf-8")

    # Asserting an impossible requirement that should fail
    result = runner.invoke(
        app,
        [
            "test",
            str(f1),
            str(f2),
            "--mock",
            "--assert",
            "similarity >= 0.999",
        ],
    )

    # Must exit with code 1 on regression failure
    assert result.exit_code == 1


def test_cli_cache_commands():
    res1 = runner.invoke(app, ["cache", "stats"])
    assert res1.exit_code == 0
    res2 = runner.invoke(app, ["cache", "clear"])
    assert res2.exit_code == 0
