"""Test suite for Platform capabilities: Git Hook Installer, SQLite Telemetry Database,

and Studio SSE Streaming.
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from promptdiff.cli.app import app
from promptdiff.cli.hooks import GitHookInstaller
from promptdiff.core.db import TelemetryDatabase
from promptdiff.core.models import (
    ComparisonResult,
    DiffReport,
    EvaluatorScore,
    RegressionVerdict,
    RunResult,
    TestCase,
)


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


def test_telemetry_database(tmp_path: Path) -> None:
    """Test recording runs and querying historical telemetry from SQLite."""
    db_file = tmp_path / "telemetry_test.db"
    db = TelemetryDatabase(db_path=str(db_file))

    tc = TestCase(id="tc_billing_1", vars={"query": "Invoice error"})
    r1 = RunResult(
        prompt_name="v1",
        test_case_id="tc_billing_1",
        rendered_prompt="v1 prompt",
        output="output 1",
        latency_ms=120.0,
        prompt_tokens=10,
        completion_tokens=15,
        total_tokens=25,
        cost_usd=0.0012,
        model="gpt-4o",
    )
    r2 = RunResult(
        prompt_name="v2",
        test_case_id="tc_billing_1",
        rendered_prompt="v2 prompt",
        output="output 2",
        latency_ms=85.0,
        prompt_tokens=10,
        completion_tokens=10,
        total_tokens=20,
        cost_usd=0.0009,
        model="gpt-4o",
    )
    score_fail = EvaluatorScore(
        name="latency",
        passed=False,
        v1_score=120.0,
        v2_score=85.0,
        delta=-35.0,
        delta_pct=-29.1,
        message="Failed assertion",
    )
    comp = ComparisonResult(
        test_case=tc,
        v1_result=r1,
        v2_result=r2,
        scores={"latency": score_fail},
    )
    verdict = RegressionVerdict(
        passed=False,
        status="REGRESSION_DETECTED",
        failed_assertions=["latency < 50ms"],
        total_cost_v1=0.0012,
        total_cost_v2=0.0009,
        cost_delta_pct=-25.0,
        avg_latency_v1=120.0,
        avg_latency_v2=85.0,
        latency_delta_pct=-29.1,
    )
    report = DiffReport(
        run_id="run_test_001",
        v1_name="prompt_v1",
        v2_name="prompt_v2",
        model_v1="gpt-4o",
        model_v2="gpt-4o",
        comparisons=[comp],
        verdict=verdict,
        evaluators=["latency"],
        total_cases=1,
    )

    db.record_run(report)

    # Query recent runs
    recent = db.get_recent_runs(limit=5)
    assert len(recent) == 1
    assert recent[0].run_id == "run_test_001"
    assert recent[0].passed is False
    assert recent[0].cost_delta_pct == -25.0

    # Query failure hotspots
    hotspots = db.get_failure_hotspots(limit=5)
    assert len(hotspots) == 1
    assert hotspots[0].test_case_id == "tc_billing_1"
    assert hotspots[0].failure_count == 1


def test_cli_db_commands(tmp_path: Path) -> None:
    """Test CLI commands: promptdiff db stats and promptdiff db hotspots."""
    runner = CliRunner()

    res_stats = runner.invoke(app, ["db", "stats"])
    assert res_stats.exit_code == 0

    res_hotspots = runner.invoke(app, ["db", "hotspots"])
    assert res_hotspots.exit_code == 0


def test_cli_install_hook(tmp_path: Path) -> None:
    """Test CLI command: promptdiff install-hook."""
    runner = CliRunner()
    git_dir = tmp_path / ".git"
    git_dir.mkdir()

    res = runner.invoke(app, ["install-hook", "--dir", str(tmp_path)])
    assert res.exit_code == 0
    assert "Successfully installed" in res.stdout
