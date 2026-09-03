"""Tests migrated to feature-focused test suite."""

from __future__ import annotations

from pathlib import Path

from promptdiff.core.db import TelemetryDatabase
from promptdiff.core.models import (
    ComparisonResult,
    DiffReport,
    EvaluatorScore,
    RegressionVerdict,
    RunResult,
    TestCase,
)


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


def test_telemetry_database_concurrent_writes(tmp_path: Path) -> None:
    """Verify SQLite WAL mode allows concurrent writes without database lock errors."""
    from concurrent.futures import ThreadPoolExecutor

    db_file = tmp_path / "concurrent_telemetry.db"
    db = TelemetryDatabase(db_path=str(db_file))

    # Verify WAL mode is active
    with db._get_connection() as conn:
        cursor = conn.execute("PRAGMA journal_mode")
        mode = cursor.fetchone()[0]
        assert str(mode).lower() == "wal"

    def write_worker(idx: int) -> str:
        worker_db = TelemetryDatabase(db_path=str(db_file))
        tc = TestCase(id=f"tc_{idx}", vars={"query": f"Query {idx}"})
        r1 = RunResult(
            prompt_name="v1",
            test_case_id=f"tc_{idx}",
            rendered_prompt="v1",
            output="out",
            latency_ms=50.0,
            prompt_tokens=5,
            completion_tokens=5,
            total_tokens=10,
            cost_usd=0.0001,
            model="gpt-4o",
        )
        comp = ComparisonResult(test_case=tc, v1_result=r1, v2_result=r1)
        rep = DiffReport(
            run_id=f"run_concurrent_{idx:03d}",
            v1_name="v1",
            v2_name="v2",
            model_v1="gpt-4o",
            model_v2="gpt-4o",
            comparisons=[comp],
            verdict=RegressionVerdict(passed=True),
            evaluators=["latency"],
            total_cases=1,
        )
        worker_db.record_run(rep)
        return rep.run_id

    # Execute 15 concurrent writes across 5 worker threads
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(write_worker, i) for i in range(15)]
        results = [f.result() for f in futures]

    assert len(results) == 15
    recent_runs = db.get_recent_runs(limit=50)
    assert len(recent_runs) == 15
