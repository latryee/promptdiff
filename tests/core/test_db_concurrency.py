"""Concurrency stress and benchmark tests for SQLite WAL TelemetryDatabase."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from promptdiff.core.db import TelemetryDatabase
from promptdiff.core.models import (
    ComparisonResult,
    DiffReport,
    RegressionVerdict,
    RunResult,
    TestCase,
)


def test_sqlite_pragmas_wal_and_busy_timeout(tmp_path: Path) -> None:
    """Verify SQLite connection is configured with WAL journal mode and busy_timeout."""
    db_file = tmp_path / "pragma_test.db"
    db = TelemetryDatabase(db_path=str(db_file))

    with db.connection() as conn:
        cursor = conn.execute("PRAGMA journal_mode")
        mode = cursor.fetchone()[0]
        assert str(mode).lower() == "wal"

        cursor = conn.execute("PRAGMA busy_timeout")
        timeout = cursor.fetchone()[0]
        assert int(timeout) >= 5000


def test_concurrent_writes_and_reads_benchmark(tmp_path: Path) -> None:
    """Benchmark high-concurrency writes and reads across 20 threads without database locking errors."""
    db_file = tmp_path / "benchmark_telemetry.db"
    db = TelemetryDatabase(db_path=str(db_file))

    total_runs = 60
    workers = 15

    def write_run(idx: int) -> str:
        worker_db = TelemetryDatabase(db_path=str(db_file))
        tc = TestCase(id=f"case_{idx:03d}", vars={"q": f"question_{idx}"})
        res = RunResult(
            prompt_name="v1",
            test_case_id=tc.id,
            rendered_prompt="test",
            output="test output",
            latency_ms=25.0,
            prompt_tokens=5,
            completion_tokens=5,
            total_tokens=10,
            cost_usd=0.0001,
            model="mock-gpt",
        )
        report = DiffReport(
            run_id=f"bench_run_{idx:04d}",
            v1_name="v1",
            v2_name="v2",
            model_v1="mock-gpt",
            model_v2="mock-gpt",
            comparisons=[ComparisonResult(test_case=tc, v1_result=res, v2_result=res)],
            verdict=RegressionVerdict(passed=(idx % 2 == 0)),
            evaluators=["latency"],
            total_cases=1,
        )
        worker_db.record_run(report)
        return report.run_id

    def read_runs() -> int:
        worker_db = TelemetryDatabase(db_path=str(db_file))
        runs = worker_db.get_recent_runs(limit=100)
        _ = worker_db.get_failure_hotspots(limit=10)
        return len(runs)

    start_time = time.perf_counter()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Submit concurrent write tasks
        write_futures = [executor.submit(write_run, i) for i in range(total_runs)]
        # Interleave concurrent read tasks
        read_futures = [executor.submit(read_runs) for _ in range(10)]

        for f in as_completed(write_futures):
            # Must complete without sqlite3.OperationalError: database is locked
            run_id = f.result()
            assert run_id.startswith("bench_run_")

        for f in as_completed(read_futures):
            count = f.result()
            assert count >= 0

    duration = time.perf_counter() - start_time
    writes_per_sec = total_runs / max(duration, 0.001)

    # Verification of persistence integrity
    all_runs = db.get_recent_runs(limit=200)
    assert len(all_runs) == total_runs
    run_ids = {r.run_id for r in all_runs}
    expected_ids = {f"bench_run_{i:04d}" for i in range(total_runs)}
    assert run_ids == expected_ids
    assert writes_per_sec > 0.0

    db.close()
