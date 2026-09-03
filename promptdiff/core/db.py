"""Persistent SQLite Historical Telemetry & Analytics Database.

Maintains a zero-dependency, local SQLite time-series database (`.promptdiff/telemetry.db`)
recording every prompt evaluation run, enabling historical trend queries, cost tracking,
and identification of high-frequency failing test cases over time.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from promptdiff.core.models import DiffReport


@dataclass
class RunSummaryRecord:
    """Historical run summary stored in SQLite."""

    run_id: str
    timestamp: float
    v1_name: str
    v2_name: str
    passed: bool
    cost_delta_pct: float
    latency_delta_pct: float
    total_cases: int


@dataclass
class FailureHotspot:
    """A test case with high failure frequency across historical runs."""

    test_case_id: str
    failure_count: int
    last_failed_timestamp: float


class TelemetryDatabase:
    """SQLite-backed historical store for prompt evaluation telemetry."""

    def __init__(self, db_path: str = ".promptdiff/telemetry.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _init_schema(self) -> None:
        """Create tables and indices if not present."""
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS evaluation_runs (
                    run_id TEXT PRIMARY KEY,
                    timestamp REAL,
                    v1_name TEXT,
                    v2_name TEXT,
                    passed INTEGER,
                    cost_delta_pct REAL,
                    latency_delta_pct REAL,
                    total_cases INTEGER
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS test_case_executions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    test_case_id TEXT,
                    passed INTEGER,
                    v1_latency_ms REAL,
                    v2_latency_ms REAL,
                    v1_cost_usd REAL,
                    v2_cost_usd REAL,
                    FOREIGN KEY(run_id) REFERENCES evaluation_runs(run_id)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_timestamp ON evaluation_runs(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tc_id ON test_case_executions(test_case_id)")
            conn.commit()

    def record_run(self, report: DiffReport) -> None:
        """Persist DiffReport results into SQLite."""
        now = time.time()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO evaluation_runs
                (run_id, timestamp, v1_name, v2_name, passed, cost_delta_pct, latency_delta_pct, total_cases)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    report.run_id,
                    now,
                    report.v1_name,
                    report.v2_name,
                    1 if report.verdict.passed else 0,
                    report.verdict.cost_delta_pct,
                    report.verdict.latency_delta_pct,
                    report.total_cases,
                ),
            )

            execution_rows = [
                (
                    report.run_id,
                    comp.test_case.id,
                    1 if all(s.passed for s in comp.scores.values()) else 0,
                    comp.v1_result.latency_ms,
                    comp.v2_result.latency_ms,
                    comp.v1_result.cost_usd,
                    comp.v2_result.cost_usd,
                )
                for comp in report.comparisons
            ]
            if execution_rows:
                conn.executemany(
                    """
                    INSERT INTO test_case_executions
                    (run_id, test_case_id, passed, v1_latency_ms, v2_latency_ms, v1_cost_usd, v2_cost_usd)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    execution_rows,
                )
            conn.commit()

    def get_recent_runs(self, limit: int = 20) -> list[RunSummaryRecord]:
        """Fetch chronological recent runs."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT run_id, timestamp, v1_name, v2_name, passed, cost_delta_pct, latency_delta_pct, total_cases
                FROM evaluation_runs
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            )
            records = []
            for row in cursor.fetchall():
                records.append(
                    RunSummaryRecord(
                        run_id=row["run_id"],
                        timestamp=row["timestamp"],
                        v1_name=row["v1_name"],
                        v2_name=row["v2_name"],
                        passed=bool(row["passed"]),
                        cost_delta_pct=row["cost_delta_pct"],
                        latency_delta_pct=row["latency_delta_pct"],
                        total_cases=row["total_cases"],
                    )
                )
            return records

    def get_failure_hotspots(self, limit: int = 5) -> list[FailureHotspot]:
        """Identify test cases with the highest regression failure frequency."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT test_case_id, COUNT(*) as fail_count, MAX(evaluation_runs.timestamp) as last_fail
                FROM test_case_executions
                JOIN evaluation_runs ON test_case_executions.run_id = evaluation_runs.run_id
                WHERE test_case_executions.passed = 0
                GROUP BY test_case_id
                ORDER BY fail_count DESC
                LIMIT ?
                """,
                (limit,),
            )
            hotspots = []
            for row in cursor.fetchall():
                hotspots.append(
                    FailureHotspot(
                        test_case_id=row["test_case_id"],
                        failure_count=row["fail_count"],
                        last_failed_timestamp=row["last_fail"] or 0.0,
                    )
                )
            return hotspots
