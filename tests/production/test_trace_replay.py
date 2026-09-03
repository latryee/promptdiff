"""Unit tests for Production Trace and OpenTelemetry / Langfuse Shadow Replayer."""

from __future__ import annotations

import json
from pathlib import Path

import promptdiff
from promptdiff.production.trace_replay import (
    ProductionTraceReplayer,
)


def test_trace_ingestion_and_pii_sanitization() -> None:
    """Ingest traces and ensure email/phone PII is redacted."""
    raw_spans = [
        {
            "trace_id": "tr_101",
            "query": "Please cancel my booking for user john.doe@example.com and contact 555-019-2834.",
        },
        {
            "id": "tr_102",
            "input": {"prompt": "What is the return policy for order #991?"},
        },
    ]

    replayer = ProductionTraceReplayer(mask_pii=True)
    report = replayer.ingest_traces(raw_spans)

    assert report.total_spans_read == 2
    assert report.valid_test_cases == 2
    assert report.masked_pii_count == 2

    # Verify redaction
    first_query = report.test_cases[0].vars["query"]
    assert "john.doe@example.com" not in first_query
    assert "[REDACTED_EMAIL]" in first_query
    assert "555-019-2834" not in first_query
    assert "[REDACTED_PHONE]" in first_query


def test_trace_ingestion_from_jsonl_file(tmp_path: Path) -> None:
    """Ingest production traces from JSONL file on disk."""
    jsonl_file = tmp_path / "traces.jsonl"
    lines = [
        json.dumps({"trace_id": "span_1", "query": "How do I upgrade my plan?"}),
        json.dumps({"trace_id": "span_2", "query": "Payment failed on checkout."}),
    ]
    jsonl_file.write_text("\n".join(lines), encoding="utf-8")

    replayer = ProductionTraceReplayer()
    report = replayer.ingest_traces(jsonl_file)

    assert report.total_spans_read == 2
    assert report.valid_test_cases == 2
    assert report.test_cases[0].id == "span_1"


def test_replay_sync_execution() -> None:
    """Replay production trace dataset across two prompt templates."""
    v1 = "You are a customer support agent. Help with: {{query}}"
    v2 = "You are an efficient customer support agent. Resolve: {{query}}"

    traces = [
        {"trace_id": "t1", "query": "Where is my invoice?"},
        {"trace_id": "t2", "query": "Reset my password."},
    ]

    replayer = ProductionTraceReplayer()
    diff_report = replayer.replay_sync(v1, v2, traces, mock=True)

    assert len(diff_report.comparisons) == 2
    assert diff_report.v1_name == "v1"
    assert diff_report.v2_name == "v2"


def test_replay_traces_cli(tmp_path: Path) -> None:
    """Test CLI command 'promptdiff replay-traces'."""
    from typer.testing import CliRunner

    from promptdiff.cli.app import app

    traces_file = tmp_path / "prod_traces.json"
    traces_file.write_text(
        json.dumps([{"trace_id": "tr_cli", "query": "Check system status."}]),
        encoding="utf-8",
    )

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "replay-traces",
            "Help with: {{query}}",
            "Assist with: {{query}}",
            "--traces",
            str(traces_file),
            "--limit",
            "5",
            "--mock",
        ],
    )
    assert res.exit_code == 0
    assert "Ingested 1 production spans" in res.output
    assert "Completed shadow replay" in res.output


def test_sdk_replay_production_traces() -> None:
    """Test top-level SDK wrapper replay_production_traces."""
    traces = [{"trace_id": "sdk_tr", "query": "What are your business hours?"}]
    diff_report = promptdiff.replay_production_traces(
        "v1: {{query}}",
        "v2: {{query}}",
        traces=traces,
        mock=True,
    )
    assert len(diff_report.comparisons) == 1
