"""Coverage boost tests for sdk.py — exercise import paths of SDK convenience functions."""

from __future__ import annotations

import json

import promptdiff
from promptdiff.sdk import (
    analyze_cache_impact,
    cascaded_judge,
    compare,
    compute_bradley_terry_ratings,
    compute_elo_ratings,
    detect_schema_breaking_changes,
    replay_production_traces,
)


def test_sdk_compare() -> None:
    report = compare(
        v1="You are a support bot. Help: {{query}}",
        v2="You are an efficient support agent. Resolve: {{query}}",
        dataset=[{"query": "reset password"}],
        mock=True,
    )
    assert len(report.comparisons) == 1


def test_sdk_analyze_cache_impact() -> None:
    res = analyze_cache_impact(
        v1="Static prefix for prompt v1: {{query}}",
        v2="Static prefix for prompt v2: {{query}}",
    )
    assert res.model_name == "claude-3-5-sonnet"


def test_sdk_detect_schema_breaking() -> None:
    v1 = json.dumps({"id": 1, "name": "test"})
    v2 = json.dumps({"id": 1, "name": "test", "extra": True})
    report = detect_schema_breaking_changes(v1, v2)
    assert report.is_compatible is True


def test_sdk_cascaded_judge() -> None:
    res = cascaded_judge("output v1", "output v2", query="test", force_mock=True)
    assert res.winner in ("v1", "v2", "tie")


def test_sdk_compute_elo() -> None:
    matches = [
        {"prompt_a": "p1", "prompt_b": "p2", "winner": "A"},
        {"prompt_a": "p1", "prompt_b": "p2", "winner": "B"},
    ]
    res = compute_elo_ratings(matches)
    assert len(res.ratings) == 2


def test_sdk_compute_bt() -> None:
    matches = [
        {"prompt_a": "x", "prompt_b": "y", "winner": "A"},
        {"prompt_a": "x", "prompt_b": "y", "winner": "B"},
    ]
    res = compute_bradley_terry_ratings(matches)
    assert len(res.ratings) == 2


def test_sdk_replay_traces() -> None:
    traces = [{"trace_id": "t1", "query": "Help me"}]
    report = replay_production_traces(
        v1="v1: {{query}}",
        v2="v2: {{query}}",
        traces=traces,
        mock=True,
    )
    assert len(report.comparisons) == 1


def test_sdk_public_api_surface() -> None:
    """Verify all new SDK functions are accessible from top-level promptdiff."""
    assert hasattr(promptdiff, "analyze_cache_impact")
    assert hasattr(promptdiff, "detect_schema_breaking_changes")
    assert hasattr(promptdiff, "cascaded_judge")
    assert hasattr(promptdiff, "compute_elo_ratings")
    assert hasattr(promptdiff, "compute_bradley_terry_ratings")
    assert hasattr(promptdiff, "replay_production_traces")
