"""Tests for studio and dashboard live WebSocket/SSE streaming."""

from __future__ import annotations

import http.server
import json
import threading

import httpx

from promptdiff.cli.dashboard import stream_live_progress
from promptdiff.cli.studio import StudioRequestHandler


def test_dashboard_stream_live_progress() -> None:
    """Test dashboard stream_live_progress generator yields structured progress events."""
    events = list(stream_live_progress(total_cases=3, delay=0.001))
    assert len(events) == 3

    assert events[0]["step"] == 1
    assert events[0]["total"] == 3
    assert events[0]["pct"] == 33
    assert events[0]["status"] == "evaluating"

    assert events[-1]["step"] == 3
    assert events[-1]["pct"] == 100
    assert events[-1]["status"] == "completed"


def test_studio_sse_progress_stream() -> None:
    """Test studio HTTP server /api/stream-progress SSE streaming endpoint."""
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), StudioRequestHandler)
    port = server.server_port
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    try:
        url = f"http://127.0.0.1:{port}/api/stream-progress"
        with httpx.stream("GET", url, timeout=5.0) as resp:
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers.get("content-type", "")

            events = []
            for line in resp.iter_lines():
                if line.startswith("data: "):
                    payload = json.loads(line[6:])
                    events.append(payload)
                    if payload.get("done"):
                        break

            assert len(events) >= 2
            assert any(e.get("done") is True for e in events)
            step_events = [e for e in events if "step" in e]
            assert len(step_events) > 0
            assert step_events[0]["step"] == 1
    finally:
        server.shutdown()
        server.server_close()


def test_studio_auth_enforced_when_key_set() -> None:
    """When PROMPTDIFF_API_KEY is configured, requests to /api/compare without key return 401."""
    import os
    from unittest.mock import MagicMock, patch

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), StudioRequestHandler)
    port = server.server_port
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    mock_report = MagicMock()
    mock_report.verdict.passed = True
    mock_report.verdict.cost_delta_pct = -5.0
    mock_report.verdict.latency_delta_pct = 2.0
    mock_report.verdict.total_cost_v1 = 0.01
    mock_report.verdict.total_cost_v2 = 0.009

    try:
        with (
            patch.dict(os.environ, {"PROMPTDIFF_API_KEY": "studio-secret-key"}),
            patch("promptdiff.cli.studio.compare", return_value=mock_report),
        ):
            url = f"http://127.0.0.1:{port}/api/compare"
            payload = {"v1": "Hello", "v2": "Hi", "dataset": [{"id": "t1", "vars": {}}]}

            # Missing API key -> 401
            resp401 = httpx.post(url, json=payload, timeout=5.0)
            assert resp401.status_code == 401
            assert "Unauthorized" in resp401.text

            # Valid API key -> 200
            resp200 = httpx.post(url, json=payload, headers={"X-API-Key": "studio-secret-key"}, timeout=5.0)
            assert resp200.status_code == 200
            assert resp200.json()["cost_delta_pct"] == -5.0
    finally:
        server.shutdown()
        server.server_close()


def test_studio_rate_limit_429() -> None:
    """When rate limit is exhausted on studio, /api/compare returns 429."""
    import os
    import time
    from unittest.mock import MagicMock, patch

    from promptdiff.cli.studio import studio_limiter

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), StudioRequestHandler)
    port = server.server_port
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    mock_report = MagicMock()
    mock_report.verdict.passed = True
    mock_report.verdict.cost_delta_pct = 0.0
    mock_report.verdict.latency_delta_pct = 0.0
    mock_report.verdict.total_cost_v1 = 0.01
    mock_report.verdict.total_cost_v2 = 0.01

    try:
        with patch.dict(os.environ, {}, clear=False), patch("promptdiff.cli.studio.compare", return_value=mock_report):
            os.environ.pop("PROMPTDIFF_API_KEY", None)
            url = f"http://127.0.0.1:{port}/api/compare"
            payload = {"v1": "Hello", "v2": "Hi", "dataset": [{"id": "t1", "vars": {}}]}

            # Force limiter tokens to 0 with current timestamp
            studio_limiter._buckets["127.0.0.1"] = (0.0, time.monotonic())

            resp429 = httpx.post(url, json=payload, timeout=5.0)
            assert resp429.status_code == 429
            assert "Rate limit exceeded" in resp429.text
            assert resp429.headers.get("retry-after") == "60"

            # Reset limiter restores access
            studio_limiter.reset()
            resp200 = httpx.post(url, json=payload, timeout=5.0)
            assert resp200.status_code == 200
    finally:
        server.shutdown()
        server.server_close()
        studio_limiter.reset()


def test_studio_cors_options() -> None:
    """OPTIONS request returns 204 with CORS preflight headers."""
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), StudioRequestHandler)
    port = server.server_port
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    try:
        url = f"http://127.0.0.1:{port}/api/compare"
        resp = httpx.options(url, timeout=5.0)
        assert resp.status_code == 204
        assert resp.headers.get("access-control-allow-origin") == "*"
        assert "X-API-Key" in resp.headers.get("access-control-allow-headers", "")
    finally:
        server.shutdown()
        server.server_close()
