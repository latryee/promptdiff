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
