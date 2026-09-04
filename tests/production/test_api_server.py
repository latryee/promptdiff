"""Tests for FastAPI API Server authentication and host security."""

from __future__ import annotations

import os
from unittest.mock import patch

from starlette.testclient import TestClient

from promptdiff.cli.server import create_app, launch_server


def test_fastapi_server_app() -> None:
    """Test FastAPI application initialization."""
    app = create_app()
    assert app is not None


def test_server_unauthenticated_when_key_unset() -> None:
    """When PROMPTDIFF_API_KEY is not set, endpoints are accessible without auth."""
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("PROMPTDIFF_API_KEY", None)
        app = create_app()
        client = TestClient(app)

        res = client.get("/")
        assert res.status_code == 200

        res = client.post(
            "/api/v1/shrink",
            json={"prompt": "Hello world", "target_reduction": 0.2, "mock": True},
        )
        assert res.status_code == 200
        assert "compressed_prompt" in res.json()


def test_server_auth_enforced_401_when_key_missing() -> None:
    """When PROMPTDIFF_API_KEY is configured, requests without X-API-Key return 401."""
    with patch.dict(os.environ, {"PROMPTDIFF_API_KEY": "test-super-secret-123"}):
        app = create_app()
        client = TestClient(app)

        res = client.get("/")
        assert res.status_code == 200

        res = client.post(
            "/api/v1/shrink",
            json={"prompt": "Hello world", "target_reduction": 0.2, "mock": True},
        )
        assert res.status_code == 401
        assert "Unauthorized" in res.json().get("detail", "")


def test_server_auth_enforced_401_when_key_invalid() -> None:
    """When PROMPTDIFF_API_KEY is configured, requests with invalid X-API-Key return 401."""
    with patch.dict(os.environ, {"PROMPTDIFF_API_KEY": "test-super-secret-123"}):
        app = create_app()
        client = TestClient(app)

        res = client.post(
            "/api/v1/shrink",
            headers={"X-API-Key": "wrong-key"},
            json={"prompt": "Hello world", "target_reduction": 0.2, "mock": True},
        )
        assert res.status_code == 401


def test_server_auth_success_with_valid_key() -> None:
    """When PROMPTDIFF_API_KEY is configured, requests with valid X-API-Key return 200."""
    with patch.dict(os.environ, {"PROMPTDIFF_API_KEY": "test-super-secret-123"}):
        app = create_app()
        client = TestClient(app)

        res = client.post(
            "/api/v1/shrink",
            headers={"X-API-Key": "test-super-secret-123"},
            json={"prompt": "Hello world", "target_reduction": 0.2, "mock": True},
        )
        assert res.status_code == 200
        assert "compressed_prompt" in res.json()


def test_launch_server_rebinding_warning_when_key_unset() -> None:
    """launch_server warns and falls back to 127.0.0.1 if key is unset and host is public."""
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("PROMPTDIFF_API_KEY", None)
        with patch("uvicorn.run") as mock_run, patch("promptdiff.cli.server.logger.warning") as mock_warn:
            launch_server(host="0.0.0.0", port=8000)
            mock_warn.assert_called_once()
            assert "Rebinding server to '127.0.0.1'" in mock_warn.call_args[0][0]
            mock_run.assert_called_once()
            assert mock_run.call_args[1]["host"] == "127.0.0.1"
