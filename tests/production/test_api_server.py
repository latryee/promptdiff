"""Tests for FastAPI API Server authentication and host security."""

from __future__ import annotations

import os
from unittest.mock import patch

from starlette.testclient import TestClient

import promptdiff
from promptdiff.cli.server import create_app, launch_server


def test_fastapi_server_app() -> None:
    """Test FastAPI application initialization and dynamic version match."""
    app = create_app()
    assert app is not None
    assert app.version == promptdiff.__version__

    client = TestClient(app)
    res = client.get("/")
    assert res.status_code == 200
    assert res.json()["version"] == promptdiff.__version__


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


def test_cors_wildcard_origin_never_allows_credentials() -> None:
    """Wildcard origin must never have allow_credentials=True in response headers."""
    app = create_app(cors_origins=["*"], allow_credentials=True)
    client = TestClient(app)
    res = client.options(
        "/api/v1/shrink",
        headers={
            "Origin": "https://attacker.com",
            "Access-Control-Request-Method": "POST",
        },
    )
    # Access-Control-Allow-Credentials must not be 'true' when origin is wildcard
    assert res.headers.get("access-control-allow-credentials") != "true"


def test_cors_explicit_origin_allows_credentials() -> None:
    """Explicit trusted origin can allow credentials."""
    app = create_app(cors_origins=["https://trusted.promptdiff.com"], allow_credentials=True)
    client = TestClient(app)
    res = client.options(
        "/api/v1/shrink",
        headers={
            "Origin": "https://trusted.promptdiff.com",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert res.headers.get("access-control-allow-origin") == "https://trusted.promptdiff.com"
    assert res.headers.get("access-control-allow-credentials") == "true"


def test_rate_limiter_exceeded_returns_429() -> None:
    """Rate limiter returns 429 when client exceeds request limit."""
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("PROMPTDIFF_API_KEY", None)
        # Configure a tight limit of 2 requests per minute for testing
        app = create_app(rate_limit_per_minute=2)
        client = TestClient(app)

        payload = {"prompt": "Tell me a joke", "model": "gpt-4o", "mock": True}

        # Request 1: OK
        res1 = client.post("/api/v1/fuzz", json=payload)
        assert res1.status_code == 200

        # Request 2: OK
        res2 = client.post("/api/v1/fuzz", json=payload)
        assert res2.status_code == 200

        # Request 3: Exceeded -> 429
        res3 = client.post("/api/v1/fuzz", json=payload)
        assert res3.status_code == 429
        assert "Rate limit exceeded" in res3.json().get("detail", "")
        assert res3.headers.get("retry-after") == "60"


def test_token_bucket_limiter_unit() -> None:
    """Direct unit test of TokenBucketRateLimiter logic."""
    from promptdiff.cli._server_security import TokenBucketRateLimiter

    limiter = TokenBucketRateLimiter(rate_per_minute=3)
    assert limiter.acquire("ip1") is True
    assert limiter.acquire("ip1") is True
    assert limiter.acquire("ip1") is True
    assert limiter.acquire("ip1") is False

    # Independent IP is not blocked
    assert limiter.acquire("ip2") is True
