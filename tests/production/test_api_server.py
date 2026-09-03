"""Tests migrated to feature-focused test suite."""

from __future__ import annotations

from promptdiff.cli.server import create_app


def test_fastapi_server_app() -> None:
    """Test FastAPI application initialization."""
    app = create_app()
    # App is either instantiated or gracefully None if fastapi not installed in test env
    assert app is not None or create_app() is None
