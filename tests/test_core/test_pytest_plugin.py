"""Tests migrated to feature-focused test suite."""

from __future__ import annotations

import pytest

from promptdiff.core.models import TestCase


def test_pytest_plugin_fixture(promptdiff_eval: pytest.FixtureRequest) -> None:
    """Test pytest fixture provided by pytest_plugin."""
    report = promptdiff_eval(
        v1="Say hello: {{name}}",
        v2="Greet user: {{name}}",
        dataset=[TestCase(id="1", vars={"name": "Alice"})],
        mock=True,
    )
    assert report.total_cases == 1
    assert report.verdict.passed is True
