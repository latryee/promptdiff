"""Pytest Test Configuration and Fixtures for promptdiff."""

from pathlib import Path

import pytest

from promptdiff.core.cache import DiskCache
from promptdiff.core.models import PromptVersion, TestCase
from promptdiff.providers.mock_provider import MockProvider

pytest_plugins = ["promptdiff.pytest_plugin"]


@pytest.fixture
def tmp_cache(tmp_path: Path) -> DiskCache:
    """Fixture providing a temporary disk cache."""
    return DiskCache(cache_dir=tmp_path / "cache", enabled=True)


@pytest.fixture
def mock_provider() -> MockProvider:
    """Fixture providing a fast mock provider without sleep delays."""
    return MockProvider(model_name="mock-gpt-4o", simulate_delay=False)


@pytest.fixture
def sample_test_case() -> TestCase:
    """Fixture providing a sample testcase."""
    return TestCase(
        id="tc_sample",
        description="Sample test scenario",
        vars={"query": "How do I upgrade to Pro plan?", "user": "Alice"},
    )


@pytest.fixture
def sample_prompts() -> tuple[PromptVersion, PromptVersion]:
    """Fixture providing a pair of v1 and v2 prompt versions."""
    v1 = PromptVersion(
        name="v1",
        template="You are a helpful assistant. Answer {{query}} for {{user}}.",
        model="gpt-4o",
    )
    v2 = PromptVersion(
        name="v2",
        template="You are a concise assistant. Answer {{query}} in bullet points for {{user}}.",
        model="gpt-4o",
    )
    return v1, v2
