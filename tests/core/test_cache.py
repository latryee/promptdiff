"""Unit tests for SQLite persistent disk cache."""

import pytest

from promptdiff.core.cache import DiskCache
from promptdiff.core.models import RunResult


def test_cache_key_computation():
    key1 = DiskCache.compute_key(
        prompt_text="Hello world",
        system_prompt="You are an assistant",
        model="gpt-4o",
        temperature=0.0,
    )
    key2 = DiskCache.compute_key(
        prompt_text="Hello world",
        system_prompt="You are an assistant",
        model="gpt-4o",
        temperature=0.0,
    )
    key3 = DiskCache.compute_key(
        prompt_text="Hello world 2",
        system_prompt="You are an assistant",
        model="gpt-4o",
        temperature=0.0,
    )
    assert key1 == key2
    assert key1 != key3


def test_cache_get_set_clear(tmp_cache: DiskCache):
    key = DiskCache.compute_key("test prompt", model="gpt-4o")
    assert tmp_cache.get(key) is None

    result = RunResult(
        prompt_name="v1",
        test_case_id="tc_1",
        rendered_prompt="test prompt",
        output="Result output text",
        latency_ms=120.0,
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        cost_usd=0.0001,
        model="gpt-4o",
    )

    tmp_cache.set(key, result)
    assert tmp_cache.count() == 1

    cached = tmp_cache.get(key)
    assert cached is not None
    assert cached.output == "Result output text"
    assert cached.cached is True

    cleared = tmp_cache.clear()
    assert cleared == 1
    assert tmp_cache.count() == 0
    assert tmp_cache.get(key) is None


@pytest.mark.asyncio
async def test_cache_async_methods(tmp_cache: DiskCache) -> None:
    """Verify asynchronous non-blocking cache operations."""
    key = DiskCache.compute_key("async prompt", model="gpt-4o")
    assert await tmp_cache.async_get(key) is None

    result = RunResult(
        prompt_name="v1",
        test_case_id="tc_async",
        rendered_prompt="async prompt",
        output="Async result text",
        latency_ms=110.0,
        prompt_tokens=12,
        completion_tokens=18,
        total_tokens=30,
        cost_usd=0.0001,
        model="gpt-4o",
    )

    await tmp_cache.async_set(key, result)
    assert await tmp_cache.async_count() == 1

    cached = await tmp_cache.async_get(key)
    assert cached is not None
    assert cached.output == "Async result text"
    assert cached.cached is True

    cleared = await tmp_cache.async_clear()
    assert cleared == 1
    assert await tmp_cache.async_count() == 0
    assert await tmp_cache.async_get(key) is None


def test_cache_ttl_invalidation_and_pruning(tmp_path: pytest.TempPathFactory) -> None:
    """Verify TTL-based cache expiration and automated pruning."""
    import sqlite3
    from pathlib import Path

    cache_dir = Path(str(tmp_path)) / "ttl_cache"
    cache = DiskCache(cache_dir=cache_dir, ttl=60)  # Robust TTL (manual backdating tests expiration)

    key = DiskCache.compute_key("ttl prompt", model="gpt-4o")
    result = RunResult(
        prompt_name="v1",
        test_case_id="tc_ttl",
        rendered_prompt="ttl prompt",
        output="Fresh output",
        latency_ms=100.0,
        prompt_tokens=10,
        completion_tokens=10,
        total_tokens=20,
        cost_usd=0.0001,
        model="gpt-4o",
    )

    cache.set(key, result)
    assert cache.get(key) is not None

    # Manually backdate created_at in SQLite to simulate expiry
    with sqlite3.connect(cache.db_path) as conn:
        conn.execute("UPDATE prompt_cache SET created_at = '2020-01-01 00:00:00' WHERE hash_key = ?", (key,))
        conn.commit()

    # Accessing expired entry should return None and prune it
    assert cache.get(key) is None
    assert cache.count() == 0

    # Test prune_expired
    cache.set(key, result)
    with sqlite3.connect(cache.db_path) as conn:
        conn.execute("UPDATE prompt_cache SET created_at = '2020-01-01 00:00:00' WHERE hash_key = ?", (key,))
        conn.commit()

    pruned = cache.prune_expired()
    assert pruned == 1
    assert cache.count() == 0
