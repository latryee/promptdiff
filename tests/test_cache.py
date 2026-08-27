"""Unit tests for SQLite persistent disk cache."""

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
