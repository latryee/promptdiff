"""Deterministic Disk Cache for promptdiff.

Avoids duplicate LLM API calls, saves money, and provides instant evaluation feedback
using SHA-256 hashed SQLite persistent cache.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from pathlib import Path

from promptdiff.core.models import RunResult


class DiskCache:
    """Persistent SQLite-backed cache for prompt runs."""

    def __init__(self, cache_dir: Path | None = None, enabled: bool = True):
        self.enabled = enabled
        if cache_dir is None:
            self.cache_dir = Path.cwd() / ".promptdiff_cache"
        else:
            self.cache_dir = Path(cache_dir)

        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = self.cache_dir / "cache.sqlite"
            self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite cache table."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS prompt_cache (
                    hash_key TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    @staticmethod
    def compute_key(
        prompt_text: str,
        system_prompt: str | None = None,
        model: str = "gpt-4o",
        temperature: float = 0.0,
        max_tokens: int | None = 2048,
    ) -> str:
        """Compute SHA-256 hash for deterministic execution parameters."""
        raw_key = json.dumps(
            {
                "prompt": prompt_text,
                "system": system_prompt or "",
                "model": model.strip().lower(),
                "temperature": round(temperature, 4),
                "max_tokens": max_tokens or 0,
            },
            sort_keys=True,
        )
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def get(self, hash_key: str) -> RunResult | None:
        """Retrieve cached result if available and valid."""
        if not self.enabled:
            return None

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT data FROM prompt_cache WHERE hash_key = ?",
                    (hash_key,),
                )
                row = cursor.fetchone()
                if row:
                    data = json.loads(row[0])
                    result = RunResult.model_validate(data)
                    result.cached = True
                    return result
        except Exception:
            return None
        return None

    def set(self, hash_key: str, result: RunResult) -> None:
        """Store run result in cache."""
        if not self.enabled:
            return

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO prompt_cache (hash_key, data) VALUES (?, ?)",
                    (hash_key, json.dumps(result.model_dump())),
                )
                conn.commit()
        except Exception:
            pass

    def clear(self) -> int:
        """Clear entire cache."""
        if not self.enabled or not self.db_path.exists():
            return 0

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM prompt_cache")
            count = cursor.rowcount
            conn.commit()
            return count

    def count(self) -> int:
        """Get number of cached entries."""
        if not self.enabled or not hasattr(self, "db_path") or not self.db_path.exists():
            return 0
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT count(*) FROM prompt_cache")
            row = cursor.fetchone()
            return row[0] if row else 0

    async def async_get(self, hash_key: str) -> RunResult | None:
        """Asynchronously retrieve cached result without blocking the event loop."""
        return await asyncio.to_thread(self.get, hash_key)

    async def async_set(self, hash_key: str, result: RunResult) -> None:
        """Asynchronously store run result in cache without blocking the event loop."""
        await asyncio.to_thread(self.set, hash_key, result)

    async def async_clear(self) -> int:
        """Asynchronously clear entire cache without blocking the event loop."""
        return await asyncio.to_thread(self.clear)

    async def async_count(self) -> int:
        """Asynchronously get number of cached entries without blocking the event loop."""
        return await asyncio.to_thread(self.count)
