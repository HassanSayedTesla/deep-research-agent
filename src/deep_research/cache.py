"""A tiny, inspectable, TTL-based cache for web search results.

Research runs hammer the same handful of questions, and a run that is
replayed or resumed re-asks most of them. Caching keeps the bill down and makes
local iteration much faster.

Storage is a single JSON file so it can be diffed, inspected and deleted by
hand. Writes are atomic and serialised behind an `asyncio.Lock` so concurrent
search workers cannot corrupt the file.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
import time
from collections.abc import Callable
from pathlib import Path


class SearchCache:
    """Key/value cache with a time-to-live, persisted as one JSON file."""

    def __init__(self, path: Path | str, ttl_seconds: float = 24 * 3600) -> None:
        self.path = Path(path)
        self.ttl_seconds = ttl_seconds
        self._lock = asyncio.Lock()
        self._entries: dict[str, dict[str, float | str]] = {}
        self._loaded = False
        self.hits = 0
        self.misses = 0

    # -- keys ---------------------------------------------------------------
    @staticmethod
    def make_key(namespace: str, *parts: str) -> str:
        """Build a stable key from a namespace and its components."""
        digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()
        return f"{namespace}:{digest}"

    # -- lifecycle ----------------------------------------------------------
    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # A corrupt cache must never break a run; start over instead.
            return
        if isinstance(raw, dict):
            self._entries = {k: v for k, v in raw.items() if isinstance(v, dict)}

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self._entries, indent=2, sort_keys=True)
        # Atomic replace: a crash mid-write leaves the old file intact.
        fd, tmp_name = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
            os.replace(tmp_name, self.path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise

    def _is_fresh(self, entry: dict[str, float | str]) -> bool:
        stored_at = float(entry.get("stored_at", 0.0))
        return (time.time() - stored_at) < self.ttl_seconds

    # -- api ----------------------------------------------------------------
    def get(self, key: str) -> str | None:
        self._load()
        entry = self._entries.get(key)
        if entry is None or not self._is_fresh(entry):
            self.misses += 1
            return None
        self.hits += 1
        return str(entry["value"])

    def set(self, key: str, value: str) -> None:
        self._load()
        self._entries[key] = {"value": value, "stored_at": time.time()}
        self._write()

    def prune(self) -> int:
        """Drop expired entries. Returns how many were removed."""
        self._load()
        stale = [key for key, entry in self._entries.items() if not self._is_fresh(entry)]
        for key in stale:
            del self._entries[key]
        if stale:
            self._write()
        return len(stale)

    def clear(self) -> None:
        self._load()
        self._entries.clear()
        self._write()

    async def wrap(self, key: str, producer: Callable[[], object]) -> str:
        """Return the cached value for `key`, or produce, store and return a new one.

        Two callers racing on the same cold key will both run `producer`; that
        is intentional — a single-flight lock would hold the event loop open
        for the whole network round trip, and a duplicated search is far
        cheaper than the complexity.
        """
        async with self._lock:
            self._load()
            entry = self._entries.get(key)
            if entry is not None and self._is_fresh(entry):
                self.hits += 1
                return str(entry["value"])
            self.misses += 1

        value = await producer()  # type: ignore[misc]
        text = value if isinstance(value, str) else str(value)

        async with self._lock:
            self._entries[key] = {"value": text, "stored_at": time.time()}
            self._write()
        return text

    def stats(self) -> dict[str, int]:
        total = self.hits + self.misses
        return {
            "entries": len(self._entries),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate_pct": round(100 * self.hits / total, 1) if total else 0.0,
        }
