"""Unit tests for the search cache."""

from __future__ import annotations

import time
from pathlib import Path

from deep_research.cache import SearchCache


def test_keys_are_stable_and_namespaced():
    a = SearchCache.make_key("tavily", "gearbox", "3")
    b = SearchCache.make_key("tavily", "gearbox", "3")
    c = SearchCache.make_key("tavily", "gearbox", "4")

    assert a == b
    assert a != c
    assert a.startswith("tavily:")
    assert len(a) > len("tavily:")


def test_set_then_get(tmp_path: Path):
    cache = SearchCache(tmp_path / "c.json")
    cache.set("k", "v")

    assert cache.get("k") == "v"
    assert cache.hits == 1


def test_missing_key_is_a_miss(tmp_path: Path):
    cache = SearchCache(tmp_path / "c.json")

    assert cache.get("nope") is None
    assert cache.misses == 1


def test_entries_survive_a_reload(tmp_path: Path):
    path = tmp_path / "c.json"
    SearchCache(path).set("k", "v")

    assert SearchCache(path).get("k") == "v"


def test_expired_entries_are_not_returned(tmp_path: Path):
    cache = SearchCache(tmp_path / "c.json", ttl_seconds=0.0)
    cache.set("k", "v")
    time.sleep(0.01)

    assert cache.get("k") is None


def test_prune_drops_only_expired_entries(tmp_path: Path):
    fresh = SearchCache(tmp_path / "fresh.json", ttl_seconds=3600)
    stale = SearchCache(tmp_path / "stale.json", ttl_seconds=3600)
    fresh.set("keep", "v")

    stale.set("drop", "v")
    stale._entries["drop"]["stored_at"] = 0.0  # backdate past the ttl

    assert stale.prune() == 1
    assert stale.get("drop") is None
    assert fresh.prune() == 0
    assert fresh.get("keep") == "v"


def test_corrupt_cache_file_is_ignored_not_fatal(tmp_path: Path):
    path = tmp_path / "c.json"
    path.write_text("{not json", encoding="utf-8")

    cache = SearchCache(path)
    assert cache.get("anything") is None
    cache.set("k", "v")  # and the cache must still be usable
    assert cache.get("k") == "v"


def test_clear_empties_the_cache(tmp_path: Path):
    cache = SearchCache(tmp_path / "c.json")
    cache.set("k", "v")
    cache.clear()

    assert cache.get("k") is None


async def test_wrap_caches_the_produced_value(tmp_path: Path):
    cache = SearchCache(tmp_path / "c.json")
    produced = 0

    async def producer() -> str:
        nonlocal produced
        produced += 1
        return "value"

    assert await cache.wrap("k", producer) == "value"
    assert await cache.wrap("k", producer) == "value"
    assert produced == 1
    assert cache.hits == 1


async def test_wrap_coerces_non_string_results(tmp_path: Path):
    cache = SearchCache(tmp_path / "c.json")

    async def producer() -> int:
        return 42

    assert await cache.wrap("k", producer) == "42"


def test_cache_file_is_valid_json_after_a_write(tmp_path: Path):
    import json

    path = tmp_path / "nested" / "c.json"
    cache = SearchCache(path)
    cache.set("k", "v")
    cache._write()

    assert json.loads(path.read_text(encoding="utf-8"))["k"]["value"] == "v"


def test_stats_summarise_hit_rate(tmp_path: Path):
    cache = SearchCache(tmp_path / "c.json")
    cache.set("a", "1")
    cache.get("a")
    cache.get("b")

    stats = cache.stats()
    assert stats["entries"] == 1
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_rate_pct"] == 50.0


def test_stats_of_an_untouched_cache(tmp_path: Path):
    assert SearchCache(tmp_path / "c.json").stats()["hit_rate_pct"] == 0.0
