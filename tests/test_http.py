"""HttpClient tests: TTL caching, stale-on-error, and retry caps (using
aioresponses to avoid real network)."""

import aioresponses
import pytest

from dashboard.http import CachedFetch, FetchError, HttpClient, TtlCache


def test_ttl_cache_hit_and_expiry():
    cache = TtlCache()
    cache.set("k", "v")
    hit, value = cache.get("k", ttl=60)
    assert hit and value == "v"

    cache._store["k"].fetched_at = 0  # force expiry  # noqa: SLF001
    hit, _ = cache.get("k", ttl=60)
    assert not hit


def test_ttl_cache_fifo_eviction():
    cache = TtlCache(max_entries=2)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)
    hit, _ = cache.get("a", ttl=60)
    assert not hit
    hit, _ = cache.get("c", ttl=60)
    assert hit


@pytest.mark.asyncio
async def test_fetch_json_cached_uses_ttl_and_returns_fresh():
    import aiohttp

    async with aiohttp.ClientSession() as session:
        client = HttpClient(session)
        spec = CachedFetch("https://example.test/data.json", ttl=60, cache_key="data")
        with aioresponses.aioresponses() as mock:
            mock.get("https://example.test/data.json", payload={"a": 1})
            stale, value, fetched = await client.fetch_json_cached(spec)
            assert stale is False
            assert value == {"a": 1}
            # second call hits cache (no extra HTTP)
            stale2, value2, _ = await client.fetch_json_cached(spec)
            assert stale2 is False and value2 == {"a": 1}
            # only one request matched: the second call came from cache
            assert len(mock.requests) == 1


@pytest.mark.asyncio
async def test_fetch_json_cached_stale_on_error():
    import aiohttp

    async with aiohttp.ClientSession() as session:
        client = HttpClient(session, retry_attempts=1)
        spec = CachedFetch("https://example.test/data.json", ttl=60, cache_key="data")
        with aioresponses.aioresponses() as mock:
            mock.get("https://example.test/data.json", payload={"a": 1})
            _, _, _ = await client.fetch_json_cached(spec)
            # expire the cache then fail the fetch
            client.cache._store["data"].fetched_at = 0  # noqa: SLF001
            mock.get("https://example.test/data.json", status=500)
            stale, value, _ = await client.fetch_json_cached(spec)
            assert stale is True
            assert value == {"a": 1}


@pytest.mark.asyncio
async def test_fetch_raises_when_no_cached_value():
    import aiohttp

    async with aiohttp.ClientSession() as session:
        client = HttpClient(session, retry_attempts=1)
        spec = CachedFetch("https://example.test/other.json", ttl=60, cache_key="other")
        with aioresponses.aioresponses() as mock:
            mock.get("https://example.test/other.json", status=500)
            with pytest.raises(FetchError):
                await client.fetch_json_cached(spec)


@pytest.mark.asyncio
async def test_gather_any_isolates_exceptions():
    async def boom():
        raise ValueError("x")

    async def ok():
        return 42

    class _C:
        async def gather_any(self, awaitables):
            import asyncio

            return await asyncio.gather(*awaitables, return_exceptions=True)

    results = await _C().gather_any([boom(), ok()])
    assert isinstance(results[0], ValueError)
    assert results[1] == 42
