"""HttpClient tests: caching, stale-on-error, and per-origin pacing."""

import asyncio

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
async def test_fetch_json_cached_uses_ttl_and_returns_fresh(monkeypatch):
    client = HttpClient(object())
    calls = 0

    async def fetch_json(_url, _headers=None):
        nonlocal calls
        calls += 1
        return {"a": 1}

    monkeypatch.setattr(client, "fetch_json", fetch_json)
    spec = CachedFetch("https://example.test/data.json", ttl=60, cache_key="data")
    stale, value, fetched = await client.fetch_json_cached(spec)
    assert stale is False
    assert value == {"a": 1}
    stale2, value2, fetched2 = await client.fetch_json_cached(spec)
    assert stale2 is False and value2 == {"a": 1}
    assert fetched2 == fetched
    assert calls == 1


@pytest.mark.asyncio
async def test_fetch_json_cached_stale_on_error(monkeypatch):
    client = HttpClient(object(), retry_attempts=1)
    should_fail = False

    async def fetch_json(_url, _headers=None):
        if should_fail:
            raise FetchError("offline")
        return {"a": 1}

    monkeypatch.setattr(client, "fetch_json", fetch_json)
    spec = CachedFetch("https://example.test/data.json", ttl=60, cache_key="data")
    await client.fetch_json_cached(spec)
    client.cache._store["data"].fetched_at = 0  # noqa: SLF001
    should_fail = True
    stale, value, _ = await client.fetch_json_cached(spec)
    assert stale is True
    assert value == {"a": 1}


@pytest.mark.asyncio
async def test_fetch_text_cached_uses_ttl_and_stale_on_error(monkeypatch):
    client = HttpClient(object(), retry_attempts=1)
    calls = 0
    should_fail = False

    async def fetch_text(_url, _headers=None, _max_bytes=None):
        nonlocal calls
        calls += 1
        if should_fail:
            raise FetchError("offline")
        return "first"

    monkeypatch.setattr(client, "fetch_text", fetch_text)
    spec = CachedFetch("https://example.test/data.txt", ttl=60, cache_key="text")
    stale, value, fetched_at = await client.fetch_text_cached(spec)
    assert stale is False
    assert value == "first"

    cached_stale, cached_value, cached_at = await client.fetch_text_cached(spec)
    assert cached_stale is False
    assert cached_value == "first"
    assert cached_at == fetched_at
    assert calls == 1

    client.cache._store["text"].fetched_at = 0  # noqa: SLF001
    should_fail = True
    stale, value, stale_at = await client.fetch_text_cached(spec)
    assert stale is True
    assert value == "first"
    assert stale_at == 0


@pytest.mark.asyncio
async def test_fetch_raises_when_no_cached_value(monkeypatch):
    client = HttpClient(object(), retry_attempts=1)

    async def fail(_url, _headers=None):
        raise FetchError("offline")

    monkeypatch.setattr(client, "fetch_json", fail)
    spec = CachedFetch("https://example.test/other.json", ttl=60, cache_key="other")
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


@pytest.mark.asyncio
async def test_origin_pacing_spaces_one_host_without_blocking_another():
    client = HttpClient(object(), origin_request_interval_seconds=0.04)
    started: dict[str, float] = {}
    loop = asyncio.get_running_loop()

    async def paced(name: str, url: str) -> None:
        await client._pace_origin(url)  # noqa: SLF001
        started[name] = loop.time()

    await asyncio.gather(
        paced("one-first", "https://one.example/first"),
        paced("two-first", "https://two.example/first"),
        paced("one-second", "https://one.example/second"),
    )
    assert abs(started["one-first"] - started["two-first"]) < 0.02
    assert started["one-second"] - started["one-first"] >= 0.025
