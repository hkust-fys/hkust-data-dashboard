"""Shared async HTTP layer: one injected aiohttp session, validated fetching,
source-specific TTL caching, and stale-on-error behavior.

All providers receive the ``HttpClient`` instance created once in ``bot.py``.
There are no import-time side effects and no module-level network calls.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

import aiohttp

log = logging.getLogger(__name__)

USER_AGENT = "hkust-data-dashboard/2.0 (+https://github.com/hkust-fys/hkust-data-dashboard)"


def _with_user_agent(headers: dict[str, str] | None) -> dict[str, str]:
    """Return a copy of ``headers`` with the dashboard User-Agent set if absent.

    Some sources (HKeMobility) reject aiohttp's default Python user agent with
    HTTP 403, so every request must identify itself as the dashboard.
    """
    merged = dict(headers or {})
    if not any(key.lower() == "user-agent" for key in merged):
        merged["User-Agent"] = USER_AGENT
    return merged


# Bound response sizes to keep memory sane.
MAX_BYTES_TEXT = 2 * 1024 * 1024  # 2 MiB
MAX_BYTES_IMAGE = 4 * 1024 * 1024  # 4 MiB

RETRY_BASE_DELAY = 0.5
RETRY_MAX_DELAY = 8.0
RETRY_ATTEMPTS = 3
ORIGIN_REQUEST_INTERVAL_SECONDS = 0.06
ORIGIN_REQUEST_INTERVAL_OVERRIDES_SECONDS = {
    # The GMB host returns 403 for short bursts well below our generic pace.
    # Five starts/second keeps gate, probe, and route-metadata calls on one
    # shared origin budget.
    "data.etagmb.gov.hk": 0.2,
}


class FetchError(RuntimeError):
    """Raised when a fetch fails validation (status, size, content type)."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass
class CacheEntry:
    value: Any
    fetched_at: float = field(default_factory=time.time)


class TtlCache:
    """Keyed TTL cache with a maximum entry count (FIFO eviction)."""

    def __init__(self, max_entries: int = 64) -> None:
        self._store: dict[str, CacheEntry] = {}
        self._max = max_entries

    def get(self, key: str, ttl: float) -> tuple[bool, Any]:
        """Return (hit, value). Expired entries count as misses."""
        entry = self._store.get(key)
        if entry is None:
            return False, None
        if time.time() - entry.fetched_at >= ttl:
            return False, None
        return True, entry.value

    def set(self, key: str, value: Any) -> None:
        self._store[key] = CacheEntry(value)
        if len(self._store) > self._max:
            oldest = min(self._store, key=lambda k: self._store[k].fetched_at)
            del self._store[oldest]

    def clear(self) -> None:
        self._store.clear()


@dataclass
class CachedFetch:
    """A fetch strategy: URL template + per-source TTL in seconds."""

    url: str
    ttl: float
    cache_key: str = ""
    timeout: float | None = None

    def key(self, **kwargs: Any) -> str:
        if self.cache_key:
            return self.cache_key.format(**kwargs)
        return self.url.format(**kwargs)


class HttpClient:
    """Wraps one aiohttp session with validation, retries, and TTL caching."""

    def __init__(
        self,
        session: aiohttp.ClientSession,
        timeout_seconds: float = 10.0,
        cache: TtlCache | None = None,
        retry_attempts: int = RETRY_ATTEMPTS,
        origin_request_interval_seconds: float = ORIGIN_REQUEST_INTERVAL_SECONDS,
        origin_request_interval_overrides_seconds: dict[str, float] | None = None,
    ) -> None:
        self.session = session
        self.timeout_seconds = timeout_seconds
        self.cache = cache or TtlCache()
        self.retry_attempts = retry_attempts
        self.origin_request_interval_seconds = max(0.0, origin_request_interval_seconds)
        self.origin_request_interval_overrides_seconds = dict(
            ORIGIN_REQUEST_INTERVAL_OVERRIDES_SECONDS
        )
        if origin_request_interval_overrides_seconds:
            self.origin_request_interval_overrides_seconds.update(
                origin_request_interval_overrides_seconds
            )
        self._origin_locks: dict[str, asyncio.Lock] = {}
        self._origin_next_request: dict[str, float] = {}

    # -- low-level ---------------------------------------------------------

    async def _pace_origin(self, url: str) -> None:
        """Space requests to one HTTP origin while leaving other origins free."""
        origin = urlsplit(url).netloc.lower()
        interval = max(
            self.origin_request_interval_seconds,
            self.origin_request_interval_overrides_seconds.get(origin, 0.0),
        )
        if not interval:
            return
        lock = self._origin_locks.setdefault(origin, asyncio.Lock())
        async with lock:
            loop = asyncio.get_running_loop()
            delay = self._origin_next_request.get(origin, 0.0) - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
            self._origin_next_request[origin] = loop.time() + interval

    async def _request_bytes(self, url: str, headers: dict[str, str] | None) -> bytes:
        """GET with bounded size; raises FetchError on bad status/content-type."""
        for attempt in range(1, self.retry_attempts + 1):
            try:
                await self._pace_origin(url)
                async with self.session.get(
                    url,
                    headers=_with_user_agent(headers),
                    timeout=aiohttp.ClientTimeout(total=self.timeout_seconds),
                ) as resp:
                    if resp.status != 200:
                        raise FetchError(
                            f"HTTP {resp.status} for {url}", status_code=resp.status
                        )
                    ct = resp.headers.get("Content-Type", "")
                    if not ct or ct.lower().startswith("text/html"):
                        # Some providers send HTML error pages on failure; treat as error.
                        raise FetchError(f"Unexpected content type {ct!r} for {url}")
                    # read() decompresses gzip and returns the full body;
                    # a bounded read() would truncate it.
                    data = await resp.read()
                    if len(data) > MAX_BYTES_IMAGE:
                        raise FetchError(f"Response too large for {url}")
                    return data
            except (TimeoutError, aiohttp.ClientError, FetchError) as exc:
                # Retrying a forbidden request immediately amplifies an origin
                # rate limit. Other terminal client errors are equally unlikely
                # to recover without changing the request.
                if (
                    isinstance(exc, FetchError)
                    and exc.status_code is not None
                    and 400 <= exc.status_code < 500
                    and exc.status_code not in {408, 429}
                ):
                    raise
                if attempt >= self.retry_attempts:
                    raise
                delay = min(RETRY_BASE_DELAY * (2 ** (attempt - 1)), RETRY_MAX_DELAY)
                log.warning(
                    "fetch %s failed (attempt %d): %s; retrying in %.1fs", url, attempt, exc, delay
                )
                await asyncio.sleep(delay)
        raise FetchError(f"Exhausted retries for {url}")

    # -- typed helpers ------------------------------------------------------

    async def fetch_bytes(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        max_bytes: int = MAX_BYTES_IMAGE,
    ) -> bytes:
        data = await self._request_bytes(url, headers)
        if len(data) > max_bytes:
            raise FetchError(f"Response too large for {url}")
        return data

    async def fetch_text(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        max_bytes: int = MAX_BYTES_TEXT,
    ) -> str:
        data = await self.fetch_bytes(url, headers, max_bytes)
        return data.decode("utf-8", errors="replace")

    async def fetch_json(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        max_bytes: int = MAX_BYTES_TEXT,
    ) -> Any:
        text = await self.fetch_text(url, headers, max_bytes)
        try:
            return __import__("json").loads(text)
        except ValueError as exc:
            raise FetchError(f"Invalid JSON from {url}") from exc

    async def fetch_xml_text(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        max_bytes: int = MAX_BYTES_TEXT,
    ) -> str:
        text = await self.fetch_text(url, headers, max_bytes)
        if "<" not in text[:200]:
            raise FetchError(f"Not XML from {url}")
        return text

    async def post_form_json(
        self,
        url: str,
        data: dict[str, str],
        headers: dict[str, str] | None = None,
        max_bytes: int = MAX_BYTES_TEXT,
        timeout_seconds: float | None = None,
        attempts: int | None = None,
    ) -> Any:
        """POST form-encoded data and parse the JSON response."""
        total_timeout = aiohttp.ClientTimeout(
            total=timeout_seconds or max(self.timeout_seconds, 30.0)
        )
        tries = attempts or self.retry_attempts
        body = b""
        for attempt in range(1, tries + 1):
            try:
                await self._pace_origin(url)
                async with self.session.post(
                    url,
                    data=data,
                    headers=_with_user_agent(headers),
                    timeout=total_timeout,
                ) as resp:
                    if resp.status != 200:
                        raise FetchError(
                            f"HTTP {resp.status} for {url}", status_code=resp.status
                        )
                    ct = resp.headers.get("Content-Type", "")
                    if "json" not in ct.lower():
                        raise FetchError(f"Unexpected content type {ct!r} for {url}")
                    body = await resp.read()
                    if len(body) > max_bytes:
                        raise FetchError(f"Response too large for {url}")
                    break
            except (TimeoutError, aiohttp.ClientError, FetchError) as exc:
                if (
                    isinstance(exc, FetchError)
                    and exc.status_code is not None
                    and 400 <= exc.status_code < 500
                    and exc.status_code not in {408, 429}
                ):
                    raise
                if attempt >= tries:
                    raise
                delay = min(RETRY_BASE_DELAY * (2 ** (attempt - 1)), RETRY_MAX_DELAY)
                log.warning(
                    "POST %s failed (attempt %d): %s; retrying in %.1fs",
                    url, attempt, exc, delay,
                )
                await asyncio.sleep(delay)
        else:
            raise FetchError(f"Exhausted retries for {url}")
        try:
            return json.loads(body.decode("utf-8", errors="replace"))
        except ValueError as exc:
            raise FetchError(f"Invalid JSON from {url}") from exc

    # -- cached fetch with stale-on-error ------------------------------------

    async def _fetch_cached(
        self,
        spec: CachedFetch,
        fetcher: Callable[[str], Awaitable[Any]],
        **url_kwargs: Any,
    ) -> tuple[bool, Any, float]:
        """Cache any typed fetcher while retaining an expired value on error."""
        key = spec.key(**url_kwargs)
        hit, value = self.cache.get(key, spec.ttl)
        if hit:
            return False, value, self.cache._store[key].fetched_at  # noqa: SLF001
        try:
            value = await fetcher(spec.url.format(**url_kwargs))
        except FetchError as exc:
            old_hit, old = self.cache.get(key, ttl=float("inf"))
            if old_hit:
                log.warning("stale-on-error for %s: %s", key, exc)
                return True, old, self.cache._store[key].fetched_at  # noqa: SLF001
            raise
        self.cache.set(key, value)
        return False, value, self.cache._store[key].fetched_at

    async def fetch_text_cached(
        self,
        spec: CachedFetch,
        headers: dict[str, str] | None = None,
        max_bytes: int = MAX_BYTES_TEXT,
        **url_kwargs: Any,
    ) -> tuple[bool, str, float]:
        """Cached UTF-8 text with stale-on-error fallback."""
        return await self._fetch_cached(
            spec,
            lambda url: self.fetch_text(url, headers, max_bytes),
            **url_kwargs,
        )

    async def fetch_xml_text_cached(
        self,
        spec: CachedFetch,
        headers: dict[str, str] | None = None,
        max_bytes: int = MAX_BYTES_TEXT,
        **url_kwargs: Any,
    ) -> tuple[bool, str, float]:
        """Cached XML text, preserving the existing lightweight validation."""
        return await self._fetch_cached(
            spec,
            lambda url: self.fetch_xml_text(url, headers, max_bytes),
            **url_kwargs,
        )

    async def fetch_json_cached(
        self,
        spec: CachedFetch,
        headers: dict[str, str] | None = None,
        **url_kwargs: Any,
    ) -> tuple[bool, Any, float]:
        """Return (is_stale, value, fetched_at_unix).

        On a fresh cache hit: (False, value, fetched_at).
        On a cache miss, fetch; if the fetch fails and a previous value exists,
        return it marked stale: (True, value, previous_fetched_at).
        """
        return await self._fetch_cached(
            spec,
            lambda url: self.fetch_json(url, headers),
            **url_kwargs,
        )

    def utcnow(self) -> datetime:
        return datetime.now(UTC)

    async def gather_any(self, awaitables: list[Any]) -> list[Any]:
        """Run awaitables concurrently; return results with exceptions included
        so callers can isolate per-source failures."""
        return await asyncio.gather(*awaitables, return_exceptions=True)


def as_datetime(value: Any) -> datetime | None:
    """Parse an ISO-8601 string (or return a datetime as-is)."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value:
        from datetime import datetime as _dt

        try:
            return _dt.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None
