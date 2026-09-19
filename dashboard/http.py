"""Shared async HTTP layer: one injected aiohttp session, validated fetching,
source-specific TTL caching, and stale-on-error behavior.

All providers receive the ``HttpClient`` instance created once in ``bot.py``.
There are no import-time side effects and no module-level network calls.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit

import aiohttp

log = logging.getLogger(__name__)

USER_AGENT = "hkust-data-dashboard/2.0 (+https://github.com/hkust-fys/hkust-data-dashboard)"

_HKO_API_HOST = "data.weather.gov.hk"
_HKO_API_PATH = "/weatherAPI/opendata/weather.php"
_HKO_SAFE_QUERY_VALUES = {
    "dataType": frozenset({"rhrread", "warnsum", "warningInfo"}),
    "lang": frozenset({"en", "tc"}),
}
_SAFE_FETCH_ERROR_DETAILS = frozenset(
    {
        "HKO warning details contained no usable static PNG catalog",
        "No route extent for official road lookup",
        "Invalid route extent for official road lookup",
        "Invalid official road geometry response",
        "Invalid official road coordinates",
        "Official road geometry was empty",
        "Official road geometry exceeded pagination limit",
        "Invalid complete road geometry",
        "Invalid road coordinate",
        "Incomplete road pagination",
        "Invalid TD special news XML",
        "Invalid TD special news HTML",
        "Unrecognized TD special news item schema",
        "Unrecognized TD special news page",
        "Unrecognized TD Chinese special news page",
        "Invalid RTHK traffic news HTML",
        "Unrecognized RTHK traffic news page",
    }
)


def safe_endpoint(url: str) -> str:
    """Return a log-safe endpoint, retaining only allowlisted HKO selectors."""
    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname
    except (TypeError, ValueError):
        return "<invalid endpoint>"
    if not hostname:
        return "<invalid endpoint>"

    try:
        port = parsed.port
    except ValueError:
        port = None
    authority = (
        f"[{hostname}]" if ":" in hostname and not hostname.startswith("[") else hostname
    )
    if port is not None:
        authority = f"{authority}:{port}"

    # Discord webhook tokens are path components, not query parameters. Hide
    # both webhook path credentials so a failure cannot expose either value.
    path_parts = parsed.path.split("/")
    for index, part in enumerate(path_parts):
        if part.casefold() == "webhooks":
            for secret_index in (index + 1, index + 2):
                if secret_index < len(path_parts):
                    path_parts[secret_index] = "<redacted>"
            break
    path = "/".join(path_parts) or "/"

    prefix = f"{parsed.scheme.lower()}://" if parsed.scheme else "//"
    endpoint = f"{prefix}{authority}{path}"

    if parsed.hostname.casefold() == _HKO_API_HOST and parsed.path == _HKO_API_PATH:
        try:
            query_parts = parse_qsl(
                parsed.query, keep_blank_values=False, max_num_fields=12
            )
        except ValueError:
            query_parts = []
        safe_values: dict[str, str] = {}
        seen: set[str] = set()
        duplicates: set[str] = set()
        for key, value in query_parts:
            allowed_values = _HKO_SAFE_QUERY_VALUES.get(key)
            if allowed_values is None:
                continue
            if key in seen:
                duplicates.add(key)
                safe_values.pop(key, None)
                continue
            seen.add(key)
            if value in allowed_values:
                safe_values[key] = value
        for key in duplicates:
            safe_values.pop(key, None)
        safe_query = [
            (key, safe_values[key])
            for key in ("dataType", "lang")
            if key in safe_values
        ]
        if safe_query:
            endpoint += f"?{urlencode(safe_query)}"
    return endpoint


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
# A live GMB ETA burst returned 403 despite staying within 20 starts per 30s.
# One shared origin clock spaces ETA starts so the 21st cannot start before the
# first leaves that window. Metadata keeps its faster, separately paced loader
# so a cold route-geometry refresh can still finish within its provider budget.
GMB_ETA_REQUEST_INTERVAL_SECONDS = 1.5
ORIGIN_REQUEST_INTERVAL_OVERRIDES_SECONDS = {
    # Keep GMB metadata requests on one shared origin budget.
    "data.etagmb.gov.hk": 0.2,
}


class FetchError(RuntimeError):
    """Raised when an HTTP request or response validation fails."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        method: str | None = None,
        endpoint: str | None = None,
        error_type: str | None = None,
        safe_detail: str | None = None,
        retry_after_s: float | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.method = method
        self.endpoint = endpoint
        self.error_type = error_type
        self.safe_detail = safe_detail
        self.retry_after_s = retry_after_s


def _http_failure(
    method: str,
    url: str,
    error_type: str,
    *,
    status_code: int | None = None,
    safe_detail: str | None = None,
    retry_after_s: float | None = None,
) -> FetchError:
    """Build a failure from sanitized request context and diagnostic detail."""
    method = method.upper()
    endpoint = safe_endpoint(url)
    message = f"{method} {endpoint} failed: {error_type}"
    if status_code is not None:
        message += f" (HTTP {status_code})"
    if safe_detail:
        message += f": {safe_detail}"
    return FetchError(
        message,
        status_code=status_code,
        method=method,
        endpoint=endpoint,
        error_type=error_type,
        safe_detail=safe_detail,
        retry_after_s=retry_after_s,
    )


def _with_http_context(method: str, url: str, error: Exception) -> FetchError:
    """Attach safe request context to transport and typed-fetch failures."""
    status_code = error.status_code if isinstance(error, FetchError) else None
    error_type = error.error_type if isinstance(error, FetchError) else None
    if status_code is not None:
        error_type = "HTTPStatusError"
    if not error_type:
        error_type = type(error).__name__
    safe_detail = None
    if isinstance(error, FetchError):
        if error.safe_detail in _SAFE_FETCH_ERROR_DETAILS:
            safe_detail = error.safe_detail
        elif str(error) in _SAFE_FETCH_ERROR_DETAILS:
            safe_detail = str(error)
    retry_after_s = error.retry_after_s if isinstance(error, FetchError) else None
    return _http_failure(
        method,
        url,
        error_type,
        status_code=status_code,
        safe_detail=safe_detail,
        retry_after_s=retry_after_s,
    )


def _retry_after_seconds(value: str | None) -> float | None:
    """Parse Retry-After into a safe delay without retaining its raw value."""
    if not value:
        return None
    if len(value) > 128:
        return None
    value = value.strip()
    if value.isascii() and value.isdecimal():
        try:
            delay = float(int(value))
        except (OverflowError, ValueError):
            return None
        return delay if math.isfinite(delay) else None

    try:
        retry_at = parsedate_to_datetime(value)
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=UTC)
        delay = (retry_at.astimezone(UTC) - datetime.now(UTC)).total_seconds()
    except (OverflowError, TypeError, ValueError):
        return None
    return max(0.0, delay) if math.isfinite(delay) else None


def _log_http_failure(
    method: str,
    url: str,
    failure: FetchError,
    *,
    attempt: int,
    total_attempts: int,
    started_at: float,
    timeout_s: float,
    retry_delay_s: float | None = None,
) -> None:
    """Log one failed attempt using only sanitized endpoint/error metadata."""
    fields = [
        f"method={method.upper()}",
        f"endpoint={failure.endpoint or safe_endpoint(url)}",
        f"error_type={failure.error_type or type(failure).__name__}",
        f"status_code={failure.status_code if failure.status_code is not None else '-'}",
        f"attempt={attempt}/{total_attempts}",
        f"elapsed_s={max(0.0, time.monotonic() - started_at):.3f}",
        f"timeout_s={timeout_s:.3f}",
    ]
    if failure.retry_after_s is not None:
        fields.append(f"retry_after_s={failure.retry_after_s:.3f}")
    if retry_delay_s is not None:
        fields.append(f"retry_delay_s={retry_delay_s:.3f}")
        fields.append("retrying=true")
    log.warning("HTTP request failed %s", " ".join(fields))


class RequestNotStarted(RuntimeError):
    """Raised when admission or pacing aborts before the HTTP request starts."""


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
        gmb_eta_request_interval_seconds: float = GMB_ETA_REQUEST_INTERVAL_SECONDS,
    ) -> None:
        self.session = session
        self.timeout_seconds = timeout_seconds
        self.cache = cache or TtlCache()
        self.retry_attempts = retry_attempts
        self.origin_request_interval_seconds = max(0.0, origin_request_interval_seconds)
        self.gmb_eta_request_interval_seconds = max(0.0, gmb_eta_request_interval_seconds)
        self.origin_request_interval_overrides_seconds = dict(
            ORIGIN_REQUEST_INTERVAL_OVERRIDES_SECONDS
        )
        if origin_request_interval_overrides_seconds:
            self.origin_request_interval_overrides_seconds.update(
                origin_request_interval_overrides_seconds
            )
        self._origin_locks: dict[str, asyncio.Lock] = {}
        self._origin_next_request: dict[str, float] = {}

    def _origin_interval(self, url: str) -> float:
        """Return pacing for a URL, with stricter pacing for GMB ETA calls."""
        parsed = urlsplit(url)
        origin = parsed.netloc.lower()
        interval = max(
            self.origin_request_interval_seconds,
            self.origin_request_interval_overrides_seconds.get(origin, 0.0),
        )
        if origin == "data.etagmb.gov.hk" and parsed.path.startswith("/eta/"):
            interval = max(interval, self.gmb_eta_request_interval_seconds)
        return interval

    # -- low-level ---------------------------------------------------------

    async def _pace_origin(self, url: str) -> None:
        """Space requests to one HTTP origin while leaving other origins free."""
        origin = urlsplit(url).netloc.lower()
        interval = self._origin_interval(url)
        if not interval:
            return
        lock = self._origin_locks.setdefault(origin, asyncio.Lock())
        async with lock:
            loop = asyncio.get_running_loop()
            delay = self._origin_next_request.get(origin, 0.0) - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
            self._origin_next_request[origin] = loop.time() + interval

    async def _request_bytes(
        self,
        url: str,
        headers: dict[str, str] | None,
        *,
        allow_html: bool = False,
        max_bytes: int = MAX_BYTES_IMAGE,
    ) -> bytes:
        """GET with bounded size; raises FetchError on bad status/content-type."""
        last_started_error: FetchError | None = None
        for attempt in range(1, self.retry_attempts + 1):
            attempt_started: float | None = None
            try:
                await self._pace_origin(url)
                attempt_started = time.monotonic()
                async with self.session.get(
                    url,
                    headers=_with_user_agent(headers),
                    timeout=aiohttp.ClientTimeout(total=self.timeout_seconds),
                ) as resp:
                    if resp.status != 200:
                        retry_after_s = (
                            _retry_after_seconds(resp.headers.get("Retry-After"))
                            if resp.status in {429, 503}
                            else None
                        )
                        raise _http_failure(
                            "GET",
                            url,
                            "HTTPStatusError",
                            status_code=resp.status,
                            retry_after_s=retry_after_s,
                        )
                    ct = resp.headers.get("Content-Type", "")
                    content_type = ct.lower()
                    is_html = content_type.startswith(("text/html", "application/xhtml+xml"))
                    if not ct or (allow_html and not is_html) or (is_html and not allow_html):
                        # Some providers send HTML error pages on failure; treat as error.
                        raise _http_failure("GET", url, "UnexpectedContentType")
                    # read() decompresses gzip and returns the full body;
                    # a bounded read() would truncate it.
                    data = await resp.read()
                    if len(data) > max_bytes:
                        raise _http_failure("GET", url, "ResponseTooLarge")
                    return data
            except RequestNotStarted:
                if last_started_error is not None:
                    raise last_started_error from None
                raise
            except (TimeoutError, aiohttp.ClientError, FetchError) as exc:
                failure = _with_http_context("GET", url, exc)
                # Retrying a forbidden request immediately amplifies an origin
                # rate limit. Other terminal client errors are equally unlikely
                # to recover without changing the request.
                terminal_status = (
                    failure.status_code is not None
                    and 400 <= failure.status_code < 500
                    and failure.status_code not in {408, 429}
                )
                retry_delay_s = None
                if not terminal_status and attempt < self.retry_attempts:
                    retry_delay_s = min(
                        RETRY_BASE_DELAY * (2 ** (attempt - 1)), RETRY_MAX_DELAY
                    )
                _log_http_failure(
                    "GET",
                    url,
                    failure,
                    attempt=attempt,
                    total_attempts=self.retry_attempts,
                    started_at=(
                        attempt_started
                        if attempt_started is not None
                        else time.monotonic()
                    ),
                    timeout_s=self.timeout_seconds,
                    retry_delay_s=retry_delay_s,
                )
                if retry_delay_s is None:
                    raise failure from None
                last_started_error = failure
                await asyncio.sleep(retry_delay_s)
        raise _http_failure("GET", url, "RetriesExhausted")

    # -- typed helpers ------------------------------------------------------

    async def fetch_bytes(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        max_bytes: int = MAX_BYTES_IMAGE,
    ) -> bytes:
        data = await self._request_bytes(url, headers, max_bytes=max_bytes)
        if len(data) > max_bytes:
            raise _http_failure("GET", url, "ResponseTooLarge")
        return data

    async def fetch_text(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        max_bytes: int = MAX_BYTES_TEXT,
    ) -> str:
        data = await self.fetch_bytes(url, headers, max_bytes)
        return data.decode("utf-8", errors="replace")

    async def fetch_html(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        max_bytes: int = MAX_BYTES_TEXT,
    ) -> str:
        """Fetch bounded HTML explicitly; ordinary API fetches still reject it."""
        data = await self._request_bytes(url, headers, allow_html=True, max_bytes=max_bytes)
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
        except ValueError:
            raise _http_failure("GET", url, "InvalidJSON") from None

    async def fetch_xml_text(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        max_bytes: int = MAX_BYTES_TEXT,
    ) -> str:
        text = await self.fetch_text(url, headers, max_bytes)
        if "<" not in text[:200]:
            raise _http_failure("GET", url, "InvalidXML")
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
        request_timeout_s = timeout_seconds or max(self.timeout_seconds, 30.0)
        total_timeout = aiohttp.ClientTimeout(total=request_timeout_s)
        tries = attempts or self.retry_attempts
        body = b""
        last_started_error: FetchError | None = None
        for attempt in range(1, tries + 1):
            attempt_started: float | None = None
            try:
                await self._pace_origin(url)
                attempt_started = time.monotonic()
                async with self.session.post(
                    url,
                    data=data,
                    headers=_with_user_agent(headers),
                    timeout=total_timeout,
                ) as resp:
                    if resp.status != 200:
                        retry_after_s = (
                            _retry_after_seconds(resp.headers.get("Retry-After"))
                            if resp.status in {429, 503}
                            else None
                        )
                        raise _http_failure(
                            "POST",
                            url,
                            "HTTPStatusError",
                            status_code=resp.status,
                            retry_after_s=retry_after_s,
                        )
                    ct = resp.headers.get("Content-Type", "")
                    if "json" not in ct.lower():
                        raise _http_failure("POST", url, "UnexpectedContentType")
                    body = await resp.read()
                    if len(body) > max_bytes:
                        raise _http_failure("POST", url, "ResponseTooLarge")
                    break
            except RequestNotStarted:
                if last_started_error is not None:
                    raise last_started_error from None
                raise
            except (TimeoutError, aiohttp.ClientError, FetchError) as exc:
                failure = _with_http_context("POST", url, exc)
                terminal_status = (
                    failure.status_code is not None
                    and 400 <= failure.status_code < 500
                    and failure.status_code not in {408, 429}
                )
                retry_delay_s = None
                if not terminal_status and attempt < tries:
                    retry_delay_s = min(
                        RETRY_BASE_DELAY * (2 ** (attempt - 1)), RETRY_MAX_DELAY
                    )
                _log_http_failure(
                    "POST",
                    url,
                    failure,
                    attempt=attempt,
                    total_attempts=tries,
                    started_at=(
                        attempt_started
                        if attempt_started is not None
                        else time.monotonic()
                    ),
                    timeout_s=request_timeout_s,
                    retry_delay_s=retry_delay_s,
                )
                if retry_delay_s is None:
                    raise failure from None
                last_started_error = failure
                await asyncio.sleep(retry_delay_s)
        else:
            raise _http_failure("POST", url, "RetriesExhausted")
        try:
            return json.loads(body.decode("utf-8", errors="replace"))
        except ValueError:
            raise _http_failure("POST", url, "InvalidJSON") from None

    # -- cached fetch with stale-on-error ------------------------------------

    async def _fetch_cached(
        self,
        spec: CachedFetch,
        fetcher: Callable[[str], Awaitable[Any]],
        **url_kwargs: Any,
    ) -> tuple[bool, Any, float]:
        """Cache any typed fetcher while retaining an expired value on error."""
        key = spec.key(**url_kwargs)
        url = spec.url.format(**url_kwargs)
        hit, value = self.cache.get(key, spec.ttl)
        if hit:
            return False, value, self.cache._store[key].fetched_at  # noqa: SLF001
        try:
            value = await fetcher(url)
        except (TimeoutError, aiohttp.ClientError, FetchError) as exc:
            failure = _with_http_context("GET", url, exc)
            old_hit, old = self.cache.get(key, ttl=float("inf"))
            if old_hit:
                fetched_at = self.cache._store[key].fetched_at  # noqa: SLF001
                cache_age_s = max(0.0, time.time() - fetched_at)
                log.warning(
                    "stale-on-error: %s cache_age_s=%.3f ttl_s=%.3f",
                    failure,
                    cache_age_s,
                    spec.ttl,
                )
                return True, old, fetched_at
            raise failure from None
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

    async def fetch_html_cached(
        self,
        spec: CachedFetch,
        headers: dict[str, str] | None = None,
        max_bytes: int = MAX_BYTES_TEXT,
        validator: Callable[[str], None] | None = None,
        **url_kwargs: Any,
    ) -> tuple[bool, str, float]:
        """Cached bounded HTML with stale-on-error fallback."""
        return await self._fetch_cached(
            spec,
            lambda url: self._fetch_validated_html(url, headers, max_bytes, validator),
            **url_kwargs,
        )

    async def _fetch_validated_html(
        self,
        url: str,
        headers: dict[str, str] | None,
        max_bytes: int,
        validator: Callable[[str], None] | None,
    ) -> str:
        value = await self.fetch_html(url, headers, max_bytes)
        if validator is not None:
            try:
                validator(value)
            except FetchError as exc:
                raise _with_http_context("GET", url, exc) from None
        return value

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
