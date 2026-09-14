"""HttpClient tests: caching, stale-on-error, and per-origin pacing."""

import asyncio
import time

import aiohttp
import pytest

from dashboard.http import (
    USER_AGENT,
    CachedFetch,
    FetchError,
    HttpClient,
    RequestNotStarted,
    TtlCache,
)


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
@pytest.mark.parametrize(
    "failure",
    [TimeoutError("timed out"), aiohttp.ClientError("connection failed")],
)
async def test_fetch_json_cached_stale_on_terminal_transport_error(monkeypatch, failure):
    client = HttpClient(object(), retry_attempts=1)
    calls = 0

    async def fetch_json(_url, _headers=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            return {"a": 1}
        raise failure

    monkeypatch.setattr(client, "fetch_json", fetch_json)
    spec = CachedFetch("https://example.test/data.json", ttl=60, cache_key="data")
    _, value, _ = await client.fetch_json_cached(spec)
    client.cache._store["data"].fetched_at = time.time() - 61  # noqa: SLF001
    fetched_at = client.cache._store["data"].fetched_at  # noqa: SLF001

    stale, stale_value, stale_at = await client.fetch_json_cached(spec)
    assert stale is True
    assert stale_value == value
    assert stale_at == fetched_at

    client.cache._store["data"].fetched_at = 0  # noqa: SLF001
    monkeypatch.setattr(client, "fetch_json", lambda *_args, **_kwargs: _recovered_json())
    stale, value, recovered_at = await client.fetch_json_cached(spec)
    assert stale is False
    assert value == {"a": 2}
    assert recovered_at != fetched_at


async def _recovered_json():
    return {"a": 2}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", [asyncio.CancelledError(), RequestNotStarted("admission"), ValueError("bug")]
)
async def test_fetch_cached_propagates_non_transport_control_errors(monkeypatch, failure):
    client = HttpClient(object(), retry_attempts=1)

    async def fetch_json(_url, _headers=None):
        return {"a": 1}

    monkeypatch.setattr(client, "fetch_json", fetch_json)
    spec = CachedFetch("https://example.test/data.json", ttl=60, cache_key="control")
    await client.fetch_json_cached(spec)
    client.cache._store["control"].fetched_at = 0  # noqa: SLF001

    async def fail(_url, _headers=None):
        raise failure

    monkeypatch.setattr(client, "fetch_json", fail)
    with pytest.raises(type(failure)):
        await client.fetch_json_cached(spec)


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
async def test_fetch_html_explicitly_allows_html_but_api_text_rejects_it():
    class Response:
        status = 200
        headers = {"Content-Type": "text/html; charset=utf-8"}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def read(self):
            return b"<html><img alt='STRONG MONSOON SIGNAL'></html>"

    class Session:
        def get(self, *_args, **_kwargs):
            return Response()

    client = HttpClient(Session(), retry_attempts=1, origin_request_interval_seconds=0)
    assert "STRONG MONSOON" in await client.fetch_html("https://example.test/details")
    with pytest.raises(FetchError, match="Unexpected content type"):
        await client.fetch_text("https://example.test/details")
    Response.headers = {"Content-Type": "application/xhtml+xml"}
    assert "STRONG MONSOON" in await client.fetch_html("https://example.test/details")
    with pytest.raises(FetchError, match="Unexpected content type"):
        await client.fetch_text("https://example.test/details")


@pytest.mark.asyncio
async def test_fetch_html_cached_validator_keeps_last_good_catalog(monkeypatch):
    client = HttpClient(object(), retry_attempts=1)
    values = ["<img>", "invalid", "invalid"]

    async def fetch_html(_url, _headers=None, _max_bytes=None):
        return values.pop(0)

    def require_catalog(value):
        if value != "<img>":
            raise FetchError("invalid catalog")

    monkeypatch.setattr(client, "fetch_html", fetch_html)
    spec = CachedFetch("https://example.test/details", ttl=60, cache_key="details")
    stale, value, _ = await client.fetch_html_cached(spec, validator=require_catalog)
    assert not stale and value == "<img>"
    client.cache._store["details"].fetched_at = 0  # noqa: SLF001
    stale, value, _ = await client.fetch_html_cached(spec, validator=require_catalog)
    assert stale and value == "<img>"

    fresh_client = HttpClient(object(), retry_attempts=1)
    monkeypatch.setattr(fresh_client, "fetch_html", fetch_html)
    with pytest.raises(FetchError, match="invalid catalog"):
        await fresh_client.fetch_html_cached(spec, validator=require_catalog)


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


@pytest.mark.asyncio
async def test_origin_specific_pacing_override():
    client = HttpClient(
        object(),
        origin_request_interval_seconds=0,
        origin_request_interval_overrides_seconds={"slow.example": 0.04},
    )
    started: list[float] = []
    loop = asyncio.get_running_loop()

    async def paced() -> None:
        await client._pace_origin("https://slow.example/data")  # noqa: SLF001
        started.append(loop.time())

    await asyncio.gather(paced(), paced())

    assert started[1] - started[0] >= 0.025


def test_gmb_default_origin_pacing_override():
    client = HttpClient(object())

    assert client.gmb_eta_request_interval_seconds == 1.5
    assert client._origin_interval("https://data.etagmb.gov.hk/eta/stop/1") == 1.5  # noqa: SLF001
    assert client._origin_interval("https://data.etagmb.gov.hk/eta/route-stop/1/2") == 1.5  # noqa: SLF001
    assert client._origin_interval("https://data.etagmb.gov.hk/route-stop/1/2") == 0.2  # noqa: SLF001
    assert client._origin_interval("https://data.etagmb.gov.hk/stop/1") == 0.2  # noqa: SLF001
    assert client._origin_interval("https://other.example/eta/stop/1") == 0.06  # noqa: SLF001


def test_gmb_eta_pacing_is_configurable():
    client = HttpClient(object(), gmb_eta_request_interval_seconds=0)
    assert client._origin_interval("https://data.etagmb.gov.hk/eta/stop/1") == 0.2  # noqa: SLF001
    disabled = HttpClient(
        object(),
        origin_request_interval_seconds=0,
        origin_request_interval_overrides_seconds={"data.etagmb.gov.hk": 0},
        gmb_eta_request_interval_seconds=0,
    )
    assert disabled._origin_interval("https://data.etagmb.gov.hk/eta/stop/1") == 0  # noqa: SLF001
    custom = HttpClient(object(), gmb_eta_request_interval_seconds=0.04)
    assert custom._origin_interval("https://data.etagmb.gov.hk/eta/stop/1") == 0.2  # noqa: SLF001
    custom = HttpClient(
        object(),
        origin_request_interval_seconds=0,
        origin_request_interval_overrides_seconds={"data.etagmb.gov.hk": 0},
        gmb_eta_request_interval_seconds=0.04,
    )
    assert custom._origin_interval("https://data.etagmb.gov.hk/eta/stop/1") == 0.04  # noqa: SLF001


@pytest.mark.asyncio
async def test_http_403_is_not_retried():
    class ForbiddenResponse:
        status = 403
        headers: dict[str, str] = {}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

    class ForbiddenSession:
        def __init__(self) -> None:
            self.calls = 0

        def get(self, *_args, **_kwargs):
            self.calls += 1
            return ForbiddenResponse()

    session = ForbiddenSession()
    client = HttpClient(
        session,
        retry_attempts=3,
        origin_request_interval_seconds=0,
        origin_request_interval_overrides_seconds={"data.etagmb.gov.hk": 0},
    )

    with pytest.raises(FetchError, match="HTTP 403"):
        await client.fetch_json("https://example.test/data")

    assert session.calls == 1


@pytest.mark.asyncio
async def test_post_http_403_is_not_retried():
    class ForbiddenResponse:
        status = 403
        headers: dict[str, str] = {}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

    class ForbiddenSession:
        def __init__(self) -> None:
            self.calls = 0

        def post(self, *_args, **_kwargs):
            self.calls += 1
            return ForbiddenResponse()

    session = ForbiddenSession()
    client = HttpClient(session, retry_attempts=3, origin_request_interval_seconds=0)

    with pytest.raises(FetchError, match="HTTP 403"):
        await client.post_form_json(
            "https://example.test/data", {"data": "query"}, attempts=3
        )

    assert session.calls == 1


@pytest.mark.asyncio
async def test_post_retry_skip_rethrows_prior_started_failure(monkeypatch):
    class UnavailableResponse:
        status = 503
        headers = {"Content-Type": "application/json"}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

    class Session:
        calls = 0

        def post(self, *_args, **_kwargs):
            self.calls += 1
            return UnavailableResponse()

    session = Session()
    client = HttpClient(session, retry_attempts=2, origin_request_interval_seconds=0)
    pace_calls = 0

    async def block_retry(_url):
        nonlocal pace_calls
        pace_calls += 1
        if pace_calls == 2:
            raise RequestNotStarted("retry admission closed")

    client._pace_origin = block_retry  # noqa: SLF001
    monkeypatch.setattr("dashboard.http.RETRY_BASE_DELAY", 0.0)

    with pytest.raises(FetchError, match="HTTP 503"):
        await client.post_form_json("https://example.test/data", {"data": "query"})

    assert session.calls == 1


class _UserAgentResponse:
    status = 200
    headers = {"Content-Type": "application/json"}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def read(self):
        return b"{}"


class _UserAgentSession:
    def __init__(self) -> None:
        self.seen_headers: list[dict[str, str]] = []

    def get(self, _url, headers=None, **_kwargs):
        self.seen_headers.append(dict(headers))
        return _UserAgentResponse()


@pytest.mark.asyncio
async def test_requests_carry_the_dashboard_user_agent():
    session = _UserAgentSession()
    client = HttpClient(
        session,
        retry_attempts=1,
        origin_request_interval_seconds=0,
    )

    headers = {"Referer": "https://hkemobility.example/"}

    await client.fetch_json("https://hkemobility.example/data", headers=headers)

    sent = session.seen_headers[0]
    assert sent["User-Agent"] == USER_AGENT
    # caller-supplied headers are preserved alongside the dashboard user agent.
    assert sent["Referer"] == "https://hkemobility.example/"


@pytest.mark.asyncio
async def test_explicit_user_agent_override_is_case_insensitive():
    session = _UserAgentSession()
    client = HttpClient(session, retry_attempts=1, origin_request_interval_seconds=0)

    await client.fetch_json(
        "https://example.test/data", headers={"user-agent": "explicit-client/1.0"}
    )

    assert session.seen_headers[0] == {"user-agent": "explicit-client/1.0"}
