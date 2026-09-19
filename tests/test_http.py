"""HttpClient tests: caching, stale-on-error, and per-origin pacing."""

import asyncio
import logging
import time
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import aiohttp
import pytest

from dashboard.http import (
    USER_AGENT,
    CachedFetch,
    FetchError,
    HttpClient,
    RequestNotStarted,
    TtlCache,
    safe_endpoint,
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
    with pytest.raises(FetchError, match="UnexpectedContentType"):
        await client.fetch_text("https://example.test/details")
    Response.headers = {"Content-Type": "application/xhtml+xml"}
    assert "STRONG MONSOON" in await client.fetch_html("https://example.test/details")
    with pytest.raises(FetchError, match="UnexpectedContentType"):
        await client.fetch_text("https://example.test/details")


@pytest.mark.asyncio
async def test_fetch_html_cached_validator_keeps_last_good_catalog(monkeypatch):
    client = HttpClient(object(), retry_attempts=1)
    values = ["<img>", "invalid", "invalid"]
    warning_detail = "HKO warning details contained no usable static PNG catalog"

    async def fetch_html(_url, _headers=None, _max_bytes=None):
        return values.pop(0)

    def require_catalog(value):
        if value != "<img>":
            raise FetchError(warning_detail)

    monkeypatch.setattr(client, "fetch_html", fetch_html)
    spec = CachedFetch(
        "https://user:password@data.weather.gov.hk/weatherAPI/opendata/weather.php"
        "?dataType=warningInfo&lang=en&api_key=query-secret",
        ttl=60,
        cache_key="details",
    )
    stale, value, _ = await client.fetch_html_cached(spec, validator=require_catalog)
    assert not stale and value == "<img>"
    client.cache._store["details"].fetched_at = 0  # noqa: SLF001
    stale, value, _ = await client.fetch_html_cached(spec, validator=require_catalog)
    assert stale and value == "<img>"

    fresh_client = HttpClient(object(), retry_attempts=1)
    monkeypatch.setattr(fresh_client, "fetch_html", fetch_html)
    with pytest.raises(FetchError) as caught:
        await fresh_client.fetch_html_cached(spec, validator=require_catalog)
    assert caught.value.method == "GET"
    assert caught.value.endpoint == (
        "https://data.weather.gov.hk/weatherAPI/opendata/weather.php"
        "?dataType=warningInfo&lang=en"
    )
    assert caught.value.error_type == "FetchError"
    assert caught.value.safe_detail == warning_detail
    assert warning_detail in str(caught.value)
    assert "password" not in str(caught.value)
    assert "query-secret" not in str(caught.value)


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
async def test_http_403_is_not_retried(caplog):
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
        timeout_seconds=6.25,
        retry_attempts=3,
        origin_request_interval_seconds=0,
        origin_request_interval_overrides_seconds={"data.etagmb.gov.hk": 0},
    )
    url = "https://user:password@example.test/data?api_key=query-secret"

    with (
        caplog.at_level(logging.WARNING, logger="dashboard.http"),
        pytest.raises(FetchError, match="HTTP 403") as caught,
    ):
        await client.fetch_json(url)

    assert session.calls == 1
    assert caught.value.method == "GET"
    assert caught.value.endpoint == "https://example.test/data"
    assert caught.value.error_type == "HTTPStatusError"
    assert caught.value.status_code == 403
    assert (
        "HTTP request failed method=GET endpoint=https://example.test/data "
        "error_type=HTTPStatusError status_code=403 attempt=1/3 "
        "elapsed_s="
    ) in caplog.text
    assert "timeout_s=6.250" in caplog.text
    assert "retry_delay_s=" not in caplog.text
    assert "user" not in str(caught.value)
    assert "password" not in str(caught.value)
    assert "query-secret" not in str(caught.value)


@pytest.mark.asyncio
async def test_post_http_403_is_not_retried(caplog):
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
    client = HttpClient(
        session,
        timeout_seconds=10,
        retry_attempts=3,
        origin_request_interval_seconds=0,
    )
    url = "https://user:password@example.test/data?api_key=query-secret"

    with (
        caplog.at_level(logging.WARNING, logger="dashboard.http"),
        pytest.raises(FetchError, match="HTTP 403") as caught,
    ):
        await client.post_form_json(
            url, {"data": "body-secret"}, attempts=3, timeout_seconds=6.25
        )

    assert session.calls == 1
    assert caught.value.method == "POST"
    assert caught.value.endpoint == "https://example.test/data"
    assert caught.value.error_type == "HTTPStatusError"
    assert caught.value.status_code == 403
    assert (
        "HTTP request failed method=POST endpoint=https://example.test/data "
        "error_type=HTTPStatusError status_code=403 attempt=1/3 "
        "elapsed_s="
    ) in caplog.text
    assert "timeout_s=6.250" in caplog.text
    assert "retry_delay_s=" not in caplog.text
    assert "user" not in str(caught.value)
    assert "password" not in str(caught.value)
    assert "query-secret" not in str(caught.value)
    assert "body-secret" not in str(caught.value)


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


def test_safe_endpoint_removes_url_secrets():
    assert (
        safe_endpoint(
            "https://alice:password@example.test:8443/data?api_key=query-secret#fragment"
        )
        == "https://example.test:8443/data"
    )
    assert safe_endpoint(
        "https://discord.example/api/webhooks/wh-id-123/webhook-token-456?wait=true"
    ) == "https://discord.example/api/webhooks/<redacted>/<redacted>"


@pytest.mark.parametrize("data_type", ["rhrread", "warnsum", "warningInfo"])
def test_safe_endpoint_preserves_only_known_hko_selectors(data_type):
    url = (
        "https://alice:password@data.weather.gov.hk/weatherAPI/opendata/weather.php"
        f"?api_key=query-secret&dataType={data_type}&lang=en"
    )

    assert safe_endpoint(url) == (
        "https://data.weather.gov.hk/weatherAPI/opendata/weather.php"
        f"?dataType={data_type}&lang=en"
    )


@pytest.mark.asyncio
async def test_http_context_drops_unvalidated_fetcherror_detail(monkeypatch):
    client = HttpClient(object(), retry_attempts=1)
    body_secret = "private body token"

    async def fetch_html(_url, _headers=None, _max_bytes=None):
        raise FetchError(f"invalid catalog: {body_secret}")

    monkeypatch.setattr(client, "fetch_html", fetch_html)
    spec = CachedFetch("https://example.test/details?token=query-secret", ttl=60)

    with pytest.raises(FetchError) as caught:
        await client.fetch_html_cached(spec)

    assert caught.value.method == "GET"
    assert caught.value.endpoint == "https://example.test/details"
    assert caught.value.safe_detail is None
    assert "private body token" not in str(caught.value)
    assert "query-secret" not in str(caught.value)


class _RaisingResponse:
    def __init__(self, url: str) -> None:
        self.url = url

    async def __aenter__(self):
        raise TimeoutError(f"timed out requesting {self.url}; response-body-secret")

    async def __aexit__(self, *_args):
        return None


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["GET", "POST"])
async def test_terminal_timeout_has_safe_method_and_endpoint(method, caplog):
    class Session:
        def __init__(self) -> None:
            self.calls = 0

        def _request(self, url):
            self.calls += 1
            return _RaisingResponse(url)

        def get(self, url, **_kwargs):
            return self._request(url)

        def post(self, url, **_kwargs):
            return self._request(url)

    url = "https://alice:password@api.example.test/v1/data?token=query-secret"
    session = Session()
    client = HttpClient(
        session,
        timeout_seconds=4.5,
        retry_attempts=1,
        origin_request_interval_seconds=0,
    )

    with (
        caplog.at_level(logging.WARNING, logger="dashboard.http"),
        pytest.raises(FetchError) as caught,
    ):
        if method == "GET":
            await client.fetch_json(url)
        else:
            await client.post_form_json(
                url, {"payload": "request-body-secret"}, timeout_seconds=4.5
            )

    error = caught.value
    assert str(error) == f"{method} https://api.example.test/v1/data failed: TimeoutError"
    assert error.method == method
    assert error.endpoint == "https://api.example.test/v1/data"
    assert error.error_type == "TimeoutError"
    assert error.status_code is None
    assert session.calls == 1
    assert (
        f"HTTP request failed method={method} "
        "endpoint=https://api.example.test/v1/data "
        "error_type=TimeoutError status_code=- attempt=1/1 elapsed_s="
    ) in caplog.text
    assert "timeout_s=4.500" in caplog.text
    assert "retry_delay_s=" not in caplog.text
    for secret in ("alice", "password", "query-secret", "response-body-secret"):
        assert secret not in str(error)
    assert "request-body-secret" not in str(error)


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["GET", "POST"])
async def test_retry_log_has_safe_method_endpoint_and_error_type(method, monkeypatch, caplog):
    class Session:
        def __init__(self) -> None:
            self.calls = 0

        def _request(self, url):
            self.calls += 1
            if self.calls == 1:
                return _RaisingResponse(url)
            return _UserAgentResponse()

        def get(self, url, **_kwargs):
            return self._request(url)

        def post(self, url, **_kwargs):
            return self._request(url)

    url = (
        "https://alice:password@discord.example/api/webhooks/wh-id-123/"
        "webhook-token-456?api_key=query-secret"
    )
    session = Session()
    client = HttpClient(
        session,
        timeout_seconds=4.5,
        retry_attempts=2,
        origin_request_interval_seconds=0,
    )
    monkeypatch.setattr("dashboard.http.RETRY_BASE_DELAY", 0.0)

    with caplog.at_level(logging.WARNING, logger="dashboard.http"):
        if method == "GET":
            result = await client.fetch_json(url)
        else:
            result = await client.post_form_json(
                url, {"payload": "request-body-secret"}, timeout_seconds=4.5
            )

    endpoint = "https://discord.example/api/webhooks/<redacted>/<redacted>"
    assert result == {}
    assert session.calls == 2
    assert (
        f"HTTP request failed method={method} endpoint={endpoint} "
        "error_type=TimeoutError status_code=- attempt=1/2 elapsed_s="
    ) in caplog.text
    assert "timeout_s=4.500" in caplog.text
    assert "retry_delay_s=0.000 retrying=true" in caplog.text
    for secret in (
        "alice", "password", "wh-id-123", "webhook-token-456", "query-secret",
        "response-body-secret", "request-body-secret",
    ):
        assert secret not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "status", "header_kind"),
    [("GET", 429, "seconds"), ("POST", 503, "date")],
)
async def test_retry_after_is_logged_as_derived_delay(
    method, status, header_kind, monkeypatch, caplog
):
    if header_kind == "seconds":
        retry_after_header = "17"
    else:
        retry_after_header = format_datetime(
            datetime.now(UTC) + timedelta(seconds=90), usegmt=True
        )

    class RetryResponse:
        def __init__(self) -> None:
            self.status = status
            self.headers = {"Retry-After": retry_after_header}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

    class Session:
        def __init__(self) -> None:
            self.calls = 0

        def _request(self, _url):
            self.calls += 1
            if self.calls == 1:
                return RetryResponse()
            return _UserAgentResponse()

        def get(self, url, **_kwargs):
            return self._request(url)

        def post(self, url, **_kwargs):
            return self._request(url)

    session = Session()
    client = HttpClient(
        session,
        timeout_seconds=4.5,
        retry_attempts=2,
        origin_request_interval_seconds=0,
    )
    monkeypatch.setattr("dashboard.http.RETRY_BASE_DELAY", 0.0)
    url = "https://api.example.test/data"

    with caplog.at_level(logging.WARNING, logger="dashboard.http"):
        if method == "GET":
            result = await client.fetch_json(url)
        else:
            result = await client.post_form_json(url, {"payload": "query"}, timeout_seconds=4.5)

    assert result == {}
    assert session.calls == 2
    assert (
        f"method={method} endpoint={url} error_type=HTTPStatusError "
        f"status_code={status} attempt=1/2 elapsed_s="
    ) in caplog.text
    assert "retry_delay_s=0.000 retrying=true" in caplog.text
    assert "timeout_s=4.500" in caplog.text
    assert "Retry-After" not in caplog.text
    logged_retry_after = float(caplog.text.split("retry_after_s=", maxsplit=1)[1].split()[0])
    if header_kind == "seconds":
        assert logged_retry_after == 17
    else:
        assert 88 <= logged_retry_after <= 90
        assert retry_after_header not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["GET", "POST"])
async def test_oversized_retry_after_does_not_mask_status_or_leak_header(method, caplog):
    oversized_header = "9" * 5000

    class Response:
        status = 503
        headers = {"Retry-After": oversized_header}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

    class Session:
        def get(self, *_args, **_kwargs):
            return Response()

        def post(self, *_args, **_kwargs):
            return Response()

    url = "https://api.example.test/data"
    client = HttpClient(
        Session(), timeout_seconds=4.5, retry_attempts=1, origin_request_interval_seconds=0
    )

    with (
        caplog.at_level(logging.WARNING, logger="dashboard.http"),
        pytest.raises(FetchError) as caught,
    ):
        if method == "GET":
            await client.fetch_json(url)
        else:
            await client.post_form_json(url, {"payload": "query"}, timeout_seconds=4.5)

    assert caught.value.status_code == 503
    assert caught.value.error_type == "HTTPStatusError"
    assert "Retry-After" not in caplog.text
    assert oversized_header not in caplog.text
    assert "retry_after_s=" not in caplog.text
    assert (
        f"method={method} endpoint={url} error_type=HTTPStatusError "
        "status_code=503 attempt=1/1 elapsed_s="
    ) in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["GET", "POST"])
async def test_attempt_elapsed_starts_after_origin_pacing(method, monkeypatch, caplog):
    class FakeClock:
        def __init__(self) -> None:
            self.values = iter([100.0, 102.5])
            self.calls = 0

        def monotonic(self) -> float:
            self.calls += 1
            return next(self.values)

        @staticmethod
        def time() -> float:
            return time.time()

    class Session:
        def get(self, url, **_kwargs):
            return _RaisingResponse(url)

        def post(self, url, **_kwargs):
            return _RaisingResponse(url)

    clock = FakeClock()
    client = HttpClient(
        Session(), timeout_seconds=4.5, retry_attempts=1, origin_request_interval_seconds=0
    )

    async def pace(_url):
        assert clock.calls == 0

    client._pace_origin = pace  # noqa: SLF001
    monkeypatch.setattr("dashboard.http.time", clock)
    url = "https://api.example.test/data"

    with (
        caplog.at_level(logging.WARNING, logger="dashboard.http"),
        pytest.raises(FetchError),
    ):
        if method == "GET":
            await client.fetch_json(url)
        else:
            await client.post_form_json(url, {"payload": "query"}, timeout_seconds=4.5)

    assert clock.calls == 2
    assert "elapsed_s=2.500" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["GET", "POST"])
async def test_typed_json_decode_error_has_safe_request_context(method):
    class InvalidJsonResponse(_UserAgentResponse):
        async def read(self):
            return b"response-body-secret"

    class Session:
        def get(self, *_args, **_kwargs):
            return InvalidJsonResponse()

        def post(self, *_args, **_kwargs):
            return InvalidJsonResponse()

    url = "https://alice:password@api.example.test/data?api_key=query-secret"
    client = HttpClient(Session(), retry_attempts=1, origin_request_interval_seconds=0)

    with pytest.raises(FetchError) as caught:
        if method == "GET":
            await client.fetch_json(url)
        else:
            await client.post_form_json(url, {"payload": "request-body-secret"})

    error = caught.value
    assert error.method == method
    assert error.endpoint == "https://api.example.test/data"
    assert error.error_type == "InvalidJSON"
    for secret in ("alice", "password", "query-secret", "response-body-secret"):
        assert secret not in str(error)
    assert "request-body-secret" not in str(error)


@pytest.mark.asyncio
async def test_stale_cache_log_shows_safe_endpoint_not_cache_key(monkeypatch, caplog):
    url = "https://alice:password@example.test/data?api_key=query-secret"
    client = HttpClient(object(), retry_attempts=1)
    calls = 0

    async def fetch_json(_url, _headers=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            return {"ok": True}
        raise TimeoutError(f"timed out for {url}")

    monkeypatch.setattr(client, "fetch_json", fetch_json)
    spec = CachedFetch(url, ttl=60)
    await client.fetch_json_cached(spec)
    client.cache._store[spec.key()].fetched_at = time.time() - 75.0  # noqa: SLF001

    with caplog.at_level(logging.WARNING, logger="dashboard.http"):
        stale, value, _ = await client.fetch_json_cached(spec)

    assert stale is True
    assert value == {"ok": True}
    assert "stale-on-error: GET https://example.test/data failed: TimeoutError" in caplog.text
    assert "ttl_s=60.000" in caplog.text
    assert "cache_age_s=" in caplog.text
    logged_age = float(caplog.text.split("cache_age_s=", maxsplit=1)[1].split()[0])
    assert 75 <= logged_age < 76
    for secret in ("alice", "password", "api_key", "query-secret"):
        assert secret not in caplog.text
