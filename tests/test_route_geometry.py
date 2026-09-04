"""HKeMobility official route-geometry provider contracts."""

from __future__ import annotations

import asyncio
import sys
import types

import pytest

from dashboard.providers import route_geometry
from dashboard.providers.route_geometry import (
    RouteGeometry,
    RouteLine,
    RouteSpec,
    Stop,
)

EXPECTED_SPECS = (
    ("KMB", "91", "outbound", 1395, 1),
    ("KMB", "91", "inbound", 1395, 2),
    ("KMB", "91M", "outbound", 1398, 1),
    ("KMB", "91M", "inbound", 1398, 2),
    ("KMB", "91P", "outbound", 8093, 1),
    ("KMB", "91P", "inbound", 8426, 1),
    ("KMB", "291P", "outbound", 8710, 1),
    ("CTB", "792M", "outbound", 1616, 1),
    ("CTB", "792M", "inbound", 1616, 2),
    ("GMB", "11", "seq-1", 2004791, 1),
    ("GMB", "11", "seq-2", 2004791, 2),
    ("GMB", "11B", "seq-1", 2004828, 1),
    ("GMB", "11M", "seq-2", 2004825, 2),
    ("GMB", "11S", "seq-1", 2004826, 1),
    ("GMB", "11S", "seq-2", 2004826, 2),
    ("GMB", "12", "seq-1", 2004764, 1),
    ("GMB", "12", "seq-2", 2004764, 2),
    ("GMB", "104", "seq-1", 2007200, 1),
)


def _valid_line(spec: RouteSpec, suffix: str = "") -> RouteLine:
    stops = [
        Stop(f"{spec.route_id}-a", f"Alpha {suffix}", 22.3000, 114.1000),
        Stop(f"{spec.route_id}-b", f"Bravo {suffix}", 22.3000, 114.1020),
    ]
    return RouteLine(
        spec.route,
        spec.operator,
        spec.bound,
        stops,
        [(22.3000, 114.1000), (22.3000, 114.1010), (22.3000, 114.1020)],
        [0.0, 200.0],
        destination=f"Destination {suffix}",
        path_source="hkemobility",
    )


def test_route_specs_cover_all_tracked_directions_with_official_mapping():
    assert (
        tuple(
            (spec.operator, spec.route, spec.bound, spec.route_id, spec.sequence)
            for spec in route_geometry.ROUTE_SPECS
        )
        == EXPECTED_SPECS
    )


def test_hkemobility_lat_lon_shape_builds_monotonic_stop_offsets():
    spec = RouteSpec("GMB", "11", "seq-1", 2004791, 1)
    stops = [
        Stop("a", "Start", 22.3000, 114.1000),
        Stop("b", "Middle", 22.3000, 114.1010),
        Stop("c", "End", 22.3000, 114.1020),
    ]
    raw = {
        "e": "Choi Hung",
        "sh": {"coordinates": [[[22.3000, 114.1000], [22.3002, 114.1010], [22.3000, 114.1020]]]},
    }

    line = route_geometry._build_line(spec, stops, raw)

    assert line.path == [(22.3000, 114.1000), (22.3002, 114.1010), (22.3000, 114.1020)]
    assert line.destination == "Choi Hung"
    assert line.path_source == "hkemobility"
    assert len(line.stop_offsets) == len(stops)
    assert line.stop_offsets[0] < line.stop_offsets[1] < line.stop_offsets[2]
    assert route_geometry._valid_line(line)


def test_trailing_extension_stops_outside_spatial_line_keep_verified_prefix():
    spec = RouteSpec("KMB", "91P", "outbound", 8093, 1)
    stops = [
        Stop("a", "Current start", 22.3000, 114.1000),
        Stop("b", "Current terminus", 22.3000, 114.1020),
        Stop("c", "Future extension", 22.3050, 114.1050),
        Stop("d", "Future terminus", 22.3070, 114.1070),
    ]
    raw = {
        "e": "HKUST",
        "sh": {"coordinates": [[[22.3000, 114.1000], [22.3000, 114.1020]]]},
    }

    line = route_geometry._build_line(spec, stops, raw)

    assert [stop.stop_id for stop in line.stops] == ["a", "b"]
    assert len(line.stop_offsets) == 2
    assert route_geometry._valid_line(line)


def test_mid_route_stop_gap_does_not_create_partial_geometry():
    spec = RouteSpec("KMB", "91P", "outbound", 8093, 1)
    stops = [
        Stop("a", "Start", 22.3000, 114.1000),
        Stop("b", "Before gap", 22.3000, 114.1010),
        Stop("c", "Bad middle stop", 22.3050, 114.1050),
        Stop("d", "Later existing stop", 22.3000, 114.1020),
    ]
    raw = {
        "e": "HKUST",
        "sh": {"coordinates": [[[22.3000, 114.1000], [22.3000, 114.1020]]]},
    }

    line = route_geometry._build_line(spec, stops, raw)

    assert line.stops == stops
    assert line.stop_offsets == []
    assert not route_geometry._valid_line(line)


class _SpatialClient:
    def __init__(self) -> None:
        self.requests: list[tuple[str, dict[str, str] | None]] = []

    class _Response:
        ok = True
        status = 200

        async def json(self):
            return {"e": "Destination", "sh": {"coordinates": []}}

    async def get(self, url: str, headers: dict[str, str] | None = None, timeout: int = 0):
        self.requests.append((url, headers))
        assert timeout == route_geometry.SPATIAL_REQUEST_TIMEOUT_MS
        return self._Response()


@pytest.mark.parametrize(
    ("spec", "referer"),
    [
        (RouteSpec("KMB", "91", "outbound", 1395, 1), "/bus"),
        (RouteSpec("CTB", "792M", "inbound", 1616, 2), "/bus"),
        (RouteSpec("GMB", "11", "seq-2", 2004791, 2), "/gmb"),
    ],
)
async def test_spatial_request_uses_route_sequence_and_operator_referer(spec, referer):
    client = _SpatialClient()

    await route_geometry._fetch_spatial(client, spec)

    url, headers = client.requests[0]
    assert url.endswith(f"/{spec.route_id}/sequence/{spec.sequence}/spatial")
    assert headers == {
        "Referer": f"https://www.hkemobility.gov.hk/en/public-transport{referer}",
        "User-Agent": route_geometry.HKEMOBILITY_USER_AGENT,
    }
    assert headers["User-Agent"] == (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36 Edg/151.0.0.0"
    )


def test_system_ca_option_preserves_existing_node_options(monkeypatch):
    monkeypatch.setattr(route_geometry.os, "name", "nt")
    monkeypatch.setenv("NODE_OPTIONS", "--trace-warnings")
    previous = route_geometry._append_system_ca_node_option()
    assert previous == "--trace-warnings"
    assert "--trace-warnings" in route_geometry.os.environ["NODE_OPTIONS"]
    assert "--use-system-ca" in route_geometry.os.environ["NODE_OPTIONS"]
    route_geometry._restore_node_options(previous)
    assert route_geometry.os.environ["NODE_OPTIONS"] == "--trace-warnings"


async def test_spatial_batch_uses_one_shared_playwright_context(monkeypatch):
    contexts: list[_SpatialClient] = []

    class Request:
        async def new_context(self, *, user_agent):
            assert user_agent == route_geometry.HKEMOBILITY_USER_AGENT
            context = _SpatialClient()
            context.disposed = False

            async def dispose():
                context.disposed = True

            context.dispose = dispose
            contexts.append(context)
            return context

    class Manager:
        async def __aenter__(self):
            return types.SimpleNamespace(request=Request())

        async def __aexit__(self, *_args):
            return None

    fake_api = types.ModuleType("playwright.async_api")
    fake_api.async_playwright = lambda: Manager()
    monkeypatch.setitem(sys.modules, "playwright.async_api", fake_api)
    monkeypatch.setattr(route_geometry, "PACE_MAX_INTERVAL_SECONDS", 0)
    monkeypatch.setattr(route_geometry, "PACE_WINDOW_SECONDS", 0)

    results = await route_geometry._fetch_spatial_batch()

    assert len(results) == len(route_geometry.ROUTE_SPECS)
    assert len(contexts) == 1
    assert len(contexts[0].requests) == len(route_geometry.ROUTE_SPECS)
    assert contexts[0].disposed is True


async def test_paced_gather_interleaves_hosts_but_spaces_one_host(monkeypatch):
    monkeypatch.setattr(route_geometry, "PACE_WINDOW_SECONDS", 0.05)
    monkeypatch.setattr(route_geometry, "PACE_MAX_INTERVAL_SECONDS", 0.05)
    started: dict[str, float] = {}
    loop = asyncio.get_running_loop()

    async def job(name: str) -> str:
        started[name] = loop.time()
        await asyncio.sleep(0.001)
        return name

    results = await route_geometry._paced_gather(
        [
            ("https://one.example/first", lambda: job("one-first")),
            ("https://two.example/first", lambda: job("two-first")),
            ("https://one.example/second", lambda: job("one-second")),
        ]
    )

    assert results == ["one-first", "two-first", "one-second"]
    assert abs(started["one-first"] - started["two-first"]) < 0.01
    assert started["one-second"] - started["one-first"] >= 0.03


def test_cache_serialization_round_trip_preserves_hkemobility_metadata():
    line = _valid_line(route_geometry.ROUTE_SPECS[0], "fresh")
    geometry = RouteGeometry([line], line.stops, 123.0)

    restored = route_geometry._deserialize(route_geometry._serialize(geometry))

    assert restored.fetched_at == 123.0
    assert restored.routes == [line]
    assert restored.stops == line.stops


async def test_invalid_refreshed_line_keeps_its_cached_line_while_other_route_updates(
    monkeypatch, tmp_path
):
    retained_spec, updated_spec = route_geometry.ROUTE_SPECS[:2]
    cached_line = _valid_line(retained_spec, "cached")
    updated_line = _valid_line(updated_spec, "updated")
    invalid = RouteLine(retained_spec.route, retained_spec.operator, retained_spec.bound)
    cached = RouteGeometry([cached_line, _valid_line(updated_spec, "old")], [], 42.0)

    async def load(_client):
        return [invalid, updated_line]

    monkeypatch.setattr(route_geometry, "_load_route_lines", load)
    route_geometry._refresh_retry_after.pop(str(tmp_path), None)
    result = await route_geometry._refresh_route_geometry(object(), str(tmp_path), cached)

    assert result.routes == [cached_line, updated_line]
    assert result.fetched_at == 42.0
    assert route_geometry._refresh_retry_after[str(tmp_path)] > asyncio.get_running_loop().time()


async def test_incomplete_refresh_cooldown_suppresses_stale_cache_retry(monkeypatch, tmp_path):
    cached_line = _valid_line(route_geometry.ROUTE_SPECS[0], "cached")
    cached = RouteGeometry([cached_line], cached_line.stops, 1.0)
    cache_dir = str(tmp_path)
    route_geometry._refresh_retry_after[cache_dir] = (
        asyncio.get_running_loop().time() + route_geometry.GEOMETRY_FAILURE_COOLDOWN_SECONDS
    )
    monkeypatch.setattr(route_geometry, "_load_disk_cache", lambda _cache_dir: cached)

    async def should_not_refresh(_client, _cache_dir, _cached):
        raise AssertionError("cooldown should suppress another direction sweep")

    monkeypatch.setattr(route_geometry, "_refresh_route_geometry", should_not_refresh)
    result = await route_geometry.fetch_route_geometry(object(), cache_dir)

    assert result is cached
    route_geometry._refresh_retry_after.pop(cache_dir, None)


async def test_empty_or_failed_refresh_never_erases_last_good_cache(monkeypatch, tmp_path):
    cached_line = _valid_line(route_geometry.ROUTE_SPECS[0], "cached")
    cached = RouteGeometry([cached_line], cached_line.stops, 42.0)
    route_geometry._save_disk_cache(cached, str(tmp_path))

    async def empty(_client):
        return []

    monkeypatch.setattr(route_geometry, "_load_route_lines", empty)
    empty_result = await route_geometry._refresh_route_geometry(object(), str(tmp_path), cached)
    assert empty_result.routes == [cached_line]
    assert route_geometry._load_disk_cache(str(tmp_path)).routes == [cached_line]

    async def failed(_client):
        raise RuntimeError("unavailable")

    monkeypatch.setattr(route_geometry, "_load_route_lines", failed)
    failed_result = await route_geometry._refresh_route_geometry(object(), str(tmp_path), cached)
    assert failed_result is cached
    assert route_geometry._load_disk_cache(str(tmp_path)).routes == [cached_line]


async def test_public_stops_include_bus_operators_and_exclude_gmb(monkeypatch, tmp_path):
    kmb = _valid_line(route_geometry.ROUTE_SPECS[0], "kmb")
    ctb = _valid_line(route_geometry.ROUTE_SPECS[7], "ctb")
    gmb = _valid_line(route_geometry.ROUTE_SPECS[9], "gmb")

    async def load(_client):
        return [kmb, ctb, gmb]

    monkeypatch.setattr(route_geometry, "_load_route_lines", load)
    result = await route_geometry._refresh_route_geometry(object(), str(tmp_path), None)

    assert {(stop.name, stop.stop_id) for stop in result.stops} == {
        (stop.name, stop.stop_id) for line in (kmb, ctb) for stop in line.stops
    }
    assert all("gmb" not in stop.name for stop in result.stops)


async def test_cold_geometry_refresh_can_detach_from_presenter(monkeypatch, tmp_path):
    started = asyncio.Event()
    release = asyncio.Event()
    cache_dir = str(tmp_path)
    refresh_calls = 0

    async def refresh(_client, _cache_dir, _cached):
        nonlocal refresh_calls
        refresh_calls += 1
        started.set()
        await release.wait()
        return RouteGeometry()

    monkeypatch.setattr(route_geometry, "_load_disk_cache", lambda _cache_dir: None)
    monkeypatch.setattr(route_geometry, "_refresh_route_geometry", refresh)
    route_geometry._refresh_retry_after.pop(cache_dir, None)

    immediate = await route_geometry.fetch_route_geometry(
        object(), cache_dir, wait_for_refresh=False
    )
    await asyncio.wait_for(started.wait(), timeout=1)
    assert immediate.routes == []
    assert cache_dir in route_geometry._refresh_tasks  # noqa: SLF001
    waiter = asyncio.create_task(route_geometry.fetch_route_geometry(object(), cache_dir))
    await asyncio.sleep(0)
    assert refresh_calls == 1
    assert not waiter.done()

    release.set()
    assert (await waiter).routes == []
    await asyncio.sleep(0)
    assert cache_dir not in route_geometry._refresh_tasks  # noqa: SLF001


async def test_cancelled_background_refresh_is_normal_shutdown(monkeypatch):
    task = asyncio.create_task(asyncio.sleep(60))
    route_geometry._refresh_tasks["cancel-test"] = task
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    route_geometry._finish_refresh(task, "cancel-test")

    assert "cancel-test" not in route_geometry._refresh_tasks
    assert "cancel-test" not in route_geometry._refresh_retry_after


@pytest.mark.asyncio
async def test_shutdown_background_refreshes_drains_registry():
    cache_key = "shutdown-test"
    route_geometry._refresh_shutdown = False
    route_geometry._refresh_retry_after[cache_key] = 123.0
    task = asyncio.create_task(asyncio.sleep(60))
    route_geometry._refresh_tasks[cache_key] = task

    await route_geometry.shutdown_background_refreshes()

    assert task.cancelled()
    assert route_geometry._refresh_tasks == {}
    assert route_geometry._refresh_retry_after == {}
