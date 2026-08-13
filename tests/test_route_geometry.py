"""Road routing and route-geometry cache contracts."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import aiohttp

from dashboard.providers import route_geometry
from dashboard.providers.route_geometry import RouteGeometry, RouteLine, Stop


class _OsrmClient:
    async def fetch_json_cached(self, spec, headers=None):
        assert headers and headers["User-Agent"].startswith("hkust-data-dashboard/")
        coordinates = spec.url.split("/driving/", 1)[1].split("?", 1)[0].split(";")
        points = [[float(value) for value in item.split(",")] for item in coordinates]
        legs = []
        for first, second in zip(points, points[1:], strict=False):
            # lon/lat GeoJSON with a road bend between every TD waypoint.
            bend = [first[0], (first[1] + second[1]) / 2 + 0.0003]
            legs.append(
                {
                    "steps": [
                        {"geometry": {"coordinates": [first, bend, second]}},
                    ]
                }
            )
        return False, {
            "code": "Ok",
            "waypoints": [{"location": point} for point in points],
            "routes": [{"legs": legs}],
        }, 1.0


async def test_osrm_geometry_uses_every_td_stop_and_monotonic_offsets():
    stops = [
        Stop("A", "A", 22.3178, 114.2640),
        Stop("B", "B", 22.3178, 114.2660),
        Stop("C", "C", 22.3178, 114.2680),
    ]
    routed = await route_geometry._fetch_osrm_chunk(_OsrmClient(), stops, "test")
    assert routed is not None
    path, offsets = routed
    assert len(path) == 5
    assert len(offsets) == len(stops)
    assert offsets[0] == 0
    assert offsets[0] < offsets[1] < offsets[2]
    assert path[1][0] > stops[0].lat  # follows the supplied road bend


class _RemoteSnapClient(_OsrmClient):
    async def fetch_json_cached(self, spec, headers=None):
        stale, raw, fetched_at = await super().fetch_json_cached(spec, headers)
        raw["waypoints"][1]["location"][0] += 0.01
        return stale, raw, fetched_at


async def test_osrm_geometry_rejects_a_td_stop_snapped_to_a_remote_road():
    stops = [
        Stop("A", "A", 22.3178, 114.2640),
        Stop("B", "B", 22.3178, 114.2660),
        Stop("C", "C", 22.3178, 114.2680),
    ]
    assert await route_geometry._fetch_osrm_chunk(_RemoteSnapClient(), stops, "test") is None


class _HttpsFailureClient(_OsrmClient):
    def __init__(self):
        self.urls = []

    async def fetch_json_cached(self, spec, headers=None):
        self.urls.append(spec.url)
        if spec.url.startswith("https://"):
            raise aiohttp.ClientConnectorCertificateError(None, RuntimeError("expired"))
        return await super().fetch_json_cached(spec, headers)


async def test_osrm_https_failure_uses_public_http_geometry_fallback(monkeypatch):
    monkeypatch.delenv(route_geometry.OSRM_BASE_URL_ENV, raising=False)
    stops = [
        Stop("A", "A", 22.3178, 114.2640),
        Stop("B", "B", 22.3178, 114.2660),
    ]
    client = _HttpsFailureClient()

    assert await route_geometry._fetch_osrm_chunk(client, stops, "test") is not None
    assert client.urls[0].startswith("https://router.project-osrm.org/")
    assert client.urls[1].startswith("http://router.project-osrm.org/")


async def test_osrm_does_not_downgrade_a_configured_endpoint(monkeypatch):
    monkeypatch.setenv(route_geometry.OSRM_BASE_URL_ENV, "https://routing.example.test")
    stops = [Stop("A", "A", 22.3178, 114.2640), Stop("B", "B", 22.3178, 114.2660)]
    client = _HttpsFailureClient()

    assert await route_geometry._fetch_osrm_chunk(client, stops, "test") is None
    assert len(client.urls) == 1
    assert client.urls[0].startswith("https://routing.example.test/")


def test_route_geometry_cache_round_trip_preserves_path_and_offsets():
    stops = [Stop("A", "A", 22.31, 114.26), Stop("B", "B", 22.32, 114.27)]
    line = RouteLine(
        "11", "GMB", "seq-1", stops,
        [(22.31, 114.26), (22.315, 114.265), (22.32, 114.27)],
        [0.0, 1500.0],
    )
    restored = route_geometry._deserialize(route_geometry._serialize(RouteGeometry([line], stops, 1)))
    assert restored.routes[0].path == line.path
    assert restored.routes[0].stop_offsets == line.stop_offsets
    assert math.isclose(restored.routes[0].stop_offsets[-1], 1500.0)


def test_kmb_91_path_through_hang_hau_village_is_rejected():
    stops = [
        Stop("A", "Clear Water Bay Road", 22.3200, 114.2690),
        Stop("B", "Ngan Ying Road", 22.3220, 114.2650),
    ]
    path = [(stops[0].lat, stops[0].lon), route_geometry.HANG_HAU_VILLAGE, (stops[1].lat, stops[1].lon)]
    line = RouteLine("91", "KMB", "outbound", stops, path, [0.0, route_geometry._path_length(path)])
    assert not route_geometry._valid_route_line(line)


def _valid_geometry(fetched_at: float) -> RouteGeometry:
    stops = [Stop("A", "A", 22.31, 114.26), Stop("B", "B", 22.32, 114.27)]
    path = [(stops[0].lat, stops[0].lon), (stops[1].lat, stops[1].lon)]
    line = RouteLine(
        "11",
        "GMB",
        "seq-1",
        stops,
        path,
        [0.0, route_geometry._path_length(path)],
    )
    return RouteGeometry([line], stops, fetched_at)


async def test_partial_geometry_refresh_never_publishes_gmb_only_stop_glyphs(tmp_path, monkeypatch):
    """GMB route stops remain usable geometry, never public map-stop inputs."""
    public = [Stop("K-1", "Public bus stop", 22.31, 114.26), Stop("K-2", "Next", 22.32, 114.27)]
    minibus = [Stop("G-1", "Minibus-only", 22.33, 114.28), Stop("G-2", "Next", 22.34, 114.29)]

    def line(route: str, operator: str, stops: list[Stop]) -> RouteLine:
        path = [(stop.lat, stop.lon) for stop in stops]
        return RouteLine(route, operator, "outbound", stops, path, [0.0, route_geometry._path_length(path)])

    kmb_line = line("91", "KMB", public)
    gmb_line = line("11", "GMB", minibus)

    async def partial_refresh(_client):
        return [kmb_line, gmb_line]

    monkeypatch.setattr(route_geometry, "_load_route_lines", partial_refresh)
    refreshed = await route_geometry._refresh_route_geometry(object(), str(tmp_path), None)

    assert refreshed.routes == [kmb_line, gmb_line]
    assert refreshed.stops == [public[1], public[0]]  # provider sorts public inputs by name


async def test_expired_last_good_geometry_is_retained_when_refresh_fails(tmp_path, monkeypatch):
    stale_time = time.time() - route_geometry.TD_ROUTES_TTL_SECONDS - 60
    stale = _valid_geometry(stale_time)
    route_geometry._save_disk_cache(stale, str(tmp_path))

    async def failed_refresh(_client):
        return []

    monkeypatch.setattr(route_geometry, "_load_route_lines", failed_refresh)
    recovered = await route_geometry.fetch_route_geometry(object(), str(tmp_path))

    assert recovered.routes[0].path == stale.routes[0].path
    assert recovered.fetched_at == stale_time
    persisted = route_geometry._load_disk_cache(str(tmp_path))
    assert persisted is not None
    assert persisted.routes[0].path == stale.routes[0].path


async def test_expired_disk_geometry_returns_immediately_and_cools_failed_refresh(tmp_path, monkeypatch):
    stale = _valid_geometry(time.time() - route_geometry.TD_ROUTES_TTL_SECONDS - 1)
    route_geometry._save_disk_cache(stale, str(tmp_path))
    attempts = 0

    async def slow_refresh(_client):
        nonlocal attempts
        attempts += 1
        await __import__("asyncio").sleep(60)
        return []

    monkeypatch.setattr(route_geometry, "_load_route_lines", slow_refresh)
    monkeypatch.setattr(route_geometry, "GEOMETRY_REFRESH_TIMEOUT_SECONDS", 0.01)
    started = time.monotonic()
    recovered = await route_geometry.fetch_route_geometry(object(), str(tmp_path))
    assert time.monotonic() - started < 0.1
    assert recovered.routes[0].path == stale.routes[0].path
    await __import__("asyncio").sleep(0.03)
    assert attempts == 1
    await route_geometry.fetch_route_geometry(object(), str(tmp_path))
    assert attempts == 1


def test_poisoned_empty_cache_is_rejected_and_cannot_overwrite_last_good(tmp_path):
    good = _valid_geometry(time.time())
    route_geometry._save_disk_cache(good, str(tmp_path))
    cache_path = route_geometry._cache_file(str(tmp_path))
    original = Path(cache_path).read_bytes()

    route_geometry._save_disk_cache(RouteGeometry([], [], time.time()), str(tmp_path))
    assert Path(cache_path).read_bytes() == original

    poisoned = route_geometry._serialize(good)
    poisoned["routes"][0]["path"] = []
    Path(cache_path).write_text(json.dumps(poisoned), encoding="utf-8")
    assert route_geometry._load_disk_cache(str(tmp_path)) is None
