"""Provider collection and adaptation for the dashboard runtime.

This module owns the provider dependency graph and the conversion from raw
provider results to renderer payloads.  It deliberately has no dependency on
the Discord runtime or on :mod:`bot` globals.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import TypeAlias

from dashboard import maps, road_policy
from dashboard.config import Settings
from dashboard.http import HttpClient
from dashboard.models import DashboardPayload, WeatherConditions
from dashboard.providers import route_geometry as route_geometry_provider
from dashboard.providers import tracked_roads as tracked_roads_provider
from dashboard.providers import traffic as traffic_provider
from dashboard.providers import transit
from dashboard.providers import weather as weather_provider
from dashboard.render import build_payload

log = logging.getLogger(__name__)
ProviderResults: TypeAlias = dict[str, object]
ResultCallback: TypeAlias = Callable[[str, object], None]
RoadPath: TypeAlias = list[tuple[float, float]]
MapPaths: TypeAlias = tuple[list[RoadPath], list[RoadPath]]
TRACKED_ROADS_WAIT_SECONDS = 5.0


def map_road_paths_from_results(traffic_result: object, roads: object) -> MapPaths:
    """Derive affected and important map paths from published results."""
    important_paths = road_policy.important_road_paths(roads)
    if not (isinstance(traffic_result, tuple) and len(traffic_result) >= 3):
        return [], important_paths
    segments_near = getattr(roads, "segments_near", None)
    if segments_near is None:
        return [], important_paths
    paths = []
    seen_paths: set[tuple[tuple[float, float], ...]] = set()
    for incident in traffic_result[1] or []:
        latitude = getattr(incident, "latitude", None)
        longitude = getattr(incident, "longitude", None)
        keys = traffic_provider.resolve_incident_road_keys(incident, roads)
        valid = (
            isinstance(latitude, (int, float))
            and isinstance(longitude, (int, float))
            and 22.0 <= latitude <= 23.0
            and 113.5 <= longitude <= 114.7
        )
        if not valid:
            if latitude is not None or longitude is not None:
                continue
            near = str(getattr(incident, "near_landmark", "") or "").strip()
            between = str(getattr(incident, "between_landmark", "") or "").strip()
            if near or between:
                keys = traffic_provider.resolve_incident_road_keys(
                    incident, roads, prefer_refinement=True
                )
                if not keys:
                    continue
            latitude = longitude = None
        if not keys:
            continue
        for path in segments_near(keys, latitude, longitude) or ():
            normalized = tuple((float(lat), float(lon)) for lat, lon in path)
            if len(normalized) >= 2 and normalized not in seen_paths:
                seen_paths.add(normalized)
                paths.append(list(normalized))
    return paths, important_paths


async def fetch_traffic_map_from_results(
    client: HttpClient, settings: Settings, results: ProviderResults, tracker: object
) -> object:
    """Render a map using retained provider inputs without network joins."""
    transit_result = results.get("transit")
    groups = (
        transit_result[0] if isinstance(transit_result, tuple) and len(transit_result) == 3 else []
    )
    roads = results.get("tracked_roads")
    if roads is None or isinstance(roads, Exception):
        roads = tracked_roads_provider.fallback_roads()
    affected, important = map_road_paths_from_results(results.get("traffic"), roads)
    return await maps.fetch_traffic_map(
        client,
        groups=groups,
        cache_dir=settings.cache_dir,
        affected_road_paths=affected,
        tracker=tracker,
        important_road_paths=important,
    )


async def collect_all(
    client: HttpClient,
    settings: Settings,
    on_result: ResultCallback | None = None,
    tracker=None,
    include_traffic_map: bool = True,
    tracked_roads_wait_seconds: float | None = None,
) -> ProviderResults:
    """Collect independent providers and publish each result as it settles."""
    tasks: dict[str, asyncio.Task] = {}
    wait_seconds = (
        TRACKED_ROADS_WAIT_SECONDS
        if tracked_roads_wait_seconds is None
        else tracked_roads_wait_seconds
    )

    async def tracked() -> object:
        return await tracked_roads_provider.fetch_tracked_roads(
            client, cache_dir=settings.cache_dir, wait_for_refresh=False
        )

    async def transit_task() -> object:
        return await transit.fetch_transit_etas(client)

    async def weather() -> object:
        return await weather_provider.fetch_weather_conditions(client)

    async def traffic() -> object:
        try:
            roads = await asyncio.wait_for(
                asyncio.shield(tasks["tracked_roads"]), timeout=wait_seconds
            )
        except Exception:
            roads = tracked_roads_provider.fallback_roads()
        return await traffic_provider.fetch_traffic_data(client, roads)

    async def traffic_map() -> object:
        try:
            result = await tasks["transit"]
            groups = result[0] if isinstance(result, tuple) and len(result) == 3 else []
        except Exception as exc:
            log.warning("traffic map: transit groups unavailable: %s", exc)
            groups = []
        try:
            tr = await tasks["traffic"]
        except Exception:
            tr = None
        try:
            roads = await asyncio.wait_for(
                asyncio.shield(tasks["tracked_roads"]), timeout=wait_seconds
            )
        except Exception:
            roads = tracked_roads_provider.fallback_roads()
        affected, important = map_road_paths_from_results(tr, roads)
        return await maps.fetch_traffic_map(
            client,
            groups=groups,
            cache_dir=settings.cache_dir,
            affected_road_paths=affected,
            tracker=tracker,
            important_road_paths=important,
        )

    for name, coro in (
        ("tracked_roads", tracked()),
        ("transit", transit_task()),
        ("weather", weather()),
        ("traffic", traffic()),
    ):
        tasks[name] = asyncio.create_task(coro)
    if include_traffic_map:
        tasks["traffic_map"] = asyncio.create_task(traffic_map())
    results: ProviderResults = {}
    names = {task: name for name, task in tasks.items()}
    pending = set(tasks.values())
    try:
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                name = names[task]
                try:
                    value = task.result()
                except Exception as exc:
                    log.warning("provider %s failed: %s", name, exc)
                    value = exc
                results[name] = value
                if on_result is not None:
                    on_result(name, value)
        return results
    finally:
        remaining = [task for task in tasks.values() if not task.done()]
        for task in remaining:
            task.cancel()
        if remaining:
            await asyncio.gather(*remaining, return_exceptions=True)


def to_payload(results: ProviderResults) -> DashboardPayload:
    """Adapt isolated provider results to the renderer contract."""
    errors: list[str] = []
    tr = results.get("transit")
    if isinstance(tr, Exception):
        errors.append("transit ETA unavailable")
        groups = []
        transit_time = None
    elif isinstance(tr, tuple) and len(tr) == 3:
        groups, transit_time, failed = tr
        errors.extend(f"{op} ETA unavailable" for op in failed)
    else:
        groups = []
        transit_time = None
    wr = results.get("weather")
    weather = None
    if isinstance(wr, Exception):
        errors.append("HKO weather unavailable")
    elif isinstance(wr, tuple) and len(wr) == 3:
        snap, warnings, warning_time = wr
        weather = WeatherConditions(warnings=warnings, snapshot=snap, warning_time=warning_time)
    elif isinstance(wr, WeatherConditions):
        weather = wr
    traffic_result = results.get("traffic")
    source_times = {}
    if isinstance(traffic_result, Exception):
        errors.append("TD traffic unavailable")
        statuses, incidents, roadworks, capture, stale = [], [], [], None, []
    elif isinstance(traffic_result, tuple) and len(traffic_result) >= 6:
        statuses, incidents, roadworks, capture, stale, source_times = traffic_result[:6]
    elif isinstance(traffic_result, tuple) and len(traffic_result) >= 5:
        statuses, incidents, roadworks, capture, stale = traffic_result
    elif isinstance(traffic_result, tuple) and len(traffic_result) == 4:
        statuses, incidents, roadworks, capture = traffic_result
        stale = []
    else:
        statuses, incidents, roadworks, capture, stale = [], [], [], None, []
    present = "traffic_map" in results
    mr = results.get("traffic_map")
    map_webp = mr[0] if isinstance(mr, tuple) and len(mr) >= 2 else None
    if isinstance(mr, Exception):
        map_webp = None
    if map_webp is None and present:
        errors.append("traffic map unavailable")
    roads = results.get("tracked_roads")
    if isinstance(roads, Exception):
        roads = None
    return build_payload(
        weather=weather,
        groups=groups,
        statuses=statuses,
        incidents=incidents,
        capture_time=capture,
        traffic_map_webp=map_webp,
        traffic_map_initializing=not present,
        transit_source_time=transit_time,
        map_source_time=None,
        roadworks=roadworks,
        traffic_stale_sources=stale,
        traffic_source_times=source_times,
        traffic_source_time=capture,
        errors=errors,
        roads=roads,
    )


async def shutdown_background_resources() -> None:
    """Drain provider/browser resources; one failure never strands siblings."""
    components = (
        maps.shutdown_gmaps_browser,
        route_geometry_provider.shutdown_background_refreshes,
        tracked_roads_provider.shutdown_background_refreshes,
        transit.shutdown_background_refreshes,
    )
    for close in components:
        try:
            await close()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("background resource shutdown failed: %s", type(exc).__name__)
