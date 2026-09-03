"""Official HKeMobility route geometry for estimated map markers.

The provider combines official operator stop sequences/coordinates with the
route lines shown by HKeMobility. Requests to each upstream host are paced
across the dashboard update window. Geometry is cached on disk for 12 hours;
failed directions retain their last-good line without blocking other routes.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

from dashboard.http import CachedFetch, HttpClient

log = logging.getLogger(__name__)

ROUTE_TTL_SECONDS = 12 * 3600.0
GEOMETRY_CACHE_VERSION = 11
GEOMETRY_CACHE_NAME = "route-geometry.json"
GEOMETRY_REFRESH_TIMEOUT_SECONDS = 45.0
GEOMETRY_FAILURE_COOLDOWN_SECONDS = 5 * 60.0
SPATIAL_REQUEST_TIMEOUT_MS = 10_000
PACE_WINDOW_SECONDS = 12.0
PACE_MAX_INTERVAL_SECONDS = 0.35
MAX_STOP_DISTANCE_METRES = 110.0

KMB_ROUTE_STOP_URL = "https://data.etabus.gov.hk/v1/transport/kmb/route-stop/{route}/{bound}/1"
KMB_STOP_URL = "https://data.etabus.gov.hk/v1/transport/kmb/stop/{stop_id}"
CTB_ROUTE_STOP_URL = "https://rt.data.gov.hk/v2/transport/citybus/route-stop/CTB/{route}/{bound}"
CTB_STOP_URL = "https://rt.data.gov.hk/v2/transport/citybus/stop/{stop_id}"
GMB_ROUTE_STOP_URL = "https://data.etagmb.gov.hk/route-stop/{route_id}/{sequence}"
GMB_STOP_URL = "https://data.etagmb.gov.hk/stop/{stop_id}"
HKEMOBILITY_SPATIAL_URL = (
    "https://www.hkemobility.gov.hk/api/drss/public-transport-routes/"
    "{route_id}/sequence/{sequence}/spatial"
)
HKEMOBILITY_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36 Edg/151.0.0.0"
)
HKEMOBILITY_BUS_REFERER = "https://www.hkemobility.gov.hk/en/public-transport/bus"
HKEMOBILITY_GMB_REFERER = "https://www.hkemobility.gov.hk/en/public-transport/gmb"


def _append_system_ca_node_option() -> str | None:
    """Add Playwright's system-CA flag on Windows without losing options."""
    if os.name != "nt":
        return None
    previous = os.environ.get("NODE_OPTIONS")
    options = previous or ""
    if "--use-system-ca" not in options.split():
        os.environ["NODE_OPTIONS"] = f"{options} --use-system-ca".strip()
    return previous


def _restore_node_options(previous: str | None) -> None:
    if os.name != "nt":
        return
    if previous is None:
        os.environ.pop("NODE_OPTIONS", None)
    else:
        os.environ["NODE_OPTIONS"] = previous


@dataclass(frozen=True)
class RouteSpec:
    operator: str
    route: str
    bound: str
    route_id: int
    sequence: int


ROUTE_SPECS: tuple[RouteSpec, ...] = (
    RouteSpec("KMB", "91", "outbound", 1395, 1),
    RouteSpec("KMB", "91", "inbound", 1395, 2),
    RouteSpec("KMB", "91M", "outbound", 1398, 1),
    RouteSpec("KMB", "91M", "inbound", 1398, 2),
    RouteSpec("KMB", "91P", "outbound", 8093, 1),
    RouteSpec("KMB", "91P", "inbound", 8426, 1),
    RouteSpec("KMB", "291P", "outbound", 8710, 1),
    RouteSpec("CTB", "792M", "outbound", 1616, 1),
    RouteSpec("CTB", "792M", "inbound", 1616, 2),
    RouteSpec("GMB", "11", "seq-1", 2004791, 1),
    RouteSpec("GMB", "11", "seq-2", 2004791, 2),
    RouteSpec("GMB", "11B", "seq-1", 2004828, 1),
    RouteSpec("GMB", "11M", "seq-2", 2004825, 2),
    RouteSpec("GMB", "11S", "seq-1", 2004826, 1),
    RouteSpec("GMB", "11S", "seq-2", 2004826, 2),
    RouteSpec("GMB", "12", "seq-1", 2004764, 1),
    RouteSpec("GMB", "12", "seq-2", 2004764, 2),
    RouteSpec("GMB", "104", "seq-1", 2007200, 1),
)


@dataclass(frozen=True)
class Stop:
    stop_id: str
    name: str
    lat: float
    lon: float


@dataclass
class RouteLine:
    route: str
    operator: str
    bound: str
    stops: list[Stop] = field(default_factory=list)
    path: list[tuple[float, float]] = field(default_factory=list)
    stop_offsets: list[float] = field(default_factory=list)
    destination: str = ""
    path_source: str = ""


@dataclass
class RouteGeometry:
    routes: list[RouteLine] = field(default_factory=list)
    stops: list[Stop] = field(default_factory=list)
    fetched_at: float = 0.0


@dataclass(frozen=True)
class ProbeStop:
    """An official stop chosen for downstream ETA polling."""

    operator: str
    route: str
    bound: str
    stop_id: str
    route_id: int | None
    sequence: int | None
    index: int  # position in the official stop sequence


def _spec_for_line(line: RouteLine) -> RouteSpec | None:
    for spec in ROUTE_SPECS:
        if (spec.operator, spec.route, spec.bound) == (line.operator, line.route, line.bound):
            return spec
    return None


def select_probe_stops(
    lines: Iterable[RouteLine],
    mandatory_stop_ids: Iterable[str] = (),
    *,
    max_anchors: int | None = None,
) -> list[ProbeStop]:
    """Select deterministic official occurrences for each route.

    Production's default selects every stop so presence/absence brackets are
    observable. An explicit ``max_anchors`` retains the legacy sparse mode,
    protecting both termini and mandatory occurrences before filling evenly
    spaced interior positions. The transit layer deduplicates fetch groups.
    """
    required = {str(stop_id) for stop_id in mandatory_stop_ids}
    probes: list[ProbeStop] = []
    for line in lines:
        stops = line.stops
        if not stops:
            continue
        spec = _spec_for_line(line)
        # Production needs absence evidence at every official occurrence.
        # Keep an explicit limit for sparse/backwards-compatible callers.
        limit = len(stops) if max_anchors is None else max(1, int(max_anchors))
        chosen: list[int] = []
        def add(index: int, stops=stops, chosen=chosen) -> None:
            if 0 <= index < len(stops) and index not in chosen:
                chosen.append(index)
        add(0)
        add(len(stops) - 1)
        for index, stop in enumerate(stops):
            if stop.stop_id in required:
                add(index)
        available = [i for i in range(1, len(stops) - 1) if i not in chosen]
        remaining = max(0, limit - len(chosen))
        for rank in range(remaining):
            target = (rank + 1) * (len(stops) - 1) / (remaining + 1)
            candidates = sorted(available, key=lambda value: (abs(value - target), value))
            if not candidates:
                break
            add(candidates[0])
            available.remove(candidates[0])
        for index in sorted(chosen):
            probes.append(
                ProbeStop(
                    operator=line.operator,
                    route=line.route,
                    bound=line.bound,
                    stop_id=stops[index].stop_id,
                    route_id=spec.route_id if spec else None,
                    sequence=spec.sequence if spec else None,
                    index=index,
                )
            )
    return probes


_refresh_tasks: dict[str, asyncio.Task[RouteGeometry]] = {}
_refresh_retry_after: dict[str, float] = {}
_refresh_shutdown = False


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat_scale = 111_320.0
    lon_scale = lat_scale * math.cos(math.radians((a[0] + b[0]) / 2))
    return math.hypot((b[0] - a[0]) * lat_scale, (b[1] - a[1]) * lon_scale)


def _append_point(path: list[tuple[float, float]], point: tuple[float, float]) -> None:
    if not path or _distance(path[-1], point) > 0.2:
        path.append(point)


def _path_length(path: list[tuple[float, float]]) -> float:
    return sum(_distance(first, second) for first, second in zip(path, path[1:], strict=False))


def _point_at_offset(path: list[tuple[float, float]], offset: float) -> tuple[float, float] | None:
    if len(path) < 2 or offset < 0:
        return None
    travelled = 0.0
    for first, second in zip(path, path[1:], strict=False):
        length = _distance(first, second)
        if travelled + length >= offset:
            fraction = 0.0 if length == 0 else (offset - travelled) / length
            return (
                first[0] + (second[0] - first[0]) * fraction,
                first[1] + (second[1] - first[1]) * fraction,
            )
        travelled += length
    return path[-1] if math.isclose(offset, travelled, abs_tol=1.0) else None


def _segment_projection(
    first: tuple[float, float], second: tuple[float, float], point: tuple[float, float]
) -> tuple[float, float]:
    lat_scale = 111_320.0
    lon_scale = lat_scale * math.cos(math.radians((first[0] + second[0]) / 2))
    dx, dy = (second[1] - first[1]) * lon_scale, (second[0] - first[0]) * lat_scale
    px, py = (point[1] - first[1]) * lon_scale, (point[0] - first[0]) * lat_scale
    length_sq = dx * dx + dy * dy
    fraction = 0.0 if length_sq == 0 else min(1.0, max(0.0, (px * dx + py * dy) / length_sq))
    projected = (
        first[0] + (second[0] - first[0]) * fraction,
        first[1] + (second[1] - first[1]) * fraction,
    )
    return _distance(first, projected), _distance(point, projected)


def _monotonic_stop_offsets(
    path: list[tuple[float, float]], stops: list[Stop]
) -> list[float] | None:
    """Map ordered stops to later occurrences on an official route line."""
    offsets = _monotonic_stop_prefix_offsets(path, stops)
    return offsets if len(offsets) == len(stops) else None


def _monotonic_stop_prefix_offsets(
    path: list[tuple[float, float]], stops: list[Stop]
) -> list[float]:
    """Return offsets for the longest leading stop sequence the line covers."""
    offsets: list[float] = []
    minimum = -0.1
    for stop in stops:
        travelled = 0.0
        candidates: list[tuple[float, float]] = []
        for first, second in zip(path, path[1:], strict=False):
            along, distance = _segment_projection(first, second, (stop.lat, stop.lon))
            offset = travelled + along
            if offset > minimum and distance <= MAX_STOP_DISTANCE_METRES:
                candidates.append((distance, offset))
            travelled += _distance(first, second)
        if not candidates:
            break
        _distance_to_line, selected = min(candidates)
        offsets.append(selected)
        minimum = selected + 0.1
    return offsets


def _minimum_stop_distance(path: list[tuple[float, float]], stop: Stop) -> float:
    """Shortest distance from one official stop to any segment of the line."""
    return min(
        (
            _segment_projection(first, second, (stop.lat, stop.lon))[1]
            for first, second in zip(path, path[1:], strict=False)
        ),
        default=float("inf"),
    )


def _covered_stop_prefix(
    path: list[tuple[float, float]], stops: list[Stop]
) -> tuple[list[Stop], list[float]] | None:
    """Accept a verified prefix only when every omitted stop lies off the line.

    This supports an announced trailing route extension whose stops have entered
    the operator sequence before HKeMobility extends its spatial line. A middle
    mismatch remains invalid because a later stop still matches the current line.
    """
    offsets = _monotonic_stop_prefix_offsets(path, stops)
    if len(offsets) < 2 or len(offsets) == len(stops):
        return None
    omitted = stops[len(offsets) :]
    if any(_minimum_stop_distance(path, stop) <= MAX_STOP_DISTANCE_METRES for stop in omitted):
        return None
    return stops[: len(offsets)], offsets


def _valid_line(line: RouteLine) -> bool:
    if (
        line.path_source != "hkemobility"
        or len(line.stops) < 2
        or len(line.path) < 2
        or len(line.stop_offsets) != len(line.stops)
        or not line.destination
    ):
        return False
    if any(
        second <= first
        for first, second in zip(line.stop_offsets, line.stop_offsets[1:], strict=False)
    ):
        return False
    for stop, offset in zip(line.stops, line.stop_offsets, strict=True):
        point = _point_at_offset(line.path, offset)
        if point is None or _distance(point, (stop.lat, stop.lon)) > MAX_STOP_DISTANCE_METRES:
            return False
    return all(-90 <= lat <= 90 and -180 <= lon <= 180 for lat, lon in line.path)


async def _paced_gather(
    jobs: list[tuple[str, Callable[[], Awaitable[Any]]]],
) -> list[Any]:
    """Launch requests to each origin gradually; different origins overlap."""
    positions: dict[str, int] = defaultdict(int)
    totals: dict[str, int] = defaultdict(int)
    for url, _factory in jobs:
        totals[urlsplit(url).netloc] += 1

    tasks: list[Awaitable[Any]] = []
    for url, factory in jobs:
        origin = urlsplit(url).netloc
        position = positions[origin]
        positions[origin] += 1
        interval = min(
            PACE_MAX_INTERVAL_SECONDS,
            PACE_WINDOW_SECONDS / max(1, totals[origin] - 1),
        )

        async def delayed(
            delay: float = position * interval,
            request: Callable[[], Awaitable[Any]] = factory,
        ) -> Any:
            if delay:
                await asyncio.sleep(delay)
            return await request()

        tasks.append(delayed())
    return await asyncio.gather(*tasks, return_exceptions=True)


def _route_stop_url(spec: RouteSpec) -> str:
    if spec.operator == "KMB":
        return KMB_ROUTE_STOP_URL.format(route=spec.route, bound=spec.bound)
    if spec.operator == "CTB":
        return CTB_ROUTE_STOP_URL.format(route=spec.route, bound=spec.bound)
    return GMB_ROUTE_STOP_URL.format(route_id=spec.route_id, sequence=spec.sequence)


def _stop_url(operator: str, stop_id: str) -> str:
    template = {"KMB": KMB_STOP_URL, "CTB": CTB_STOP_URL, "GMB": GMB_STOP_URL}[operator]
    return template.format(stop_id=stop_id)


async def _fetch_route_stop_ids(client: HttpClient, spec: RouteSpec) -> list[str] | None:
    url = _route_stop_url(spec)
    try:
        _, raw, _ = await client.fetch_json_cached(
            CachedFetch(
                url,
                ROUTE_TTL_SECONDS,
                cache_key=f"route-stops-{spec.operator}-{spec.route}-{spec.bound}",
            )
        )
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "route-stop list failed for %s/%s/%s: %s",
            spec.operator,
            spec.route,
            spec.bound,
            type(exc).__name__,
        )
        return None
    data = raw.get("data") if isinstance(raw, dict) else None
    values = data.get("route_stops") if spec.operator == "GMB" and isinstance(data, dict) else data
    if not isinstance(values, list):
        return None
    key = "stop_id" if spec.operator == "GMB" else "stop"
    return [str(item[key]) for item in values if isinstance(item, dict) and item.get(key)]


async def _fetch_stop(client: HttpClient, operator: str, stop_id: str) -> Stop | None:
    url = _stop_url(operator, stop_id)
    try:
        _, raw, _ = await client.fetch_json_cached(
            CachedFetch(url, ROUTE_TTL_SECONDS, cache_key=f"stop-{operator}-{stop_id}")
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("stop %s/%s failed: %s", operator, stop_id, type(exc).__name__)
        return None
    data = raw.get("data") if isinstance(raw, dict) else None
    if isinstance(data, list):
        data = data[0] if data else None
    if not isinstance(data, dict):
        return None
    coordinates = data.get("coordinates") if isinstance(data.get("coordinates"), dict) else {}
    wgs84 = coordinates.get("wgs84") if isinstance(coordinates.get("wgs84"), dict) else {}
    try:
        return Stop(
            stop_id,
            str(data.get("name_en") or data.get("name_tc") or stop_id),
            float(data.get("lat") or wgs84.get("latitude")),
            float(data.get("long") or wgs84.get("longitude")),
        )
    except (TypeError, ValueError):
        return None


async def _fetch_spatial(request_context: Any, spec: RouteSpec) -> dict[str, Any] | None:
    url = HKEMOBILITY_SPATIAL_URL.format(route_id=spec.route_id, sequence=spec.sequence)
    referer = (
        HKEMOBILITY_GMB_REFERER
        if spec.operator == "GMB"
        else HKEMOBILITY_BUS_REFERER
    )
    try:
        response = await request_context.get(
            url,
            headers={"Referer": referer, "User-Agent": HKEMOBILITY_USER_AGENT},
            timeout=SPATIAL_REQUEST_TIMEOUT_MS,
        )
        if not response.ok:
            raise RuntimeError(f"HTTP {response.status}")
        raw = await response.json()
        return raw if isinstance(raw, dict) else None
    except Exception as exc:  # noqa: BLE001
        status = getattr(response, "status", None) if "response" in locals() else None
        detail = str(exc).replace("\n", " ")[:300]
        log.warning(
            "HKeMobility line failed for %s/%s/%s url=%s status=%s detail=%s",
            spec.operator,
            spec.route,
            spec.bound,
            url,
            status,
            detail,
        )
        return None


async def _fetch_spatial_batch() -> list[dict[str, Any] | None]:
    """Fetch all spatial directions through one browser-shaped API context."""
    from playwright.async_api import async_playwright

    previous_options = _append_system_ca_node_option()
    try:
        async with async_playwright() as playwright:
            # The Node driver has inherited NODE_OPTIONS by this point; restore
            # the host process environment before making requests.
            _restore_node_options(previous_options)
            request_context = await playwright.request.new_context(user_agent=HKEMOBILITY_USER_AGENT)
            try:
                results: list[dict[str, Any] | None] = []
                interval = min(
                    PACE_MAX_INTERVAL_SECONDS,
                    PACE_WINDOW_SECONDS / max(1, len(ROUTE_SPECS) - 1),
                )
                for index, spec in enumerate(ROUTE_SPECS):
                    if index:
                        await asyncio.sleep(interval)
                    results.append(await _fetch_spatial(request_context, spec))
                return results
            finally:
                await request_context.dispose()
    finally:
        _restore_node_options(previous_options)


def _build_line(spec: RouteSpec, stops: list[Stop], raw: dict[str, Any] | None) -> RouteLine:
    line = RouteLine(spec.route, spec.operator, spec.bound, stops=stops)
    if not raw:
        return line
    shape = raw.get("sh")
    coordinates = shape.get("coordinates") if isinstance(shape, dict) else None
    path: list[tuple[float, float]] = []
    if isinstance(coordinates, list):
        for segment in coordinates:
            if not isinstance(segment, list):
                continue
            for point in segment:
                if isinstance(point, list) and len(point) >= 2:
                    try:
                        # HKeMobility uses latitude,longitude despite the
                        # GeoJSON-like shape wrapper.
                        _append_point(path, (float(point[0]), float(point[1])))
                    except (TypeError, ValueError):
                        continue
    offsets = _monotonic_stop_offsets(path, stops)
    if offsets is None and (covered := _covered_stop_prefix(path, stops)) is not None:
        covered_stops, offsets = covered
        log.info(
            "HKeMobility line %s/%s/%s currently covers %d/%d stops; "
            "omitting trailing stops outside the spatial line",
            spec.operator,
            spec.route,
            spec.bound,
            len(covered_stops),
            len(stops),
        )
        line.stops = covered_stops
    line.path = path
    line.stop_offsets = offsets or []
    line.destination = str(raw.get("e") or "")
    line.path_source = "hkemobility" if offsets else ""
    return line


async def _load_route_lines(client: HttpClient) -> list[RouteLine]:
    route_jobs = [
        (
            _route_stop_url(spec),
            lambda spec=spec: _fetch_route_stop_ids(client, spec),
        )
        for spec in ROUTE_SPECS
    ]
    stop_id_results = await _paced_gather(route_jobs)
    stop_ids_by_spec = {
        spec: value if isinstance(value, list) else []
        for spec, value in zip(ROUTE_SPECS, stop_id_results, strict=True)
    }

    unique_stops = sorted(
        {
            (spec.operator, stop_id)
            for spec, stop_ids in stop_ids_by_spec.items()
            for stop_id in stop_ids
        }
    )
    stop_jobs = [
        (
            _stop_url(operator, stop_id),
            lambda operator=operator, stop_id=stop_id: _fetch_stop(client, operator, stop_id),
        )
        for operator, stop_id in unique_stops
    ]
    async def load_spatial_safely() -> list[Any]:
        try:
            return await _fetch_spatial_batch()
        except Exception as exc:  # noqa: BLE001
            log.warning("HKeMobility spatial client failed: %s", type(exc).__name__)
            return [None] * len(ROUTE_SPECS)

    stop_results, spatial_results = await asyncio.gather(
        _paced_gather(stop_jobs),
        load_spatial_safely(),
    )
    stop_lookup = {
        key: value
        for key, value in zip(unique_stops, stop_results, strict=True)
        if isinstance(value, Stop)
    }

    lines: list[RouteLine] = []
    for spec, raw in zip(ROUTE_SPECS, spatial_results, strict=True):
        expected_stop_ids = stop_ids_by_spec[spec]
        stops = [
            stop_lookup[(spec.operator, stop_id)]
            for stop_id in expected_stop_ids
            if (spec.operator, stop_id) in stop_lookup
        ]
        # A partial stop sequence is not official route geometry.  Reject only
        # this direction so its last-good cache can survive independently.
        complete_stops = bool(expected_stop_ids) and len(stops) == len(expected_stop_ids)
        line = _build_line(
            spec,
            stops,
            raw if complete_stops and isinstance(raw, dict) else None,
        )
        if not _valid_line(line):
            log.warning(
                "official geometry unavailable for %s/%s/%s", spec.operator, spec.route, spec.bound
            )
        lines.append(line)
    return lines


def _serialize(geometry: RouteGeometry) -> dict[str, Any]:
    return {
        "version": GEOMETRY_CACHE_VERSION,
        "fingerprint": _cache_fingerprint(),
        "fetched_at": geometry.fetched_at,
        "routes": [
            {
                "route": line.route,
                "operator": line.operator,
                "bound": line.bound,
                "stops": [[stop.stop_id, stop.name, stop.lat, stop.lon] for stop in line.stops],
                "path": line.path,
                "stop_offsets": line.stop_offsets,
                "destination": line.destination,
                "path_source": line.path_source,
            }
            for line in geometry.routes
        ],
        "stops": [[stop.stop_id, stop.name, stop.lat, stop.lon] for stop in geometry.stops],
    }


def _deserialize(data: dict[str, Any]) -> RouteGeometry:
    geometry = RouteGeometry(fetched_at=float(data.get("fetched_at") or 0))
    for raw in data.get("routes") or []:
        line = RouteLine(str(raw["route"]), str(raw["operator"]), str(raw["bound"]))
        line.stops = [
            Stop(str(s[0]), str(s[1]), float(s[2]), float(s[3])) for s in raw.get("stops") or []
        ]
        line.path = [(float(p[0]), float(p[1])) for p in raw.get("path") or []]
        line.stop_offsets = [float(value) for value in raw.get("stop_offsets") or []]
        line.destination = str(raw.get("destination") or "")
        line.path_source = str(raw.get("path_source") or "")
        geometry.routes.append(line)
    geometry.stops = [
        Stop(str(s[0]), str(s[1]), float(s[2]), float(s[3])) for s in data.get("stops") or []
    ]
    return geometry


def _cache_fingerprint() -> str:
    payload = [
        (spec.operator, spec.route, spec.bound, spec.route_id, spec.sequence)
        for spec in ROUTE_SPECS
    ]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()[:20]


def _cache_file(cache_dir: str) -> str:
    return os.path.join(cache_dir, "maps", GEOMETRY_CACHE_NAME)


def _load_disk_cache(cache_dir: str = ".cache") -> RouteGeometry | None:
    try:
        with open(_cache_file(cache_dir), encoding="utf-8") as file:
            raw = json.load(file)
        if (
            raw.get("version") != GEOMETRY_CACHE_VERSION
            or raw.get("fingerprint") != _cache_fingerprint()
        ):
            return None
        geometry = _deserialize(raw)
        geometry.routes = [line for line in geometry.routes if _valid_line(line)]
        return geometry if geometry.routes else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _save_disk_cache(geometry: RouteGeometry, cache_dir: str = ".cache") -> None:
    valid = [line for line in geometry.routes if _valid_line(line)]
    if not valid:
        return
    serializable = RouteGeometry(valid, geometry.stops, geometry.fetched_at)
    try:
        path = _cache_file(cache_dir)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        temporary = f"{path}.tmp"
        with open(temporary, "w", encoding="utf-8") as file:
            json.dump(_serialize(serializable), file)
        os.replace(temporary, path)
    except OSError as exc:
        log.warning("route geometry cache write failed: %s", exc)


async def _refresh_route_geometry(
    client: HttpClient, cache_dir: str, cached: RouteGeometry | None
) -> RouteGeometry:
    try:
        refreshed = await asyncio.wait_for(
            _load_route_lines(client), GEOMETRY_REFRESH_TIMEOUT_SECONDS
        )
    except Exception as exc:  # noqa: BLE001
        _refresh_retry_after[cache_dir] = time.monotonic() + GEOMETRY_FAILURE_COOLDOWN_SECONDS
        log.warning("route geometry refresh failed: %s", type(exc).__name__)
        return cached or RouteGeometry()

    cached_lines = {
        (line.operator, line.route, line.bound): line for line in (cached.routes if cached else [])
    }
    refreshed_lines = {(line.operator, line.route, line.bound): line for line in refreshed}
    routes: list[RouteLine] = []
    complete = True
    for spec in ROUTE_SPECS:
        key = (spec.operator, spec.route, spec.bound)
        line = refreshed_lines.get(key)
        if line is not None and _valid_line(line):
            routes.append(line)
            continue
        complete = False
        if key in cached_lines:
            routes.append(cached_lines[key])

    public_stops: dict[tuple[str, str], Stop] = {}
    for line in routes:
        if line.operator == "GMB":
            continue
        for stop in line.stops:
            public_stops.setdefault((line.operator, stop.stop_id), stop)
    geometry = RouteGeometry(
        routes,
        sorted(public_stops.values(), key=lambda stop: stop.name),
        time.time()
        if complete and len(routes) == len(ROUTE_SPECS)
        else (cached.fetched_at if cached else 0),
    )
    if not complete or len(routes) != len(ROUTE_SPECS):
        _refresh_retry_after[cache_dir] = (
            time.monotonic() + GEOMETRY_FAILURE_COOLDOWN_SECONDS
        )
        log.warning(
            "route geometry refresh incomplete: retained %d/%d directions; retrying after cooldown",
            len(routes),
            len(ROUTE_SPECS),
        )
    _save_disk_cache(geometry, cache_dir)
    return geometry


def _finish_refresh(task: asyncio.Task[RouteGeometry], cache_dir: str) -> None:
    _refresh_tasks.pop(cache_dir, None)
    if _refresh_shutdown:
        return
    if task.cancelled():
        # Cancellation is normal during shutdown; it must not schedule a
        # failure cooldown or emit an unhandled done-callback exception.
        return
    try:
        task.result()
    except Exception as exc:  # noqa: BLE001
        _refresh_retry_after[cache_dir] = time.monotonic() + GEOMETRY_FAILURE_COOLDOWN_SECONDS
        log.warning("background route geometry refresh failed: %s", type(exc).__name__)


async def fetch_route_geometry(client: HttpClient, cache_dir: str = ".cache") -> RouteGeometry:
    """Return fresh/last-good geometry and refresh expired cache in background."""
    global _refresh_shutdown
    _refresh_shutdown = False
    cached = _load_disk_cache(cache_dir)
    if cached is not None:
        if time.time() - cached.fetched_at <= ROUTE_TTL_SECONDS:
            return cached
        if cache_dir not in _refresh_tasks and time.monotonic() >= _refresh_retry_after.get(
            cache_dir, 0
        ):
            task = asyncio.create_task(_refresh_route_geometry(client, cache_dir, cached))
            _refresh_tasks[cache_dir] = task
            task.add_done_callback(
                lambda done, cache_dir=cache_dir: _finish_refresh(done, cache_dir)
            )
        return cached
    if time.monotonic() < _refresh_retry_after.get(cache_dir, 0):
        return RouteGeometry()
    return await _refresh_route_geometry(client, cache_dir, None)


async def shutdown_background_refreshes() -> None:
    """Cancel and drain refreshes before the shared HTTP session is closed."""
    global _refresh_shutdown
    _refresh_shutdown = True
    tasks = list(_refresh_tasks.values())
    for task in tasks:
        if not task.done():
            task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    _refresh_tasks.clear()
    _refresh_retry_after.clear()
