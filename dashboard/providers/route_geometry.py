"""Official TD route-stop sequences and OSM-backed road geometry.

This provider loads official KMB/Citybus/GMB stop coordinates and ordered
sequences, then routes those waypoints over OSM roads. The versioned cache is
rooted in the caller's ``cache_dir``. There
are no import-time network or filesystem side effects.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field

import aiohttp

from dashboard.http import USER_AGENT, CachedFetch, HttpClient

log = logging.getLogger(__name__)

TD_ROUTES_TTL_SECONDS = 12 * 3600.0  # 12 hours: route/stop lists change rarely
GEOMETRY_CACHE_VERSION = 6
GEOMETRY_CACHE_NAME = "route-geometry.json"
DEFAULT_OSRM_BASE_URL = "https://router.project-osrm.org"
OSRM_BASE_URL_ENV = "OSRM_BASE_URL"
OSRM_MAX_WAYPOINTS = 16
# Geometry is an enhancement to the live dashboard, not a reason to hold an
# update hostage.  A cold cache gets a short, whole-refresh budget; a last-good
# disk cache is returned immediately and refreshed in the background.
GEOMETRY_REFRESH_TIMEOUT_SECONDS = 12.0
GEOMETRY_FAILURE_COOLDOWN_SECONDS = 5 * 60.0
_refresh_tasks: dict[str, object] = {}
_refresh_retry_after: dict[str, float] = {}

# Reviewed local exclusion: Hang Hau Village (OSM node 7428926981) is not on
# KMB 91's TD stop sequence.  OSRM's generic driving profile can otherwise
# choose its local lanes between the adjacent CWB Road stops.  Reject rather
# than inventing a straight replacement; the next refresh can use a better
# routed path.
HANG_HAU_VILLAGE = (22.3211924, 114.2668510)
HANG_HAU_VILLAGE_EXCLUSION_METRES = 180.0

# The routes we display, keyed by operator. "bounds" are the TD direction
# values (outbound/inbound for KMB/CTB; route_seq 1/2 for GMB).
KMB_ROUTES: dict[str, tuple[str, str]] = {
    "91": ("outbound", "inbound"),
    "91M": ("outbound", "inbound"),
    "91P": ("outbound", "inbound"),
    "291P": ("outbound", "inbound"),
}
CTB_ROUTES: dict[str, tuple[str, str]] = {
    "792M": ("outbound", "inbound"),
}
# GMB route_id -> (route_code, route_seqs)
GMB_ROUTES: dict[int, tuple[str, tuple[int, ...]]] = {
    2004791: ("11", (1, 2)),
    2004828: ("11B", (1,)),
    2004825: ("11M", (2,)),
    2004826: ("11S", (1, 2)),
    2004764: ("12", (1, 2)),
    2007200: ("104", (1,)),
}

KMB_ROUTE_STOP_URL = "https://data.etabus.gov.hk/v1/transport/kmb/route-stop/{route}/{bound}/1"
KMB_STOP_URL = "https://data.etabus.gov.hk/v1/transport/kmb/stop/{stop_id}"
CTB_ROUTE_STOP_URL = "https://rt.data.gov.hk/v2/transport/citybus/route-stop/CTB/{route}/{bound}"
CTB_STOP_URL = "https://rt.data.gov.hk/v2/transport/citybus/stop/{stop_id}"
GMB_ROUTE_STOP_URL = "https://data.etagmb.gov.hk/route-stop/{route_id}/{seq}"
GMB_STOP_URL = "https://data.etagmb.gov.hk/stop/{stop_id}"


@dataclass(frozen=True)
class Stop:
    """One official TD stop on a route: id, name, WGS84 coords."""

    stop_id: str
    name: str
    lat: float
    lon: float


@dataclass
class RouteLine:
    """One direction of one route: the ordered stop sequence."""

    route: str
    operator: str
    bound: str  # "outbound"/"inbound" (KMB/CTB) or "seq-1"/"seq-2" (GMB)
    stops: list[Stop] = field(default_factory=list)
    path: list[tuple[float, float]] = field(default_factory=list)
    stop_offsets: list[float] = field(default_factory=list)


@dataclass
class RouteGeometry:
    """Cached official route sequences and deduplicated public stops."""

    routes: list[RouteLine] = field(default_factory=list)
    stops: list[Stop] = field(default_factory=list)
    fetched_at: float = 0.0


async def _fetch_stop(
    client: HttpClient, url: str, cache_key: str
) -> tuple[str, float, float] | None:
    """Fetch one stop's coords (cached 12h). Returns (name, lat, lon)."""
    spec = CachedFetch(url, TD_ROUTES_TTL_SECONDS, cache_key=cache_key)
    try:
        _, raw, _ = await client.fetch_json_cached(spec)
    except Exception as exc:  # noqa: BLE001
        log.warning("TD stop %s fetch failed: %s", cache_key, exc)
        return None
    data = (raw or {}).get("data") if isinstance(raw, dict) else None
    if not isinstance(data, dict):
        return None
    try:
        name = data.get("name_en") or data.get("name_tc") or cache_key
        lat = float(data.get("lat") or data.get("coordinates", {}).get("wgs84", {}).get("latitude"))
        lon = float(
            data.get("long") or data.get("coordinates", {}).get("wgs84", {}).get("longitude")
        )
        return name, lat, lon
    except (TypeError, ValueError):
        return None


async def _fetch_kmb_route_stops(client: HttpClient, route: str, bound: str) -> list[str] | None:
    spec = CachedFetch(
        KMB_ROUTE_STOP_URL.format(route=route, bound=bound),
        TD_ROUTES_TTL_SECONDS,
        cache_key=f"td-kmb-route-stop-{route}-{bound}",
    )
    try:
        _, raw, _ = await client.fetch_json_cached(spec)
    except Exception as exc:  # noqa: BLE001
        log.warning("KMB %s %s route-stop failed: %s", route, bound, exc)
        return None
    data = (raw or {}).get("data") if isinstance(raw, dict) else None
    if not isinstance(data, list):
        return None
    return [str(s.get("stop")) for s in data if isinstance(s, dict) and s.get("stop")]


async def _fetch_ctb_route_stops(client: HttpClient, route: str, bound: str) -> list[str] | None:
    spec = CachedFetch(
        CTB_ROUTE_STOP_URL.format(route=route, bound=bound),
        TD_ROUTES_TTL_SECONDS,
        cache_key=f"td-ctb-route-stop-{route}-{bound}",
    )
    try:
        _, raw, _ = await client.fetch_json_cached(spec)
    except Exception as exc:  # noqa: BLE001
        log.warning("CTB %s %s route-stop failed: %s", route, bound, exc)
        return None
    data = (raw or {}).get("data") if isinstance(raw, dict) else None
    if not isinstance(data, list):
        return None
    return [str(s.get("stop")) for s in data if isinstance(s, dict) and s.get("stop")]


async def _fetch_gmb_route_stops(client: HttpClient, route_id: int, seq: int) -> list[str] | None:
    spec = CachedFetch(
        GMB_ROUTE_STOP_URL.format(route_id=route_id, seq=seq),
        TD_ROUTES_TTL_SECONDS,
        cache_key=f"td-gmb-route-stop-{route_id}-{seq}",
    )
    try:
        _, raw, _ = await client.fetch_json_cached(spec)
    except Exception as exc:  # noqa: BLE001
        log.warning("GMB %s seq %s route-stop failed: %s", route_id, seq, exc)
        return None
    data = (raw or {}).get("data") if isinstance(raw, dict) else None
    route_stops = (data or {}).get("route_stops") if isinstance(data, dict) else None
    if not isinstance(route_stops, list):
        return None
    return [str(s.get("stop_id")) for s in route_stops if isinstance(s, dict) and s.get("stop_id")]


async def _load_route_lines(client: HttpClient) -> list[RouteLine]:
    """Fetch every tracked route's stop sequences + stop coordinates."""
    jobs: list[RouteLine] = []
    # KMB
    for route, (outb, inb) in KMB_ROUTES.items():
        for bound in (outb, inb):
            jobs.append(RouteLine(route, "KMB", bound))
    # CTB
    for route, (outb, inb) in CTB_ROUTES.items():
        for bound in (outb, inb):
            jobs.append(RouteLine(route, "CTB", bound))
    # GMB
    for _route_id, (code, seqs) in GMB_ROUTES.items():
        for seq in seqs:
            jobs.append(RouteLine(code, "GMB", f"seq-{seq}"))

    # fetch all route-stop lists concurrently
    stop_lists = await client.gather_any(
        [
            _fetch_kmb_route_stops(client, r.route, r.bound)
            if r.operator == "KMB"
            else _fetch_ctb_route_stops(client, r.route, r.bound)
            if r.operator == "CTB"
            else _fetch_gmb_route_stops(
                client,
                next(route_id for route_id, (code, _seqs) in GMB_ROUTES.items() if code == r.route),
                int(r.bound.removeprefix("seq-")),
            )
            for r in jobs
        ]
    )
    for line, stop_ids in zip(jobs, stop_lists, strict=True):
        if not stop_ids:
            continue
        # fetch each stop's coords (cached)
        for sid in stop_ids:
            if line.operator == "KMB":
                url = KMB_STOP_URL.format(stop_id=sid)
            elif line.operator == "CTB":
                url = CTB_STOP_URL.format(stop_id=sid)
            else:
                url = GMB_STOP_URL.format(stop_id=sid)
            stop = await _fetch_stop(client, url, cache_key=f"td-stop-{sid}")
            if stop:
                line.stops.append(Stop(sid, stop[0], stop[1], stop[2]))
    lines = [line for line in jobs if len(line.stops) >= 2]
    geometry = await client.gather_any([_fetch_osrm_path(client, line) for line in lines])
    for line, routed in zip(lines, geometry, strict=True):
        if isinstance(routed, tuple):
            line.path, line.stop_offsets = routed
    return lines


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Approximate local WGS84 distance in metres."""
    lat_scale = 111_320.0
    lon_scale = lat_scale * math.cos(math.radians((a[0] + b[0]) / 2))
    return math.hypot((b[0] - a[0]) * lat_scale, (b[1] - a[1]) * lon_scale)


def _append_point(path: list[tuple[float, float]], point: tuple[float, float]) -> None:
    if not path or _distance(path[-1], point) > 0.2:
        path.append(point)


def _path_length(path: list[tuple[float, float]]) -> float:
    return sum(_distance(a, b) for a, b in zip(path, path[1:], strict=False))


def _point_at_offset(
    path: list[tuple[float, float]], offset: float
) -> tuple[float, float] | None:
    if not path or not math.isfinite(offset) or offset < 0:
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


def _valid_path_offsets(
    stops: list[Stop], path: list[tuple[float, float]], offsets: list[float]
) -> bool:
    """Reject incomplete, non-finite, or waypoint-detached routed geometry."""
    if len(stops) < 2 or len(path) < 2 or len(offsets) != len(stops):
        return False
    if any(
        not math.isfinite(value)
        for point in path
        for value in point
    ) or any(not math.isfinite(value) for value in offsets):
        return False
    if any(not (-90 <= lat <= 90 and -180 <= lon <= 180) for lat, lon in path):
        return False
    if not math.isclose(offsets[0], 0.0, abs_tol=0.01):
        return False
    if any(second <= first for first, second in zip(offsets, offsets[1:], strict=False)):
        return False
    total = _path_length(path)
    if not math.isclose(offsets[-1], total, rel_tol=0.001, abs_tol=1.0):
        return False
    for stop, offset in zip(stops, offsets, strict=True):
        point = _point_at_offset(path, offset)
        if point is None or _distance(point, (stop.lat, stop.lon)) > 110:
            return False
    return True


def _valid_route_line(line: RouteLine) -> bool:
    if not line.route or line.operator not in {"KMB", "CTB", "GMB"} or not line.bound:
        return False
    if any(
        not stop.stop_id
        or not math.isfinite(stop.lat)
        or not math.isfinite(stop.lon)
        or not (-90 <= stop.lat <= 90 and -180 <= stop.lon <= 180)
        for stop in line.stops
    ):
        return False
    if not _valid_path_offsets(line.stops, line.path, line.stop_offsets):
        return False
    return not (
        line.operator == "KMB"
        and line.route == "91"
        and any(_distance(point, HANG_HAU_VILLAGE) < HANG_HAU_VILLAGE_EXCLUSION_METRES for point in line.path)
    )


async def _fetch_osrm_chunk(
    client: HttpClient, stops: list[Stop], cache_key: str
) -> tuple[list[tuple[float, float]], list[float]] | None:
    coordinates = ";".join(f"{stop.lon:.7f},{stop.lat:.7f}" for stop in stops)
    query = "?alternatives=false&steps=true&overview=false&geometries=geojson"
    configured = _configured_osrm_base_url()
    for index, base_url in enumerate(_osrm_base_urls()):
        url = f"{base_url}/route/v1/driving/{coordinates}{query}"
        try:
            _, raw, _ = await client.fetch_json_cached(
                CachedFetch(
                    url,
                    TD_ROUTES_TTL_SECONDS,
                    cache_key=f"{cache_key}-transport-{index}",
                    timeout=30.0,
                ),
                headers={"User-Agent": USER_AGENT},
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("OSRM route chunk failed over %s: %s", url.split(":", 1)[0], type(exc).__name__)
            # The public project server has occasionally presented an expired
            # certificate.  Keep verification for HTTPS and only use cleartext
            # for this exact public, coordinate-only request and exact TLS
            # failure.  A configured endpoint must never be silently downgraded.
            if not (
                index == 0
                and configured == DEFAULT_OSRM_BASE_URL
                and isinstance(exc, aiohttp.ClientConnectorCertificateError)
            ):
                break
            continue
        routed = _parse_osrm_response(raw, stops)
        if routed is not None:
            return routed
        log.warning("OSRM returned invalid route geometry over %s", url.split(":", 1)[0])
    return None


def _configured_osrm_base_url() -> str:
    """Return the configured OSRM endpoint and its transport-only fallback.

    OSRM requests contain only public TD coordinates and request public OSM
    road geometry; they never contain credentials, cookies, or user data.  We
    therefore permit HTTP only as a last-resort transport fallback when the
    configured HTTPS endpoint cannot be used (for example, an expired public
    server certificate).  TLS verification remains enabled for HTTPS.
    """
    configured = os.getenv(OSRM_BASE_URL_ENV, DEFAULT_OSRM_BASE_URL).strip().rstrip("/")
    if not configured.startswith(("https://", "http://")):
        log.warning("ignoring invalid %s; using the default endpoint", OSRM_BASE_URL_ENV)
        configured = DEFAULT_OSRM_BASE_URL
    return configured


def _osrm_base_urls() -> tuple[str, ...]:
    configured = _configured_osrm_base_url()
    if configured == DEFAULT_OSRM_BASE_URL:
        return (configured, "http://router.project-osrm.org")
    return (configured,)


def _parse_osrm_response(
    raw: object, stops: list[Stop]
) -> tuple[list[tuple[float, float]], list[float]] | None:
    """Validate an OSRM route and retain arclength offsets for every TD stop."""
    routes = raw.get("routes") if isinstance(raw, dict) and raw.get("code") == "Ok" else None
    if not isinstance(routes, list) or not routes or not isinstance(routes[0], dict):
        return None
    waypoints = raw.get("waypoints")
    if not isinstance(waypoints, list) or len(waypoints) != len(stops):
        return None
    for waypoint, stop in zip(waypoints, stops, strict=True):
        location = waypoint.get("location") if isinstance(waypoint, dict) else None
        if not isinstance(location, list) or len(location) < 2:
            return None
        try:
            snapped = (float(location[1]), float(location[0]))
        except (TypeError, ValueError):
            return None
        # Reject a route if OSRM had to snap an official TD stop to a remote
        # road.  This is especially important around divided carriageways and
        # hillside roads where the nearest drivable edge may be misleading.
        if _distance(snapped, (stop.lat, stop.lon)) > 100:
            return None
    legs = routes[0].get("legs")
    if not isinstance(legs, list) or len(legs) != len(stops) - 1:
        return None
    path: list[tuple[float, float]] = []
    offsets = [0.0]
    for leg in legs:
        steps = leg.get("steps") if isinstance(leg, dict) else None
        if not isinstance(steps, list) or not steps:
            return None
        before_leg = len(path)
        for step in steps:
            geometry = step.get("geometry") if isinstance(step, dict) else None
            coords = geometry.get("coordinates") if isinstance(geometry, dict) else None
            if not isinstance(coords, list):
                return None
            for coord in coords:
                if not isinstance(coord, list) or len(coord) < 2:
                    return None
                try:
                    _append_point(path, (float(coord[1]), float(coord[0])))
                except (TypeError, ValueError):
                    return None
        if len(path) <= before_leg:
            return None
        offsets.append(_path_length(path))
    if not _valid_path_offsets(stops, path, offsets):
        return None
    return path, offsets


async def _fetch_osrm_path(
    client: HttpClient, line: RouteLine
) -> tuple[list[tuple[float, float]], list[float]] | None:
    """Route every official TD stop through OSM roads in overlapping chunks."""
    complete_path: list[tuple[float, float]] = []
    complete_offsets: list[float] = []
    start = 0
    while start < len(line.stops) - 1:
        end = min(len(line.stops), start + OSRM_MAX_WAYPOINTS)
        chunk = line.stops[start:end]
        digest = hashlib.sha256(
            ";".join(f"{s.stop_id}:{s.lat:.6f}:{s.lon:.6f}" for s in chunk).encode()
        ).hexdigest()[:20]
        routed = await _fetch_osrm_chunk(
            client, chunk, f"osm-osrm-route-{line.operator}-{line.route}-{line.bound}-{digest}"
        )
        if routed is None:
            return None
        chunk_path, chunk_offsets = routed
        base = complete_offsets[-1] if complete_offsets else 0.0
        for point in chunk_path:
            _append_point(complete_path, point)
        if not complete_offsets:
            complete_offsets.extend(chunk_offsets)
        else:
            complete_offsets.extend(base + offset for offset in chunk_offsets[1:])
        start = end - 1
    if len(complete_offsets) != len(line.stops):
        return None
    if not _valid_path_offsets(line.stops, complete_path, complete_offsets):
        return None
    candidate = RouteLine(line.route, line.operator, line.bound, line.stops, complete_path, complete_offsets)
    if not _valid_route_line(candidate):
        log.warning("OSRM route %s %s entered excluded Hang Hau Village lanes", line.route, line.bound)
        return None
    return complete_path, complete_offsets


def _serialize(geo: RouteGeometry) -> dict:
    return {
        "version": GEOMETRY_CACHE_VERSION,
        "fingerprint": _cache_fingerprint(),
        "fetched_at": geo.fetched_at,
        "routes": [
            {
                "route": r.route,
                "operator": r.operator,
                "bound": r.bound,
                "stops": [[s.stop_id, s.name, s.lat, s.lon] for s in r.stops],
                "path": [[lat, lon] for lat, lon in r.path],
                "stop_offsets": r.stop_offsets,
            }
            for r in geo.routes
        ],
        "stops": [[s.stop_id, s.name, s.lat, s.lon] for s in geo.stops],
    }


def _deserialize(data: dict) -> RouteGeometry:
    geo = RouteGeometry(fetched_at=float(data.get("fetched_at") or 0))
    for r in data.get("routes") or []:
        line = RouteLine(r["route"], r["operator"], r["bound"])
        line.stops = [Stop(s[0], s[1], float(s[2]), float(s[3])) for s in r.get("stops") or []]
        line.path = [(float(p[0]), float(p[1])) for p in r.get("path") or []]
        line.stop_offsets = [float(value) for value in r.get("stop_offsets") or []]
        geo.routes.append(line)
    geo.stops = [Stop(s[0], s[1], float(s[2]), float(s[3])) for s in data.get("stops") or []]
    return geo


def _cache_fingerprint() -> str:
    payload = json.dumps(
        {"kmb": KMB_ROUTES, "ctb": CTB_ROUTES, "gmb": GMB_ROUTES},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:20]


def _expected_route_keys() -> set[tuple[str, str, str]]:
    keys = {
        ("KMB", route, bound)
        for route, bounds in KMB_ROUTES.items()
        for bound in bounds
    }
    keys.update(
        ("CTB", route, bound)
        for route, bounds in CTB_ROUTES.items()
        for bound in bounds
    )
    keys.update(
        ("GMB", code, f"seq-{seq}")
        for code, seqs in GMB_ROUTES.values()
        for seq in seqs
    )
    return keys


def _cache_file(cache_dir: str) -> str:
    return os.path.join(cache_dir, "maps", GEOMETRY_CACHE_NAME)


def _load_disk_cache(cache_dir: str = ".cache") -> RouteGeometry | None:
    """Return structurally valid on-disk geometry, including expired last-good data."""
    try:
        path = _cache_file(cache_dir)
        if not os.path.isfile(path):
            return None
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if (
            data.get("version") != GEOMETRY_CACHE_VERSION
            or data.get("fingerprint") != _cache_fingerprint()
        ):
            return None
        geo = _deserialize(data)
        geo.routes = [line for line in geo.routes if _valid_route_line(line)]
        if not geo.routes:
            return None
        return geo
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return None


def _save_disk_cache(geo: RouteGeometry, cache_dir: str = ".cache") -> None:
    valid_routes = [line for line in geo.routes if _valid_route_line(line)]
    # A provider outage must never replace last-good geometry with an empty or
    # structurally poisoned cache document.
    if not valid_routes:
        return
    serializable = RouteGeometry(valid_routes, geo.stops, geo.fetched_at)
    try:
        path = _cache_file(cache_dir)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        temporary = f"{path}.tmp"
        with open(temporary, "w", encoding="utf-8") as f:
            json.dump(_serialize(serializable), f)
        os.replace(temporary, path)
    except OSError as exc:
        log.warning("route geometry cache write failed: %s", exc)


async def _refresh_route_geometry(
    client: HttpClient, cache_dir: str, cached: RouteGeometry | None
) -> RouteGeometry:
    """Refresh geometry under one bounded attempt, retaining every last-good line."""
    try:
        refreshed = await asyncio.wait_for(
            _load_route_lines(client), timeout=GEOMETRY_REFRESH_TIMEOUT_SECONDS
        )
    except Exception as exc:  # noqa: BLE001
        _refresh_retry_after[cache_dir] = time.monotonic() + GEOMETRY_FAILURE_COOLDOWN_SECONDS
        log.warning("route geometry refresh deferred after %s", type(exc).__name__)
        return cached or RouteGeometry()
    cached_by_route = {
        (line.operator, line.route, line.bound): line for line in (cached.routes if cached else [])
    }
    routes: list[RouteLine] = []
    refresh_complete = True
    seen: set[tuple[str, str, str]] = set()
    for line in refreshed:
        key = (line.operator, line.route, line.bound)
        seen.add(key)
        if _valid_route_line(line):
            routes.append(line)
        elif key in cached_by_route:
            routes.append(cached_by_route[key])
            refresh_complete = False
        else:
            refresh_complete = False
    for key, line in cached_by_route.items():
        if key not in seen:
            routes.append(line)
            refresh_complete = False
    if seen != _expected_route_keys():
        refresh_complete = False

    # Only fixed public-bus stops are map glyph candidates.  GMB sequences
    # remain intact in ``routes`` for ETA interpolation, but publishing their
    # stops here lets a partial geometry refresh accidentally turn a
    # minibus-only stop into a public map glyph.  The renderer retains the
    # same filter as defence in depth.
    stop_map: dict[str, Stop] = {}
    for line in routes:
        if line.operator == "GMB":
            continue
        for s in line.stops:
            stop_map.setdefault(s.stop_id, s)
    geo = RouteGeometry(
        routes=routes,
        stops=sorted(stop_map.values(), key=lambda s: s.name),
        # Retaining an expired route keeps the aggregate cache expired so the
        # next provider refresh will retry it rather than freshening stale data.
        fetched_at=time.time() if refresh_complete and routes else (cached.fetched_at if cached else 0),
    )
    _save_disk_cache(geo, cache_dir)
    return geo


def _finish_background_refresh(task: asyncio.Task, cache_dir: str) -> None:
    _refresh_tasks.pop(cache_dir, None)
    try:
        task.result()
    except Exception as exc:  # noqa: BLE001
        _refresh_retry_after[cache_dir] = time.monotonic() + GEOMETRY_FAILURE_COOLDOWN_SECONDS
        log.warning("background route geometry refresh failed: %s", type(exc).__name__)


async def fetch_route_geometry(client: HttpClient, cache_dir: str = ".cache") -> RouteGeometry:
    """Return last-good disk geometry immediately and refresh it in the background.

    A cold cache waits only for the short refresh budget.  Subsequent callers
    never serially retry a failed public routing service on the 15-second map
    cadence.
    """
    cached = _load_disk_cache(cache_dir)
    if cached is not None:
        if time.time() - cached.fetched_at <= TD_ROUTES_TTL_SECONDS:
            return cached
        task = _refresh_tasks.get(cache_dir)
        if task is None and time.monotonic() >= _refresh_retry_after.get(cache_dir, 0.0):
            task = asyncio.create_task(_refresh_route_geometry(client, cache_dir, cached))
            _refresh_tasks[cache_dir] = task
            task.add_done_callback(lambda done: _finish_background_refresh(done, cache_dir))
        return cached
    if time.monotonic() < _refresh_retry_after.get(cache_dir, 0.0):
        return RouteGeometry()
    return await _refresh_route_geometry(client, cache_dir, None)
