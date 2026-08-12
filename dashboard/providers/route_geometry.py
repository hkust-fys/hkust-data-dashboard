"""Official TD route-stop metadata used for public-stop map markers.

This provider only loads official KMB/Citybus stop coordinates (plus route sequences for marker-side
context). The versioned cache is rooted in the caller's ``cache_dir``. There
are no import-time network or filesystem side effects.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field

from dashboard.http import CachedFetch, HttpClient

log = logging.getLogger(__name__)

TD_ROUTES_TTL_SECONDS = 12 * 3600.0  # 12 hours: route/stop lists change rarely
GEOMETRY_CACHE_VERSION = 4
GEOMETRY_CACHE_NAME = "route-geometry.json"

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
    return [line for line in jobs if len(line.stops) >= 2]


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


def _cache_file(cache_dir: str) -> str:
    return os.path.join(cache_dir, "maps", GEOMETRY_CACHE_NAME)


def _load_disk_cache(cache_dir: str = ".cache") -> RouteGeometry | None:
    """Return the on-disk geometry if fresh (<= 12h), else None."""
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
        if time.time() - geo.fetched_at <= TD_ROUTES_TTL_SECONDS and geo.routes:
            return geo
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return None


def _save_disk_cache(geo: RouteGeometry, cache_dir: str = ".cache") -> None:
    try:
        path = _cache_file(cache_dir)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_serialize(geo), f)
    except OSError as exc:
        log.warning("route geometry cache write failed: %s", exc)


async def fetch_route_geometry(client: HttpClient, cache_dir: str = ".cache") -> RouteGeometry:
    """Fetch official route sequences and public stops, cached for 12 hours."""
    cached = _load_disk_cache(cache_dir)
    if cached is not None:
        return cached
    routes = await _load_route_lines(client)
    # deduplicated stops across all routes (official bus stops only; GMB
    # minibus stops excluded — minibuses board/unboard anywhere)
    stop_map: dict[str, Stop] = {}
    for line in routes:
        if line.operator == "GMB":
            continue  # minibuses: no stop markers
        for s in line.stops:
            stop_map.setdefault(s.stop_id, s)
    geo = RouteGeometry(
        routes=routes,
        stops=sorted(stop_map.values(), key=lambda s: s.name),
        fetched_at=time.time(),
    )
    _save_disk_cache(geo, cache_dir)
    return geo
