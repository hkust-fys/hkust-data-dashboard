"""Resolve traffic sections using official places, roads and bus geometry.

Road-name membership controls which notices we retain. It never, by itself,
proves that a bus passes the reported section. Missing/ambiguous locations fail
open for the notice and closed for its bus list.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import time
import unicodedata
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlencode

from dashboard.destinations import short_destination
from dashboard.http import FetchError, HttpClient
from dashboard.models import TrafficIncident
from dashboard.providers.official_roads import ROAD_CENTRELINE_QUERY_URL
from dashboard.providers.route_geometry import RouteLine
from dashboard.providers.tracked_roads import (
    _association_samples,
    _metres_between,
    _nearest_path_projection,
    _point_at,
    _point_to_segment_projection_metres,
    _unit_heading,
)

log = logging.getLogger(__name__)
LOCATION_SEARCH_URL = "https://www.map.gov.hk/gs/api/v1.0.0/locationSearch"
ROAD_COVERAGE_THRESHOLD = 0.5
GEOMETRY_TTL_SECONDS = 7 * 24 * 3600
FAILED_LOOKUP_TTL_SECONDS = 30 * 60
MAX_CACHE_ENTRIES = 128
MAX_CACHE_BYTES = 8 * 1024 * 1024
NEAR_ROAD_METRES = 180.0
NEAR_SECTION_HALF_METRES = 150.0
ROUTE_DISTANCE_METRES = 30.0
Point = tuple[float, float]
_stores: dict[str, dict] = {}


def _normal_place(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    # Transit operator words do not change the named station's identity.
    value = re.sub(r"\bmtr\b|港鐵", "", value)
    return re.sub(r"[\W_]+", "", value)


@lru_cache(maxsize=1)
def _hk80_transformer():
    # Construction reads PROJ's database, so it must remain runtime-only.
    from pyproj import Transformer

    return Transformer.from_crs(2326, 4326, always_xy=True)


def exact_place_point(query: str, rows: object) -> Point | None:
    """Accept exact bilingual names only; distant namesakes remain ambiguous."""
    if not isinstance(rows, list):
        return None
    matched: list[Point] = []
    for row in rows[:100]:
        if not isinstance(row, dict):
            continue
        names = [str(row.get(key) or "") for key in ("nameEN", "nameZH")]
        if _normal_place(query) not in {_normal_place(name) for name in names if name}:
            continue
        try:
            if "latitude" in row and "longitude" in row:
                lat, lon = float(row["latitude"]), float(row["longitude"])
            else:
                lon, lat = _hk80_transformer().transform(float(row["x"]), float(row["y"]))
        except (KeyError, TypeError, ValueError):
            continue
        if 22 <= lat <= 23 and 113.5 <= lon <= 114.7:
            matched.append((lat, lon))
    if not matched or any(_metres_between(matched[0], point) > 150 for point in matched):
        return None
    return tuple(sum(point[i] for point in matched) / len(matched) for i in (0, 1))


def location_terms(incident: TrafficIncident) -> tuple[tuple[str, ...], str]:
    """Extract near/between anchors and a direction, never a road-wide guess."""
    text = " ".join((incident.description, incident.title))
    anchors: tuple[str, ...] = ()
    between = re.search(
        r"\bbetween\s+(.+?)\s+and\s+(.+?)(?=\s+(?:is|are|which|due)\b|[.,;]|$)",
        text, re.I,
    )
    near = re.search(r"\bnear\s+(.+?)(?=\s+(?:is|are|was|which|due)\b|[.,;]|$)", text, re.I)
    if between:
        anchors = (between[1].strip(), between[2].strip())
    elif near:
        anchors = (near[1].strip(),)
    elif incident.near_landmark:
        anchors = (incident.near_landmark.strip(),)
    else:
        match = re.search(r"(?:介乎|由)(.+?)(?:與|至|到)(.+?)(?:之間|一段)", text)
        if match:
            anchors = (match[1].strip(), match[2].strip())
        else:
            match = re.search(r"近(.+?)(?=的|有|，|。|$)", text)
            if match:
                place = re.sub(r"(?:慢線|快線|中線|部分行車線|所有行車線)$", "", match[1])
                anchors = (place.strip(),)
    direction = incident.direction.strip()
    match = re.search(r"\(([^()]+?)\s+bounds?\)", text, re.I)
    if match or (match := re.search(r"往(.{1,20}?)方向", text)):
        direction = match[1].strip()
    elif not direction and (match := re.search(r"\b(?:north|south|east|west)bound\b|[東西南北]行|來回方向", text, re.I)):
        direction = match[0]
    direction = re.sub(r"\s+bounds?$", "", direction, flags=re.I).strip(" ()")
    return anchors, direction


def _load_store(cache_dir: str) -> dict:
    key = str(Path(cache_dir).resolve())
    if key not in _stores:
        path = Path(cache_dir) / "maps" / "traffic-locations.json"
        try:
            raw = json.loads(path.read_text(encoding="utf-8")) if path.stat().st_size <= MAX_CACHE_BYTES else {}
            entries = raw.get("entries", {}) if raw.get("version") == 1 else {}
            if not isinstance(entries, dict):
                entries = {}
        except (OSError, ValueError, TypeError):
            entries = {}
        _stores[key] = dict(list(entries.items())[-MAX_CACHE_ENTRIES:])
    return _stores[key]


async def _cached_lookup(client, cache_dir, key, fetch):
    store = _load_store(cache_dir)
    entry = store.get(key)
    if isinstance(entry, dict):
        ttl = GEOMETRY_TTL_SECONDS if entry.get("value") else FAILED_LOOKUP_TTL_SECONDS
        if 0 <= time.time() - float(entry.get("at", 0)) < ttl:
            return entry.get("value")
    try:
        async with asyncio.timeout(12):
            value = await fetch()
    except Exception as exc:
        log.info("traffic location lookup unavailable: %s", type(exc).__name__)
        value = None
    store[key] = {"at": time.time(), "value": value}
    while len(store) > MAX_CACHE_ENTRIES:
        del store[min(store, key=lambda item: float(store[item].get("at", 0)))]
    try:
        path = Path(cache_dir) / "maps" / "traffic-locations.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps({"version": 1, "entries": store}, ensure_ascii=False)
        if len(content.encode("utf-8")) <= MAX_CACHE_BYTES:
            temporary = path.with_suffix(".tmp")
            temporary.write_text(content, encoding="utf-8")
            temporary.replace(path)
    except OSError:
        log.warning("could not save traffic location cache")
    return value


async def _place(client: HttpClient, name: str, cache_dir: str) -> Point | None:
    async def fetch():
        raw = await client.fetch_json(f"{LOCATION_SEARCH_URL}?{urlencode({'q': name})}")
        return exact_place_point(name, raw)

    result = await _cached_lookup(client, cache_dir, "place:" + _normal_place(name), fetch)
    return tuple(result) if result else None


async def _whole_road(client: HttpClient, name: str, cache_dir: str) -> list[list[Point]]:
    async def fetch():
        paths = []
        for page in range(8):
            params = {
                "f": "json", "where": "UPPER(ENGLISHSTREETNAME) = '" + name.upper().replace("'", "''") + "'",
                "outFields": "ENGLISHSTREETNAME", "returnGeometry": "true", "outSR": 4326,
                "geometryPrecision": 6, "maxAllowableOffset": 0.00001,
                "resultRecordCount": 1000, "resultOffset": page * 1000, "orderByFields": "OBJECTID",
            }
            raw = await client.fetch_json(f"{ROAD_CENTRELINE_QUERY_URL}?{urlencode(params)}")
            if not isinstance(raw, dict) or "error" in raw or not isinstance(raw.get("features"), list):
                raise FetchError("Invalid complete road geometry")
            for feature in raw["features"]:
                for raw_path in (feature.get("geometry") or {}).get("paths", []):
                    points = [(float(p[1]), float(p[0])) for p in raw_path]
                    if len(points) < 2 or not all(22 <= p[0] <= 23 and 113.5 <= p[1] <= 114.7 for p in points):
                        raise FetchError("Invalid road coordinate")
                    paths.append(points)
            if not raw.get("exceededTransferLimit"):
                return paths
        raise FetchError("Incomplete road pagination")

    return await _cached_lookup(client, cache_dir, "road:" + name.casefold(), fetch) or []


def _crop(path, start, end):
    cumulative = [0.0]
    for first, second in zip(path, path[1:], strict=False):
        cumulative.append(cumulative[-1] + _metres_between(first, second))
    start, end = max(0.0, start), min(cumulative[-1], end)
    if end <= start:
        return []
    return [_point_at(path, cumulative, start), *[
        point for offset, point in zip(cumulative, path, strict=True) if start < offset < end
    ], _point_at(path, cumulative, end)]


def locate_section(paths: list[list[Point]], anchors: tuple[Point, ...]) -> list[list[Point]]:
    """Bound a near location or a between interval on the actual named road."""
    if not anchors:
        return paths
    result = []
    if len(anchors) == 2:
        # Only a continuous path proves the between interval; disconnected
        # features must not silently become two isolated 'near' locations.
        for path in paths:
            first, second = (_nearest_path_projection(anchor, tuple(path)) for anchor in anchors)
            if first and second and max(first[0], second[0]) <= NEAR_ROAD_METRES:
                low, high = sorted((first[3], second[3]))
                if section := _crop(path, low, high):
                    result.append(section)
        return result
    for path in paths:
        nearest = _nearest_path_projection(anchors[0], tuple(path))
        if nearest and nearest[0] <= NEAR_ROAD_METRES and (section := _crop(
            path, nearest[3] - NEAR_SECTION_HALF_METRES, nearest[3] + NEAR_SECTION_HALF_METRES,
        )):
            result.append(section)
    return result


def _direction_vector(direction: str, point: Point, destination: Point | None):
    cardinal = {
        "north": (1.0, 0.0), "northbound": (1.0, 0.0), "北行": (1.0, 0.0),
        "south": (-1.0, 0.0), "southbound": (-1.0, 0.0), "南行": (-1.0, 0.0),
        "east": (0.0, 1.0), "eastbound": (0.0, 1.0), "東行": (0.0, 1.0),
        "west": (0.0, -1.0), "westbound": (0.0, -1.0), "西行": (0.0, -1.0),
    }
    if direction.casefold() in cardinal:
        return cardinal[direction.casefold()]
    return _unit_heading(point, destination) if destination else None


def route_coverage(paths: list[list[Point]], line: RouteLine, direction: str = "", destination: Point | None = None) -> float:
    """Length-weighted fraction of a road/section following this bus path."""
    if len(line.path) < 2:
        return 0.0
    directional = bool(direction and direction.casefold() not in {"both", "both bounds", "both directions", "來回", "來回方向"})
    segments = []
    for start, end in zip(line.path, line.path[1:], strict=False):
        heading = _unit_heading(start, end)
        if heading:
            segments.append((start, end, heading))
    # A coarse spatial index avoids comparing every sample with every segment.
    scale = 2000.0
    grid: dict[tuple[int, int], list] = {}
    for segment in segments:
        start, end, _ = segment
        for y in range(math.floor(min(start[0], end[0]) * scale) - 1, math.floor(max(start[0], end[0]) * scale) + 2):
            for x in range(math.floor(min(start[1], end[1]) * scale) - 1, math.floor(max(start[1], end[1]) * scale) + 2):
                grid.setdefault((y, x), []).append(segment)
    total = supported = 0.0
    for path in paths:
        for point, road_heading, length in _association_samples(path, step_metres=20):
            total += length
            wanted = _direction_vector(direction, point, destination) if directional else None
            if directional and wanted is None:
                continue
            for start, end, heading in grid.get((math.floor(point[0] * scale), math.floor(point[1] * scale)), ()):
                if abs(sum(a * b for a, b in zip(heading, road_heading, strict=True))) < 0.9:
                    continue
                if wanted and sum(a * b for a, b in zip(heading, wanted, strict=True)) < 0.25:
                    continue
                distance, ratio = _point_to_segment_projection_metres(point, start, end)
                if 0 <= ratio <= 1 and distance <= ROUTE_DISTANCE_METRES:
                    supported += length
                    break
    return supported / total if total else 0.0


async def enrich_incident_locations(client, incidents, roads, lines, cache_dir=".cache"):
    """Retain every notice while independently resolving its possible buses."""
    from dashboard.providers.traffic import resolve_incident_road_keys

    if roads is None or not lines:
        return [replace(incident, location_resolution="unresolved") for incident in incidents]
    semaphore = asyncio.Semaphore(2)

    async def enrich(incident):
        async with semaphore:
            try:
                async with asyncio.timeout(20):
                    keys = resolve_incident_road_keys(incident, roads)
                    anchors, direction = location_terms(incident)
                    # Unparsed explicit location language must not invoke the
                    # road-coverage fallback (e.g. 'between' in unfamiliar form).
                    explicit_location = bool(anchors or incident.near_landmark or incident.between_landmark or re.search(
                        r"\b(?:near|between|junction)\b|近|交界|介乎|一段", incident.description, re.I,
                    ))
                    if incident.latitude is not None and incident.longitude is not None:
                        if not (22 <= incident.latitude <= 23 and 113.5 <= incident.longitude <= 114.7):
                            return replace(incident, location_resolution="unresolved")
                        anchor_points = ((incident.latitude, incident.longitude),)
                        explicit_location = True
                    else:
                        anchor_points = tuple([await _place(client, anchor, cache_dir) for anchor in anchors])
                        if anchors and any(point is None for point in anchor_points) and incident.translated_description:
                            # Use the officially paired Chinese notice when
                            # TD's English place spelling differs from LandsD.
                            chinese_anchors, chinese_direction = location_terms(replace(
                                incident, description=incident.translated_description, title="", direction="",
                            ))
                            if len(chinese_anchors) == len(anchors):
                                anchor_points = tuple([await _place(client, anchor, cache_dir) for anchor in chinese_anchors])
                                direction = chinese_direction or direction
                        if explicit_location and (not anchors or any(point is None for point in anchor_points)):
                            return replace(incident, location_resolution="unresolved")
                    direction_point = None
                    if direction and direction.casefold() not in {
                        "both", "both bounds", "both directions", "來回", "來回方向",
                        "north", "south", "east", "west", "northbound", "southbound", "eastbound", "westbound", "北行", "南行", "東行", "西行",
                    }:
                        direction_point = await _place(client, direction, cache_dir)
                    paths = []
                    for key in keys:
                        paths.extend(await _whole_road(client, roads.display_name(key), cache_dir))
                    sections = locate_section(paths, anchor_points)
                    if not sections:
                        return replace(incident, location_resolution="unresolved")
                    def compare():
                        found = []
                        for line in lines:
                            coverage = route_coverage(sections, line, direction, direction_point)
                            if coverage + 1e-9 >= ROAD_COVERAGE_THRESHOLD:
                                destination = short_destination(line.destination or (line.stops[-1].name if line.stops else ""))
                                label = f"{line.route} → {destination}" if destination else line.route
                                if label not in found:
                                    found.append(label)
                        return tuple(found)
                    affected = await asyncio.to_thread(compare)
                    return replace(
                        incident, affected_routes=affected,
                        affected_paths=tuple(
                            tuple((float(lat), float(lon)) for lat, lon in path)
                            for path in sections
                        ) if explicit_location else (),
                        location_resolution="section" if explicit_location else "road_coverage",
                    )
            except Exception as exc:
                log.info("traffic section unresolved: %s", type(exc).__name__)
                return replace(incident, location_resolution="unresolved")

    return list(await asyncio.gather(*(enrich(incident) for incident in incidents)))
