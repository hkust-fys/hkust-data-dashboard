"""Tracked-road derivation from official route geometry via OpenStreetMap.

The dashboard listens for TD traffic news on every road its buses actually
travel.  Rather than hand-maintaining that list, each official HKeMobility
route line is sampled and the sampled points are matched against named OSM
highways.  ONE batched Overpass query (``out geom``) covers every direction;
matching against the returned way geometry happens locally.  The result is
cached on disk with last-good retention; a failure falls back to the last good
table, then to a small curated seed so traffic-news filtering never goes blind.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field

import aiohttp

from dashboard.http import HttpClient
from dashboard.providers.route_geometry import RouteLine, fetch_route_geometry

log = logging.getLogger(__name__)

ROADS_TTL_SECONDS = 12 * 3600.0
ROADS_CACHE_VERSION = 4
ROADS_CACHE_NAME = "tracked-roads.json"
OVERPASS_TIMEOUT_SECONDS = 10.0
OVERPASS_ATTEMPTS = 1
OVERPASS_HTTP_FALLBACK_ATTEMPTS = 2
ROADS_FAILURE_COOLDOWN_SECONDS = 30 * 60.0

# Public-coordinate Overpass queries follow the same rule as the retired OSRM
# fallback: after an actual TLS certificate failure, plain HTTP is allowed for
# these keyless public endpoints only.
OVERPASS_URLS: tuple[str, ...] = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "http://overpass-api.de/api/interpreter",
)

# Curated seed used only when neither fresh nor cached OSM data is available.
FALLBACK_ROADS: tuple[str, ...] = (
    "Clear Water Bay Road",
    "New Clear Water Bay Road",
    "Lung Cheung Road",
    "Hiram's Highway",
    "Sai Kung Road",
    "Tai Po Tsai Road",
    "University Road",
    "Ngan Ying Road",
    "Ying Yip Road",
    "Hang Hau Road",
    "Wan Po Road",
    "Po Ning Road",
    "Po Shun Road",
    "Tseung Kwan O Tunnel Road",
    "Po Lam Road",
    "Chun Ying Street",
)

# OSM highway values that are not roads a bus route news item would name.
_EXCLUDED_HIGHWAY_TYPES = {
    "path",
    "footway",
    "cycleway",
    "steps",
    "track",
    "bridleway",
    "pedestrian",
}

SAMPLE_STEP_METRES = 150.0
MATCH_RADIUS_METRES = 30.0
ASSOCIATION_SAMPLE_STEP_METRES = 20.0
MAX_ASSOCIATION_HEADING_DEGREES = 25.0
MIN_ALIGNED_SAMPLES = 3
MIN_ALIGNED_OVERLAP_METRES = 50.0
MIN_ALIGNED_RUN_METRES = 40.0
MIN_COMPLETE_SHORT_WAY_METRES = 35.0
COMPLETE_SHORT_WAY_SAMPLE_STEP_METRES = 5.0
COMPLETE_SHORT_WAY_MATCH_RADIUS_METRES = 8.0
MAX_COMPLETE_SHORT_WAY_HEADING_DEGREES = 10.0
MIN_COMPLETE_SHORT_WAY_COVERAGE = 0.85
MAX_COMPLETE_SHORT_WAY_GAP_METRES = 6.0
# A TD news coordinate must be genuinely on the named OSM way.  This generous
# bound covers GPS/map snapping error without turning an incident into a whole
# route or whole-road highlight.  The returned line is capped at 500 m either
# side of the nearest projected point.
SEGMENT_MATCH_RADIUS_METRES = 120.0
SEGMENT_HALF_LENGTH_METRES = 500.0
# Coordinate-less TD notices may highlight only a genuinely short named way.
# This avoids turning an ambiguous notice into a whole-corridor overlay.
SHORT_ROAD_FALLBACK_MAX_METRES = 900.0


@dataclass(frozen=True)
class TrackedRoads:
    """Road names heard in TD feeds plus the routes serving each road."""

    display_names: dict[str, str] = field(default_factory=dict)  # key -> display
    aliases: dict[str, str] = field(default_factory=dict)  # key -> match phrase
    road_routes: dict[str, tuple[str, ...]] = field(default_factory=dict)
    # Canonical road key -> one or more OSM way polylines.  Only ways which
    # intersected a tracked route are retained; fallback roads have no paths.
    paths: dict[str, tuple[tuple[tuple[float, float], ...], ...]] = field(default_factory=dict)
    source: str = ""
    fetched_at: float = 0.0

    def match(self, text: str) -> list[str]:
        """Return canonical road keys whose aliases appear in ``text``.

        Longer aliases win: "New Clear Water Bay Road" suppresses the
        overlapping "clear water bay road" substring match so one notice maps
        to one road.
        """
        lowered = text.lower().replace("\u2019", "'")
        hits: list[tuple[int, int, str]] = []
        for key, alias in self.aliases.items():
            phrase = alias.replace("\u2019", "'")
            start = lowered.find(phrase)
            while start != -1:
                hits.append((start, start + len(phrase), key))
                start = lowered.find(phrase, start + 1)
        kept: list[tuple[int, int, str]] = [
            hit
            for hit in hits
            if not any(
                other is not hit
                and other[0] <= hit[0]
                and hit[1] <= other[1]
                and (other[1] - other[0]) > (hit[1] - hit[0])
                for other in hits
            )
        ]
        seen: set[str] = set()
        ordered: list[str] = []
        for _start, _end, key in sorted(kept):
            if key not in seen:
                seen.add(key)
                ordered.append(key)
        return ordered

    def routes_for_text(self, text: str) -> list[str]:
        """Sorted route labels serving any road matched in ``text``."""
        matched: set[str] = set()
        for key in self.match(text):
            matched.update(self.road_routes.get(key, ()))
        return _sorted_routes(matched)

    def routes_for_keys(self, keys: list[str]) -> list[str]:
        """Sorted route labels serving the given canonical road keys."""
        matched: set[str] = set()
        for key in keys:
            matched.update(self.road_routes.get(key, ()))
        return _sorted_routes(matched)

    def display_name(self, key: str) -> str:
        return self.display_names.get(key, key.replace("-", " ").title())

    def segments_near(
        self,
        keys: list[str],
        latitude: float | None = None,
        longitude: float | None = None,
    ) -> list[list[tuple[float, float]]]:
        """Crop matched OSM ways around a TD coordinate.

        With an anchor, no cached path or path within 120 m yields no segment.
        Each anchored result is limited to 500 m on either side of the nearest
        projected point. Without an anchor, only roads no longer than 900 m
        are returned and an informational message is emitted, so ambiguous notices never
        become whole-corridor guesses.
        """
        results: list[list[tuple[float, float]]] = []
        if latitude is None and longitude is None:
            for key in keys:
                paths = self.paths.get(key, ())
                if not paths:
                    continue
                length = sum(_path_length_metres(path) for path in paths)
                if length <= SHORT_ROAD_FALLBACK_MAX_METRES:
                    log.info(
                        "using coordinate-less short-road traffic segment for %s (%.0f m)",
                        self.display_name(key),
                        length,
                    )
                    results.extend(list(path) for path in paths)
            return results
        if latitude is None or longitude is None:
            return []
        anchor = (latitude, longitude)
        for key in keys:
            paths = self.paths.get(key, ())
            candidates = [
                (nearest[0], path, nearest)
                for path in paths
                if len(path) >= 2
                for nearest in [_nearest_path_projection(anchor, path)]
                if nearest is not None
            ]
            if not candidates:
                continue
            distance, path, nearest = min(candidates, key=lambda item: item[0])
            if distance > SEGMENT_MATCH_RADIUS_METRES:
                continue
            _distance, _segment_index, _ratio, along = nearest
            cumulative = [0.0]
            for first, second in zip(path, path[1:], strict=False):
                cumulative.append(cumulative[-1] + _metres_between(first, second))
            start = max(0.0, along - SEGMENT_HALF_LENGTH_METRES)
            end = min(cumulative[-1], along + SEGMENT_HALF_LENGTH_METRES)
            cropped: list[tuple[float, float]] = [_point_at(path, cumulative, start)]
            for index, point in enumerate(path[1:], 1):
                if start < cumulative[index] < end:
                    cropped.append(point)
            cropped.append(_point_at(path, cumulative, end))
            if cropped[-1] != cropped[-2]:
                results.append(cropped)
        return results


def _nearest_path_projection(
    point: tuple[float, float], path: tuple[tuple[float, float], ...]
) -> tuple[float, int, float, float] | None:
    best: tuple[float, int, float, float] | None = None
    along = 0.0
    for index, (start, end) in enumerate(zip(path, path[1:], strict=False)):
        lat_scale = 111_320.0
        lon_scale = lat_scale * math.cos(math.radians((start[0] + end[0]) / 2))
        ex, ey = (end[0] - start[0]) * lat_scale, (end[1] - start[1]) * lon_scale
        px, py = (point[0] - start[0]) * lat_scale, (point[1] - start[1]) * lon_scale
        length_sq = ex * ex + ey * ey
        ratio = 0.0 if length_sq == 0 else max(0.0, min(1.0, (px * ex + py * ey) / length_sq))
        distance = math.hypot(px - ratio * ex, py - ratio * ey)
        candidate = (distance, index, ratio, along + ratio * math.sqrt(length_sq))
        if best is None or candidate[0] < best[0]:
            best = candidate
        along += math.sqrt(length_sq)
    return best


def _point_at(
    path: tuple[tuple[float, float], ...], cumulative: list[float], distance: float
) -> tuple[float, float]:
    for index, (start, end) in enumerate(zip(path, path[1:], strict=False), 1):
        if distance <= cumulative[index]:
            span = cumulative[index] - cumulative[index - 1]
            ratio = 0.0 if span == 0 else (distance - cumulative[index - 1]) / span
            return (start[0] + ratio * (end[0] - start[0]), start[1] + ratio * (end[1] - start[1]))
    return path[-1]


def _path_length_metres(path: tuple[tuple[float, float], ...]) -> float:
    return sum(_metres_between(first, second) for first, second in zip(path, path[1:], strict=False))


def _sorted_routes(routes: set[str]) -> list[str]:
    def sort_key(route: str) -> tuple[int, str]:
        digits = "".join(ch for ch in route if ch.isdigit())
        letters = "".join(ch for ch in route if not ch.isdigit())
        return (int(digits) if digits else 9999, letters)

    return sorted(routes, key=sort_key)


def _sample_path(
    path: list[tuple[float, float]], step_metres: float = SAMPLE_STEP_METRES
) -> list[tuple[float, float]]:
    lat_scale = 111_320.0
    if len(path) < 2:
        return list(path)
    out = [path[0]]
    acc = 0.0
    for first, second in zip(path, path[1:], strict=False):
        mid_lat = (first[0] + second[0]) / 2
        lon_scale = lat_scale * math.cos(math.radians(mid_lat))
        acc += math.hypot(
            (second[0] - first[0]) * lat_scale, (second[1] - first[1]) * lon_scale
        )
        if acc >= step_metres:
            out.append(second)
            acc = 0.0
    return out


def build_overpass_query(
    points: list[tuple[float, float]], timeout_seconds: float = 10.0
) -> str:
    """One union query returning geometry of named highways near all samples."""
    clauses = "\n".join(
        f'  way(around:{MATCH_RADIUS_METRES:.0f},{lat:.6f},{lon:.6f})["highway"]["name"];'
        for lat, lon in points
    )
    timeout = max(1, int(timeout_seconds))
    return f"[out:json][timeout:{timeout}];\n(\n{clauses}\n);\nout geom;"


def parse_overpass_roads(raw: dict) -> list[str]:
    """Extract English road names from an Overpass response (tags-only output).

    Retained for tests and quick diagnostics; the runtime path uses
    ``collect_way_roads`` + ``roads_for_line`` on ``out geom`` responses.
    """
    names: list[str] = []
    seen: set[str] = set()
    for element in raw.get("elements") or []:
        if not isinstance(element, dict):
            continue
        tags = element.get("tags") or {}
        highway = str(tags.get("highway") or "")
        if highway in _EXCLUDED_HIGHWAY_TYPES:
            continue
        name = str(tags.get("name:en") or tags.get("name") or "").strip()
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        names.append(name)
    return names


def _metres_between(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat_scale = 111_320.0
    lon_scale = lat_scale * math.cos(math.radians((a[0] + b[0]) / 2))
    return math.hypot((b[0] - a[0]) * lat_scale, (b[1] - a[1]) * lon_scale)


def _point_to_segment_projection_metres(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> tuple[float, float]:
    """Distance and unclamped along-segment ratio in local metric space."""
    lat_scale = 111_320.0
    lon_scale = lat_scale * math.cos(math.radians((start[0] + end[0]) / 2))
    px = (point[0] - start[0]) * lat_scale
    py = (point[1] - start[1]) * lon_scale
    ex = (end[0] - start[0]) * lat_scale
    ey = (end[1] - start[1]) * lon_scale
    length_sq = ex * ex + ey * ey
    raw_ratio = 0.0 if length_sq == 0 else (px * ex + py * ey) / length_sq
    ratio = max(0.0, min(1.0, raw_ratio))
    dx = px - ratio * ex
    dy = py - ratio * ey
    return math.hypot(dx, dy), raw_ratio


def _unit_heading(
    start: tuple[float, float], end: tuple[float, float]
) -> tuple[float, float] | None:
    """Return a local metric heading; direction is ignored by the caller."""
    lat_scale = 111_320.0
    lon_scale = lat_scale * math.cos(math.radians((start[0] + end[0]) / 2))
    north = (end[0] - start[0]) * lat_scale
    east = (end[1] - start[1]) * lon_scale
    length = math.hypot(north, east)
    if length == 0:
        return None
    return (north / length, east / length)


def _association_samples(
    path: list[tuple[float, float]],
    step_metres: float = ASSOCIATION_SAMPLE_STEP_METRES,
) -> list[tuple[tuple[float, float], tuple[float, float], float]]:
    """Densely sample a route with local headings and represented lengths."""
    samples: list[tuple[tuple[float, float], tuple[float, float], float]] = []
    for start, end in zip(path, path[1:], strict=False):
        length = _metres_between(start, end)
        heading = _unit_heading(start, end)
        if length == 0 or heading is None:
            continue
        pieces = max(1, math.ceil(length / step_metres))
        represented = length / pieces
        for index in range(pieces):
            ratio = (index + 0.5) / pieces
            point = (
                start[0] + ratio * (end[0] - start[0]),
                start[1] + ratio * (end[1] - start[1]),
            )
            samples.append((point, heading, represented))
    return samples


def _way_segment(
    start: tuple[float, float], end: tuple[float, float]
) -> tuple[
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
    tuple[float, float, float, float],
] | None:
    heading = _unit_heading(start, end)
    if heading is None:
        return None
    lat_pad = MATCH_RADIUS_METRES / 111_320.0
    mid_lat = (start[0] + end[0]) / 2
    lon_scale = 111_320.0 * max(0.1, math.cos(math.radians(mid_lat)))
    lon_pad = MATCH_RADIUS_METRES / lon_scale
    bounds = (
        min(start[0], end[0]) - lat_pad,
        max(start[0], end[0]) + lat_pad,
        min(start[1], end[1]) - lon_pad,
        max(start[1], end[1]) + lon_pad,
    )
    return (start, end, heading, bounds)


def _complete_short_way_matches_route(
    way_points: list[tuple[float, float]],
    route_segments: list[tuple],
) -> bool:
    """Accept only a nearly complete, tightly registered short named way."""
    way_length = sum(
        _metres_between(start, end)
        for start, end in zip(way_points, way_points[1:], strict=False)
    )
    if not (MIN_COMPLETE_SHORT_WAY_METRES <= way_length < MIN_ALIGNED_OVERLAP_METRES):
        return False

    samples = _association_samples(
        way_points,
        step_metres=COMPLETE_SHORT_WAY_SAMPLE_STEP_METRES,
    )
    if not samples:
        return False
    cosine_limit = math.cos(math.radians(MAX_COMPLETE_SHORT_WAY_HEADING_DEGREES))
    supported: list[bool] = []
    supported_length = 0.0
    current_gap = 0.0
    longest_gap = 0.0
    for point, way_heading, represented in samples:
        matched = False
        for start, end, route_heading, bounds in route_segments:
            min_lat, max_lat, min_lon, max_lon = bounds
            if not (
                min_lat <= point[0] <= max_lat and min_lon <= point[1] <= max_lon
            ):
                continue
            alignment = abs(
                way_heading[0] * route_heading[0]
                + way_heading[1] * route_heading[1]
            )
            if alignment < cosine_limit:
                continue
            distance, ratio = _point_to_segment_projection_metres(point, start, end)
            if (
                0.0 <= ratio <= 1.0
                and distance <= COMPLETE_SHORT_WAY_MATCH_RADIUS_METRES
            ):
                matched = True
                break
        supported.append(matched)
        if matched:
            supported_length += represented
            current_gap = 0.0
        else:
            current_gap += represented
            longest_gap = max(longest_gap, current_gap)

    return (
        supported[0]
        and supported[-1]
        and supported_length / way_length >= MIN_COMPLETE_SHORT_WAY_COVERAGE
        and longest_gap <= MAX_COMPLETE_SHORT_WAY_GAP_METRES
    )


def collect_way_roads(raw: dict) -> list[dict]:
    """Normalize Overpass elements into {name, name_en, points} road ways."""
    ways: list[dict] = []
    for element in raw.get("elements") or []:
        if not isinstance(element, dict) or element.get("type") != "way":
            continue
        tags = element.get("tags") or {}
        highway = str(tags.get("highway") or "")
        if highway in _EXCLUDED_HIGHWAY_TYPES:
            continue
        name = str(tags.get("name") or "").strip()
        name_en = str(tags.get("name:en") or "").strip()
        if not name and not name_en:
            continue
        geometry = element.get("geometry") or []
        points = [
            (float(node["lat"]), float(node["lon"]))
            for node in geometry
            if isinstance(node, dict) and "lat" in node and "lon" in node
        ]
        if len(points) < 2:
            continue
        ways.append({"name": name, "name_en": name_en, "points": points})
    return ways


def roads_for_line(line_points: list[tuple[float, float]], ways: list[dict]) -> list[str]:
    """Road names with sustained, heading-aligned overlap with a route line."""
    route_samples = _association_samples(line_points)
    if not route_samples:
        return []

    route_segments = [
        segment
        for start, end in zip(line_points, line_points[1:], strict=False)
        if (segment := _way_segment(start, end)) is not None
    ]
    grouped: dict[str, tuple[str, list[tuple], list[list[tuple[float, float]]]]] = {}
    for way in ways:
        display = str(way.get("name_en") or way.get("name") or "").strip()
        if not display:
            continue
        key = display.casefold()
        if key not in grouped:
            grouped[key] = (display, [], [])
        segments = grouped[key][1]
        points = list(way.get("points") or [])
        way_segments: list[tuple] = []
        for start, end in zip(points, points[1:], strict=False):
            segment = _way_segment(start, end)
            if segment is not None:
                segments.append(segment)
                way_segments.append(segment)
        if way_segments:
            grouped[key][2].append(points)

    cosine_limit = math.cos(math.radians(MAX_ASSOCIATION_HEADING_DEGREES))
    names: list[str] = []
    for display, segments, way_paths in grouped.values():
        aligned_samples = 0
        aligned_overlap = 0.0
        current_run = 0.0
        longest_run = 0.0
        for point, route_heading, represented in route_samples:
            supported = False
            for start, end, way_heading, bounds in segments:
                min_lat, max_lat, min_lon, max_lon = bounds
                if not (
                    min_lat <= point[0] <= max_lat and min_lon <= point[1] <= max_lon
                ):
                    continue
                # OSM one-way geometry may run opposite the route coordinate
                # order, so compare the undirected heading via |dot product|.
                alignment = abs(
                    route_heading[0] * way_heading[0]
                    + route_heading[1] * way_heading[1]
                )
                if alignment < cosine_limit:
                    continue
                distance, ratio = _point_to_segment_projection_metres(point, start, end)
                if 0.0 <= ratio <= 1.0 and distance <= MATCH_RADIUS_METRES:
                    supported = True
                    break
            if supported:
                aligned_samples += 1
                aligned_overlap += represented
                current_run += represented
                longest_run = max(longest_run, current_run)
            else:
                current_run = 0.0
        normal_overlap = (
            sum(_metres_between(start, end) for start, end, _heading, _bounds in segments)
            >= MIN_ALIGNED_OVERLAP_METRES
            and aligned_samples >= MIN_ALIGNED_SAMPLES
            and aligned_overlap >= MIN_ALIGNED_OVERLAP_METRES
            and longest_run >= MIN_ALIGNED_RUN_METRES
        )
        complete_short_way = not normal_overlap and any(
            _complete_short_way_matches_route(path, route_segments)
            for path in way_paths
        )
        if normal_overlap or complete_short_way:
            names.append(display)
    return names


def build_tracked_roads(
    lines: list[RouteLine],
    roads_by_line: list[list[str]],
    ways: list[dict] | None = None,
) -> TrackedRoads:
    """Combine per-direction OSM road lists into the shared road table."""
    display_by_key: dict[str, str] = {}
    road_routes: dict[str, set[str]] = {}
    for line, roads in zip(lines, roads_by_line, strict=True):
        label = line.route
        for name in roads:
            key = name.lower()
            display_by_key.setdefault(key, name)
            road_routes.setdefault(key, set()).add(label)
    paths_by_key: dict[str, list[tuple[tuple[float, float], ...]]] = {}
    for way in ways or []:
        display = way.get("name_en") or way.get("name")
        points = tuple(way.get("points") or ())
        key = str(display or "").lower()
        if key in display_by_key and len(points) >= 2 and points not in paths_by_key.setdefault(key, []):
            paths_by_key[key].append(points)
    return TrackedRoads(
        display_names=dict(display_by_key),
        aliases={key: key for key in display_by_key},
        road_routes={
            key: tuple(_sorted_routes(routes)) for key, routes in road_routes.items()
        },
        paths={key: tuple(value) for key, value in paths_by_key.items()},
        source="osm",
    )


def _fallback_roads() -> TrackedRoads:
    return TrackedRoads(
        display_names={name.lower(): name for name in FALLBACK_ROADS},
        aliases={name.lower(): name.lower() for name in FALLBACK_ROADS},
        road_routes={},
        source="fallback",
    )


def fallback_roads() -> TrackedRoads:
    """Curated seed used when OSM derivation is unavailable."""
    return _fallback_roads()


def _cache_file(cache_dir: str) -> str:
    return os.path.join(cache_dir, "maps", ROADS_CACHE_NAME)


def _parse_cached_paths(raw_paths: object) -> dict[str, tuple[tuple[tuple[float, float], ...], ...]]:
    """Read the current multi-path shape and the brief singular-path shape."""
    if not isinstance(raw_paths, dict):
        return {}
    parsed: dict[str, tuple[tuple[tuple[float, float], ...], ...]] = {}
    for key, raw_value in raw_paths.items():
        if not isinstance(key, str) or not isinstance(raw_value, list) or not raw_value:
            continue
        # Older development caches stored one path directly as [[lat, lon], ...].
        if isinstance(raw_value[0], list) and len(raw_value[0]) >= 2 and isinstance(
            raw_value[0][0], (int, float)
        ):
            raw_value = [raw_value]
        paths: list[tuple[tuple[float, float], ...]] = []
        for raw_path in raw_value:
            if not isinstance(raw_path, list) or len(raw_path) < 2:
                continue
            try:
                path = tuple((float(point[0]), float(point[1])) for point in raw_path)
            except (IndexError, TypeError, ValueError):
                continue
            if path not in paths:
                paths.append(path)
        if paths:
            parsed[key] = tuple(paths)
    return parsed


def _load_disk_cache(cache_dir: str) -> TrackedRoads | None:
    try:
        with open(_cache_file(cache_dir), encoding="utf-8") as file:
            raw = json.load(file)
        if raw.get("version") != ROADS_CACHE_VERSION:
            return None
        display_raw = raw.get("display_names")
        if isinstance(display_raw, dict):
            display_names = dict(display_raw)
        else:  # legacy tuple form
            display_names = {str(name).lower(): str(name) for name in display_raw or ()}
        return TrackedRoads(
            display_names=display_names,
            aliases=dict(raw.get("aliases") or {}),
            road_routes={
                key: tuple(value) for key, value in (raw.get("road_routes") or {}).items()
            },
            paths=_parse_cached_paths(raw.get("paths")),
            source=str(raw.get("source") or ""),
            fetched_at=float(raw.get("fetched_at") or 0),
        )
    except (OSError, ValueError, TypeError):
        return None


def _save_disk_cache(roads: TrackedRoads, cache_dir: str) -> None:
    if not roads.display_names:
        return
    payload = {
        "version": ROADS_CACHE_VERSION,
        "fetched_at": roads.fetched_at,
        "source": roads.source,
        "display_names": roads.display_names,
        "aliases": roads.aliases,
        "road_routes": {key: list(value) for key, value in roads.road_routes.items()},
        "paths": {
            key: [[list(point) for point in path] for path in paths]
            for key, paths in roads.paths.items()
        },
    }
    try:
        path = _cache_file(cache_dir)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        temporary = f"{path}.tmp"
        with open(temporary, "w", encoding="utf-8") as file:
            json.dump(payload, file)
        os.replace(temporary, path)
    except OSError as exc:
        log.warning("tracked-roads cache write failed: %s", exc)


def _is_plain_http(url: str) -> bool:
    return url.startswith("http://")


def _is_tls_failure(exc: BaseException) -> bool:
    """True when an HTTPS mirror failed with a real TLS/certificate error."""
    return isinstance(exc, aiohttp.ClientConnectorCertificateError)


async def _fetch_overpass(
    client: HttpClient, query: str, timeout_seconds: float | None = None
) -> dict:
    """Query Overpass mirrors in order; HTTP is a certificate-only escape hatch.

    The plain-HTTP mirror is never contacted unless an HTTPS mirror has already
    reported a real TLS certificate failure, and even then it is retried once so
    that a single dropped connection does not discard the whole refresh.
    """
    budget = float(timeout_seconds or getattr(client, "timeout_seconds", OVERPASS_TIMEOUT_SECONDS))
    saw_tls_failure = False
    last_error: Exception | None = None
    started = time.monotonic()
    try:
        async with asyncio.timeout(max(0.001, budget)):
            for url in OVERPASS_URLS:
                if _is_plain_http(url) and not saw_tls_failure:
                    log.info("skipping plain-HTTP Overpass fallback: no TLS certificate failure seen")
                    continue
                try:
                    remaining = max(0.001, budget - (time.monotonic() - started))
                    return await client.post_form_json(
                        url,
                        {"data": query},
                        timeout_seconds=remaining,
                        attempts=(
                            OVERPASS_HTTP_FALLBACK_ATTEMPTS if _is_plain_http(url) else OVERPASS_ATTEMPTS
                        ),
                    )
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    if _is_tls_failure(exc):
                        saw_tls_failure = True
                        log.warning(
                            "Overpass TLS certificate failure reported by %s; "
                            "plain-HTTP fallback is now authorised",
                            url.split("/")[2],
                        )
                    else:
                        log.warning("Overpass %s failed: %s", url.split("/")[2], type(exc).__name__)
    except TimeoutError as exc:
        raise TimeoutError(f"Overpass refresh exceeded {budget:.1f}s budget") from exc
    assert last_error is not None
    raise last_error


_refresh_tasks: dict[str, asyncio.Task[TrackedRoads]] = {}
_refresh_retry_after: dict[str, float] = {}
_refresh_shutdown = False


async def _refresh_roads(client: HttpClient, cache_dir: str) -> TrackedRoads:
    geometry = await fetch_route_geometry(client, cache_dir=cache_dir)
    lines = [line for line in geometry.routes if len(line.path) >= 2]
    if not lines:
        raise RuntimeError("no route geometry available for road derivation")

    # One polite batched query: every direction's samples in a single union.
    samples = [
        point
        for line in lines
        for point in _sample_path(line.path)
    ]
    budget = float(getattr(client, "timeout_seconds", OVERPASS_TIMEOUT_SECONDS))
    raw = await _fetch_overpass(client, build_overpass_query(samples, budget), budget)
    ways = collect_way_roads(raw)
    if not ways:
        raise RuntimeError("Overpass returned no named road geometry")

    roads_by_line = [roads_for_line(line.path, ways) for line in lines]
    failures = sum(1 for roads in roads_by_line if not roads)
    if failures == len(lines):
        raise RuntimeError("no route line matched any named OSM road")
    if failures:
        log.warning(
            "tracked roads: %d/%d directions matched no OSM road; using partial union",
            failures,
            len(lines),
        )
    roads = build_tracked_roads(lines, roads_by_line, ways)
    roads = replace_fetched_at(roads, time.time())
    _save_disk_cache(roads, cache_dir)
    return roads


def replace_fetched_at(roads: TrackedRoads, fetched_at: float) -> TrackedRoads:
    """Return a copy stamped with ``fetched_at`` (the table is frozen)."""
    return TrackedRoads(
        display_names=roads.display_names,
        aliases=roads.aliases,
        road_routes=roads.road_routes,
        paths=roads.paths,
        source=roads.source,
        fetched_at=fetched_at,
    )


def _finish_refresh(task: asyncio.Task[TrackedRoads], cache_dir: str) -> None:
    _refresh_tasks.pop(cache_dir, None)
    if _refresh_shutdown:
        return
    if task.cancelled():
        # Normal shutdown must not create a failure cooldown or an exception
        # from this task's done callback.
        return
    try:
        task.result()
    except Exception as exc:  # noqa: BLE001
        _refresh_retry_after[cache_dir] = (
            time.monotonic() + ROADS_FAILURE_COOLDOWN_SECONDS
        )
        log.warning("tracked-roads refresh failed: %s", type(exc).__name__)


async def fetch_tracked_roads(
    client: HttpClient, cache_dir: str = ".cache", *, wait_for_refresh: bool = True
) -> TrackedRoads:
    """Return fresh/last-good tracked roads, refreshing expired data in background.

    Order of preference: fresh disk cache, then a background refresh of the
    expired one, then a blocking refresh, then the curated fallback seed.
    """
    global _refresh_shutdown
    _refresh_shutdown = False
    cached = _load_disk_cache(cache_dir)
    if cached is not None and cached.display_names:
        if time.time() - cached.fetched_at <= ROADS_TTL_SECONDS and cached.paths:
            return cached
        if cache_dir not in _refresh_tasks and time.monotonic() >= _refresh_retry_after.get(
            cache_dir, 0
        ):
            task = asyncio.create_task(_refresh_roads(client, cache_dir))
            _refresh_tasks[cache_dir] = task
            task.add_done_callback(lambda done: _finish_refresh(done, cache_dir))
        return cached
    if time.monotonic() < _refresh_retry_after.get(cache_dir, 0):
        return _fallback_roads()
    task = _refresh_tasks.get(cache_dir)
    if task is None:
        task = asyncio.create_task(_refresh_roads(client, cache_dir))
        _refresh_tasks[cache_dir] = task
        task.add_done_callback(lambda done: _finish_refresh(done, cache_dir))
    if not wait_for_refresh:
        return cached or _fallback_roads()
    try:
        return await asyncio.shield(task)
    except Exception as exc:  # noqa: BLE001
        _refresh_retry_after[cache_dir] = (
            time.monotonic() + ROADS_FAILURE_COOLDOWN_SECONDS
        )
        log.warning("tracked-roads initial refresh failed: %s", type(exc).__name__)
        if cached is not None and cached.display_names:
            return cached
        return _fallback_roads()


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
