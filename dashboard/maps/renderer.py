"""Projection and PIL rendering for two live directional route polylines."""

from __future__ import annotations

import io
import math
import os
from collections.abc import Iterable
from typing import NamedTuple

from PIL import Image, ImageDraw, ImageFont

from dashboard.maps.tiles import TILE_SIZE
from dashboard.models import Operator, RouteEtaGroup

MAP_WIDTH = 1920
MAP_HEIGHT = 1080
BASE_MAP_LAT = 22.3274138
BASE_MAP_LON = 114.2331738
BASE_MAP_ZOOM = 15.0
HKUST_LAT, HKUST_LON = 22.3364, 114.2656
STOP_MARKER_SIZE = 8
DIRECTION_OFFSET_PX = 3.5

CONGESTION_COLORS = {
    "green": (34, 197, 94),
    "amber": (245, 158, 11),
    "red": (239, 68, 68),
    "unknown": (107, 114, 128),
}
SPEED_COLORS = {
    "NORMAL": CONGESTION_COLORS["green"],
    "SLOW": CONGESTION_COLORS["amber"],
    "TRAFFIC_JAM": CONGESTION_COLORS["red"],
}
OPERATOR_COLORS = {
    Operator.KMB: (225, 29, 72),
    Operator.CITYBUS: (250, 204, 21),
    Operator.GMB: (22, 163, 74),
}
SHUTTLE_STOP_COLOR = (37, 99, 235)
PUBLIC_STOP_COLOR = (139, 92, 246)
GATE_PIN_COLOR = (37, 99, 235)

# Official TD KMB stop coordinates, not approximate campus-centre pins.
GATE_PINS = (
    ("HKUST (N)", 22.338678, 114.261946, True),
    ("HKUST (S)", 22.333360, 114.262881, False),
)

SHUTTLE_STOPS = (
    ("Diamond Hill", 22.3399, 114.2031),
    ("Tseung Kwan O", 22.3071, 114.2597),
    ("Hang Hau", 22.3170, 114.2670),
    ("Po Lam", 22.3229, 114.2585),
    ("Choi Hung", 22.3346, 114.2107),
    ("Kwun Tong", 22.3119, 114.2260),
)


class BusMarker(NamedTuple):
    routes: tuple[str, ...]
    x: float
    y: float
    operator: Operator
    heading: float


class LabelPlacement(NamedTuple):
    text: str
    rect: tuple[float, float, float, float]
    marker: tuple[float, float]
    operator: Operator
    heading: float


def mercator_y(lat: float) -> float:
    return (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2


def fit_view(
    points: list[tuple[float, float]] | None = None,
    size: tuple[int, int] = (MAP_WIDTH, MAP_HEIGHT),
) -> tuple[float, float, float]:
    return BASE_MAP_LAT, BASE_MAP_LON, BASE_MAP_ZOOM


def project(
    lat: float,
    lon: float,
    center_lat: float = BASE_MAP_LAT,
    center_lon: float = BASE_MAP_LON,
    zoom: float = BASE_MAP_ZOOM,
    size: tuple[int, int] = (MAP_WIDTH, MAP_HEIGHT),
) -> tuple[float, float]:
    scale = TILE_SIZE * 2**zoom
    x = (lon - center_lon) / 360 * scale + size[0] / 2
    y = (mercator_y(lat) - mercator_y(center_lat)) * scale + size[1] / 2
    return x, y


def congestion_color(delay_min: float | None) -> tuple[int, int, int]:
    if delay_min is None:
        return CONGESTION_COLORS["unknown"]
    if delay_min >= 5:
        return CONGESTION_COLORS["red"]
    if delay_min >= 2:
        return CONGESTION_COLORS["amber"]
    return CONGESTION_COLORS["green"]


def offset_polyline(points: list[tuple[float, float]], offset: float) -> list[tuple[float, float]]:
    """Offset a screen-space path to the left of its travel direction."""
    if len(points) < 2 or not offset:
        return points
    shifted: list[tuple[float, float]] = []
    for index, point in enumerate(points):
        before = points[max(0, index - 1)]
        after = points[min(len(points) - 1, index + 1)]
        dx, dy = after[0] - before[0], after[1] - before[1]
        length = math.hypot(dx, dy)
        shifted.append(
            point
            if not length
            else (point[0] - dy / length * offset, point[1] + dx / length * offset)
        )
    return shifted


def offset_where_overlapping(
    points: list[tuple[float, float]],
    other: list[tuple[float, float]],
    offset: float = DIRECTION_OFFSET_PX,
    threshold: float = 8.0,
) -> list[tuple[float, float]]:
    """Separate opposing lanes only where their road geometry overlaps."""
    if len(points) < 2 or len(other) < 2:
        return points

    def distance_to_segment(point, start, end) -> float:
        dx, dy = end[0] - start[0], end[1] - start[1]
        denominator = dx * dx + dy * dy
        ratio = (
            0.0
            if not denominator
            else max(
                0.0,
                min(
                    1.0,
                    ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / denominator,
                ),
            )
        )
        return math.hypot(
            point[0] - start[0] - ratio * dx,
            point[1] - start[1] - ratio * dy,
        )

    shifted = offset_polyline(points, offset)
    segments = list(zip(other, other[1:], strict=False))
    return [
        moved
        if min(distance_to_segment(point, start, end) for start, end in segments) <= threshold
        else point
        for point, moved in zip(points, shifted, strict=True)
    ]


def predict_buses(
    groups: list[RouteEtaGroup], route_lines: Iterable[object]
) -> list[tuple[str, float, float, Operator, float]]:
    """Estimate buses on their matching official HKeMobility route geometry.

    The ETA is only evidence of time to the target stop, not GPS.  We use a
    deliberately coarse two-minutes-per-stop pace, then interpolate by
    arclength on the official HKeMobility spatial line and official stop order.
    A line without validated road geometry produces no estimate.
    """
    buses: list[tuple[str, float, float, Operator, float]] = []
    lines = list(route_lines)
    operator_codes = {
        Operator.KMB: "KMB",
        Operator.CITYBUS: "CTB",
        Operator.GMB: "GMB",
    }
    gate_coords = {"N": (22.338678, 114.261946), "S": (22.333360, 114.262881)}
    for group in groups:
        target_gate = gate_coords.get(group.gate.upper())
        if target_gate is None:
            continue
        candidates: list[tuple[int, int, object]] = []
        destination_tokens = {
            token for token in group.destination.lower().split() if len(token) >= 3
        }
        for line in lines:
            if (
                str(getattr(line, "route", "")) != group.route
                or str(getattr(line, "operator", "")) != operator_codes[group.operator]
            ):
                continue
            stops = list(getattr(line, "stops", ()))
            path = list(getattr(line, "path", ()))
            stop_offsets = list(getattr(line, "stop_offsets", ()))
            if len(stops) < 3 or len(path) < 2 or len(stop_offsets) != len(stops):
                continue
            gate_distances = [
                math.hypot(
                    float(stops[index].lat) - target_gate[0],
                    float(stops[index].lon) - target_gate[1],
                )
                for index in range(len(stops))
            ]
            gate_distance = min(gate_distances)
            circular = (
                group.operator == Operator.GMB
                and group.route == "104"
                and str(stops[0].stop_id) == str(stops[-1].stop_id)
            )
            # The 104 ETA is for its next departure from the repeated HKUST
            # stop.  Use the final loop occurrence so walking backwards follows
            # the bus that is currently completing the circular trip.
            if circular:
                gate_index = max(
                    index
                    for index, distance in enumerate(gate_distances)
                    if math.isclose(distance, gate_distance, abs_tol=1e-8)
                )
            else:
                gate_index = gate_distances.index(gate_distance)
            # A one-way target at either terminus has no defensible upstream
            # placement.  The repeated final stop of circular 104 is the sole
            # exception; the predicted marker itself still stays off endpoints.
            if (
                gate_distance > 0.004
                or gate_index == 0
                or (gate_index == len(stops) - 1 and not circular)
            ):
                continue
            downstream_names = " ".join(stop.name.lower() for stop in stops[gate_index:])
            destination_score = sum(token in downstream_names for token in destination_tokens)
            candidates.append((destination_score, gate_index, line))
        if not candidates:
            continue
        _score, gate_index, line = max(candidates, key=lambda item: (item[0], item[1]))
        stops = list(line.stops)
        path = list(line.path)
        stop_offsets = list(line.stop_offsets)
        for row in group.rows:
            if row.minutes is None:
                continue
            # ETA is measured at the target stop.  Walk back through the
            # matching official sequence at roughly two minutes per stop, and linearly
            # interpolate only within that official geometry segment.
            position = gate_index - max(0, row.minutes) / 2
            if position <= 0 or position >= len(stops) - 1:
                continue
            index = math.floor(position)
            fraction = position - index
            target_offset = (
                stop_offsets[index] + (stop_offsets[index + 1] - stop_offsets[index]) * fraction
            )
            located = _point_at_path_offset(path, target_offset)
            if located is None:
                continue
            lat, lon, heading = located
            # Identify the vehicle's service direction at a glance.  This is
            # particularly important where the same route number has a loop.
            # GMB 104 is a circular service.  Stop 13 (Ngau Tau Kok Road,
            # outside Lotus Tower) is the approved turn point: before it the
            # vehicle is heading for Kwun Tong; from it onward it is returning
            # to HKUST.
            if group.operator == Operator.GMB and group.route == "104":
                terminus = "Kwun Tong" if position < 12 else "HKUST"
            else:
                # Keep marker wording consistent with the compact ETA embed.
                terminus = group.destination
            buses.append((f"{group.route} {terminus}", lat, lon, group.operator, heading))
    return buses


def _path_segment_length(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat_scale = 111_320.0
    lon_scale = lat_scale * math.cos(math.radians((a[0] + b[0]) / 2))
    return math.hypot((b[0] - a[0]) * lat_scale, (b[1] - a[1]) * lon_scale)


def _point_at_path_offset(
    path: list[tuple[float, float]], target: float
) -> tuple[float, float, float] | None:
    if len(path) < 2 or target < 0:
        return None
    travelled = 0.0
    for a, b in zip(path, path[1:], strict=False):
        length = _path_segment_length(a, b)
        if length and travelled + length >= target:
            fraction = (target - travelled) / length
            lat = a[0] + (b[0] - a[0]) * fraction
            lon = a[1] + (b[1] - a[1]) * fraction
            return lat, lon, math.atan2(b[0] - a[0], b[1] - a[1])
        travelled += length
    return None


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "Arial.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _background(
    cache_dir: str,
    center: tuple[float, float] = (BASE_MAP_LAT, BASE_MAP_LON),
    zoom: float = BASE_MAP_ZOOM,
    size: tuple[int, int] = (MAP_WIDTH, MAP_HEIGHT),
) -> Image.Image:
    cache_path = os.path.join(cache_dir, "gmaps_base.png")
    if os.path.exists(cache_path):
        try:
            return Image.open(cache_path).convert("RGB")
        except Exception:  # noqa: BLE001
            pass
    return Image.new("RGB", size, (240, 242, 245))


def _arrow(
    draw: ImageDraw.ImageDraw,
    point: tuple[float, float],
    angle: float,
    color: tuple[int, int, int, int],
) -> None:
    """Draw the original small, thin roadside arrowhead style."""
    tip = (point[0] + math.cos(angle) * 7, point[1] + math.sin(angle) * 7)
    back = (point[0] - math.cos(angle) * 6, point[1] - math.sin(angle) * 6)
    draw.line((back, tip), fill=color, width=2)
    for delta in (-0.65, 0.65):
        arm = (tip[0] - math.cos(angle + delta) * 6, tip[1] - math.sin(angle + delta) * 6)
        draw.line((tip, arm), fill=color, width=2)


def arrow_positions(
    points: list[tuple[float, float]], spacing: float = 180.0, phase: float = 0.0
) -> list[tuple[tuple[float, float], float]]:
    """Distance-based arrow placement, independent of vertex density."""
    lengths = [
        math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:], strict=False)
    ]
    total = sum(lengths)
    if total < 18:
        return []
    distances: list[float] = []
    position = 25 + phase
    while position < total - 25:
        distances.append(position)
        position += spacing
    if not distances:
        distances = [total / 2]
    out: list[tuple[tuple[float, float], float]] = []
    for target in distances:
        travelled = 0.0
        pairs = zip(points, points[1:], strict=False)
        for (a, b), length in zip(pairs, lengths, strict=True):
            if length and travelled + length >= target:
                ratio = (target - travelled) / length
                point = (a[0] + (b[0] - a[0]) * ratio, a[1] + (b[1] - a[1]) * ratio)
                out.append((point, math.atan2(b[1] - a[1], b[0] - a[0])))
                break
            travelled += length
    return out


def roadside_arrow_positions(
    points: list[tuple[float, float]], phase: float = 0.0
) -> list[tuple[tuple[float, float], float]]:
    """Place arrowheads eight pixels beside, rather than on, the route."""
    return [
        ((point[0] - math.sin(angle) * 8, point[1] + math.cos(angle) * 8), angle)
        for point, angle in arrow_positions(points, phase=phase)
    ]


def _draw_marker_on_left(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    heading: float,
    color: tuple[int, int, int],
    square: bool,
) -> tuple[float, float]:
    # Geographic heading uses +latitude as up.  After converting to screen
    # travel (dx, dy), its left normal is (dy, -dx).
    dx, dy = math.cos(heading), -math.sin(heading)
    x, y = x + dy * 7, y - dx * 7
    # PIL bounds are inclusive.  Integer bounds ending at start + size - 1
    # make the circle diameter and square side exactly STOP_MARKER_SIZE px.
    left = round(x - STOP_MARKER_SIZE / 2)
    top = round(y - STOP_MARKER_SIZE / 2)
    bounds = (left, top, left + STOP_MARKER_SIZE - 1, top + STOP_MARKER_SIZE - 1)
    if square:
        draw.rectangle(bounds, fill=color + (255,), outline=(255, 255, 255, 255), width=1)
    else:
        draw.ellipse(bounds, fill=color + (255,), outline=(255, 255, 255, 255), width=1)
    return x, y


def _nearest_road_heading(
    lat: float, lon: float, paths: Iterable[list[tuple[float, float]]]
) -> float:
    best = (float("inf"), 0.0)
    for path in paths:
        for a, b in zip(path, path[1:], strict=False):
            dy, dx = b[0] - a[0], b[1] - a[1]
            length_squared = dx * dx + dy * dy
            ratio = 0.0
            if length_squared:
                ratio = max(
                    0.0,
                    min(
                        1.0,
                        ((lon - a[1]) * dx + (lat - a[0]) * dy) / length_squared,
                    ),
                )
            nearest = (a[0] + ratio * dy, a[1] + ratio * dx)
            distance = (lat - nearest[0]) ** 2 + (lon - nearest[1]) ** 2
            if distance < best[0]:
                best = (distance, math.atan2(dy, dx))
    return best[1]


def _angular_distance(a: float, b: float) -> float:
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


def _stop_direction_headings(route_lines: Iterable[object]) -> dict[str, list[float]]:
    """Map stop IDs to deduplicated official route travel headings.

    A stop shared by eastbound and westbound sequences retains both headings;
    headings from overlapping routes within 20 degrees collapse to one marker.
    """
    headings: dict[str, list[float]] = {}
    for line in route_lines:
        stops = list(line.stops)
        path = list(getattr(line, "path", ()))
        offsets = list(getattr(line, "stop_offsets", ()))
        for index, stop in enumerate(stops):
            if len(stops) < 2:
                continue
            located = (
                _point_at_path_offset(path, max(0.0, offsets[index] - 0.5))
                if len(path) >= 2 and len(offsets) == len(stops)
                else None
            )
            if located is not None:
                heading = located[2]
            elif index == 0:
                before, after = stops[0], stops[1]
                heading = math.atan2(
                    float(after.lat) - float(before.lat), float(after.lon) - float(before.lon)
                )
            elif index == len(stops) - 1:
                before, after = stops[-2], stops[-1]
                heading = math.atan2(
                    float(after.lat) - float(before.lat), float(after.lon) - float(before.lon)
                )
            else:
                before, after = stops[index - 1], stops[index + 1]
                heading = math.atan2(
                    float(after.lat) - float(before.lat), float(after.lon) - float(before.lon)
                )
            values = headings.setdefault(str(stop.stop_id), [])
            if all(_angular_distance(heading, existing) >= math.radians(20) for existing in values):
                values.append(heading)
    return headings


def _normalized_stop_name(name: str) -> set[str]:
    import re

    aliases = {"rd": "road", "stn": "station", "ctr": "centre"}
    ignored = {"bus", "stop", "public", "transport", "interchange"}
    tokens = re.findall(r"[a-z0-9]+", name.lower())
    return {aliases.get(token, token) for token in tokens if token not in ignored}


def _stop_names_match(first: str, second: str) -> bool:
    a, b = _normalized_stop_name(first), _normalized_stop_name(second)
    if not a or not b:
        return False
    return a == b or (len(a & b) >= 2 and len(a & b) / min(len(a), len(b)) >= 0.75)


def _merged_public_stop_markers(
    public_stops: Iterable[object],
    route_lines: Iterable[object],
    route_paths: Iterable[list[tuple[float, float]]],
    center_lat: float = BASE_MAP_LAT,
    center_lon: float = BASE_MAP_LON,
    zoom: float = BASE_MAP_ZOOM,
    size: tuple[int, int] = (MAP_WIDTH, MAP_HEIGHT),
    location_tolerance: float = 16.0,
    angular_tolerance: float = math.radians(20),
) -> list[tuple[float, float, float]]:
    """Merge same-place/same-direction stops across operators.

    Official operators publish slightly different coordinates for the same
    physical stop.  The 16 px tolerance covers the observed 792M versus
    91/91M offsets while the heading check retains opposite road sides.
    """
    route_lines = list(route_lines)
    # Green minibuses may board/alight away from a fixed sign.  Preserve their
    # official stops for route geometry and ETA matching, but do not turn them
    # into public-stop glyphs.  A physically shared TD stop remains visible
    # when a KMB/Citybus route also serves it.
    fixed_stop_ids = {
        str(stop.stop_id)
        for line in route_lines
        if str(getattr(line, "operator", "")) != "GMB"
        for stop in line.stops
    }
    headings_by_id = _stop_direction_headings(route_lines)
    candidates: list[tuple[float, float, float, str, str]] = []
    paths = list(route_paths)
    for stop in public_stops:
        if str(stop.stop_id) not in fixed_stop_ids:
            continue
        lat, lon = float(stop.lat), float(stop.lon)
        x, y = project(lat, lon, center_lat, center_lon, zoom, size)
        headings = headings_by_id.get(str(stop.stop_id)) or [_nearest_road_heading(lat, lon, paths)]
        for heading in headings:
            candidates.append((x, y, heading, str(stop.stop_id), str(stop.name)))
    candidates.sort(key=lambda item: (round(item[1], 4), round(item[0], 4), item[3], item[2]))
    merged_named: list[tuple[float, float, float, str]] = []
    for x, y, heading, _stop_id, name in candidates:
        if any(
            math.hypot(x - old_x, y - old_y)
            <= (location_tolerance if _stop_names_match(name, old_name) else 6.0)
            and _angular_distance(heading, old_heading) <= angular_tolerance
            for old_x, old_y, old_heading, old_name in merged_named
        ):
            continue
        merged_named.append((x, y, heading, name))
    return [(x, y, heading) for x, y, heading, _name in merged_named]


def _rects_overlap(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
    padding: float = 2,
) -> bool:
    return not (
        first[2] + padding <= second[0]
        or second[2] + padding <= first[0]
        or first[3] + padding <= second[1]
        or second[3] + padding <= first[1]
    )


def _layout_bus_labels(
    markers: Iterable[BusMarker],
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    size: tuple[int, int],
) -> list[LabelPlacement]:
    """Lay out integrated route pills with their directional glyph on the road.

    The white triangle's centre is the interpolated road anchor.  The coloured
    pill is vertically centred on that point and grows from the triangle's
    leading compartment into the route text, making one compact marker.
    """
    # Several ETA rows can describe vehicles that are visually indistinguishable
    # at this zoom.  A centred pill cannot be displaced without breaking the
    # road-anchor contract, so coalesce a same-operator/same-direction convoy
    # into one route pill instead.  Keep the first actual road point as the
    # triangle anchor; do not invent an averaged off-road point.
    ordered_markers = sorted(
        markers, key=lambda marker: (marker.y, marker.x, marker.operator.value, marker.routes)
    )
    marker_list: list[BusMarker] = []
    for marker in ordered_markers:
        nearby = next(
            (
                existing
                for existing in reversed(marker_list)
                if existing.operator == marker.operator
                and _angular_distance(existing.heading, marker.heading) <= math.radians(12)
                and math.hypot(existing.x - marker.x, existing.y - marker.y) <= 18
            ),
            None,
        )
        if nearby is None:
            marker_list.append(marker)
            continue
        index = marker_list.index(nearby)
        marker_list[index] = nearby._replace(
            routes=tuple(sorted(set(nearby.routes) | set(marker.routes)))
        )
    placed: list[LabelPlacement] = []
    for marker in marker_list:
        text = "/".join(marker.routes)
        text_width = float(draw.textlength(text, font=font))
        # Reserve a leading compartment for the 12 px triangle and a text
        # gutter, so the triangle cannot be buried in the route number.
        pill_width, pill_height = text_width + 20, 18.0
        chosen = None
        # Prefer normal reading order: triangle then route text.  Only flip at
        # a map edge or an existing pill; either choice retains the triangle
        # on the road and vertically centred in its box.
        for text_side in (1, -1):
            if text_side > 0:
                left, right = marker.x - 7, marker.x + pill_width - 7
            else:
                left, right = marker.x - pill_width + 7, marker.x + 7
            rect = (left, marker.y - pill_height / 2, right, marker.y + pill_height / 2)
            if rect[0] < 2 or rect[1] < 2 or rect[2] > size[0] - 2 or rect[3] > size[1] - 2:
                continue
            covers_other_anchor = any(
                math.hypot(marker.x - other.x, marker.y - other.y) > 10
                and _rects_overlap(
                    rect,
                    (other.x - 5, other.y - 5, other.x + 5, other.y + 5),
                )
                for other in marker_list
                if other is not marker
            )
            conflicts = False
            for existing in placed:
                if not _rects_overlap(rect, existing.rect):
                    continue
                shares_anchor = (
                    math.hypot(
                        marker.x - existing.marker[0],
                        marker.y - existing.marker[1],
                    )
                    <= 10
                )
                candidate_side = math.copysign(1, (left + right) / 2 - marker.x)
                existing_side = math.copysign(
                    1,
                    (existing.rect[0] + existing.rect[2]) / 2 - existing.marker[0],
                )
                # Two operators at the same road point use opposite sides of
                # their shared arrow compartment. Suppressing both was the
                # source of apparently missing bus/minibus markers.
                if not shares_anchor or candidate_side == existing_side:
                    conflicts = True
                    break
            if covers_other_anchor or conflicts:
                continue
            chosen = rect
            break
        if chosen is not None:
            placed.append(
                LabelPlacement(text, chosen, (marker.x, marker.y), marker.operator, marker.heading)
            )
    return placed


def _draw_bus_route_marker(
    draw: ImageDraw.ImageDraw,
    placement: LabelPlacement,
    color: tuple[int, int, int],
    font: ImageFont.ImageFont,
) -> None:
    """Draw one coloured route pill with its white road-centred triangle."""
    left, top, right, bottom = placement.rect
    anchor_x, anchor_y = placement.marker
    # The anchor is the interpolated road position.  It is intentionally part
    # of the pill, so the visible white direction triangle cannot drift away
    # from the road when label rows are stacked.
    draw.rounded_rectangle(
        placement.rect, radius=4, fill=color + (245,), outline=(20, 20, 20, 255), width=1
    )
    draw.polygon(
        _bus_direction_arrow_triangle((anchor_x, anchor_y), placement.heading),
        fill=(255, 255, 255, 255),
    )
    # Text starts after the reserved triangle compartment.  On the rare
    # edge-flipped pill it grows cleanly to the triangle's left instead.
    text_x = anchor_x + 9 if anchor_x <= (left + right) / 2 else left + 4
    draw.text((text_x, top + 2), placement.text, fill=(255, 255, 255, 255), font=font)


def _bus_direction_arrow_triangle(
    center: tuple[float, float], heading: float
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Return a heading triangle whose geometric centre is the road anchor."""
    arrow_center = center
    arrow_dx, arrow_dy = math.cos(heading), -math.sin(heading)
    arrow_left = (-arrow_dy, arrow_dx)
    arrow_tip = (
        arrow_center[0] + arrow_dx * 6,
        arrow_center[1] + arrow_dy * 6,
    )
    return (
        arrow_tip,
        (
            arrow_center[0] - arrow_dx * 3 + arrow_left[0] * 3,
            arrow_center[1] - arrow_dy * 3 + arrow_left[1] * 3,
        ),
        (
            arrow_center[0] - arrow_dx * 3 - arrow_left[0] * 3,
            arrow_center[1] - arrow_dy * 3 - arrow_left[1] * 3,
        ),
    )


def _merge_bus_markers(
    predictions: Iterable[tuple[str, float, float, Operator, float]],
    center_lat: float,
    center_lon: float,
    zoom: float,
    size: tuple[int, int],
) -> list[BusMarker]:
    grouped: dict[tuple[Operator, str, int, int, int], tuple[set[str], float, float, float]] = {}
    projected = []
    for route, lat, lon, operator, heading in predictions:
        x, y = project(lat, lon, center_lat, center_lon, zoom, size)
        if not (0 <= x < size[0] and 0 <= y < size[1]):
            continue
        projected.append((operator.value, route, x, y, operator, heading))
    for _operator_name, route, x, y, operator, heading in sorted(projected):
        key = (
            operator,
            route,
            round(x / 4),
            round(y / 4),
            round(heading / math.radians(10)),
        )
        routes, *_rest = grouped.setdefault(key, (set(), x, y, heading))
        routes.add(route)
    return [
        BusMarker(tuple(sorted(routes)), x, y, operator, heading)
        for (operator, _route, _x, _y, _heading), (routes, x, y, heading) in sorted(
            grouped.items(),
            key=lambda item: (item[0][0].value, item[1][1], item[1][2], item[0][1:]),
        )
    ]


def _draw_gate_pins(
    draw: ImageDraw.ImageDraw,
    center_lat: float = BASE_MAP_LAT,
    center_lon: float = BASE_MAP_LON,
    zoom: float = BASE_MAP_ZOOM,
    size: tuple[int, int] = (MAP_WIDTH, MAP_HEIGHT),
) -> None:
    font = _font(13)
    for label, lat, lon, label_above in GATE_PINS:
        x, y = project(lat, lon, center_lat, center_lon, zoom, size)
        draw.ellipse(
            (x - 8, y - 8, x + 8, y + 8),
            fill=GATE_PIN_COLOR + (235,),
            outline=(255, 255, 255, 255),
            width=2,
        )
        text_width = draw.textlength(label, font=font)
        text_x = min(max(x + 11, 3), size[0] - text_width - 3)
        text_y = y - 19 if label_above else y + 9
        draw.rounded_rectangle(
            (text_x - 3, text_y - 2, text_x + text_width + 3, text_y + 16),
            radius=3,
            fill=(255, 255, 255, 215),
        )
        draw.text((text_x, text_y), label, fill=GATE_PIN_COLOR + (255,), font=font)


def _draw_legend(draw: ImageDraw.ImageDraw, size: tuple[int, int]) -> None:
    """Explain only dashboard-authored estimates and stop glyphs."""
    font = _font(12)
    x, y = 10, size[1] - 112
    draw.rounded_rectangle(
        (x - 6, y - 7, x + 455, y + 94),
        radius=7,
        fill=(255, 255, 255, 225),
        outline=(180, 180, 180, 235),
        width=1,
    )
    draw.text((x, y), "Estimated buses (not GPS)", fill=(30, 30, 30, 255), font=font)
    cursor_x = x + 158
    for operator, color in OPERATOR_COLORS.items():
        draw.rounded_rectangle(
            (cursor_x, y + 1, cursor_x + 12, y + 10),
            radius=2,
            fill=color + (255,),
            outline=(20, 20, 20, 255),
        )
        draw.text((cursor_x + 15, y - 2), operator.value, fill=(30, 30, 30, 255), font=font)
        cursor_x += 23 + draw.textlength(operator.value, font=font)

    row_y = y + 27
    draw.ellipse(
        (x, row_y, x + STOP_MARKER_SIZE - 1, row_y + STOP_MARKER_SIZE - 1),
        fill=SHUTTLE_STOP_COLOR + (255,),
        outline=(255, 255, 255, 255),
    )
    draw.text((x + 13, row_y - 3), "shuttle stop", fill=(30, 30, 30, 255), font=font)
    public_x = x + 112
    draw.rectangle(
        (public_x, row_y, public_x + STOP_MARKER_SIZE - 1, row_y + STOP_MARKER_SIZE - 1),
        fill=PUBLIC_STOP_COLOR + (255,),
        outline=(255, 255, 255, 255),
    )
    draw.text((public_x + 13, row_y - 3), "public bus stop", fill=(30, 30, 30, 255), font=font)
    draw.text(
        (x, y + 54),
        "Route geometry © Transport Department HKeMobility",
        fill=(60, 60, 60, 255),
        font=font,
    )


def render_map(
    groups: list[RouteEtaGroup],
    cache_dir: str,
    public_stops: Iterable[object] = (),
    route_lines: Iterable[object] = (),
    base_image: Image.Image | None = None,
) -> bytes:
    route_lines = list(route_lines)
    public_stops = list(public_stops)
    center_lat, center_lon, zoom = BASE_MAP_LAT, BASE_MAP_LON, BASE_MAP_ZOOM
    if base_image is not None:
        canvas = base_image.copy().convert("RGBA")
    else:
        canvas = _background(cache_dir, (center_lat, center_lon), zoom).convert("RGBA")
    draw = ImageDraw.Draw(canvas, "RGBA")
    size = canvas.size

    route_paths = [list(line.path) for line in route_lines if len(getattr(line, "path", ())) >= 2]

    # All stop glyphs use the exact same 8 px measure
    for _label, lat, lon in SHUTTLE_STOPS:
        x, y = project(lat, lon, center_lat, center_lon, zoom, size)
        _draw_marker_on_left(
            draw,
            x,
            y,
            _nearest_road_heading(lat, lon, route_paths),
            SHUTTLE_STOP_COLOR,
            square=False,
        )

    _draw_gate_pins(draw, center_lat, center_lon, zoom, size)

    for x, y, heading in _merged_public_stop_markers(
        public_stops, route_lines, route_paths, center_lat, center_lon, zoom, size
    ):
        _draw_marker_on_left(draw, x, y, heading, PUBLIC_STOP_COLOR, square=True)

    font = _font(13)
    bus_markers = _merge_bus_markers(
        predict_buses(groups, route_lines), center_lat, center_lon, zoom, size
    )
    for placement in _layout_bus_labels(bus_markers, draw, font, size):
        _draw_bus_route_marker(
            draw,
            placement,
            OPERATOR_COLORS.get(placement.operator, (100, 100, 100)),
            font,
        )

    _draw_legend(draw, size)

    draw.text(
        (size[0] - 150, size[1] - 22),
        "Map data © Google",
        fill=(60, 60, 60, 255),
        font=_font(12),
    )
    buffer = io.BytesIO()
    canvas.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()
