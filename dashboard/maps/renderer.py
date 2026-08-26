"""Projection and PIL rendering for two live directional route polylines."""

from __future__ import annotations

import io
import math
import os
from collections.abc import Iterable
from functools import lru_cache
from typing import NamedTuple

from PIL import Image, ImageDraw, ImageFont

from dashboard.maps.tiles import BASE_CACHE_FILENAME, TILE_SIZE
from dashboard.models import Operator

MAP_WIDTH = 960
MAP_HEIGHT = 540
LEGEND_WIDTH = 420
LEGEND_HEIGHT = 110
LEGEND_DISPLAY_WIDTH = 840
LEGEND_BAND_HEIGHT = 240
MIN_MAP_WIDTH = 720
MIN_MAP_HEIGHT = 405
BASE_MAP_LAT = 22.3274138
BASE_MAP_LON = 114.2331738
BASE_MAP_ZOOM = 14.0
STOP_MARKER_SIZE = 8

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
    unreliable: bool = False


class LabelPlacement(NamedTuple):
    text: str
    rect: tuple[float, float, float, float]
    marker: tuple[float, float]
    operator: Operator
    heading: float
    unreliable: bool = False


def mercator_y(lat: float) -> float:
    return (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2


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
    cache_path = os.path.join(cache_dir, BASE_CACHE_FILENAME)
    if os.path.exists(cache_path):
        try:
            image = Image.open(cache_path).convert("RGB")
            return image.resize(size, Image.Resampling.LANCZOS) if image.size != size else image
        except Exception:  # noqa: BLE001
            pass
    return Image.new("RGB", size, (240, 242, 245))


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
    placed_rects: Iterable[tuple[float, float, float, float]] = (),
) -> list[LabelPlacement]:
    """Lay out integrated route pills with their directional glyph on the road.

    The white triangle's centre is the interpolated road anchor.  The coloured
    pill is vertically centred on that point and grows from the triangle's
    leading compartment into the route text, making one compact marker.
    Colliding pills stack vertically above the road point instead of being
    dropped; ``placed_rects`` (e.g. legend bounds) are avoided too.
    """
    # Several ETA rows can describe vehicles that are visually indistinguishable
    # at this zoom.  A centred pill cannot be displaced without breaking the
    # road-anchor contract, so coalesce a same-operator/same-direction convoy
    # into one route pill instead.  Keep the first actual road point as the
    # triangle anchor; do not invent an averaged off-road point.
    ordered_markers = sorted(
        markers, key=lambda marker: (marker.y, marker.x, marker.operator.value, marker.routes)
    )
    # Same road point — or a neighbouring one within ~30 px, which reads as
    # the same junction at this zoom — forms ONE stacked label group. Pills
    # from different roads must never extend into each other.
    anchor_groups: list[list[BusMarker]] = []
    for marker in ordered_markers:
        group = next(
            (
                existing
                for existing in anchor_groups
                if math.hypot(existing[0].x - marker.x, existing[0].y - marker.y) <= 30
            ),
            None,
        )
        if group is None:
            anchor_groups.append([marker])
        else:
            group.append(marker)

    placed: list[LabelPlacement] = []
    occupied: list[tuple[float, float, float, float]] = list(placed_rects)
    pill_height = 18.0
    for group in anchor_groups:
        # Deterministic order within a stack: operator then routes.
        group = sorted(group, key=lambda m: (m.operator.value, m.routes))
        for index, marker in enumerate(group):
            text = "/".join(marker.routes)
            pill_width = float(draw.textlength(text, font=font)) + 20
            chosen = None
            pointer = None
            # Candidate rows: the road row first (pill vertically centred on
            # the anchor, white triangle exactly at the road point), then a
            # tidy upward stack. Every candidate keeps the FULL pill size —
            # never squashed — so boxes stay readable at map edges.
            candidate_offsets: list[float] = [0.0]
            while len(candidate_offsets) < index + 1:
                candidate_offsets.append(
                    candidate_offsets[-1] - (pill_height + 4)
                )
            # If the upward stack runs out of room, keep searching DOWNWARD
            # below the road point before falling back to an overlap.
            extra = 0
            while len(candidate_offsets) < len(group) * 2:
                extra += 1
                candidate_offsets.append(extra * (pill_height + 4))
            for offset_y in candidate_offsets:
                top = marker.y + offset_y - pill_height / 2
                bottom = top + pill_height
                for text_side in (1, -1):
                    if text_side > 0:
                        left, right = marker.x - 7, marker.x + pill_width - 7
                    else:
                        left, right = marker.x - pill_width + 7, marker.x + 7
                    rect = (left, top, right, bottom)
                    if (
                        rect[0] < 2
                        or rect[1] < 2
                        or rect[2] > size[0] - 2
                        or rect[3] > size[1] - 2
                    ):
                        continue
                    if any(_rects_overlap(rect, other) for other in occupied):
                        continue
                    chosen = rect
                    # The white pointer rides INSIDE its pill: exactly on the
                    # road point for the road row; centred in stacked pills so
                    # direction stays interpretable above the road.
                    pointer = (
                        (marker.x, marker.y)
                        if offset_y == 0.0 and abs((top + bottom) / 2 - marker.y) < 1
                        else ((left + right) / 2 - text_side * (pill_width / 2 - 9), top + pill_height / 2)
                    )
                    break
                if chosen is not None:
                    break
            if chosen is None:
                # No clean slot in the stack columns: spiral outward over
                # progressively larger offsets (up, down, left, right) until
                # a free full-size slot exists. A pill NEVER ends up on top
                # of another one.
                chosen = None
                for radius in range(1, 40):
                    for dx, dy in (
                        (-radius * 6, 0),
                        (radius * 6, 0),
                        (0, radius * (pill_height + 4)),
                        (0, -radius * (pill_height + 4)),
                        (-radius * 6, -radius * (pill_height + 4)),
                        (radius * 6, -radius * (pill_height + 4)),
                        (-radius * 6, radius * (pill_height + 4)),
                        (radius * 6, radius * (pill_height + 4)),
                    ):
                        left = marker.x + dx - 7
                        right = left + pill_width
                        top = marker.y + dy - pill_height / 2
                        bottom = top + pill_height
                        rect = (left, top, right, bottom)
                        if (
                            rect[0] < 2
                            or rect[1] < 2
                            or rect[2] > size[0] - 2
                            or rect[3] > size[1] - 2
                        ):
                            continue
                        if any(_rects_overlap(rect, other) for other in occupied):
                            continue
                        chosen = rect
                        pointer = ((left + right) / 2 - (pill_width / 2 - 9), top + pill_height / 2)
                        break
                    if chosen is not None:
                        break
            if chosen is None:
                # Truly nowhere free (map wall-to-wall buses): keep the
                # full-size pill on the road row, clipped into the canvas —
                # never squashed.
                left = min(max(2.0, marker.x - 7), size[0] - pill_width - 2)
                top = min(
                    max(2.0, marker.y - pill_height / 2), size[1] - pill_height - 2
                )
                chosen = (left, top, left + pill_width, top + pill_height)
                pointer = (marker.x, marker.y)
            occupied.append(chosen)
            placed.append(
                LabelPlacement(
                    text,
                    chosen,
                    pointer,
                    marker.operator,
                    marker.heading,
                    marker.unreliable,
                )
            )
    return placed


def _dashed_rounded_rectangle(
    draw: ImageDraw.ImageDraw,
    rect: tuple[float, float, float, float],
    radius: int,
    fill: tuple[int, int, int, int],
    outline: tuple[int, int, int, int],
    dash: int = 4,
    gap: int = 3,
) -> None:
    """Rounded rectangle with a dashed outline (PIL lacks dash support)."""
    left, top, right, bottom = rect
    draw.rounded_rectangle(rect, radius=radius, fill=fill)

    def dashed_line(a, b):
        length = math.hypot(b[0] - a[0], b[1] - a[1])
        if length <= 0:
            return
        ux, uy = (b[0] - a[0]) / length, (b[1] - a[1]) / length
        travelled = 0.0
        while travelled < length:
            end = min(travelled + dash, length)
            draw.line(
                (
                    (a[0] + ux * travelled, a[1] + uy * travelled),
                    (a[0] + ux * end, a[1] + uy * end),
                ),
                fill=outline,
                width=1,
            )
            travelled = end + gap

    # Straight segments only (corner arcs stay undashed; visually equivalent).
    dashed_line((left + radius, top), (right - radius, top))
    dashed_line((left + radius, bottom), (right - radius, bottom))
    dashed_line((left, top + radius), (left, bottom - radius))
    dashed_line((right, top + radius), (right, bottom - radius))


def _draw_bus_route_marker(
    draw: ImageDraw.ImageDraw,
    placement: LabelPlacement,
    color: tuple[int, int, int],
    font: ImageFont.ImageFont,
    unreliable: bool = False,
) -> None:
    """Draw one coloured route pill with its white pointer inside it."""
    left, top, right, bottom = placement.rect
    anchor_x, anchor_y = placement.marker
    if unreliable:
        # Timetable-derived estimate: paler fill, dashed outline, dim text.
        pale = tuple(int(c + (255 - c) * 0.55) for c in color)
        draw.rounded_rectangle(placement.rect, radius=4, fill=pale + (200,))
        _dashed_rounded_rectangle(
            draw,
            placement.rect,
            radius=4,
            fill=pale + (0,),
            outline=(90, 90, 90, 235),
        )
        text_fill = (235, 235, 235, 255)
    else:
        draw.rounded_rectangle(
            placement.rect, radius=4, fill=color + (245,), outline=(20, 20, 20, 255), width=1
        )
        text_fill = (255, 255, 255, 255)
    # The white direction triangle lives INSIDE the pill (road point when the
    # pill sits on the road row; pill-centred when stacked above it), so the
    # box and its pointer are always one interpretable unit.
    draw.polygon(
        _bus_direction_arrow_triangle((anchor_x, anchor_y), placement.heading),
        fill=(255, 255, 255, 255),
    )
    text_x = (
        anchor_x + 9
        if anchor_x <= (left + right) / 2
        else left + 4
    )
    draw.text((text_x, top + 2), placement.text, fill=text_fill, font=font)


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
    estimates: Iterable[object],
    center_lat: float,
    center_lon: float,
    zoom: float,
    size: tuple[int, int],
) -> list[BusMarker]:
    grouped: dict[
        tuple[Operator, str, int, int, int, bool],
        tuple[set[str], float, float, float],
    ] = {}
    projected = []
    for estimate in estimates:
        x, y = project(estimate.lat, estimate.lon, center_lat, center_lon, zoom, size)
        if not (0 <= x < size[0] and 0 <= y < size[1]):
            continue
        # Group by the full label so same-route opposite-destination markers
        # (e.g. 91 Diamond Hill vs 91 Clear Water Bay) stay distinct.
        projected.append(
            (
                estimate.operator.value,
                estimate.label,
                x,
                y,
                estimate.operator,
                estimate.heading,
                bool(getattr(estimate, "unreliable", False)),
            )
        )
    for row in sorted(projected):
        _operator_name, label, x, y, operator, heading, unreliable = row
        key = (
            operator,
            label,
            round(x / 4),
            round(y / 4),
            round(heading / math.radians(10)),
            unreliable,
        )
        routes, *_rest = grouped.setdefault(key, (set(), x, y, heading))
        routes.add(label)
    return [
        BusMarker(tuple(sorted(routes)), x, y, operator, heading, unreliable)
        for (operator, _route, _x, _y, _heading, unreliable), (routes, x, y, heading)
        in sorted(
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
    x, y = 10, max(8, size[1] - 102)
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
    # Unreliable (timetable-derived) marker swatch: pale with dashed outline.
    pale_kmb = tuple(int(c + (255 - c) * 0.55) for c in OPERATOR_COLORS[Operator.KMB])
    unreliable_x = x + 240
    _dashed_rounded_rectangle(
        draw,
        (unreliable_x, row_y, unreliable_x + 26, row_y + 13),
        radius=3,
        fill=pale_kmb + (200,),
        outline=(90, 90, 90, 235),
    )
    draw.text(
        (unreliable_x + 31, row_y - 3),
        "timetable only",
        fill=(30, 30, 30, 255),
        font=font,
    )

    # Match the map's high-contrast, no-fill rectangle indicator.
    draw.rectangle(
        (x, y + 54, x + 20, y + 64),
        outline=ALERT_RECT_COLOR,
        width=4,
    )
    draw.text(
        (x + 27, y + 53),
        "traffic-news segment",
        fill=(30, 30, 30, 255),
        font=font,
    )
    attribution = "Map data © Google · Route geometry © Transport Department HKeMobility"
    draw.text(
        (x, size[1] - 16),
        attribution,
        fill=(105, 105, 105, 255),
        font=_font(8),
    )


@lru_cache(maxsize=1)
def render_legend() -> bytes:
    """Return the deterministic standalone legend PNG used beside the map."""
    canvas = Image.new("RGB", (LEGEND_WIDTH, LEGEND_HEIGHT), (246, 247, 249))
    _draw_legend(ImageDraw.Draw(canvas, "RGBA"), canvas.size)
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG", optimize=False)
    return buffer.getvalue()


def _append_legend_band(canvas: Image.Image) -> Image.Image:
    """Append a readable, opaque legend band below the untouched map image."""
    map_image = canvas.convert("RGB")
    legend = Image.open(io.BytesIO(render_legend())).convert("RGB")
    legend_height = round(LEGEND_DISPLAY_WIDTH * legend.height / legend.width)
    legend = legend.resize((LEGEND_DISPLAY_WIDTH, legend_height), Image.Resampling.LANCZOS)
    composite = Image.new(
        "RGB", (map_image.width, map_image.height + LEGEND_BAND_HEIGHT), (246, 247, 249)
    )
    composite.paste(map_image, (0, 0))
    composite.paste(
        legend,
        ((map_image.width - LEGEND_DISPLAY_WIDTH) // 2, map_image.height + 20),
    )
    return composite


ALERT_RECT_COLOR = (180, 0, 255, 255)
ALERT_RECT_PADDING = 7
ALERT_RECT_OUTLINE_WIDTH = 4


def _draw_alerted_road_rectangles(
    canvas: Image.Image,
    affected_road_paths: Iterable[Iterable[tuple[float, float]]],
    center_lat: float,
    center_lon: float,
    zoom: float,
    size: tuple[int, int],
) -> int:
    """Draw merged, transparent rectangles around TD-affected road sections.

    Rectangles are padded in screen space, merged transitively when they touch,
    and clipped to the map. The transparent centre preserves Google's traffic
    layer. Drawn beneath markers so pills and stops stay readable.
    """
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay, "RGBA")
    rectangles: list[tuple[float, float, float, float]] = []
    for raw_path in affected_road_paths:
        path = list(raw_path)
        if len(path) < 2:
            continue
        points = [project(lat, lon, center_lat, center_lon, zoom, size) for lat, lon in path]
        if len(points) < 2:
            continue
        left = max(0.0, min(point[0] for point in points) - ALERT_RECT_PADDING)
        top = max(0.0, min(point[1] for point in points) - ALERT_RECT_PADDING)
        right = min(float(size[0] - 1), max(point[0] for point in points) + ALERT_RECT_PADDING)
        bottom = min(float(size[1] - 1), max(point[1] for point in points) + ALERT_RECT_PADDING)
        if left <= right and top <= bottom:
            rectangles.append((left, top, right, bottom))

    # Merge transitively: expanding a merged rectangle can make it touch a
    # later rectangle, so keep scanning until no merge remains.
    merged: list[tuple[float, float, float, float]] = []
    for rectangle in rectangles:
        candidate = rectangle
        changed = True
        while changed:
            changed = False
            remaining: list[tuple[float, float, float, float]] = []
            for other in merged:
                if _rects_overlap(candidate, other, padding=2):
                    candidate = (
                        min(candidate[0], other[0]),
                        min(candidate[1], other[1]),
                        max(candidate[2], other[2]),
                        max(candidate[3], other[3]),
                    )
                    changed = True
                else:
                    remaining.append(other)
            merged = remaining
        merged.append(candidate)

    for left, top, right, bottom in merged:
        overlay_draw.rectangle(
            (left, top, right, bottom),
            outline=ALERT_RECT_COLOR,
            width=ALERT_RECT_OUTLINE_WIDTH,
        )
    canvas.alpha_composite(overlay)
    return len(merged)


def render_map(
    estimates: list,
    cache_dir: str,
    public_stops: Iterable[object] = (),
    route_lines: Iterable[object] = (),
    base_image: Image.Image | None = None,
    affected_road_paths: Iterable[Iterable[tuple[float, float]]] = (),
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

    # Traffic-news road indicators go beneath everything dashboard-drawn.
    _draw_alerted_road_rectangles(
        canvas,
        affected_road_paths,
        center_lat,
        center_lon,
        zoom,
        size,
    )

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
    bus_markers = _merge_bus_markers(estimates, center_lat, center_lon, zoom, size)
    for placement in _layout_bus_labels(bus_markers, draw, font, size):
        _draw_bus_route_marker(
            draw,
            placement,
            OPERATOR_COLORS.get(placement.operator, (100, 100, 100)),
            font,
            unreliable=placement.unreliable,
        )

    buffer = io.BytesIO()
    # Discord/mobile payload target: retain full dimensions while adapting
    # quality. RGB WebP keeps traffic colours (no palette quantization).
    image = _append_legend_band(canvas)
    minimum_height = round(image.height * MIN_MAP_WIDTH / image.width)
    # Keep a readable mobile floor while reducing dimensions only as needed.
    while True:
        for quality in (82, 78, 74, 70, 65, 60):
            buffer.seek(0)
            buffer.truncate(0)
            image.save(buffer, format="WEBP", quality=quality, method=6)
            if buffer.tell() <= 100_000:
                return buffer.getvalue()
        if image.size == (MIN_MAP_WIDTH, minimum_height):
            raise ValueError("map WebP exceeds 100 KB at readable minimum dimensions")
        next_width = max(MIN_MAP_WIDTH, round(image.width * 0.9))
        next_height = round(next_width * image.height / image.width)
        if (next_width, next_height) == image.size:
            next_width, next_height = MIN_MAP_WIDTH, minimum_height
        image = image.resize((next_width, next_height), Image.Resampling.LANCZOS)
