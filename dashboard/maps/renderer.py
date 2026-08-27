"""Projection and PIL rendering for two live directional route polylines."""

from __future__ import annotations

import io
import math
import os
from array import array
from collections.abc import Iterable
from functools import lru_cache
from typing import NamedTuple

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

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
# Solid traffic-stroke cores sampled from the cached 2026-08-28 Google frame.
# The third entry is the captured amber transition used for the orange key.
GOOGLE_TRAFFIC_COLORS = (
    (22, 224, 152),
    (255, 224, 104),
    (250, 198, 49),
    (247, 74, 85),
    (169, 39, 39),
)
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
    rows: tuple[tuple[str, Operator, bool], ...] = ()


class RenderMetrics(NamedTuple):
    """Immutable authored-pixel scale for a native-resolution render."""

    scale: float

    def px(self, value: float, minimum: float = 0.0) -> float:
        return max(minimum, value * self.scale)

    def integer(self, value: float, minimum: int = 1) -> int:
        return max(minimum, round(value * self.scale))

    def font_size(self, value: float, minimum: int = 7) -> int:
        return self.integer(value, minimum)


DEFAULT_METRICS = RenderMetrics(1.0)


class _OversizedMapError(ValueError):
    """A native render candidate cannot meet Discord's payload limit."""


class TrafficOccupancy:
    """Integral mask of saturated Google traffic-layer road pixels."""

    __slots__ = ("height", "integral", "snap_integral", "stride", "width")

    def __init__(self, mask: Image.Image, snap_mask: Image.Image | None = None) -> None:
        self.width, self.height = mask.size
        self.stride = self.width + 1
        self.integral = self._build_integral(mask)
        self.snap_integral = self._build_integral(snap_mask if snap_mask is not None else mask)

    def _build_integral(self, mask: Image.Image) -> array:
        integral = array("I", [0]) * ((self.width + 1) * (self.height + 1))
        pixels = mask.load()
        for y in range(1, self.height + 1):
            row_sum = 0
            row = y * self.stride
            previous = (y - 1) * self.stride
            for x in range(1, self.width + 1):
                row_sum += 1 if pixels[x - 1, y - 1] else 0
                integral[row + x] = integral[previous + x] + row_sum
        return integral

    def _overlap(
        self, rect: tuple[float, float, float, float], integral: array
    ) -> tuple[int, int]:
        left = max(0, min(self.width, math.floor(rect[0])))
        top = max(0, min(self.height, math.floor(rect[1])))
        right = max(left, min(self.width, math.ceil(rect[2])))
        bottom = max(top, min(self.height, math.ceil(rect[3])))
        area = (right - left) * (bottom - top)
        if not area:
            return 0, 0
        stride = self.stride
        count = (
            integral[bottom * stride + right]
            - integral[top * stride + right]
            - integral[bottom * stride + left]
            + integral[top * stride + left]
        )
        return count, area

    def overlap(self, rect: tuple[float, float, float, float]) -> tuple[int, int]:
        """Return broad traffic pixels and clipped area covered by ``rect``."""
        return self._overlap(rect, self.integral)

    def _snap_overlap(self, rect: tuple[float, float, float, float]) -> tuple[int, int]:
        return self._overlap(rect, self.snap_integral)

    def snap_anchor(
        self,
        anchor: tuple[float, float],
        heading: float,
        metrics: RenderMetrics = DEFAULT_METRICS,
    ) -> tuple[float, float]:
        """Snap perpendicularly to a credible, route-aligned traffic band.

        Route progress and heading remain authoritative. Hong Kong's left-side
        traffic is encoded by positive offsets along screen travel's left
        normal ``(dy, -dx)``. Density and continuous along-heading support can
        outweigh side preference, preventing a small icon from beating a road.
        """
        dx, dy = math.cos(heading), -math.sin(heading)
        nx, ny = dy, -dx
        radius = metrics.px(10)
        probe_radius = metrics.px(1.15, minimum=0.75)
        along_offsets = tuple(item * metrics.scale for item in (-8, -4, 0, 4, 8))
        samples: list[tuple[int, int]] = []
        for offset in range(-math.ceil(radius), math.ceil(radius) + 1):
            if abs(offset) > radius:
                continue
            support = []
            for along in along_offsets:
                x = anchor[0] + nx * offset + dx * along
                y = anchor[1] + ny * offset + dy * along
                count, _area = self._snap_overlap(
                    (x - probe_radius, y - probe_radius,
                     x + probe_radius, y + probe_radius)
                )
                support.append(count)
            # A route-aligned traffic stroke persists over the full 16
            # logical-pixel span. Isolated circular POIs and short labels do not.
            if all(count > 0 for count in support):
                samples.append((offset, sum(support)))
        if not samples:
            return anchor

        bands: list[list[tuple[int, int]]] = []
        for sample in samples:
            if not bands or sample[0] > bands[-1][-1][0] + 1:
                bands.append([sample])
            else:
                bands[-1].append(sample)
        def band_details(band: list[tuple[int, int]]) -> tuple[float, float]:
            total = sum(count for _offset, count in band)
            center = sum(offset * count for offset, count in band) / total
            # Width/density dominate a modest left-side bonus. A strong road
            # band can therefore beat a weak saturated speck on the left.
            quality = total / len(band) + len(band) * 2.0
            side_bonus = 8.0 if center > 0 else (2.0 if abs(center) <= 1 else 0.0)
            score = quality + side_bonus - abs(center) * 0.35
            return score, center

        scored = [(band_details(band), band) for band in bands]
        (_quality, center), band = max(
            scored,
            key=lambda item: (item[0][0], item[0][1] > 0, -abs(item[0][1])),
        )
        total = sum(count for _offset, count in band)
        offset = sum(offset * count for offset, count in band) / total
        return anchor[0] + nx * offset, anchor[1] + ny * offset


def _traffic_occupancy(
    pristine_base: Image.Image, metrics: RenderMetrics = DEFAULT_METRICS
) -> TrafficOccupancy:
    """Identify Google traffic strokes without classifying pale map greenery.

    Google traffic roads are strongly saturated red, amber/yellow, or green.
    Requiring both saturation and value excludes the low-saturation mint land
    fill while retaining compressed/anti-aliased traffic cores. A one-logical-
    pixel dilation keeps thin strokes meaningful in lower-resolution retries.
    """
    rgb = pristine_base.convert("RGB")
    hue, saturation, value = rgb.convert("HSV").split()
    hue_mask = hue.point(
        [255 if item <= 48 or item >= 245 or 50 <= item <= 112 else 0 for item in range(256)]
    )
    saturation_mask = saturation.point([255 if item >= 105 else 0 for item in range(256)])
    value_mask = value.point([255 if item >= 85 else 0 for item in range(256)])
    mask = ImageChops.multiply(ImageChops.multiply(hue_mask, saturation_mask), value_mask)
    red, green, blue = rgb.split()
    snap_mask = Image.new("L", rgb.size, 0)
    tolerance = 38
    for palette_red, palette_green, palette_blue in GOOGLE_TRAFFIC_COLORS:
        red_match = red.point(
            [255 if abs(item - palette_red) <= tolerance else 0 for item in range(256)]
        )
        green_match = green.point(
            [255 if abs(item - palette_green) <= tolerance else 0 for item in range(256)]
        )
        blue_match = blue.point(
            [255 if abs(item - palette_blue) <= tolerance else 0 for item in range(256)]
        )
        matched = ImageChops.multiply(ImageChops.multiply(red_match, green_match), blue_match)
        snap_mask = ImageChops.lighter(snap_mask, matched)
    dilation = metrics.integer(1)
    if dilation:
        mask = mask.filter(ImageFilter.MaxFilter(dilation * 2 + 1))
        snap_mask = snap_mask.filter(ImageFilter.MaxFilter(dilation * 2 + 1))
    return TrafficOccupancy(mask, snap_mask)


def _render_metrics(size: tuple[int, int]) -> RenderMetrics:
    return RenderMetrics(size[0] / MAP_WIDTH)


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
    metrics: RenderMetrics = DEFAULT_METRICS,
) -> tuple[float, float]:
    # Geographic heading uses +latitude as up.  After converting to screen
    # travel (dx, dy), its left normal is (dy, -dx).
    dx, dy = math.cos(heading), -math.sin(heading)
    x, y = x + dy * metrics.px(7), y - dx * metrics.px(7)
    # PIL bounds are inclusive.  Integer bounds ending at start + size - 1
    # make the circle diameter and square side exactly STOP_MARKER_SIZE px.
    marker_size = metrics.integer(STOP_MARKER_SIZE, minimum=4)
    left = round(x - marker_size / 2)
    top = round(y - marker_size / 2)
    bounds = (left, top, left + marker_size - 1, top + marker_size - 1)
    outline_width = metrics.integer(1)
    if square:
        draw.rectangle(
            bounds, fill=color + (255,), outline=(255, 255, 255, 255),
            width=outline_width,
        )
    else:
        draw.ellipse(
            bounds, fill=color + (255,), outline=(255, 255, 255, 255),
            width=outline_width,
        )
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
    metrics: RenderMetrics = DEFAULT_METRICS,
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
            <= metrics.px(location_tolerance if _stop_names_match(name, old_name) else 6.0)
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
    metrics: RenderMetrics = DEFAULT_METRICS,
    traffic: TrafficOccupancy | None = None,
) -> list[LabelPlacement]:
    """Lay out independent route labels and their anchored directional arrows.

    The operator-colored arrow stays tied to the immutable road anchor.
    Labels are independently displaced with a clear gap and avoid obstacles.
    """
    # Several ETA rows can describe vehicles at one visual anchor. Coalesce
    # compatible cross-operator/same-direction anchors into one multi-row box,
    # while keeping the first actual road point as the arrow anchor.
    ordered_markers = sorted(
        markers, key=lambda marker: (marker.y, marker.x, marker.operator.value, marker.routes)
    )
    # Only genuinely overlapping anchors form one multi-row label group;
    # nearby but distinct vehicles remain independent.
    anchor_groups: list[list[BusMarker]] = []
    for marker in ordered_markers:
        group = next(
            (
                existing
                for existing in anchor_groups
                if (
                    math.hypot(existing[0].x - marker.x, existing[0].y - marker.y)
                    <= metrics.px(8)
                    and _angular_distance(existing[0].heading, marker.heading)
                    <= math.radians(28)
                )
            ),
            None,
        )
        if group is None:
            anchor_groups.append([marker])
        else:
            group.append(marker)

    # Keep opposite carriageways visually legible when their geographic
    # anchors coincide.  This is a display-only nudge; source coordinates and
    # travel headings remain unchanged.  Apply it before label placement so
    # the arrow and its label use the same deterministic visual anchor.
    adjusted_groups: set[int] = set()
    for left_index, left_group in enumerate(anchor_groups):
        for right_group in anchor_groups[left_index + 1 :]:
            first, second = left_group[0], right_group[0]
            if (
                math.hypot(first.x - second.x, first.y - second.y) <= metrics.px(8)
                and _angular_distance(first.heading, second.heading) >= math.radians(120)
            ):
                first_dx, first_dy = math.cos(first.heading), -math.sin(first.heading)
                second_dx, second_dy = math.cos(second.heading), -math.sin(second.heading)
                # Nudge each heading independently towards its own left-side
                # carriageway; never infer the second side by simple negation.
                first_nx, first_ny = first_dy, -first_dx
                second_nx, second_ny = second_dy, -second_dx
                offset = metrics.px(2.0)
                right_index = left_index + 1 + anchor_groups[left_index + 1 :].index(right_group)
                if left_index in adjusted_groups or right_index in adjusted_groups:
                    continue
                left_group[:] = [
                    m._replace(x=m.x + first_nx * offset, y=m.y + first_ny * offset)
                    for m in left_group
                ]
                right_group[:] = [
                    m._replace(x=m.x + second_nx * offset, y=m.y + second_ny * offset)
                    for m in right_group
                ]
                adjusted_groups.update((left_index, right_index))

    placed: list[LabelPlacement] = []
    occupied: list[tuple[float, float, float, float]] = list(placed_rects)
    arrow_footprints = [
        _arrow_footprint((group[0].x, group[0].y), metrics) for group in anchor_groups
    ]
    pill_height = metrics.px(18.0)
    edge_margin = metrics.px(2.0)
    label_gap = metrics.px(10.0)
    row_gap = metrics.px(4.0)
    text_padding = metrics.px(8.0)
    spiral_step = metrics.px(6.0)
    collision_padding = metrics.px(2.0)

    def score(
        rect: tuple[float, float, float, float],
        displacement: float,
        side: int,
    ) -> tuple[float, float, int, float]:
        """Prefer clear labels nearby, but cap how far traffic can push one."""
        traffic_penalty = 0.0
        if traffic is not None:
            overlap, area = traffic.overlap(rect)
            if overlap and area:
                # Any road overlap is meaningful; density adds only a bounded
                # increment. At most two logical label rows of displacement
                # can be justified, so unavoidable traffic stays local.
                traffic_penalty = metrics.px(24) + metrics.px(20) * overlap / area
        return displacement + traffic_penalty, traffic_penalty, side, rect[0]
    for group in anchor_groups:
        # Deterministic order within a stack: operator then routes.
        group = sorted(group, key=lambda m: (m.operator.value, m.routes))
        if len(group) > 1:
            anchor = group[0]
            rows = tuple(
                ("/".join(marker.routes), marker.operator, marker.unreliable)
                for marker in group
            )
            row_height = pill_height
            cluster_height = row_height * len(rows)
            cluster_width = max(
                float(draw.textlength(text, font=font)) + text_padding
                for text, _operator, _unreliable in rows
            )
            chosen = None
            # Anchor first, then symmetric escape candidates.  The road arrow
            # always remains at the immutable anchor even when the label moves.
            cluster_candidates: list[
                tuple[tuple[float, float, float, float], tuple[float, float, int, float]]
            ] = []
            for offset_y in (
                0.0,
                -(cluster_height + row_gap), cluster_height + row_gap,
                -2 * (cluster_height + row_gap), 2 * (cluster_height + row_gap),
            ):
                top = anchor.y + offset_y - cluster_height / 2
                for side in (1, -1):
                    left = (
                        anchor.x + label_gap
                        if side > 0 else anchor.x - label_gap - cluster_width
                    )
                    candidate = (left, top, left + cluster_width, top + cluster_height)
                    if (
                        candidate[0] < edge_margin or candidate[1] < edge_margin
                        or candidate[2] > size[0] - edge_margin
                        or candidate[3] > size[1] - edge_margin
                        or any(_rects_overlap(candidate, footprint, padding=0) for footprint in arrow_footprints)
                        or any(
                            _rects_overlap(candidate, other, padding=collision_padding)
                            for other in occupied
                        )
                    ):
                        continue
                    cluster_candidates.append(
                        (candidate, score(candidate, abs(offset_y), 0 if side < 0 else 1))
                    )
            if cluster_candidates:
                chosen, _score = min(cluster_candidates, key=lambda item: item[1])
            if chosen is None:
                # Bounded spiral search handles a busy junction without
                # clamping a group onto another occupied label.
                spiral_candidates = []
                for radius in range(1, 40):
                    for dx, dy in (
                        (-radius * spiral_step, 0), (radius * spiral_step, 0),
                        (0, -radius * (cluster_height + row_gap)),
                        (0, radius * (cluster_height + row_gap)),
                    ):
                        left = anchor.x + dx + (
                            label_gap if dx >= 0 else -label_gap - cluster_width
                        )
                        top = anchor.y + dy - cluster_height / 2
                        candidate = (left, top, left + cluster_width, top + cluster_height)
                        if (candidate[0] >= edge_margin and candidate[1] >= edge_margin
                                and candidate[2] <= size[0] - edge_margin
                                and candidate[3] <= size[1] - edge_margin
                                and not any(
                                    _rects_overlap(candidate, other, padding=collision_padding)
                                    for other in occupied
                                )
                                and not any(_rects_overlap(candidate, footprint, padding=0) for footprint in arrow_footprints)):
                            spiral_candidates.append(
                                (candidate, score(
                                    candidate,
                                    math.hypot(dx, dy),
                                    0 if dx < 0 else 1,
                                ))
                            )
                    # Traffic avoidance is capped below two label rows, so
                    # farther rings cannot beat a valid candidate found here.
                    if spiral_candidates and radius * spiral_step > metrics.px(48):
                        break
                if spiral_candidates:
                    chosen, _score = min(spiral_candidates, key=lambda item: item[1])
                if chosen is None:
                    # Preserve every arrow even when label-label overlap is
                    # unavoidable: search the canvas before the final clamp.
                    grid_step = metrics.integer(6)
                    start = max(1, round(edge_margin))
                    grid_candidates = []
                    for top in range(
                        start, max(start + 1, size[1] - math.ceil(cluster_height)), grid_step
                    ):
                        for left in range(
                            start, max(start + 1, size[0] - math.ceil(cluster_width)), grid_step
                        ):
                            candidate = (float(left), float(top), left + cluster_width, top + cluster_height)
                            if not any(_rects_overlap(candidate, footprint, padding=0) for footprint in arrow_footprints):
                                displacement = math.hypot(
                                    (candidate[0] + candidate[2]) / 2 - anchor.x,
                                    (candidate[1] + candidate[3]) / 2 - anchor.y,
                                )
                                grid_candidates.append(
                                    (candidate, score(candidate, displacement, 0))
                                )
                    if grid_candidates:
                        chosen, _score = min(grid_candidates, key=lambda item: item[1])
                if chosen is None:
                    top = min(
                        max(edge_margin, anchor.y - cluster_height / 2),
                        size[1] - cluster_height - edge_margin,
                    )
                    fallback_candidates = []
                    for side in (1, -1):
                        proposed = (
                            anchor.x + label_gap
                            if side > 0 else anchor.x - label_gap - cluster_width
                        )
                        left = min(
                            max(edge_margin, proposed),
                            size[0] - cluster_width - edge_margin,
                        )
                        candidate = (left, top, left + cluster_width, top + cluster_height)
                        if not _rects_overlap(
                            candidate, _arrow_footprint((anchor.x, anchor.y), metrics),
                            padding=0,
                        ):
                            fallback_candidates.append((
                                candidate, score(candidate, 0, 0 if side < 0 else 1)
                            ))
                    if fallback_candidates:
                        chosen, _score = min(fallback_candidates, key=lambda item: item[1])
                    else:
                        left = min(
                            max(edge_margin, anchor.x + label_gap),
                            size[0] - cluster_width - edge_margin,
                        )
                        chosen = (left, top, left + cluster_width, top + cluster_height)
            occupied.append(chosen)
            placed.append(LabelPlacement(
                "/".join(text for text, _operator, _unreliable in rows),
                chosen, (anchor.x, anchor.y), anchor.operator, anchor.heading,
                all(unreliable for _text, _operator, unreliable in rows), rows,
            ))
            continue
        for marker in group:
            text = "/".join(marker.routes)
            pill_width = float(draw.textlength(text, font=font)) + text_padding
            chosen = None
            # Candidate rows are scored symmetrically around the anchor. Every
            # candidate keeps the FULL pill size —
            # never squashed — so boxes stay readable at map edges.
            label_step = pill_height + row_gap
            candidate_offsets = [0.0, -label_step, label_step, -2 * label_step, 2 * label_step]
            candidates: list[
                tuple[tuple[float, float, float, float], tuple[float, float, int, float]]
            ] = []
            for offset_y in candidate_offsets:
                top = marker.y + offset_y - pill_height / 2
                bottom = top + pill_height
                for text_side in (1, -1):
                    if text_side > 0:
                        left, right = marker.x + label_gap, marker.x + label_gap + pill_width
                    else:
                        left, right = marker.x - label_gap - pill_width, marker.x - label_gap
                    rect = (left, top, right, bottom)
                    if (
                        rect[0] < edge_margin
                        or rect[1] < edge_margin
                        or rect[2] > size[0] - edge_margin
                        or rect[3] > size[1] - edge_margin
                    ):
                        continue
                    if any(
                        _rects_overlap(rect, other, padding=collision_padding)
                        for other in occupied
                    ):
                        continue
                    if any(_rects_overlap(rect, footprint, padding=0) for footprint in arrow_footprints):
                        continue
                    # Prefer the nearest unobstructed row; when left/right
                    # are equally good, choose left for stable, edge-friendly
                    # presentation of long route names.
                    candidates.append((
                        rect, score(rect, abs(offset_y), 0 if text_side < 0 else 1)
                    ))
            if candidates:
                chosen, _score = min(candidates, key=lambda item: item[1])
            if chosen is None:
                # No clean slot in the stack columns: spiral outward over
                # progressively larger offsets (up, down, left, right) until
                # a free full-size slot exists. A pill NEVER ends up on top
                # of another one.
                chosen = None
                spiral_candidates = []
                for radius in range(1, 40):
                    for dx, dy in (
                        (-radius * spiral_step, 0),
                        (radius * spiral_step, 0),
                        (0, radius * (pill_height + row_gap)),
                        (0, -radius * (pill_height + row_gap)),
                        (-radius * spiral_step, -radius * (pill_height + row_gap)),
                        (radius * spiral_step, -radius * (pill_height + row_gap)),
                        (-radius * spiral_step, radius * (pill_height + row_gap)),
                        (radius * spiral_step, radius * (pill_height + row_gap)),
                    ):
                        left = marker.x + dx + (
                            label_gap if dx >= 0 else -label_gap - pill_width
                        )
                        right = left + pill_width
                        top = marker.y + dy - pill_height / 2
                        bottom = top + pill_height
                        rect = (left, top, right, bottom)
                        if (
                            rect[0] < edge_margin
                            or rect[1] < edge_margin
                            or rect[2] > size[0] - edge_margin
                            or rect[3] > size[1] - edge_margin
                        ):
                            continue
                        if any(
                            _rects_overlap(rect, other, padding=collision_padding)
                            for other in occupied
                        ):
                            continue
                        if any(_rects_overlap(rect, footprint, padding=0) for footprint in arrow_footprints):
                            continue
                        spiral_candidates.append((
                            rect, score(
                                rect, math.hypot(dx, dy), 0 if dx < 0 else 1
                            )
                        ))
                    if spiral_candidates and radius * spiral_step > metrics.px(48):
                        break
                if spiral_candidates:
                    chosen, _score = min(spiral_candidates, key=lambda item: item[1])
            if chosen is None:
                # Truly nowhere free (map wall-to-wall buses): keep the
                # full-size pill on the road row, clipped into the canvas —
                # never squashed.
                top = min(
                    max(edge_margin, marker.y - pill_height / 2),
                    size[1] - pill_height - edge_margin,
                )
                fallback_candidates = []
                for side in (1, -1):
                    proposed = (
                        marker.x + label_gap
                        if side > 0 else marker.x - label_gap - pill_width
                    )
                    left = min(
                        max(edge_margin, proposed),
                        size[0] - pill_width - edge_margin,
                    )
                    candidate = (left, top, left + pill_width, top + pill_height)
                    if not _rects_overlap(
                        candidate, _arrow_footprint((marker.x, marker.y), metrics),
                        padding=0,
                    ):
                        fallback_candidates.append((
                            candidate, score(candidate, 0, 0 if side < 0 else 1)
                        ))
                if fallback_candidates:
                    chosen, _score = min(fallback_candidates, key=lambda item: item[1])
                else:
                    left = min(
                        max(edge_margin, marker.x + label_gap),
                        size[0] - pill_width - edge_margin,
                    )
                    chosen = (left, top, left + pill_width, top + pill_height)
            occupied.append(chosen)
            placed.append(
                LabelPlacement(
                    text,
                    chosen,
                    (marker.x, marker.y),
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
    metrics: RenderMetrics = DEFAULT_METRICS,
) -> None:
    """Rounded rectangle with a dashed outline (PIL lacks dash support)."""
    left, top, right, bottom = rect
    scaled_radius = metrics.integer(radius)
    scaled_dash = metrics.px(dash, minimum=1.0)
    scaled_gap = metrics.px(gap, minimum=1.0)
    draw.rounded_rectangle(rect, radius=scaled_radius, fill=fill)

    def dashed_line(a, b):
        length = math.hypot(b[0] - a[0], b[1] - a[1])
        if length <= 0:
            return
        ux, uy = (b[0] - a[0]) / length, (b[1] - a[1]) / length
        travelled = 0.0
        while travelled < length:
            end = min(travelled + scaled_dash, length)
            draw.line(
                (
                    (a[0] + ux * travelled, a[1] + uy * travelled),
                    (a[0] + ux * end, a[1] + uy * end),
                ),
                fill=outline,
                width=metrics.integer(1),
            )
            travelled = end + scaled_gap

    # Straight segments only (corner arcs stay undashed; visually equivalent).
    dashed_line((left + scaled_radius, top), (right - scaled_radius, top))
    dashed_line((left + scaled_radius, bottom), (right - scaled_radius, bottom))
    dashed_line((left, top + scaled_radius), (left, bottom - scaled_radius))
    dashed_line((right, top + scaled_radius), (right, bottom - scaled_radius))


def _draw_bus_route_marker(
    draw: ImageDraw.ImageDraw,
    placement: LabelPlacement,
    color: tuple[int, int, int],
    font: ImageFont.ImageFont,
    unreliable: bool = False,
    phase: str = "all",
    metrics: RenderMetrics = DEFAULT_METRICS,
) -> None:
    """Draw one coloured route label with its arrow at the road anchor."""
    left, top, right, bottom = placement.rect
    anchor_x, anchor_y = placement.marker
    # A short palette-colored leader preserves the association when displaced.
    # It terminates exactly at the label edge.
    end_x = min(max(anchor_x, left), right)
    end_y = min(max(anchor_y, top), bottom)
    if phase in {"all", "connector"} and (end_x, end_y) != (anchor_x, anchor_y):
        palette = list(dict.fromkeys(
            OPERATOR_COLORS.get(op, (100, 100, 100))
            for _, op, _ in placement.rows
        )) or [color]
        for index, connector_color in enumerate(palette):
            start = index / len(palette)
            finish = (index + 1) / len(palette)
            draw.line((anchor_x + (end_x - anchor_x) * start,
                       anchor_y + (end_y - anchor_y) * start,
                       anchor_x + (end_x - anchor_x) * finish,
                       anchor_y + (end_y - anchor_y) * finish),
                      fill=connector_color + (220,), width=metrics.integer(1))
    if phase == "connector":
        return
    if phase == "arrow":
        colors = list(dict.fromkeys(OPERATOR_COLORS.get(operator, (100, 100, 100)) for _, operator, _ in placement.rows)) if placement.rows else [color]
        _draw_colored_bus_arrow(
            draw, (anchor_x, anchor_y), placement.heading, colors, metrics
        )
        return
    def text_origin(text: str, row_left: float, row_right: float) -> float:
        width = draw.textlength(text, font=font)
        padding = metrics.px(4)
        return min(row_left + padding, row_right - width - padding)

    if placement.rows:
        row_height = (bottom - top) / len(placement.rows)
        # Paint each row independently so a mixed KMB/Citybus/GMB cluster is
        # still immediately identifiable.  The outer outline is shared.
        for index, (text, operator, row_unreliable) in enumerate(placement.rows):
            row_top = top + index * row_height
            row_bottom = top + (index + 1) * row_height
            row_color = OPERATOR_COLORS.get(operator, (100, 100, 100))
            if row_unreliable:
                row_color = tuple(int(c + (255 - c) * 0.55) for c in row_color)
            row_rect = (left, row_top, right, row_bottom)
            draw.rectangle(row_rect, fill=row_color + (220,))
            if row_unreliable:
                _dashed_rounded_rectangle(
                    draw, row_rect, radius=3, fill=row_color + (0,),
                    outline=(90, 90, 90, 235),
                    metrics=metrics,
                )
            draw.text(
                (text_origin(text, left, right), row_top + metrics.px(2)), text,
                fill=(235, 235, 235, 255) if row_unreliable else (255, 255, 255, 255),
                font=font,
            )
        draw.rounded_rectangle(
            placement.rect, radius=metrics.integer(4),
            outline=(20, 20, 20, 255), width=metrics.integer(1),
        )
        colors = list(dict.fromkeys(OPERATOR_COLORS.get(operator, (100, 100, 100)) for _, operator, _ in placement.rows))
        if phase != "label":
            _draw_colored_bus_arrow(
                draw, (anchor_x, anchor_y), placement.heading, colors, metrics
            )
        return
    if unreliable:
        # Timetable-derived estimate: paler fill, dashed outline, dim text.
        pale = tuple(int(c + (255 - c) * 0.55) for c in color)
        draw.rounded_rectangle(
            placement.rect, radius=metrics.integer(4), fill=pale + (200,)
        )
        _dashed_rounded_rectangle(
            draw,
            placement.rect,
            radius=4,
            fill=pale + (0,),
            outline=(90, 90, 90, 235),
            metrics=metrics,
        )
        text_fill = (235, 235, 235, 255)
    else:
        draw.rounded_rectangle(
            placement.rect, radius=metrics.integer(4), fill=color + (245,),
            outline=(20, 20, 20, 255), width=metrics.integer(1)
        )
        text_fill = (255, 255, 255, 255)
    if phase != "label":
        _draw_colored_bus_arrow(
            draw, (anchor_x, anchor_y), placement.heading, [color], metrics
        )
    text_x = text_origin(placement.text, left, right)
    draw.text((text_x, top + metrics.px(2)), placement.text, fill=text_fill, font=font)


def _bus_direction_arrow_triangle(
    center: tuple[float, float],
    heading: float,
    metrics: RenderMetrics = DEFAULT_METRICS,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Return a heading triangle whose geometric centre is the road anchor."""
    arrow_center = center
    arrow_dx, arrow_dy = math.cos(heading), -math.sin(heading)
    arrow_left = (-arrow_dy, arrow_dx)
    arrow_tip = (
        arrow_center[0] + arrow_dx * metrics.px(6),
        arrow_center[1] + arrow_dy * metrics.px(6),
    )
    return (
        arrow_tip,
        (
            arrow_center[0] - arrow_dx * metrics.px(3) + arrow_left[0] * metrics.px(3),
            arrow_center[1] - arrow_dy * metrics.px(3) + arrow_left[1] * metrics.px(3),
        ),
        (
            arrow_center[0] - arrow_dx * metrics.px(3) - arrow_left[0] * metrics.px(3),
            arrow_center[1] - arrow_dy * metrics.px(3) - arrow_left[1] * metrics.px(3),
        ),
    )


def _merge_bus_markers(
    estimates: Iterable[object],
    center_lat: float,
    center_lon: float,
    zoom: float,
    size: tuple[int, int],
    metrics: RenderMetrics = DEFAULT_METRICS,
    traffic: TrafficOccupancy | None = None,
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
        if traffic is not None:
            x, y = traffic.snap_anchor((x, y), estimate.heading, metrics)
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
            round(x / metrics.px(4, minimum=1.0)),
            round(y / metrics.px(4, minimum=1.0)),
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
    metrics: RenderMetrics = DEFAULT_METRICS,
) -> None:
    font = _font(metrics.font_size(13))
    for label, lat, lon, label_above in GATE_PINS:
        x, y = project(lat, lon, center_lat, center_lon, zoom, size)
        draw.ellipse(
            (x - metrics.px(8), y - metrics.px(8),
             x + metrics.px(8), y + metrics.px(8)),
            fill=GATE_PIN_COLOR + (235,),
            outline=(255, 255, 255, 255),
            width=metrics.integer(2),
        )
        text_width = draw.textlength(label, font=font)
        edge = metrics.px(3)
        text_x = min(max(x + metrics.px(11), edge), size[0] - text_width - edge)
        text_y = y - metrics.px(19) if label_above else y + metrics.px(9)
        draw.rounded_rectangle(
            (text_x - metrics.px(3), text_y - metrics.px(2),
             text_x + text_width + metrics.px(3), text_y + metrics.px(16)),
            radius=metrics.integer(3),
            fill=(255, 255, 255, 215),
        )
        draw.text((text_x, text_y), label, fill=GATE_PIN_COLOR + (255,), font=font)


def _draw_legend(
    draw: ImageDraw.ImageDraw,
    size: tuple[int, int],
    metrics: RenderMetrics = DEFAULT_METRICS,
    origin: tuple[float, float] = (0.0, 0.0),
) -> None:
    """Explain only dashboard-authored estimates and stop glyphs."""
    del size  # The artwork retains a stable logical 420 x 110 coordinate space.
    font = _font(metrics.font_size(12))
    x = origin[0] + metrics.px(10)
    y = origin[1] + metrics.px(8)
    draw.text((x, y), "Estimated buses (not GPS)", fill=(30, 30, 30, 255), font=font)
    cursor_x = x + metrics.px(158)
    for operator, color in OPERATOR_COLORS.items():
        draw.rounded_rectangle(
            (cursor_x, y + metrics.px(1), cursor_x + metrics.px(12),
             y + metrics.px(10)),
            radius=metrics.integer(2),
            fill=color + (255,),
            outline=(20, 20, 20, 255),
            width=metrics.integer(1),
        )
        draw.text(
            (cursor_x + metrics.px(15), y - metrics.px(2)), operator.value,
            fill=(30, 30, 30, 255), font=font,
        )
        cursor_x += metrics.px(23) + draw.textlength(operator.value, font=font)

    row_y = y + metrics.px(27)
    marker_size = metrics.integer(STOP_MARKER_SIZE, minimum=4)
    draw.ellipse(
        (x, row_y, x + marker_size - 1, row_y + marker_size - 1),
        fill=SHUTTLE_STOP_COLOR + (255,),
        outline=(255, 255, 255, 255),
        width=metrics.integer(1),
    )
    draw.text(
        (x + metrics.px(13), row_y - metrics.px(3)), "shuttle stop",
        fill=(30, 30, 30, 255), font=font,
    )
    public_x = x + metrics.px(112)
    draw.rectangle(
        (public_x, row_y, public_x + marker_size - 1, row_y + marker_size - 1),
        fill=PUBLIC_STOP_COLOR + (255,),
        outline=(255, 255, 255, 255),
        width=metrics.integer(1),
    )
    draw.text(
        (public_x + metrics.px(13), row_y - metrics.px(3)), "public bus stop",
        fill=(30, 30, 30, 255), font=font,
    )
    # Unreliable (timetable-derived) marker swatch: pale with dashed outline.
    pale_kmb = tuple(int(c + (255 - c) * 0.55) for c in OPERATOR_COLORS[Operator.KMB])
    unreliable_x = x + metrics.px(240)
    _dashed_rounded_rectangle(
        draw,
        (unreliable_x, row_y, unreliable_x + metrics.px(26),
         row_y + metrics.px(13)),
        radius=3,
        fill=pale_kmb + (200,),
        outline=(90, 90, 90, 235),
        metrics=metrics,
    )
    draw.text(
        (unreliable_x + metrics.px(31), row_y - metrics.px(3)),
        "timetable only",
        fill=(30, 30, 30, 255),
        font=font,
    )

    # Match the map's high-contrast, no-fill rectangle indicator.
    draw.rectangle(
        (x, y + metrics.px(54), x + metrics.px(20), y + metrics.px(64)),
        outline=ALERT_RECT_COLOR,
        width=metrics.integer(4),
    )
    draw.text(
        (x + metrics.px(27), y + metrics.px(53)),
        "traffic-news segment",
        fill=(30, 30, 30, 255),
        font=font,
    )
    # Keep the Google traffic key swatch-first, matching the other legend
    # entries. The label follows the five sampled road colours.
    traffic_x = x + metrics.px(178)
    compact_font = _font(metrics.font_size(9, minimum=6))
    swatch_x = traffic_x
    for color in GOOGLE_TRAFFIC_COLORS:
        draw.rounded_rectangle(
            (swatch_x, y + metrics.px(55), swatch_x + metrics.px(13),
             y + metrics.px(64)),
            radius=metrics.integer(2), fill=color + (255,),
            outline=(50, 50, 50, 255), width=metrics.integer(1),
        )
        swatch_x += metrics.px(17)
    draw.text(
        (swatch_x + metrics.px(3), y + metrics.px(55)), "Google traffic",
        fill=(45, 45, 45, 255), font=compact_font,
    )
    attribution = "Map data © Google · Route geometry © Transport Department HKeMobility"
    draw.text(
        (x, origin[1] + metrics.px(94)),
        attribution,
        fill=(65, 65, 65, 255),
        font=_font(metrics.font_size(9, minimum=7)),
    )


def _arrow_footprint(
    anchor: tuple[float, float], metrics: RenderMetrics = DEFAULT_METRICS
) -> tuple[float, float, float, float]:
    radius = metrics.px(6)
    return (anchor[0] - radius, anchor[1] - radius,
            anchor[0] + radius, anchor[1] + radius)


def _draw_colored_bus_arrow(
    draw: ImageDraw.ImageDraw,
    center: tuple[float, float],
    heading: float,
    colors: list[tuple[int, int, int]],
    metrics: RenderMetrics = DEFAULT_METRICS,
) -> None:
    """Draw an anchored operator-colored arrow, split into equal wedges."""
    triangle = _bus_direction_arrow_triangle(center, heading, metrics)
    if not colors:
        colors = [(100, 100, 100)]
    if len(colors) == 1:
        draw.polygon(triangle, fill=colors[0] + (255,))
        draw.line(
            (*triangle, triangle[0]), fill=(25, 25, 25, 255),
            width=metrics.integer(1), joint="curve",
        )
        return
    tip, left, right = triangle
    # Equal strips across the arrow base, each joined to the tip.
    for index, color in enumerate(colors):
        start = index / len(colors)
        end = (index + 1) / len(colors)
        a = (
            left[0] + (right[0] - left[0]) * start,
            left[1] + (right[1] - left[1]) * start,
        )
        b = (
            left[0] + (right[0] - left[0]) * end,
            left[1] + (right[1] - left[1]) * end,
        )
        draw.polygon((tip, a, b), fill=color + (255,))
    draw.line(
        (tip, left, right, tip), fill=(25, 25, 25, 255),
        width=metrics.integer(1), joint="curve",
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
    scale = map_image.width / MAP_WIDTH
    band_height = max(1, round(LEGEND_BAND_HEIGHT * scale))
    legend = Image.new("RGB", (map_image.width, band_height), (246, 247, 249))
    # The logical 420 x 110 legend is authored directly at its 2x display
    # scale. This keeps type and shapes crisp instead of resizing a raster.
    legend_metrics = RenderMetrics(2 * scale)
    _draw_legend(
        ImageDraw.Draw(legend, "RGBA"), legend.size, legend_metrics,
        origin=(60 * scale, 20 * scale),
    )
    composite = Image.new(
        "RGB", (map_image.width, map_image.height + band_height), (246, 247, 249)
    )
    composite.paste(map_image, (0, 0))
    composite.paste(legend, (0, map_image.height))
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
    metrics: RenderMetrics = DEFAULT_METRICS,
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
        padding = metrics.px(ALERT_RECT_PADDING)
        left = max(0.0, min(point[0] for point in points) - padding)
        top = max(0.0, min(point[1] for point in points) - padding)
        right = min(float(size[0] - 1), max(point[0] for point in points) + padding)
        bottom = min(float(size[1] - 1), max(point[1] for point in points) + padding)
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
                if _rects_overlap(candidate, other, padding=metrics.px(2)):
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
            width=metrics.integer(ALERT_RECT_OUTLINE_WIDTH),
        )
    canvas.alpha_composite(overlay)
    return len(merged)


def _render_map_once(
    estimates: list,
    cache_dir: str,
    public_stops: Iterable[object] = (),
    route_lines: Iterable[object] = (),
    base_image: Image.Image | None = None,
    affected_road_paths: Iterable[Iterable[tuple[float, float]]] = (),
) -> bytes:
    route_lines = list(route_lines)
    public_stops = list(public_stops)
    center_lat, center_lon = BASE_MAP_LAT, BASE_MAP_LON
    if base_image is not None:
        canvas = base_image.copy().convert("RGBA")
    else:
        canvas = _background(cache_dir, (center_lat, center_lon), BASE_MAP_ZOOM).convert("RGBA")
    size = canvas.size
    metrics = _render_metrics(size)
    zoom = BASE_MAP_ZOOM + math.log2(metrics.scale)
    traffic = _traffic_occupancy(canvas, metrics)
    draw = ImageDraw.Draw(canvas, "RGBA")

    route_paths = [list(line.path) for line in route_lines if len(getattr(line, "path", ())) >= 2]

    # Traffic-news road indicators go beneath everything dashboard-drawn.
    _draw_alerted_road_rectangles(
        canvas,
        affected_road_paths,
        center_lat,
        center_lon,
        zoom,
        size,
        metrics,
    )

    # All stop glyphs share the same logical 8 px measure at every candidate scale.
    for _label, lat, lon in SHUTTLE_STOPS:
        x, y = project(lat, lon, center_lat, center_lon, zoom, size)
        _draw_marker_on_left(
            draw,
            x,
            y,
            _nearest_road_heading(lat, lon, route_paths),
            SHUTTLE_STOP_COLOR,
            square=False,
            metrics=metrics,
        )

    _draw_gate_pins(draw, center_lat, center_lon, zoom, size, metrics)

    for x, y, heading in _merged_public_stop_markers(
        public_stops, route_lines, route_paths, center_lat, center_lon, zoom, size,
        metrics=metrics,
    ):
        _draw_marker_on_left(
            draw, x, y, heading, PUBLIC_STOP_COLOR, square=True, metrics=metrics
        )

    font = _font(metrics.font_size(13))
    bus_markers = _merge_bus_markers(
        estimates, center_lat, center_lon, zoom, size, metrics, traffic
    )
    placements = _layout_bus_labels(
        bus_markers, draw, font, size, metrics=metrics, traffic=traffic
    )
    for placement in placements:
        _draw_bus_route_marker(
            draw, placement, OPERATOR_COLORS.get(placement.operator, (100, 100, 100)),
            font, unreliable=placement.unreliable, phase="connector", metrics=metrics,
        )
    for placement in placements:
        _draw_bus_route_marker(
            draw,
            placement,
            OPERATOR_COLORS.get(placement.operator, (100, 100, 100)),
            font,
            unreliable=placement.unreliable,
            phase="label",
            metrics=metrics,
        )
    for placement in placements:
        _draw_bus_route_marker(
            draw,
            placement,
            OPERATOR_COLORS.get(placement.operator, (100, 100, 100)),
            font,
            unreliable=placement.unreliable,
            phase="arrow",
            metrics=metrics,
        )

    buffer = io.BytesIO()
    # Discord/mobile payload target: retain full dimensions while adapting
    # quality. RGB WebP keeps traffic colours (no palette quantization).
    image = _append_legend_band(canvas)
    # The caller rerenders overlays at smaller native resolutions when needed;
    # this pass must never resize an already-composed image.
    for quality in (82, 78, 74, 70, 65, 60):
        buffer.seek(0)
        buffer.truncate(0)
        image.save(buffer, format="WEBP", quality=quality, method=6)
        if buffer.tell() <= 100_000:
            return buffer.getvalue()
    raise _OversizedMapError("map WebP exceeds 100 KB at native resolution")


def render_map(
    estimates: list,
    cache_dir: str,
    public_stops: Iterable[object] = (),
    route_lines: Iterable[object] = (),
    base_image: Image.Image | None = None,
    affected_road_paths: Iterable[Iterable[tuple[float, float]]] = (),
) -> bytes:
    """Render overlays natively at progressively smaller fixed resolutions."""
    estimates = list(estimates)
    public_stops = list(public_stops)
    route_lines = list(route_lines)
    affected_road_paths = [list(path) for path in affected_road_paths]
    pristine = (
        base_image.copy().convert("RGBA")
        if base_image is not None
        else _background(cache_dir).convert("RGBA")
    )
    for width, height in ((960, 540), (864, 486), (778, 438), (720, 405)):
        candidate = (
            pristine.resize((width, height), Image.Resampling.LANCZOS)
            if pristine is not None and pristine.size != (width, height)
            else pristine
        )
        try:
            return _render_map_once(
                estimates, cache_dir, public_stops, route_lines, candidate,
                affected_road_paths,
            )
        except _OversizedMapError:
            if width == MIN_MAP_WIDTH:
                raise
    raise _OversizedMapError("map WebP exceeds 100 KB at readable minimum dimensions")
