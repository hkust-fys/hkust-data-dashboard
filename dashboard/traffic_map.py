"""Headless traffic map renderer.

Builds a 1024x600 map over the HKUST approach area using OpenStreetMap
standard tiles as background (attribution required) and overlays fresh TD
detector observations, incidents, and road labels. The map bounds are
auto-fitted to the detector points actually captured (plus HKUST and padding),
so the content is centered rather than offset. Tiles are cached on disk for at
least 7 days; the PNG is returned in memory — no historical maps are kept.
"""

from __future__ import annotations

import io
import logging
import os
import time
from datetime import datetime

from PIL import Image, ImageDraw, ImageFont

from dashboard.models import (
    Roadwork,
    SpeedBand,
    TrafficCorridorStatus,
    TrafficIncident,
)

log = logging.getLogger(__name__)

MAP_WIDTH = 1024
MAP_HEIGHT = 600

# Fallback map bounds (approx. HKUST approach area) used when no detector
# points are captured; otherwise the map auto-fits the live data.
FALLBACK_LON_MIN, FALLBACK_LON_MAX = 114.17, 114.33
FALLBACK_LAT_MIN, FALLBACK_LAT_MAX = 22.30, 22.40

# HKUST campus centre (always included in the fitted bounds).
HKUST_LON, HKUST_LAT = 114.2656, 22.3364

# Padding around the fitted points (degrees), so markers are not clipped.
LON_PAD = 0.012
LAT_PAD = 0.010
# Minimum span so a single point still produces a readable map.
MIN_LON_SPAN = 0.08
MIN_LAT_SPAN = 0.05

OSM_TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
OSM_USER_AGENT = "hkust-data-dashboard/2.0 (HKUST FYS campus dashboard; contact: repo maintainers)"
OSM_TILE_TTL_DAYS = 7

TILE_SIZE = 256

# Speed band colors (dashboard heuristics).
BAND_COLORS = {
    SpeedBand.RED: (220, 38, 38),
    SpeedBand.AMBER: (245, 158, 11),
    SpeedBand.GREEN: (34, 197, 94),
    SpeedBand.GRAY: (156, 163, 175),
}
INCIDENT_COLOR = (139, 92, 246)
HKUST_COLOR = (59, 130, 246)
LABEL_COLOR = (30, 30, 30)
BG_COLOR = (238, 240, 243)


def _fit_bounds(statuses: list[TrafficCorridorStatus]) -> tuple[float, float, float, float]:
    """Compute (lon_min, lon_max, lat_min, lat_max) covering the captured
    detector points plus HKUST, with padding and minimum spans."""
    lons = [HKUST_LON]
    lats = [HKUST_LAT]
    for status in statuses:
        for obs in status.observations:
            if obs.latitude and obs.longitude:
                lats.append(obs.latitude)
                lons.append(obs.longitude)
    lon_min, lon_max = min(lons), max(lons)
    lat_min, lat_max = min(lats), max(lats)

    lon_min -= LON_PAD
    lon_max += LON_PAD
    lat_min -= LAT_PAD
    lat_max += LAT_PAD

    # Enforce minimum spans (center the existing span inside them).
    if lon_max - lon_min < MIN_LON_SPAN:
        mid = (lon_min + lon_max) / 2
        lon_min, lon_max = mid - MIN_LON_SPAN / 2, mid + MIN_LON_SPAN / 2
    if lat_max - lat_min < MIN_LAT_SPAN:
        mid = (lat_min + lat_max) / 2
        lat_min, lat_max = mid - MIN_LAT_SPAN / 2, mid + MIN_LAT_SPAN / 2
    return lon_min, lon_max, lat_min, lat_max


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    """True when two (x0, y0, x1, y1) rectangles intersect."""
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _tile_coords(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    import math

    lat_rad = math.radians(lat)
    n = 2**zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def _tile_path(cache_dir: str, zoom: int, x: int, y: int) -> str:
    return os.path.join(cache_dir, "osm", str(zoom), str(x), f"{y}.png")


def _load_tile(
    cache_dir: str, zoom: int, x: int, y: int
) -> Image.Image | None:
    """Load a cached tile, downloading only when stale/missing."""
    path = _tile_path(cache_dir, zoom, x, y)
    if os.path.isfile(path):
        age_days = (time.time() - os.path.getmtime(path)) / 86400.0
        if age_days <= OSM_TILE_TTL_DAYS:
            try:
                return Image.open(path).convert("RGB")
            except OSError:
                pass

    # Bound the download; download synchronously via urllib is acceptable here
    # because the renderer runs inside the async loop rarely.
    import urllib.request

    url = OSM_TILE_URL.format(z=zoom, x=x, y=y)
    req = urllib.request.Request(url, headers={"User-Agent": OSM_USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
            data = resp.read(2 * 1024 * 1024)
    except Exception as exc:  # noqa: BLE001
        log.warning("OSM tile download failed for %s: %s", url, exc)
        return None
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except OSError as exc:
        log.warning("Invalid tile data for %s: %s", url, exc)
        return None
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    return img


def _stich_background(
    cache_dir: str, zoom: int, x_min: int, y_min: int, width_tiles: int, height_tiles: int
) -> Image.Image:
    """Stitch the fixed tile set into one background image (neutral fallback).

    The stitched tiles are blended toward white so road/land details stay
    readable but the colored speed markers stand out clearly.
    """
    bg = Image.new("RGB", (width_tiles * TILE_SIZE, height_tiles * TILE_SIZE), BG_COLOR)
    loaded = 0
    for dx in range(width_tiles):
        for dy in range(height_tiles):
            tile = _load_tile(cache_dir, zoom, x_min + dx, y_min + dy)
            if tile is not None:
                bg.paste(tile, (dx * TILE_SIZE, dy * TILE_SIZE))
                loaded += 1
    if loaded:
        # lighten: 55% white + 45% tile so speed markers stand out
        white = Image.new("RGB", bg.size, (255, 255, 255))
        bg = Image.blend(bg, white, alpha=0.55)
    return bg


def _lonlat_to_pixel(
    lon: float, lat: float, zoom: int, x_min: int, y_min: int
) -> tuple[float, float]:
    import math

    n = 2**zoom
    x = (lon + 180.0) / 360.0 * n
    lat_rad = math.radians(lat)
    y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    return (x - x_min) * TILE_SIZE, (y - y_min) * TILE_SIZE


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "Arial.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_traffic_map(
    statuses: list[TrafficCorridorStatus],
    incidents: list[TrafficIncident],
    roadworks: list[Roadwork],
    capture_time: datetime | None,
    cache_dir: str = ".cache",
    zoom: int = 14,
) -> tuple[bytes, str | None]:
    """Render the traffic map PNG.

    Returns (png_bytes, error_message). On total tile failure the same bounded
    PNG is still produced on a neutral background.
    """
    # Auto-fit the map to the captured points; fall back to the fixed box.
    if statuses:
        lon_min, lon_max, lat_min, lat_max = _fit_bounds(statuses)
    else:
        lon_min, lon_max = FALLBACK_LON_MIN, FALLBACK_LON_MAX
        lat_min, lat_max = FALLBACK_LAT_MIN, FALLBACK_LAT_MAX

    # Compute tile span covering the bounds.
    x_min, y_min = _tile_coords(lon_min, lat_max, zoom)
    x_max, y_max = _tile_coords(lon_max, lat_min, zoom)
    width_tiles = x_max - x_min + 1
    height_tiles = y_max - y_min + 1

    bg = _stich_background(cache_dir, zoom, x_min, y_min, width_tiles, height_tiles)

    # Crop to exact map size.
    # Top-left pixel of the bounds.
    px0 = _lonlat_to_pixel(lon_min, lat_max, zoom, x_min, y_min)
    canvas = bg.crop((int(px0[0]), int(px0[1]), int(px0[0]) + MAP_WIDTH, int(px0[1]) + MAP_HEIGHT))
    draw = ImageDraw.Draw(canvas)

    # Detector markers: smaller dots with collision-aware label placement so
    # close pairs (e.g. the two New Clear Water Bay Road detectors near Shun
    # Lee) stay visually separate. Markers closer than MIN_MARKER_DIST px are
    # nudged apart so the dots do not overlap.
    font_small = _load_font(13)
    marker_r = 5
    min_marker_dist = marker_r * 2 + 6

    points: list[tuple[float, float]] = []
    used_rects: list[tuple[int, int, int, int]] = []
    for status in statuses:
        for obs in status.observations:
            if obs.latitude == 0.0 and obs.longitude == 0.0:
                continue
            px = _lonlat_to_pixel(obs.longitude, obs.latitude, zoom, x_min, y_min)
            x, y = px[0] - px0[0], px[1] - px0[1]
            if not (0 <= x <= MAP_WIDTH and 0 <= y <= MAP_HEIGHT):
                continue
            # nudge away from any previously placed marker that is too close
            for ox, oy in points:
                dx, dy = x - ox, y - oy
                dist = (dx * dx + dy * dy) ** 0.5
                if 0 < dist < min_marker_dist:
                    push = (min_marker_dist - dist) / 2
                    x += dx / dist * push
                    y += dy / dist * push
            points.append((x, y))
            xi, yi = int(x), int(y)
            color = BAND_COLORS.get(obs.band, BAND_COLORS[SpeedBand.GRAY])
            draw.ellipse((xi - marker_r, yi - marker_r, xi + marker_r, yi + marker_r),
                         fill=color, outline=(255, 255, 255), width=1)
            if obs.speed_kmh is not None:
                label = f"{obs.speed_kmh:.0f}"
                lw = draw.textlength(label, font=font_small)
                # pick a label corner that does not collide with earlier labels
                candidates = [
                    (xi + marker_r + 3, yi - 10),
                    (xi - marker_r - 3 - lw, yi - 10),
                    (xi + marker_r + 3, yi + 4),
                    (xi - marker_r - 3 - lw, yi + 4),
                ]
                lx, ly = candidates[0]
                for cx, cy in candidates[1:]:
                    rect = (cx, cy, cx + lw, cy + 14)
                    if not any(_rects_overlap(rect, u) for u in used_rects):
                        lx, ly = cx, cy
                        break
                used_rects.append((lx, ly, lx + lw, ly + 14))
                draw.text((lx, ly), label, fill=LABEL_COLOR, font=font_small)

    # Incident markers.
    font_tiny = _load_font(12)
    # incidents carry no coordinates; shown in the legend below (count only)
    incident_count = len(incidents)

    # Roadworks markers (no coordinates in our model — skip drawing; shown in legend).

    # HKUST pin (approx. campus centre).
    hk_lon, hk_lat = 114.2656, 22.3364
    px = _lonlat_to_pixel(hk_lon, hk_lat, zoom, x_min, y_min)
    x, y = int(px[0] - px0[0]), int(px[1] - px0[1])
    draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=HKUST_COLOR, outline=(255, 255, 255), width=2)
    draw.text((x + 10, y - 10), "HKUST", fill=HKUST_COLOR, font=font_small)

    # Legend.
    legend_x, legend_y = 10, MAP_HEIGHT - 118
    draw.rectangle((legend_x - 6, legend_y - 8, legend_x + 210, legend_y + 104), fill=(255, 255, 255, 200))
    draw.text((legend_x, legend_y), "Speed (km/h):", fill=LABEL_COLOR, font=font_small)
    items = [
        (BAND_COLORS[SpeedBand.RED], "< 20"),
        (BAND_COLORS[SpeedBand.AMBER], "20 – 40"),
        (BAND_COLORS[SpeedBand.GREEN], "> 40"),
        (BAND_COLORS[SpeedBand.GRAY], "no data"),
    ]
    y = legend_y + 20
    for color, label in items:
        draw.ellipse((legend_x, y - 4, legend_x + 10, y + 6), fill=color)
        draw.text((legend_x + 16, y - 4), label, fill=LABEL_COLOR, font=font_small)
        y += 20
    if incident_count:
        draw.text((legend_x, y), f"⚠ {incident_count} incident(s)", fill=INCIDENT_COLOR, font=font_small)

    # Attribution + capture time.
    capture_str = capture_time.strftime("%H:%M") if capture_time else "—"
    draw.text(
        (MAP_WIDTH - 420, MAP_HEIGHT - 18),
        f"TD {capture_str} · © OpenStreetMap contributors",
        fill=(60, 60, 60),
        font=font_tiny,
    )

    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue(), None
