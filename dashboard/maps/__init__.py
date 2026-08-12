"""Small public API for dashboard map generation."""

from __future__ import annotations

import asyncio
import logging

from dashboard.maps.geometry import decode_polyline, encode_polyline
from dashboard.maps.renderer import (
    CONGESTION_COLORS,
    MAP_HEIGHT,
    MAP_WIDTH,
    congestion_color,
    fit_view,
    predict_buses,
    project,
    render_map,
)
from dashboard.maps.tiles import capture_gmaps_base
from dashboard.models import RouteEtaGroup
from dashboard.providers.route_geometry import Stop, fetch_route_geometry

log = logging.getLogger(__name__)


async def fetch_traffic_map(
    client,
    groups: list[RouteEtaGroup] | None = None,
    cache_dir: str = ".cache",
) -> tuple[bytes | None, list[object]]:
    """Capture the Google base map and render retained bus/stop markers."""
    base_image_task = asyncio.create_task(capture_gmaps_base(cache_dir=cache_dir))
    base_image = await base_image_task

    public_stops: list[Stop] = []
    route_lines: list[object] = []
    try:
        geometry = await fetch_route_geometry(client, cache_dir=cache_dir)
        public_stops = geometry.stops
        route_lines = list(geometry.routes)
    except Exception as exc:  # noqa: BLE001
        log.warning("public stop geometry unavailable: %s", type(exc).__name__)
    if not public_stops:
        public_stops = [
            Stop("HKUST-N", "HKUST North Gate", 22.338678, 114.261946),
            Stop("HKUST-S", "HKUST South Gate", 22.333360, 114.262881),
        ]
    try:
        png = await asyncio.to_thread(
            render_map,
            groups or [],
            cache_dir,
            public_stops,
            route_lines,
            base_image,
        )
        return png, []
    except Exception as exc:  # noqa: BLE001
        log.warning("map rendering failed: %s", type(exc).__name__)
        return None, []


__all__ = [
    "CONGESTION_COLORS",
    "MAP_HEIGHT",
    "MAP_WIDTH",
    "congestion_color",
    "decode_polyline",
    "encode_polyline",
    "fetch_traffic_map",
    "fit_view",
    "predict_buses",
    "project",
]
