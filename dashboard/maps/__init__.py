"""Small public API for dashboard map generation."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from dashboard.maps.positions import BusEstimate, estimate_bus_positions
from dashboard.maps.renderer import (
    MAP_HEIGHT,
    MAP_WIDTH,
    project,
    render_map,
)
from dashboard.maps.tiles import capture_gmaps_base
from dashboard.models import EtaKind, RouteEtaGroup
from dashboard.providers.route_geometry import Stop, fetch_route_geometry, select_probe_stops
from dashboard.providers.transit import CTB_STOPS, GMB_STOPS, KMB_STOPS, fetch_probe_etas

log = logging.getLogger(__name__)


# Compact display wording for marker labels: shorter than official termini
# and matching the ETA embed's shorthand.
_DESTINATION_SHORTHAND: dict[str, str] = {
    "tseung kwan o station": "TKO",
    "tseung kwan o": "TKO",
    "choi hung station": "Choi Hung",
    "choi hung station (lung cheung road)": "Choi Hung",
    "diamond hill station bus terminus": "Diamond Hill",
    "diamond hill station": "Diamond Hill",
    "clear water bay bus terminus": "Clear Water Bay",
    "hang hau village": "Hang Hau",
    "sai kung": "Sai Kung",
    "kwun tong (circular)": "Kwun Tong",
}


def _shorthand(destination: str) -> str:
    """Compact display wording for a destination string."""
    return _DESTINATION_SHORTHAND.get(
        destination.strip().lower(), destination.strip()
    )


def _destination_map(
    groups: list[RouteEtaGroup], route_lines: list[object]
) -> dict[tuple[str, str, str], str]:
    """Compact display destinations keyed by (operator-code, route, bound).

    Gate ETA rows now carry the OFFICIAL bound ("outbound"/"seq-1"/...), so
    each group's compact destination is applied to exactly its own direction
    — no gate-name heuristics, which failed for GMB raw-ID stop names and
    CTB's gate-less feed. The official line destination seeds every direction
    first as fallback; all wording passes through the shorthand normalizer.
    """
    operator_codes = {"KMB": "KMB", "Citybus": "CTB", "GMB": "GMB"}
    out: dict[tuple[str, str, str], str] = {}
    for line in route_lines:
        key = (str(line.operator), str(line.route), str(line.bound))
        destination = str(getattr(line, "destination", "") or "").strip()
        if destination:
            out[key] = _shorthand(destination)

    for group in groups or []:
        code = operator_codes.get(str(group.operator), str(group.operator))
        bound = getattr(group, "bound", None)
        if not bound:
            continue  # cannot place without an official direction
        key = (code, str(group.route), str(bound))
        if key in out:
            out[key] = _shorthand(str(group.destination))
    return out


@dataclass(frozen=True)
class _AuthoritativeEta:
    operator: str
    route: str
    bound: str
    index: int
    minutes: int
    kind: EtaKind
    authoritative: bool = True


def _authoritative_etas(
    groups: list[RouteEtaGroup], route_lines: list[object]
) -> list[_AuthoritativeEta]:
    """Convert mapped gate rows into exact route-line positions.

    Gate mappings are sourced from the transit provider's verified constants;
    no direction is inferred from a destination or stop name.
    """
    line_stops = {
        (str(line.operator), str(line.route), str(line.bound)): list(line.stops)
        for line in route_lines
    }
    out: list[_AuthoritativeEta] = []
    for group in groups:
        operator = {"Citybus": "CTB"}.get(str(group.operator), str(group.operator))
        bound = str(group.bound or "")
        if not bound:
            continue
        stop_id: str | None = None
        if operator == "KMB":
            stop_id = next(
                (spec["stop"] for spec in KMB_STOPS
                 if spec["route"] == group.route and spec["gate"] == group.gate),
                None,
            )
        elif operator == "CTB":
            stop_id = next(
                (spec["stop"] for spec in CTB_STOPS
                 if spec["route"] == group.route
                 and {"O": "outbound", "I": "inbound"}.get(spec["gate"]) == bound),
                None,
            )
        elif operator == "GMB":
            bound_seq = int(bound.removeprefix("seq-")) if bound.startswith("seq-") else None
            for candidate_stop, mappings in GMB_STOPS.items():
                if any(
                    route == group.route and gate == group.gate and seq == bound_seq
                    for route, _dest, gate, _route_id, seq in mappings
                ):
                    stop_id = str(candidate_stop)
                    break
        if stop_id is None:
            continue
        stops = line_stops.get((operator, str(group.route), bound), [])
        index = next(
            (index for index, stop in enumerate(stops) if str(stop.stop_id) == stop_id),
            None,
        )
        if index is None:
            continue
        for row in group.rows:
            if row.minutes is not None and row.kind is not EtaKind.UNAVAILABLE:
                out.append(
                    _AuthoritativeEta(
                        operator, str(group.route), bound, index,
                        max(0, int(row.minutes)), row.kind,
                    )
                )
    return out


async def fetch_traffic_map(
    client,
    groups: list[RouteEtaGroup] | None = None,
    cache_dir: str = ".cache",
    affected_road_paths: list[list[tuple[float, float]]] | None = None,
) -> tuple[bytes | None, list[object]]:
    """Capture the Google base map and render estimated bus/stop markers.

    ``affected_road_paths`` contains only matched OSM road polylines from
    current TD traffic news; it never represents a whole transit route.
    """
    base_image_task = asyncio.create_task(capture_gmaps_base(cache_dir=cache_dir))
    public_stops: list[Stop] = []
    route_lines: list[object] = []
    # Disk-backed geometry is immediate; a cold cache is independently bounded
    # by the provider.  Start it alongside browser capture so routing trouble
    # can never delay launching the required Google Maps screenshot.
    geometry_task = asyncio.create_task(fetch_route_geometry(client, cache_dir=cache_dir))
    try:
        base_image = await base_image_task
        geometry = await geometry_task
        public_stops = geometry.stops
        route_lines = list(geometry.routes)
    except Exception as exc:  # noqa: BLE001
        log.warning("public stop geometry unavailable: %s", type(exc).__name__)
    finally:
        # If the parent is cancelled (or capture/geometry fails), do not leave
        # the sibling task running beyond this map operation.
        siblings = [task for task in (base_image_task, geometry_task) if not task.done()]
        for task in siblings:
            task.cancel()
        if siblings:
            await asyncio.gather(*siblings, return_exceptions=True)
    if not public_stops:
        public_stops = [
            Stop("HKUST-N", "HKUST North Gate", 22.338678, 114.261946),
            Stop("HKUST-S", "HKUST South Gate", 22.333360, 114.262881),
        ]

    estimates: list[BusEstimate] = []
    if route_lines:
        probes = select_probe_stops(route_lines)
        # Keep this initialized: a provider failure should still leave the
        # required Google base map renderable without estimated markers.
        probe_etas = []
        try:
            probe_etas = await fetch_probe_etas(client, probes)
            estimates = estimate_bus_positions(
                probe_etas,
                route_lines,
                _destination_map(groups or [], route_lines),
                _authoritative_etas(groups or [], route_lines),
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("probe ETA estimation failed: %s", type(exc).__name__)
        log.info(
            "map markers: %d probe ETAs -> %d bus estimates",
            len(probe_etas),
            len(estimates),
        )
    try:
        webp = await asyncio.to_thread(
            render_map,
            estimates,
            cache_dir,
            public_stops,
            route_lines,
            base_image,
            affected_road_paths or [],
        )
        return webp, []
    except Exception as exc:  # noqa: BLE001
        import traceback

        log.warning(
            "map rendering failed: %s\n%s",
            type(exc).__name__,
            traceback.format_exc(limit=6),
        )
        return None, []


__all__ = [
    "MAP_HEIGHT",
    "MAP_WIDTH",
    "BusEstimate",
    "estimate_bus_positions",
    "fetch_traffic_map",
    "project",
]
