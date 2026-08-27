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
from dashboard.maps.tiles import capture_gmaps_base, shutdown_gmaps_browser
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
    "hang hau station public transport interchange": "Hang Hau",
    "po lam bus terminus": "Po Lam",
    "h.k.u.s.t. (north)": "HKUST",
    "ngau chi wan bbi - choi hung station": "Choi Hung",
    "mong kok station": "Mong Kok",
    "sai kung": "Sai Kung",
    "kwun tong (circular)": "Kwun Tong",
    "kwun tong(circular)": "Kwun Tong",
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
        if not destination:
            stops = list(getattr(line, "stops", ()) or ())
            destination = str(getattr(stops[-1], "name", "") or "").strip() if stops else ""
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
    operation_tasks: list[asyncio.Task[object]] = [base_image_task]
    public_stops: list[Stop] = []
    route_lines: list[object] = []
    # Disk-backed geometry is immediate; a cold cache is independently bounded
    # by the provider.  Start it alongside browser capture so routing trouble
    # can never delay launching the required Google Maps screenshot.
    geometry_task = asyncio.create_task(fetch_route_geometry(client, cache_dir=cache_dir))
    operation_tasks.append(geometry_task)
    base_image: object | None = None
    probe_task: asyncio.Task[object] | None = None
    try:
        # Geometry determines the probe set, so it is the first dependency we
        # await.  The browser capture remains in flight while a cold geometry
        # cache is populated, and the probe sweep starts as soon as the lines
        # are available rather than waiting for the screenshot.
        try:
            geometry = await geometry_task
        except Exception as exc:  # noqa: BLE001
            log.warning("public stop geometry unavailable: %s", type(exc).__name__)
        else:
            public_stops = geometry.stops
            route_lines = list(geometry.routes)
            probes = select_probe_stops(route_lines) if route_lines else []
            probe_task = asyncio.create_task(fetch_probe_etas(client, probes))
            operation_tasks.append(probe_task)

        # Collect independent work together.  return_exceptions keeps a
        # failed probe or capture from cancelling its sibling, while the
        # finally block below guarantees cleanup on parent cancellation.
        capture_result, probe_result = await asyncio.gather(
            base_image_task,
            probe_task if probe_task is not None else asyncio.sleep(0, result=None),
            return_exceptions=True,
        )
        if isinstance(capture_result, BaseException):
            log.warning("Google traffic map capture failed: %s", type(capture_result).__name__)
        else:
            base_image = capture_result

        estimates: list[BusEstimate] = []
        probe_etas = []
        if probe_task is not None:
            if isinstance(probe_result, BaseException):
                log.warning("probe ETA estimation failed: %s", type(probe_result).__name__)
            else:
                probe_etas = probe_result or []
                try:
                    estimates = estimate_bus_positions(
                        probe_etas,
                        route_lines,
                        _destination_map(groups or [], route_lines),
                        _authoritative_etas(groups or [], route_lines),
                    )
                except Exception as exc:  # noqa: BLE001
                    log.warning("probe ETA estimation failed: %s", type(exc).__name__)
                    estimates = []
        log.info(
            "map markers: %d probe ETAs -> %d bus estimates",
            len(probe_etas),
            len(estimates),
        )
    finally:
        # If the parent is cancelled (or an operation fails), do not leave a
        # sibling task running beyond this map operation.
        siblings = [task for task in operation_tasks if not task.done()]
        for task in siblings:
            task.cancel()
        if siblings:
            await asyncio.gather(*siblings, return_exceptions=True)
    if not public_stops:
        public_stops = [
            Stop("HKUST-N", "HKUST North Gate", 22.338678, 114.261946),
            Stop("HKUST-S", "HKUST South Gate", 22.333360, 114.262881),
        ]

    if base_image is None:
        return None, []
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
    "shutdown_gmaps_browser",
    "project",
]
