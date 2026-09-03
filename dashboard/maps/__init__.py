"""Small public API for dashboard map generation."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from dashboard.maps.marker_audit import audit_gmb_marker_pairs, audit_marker_positions
from dashboard.maps.positions import BusEstimate, estimate_bus_positions
from dashboard.maps.renderer import (
    MAP_HEIGHT,
    MAP_WIDTH,
    project,
    render_map,
)
from dashboard.maps.tiles import capture_gmaps_base, shutdown_gmaps_browser
from dashboard.maps.tracker import MarkerTracker
from dashboard.models import EtaKind, RouteEtaGroup
from dashboard.providers.route_geometry import Stop, fetch_route_geometry, select_probe_stops
from dashboard.providers.transit import CTB_STOPS, GMB_STOPS, KMB_STOPS, fetch_probe_snapshot

log = logging.getLogger(__name__)
_frame_counter = 0
_logged_marker_issue_keys: dict[tuple[object, ...], None] = {}
_MARKER_ISSUE_KEY_LIMIT = 256


def _first_marker_issue(key: tuple[object, ...]) -> bool:
    """Return whether this bounded issue detail should be logged this run."""
    if key in _logged_marker_issue_keys:
        return False
    _logged_marker_issue_keys[key] = None
    if len(_logged_marker_issue_keys) > _MARKER_ISSUE_KEY_LIMIT:
        del _logged_marker_issue_keys[next(iter(_logged_marker_issue_keys))]
    return True


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
    tracker: MarkerTracker | None = None,
) -> tuple[bytes | None, list[object]]:
    """Capture the Google base map and render estimated bus/stop markers.

    ``affected_road_paths`` contains only matched OSM road polylines from
    current TD traffic news; it never represents a whole transit route.
    """
    global _frame_counter
    _frame_counter += 1
    frame_id = _frame_counter
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
            mandatory = {
                str(spec["stop"]) for spec in KMB_STOPS
            } | {str(spec["stop"]) for spec in CTB_STOPS}
            mandatory |= {str(stop_id) for stop_id in GMB_STOPS}
            probes = (select_probe_stops(route_lines, mandatory_stop_ids=mandatory)
                      if route_lines else [])
            priority_provider = getattr(tracker, "poll_priorities", None)
            priorities = priority_provider() if callable(priority_provider) else None
            probe_task = asyncio.create_task(
                fetch_probe_snapshot(client, probes, priorities=priorities)
            )
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
        audit_estimates: list[BusEstimate] = []
        authoritative = _authoritative_etas(groups or [], route_lines)
        probe_etas = []
        if probe_task is not None:
            if isinstance(probe_result, BaseException):
                log.warning("probe ETA estimation failed: %s", type(probe_result).__name__)
            else:
                snapshot = probe_result
                positioning_rows = getattr(snapshot, "positioning_rows", None)
                probe_etas = list(
                    positioning_rows
                    if positioning_rows is not None
                    else (getattr(snapshot, "rows", ()) or ())
                )
                observed_positions: dict[tuple[str, str, str], set[int]] = {}
                for operator, route, bound, index in getattr(
                        snapshot, "positioning_checkpoints", ()):
                    observed_positions.setdefault((operator, route, bound), set()).add(index)
                try:
                    estimates = estimate_bus_positions(
                        probe_etas,
                        route_lines,
                        _destination_map(groups or [], route_lines),
                        authoritative,
                        observed_checkpoint_indices=observed_positions or {
                            tuple(route.route_key): route.observed_checkpoint_indices
                            for route in getattr(snapshot, "complete_routes", ())
                        },
                    )
                    audit_estimates = list(estimates)
                except Exception as exc:  # noqa: BLE001
                    log.warning("probe ETA estimation failed: %s", type(exc).__name__)
                    estimates = []
                if tracker is not None:
                    try:
                        estimates = await tracker.update(snapshot, estimates, route_lines)
                    except Exception as exc:  # noqa: BLE001
                        log.warning("marker tracker unavailable: %s", type(exc).__name__)
        try:
            audit = audit_marker_positions(
                probe_etas, authoritative, audit_estimates, route_lines,
                frame_id=frame_id, seed=frame_id,
            )
            marker_pairs = audit.get("gmb_marker_pairs", ())
            log.info(
                "marker audit frame=%d checks=%d inconclusive=%d issues=%d "
                "marker_pairs=%d markers=%d observed_checkpoints=%d "
                "audited_checkpoints=%d uncovered_checkpoints=%d "
                "observed_rows=%d audited_rows=%d uncovered_rows=%d status=%s",
                frame_id,
                len(audit["checks"]),
                audit["stats"].get("inconclusive", 0),
                len(audit["issues"]),
                len(marker_pairs),
                audit["stats"]["markers"],
                audit["stats"].get("observed_checkpoints", 0),
                audit["stats"].get("audited_checkpoints", 0),
                audit["stats"].get("uncovered_checkpoints", 0),
                audit["stats"].get("observed_probe_rows", 0),
                audit["stats"].get("audited_probe_rows", 0),
                audit["stats"].get("uncovered_probe_rows", 0),
                "pass" if audit["ok"] and not marker_pairs else "fail",
            )
            for issue in audit["issues"]:
                detail = issue.get("detail") or {}
                match = detail.get("match") or {}
                issue_key = (
                    tuple(issue.get("key", ())),
                    issue.get("kind", ""),
                    detail.get("checkpoint", detail.get("gate_index", "-")),
                    detail.get("reason", "one-to-one mismatch"),
                )
                if not _first_marker_issue(issue_key):
                    continue
                log.warning(
                    "marker audit mismatch frame=%d route=%s kind=%s "
                    "checkpoint=%s unmatched_source=%s source_tokens=%s "
                    "unmatched_marker=%s marker_id=%s position=%s "
                    "marker_sources=%s reason=%s",
                    frame_id,
                    "/".join(issue["key"]),
                    issue["kind"],
                    detail.get("checkpoint", detail.get("gate_index", "-")),
                    match.get("unmatched_source_values", []),
                    match.get("unmatched_source_observations", []),
                    match.get("unmatched_marker_values", []),
                    detail.get("marker_id", "-"),
                    detail.get("position", "-"),
                    detail.get("source_observations", []),
                    detail.get("reason", "one-to-one mismatch"),
                )
                log.warning(
                    "marker audit context frame=%d route=%s gate_rows=%s "
                    "checkpoint_rows=%s route_markers=%s association_rows=%s",
                    frame_id,
                    "/".join(issue["key"]),
                    detail.get("gate_rows", []),
                    detail.get("checkpoint_rows", []),
                    detail.get("route_markers", []),
                    detail.get("association_rows", []),
                )
            for pair in marker_pairs:
                common = pair.get("common_stops", ())
                pair_key = (
                    tuple(pair.get("key", ())),
                    "gmb-marker-pair",
                    pair.get("common_stop_index", "-"),
                    pair.get("classification", "nearby"),
                )
                if not _first_marker_issue(pair_key):
                    continue
                log.warning(
                    "GMB marker pair frame=%d route=%s class=%s distance=%.2fpx "
                    "markers=%s route_positions=%s marker_sources=%s common=%s",
                    frame_id,
                    "/".join(pair.get("key", ())),
                    pair.get("classification", "nearby"),
                    pair.get("pixel_distance", 0.0),
                    pair.get("marker_ids", []),
                    pair.get("route_positions", []),
                    pair.get("marker_source_observations", []),
                    common[:2],
                )
        except Exception as exc:  # audit must never prevent rendering
            log.warning("marker audit unavailable frame=%d: %s", frame_id, type(exc).__name__)
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
    "MarkerTracker",
    "estimate_bus_positions",
    "audit_marker_positions",
    "audit_gmb_marker_pairs",
    "fetch_traffic_map",
    "shutdown_gmaps_browser",
    "project",
]
