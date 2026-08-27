"""Estimated bus positions from probe-stop ETAs on official route geometry.

Each tracked direction is an official ordered stop sequence with a validated
HKeMobility road path.  An ETA of ``m`` minutes at the official stop with
index ``i`` places a bus roughly ``m / 2`` stops upstream (two minutes per
stop), clamped to the section between stop ``i - 1`` and stop ``i`` so one
ETA can never smear an estimate across several stops.  Positions are
arclength-interpolated on the official line; estimates landing on either
terminus are dropped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from dashboard.models import EtaKind, Operator


@dataclass(frozen=True)
class BusEstimate:
    """One estimated vehicle: label, road point, operator, travel heading.

    ``unreliable`` marks estimates derived from timetable ('scheduled')
    observations rather than live tracking: the position is plausible but the
    operator has not confirmed the vehicle, so the marker renders paler with
    a dashed outline.
    """

    label: str
    lat: float
    lon: float
    operator: Operator
    heading: float
    unreliable: bool = False
    route: str = ""
    bound: str = ""
    position: float | None = None


MINUTES_PER_STOP = 2.0

# Route-geometry operator codes -> dashboard Operator enum values.
_OPERATOR_BY_CODE = {
    "KMB": Operator.KMB,
    "CTB": Operator.CITYBUS,
    "GMB": Operator.GMB,
}


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


def estimate_bus_positions(
    probe_etas,
    route_lines,
    destinations: dict[tuple[str, str, str], str] | None = None,
    authoritative_etas=None,
) -> list[BusEstimate]:
    """Interpolate probe ETAs into per-section bus positions.

    ``probe_etas`` items need ``operator/route/bound/index/minutes/kind``;
    ``route_lines`` are geometry objects exposing ``route/operator/bound``,
    ``stops``, ``path``, and ``stop_offsets``.  ``destinations`` optionally
    maps ``(operator, route, bound)`` to the compact display destination used
    by the ETA embed; without it the official terminus name is used.
    """
    destination_map = destinations or {}
    lines_by_key: dict[tuple[str, str, str], object] = {}
    for line in route_lines:
        key = (str(line.operator), str(line.route), str(line.bound))
        stops = list(getattr(line, "stops", ()))
        path = list(getattr(line, "path", ()))
        offsets = list(getattr(line, "stop_offsets", ()))
        if len(stops) < 3 or len(path) < 2 or len(offsets) != len(stops):
            continue
        # Cache the line's own official destination as a label fallback; some
        # feeds store raw stop IDs in stop names, so the terminus name is the
        # next best source after the caller's destination map.
        destination_map.setdefault(
            key, str(getattr(line, "destination", "") or "").strip()
        )
        lines_by_key[key] = line

    # Operators publish ETA rows for every upcoming stop of a vehicle, and
    # far-future rows are interpolated FROM THE TIMETABLE (~2 min per stop).
    # One real bus therefore leaves a LADDER of implied positions
    # (`index - minutes / MINUTES_PER_STOP`) rising by ~1 stop per announcing
    # stop, even when flagged realtime. The truest current location is the
    # ladder's MAXIMUM implied position: the latest announcement, closest
    # schedule drift. Separate buses appear as separate ladders offset by the
    # headway, so: sort implied positions per direction, merge neighbours
    # within LADDER_GAP_STOPS into one vehicle, anchor each vehicle at its
    # maximum implied position.
    #
    # A ladder made ONLY of 'scheduled' rows is a timetable departure: once
    # its ETA has matured (minutes near 0) the bus is plausibly on the road,
    # so it renders — flagged UNRELIABLE (paler, dashed outline). A ladder
    # containing ANY realtime row is a live vehicle and renders normally.
    LADDER_GAP_STOPS = 2.2

    authoritative_keys = {
        (str(eta.operator), str(eta.route), str(eta.bound), int(eta.index))
        for eta in (authoritative_etas or [])
        if eta.minutes is not None
    }
    by_direction: dict[
        tuple[str, str, str], list[tuple[float, int, bool, bool]]
    ] = {}
    for eta, is_authoritative in [
        *((eta, False) for eta in probe_etas),
        *((eta, True) for eta in (authoritative_etas or [])),
    ]:
        if eta.minutes is None:
            continue
        if eta.kind is EtaKind.UNAVAILABLE:
            continue
        key = (str(eta.operator), str(eta.route), str(eta.bound))
        if key not in lines_by_key:
            continue
        idx = int(eta.index)
        if not is_authoritative and (
            str(eta.operator), str(eta.route), str(eta.bound), idx
        ) in authoritative_keys:
            continue
        stops_count = len(list(lines_by_key[key].stops))
        if not 0 <= idx <= stops_count - 1:
            continue
        minutes = max(0.0, float(eta.minutes))
        raw_position = idx - minutes / MINUTES_PER_STOP
        # The estimate must lie on the route. Undeparted buses never render:
        # an ETA > 0 AT the terminus (implied position < 0) means the bus has
        # not left yet. Only a matured 0-minute reading at the terminus puts
        # the bus ON it (about to depart) — same rule as 0 minutes on top of
        # HKUST.
        if minutes > 0 and raw_position < 0:
            continue
        if not 0 <= raw_position <= stops_count - 1:
            continue
        by_direction.setdefault(key, []).append(
            (raw_position, idx, eta.kind is EtaKind.SCHEDULED, is_authoritative)
        )

    candidates: dict[
        tuple[str, str, str, int],
        list[tuple[float, int, bool, frozenset[int], bool]],
    ] = {}
    for key, rows in sorted(by_direction.items()):
        operator_name, route, bound = key
        stops_count = len(list(lines_by_key[key].stops))
        # Assign each rung deterministically to the nearest compatible ladder.
        # A ladder can contain at most one ETA for a given stop: otherwise
        # same-stop ETAs can chain transitively through neighbouring rows and
        # collapse several actual departures into one vehicle.
        rows.sort(key=lambda row: (row[0], row[1], row[2]))
        ladders: list[list[tuple[float, int, bool, bool]]] = []
        for row in rows:
            position, stop_index, _scheduled, _authoritative = row
            compatible = [
                (abs(position - ladder[-1][0]), ladder_index, ladder)
                for ladder_index, ladder in enumerate(ladders)
                if stop_index not in {rung[1] for rung in ladder}
                and abs(position - ladder[-1][0]) <= LADDER_GAP_STOPS
            ]
            if compatible:
                _distance, _ladder_index, ladder = min(
                    compatible, key=lambda item: (item[0], item[1])
                )
                ladder.append(row)
            else:
                ladders.append([row])

        for ladder in ladders:
            # A direct gate ETA overrides downstream inference. Otherwise use
            # the maximum implied position, closest to the gate ETA. The ladder
            # is unreliable only when
            # EVERY rung is a timetable row (no live confirmation anywhere).
            direct = [rung for rung in ladder if rung[3]]
            position = max(rung[0] for rung in (direct or ladder))
            if position < 0 or position > stops_count - 1:
                continue
            unreliable = all(rung[2] for rung in ladder)
            # A lone scheduled probe row is not enough evidence to reconstruct
            # a vehicle on an infrequent route: it can be a stale timetable
            # departure rather than a bus currently in service.  Require
            # either live evidence, a direct/authoritative gate rung, or
            # corroboration from two distinct probe stops.  Do not apply this
            # to realtime rows, even when only one stop reported the vehicle.
            if unreliable and not direct and len({rung[1] for rung in ladder}) < 2:
                continue
            section = min(math.floor(position), stops_count - 2)
            bucket = candidates.setdefault(
                (operator_name, route, bound, section), []
            )
            bucket.append(
                (position, round(position * 1000), unreliable,
                 frozenset(rung[1] for rung in ladder), bool(direct))
            )

    # A missing middle rung (probe fetch failure) can split one bus's ladder
    # in two, yielding two markers a section apart that alternate between
    # frames. Collapse vehicle anchors that sit within one stop of each other
    # in the same direction, preferring the reliable (realtime-evidenced)
    # anchor and then the earlier position.
    vehicles: dict[
        tuple[str, str, str], list[tuple[float, bool, frozenset[int], bool]]
    ] = {}
    for (operator_name, route, bound, _section), entries in sorted(candidates.items()):
        for position, _scaled, unreliable, stop_indices, authoritative in entries:
            vehicles.setdefault((operator_name, route, bound), []).append(
                (position, unreliable, stop_indices, authoritative)
            )

    candidates2: dict[tuple[str, str, str, int], list[tuple[float, bool]]] = {}
    for key, anchors in sorted(vehicles.items()):
        # Keep spatial order while forming clusters.  Sorting by reliability
        # first can make a distant reliable anchor absorb an upstream
        # scheduled anchor, and makes the result depend on feed ordering.
        anchors.sort(key=lambda item: item[0])
        clusters: list[list[tuple[float, bool, frozenset[int], bool]]] = [[anchors[0]]]
        for anchor in anchors[1:]:
            cluster_stops = set().union(*(item[2] for item in clusters[-1]))
            # Shared stop provenance means that one source snapshot exposed
            # both departures simultaneously.  Keep them distinct even when
            # their inferred positions are close: upstream stops stop listing
            # a bus after it passes, so far-away upstream ETAs do not refute a
            # close pair confirmed at downstream stops.
            if (abs(anchor[0] - clusters[-1][-1][0]) <= 1.0
                    and cluster_stops.isdisjoint(anchor[2])):
                clusters[-1].append(anchor)
            else:
                clusters.append([anchor])
        operator_name, route, bound = key
        stops_count = len(list(lines_by_key[key].stops))
        for cluster in clusters:
            # In a mixed cluster, use the latest realtime anchor.  For a
            # scheduled-only cluster, use the latest scheduled anchor.
            direct = [anchor for anchor in cluster if anchor[3]]
            reliable = [anchor for anchor in (direct or cluster) if not anchor[1]]
            selected = reliable or direct or cluster
            position = max(anchor[0] for anchor in selected)
            unreliable = all(anchor[1] for anchor in cluster)
            section = min(math.floor(position), stops_count - 2)
            bucket = candidates2.setdefault(
                (operator_name, route, bound, section), []
            )
            bucket.append((0.0, round(position * 1000), unreliable))

    estimates: list[BusEstimate] = []
    for (operator_name, route, bound, section), entries in sorted(candidates2.items()):
        line = lines_by_key[(operator_name, route, bound)]
        stops = list(line.stops)
        path = list(line.path)
        offsets = list(line.stop_offsets)
        for _best_distance, scaled_position, unreliable in entries:
            position = scaled_position / 1000
            fraction = position - section
            target_offset = offsets[section] + (offsets[section + 1] - offsets[section]) * fraction
            located = _point_at_path_offset(path, target_offset)
            if located is None:
                continue
            lat, lon, heading = located
            operator = _OPERATOR_BY_CODE.get(operator_name)
            if operator is None:
                continue
            label = _label_for(
                route, operator_name, bound, position, stops, destination_map
            )
            estimates.append(
                BusEstimate(
                    label,
                    lat,
                    lon,
                    operator,
                    heading,
                    unreliable=unreliable,
                    route=route,
                    bound=bound,
                    position=position,
                )
            )
    return estimates


def _label_for(
    route: str,
    operator_name: str,
    bound: str,
    position: float,
    stops: list,
    destination_map: dict[tuple[str, str, str], str],
) -> str:
    """Marker wording matches the compact ETA embed; circular 104 splits sides."""
    if operator_name == "GMB" and route == "104":
        terminus = "Kwun Tong" if position < 12 else "HKUST"
        return f"104 {terminus}"
    destination = destination_map.get((operator_name, route, bound))
    if not destination:
        destination = getattr(stops[-1], "name", "") if stops else ""
    return f"{route} {destination}".strip()
