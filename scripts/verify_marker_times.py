"""Live source-to-marker timing diagnostic (no raw upstream data is printed)."""

from __future__ import annotations

import argparse
import asyncio
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import aiohttp
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.config import Settings  # noqa: E402
from dashboard.http import HttpClient  # noqa: E402
from dashboard.maps import _authoritative_etas, _destination_map  # noqa: E402
from dashboard.maps.positions import _label_for, _path_segment_length, estimate_bus_positions  # noqa: E402
from dashboard.providers.route_geometry import fetch_route_geometry, select_probe_stops  # noqa: E402
from dashboard.providers.transit import CTB_STOPS, GMB_STOPS, KMB_STOPS, fetch_probe_etas, fetch_transit_etas  # noqa: E402


def _operator(value: object) -> str:
    return {"Citybus": "CTB", "Operator.CITYBUS": "CTB"}.get(str(value), str(value))


def _gate_specs() -> dict[tuple[str, str, str], tuple[str, str]]:
    out: dict[tuple[str, str, str], tuple[str, str]] = {}
    for item in KMB_STOPS:
        out[("KMB", item["route"], {"S": "outbound", "N": "inbound"}[item["gate"]])] = (item["stop"], item["dest"])
    for item in CTB_STOPS:
        out[("CTB", item["route"], {"O": "outbound", "I": "inbound"}[item["gate"]])] = (item["stop"], item["dest"])
    for stop, routes in GMB_STOPS.items():
        for route, dest, _gate, _route_id, seq in routes:
            out[("GMB", route, f"seq-{seq}")] = (str(stop), dest)
    return out


@dataclass(frozen=True)
class Projection:
    position: float
    distance_m: float
    heading: float


def _project(line, lat: float, lon: float) -> Projection | None:
    """Project onto a route using runtime's metre-valued segment metric."""
    path, offsets = list(line.path), list(line.stop_offsets)
    if len(path) < 2 or len(offsets) != len(line.stops):
        return None
    best: tuple[float, float, float] | None = None
    travelled = 0.0
    for a, b in zip(path, path[1:], strict=False):
        dy, dx = b[0] - a[0], b[1] - a[1]
        denom = dy * dy + dx * dx
        t = 0.0 if denom == 0 else max(0.0, min(1.0, ((lat - a[0]) * dy + (lon - a[1]) * dx) / denom))
        q = (a[0] + t * dy, a[1] + t * dx)
        segment = _path_segment_length(a, b)
        distance = _path_segment_length((lat, lon), q)
        if best is None or distance < best[0]:
            best = (distance, travelled + t * segment, math.atan2(dy, dx))
        travelled += segment
    if best is None:
        return None
    distance_m, along, heading = best
    for index, (start, end) in enumerate(zip(offsets, offsets[1:], strict=False)):
        if start <= along <= end and end > start:
            return Projection(index + (along - start) / (end - start), distance_m, heading)
    return Projection(float(len(offsets) - 1), distance_m, heading)


def _angle_delta(a: float, b: float) -> float:
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


def _assign_markers(estimates, lines, destinations):
    """Assign markers, preferring estimator metadata over visual inference."""
    result: dict[tuple[str, str, str], list[tuple[object, float]]] = {}
    for marker in estimates:
        code = _operator(marker.operator)
        route = str(getattr(marker, "route", "") or marker.label.split(" ", 1)[0])
        bound = str(getattr(marker, "bound", "") or "")
        exact = [line for line in lines if _operator(line.operator) == code and line.route == route and (not bound or line.bound == bound)]
        if len(exact) == 1 and getattr(marker, "position", None) is not None:
            line = exact[0]
            result.setdefault((code, line.route, line.bound), []).append((marker, float(marker.position)))
            continue
        ranked = []
        for line in lines:
            if _operator(line.operator) != code or line.route != route or (bound and line.bound != bound):
                continue
            projection = _project(line, marker.lat, marker.lon)
            if projection:
                score = projection.distance_m + 50.0 * _angle_delta(marker.heading, projection.heading)
                ranked.append((score, line, projection))
        if ranked:
            _score, line, projection = min(ranked, key=lambda item: item[0])
            result.setdefault((code, line.route, line.bound), []).append((marker, projection.position))
    return result


def _one_to_one(source: list[int], marker: list[int]) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Pair ordered departures without reusing a marker."""
    left, right = sorted(source), sorted(marker)
    pairs = list(zip(left, right, strict=False))
    return pairs, left[len(pairs):], right[len(pairs):]


def _probe_gate_anchors(probe_etas, key, gate_index: int, stop_count: int) -> list[int]:
    """Convert downstream probe announcements into implied gate ETAs."""
    anchors = []
    for eta in probe_etas:
        if eta.minutes is None or (_operator(eta.operator), eta.route, eta.bound) != key:
            continue
        raw_position = eta.index - eta.minutes / 2.0
        if 0 <= raw_position <= stop_count - 1 and raw_position <= gate_index + 0.5:
            anchors.append(round((gate_index - raw_position) * 2))
    return sorted(anchors)


async def _run(cycles: int) -> int:
    settings = Settings.from_env(require_keys=False)
    async with aiohttp.ClientSession() as session:
        client = HttpClient(session, timeout_seconds=settings.http_timeout_seconds)
        try:
            geometry = await fetch_route_geometry(client, cache_dir=settings.cache_dir)
            lines = list(geometry.routes)
            probes = select_probe_stops(lines)
            groups, _latest, failed = await fetch_transit_etas(client)
            probe_etas = []
            for _ in range(max(1, cycles)):
                probe_etas = await fetch_probe_etas(client, probes)
            destinations = _destination_map(groups, lines)
            authoritative = _authoritative_etas(groups, lines)
            estimates = estimate_bus_positions(probe_etas, lines, destinations, authoritative)
        except Exception as exc:
            print(f"STATUS INCONCLUSIVE reason={type(exc).__name__}")
            return 2

    lines_by_key = {(_operator(line.operator), line.route, line.bound): line for line in lines}
    assigned = _assign_markers(estimates, lines, destinations)
    checks = 0
    failures: list[str] = []
    for key, line in lines_by_key.items():
        spec = _gate_specs().get(key)
        if not spec:
            continue
        stop_id, _destination = spec
        exact_authoritative = [eta for eta in authoritative if (eta.operator, eta.route, eta.bound) == key]
        excluded_undeparted = sum(1 for eta in exact_authoritative if eta.minutes > 0 and eta.index - eta.minutes / 2.0 < 0)
        source = [eta for eta in exact_authoritative if not (eta.minutes > 0 and eta.index - eta.minutes / 2.0 < 0)]
        gate_indices = [i for i, stop in enumerate(line.stops) if stop.stop_id == stop_id]
        marker_rows = assigned.get(key, [])
        if not source or not gate_indices or not marker_rows:
            continue
        gate_index = gate_indices[0]
        upstream = [(marker, pos) for marker, pos in marker_rows if pos <= gate_index + 0.5]
        if not upstream:
            continue
        source_minutes = sorted(int(eta.minutes) for eta in source)
        marker_minutes = sorted(round((gate_index - pos) * 2) for _marker, pos in upstream)
        pairs, unmatched_source, unmatched_markers = _one_to_one(source_minutes, marker_minutes)
        deltas = [abs(source_minute - marker_minute) for source_minute, marker_minute in pairs]
        probe_anchors = _probe_gate_anchors(probe_etas, key, gate_index, len(line.stops))
        checks += 1
        label_ok = all(marker.label == _label_for(key[1], key[0], key[2], pos, line.stops, destinations) for marker, pos in marker_rows)
        timing_ok = bool(deltas) and max(deltas) <= 2 and not unmatched_source
        status = "PASS" if timing_ok and label_ok else "FAIL"
        print(f"{key[0]:5} {key[1]:4} {key[2]:8} source={source_minutes} marker={marker_minutes} matched={len(pairs)}/{len(source_minutes)} unmatched_source={unmatched_source} unmatched_marker={unmatched_markers} excluded_undeparted={excluded_undeparted} probe_anchors={probe_anchors} max_delta={max(deltas) if deltas else 'NA'}m label={'ok' if label_ok else 'BAD'} unreliable=NA {status}")
        if status == "FAIL":
            failures.append("/".join(key))

    if failed or checks == 0:
        print("STATUS INCONCLUSIVE reason=" + ("upstream-failures" if failed else "insufficient-current-ETAs"))
        return 2
    if failures:
        print("STATUS FAIL routes=" + ",".join(failures))
        return 1
    print(f"STATUS PASS checks={checks} reliability=inconclusive")
    return 0


def _self_test() -> int:
    line = SimpleNamespace(path=[(0.0, 0.0), (0.0, 0.001)], stop_offsets=[0.0, 111.0], stops=[SimpleNamespace(stop_id="a"), SimpleNamespace(stop_id="b")])
    projection = _project(line, 0.0, 0.0005)
    assert projection and abs(projection.position - 0.5) < 0.05, projection
    assert _operator("Citybus") == "CTB"
    pairs, unmatched_source, unmatched_markers = _one_to_one([3, 8, 12], [2, 9])
    assert pairs == [(3, 2), (8, 9)] and unmatched_source == [12] and not unmatched_markers
    line.operator, line.route, line.bound = "CTB", "792M", "inbound"
    marker = SimpleNamespace(operator="Citybus", route="792M", bound="inbound", position=0.25, label="792M TKO", lat=0.0, lon=0.0, heading=0.0)
    assigned = _assign_markers([marker], [line], {})
    assert list(assigned) == [("CTB", "792M", "inbound")] and assigned[("CTB", "792M", "inbound")][0][1] == 0.25
    downstream = SimpleNamespace(operator="CTB", route="792M", bound="inbound", index=3, minutes=4)
    assert _probe_gate_anchors([downstream], ("CTB", "792M", "inbound"), 2, 5) == [2]
    print("STATUS PASS self-test=meters,operator,projection,metadata,no-reuse,downstream-anchor")
    return 0


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return _self_test()
    if args.cycles < 1:
        parser.error("--cycles must be >= 1")
    return asyncio.run(_run(args.cycles))


if __name__ == "__main__":
    raise SystemExit(main())
