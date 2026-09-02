"""Small public-data continuity harness for MarkerTracker."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import aiohttp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from dashboard.config import Settings  # noqa: E402
from dashboard.http import HttpClient  # noqa: E402
from dashboard.maps import _authoritative_etas, _destination_map  # noqa: E402
from dashboard.maps.positions import estimate_bus_positions  # noqa: E402
from dashboard.maps.tracker import MarkerTracker  # noqa: E402
from dashboard.providers.route_geometry import (  # noqa: E402
    fetch_route_geometry,
    select_probe_stops,
    shutdown_background_refreshes,
)
from dashboard.providers.transit import (  # noqa: E402
    CTB_STOPS,
    GMB_STOPS,
    KMB_STOPS,
    fetch_probe_snapshot,
    fetch_transit_etas,
)

OBSERVATION_SPAN = 55.0
POSITION_EPSILON = 0.05
# Tracking extrapolation can shift adjacent gaps by a fraction of a stop while
# preserving marker identity/order; retain sensitivity to real spacing errors.
GAP_TOLERANCE = 1.0
EVIDENCE_TTL = 900.0
RouteKey = tuple[str, str, str]


def route_key(v):
    return (
        str(getattr(v, "operator_code", "") or getattr(v, "operator", "")).removeprefix(
            "Operator."
        ),
        str(getattr(v, "route", "")),
        str(getattr(v, "bound", "") or ""),
    )


def parse_route(v):
    p = tuple(x.strip() for x in v.split("/"))
    if len(p) != 3 or not all(p):
        raise ValueError("route must be OPERATOR/ROUTE/BOUND")
    return p


def validate_inputs(lines, probes):
    if not lines:
        raise ValueError("route filter matched zero geometry lines")
    if not probes:
        raise ValueError("no probe stops selected")


def _positions(items):
    out = {}
    for x in items:
        out.setdefault(route_key(x), []).append(float(getattr(x, "position", 0)))
    return {k: sorted(v) for k, v in out.items()}


def _tracks(items):
    out = {}
    for x in items:
        out.setdefault(route_key(x), []).append((int(x.track_id), float(getattr(x, "position", 0))))
    return {k: sorted(v, key=lambda z: z[1]) for k, v in out.items()}


def frame_record(snapshot, candidates, tracked, route_max=None, timestamp=None):
    return {
        "utc": timestamp or datetime.now(UTC).isoformat(),
        "generations": {
            tuple(x.route_key): (int(x.generation), x.collected_at.isoformat())
            for x in snapshot.complete_routes
        },
        "candidates": _positions(candidates),
        "tracks": _tracks(tracked),
        "route_max": route_max or {},
    }


def _json_safe_record(x):
    if isinstance(x, dict):
        return {
            ("/".join(k) if isinstance(k, tuple) else str(k)): _json_safe_record(v)
            for k, v in x.items()
        }
    if isinstance(x, (tuple, list)):
        return [_json_safe_record(v) for v in x]
    return x


def _evidence_state(state=None):
    if state is None:
        return {
        "last_generation_by_route": {}, "last_ids_by_route": {},
        "latest_complete_collected_at": {},
        "minute_baselines": {}, "minute_checks": {}, "gap_checks": {},
        "gap_inconclusive": {}, "lifecycle_inconclusive": {},
        }
    for key in ("last_generation_by_route", "last_ids_by_route", "latest_complete_collected_at",
                "minute_baselines", "minute_checks", "gap_checks", "gap_inconclusive",
                "lifecycle_inconclusive"):
        state.setdefault(key, {})
    return state


def _route_maps(record):
    return {key: {int(track): pos for track, pos in values}
            for key, values in record.get("tracks", {}).items()}


def compare_adjacent(old, new, state=None):
    """Check only observed facts; omissions never manufacture a generation."""
    state = _evidence_state(state)
    if old is None:
        for key, generation in new.get("generations", {}).items():
            state["last_generation_by_route"][key] = generation[0]
            state["latest_complete_collected_at"][key] = generation[1]
        # The first complete frame is valid direct spacing evidence.
        return compare_adjacent(new, new, state)
    for key, generation in old.get("generations", {}).items():
        state["last_generation_by_route"].setdefault(key, generation[0])
        state["latest_complete_collected_at"].setdefault(key, generation[1])
    issues, checks = [], 0
    old_tracks, new_tracks = _route_maps(old), _route_maps(new)
    old_routes = {track: key for key, values in old_tracks.items() for track in values}
    new_routes = {track: key for key, values in new_tracks.items() for track in values}
    for track, old_key in old_routes.items():
        if track in new_routes and new_routes[track] != old_key:
            issues.append({"kind": "identity_route_change", "track_id": track,
                           "route": old_key, "new_route": new_routes[track]})
    keys = set(old_tracks) | set(new_tracks) | set(old.get("generations", {})) | set(new.get("generations", {}))
    for key in keys:
        a, b = old_tracks.get(key, {}), new_tracks.get(key, {})
        current_generation = new.get("generations", {}).get(key)
        last_generation = state["last_generation_by_route"].get(key)
        generation_changed = bool(current_generation and last_generation is not None
                                   and current_generation[0] != last_generation)
        if current_generation:
            # Compare against the last complete generation, including through outages.
            if last_generation is None or generation_changed:
                state["last_generation_by_route"][key] = current_generation[0]
                state["latest_complete_collected_at"][key] = current_generation[1]
            else:
                if set(a) != set(b):
                    issues.append({"kind": "identity_change_without_generation", "route": key})
                if len(a) != len(b):
                    issues.append({"kind": "cardinality_without_generation", "route": key})
        else:
            removed, added = set(a) - set(b), set(b) - set(a)
            if added:
                issues.append({"kind": "identity_change_during_omission", "route": key})
            if removed and not added:
                maximum = new.get("route_max", {}).get(key, float("inf"))
                stamp = state["latest_complete_collected_at"].get(key)
                age = ((datetime.fromisoformat(new["utc"]) - datetime.fromisoformat(stamp)).total_seconds()
                       if stamp else -1)
                if any(a[track] < maximum - POSITION_EPSILON for track in removed):
                    issues.append({"kind": "identity_change_during_omission", "route": key})
                elif age < EVIDENCE_TTL:
                    state["lifecycle_inconclusive"][key] = state["lifecycle_inconclusive"].get(key, 0) + 1
        for track, position in a.items():
            if track in b and b[track] + POSITION_EPSILON < position:
                issues.append({"kind": "backward", "route": key, "track_id": track})
        common = [track for track in a if track in b]
        if len(common) > 1 and common != [track for track in b if track in a]:
            issues.append({"kind": "identity_order_crossing", "route": key})
        # Gap evidence is a property of this complete frame, never an inferred match.
        candidates = new.get("candidates", {}).get(key, ())
        if current_generation and len(candidates) == len(b) >= 2:
            checks += len(candidates) - 1
            state["gap_checks"][key] = state["gap_checks"].get(key, 0) + len(candidates) - 1
            if any(abs((candidates[i + 1] - candidates[i]) -
                       (tuple(b.values())[i + 1] - tuple(b.values())[i])) > GAP_TOLERANCE
                   for i in range(len(candidates) - 1)):
                issues.append({"kind": "spacing_mismatch", "route": key})
        elif current_generation and len(candidates) != len(b):
            state["gap_inconclusive"][key] = state["gap_inconclusive"].get(key, 0) + 1
    return issues, checks


def minute_checks(old, new, state=None):
    issues = []
    checks = 0
    state = _evidence_state(state)
    elapsed = (datetime.fromisoformat(new["utc"]) - datetime.fromisoformat(old["utc"])).total_seconds()
    for key, a in old.get("tracks", {}).items():
        if elapsed < OBSERVATION_SPAN:
            continue
        bm = dict(new.get("tracks", {}).get(key, ()))
        for i, p in a:
            if i not in bm:
                continue
            q = bm[i]
            checks += 1
            if q + POSITION_EPSILON < p:
                issues.append({"kind": "minute_backward", "route": key, "track_id": i})
            elif q <= p + POSITION_EPSILON and q < new.get("route_max", {}).get(key, float("inf")) - POSITION_EPSILON:
                issues.append({"kind": "minute_stalled", "route": key, "track_id": i})
            state["minute_checks"][key] = state["minute_checks"].get(key, 0) + 1
    return issues, checks


def check_minute_baselines(baselines, current, *, max_baselines=256, evidence_state=None):
    """Evaluate mature baselines for surviving output identities."""
    issues = []
    checks = 0
    live = {(k, i) for k, v in current.get("tracks", {}).items() for i, _ in v}
    baselines = {x: y for x, y in baselines.items() if x in live}
    evidence_state = _evidence_state(evidence_state)
    effective_generations = dict(evidence_state["last_generation_by_route"])
    effective_generations.update({key: value[0] for key, value in current.get("generations", {}).items()})
    for identity, baseline in list(baselines.items()):
        stamp, pos, frame = baseline[:3]
        baseline_generation = baseline[3] if len(baseline) > 3 else frame.get("generations", {}).get(identity[0], (None,))[0]
        if (
            datetime.fromisoformat(current["utc"]) - datetime.fromisoformat(stamp)
        ).total_seconds() >= OBSERVATION_SPAN:
            a = dict(frame)
            a["tracks"] = {identity[0]: ((identity[1], pos),)}
            state = {"last_generation_by_route": effective_generations, "_baseline_generation": {identity[0]: baseline_generation}, "minute_checks": {}}
            e, c = minute_checks(a, current, state)
            issues.extend(e)
            checks += c
            for key, count in state["minute_checks"].items():
                evidence_state["minute_checks"][key] = evidence_state["minute_checks"].get(key, 0) + count
            baselines[identity] = (
                current["utc"],
                dict(current["tracks"][identity[0]])[identity[1]],
                current,
                baseline_generation,
            )
    for k, v in current.get("tracks", {}).items():
        for i, p in v:
            baselines.setdefault((k, i), (current["utc"], p, current, effective_generations.get(k)))
    return dict(list(baselines.items())[-max_baselines:]), issues, checks


def missing_complete_routes(requested, seen):
    return tuple(sorted(set(requested) - set(seen)))


def fresh_routes(collected, started, ended):
    """Return routes whose latest observation is process-fresh and within TTL."""
    return {key for key, stamp in collected.items()
            if stamp >= started and 0 <= (ended - stamp).total_seconds() <= EVIDENCE_TTL}


def evaluate_run(
    requested, fresh, tracks_seen, minute_count, spacing_count, violations, provider_errors=(), lifecycle_inconclusive=()
):
    if violations:
        return 1
    if provider_errors or lifecycle_inconclusive or missing_complete_routes(requested, fresh):
        return 2
    active = tracks_seen if isinstance(tracks_seen, dict) else ({key: 1 for key in requested} if tracks_seen else {})
    minutes = minute_count if isinstance(minute_count, dict) else ({key: minute_count for key in requested} if minute_count else {})
    gaps = spacing_count if isinstance(spacing_count, dict) else ({key: spacing_count for key in requested} if spacing_count else {})
    if any(not active.get(key, 0) or not minutes.get(key, 0) or not gaps.get(key, 0) for key in requested):
        return 2
    return 0


async def _run(cycles, interval, cache_dir, watch, output, fail_fast):
    settings = Settings.from_env(require_keys=False)
    previous = None
    seen = set()
    requested = set()
    violations = 0
    checks = 0
    handle = None
    started = datetime.now(UTC)
    fresh = {}
    minute_count = 0
    spacing_count = 0
    provider_errors = set()
    baselines = {}
    evidence = _evidence_state()
    completed_frames = 0
    try:
        async with aiohttp.ClientSession() as session:
            client = HttpClient(session, timeout_seconds=settings.http_timeout_seconds)
            geometry = await fetch_route_geometry(client, cache_dir=cache_dir)
            lines = [x for x in geometry.routes if not watch or route_key(x) in watch]
            requested = {route_key(x) for x in lines}
            mandatory = {str(x["stop"]) for x in KMB_STOPS + CTB_STOPS} | {
                str(x) for x in GMB_STOPS
            }
            probes = select_probe_stops(lines, mandatory_stop_ids=mandatory)
            validate_inputs(lines, probes)
            if output:
                handle = Path(output).open("w", encoding="utf-8")  # noqa: SIM115
            tracker = MarkerTracker(evidence_ttl_seconds=EVIDENCE_TTL)
            for n in range(cycles):
                groups, _, failed = await fetch_transit_etas(client)
                provider_errors.update(failed or ())
                snap = await fetch_probe_snapshot(client, probes)
                seen.update(tuple(x.route_key) for x in snap.complete_routes)
                for item in snap.complete_routes:
                    fresh[tuple(item.route_key)] = item.collected_at
                cand = estimate_bus_positions(
                    list(snap.rows),
                    lines,
                    _destination_map(groups, lines),
                    _authoritative_etas(groups, lines),
                )
                tracked = await tracker.update(snap, cand, lines)
                completed_frames += 1
                cur = frame_record(
                    snap,
                    cand,
                    tracked,
                    {route_key(x): max(0, len(getattr(x, "stops", ())) - 1) for x in lines},
                )
                iss, g = compare_adjacent(previous, cur, evidence)
                spacing_count += g
                baselines, minute_issues, minute_checks_count = check_minute_baselines(
                    baselines, cur, evidence_state=evidence
                )
                iss.extend(minute_issues)
                minute_count += minute_checks_count
                violations += len(iss)
                checks += g
                cur["issues"] = iss
                cur["counters"] = {
                    "minute_checks": dict(evidence["minute_checks"]),
                    "gap_checks": dict(evidence["gap_checks"]),
                    "gap_inconclusive": dict(evidence["gap_inconclusive"]),
                    "lifecycle_inconclusive": dict(evidence["lifecycle_inconclusive"]),
                }
                if handle:
                    handle.write(json.dumps(_json_safe_record(cur), separators=(",", ":")) + "\n")
                    handle.flush()
                print(
                    f"FRAME {n + 1}/{cycles} violations={len(iss)} status={'FAIL' if iss else 'PASS'}"
                )
                previous = cur
                if iss and fail_fast:
                    break
                if n + 1 < cycles:
                    await asyncio.sleep(interval)
    except Exception as exc:
        print(f"SUMMARY status=INCONCLUSIVE diagnostic_error={type(exc).__name__}")
        return 2
    finally:
        if handle:
            handle.close()
        await shutdown_background_refreshes()
    now = datetime.now(UTC)
    fresh = fresh_routes(fresh, started, now)
    return_code = evaluate_run(
        requested,
        fresh,
        ({key: len(values) for key, values in (previous or {}).get("tracks", {}).items()}),
        evidence["minute_checks"],
        evidence["gap_checks"],
        violations,
        provider_errors,
        evidence["lifecycle_inconclusive"],
    )
    print(
        f"SUMMARY frames={completed_frames} gap_checks={sum(evidence['gap_checks'].values())} minute_checks={sum(evidence['minute_checks'].values())} lifecycle_inconclusive={sum(evidence['lifecycle_inconclusive'].values())} violations={violations} status={'FAIL' if violations else ('INCONCLUSIVE' if return_code == 2 else 'PASS')}"
    )
    return return_code


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cycles", type=int, default=72)
    p.add_argument("--interval", type=float, default=10)
    p.add_argument("--cache-dir", default=".cache")
    p.add_argument("--watch-route", action="append", default=[])
    p.add_argument("--jsonl")
    p.add_argument("--fail-fast", action="store_true")
    a = p.parse_args()
    return asyncio.run(
        _run(
            a.cycles,
            a.interval,
            a.cache_dir,
            tuple(parse_route(x) for x in a.watch_route),
            a.jsonl,
            a.fail_fast,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
