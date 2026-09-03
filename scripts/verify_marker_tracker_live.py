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
HTTP_TIMEOUT_SECONDS = 30.0
TRACKER_BOUNDARY_FRESH_SECONDS = 5.0
RouteKey = tuple[str, str, str]
FAILED_OPERATOR_CODES = {"KMB": "KMB", "Citybus": "CTB", "GMB": "GMB"}


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
    def evidence(x):
        arrival = getattr(x, "eta_arrival_at", None)
        return {
            "bracket": getattr(x, "bracket", None),
            "eta_minutes": getattr(x, "eta_minutes", None),
            "eta_arrival_at": arrival.isoformat() if hasattr(arrival, "isoformat") else arrival,
            "boundary_age_seconds": getattr(x, "boundary_age_seconds", None),
            "source_indices": sorted(getattr(x, "source_indices", ()) or ()),
            "source_observations": sorted(getattr(x, "source_observations", ()) or ()),
        }
    def evidence_record(x):
        e = evidence(x)
        return {"position": float(getattr(x, "position", 0)), **e}
    return {
        "utc": timestamp or datetime.now(UTC).isoformat(),
        "generations": {
            tuple(x.route_key): (int(x.generation), x.collected_at.isoformat())
            for x in snapshot.complete_routes
        },
        "candidates": _positions(candidates),
        "candidate_evidence": {key: [evidence_record(x) for x in sorted(values, key=lambda x: float(x.position or 0))]
                               for key, values in ((key, [x for x in candidates if route_key(x) == key])
                                                   for key in {route_key(x) for x in candidates})},
        "tracks": _tracks(tracked),
        "track_evidence": {
            key: {int(x.track_id): evidence(x) for x in values if x.track_id is not None}
            for key, values in ((key, [x for x in tracked if route_key(x) == key])
                                for key in {route_key(x) for x in tracked})
        },
        "observed_checkpoints": _observed_checkpoint_map(snapshot),
        "route_max": route_max or {},
    }


def _json_safe_record(x):
    if isinstance(x, dict):
        return {
            ("/".join(k) if isinstance(k, tuple) else str(k)): _json_safe_record(v)
            for k, v in x.items()
        }
    if isinstance(x, (set, frozenset)):
        return [_json_safe_record(v) for v in sorted(x, key=str)]
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
        "bracket_checks": {}, "bracket_inconclusive": {},
        }
    for key in ("last_generation_by_route", "last_ids_by_route", "latest_complete_collected_at",
                "minute_baselines", "minute_checks", "gap_checks", "gap_inconclusive",
                "lifecycle_inconclusive", "bracket_checks", "bracket_inconclusive"):
        state.setdefault(key, {})
    return state


def _route_maps(record):
    return {key: {int(track): pos for track, pos in values}
            for key, values in record.get("tracks", {}).items()}


def _eta_allows_motion(old, new, key, track):
    before = old.get("track_evidence", {}).get(key, {}).get(track, {})
    after = new.get("track_evidence", {}).get(key, {}).get(track, {})
    fields = ("bracket", "eta_minutes", "eta_arrival_at", "source_indices", "source_observations")
    changed = tuple(after.get(k) for k in fields) != tuple(before.get(k) for k in fields)
    age = after.get("boundary_age_seconds")
    return changed and isinstance(age, (int, float)) and 0 <= age <= TRACKER_BOUNDARY_FRESH_SECONDS


def _observed_checkpoint_map(snapshot):
    checkpoints = getattr(snapshot, "positioning_checkpoints", None)
    if isinstance(checkpoints, dict):
        return {tuple(key): frozenset(value) for key, value in checkpoints.items()}
    if checkpoints is not None:
        observed = {}
        for operator, route, bound, index in checkpoints:
            observed.setdefault((operator, route, bound), set()).add(index)
        if observed:
            return {key: frozenset(value) for key, value in observed.items()}
    return {tuple(route.route_key): getattr(route, "observed_checkpoint_indices", frozenset())
            for route in getattr(snapshot, "complete_routes", ())}


def _direct_bracket_evidence(record, key, track, position):
    evidence = record.get("track_evidence", {}).get(key, {}).get(track, {})
    bracket = evidence.get("bracket") or ()
    sources = {int(index) for index in evidence.get("source_indices", ())}
    observed = {
        int(index) for index in record.get("observed_checkpoints", {}).get(key, ())
    }
    age = evidence.get("boundary_age_seconds")
    eta = evidence.get("eta_minutes")
    if (
        len(bracket) != 2
        or not sources
        or not isinstance(age, (int, float))
        or not 0 <= age <= TRACKER_BOUNDARY_FRESH_SECONDS
        or not isinstance(eta, (int, float))
    ):
        return False
    lower, upper = map(float, bracket)
    first_present = min(sources)
    if upper != float(first_present):
        return False
    if first_present == 0:
        if lower != 0.0:
            return False
    else:
        absent = [
            index for index in observed
            if index < first_present and index not in sources
        ]
        if not absent or lower != float(max(absent)):
            return False
    expected = first_present - min(1.0, max(0.0, float(eta) / 2.0))
    expected = min(upper, max(lower, expected))
    return lower <= position <= upper and abs(position - expected) <= POSITION_EPSILON


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
            if track in b and abs(b[track] - position) > POSITION_EPSILON and not _eta_allows_motion(old, new, key, track):
                kind = "backward_without_eta_evidence" if b[track] < position else "movement_without_eta_evidence"
                issues.append({"kind": kind, "route": key, "track_id": track,
                               "detail": "movement lacks fresh changed ETA/bracket evidence"})
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
        valid_tracks = [
            (track, position)
            for track, position in b.items()
            if _direct_bracket_evidence(new, key, track, position)
        ]
        invalid_fresh = [
            track
            for track, position in b.items()
            if len(
                new.get("track_evidence", {})
                .get(key, {})
                .get(track, {})
                .get("bracket")
                or ()
            ) == 2
            and isinstance(
                new["track_evidence"][key][track].get("boundary_age_seconds"),
                (int, float),
            )
            and 0
            <= new["track_evidence"][key][track]["boundary_age_seconds"]
            <= TRACKER_BOUNDARY_FRESH_SECONDS
            and not _direct_bracket_evidence(new, key, track, position)
        ]
        for track in invalid_fresh:
            issues.append({
                "kind": "invalid_bracket_evidence",
                "route": key,
                "track_id": track,
            })
        if len(valid_tracks) == len(b) >= 2:
            state["bracket_checks"][key] = (
                state["bracket_checks"].get(key, 0) + len(valid_tracks) - 1
            )
        elif current_generation:
            state["bracket_inconclusive"][key] = state["bracket_inconclusive"].get(key, 0) + 1
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
            if q + POSITION_EPSILON < p and not _eta_allows_motion(old, new, key, i):
                issues.append({"kind": "minute_backward", "route": key, "track_id": i})
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
    requested, fresh, tracks_seen, minute_count, spacing_count, violations, provider_errors=(), lifecycle_inconclusive=(), bracket_count=()
):
    if violations:
        return 1
    if provider_errors or lifecycle_inconclusive or missing_complete_routes(requested, fresh):
        return 2
    active = tracks_seen if isinstance(tracks_seen, dict) else ({key: 1 for key in requested} if tracks_seen else {})
    minutes = minute_count if isinstance(minute_count, dict) else ({key: minute_count for key in requested} if minute_count else {})
    gaps = spacing_count if isinstance(spacing_count, dict) else ({key: spacing_count for key in requested} if spacing_count else {})
    brackets = bracket_count if isinstance(bracket_count, dict) else ({key: bracket_count for key in requested} if bracket_count else {})
    if any(not active.get(key, 0) or not minutes.get(key, 0) or not gaps.get(key, 0) or not brackets.get(key, 0) for key in requested):
        return 2
    return 0


async def _run(cycles, interval, cache_dir, watch, output, fail_fast):
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
            client = HttpClient(session, timeout_seconds=HTTP_TIMEOUT_SECONDS)
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
                requested_operators = {key[0] for key in requested}
                provider_errors.update(
                    name
                    for name in failed or ()
                    if FAILED_OPERATOR_CODES.get(name, name) in requested_operators
                )
                snap = await fetch_probe_snapshot(client, probes, priorities=tracker.poll_priorities())
                seen.update(tuple(x.route_key) for x in snap.complete_routes)
                for item in snap.complete_routes:
                    fresh[tuple(item.route_key)] = item.collected_at
                positioning_rows = getattr(snap, "positioning_rows", None)
                rows = list(snap.rows) if positioning_rows is None else list(positioning_rows)
                observed = _observed_checkpoint_map(snap)
                cand = estimate_bus_positions(
                    rows,
                    lines,
                    _destination_map(groups, lines),
                    _authoritative_etas(groups, lines),
                    observed_checkpoint_indices=observed,
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
                    "bracket_checks": dict(evidence["bracket_checks"]),
                    "bracket_inconclusive": dict(evidence["bracket_inconclusive"]),
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
        print(
            "SUMMARY status=INCONCLUSIVE "
            f"diagnostic_error={type(exc).__name__}: {exc}"
        )
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
        evidence["bracket_checks"],
    )
    print(
        f"SUMMARY frames={completed_frames} gap_checks={sum(evidence['gap_checks'].values())} minute_checks={sum(evidence['minute_checks'].values())} bracket_checks={sum(evidence['bracket_checks'].values())} bracket_inconclusive={sum(evidence['bracket_inconclusive'].values())} lifecycle_inconclusive={sum(evidence['lifecycle_inconclusive'].values())} violations={violations} status={'FAIL' if violations else ('INCONCLUSIVE' if return_code == 2 else 'PASS')}"
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
