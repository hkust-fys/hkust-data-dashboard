"""Small public-data continuity harness for MarkerTracker."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path

import aiohttp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from dashboard.http import HttpClient  # noqa: E402
from dashboard.maps import _authoritative_etas, _destination_map  # noqa: E402
from dashboard.maps.marker_audit import audit_marker_positions  # noqa: E402
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


def frame_record(
    snapshot,
    candidates,
    tracked,
    route_max=None,
    timestamp=None,
    priorities=None,
    source_audit=None,
):
    def evidence(x):
        arrival = getattr(x, "eta_arrival_at", None)
        return {
            "bracket": getattr(x, "bracket", None),
            "unreliable": bool(getattr(x, "unreliable", False)),
            "eta_minutes": getattr(x, "eta_minutes", None),
            "eta_arrival_at": arrival.isoformat() if hasattr(arrival, "isoformat") else arrival,
            "boundary_age_seconds": getattr(x, "boundary_age_seconds", None),
            "boundary_revision": getattr(x, "boundary_revision", None),
            "bracket_eta_offsets": getattr(x, "bracket_eta_offsets", None),
            "source_indices": sorted(getattr(x, "source_indices", ()) or ()),
            "source_observations": sorted(getattr(x, "source_observations", ()) or ()),
            "priority_indices": sorted(getattr(x, "priority_indices", ()) or ()),
            "position_authoritative": getattr(x, "position_authoritative", None),
            "checkpoint_evidence": sorted(
                getattr(x, "checkpoint_evidence", ()) or ()
            ),
        }
    def evidence_record(x):
        e = evidence(x)
        return {"position": float(getattr(x, "position", 0)), **e}
    checkpoint_ages = {}
    source_rows = {}
    positioning_rows = getattr(snapshot, "positioning_rows", None)
    for input_index, row in enumerate(
        positioning_rows if positioning_rows is not None else ()
    ):
        key = route_key(row)
        arrival = getattr(row, "arrival_at", None)
        kind = getattr(row, "kind", None)
        source_rows.setdefault(key, []).append({
            "observation": ("probe", input_index),
            "index": getattr(row, "index", None),
            "minutes": getattr(row, "minutes", None),
            "kind": getattr(kind, "value", kind),
            "arrival_at": (
                arrival.isoformat() if hasattr(arrival, "isoformat") else arrival
            ),
            "cache_age_seconds": getattr(row, "cache_age_seconds", None),
            "refresh_generation": getattr(row, "refresh_generation", None),
        })
        try:
            index = int(row.index)
            age = float(row.cache_age_seconds)
        except (TypeError, ValueError, AttributeError):
            continue
        if not isfinite(age) or age < 0:
            continue
        key = route_key(row)
        by_index = checkpoint_ages.setdefault(key, {})
        by_index[index] = max(age, by_index.get(index, 0.0))
    record = {
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
        "priority_checkpoints": {
            tuple(key): sorted(int(index) for index in indices)
            for key, indices in (priorities or {}).items()
        },
        "checkpoint_ages": checkpoint_ages,
        "source_rows": source_rows,
        "route_max": route_max or {},
    }
    if source_audit is not None:
        record["source_audit"] = source_audit
    return record


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


def _audit_issue_has_duplicate_ownership(issue):
    """Keep duplicated displayed-source ownership hard without lifecycle proof."""
    detail = issue.get("detail", {}) if isinstance(issue, dict) else {}
    match = detail.get("match", {}) if isinstance(detail, dict) else {}
    return bool(match.get("duplicate_sources"))


def _source_audit_generation_context(snapshot, route_keys, previous):
    """Classify routes whose current frame carries a new complete generation."""
    current = {
        tuple(item.route_key): int(item.generation)
        for item in getattr(snapshot, "complete_routes", ())
    }
    strict = {
        key for key, generation in current.items()
        if previous.get(key) != generation
    }
    reasons = {}
    for key in route_keys:
        if key in strict:
            continue
        if key not in current:
            reasons[key] = (
                "no complete generation available; displayed population is "
                "not comparable to partial source rows"
                if key not in previous
                else
                "complete generation omitted; displayed population may be held"
            )
        else:
            reasons[key] = (
                "complete generation unchanged; displayed population may be "
                "held against newer partial source rows"
            )
    previous.update(current)
    return strict, reasons


def source_audit_result(audit, *, strict_routes=None, route_reasons=None):
    """Return a bounded frame record plus hard verifier issues.

    The auditor has already compared raw source rows with the tracker output.
    Preserve its exact issue details for diagnosis, but keep successful frame
    records compact by retaining only summary counters and inconclusive checks.
    """
    audit = audit if isinstance(audit, dict) else {}
    checks = tuple(audit.get("checks", ()) or ())
    raw_issues = tuple(audit.get("issues", ()) or ())
    strict_routes = (
        None if strict_routes is None
        else {tuple(key) for key in strict_routes}
    )
    route_reasons = {
        tuple(key): reason for key, reason in (route_reasons or {}).items()
    }
    inconclusive = [
        {
            name: check.get(name)
            for name in ("key", "kind", "checkpoint", "marker_id", "reason")
            if check.get(name) is not None
        }
        for check in checks
        if check.get("inconclusive")
    ]
    generation_context_count = 0
    hard_raw_issues = [
        issue for issue in raw_issues
        if (
            strict_routes is None
            or tuple(issue.get("key", ())) in strict_routes
            or _audit_issue_has_duplicate_ownership(issue)
        )
    ]
    deferred_issues = [
        issue for issue in raw_issues
        if issue not in hard_raw_issues
    ]
    for issue in deferred_issues:
        key = tuple(issue.get("key", ()))
        detail = issue.get("detail", {}) or {}
        inconclusive.append({
            "key": key,
            "kind": issue.get("kind"),
            "checkpoint": detail.get("checkpoint", detail.get("gate_index")),
            "reason": route_reasons.get(
                key, "no new complete generation for strict source comparison"
            ),
            "deferred_issue": True,
        })
    deferred_routes = {
        tuple(issue.get("key", ())) for issue in deferred_issues
    }
    audited_routes = {
        tuple(check.get("key", ())) for check in checks
        if check.get("key") is not None
    }
    for key in sorted(audited_routes - deferred_routes):
        if strict_routes is not None and key not in strict_routes:
            inconclusive.append({
                "key": key,
                "kind": "generation-context",
                "reason": route_reasons.get(
                    key, "no new complete generation for strict source comparison"
                ),
            })
            generation_context_count += 1
    issues = [
        {
            "kind": "source_audit_failure",
            "route": tuple(issue.get("key", ())),
            "audit_kind": issue.get("kind"),
            "detail": issue.get("detail", {}),
        }
        for issue in hard_raw_issues
    ]
    stats = dict(audit.get("stats", {}) or {})
    stats["inconclusive"] = len(inconclusive)
    return (
        {
            "ok": not hard_raw_issues,
            "check_count": len(checks) + generation_context_count,
            "inconclusive_count": len(inconclusive),
            "inconclusive_checks": inconclusive,
            "stats": stats,
            "issues": hard_raw_issues,
            "deferred_issues": deferred_issues,
        },
        issues,
    )


def _evidence_state(state=None):
    if state is None:
        return {
        "last_generation_by_route": {}, "last_ids_by_route": {},
        "latest_complete_collected_at": {},
        "minute_baselines": {}, "minute_checks": {}, "gap_checks": {},
        "gap_inconclusive": {}, "gap_inconclusive_reasons": {},
        "lifecycle_inconclusive": {},
        "bracket_checks": {}, "bracket_inconclusive": {},
        }
    for key in ("last_generation_by_route", "last_ids_by_route", "latest_complete_collected_at",
                "minute_baselines", "minute_checks", "gap_checks", "gap_inconclusive",
                "gap_inconclusive_reasons",
                "lifecycle_inconclusive", "bracket_checks", "bracket_inconclusive"):
        state.setdefault(key, {})
    return state


def _route_maps(record):
    return {key: {int(track): pos for track, pos in values}
            for key, values in record.get("tracks", {}).items()}


def _provenance_signature(evidence):
    """Return identity evidence used to match candidates to displayed tracks.

    Boundary age is deliberately excluded: it describes freshness of the
    observation, not the identity of the ETA/bracket that produced it.
    """
    if not isinstance(evidence, dict):
        return None
    bracket = evidence.get("bracket")
    eta_minutes = evidence.get("eta_minutes")
    eta_arrival_at = evidence.get("eta_arrival_at")
    source_indices = evidence.get("source_indices")
    source_observations = evidence.get("source_observations")
    if (bracket is None and eta_minutes is None and eta_arrival_at is None
            and not source_indices and not source_observations):
        return None
    observations = tuple(
        tuple(item) if isinstance(item, (list, tuple)) else item
        for item in (source_observations or ())
    )
    bracket_eta_offsets = evidence.get("bracket_eta_offsets")
    return (
        tuple(bracket) if bracket is not None else None,
        tuple(bracket_eta_offsets) if bracket_eta_offsets is not None else None,
        eta_minutes,
        eta_arrival_at,
        tuple(source_indices or ()),
        observations,
    )


def _boundary_revision(evidence):
    """Normalize a complete pair of provider endpoint revisions."""
    if not isinstance(evidence, dict):
        return None
    value = evidence.get("boundary_revision")
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        return None
    try:
        revision = (int(value[0]), int(value[1]))
    except (TypeError, ValueError):
        return None
    return revision if all(item > 0 for item in revision) else None


def _boundary_age(evidence):
    """Return a tracker-compatible nonnegative finite boundary age."""
    if not isinstance(evidence, dict):
        return None
    try:
        age = float(evidence.get("boundary_age_seconds"))
    except (TypeError, ValueError):
        return None
    return age if isfinite(age) and age >= 0.0 else None


def _boundary_evidence_attempted(evidence):
    """Whether a bracket claims fresh or revisioned boundary evidence."""
    if not isinstance(evidence, dict):
        return False
    if evidence.get("boundary_revision") is not None:
        return True
    age = _boundary_age(evidence)
    return age is not None and age <= TRACKER_BOUNDARY_FRESH_SECONDS


def _source_observation_signature(evidence):
    """Return the estimator ladder claimed by one displayed marker."""
    if not isinstance(evidence, dict):
        return None
    observations = evidence.get("source_observations") or ()
    if not observations:
        return None
    return tuple(
        tuple(item) if isinstance(item, (list, tuple)) else item
        for item in observations
    )


def _track_evidence(record, key, track):
    """Read in-memory integer or JSON-round-tripped string track keys."""
    route_evidence = record.get("track_evidence", {}).get(key, {})
    return route_evidence.get(track, route_evidence.get(str(track), {}))


def _duplicate_candidate_observation_issues(record, key):
    """Reject one current ETA observation feeding multiple marker candidates."""
    owners = {}
    issues = []
    for candidate_index, evidence in enumerate(
        record.get("candidate_evidence", {}).get(key, ())
    ):
        signature = _source_observation_signature(evidence) or ()
        for observation in signature:
            other = owners.get(observation)
            if other is not None and other != candidate_index:
                issues.append({
                    "kind": "duplicate_candidate_observation",
                    "route": key,
                    "candidate_index": candidate_index,
                    "other_candidate_index": other,
                    "source_observation": observation,
                })
            else:
                owners[observation] = candidate_index
    return issues


def _spacing_provenance_comparable(record, key, tracks, candidates):
    """Return whether positions have authoritative, one-to-one evidence."""
    candidate_evidence = record.get("candidate_evidence", {}).get(key)
    track_evidence = record.get("track_evidence", {}).get(key)
    if candidate_evidence is None or track_evidence is None:
        return False, "missing_position_evidence"
    candidate_evidence = list(candidate_evidence or ())
    track_evidence_rows = [
        _track_evidence(record, key, track)
        for track, _ in tracks
    ]
    if (
        len(candidate_evidence) != len(candidates)
        or len(track_evidence_rows) != len(tracks)
    ):
        return False, "missing_position_evidence"
    if not all(
        evidence.get("position_authoritative") is True
        for evidence in candidate_evidence + track_evidence_rows
    ):
        return False, "non_authoritative_position"
    candidate_signatures = [
        _provenance_signature(evidence) for evidence in candidate_evidence
    ]
    track_signatures = [
        _provenance_signature(evidence) for evidence in track_evidence_rows
    ]
    if not any(candidate_signatures) and not any(track_signatures):
        return True, None
    if any(signature is None for signature in candidate_signatures + track_signatures):
        return False, "provenance_mismatch"
    if Counter(candidate_signatures) != Counter(track_signatures):
        return False, "provenance_mismatch"
    return True, None


def _record_gap_inconclusive(state, key, reason):
    state["gap_inconclusive"][key] = state["gap_inconclusive"].get(key, 0) + 1
    route_reasons = state["gap_inconclusive_reasons"].setdefault(key, {})
    route_reasons[reason] = route_reasons.get(reason, 0) + 1


def _eta_allows_motion(old, new, key, track):
    before = _track_evidence(old, key, track)
    after = _track_evidence(new, key, track)
    fields = ("bracket", "bracket_eta_offsets", "eta_minutes", "eta_arrival_at",
              "source_indices", "source_observations")
    changed = tuple(after.get(k) for k in fields) != tuple(before.get(k) for k in fields)
    after_revision = _boundary_revision(after)
    before_revision = _boundary_revision(before)
    if after.get("boundary_revision") is not None:
        # A revision identifies consumed endpoint responses.  Both endpoints
        # must advance; this rejects replayed and one-sided/partial refreshes,
        # even when the resulting estimate is observed several seconds later.
        return (
            changed
            and after_revision is not None
            and _boundary_age(after) is not None
            and (before_revision is None
                 or all(a > b for a, b in zip(
                     after_revision, before_revision, strict=True
                 )))
        )
    if before_revision is not None:
        return False
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
    evidence = _track_evidence(record, key, track)
    bracket = evidence.get("bracket") or ()
    sources = {int(index) for index in evidence.get("source_indices", ())}
    observed = {
        int(index) for index in record.get("observed_checkpoints", {}).get(key, ())
    }
    revision = _boundary_revision(evidence)
    raw_revision = evidence.get("boundary_revision")
    eta = evidence.get("eta_minutes")
    eta_offsets = evidence.get("bracket_eta_offsets")
    if (
        len(bracket) != 2
        or not sources
        or not isinstance(eta, (int, float))
        or (raw_revision is not None and (revision is None or _boundary_age(evidence) is None))
        or (raw_revision is None and (
            _boundary_age(evidence) is None
            or _boundary_age(evidence) > TRACKER_BOUNDARY_FRESH_SECONDS
        ))
    ):
        return False
    lower, upper = map(float, bracket)
    if eta_offsets is not None:
        if (
            len(eta_offsets) != 2
            or int(lower) not in sources
            or int(upper) not in sources
        ):
            return False
        lower_eta, upper_eta = map(float, eta_offsets)
        if lower == upper:
            expected = lower
        elif lower_eta <= 0 < upper_eta:
            expected = lower + (upper - lower) * (
                -lower_eta / (upper_eta - lower_eta)
            )
        else:
            return False
        return (
            lower <= position <= upper
            and abs(position - expected) <= POSITION_EPSILON
        )
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
        initial_issues = []
        for key, generation in new.get("generations", {}).items():
            state["last_generation_by_route"][key] = generation[0]
            state["latest_complete_collected_at"][key] = generation[1]
            candidate_count = len(new.get("candidates", {}).get(key, ()))
            track_count = len(new.get("tracks", {}).get(key, ()))
            if candidate_count != track_count:
                initial_issues.append({
                    "kind": "cardinality_mismatch_at_complete_generation",
                    "route": key,
                    "candidate_count": candidate_count,
                    "track_count": track_count,
                })
        # The first complete frame is valid direct spacing evidence.
        issues, checks = compare_adjacent(new, new, state)
        return initial_issues + issues, checks
    for key, generation in old.get("generations", {}).items():
        state["last_generation_by_route"].setdefault(key, generation[0])
        state["latest_complete_collected_at"].setdefault(key, generation[1])
    issues, checks = [], 0
    for key, values in new.get("tracks", {}).items():
        track_ids = [int(track) for track, _position in values]
        duplicates = sorted(
            track for track, count in Counter(track_ids).items() if count > 1
        )
        for track in duplicates:
            issues.append({
                "kind": "duplicate_track_identity",
                "route": key,
                "track_id": track,
            })
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
        issues.extend(_duplicate_candidate_observation_issues(new, key))
        signatures = {}
        source_signatures = {}
        source_owners = {}
        for track in b:
            route_evidence = new.get("track_evidence", {}).get(key, {})
            track_evidence = route_evidence.get(
                track, route_evidence.get(str(track))
            )
            signature = _provenance_signature(track_evidence)
            source_signature = _source_observation_signature(track_evidence)
            duplicate = (
                source_signatures.get(source_signature)
                if source_signature is not None
                else None
            )
            if duplicate is None and signature is not None:
                duplicate = signatures.get(signature)
            if duplicate is not None:
                issues.append({
                    "kind": "duplicate_track_evidence",
                    "route": key,
                    "track_id": track,
                    "other_track_id": duplicate,
                })
            if signature is not None:
                signatures[signature] = track
            if source_signature is not None:
                for observation in source_signature:
                    other_owner = source_owners.get(observation)
                    if other_owner is not None and other_owner != track:
                        issues.append({
                            "kind": "duplicate_track_observation",
                            "route": key,
                            "track_id": track,
                            "other_track_id": other_owner,
                            "source_observation": observation,
                        })
                    else:
                        source_owners[observation] = track
                source_signatures[source_signature] = track
        current_generation = new.get("generations", {}).get(key)
        last_generation = state["last_generation_by_route"].get(key)
        generation_changed = bool(current_generation and last_generation is not None
                                   and current_generation[0] != last_generation)
        if current_generation:
            # Compare against the last complete generation, including through outages.
            if last_generation is None or generation_changed:
                state["last_generation_by_route"][key] = current_generation[0]
                state["latest_complete_collected_at"][key] = current_generation[1]
                candidate_count = len(new.get("candidates", {}).get(key, ()))
                if candidate_count != len(b):
                    issues.append({
                        "kind": "cardinality_mismatch_at_complete_generation",
                        "route": key,
                        "candidate_count": candidate_count,
                        "track_count": len(b),
                    })
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
        new_common = [track for track in b if track in a]
        if len(common) > 1 and common != new_common:
            crossing = False
            new_order = {track: index for index, track in enumerate(new_common)}
            for left_index, left_track in enumerate(common):
                for right_track in common[left_index + 1:]:
                    old_gap = a[right_track] - a[left_track]
                    if (old_gap != 0.0
                            and new_order[left_track] > new_order[right_track]):
                        crossing = True
                        break
                if crossing:
                    break
            if crossing:
                issues.append({"kind": "identity_order_crossing", "route": key})
        # Gap evidence is a property of this complete frame, never an inferred match.
        candidates = new.get("candidates", {}).get(key, ())
        provenance_comparable, spacing_reason = _spacing_provenance_comparable(
            new, key, b.items(), candidates
        )
        if current_generation and len(candidates) == len(b) >= 2 and provenance_comparable:
            checks += len(candidates) - 1
            state["gap_checks"][key] = state["gap_checks"].get(key, 0) + len(candidates) - 1
            if any(abs((candidates[i + 1] - candidates[i]) -
                       (tuple(b.values())[i + 1] - tuple(b.values())[i])) > GAP_TOLERANCE
                   for i in range(len(candidates) - 1)):
                issues.append({"kind": "spacing_mismatch", "route": key})
        elif current_generation and (len(candidates) != len(b) or not provenance_comparable):
            _record_gap_inconclusive(
                state,
                key,
                "cardinality_mismatch" if len(candidates) != len(b)
                else spacing_reason or "unknown",
            )
        valid_tracks = [
            (track, position)
            for track, position in b.items()
            if _direct_bracket_evidence(new, key, track, position)
        ]
        invalid_fresh = [
            track
            for track, position in b.items()
            if (
                len(_track_evidence(new, key, track).get("bracket") or ()) == 2
                and _boundary_evidence_attempted(
                    _track_evidence(new, key, track)
                )
                and not _direct_bracket_evidence(new, key, track, position)
            )
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
        current_position = dict(current.get("tracks", {}).get(identity[0], ())).get(
            identity[1]
        )
        if (
            current_position is not None
            and current_position + POSITION_EPSILON < pos
            and _eta_allows_motion(frame, current, identity[0], identity[1])
        ):
            # A fresh ETA correction is an allowed backward snap. Start the
            # minute window at that evidence event so a later cached frame
            # cannot misreport the already-attributed correction as new
            # unexplained motion. Ordinary generation refreshes do not reset
            # surviving route/track baselines.
            baselines[identity] = (
                current["utc"],
                current_position,
                current,
                effective_generations.get(identity[0], baseline_generation),
            )
            continue
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
    requested, fresh, tracks_seen, minute_count, spacing_count, violations,
    provider_errors=(), lifecycle_inconclusive=(), bracket_count=(),
    source_audit_inconclusive=0,
):
    if violations:
        return 1
    if (provider_errors or lifecycle_inconclusive or source_audit_inconclusive
            or missing_complete_routes(requested, fresh)):
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
    source_audit_checks = 0
    source_audit_inconclusive = 0
    source_audit_matched = 0
    source_audit_excluded_undeparted = 0
    source_audit_deferred = 0
    audit_complete_generations = {}
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
                priorities = tracker.poll_priorities()
                snap = await fetch_probe_snapshot(
                    client, probes, priorities=priorities
                )
                seen.update(tuple(x.route_key) for x in snap.complete_routes)
                for item in snap.complete_routes:
                    fresh[tuple(item.route_key)] = item.collected_at
                positioning_rows = getattr(snap, "positioning_rows", None)
                rows = list(snap.rows) if positioning_rows is None else list(positioning_rows)
                observed = _observed_checkpoint_map(snap)
                authoritative = _authoritative_etas(groups, lines)
                cand = estimate_bus_positions(
                    rows,
                    lines,
                    _destination_map(groups, lines),
                    authoritative,
                    observed_checkpoint_indices=observed,
                )
                tracked = await tracker.update(snap, cand, lines)
                audit = audit_marker_positions(
                    rows,
                    authoritative,
                    tracked,
                    lines,
                    frame_id=n + 1,
                    observed_checkpoint_indices=observed,
                )
                strict_audit_routes, audit_route_reasons = (
                    _source_audit_generation_context(
                        snap, requested, audit_complete_generations
                    )
                )
                audit_record, audit_issues = source_audit_result(
                    audit,
                    strict_routes=strict_audit_routes,
                    route_reasons=audit_route_reasons,
                )
                source_audit_checks += audit_record["check_count"]
                source_audit_inconclusive += audit_record["inconclusive_count"]
                source_audit_deferred += len(audit_record["deferred_issues"])
                source_audit_matched += int(audit_record["stats"].get("matched", 0) or 0)
                source_audit_excluded_undeparted += int(
                    audit_record["stats"].get("excluded_undeparted", 0) or 0
                )
                completed_frames += 1
                cur = frame_record(
                    snap,
                    cand,
                    tracked,
                    {route_key(x): max(0, len(getattr(x, "stops", ())) - 1) for x in lines},
                    priorities=priorities,
                    source_audit=audit_record,
                )
                iss, g = compare_adjacent(previous, cur, evidence)
                spacing_count += g
                baselines, minute_issues, minute_checks_count = check_minute_baselines(
                    baselines, cur, evidence_state=evidence
                )
                iss.extend(minute_issues)
                iss.extend(audit_issues)
                minute_count += minute_checks_count
                violations += len(iss)
                checks += g
                cur["issues"] = iss
                cur["counters"] = {
                    "minute_checks": dict(evidence["minute_checks"]),
                    "gap_checks": dict(evidence["gap_checks"]),
                    "gap_inconclusive": dict(evidence["gap_inconclusive"]),
                    "gap_inconclusive_reasons": {
                        key: dict(reasons)
                        for key, reasons in evidence["gap_inconclusive_reasons"].items()
                    },
                    "lifecycle_inconclusive": dict(evidence["lifecycle_inconclusive"]),
                    "bracket_checks": dict(evidence["bracket_checks"]),
                    "bracket_inconclusive": dict(evidence["bracket_inconclusive"]),
                    "source_audit_checks": source_audit_checks,
                    "source_audit_inconclusive": source_audit_inconclusive,
                    "source_audit_matched": source_audit_matched,
                    "source_audit_excluded_undeparted": source_audit_excluded_undeparted,
                    "source_audit_deferred": source_audit_deferred,
                }
                if handle:
                    handle.write(json.dumps(_json_safe_record(cur), separators=(",", ":")) + "\n")
                    handle.flush()
                frame_status = (
                    "FAIL" if iss else
                    "INCONCLUSIVE" if audit_record["inconclusive_count"] else
                    "PASS"
                )
                print(
                    f"FRAME {n + 1}/{cycles} violations={len(iss)} "
                    f"source_audit_matched={audit_record['stats'].get('matched', 0)} "
                    f"source_audit_excluded_undeparted="
                    f"{audit_record['stats'].get('excluded_undeparted', 0)} "
                    f"source_audit_inconclusive={audit_record['inconclusive_count']} "
                    f"source_audit_deferred={len(audit_record['deferred_issues'])} "
                    f"status={frame_status}"
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
        source_audit_inconclusive,
    )
    gap_inconclusive_reasons = Counter(
        reason
        for reasons in evidence["gap_inconclusive_reasons"].values()
        for reason, count in reasons.items()
        for _ in range(count)
    )
    print(
        f"SUMMARY frames={completed_frames} gap_checks={sum(evidence['gap_checks'].values())} gap_inconclusive={sum(evidence['gap_inconclusive'].values())} gap_inconclusive_reasons={dict(sorted(gap_inconclusive_reasons.items()))} minute_checks={sum(evidence['minute_checks'].values())} bracket_checks={sum(evidence['bracket_checks'].values())} bracket_inconclusive={sum(evidence['bracket_inconclusive'].values())} lifecycle_inconclusive={sum(evidence['lifecycle_inconclusive'].values())} source_audit_checks={source_audit_checks} source_audit_matched={source_audit_matched} source_audit_excluded_undeparted={source_audit_excluded_undeparted} source_audit_inconclusive={source_audit_inconclusive} source_audit_deferred={source_audit_deferred} violations={violations} status={'FAIL' if violations else ('INCONCLUSIVE' if return_code == 2 else 'PASS')}"
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
