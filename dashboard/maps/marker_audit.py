"""Pure source-to-marker checks for one rendered traffic-map frame.

The auditor consumes the ETA rows already fetched for the frame. It performs
no I/O and adds no traffic to the operator APIs.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from dashboard.maps.positions import (
    GATE_DOWNSTREAM_DRIFT_MINUTES,
    GATE_UPSTREAM_DRIFT_MINUTES,
    _plan_gate_associations,
)
from dashboard.providers.transit import CTB_STOPS, GMB_STOPS, KMB_STOPS

RouteKey = tuple[str, str, str]

# The diagnostic deliberately uses the renderer's logical (2x) viewport.  It
# must remain independent of a particular raster candidate or image capture.
_MAP_LAT, _MAP_LON, _MAP_ZOOM = 22.3274138, 114.2331738, 14.0
_TILE_SIZE = 256.0
_GMB_PAIR_MARKER_LIMIT = 100
_GMB_PAIR_OBSERVATION_LIMIT = 400
_GMB_PAIR_RESULT_LIMIT = 25


def _logical_anchor(lat: object, lon: object) -> tuple[float, float]:
    scale = _TILE_SIZE * 2**_MAP_ZOOM

    def mercator(value: object) -> float:
        return (1 - math.asinh(math.tan(math.radians(float(value)))) / math.pi) / 2

    return (
        (float(lon) - _MAP_LON) / 360 * scale + 1920 / 2,
        (mercator(lat) - mercator(_MAP_LAT)) * scale + 1080 / 2,
    )


def audit_gmb_marker_pairs(
    probe_etas: Sequence[object] = (),
    authoritative_etas: Sequence[object] = (),
    estimates: Sequence[object] = (),
    *,
    max_distance: float = 20.0,
    min_eta_delta: float = 10.0,
) -> list[dict[str, Any]]:
    """Find close GMB marker pairs backed by widely-separated common ETAs.

    This is intentionally a bounded, no-I/O plausibility diagnostic.  A
    source observation token is ``(kind, input index)`` as retained by the
    estimator; matching stop indices across the two marker provenances avoids
    treating merely similar labels or positions as duplicate evidence.
    """
    rows: dict[tuple[str, int], object] = {}
    for kind, source in (("probe", probe_etas), ("gate", authoritative_etas)):
        for index, row in enumerate(source):
            if index >= _GMB_PAIR_OBSERVATION_LIMIT:
                break
            if getattr(row, "minutes", None) is not None:
                rows[(kind, index)] = row
    grouped: dict[RouteKey, list[tuple[int, object, tuple[float, float]]]] = defaultdict(list)
    for marker_id, marker in enumerate(estimates):
        if marker_id >= _GMB_PAIR_MARKER_LIMIT or _operator(getattr(marker, "operator", "")) != "GMB":
            continue
        try:
            anchor = _logical_anchor(marker.lat, marker.lon)
        except (AttributeError, TypeError, ValueError):
            continue
        grouped[_key(marker)].append((marker_id, marker, anchor))

    findings: list[dict[str, Any]] = []

    def observations_by_stop(marker: object, key: RouteKey) -> dict[int, list[tuple[tuple[str, int], object]]]:
        grouped_rows: dict[int, list[tuple[tuple[str, int], object]]] = defaultdict(list)
        for raw_token in sorted(getattr(marker, "source_observations", ())):
            token = tuple(raw_token)
            row = rows.get(token)
            if row is None or _key(row) != key:
                continue
            try:
                stop_index = int(row.index)
            except (AttributeError, TypeError, ValueError):
                continue
            grouped_rows[stop_index].append((token, row))
        return grouped_rows

    for key in sorted(grouped):
        members = grouped[key]
        for left_index, (left_id, left, left_anchor) in enumerate(members):
            candidates = members[left_index + 1:]
            # A merged marker can own multiple departures from one stop; treat
            # those observations as a self-pair at zero pixel distance.
            left_rows = observations_by_stop(left, key)
            self_candidates = [
                (left_id, left, left_anchor, left_rows, True)
            ] if any(len(items) > 1 for items in left_rows.values()) else []
            for right_id, right, right_anchor in candidates:
                right_rows = observations_by_stop(right, key)
                yield_pair = (right_id, right_anchor, right_rows)
                for candidate in (yield_pair,):
                    candidate_id, candidate_anchor, candidate_rows = candidate
                    is_self = False
                    distance = math.hypot(left_anchor[0] - candidate_anchor[0], left_anchor[1] - candidate_anchor[1])
                    if distance > max_distance:
                        continue
                    common = []
                    for stop_index in sorted(set(left_rows) & set(candidate_rows)):
                        left_observations = left_rows[stop_index]
                        right_observations = candidate_rows[stop_index]
                        row_pairs = (
                            (left_observations[a], left_observations[b])
                            for a in range(len(left_observations))
                            for b in range(a + 1, len(left_observations))
                        ) if is_self else (
                            (left_observations[a], right_observations[b])
                            for a in range(len(left_observations))
                            for b in range(len(right_observations))
                        )
                        for (left_token, left_row), (right_token, right_row) in row_pairs:
                            delta = abs(float(left_row.minutes) - float(right_row.minutes))
                            if delta < min_eta_delta:
                                continue
                            common.append({
                                "index": stop_index,
                                "eta_values": [float(left_row.minutes), float(right_row.minutes)],
                                "eta_delta": delta,
                                "source_observations": [left_token, right_token],
                            })
                    if not common:
                        continue
                    classification = "stacked" if distance <= 8.0 else "nearby"
                    primary = common[0]
                    findings.append({
                        "key": key,
                        "marker_ids": [left_id, candidate_id],
                        "positions": [left_anchor, candidate_anchor],
                        "pixel_distance": distance,
                        "classification": classification,
                        "common_stops": common,
                        "common_stop_indices": [item["index"] for item in common],
                        "common_stop_index": primary["index"],
                        "eta_values": primary["eta_values"],
                        "eta_delta": primary["eta_delta"],
                        "source_observations": primary["source_observations"],
                    })
                    if len(findings) >= _GMB_PAIR_RESULT_LIMIT:
                        return findings
            for candidate_id, _candidate_marker, candidate_anchor, _candidate_rows, _is_self in self_candidates:
                distance = 0.0
                common = []
                for stop_index, observations in sorted(left_rows.items()):
                    for a in range(len(observations)):
                        for b in range(a + 1, len(observations)):
                            left_token, left_row = observations[a]
                            right_token, right_row = observations[b]
                            delta = abs(float(left_row.minutes) - float(right_row.minutes))
                            if delta >= min_eta_delta:
                                common.append({
                                    "index": stop_index,
                                    "eta_values": [float(left_row.minutes), float(right_row.minutes)],
                                    "eta_delta": delta,
                                    "source_observations": [left_token, right_token],
                                })
                if common:
                    primary = common[0]
                    findings.append({
                        "key": key,
                        "marker_ids": [left_id, candidate_id],
                        "positions": [left_anchor, candidate_anchor],
                        "pixel_distance": distance,
                        "classification": "stacked",
                        "common_stops": common,
                        "common_stop_indices": [item["index"] for item in common],
                        "common_stop_index": primary["index"],
                        "eta_values": primary["eta_values"],
                        "eta_delta": primary["eta_delta"],
                        "source_observations": primary["source_observations"],
                    })
                    if len(findings) >= _GMB_PAIR_RESULT_LIMIT:
                        return findings
            continue
    return findings


def _operator(value: object) -> str:
    text = str(getattr(value, "value", value)).strip().upper()
    return {
        "CITYBUS": "CTB",
        "OPERATOR.CITYBUS": "CTB",
        "OPERATOR.KMB": "KMB",
        "OPERATOR.GMB": "GMB",
    }.get(text, text)


def _key(value: object) -> RouteKey:
    return (
        _operator(getattr(value, "operator", "")),
        str(getattr(value, "route", "")),
        str(getattr(value, "bound", "") or ""),
    )


def _kind(row: object) -> str:
    value = getattr(row, "kind", None)
    return str(getattr(value, "value", value)).lower()


def _raw_position(row: object) -> float:
    return float(row.index) - float(row.minutes) / 2.0


def _match(
    source: Sequence[float],
    markers: Sequence[float],
    tolerance: float = 2.0,
) -> dict[str, Any]:
    """Maximise matches, then minimise total error, preserving input indices."""
    ordered_source = sorted(
        ((float(value), index) for index, value in enumerate(source)),
        key=lambda item: (item[0], item[1]),
    )
    ordered_markers = sorted(
        ((float(value), index) for index, value in enumerate(markers)),
        key=lambda item: (item[0], item[1]),
    )
    source_count = len(ordered_source)
    marker_count = len(ordered_markers)
    # State: matched count, total error, pairs of ORIGINAL input indices.
    states: list[list[tuple[int, float, tuple[tuple[int, int], ...]]]] = [
        [(0, 0.0, ()) for _ in range(marker_count + 1)]
        for _ in range(source_count + 1)
    ]
    for source_index in range(1, source_count + 1):
        for marker_index in range(1, marker_count + 1):
            choices = [
                states[source_index - 1][marker_index],
                states[source_index][marker_index - 1],
            ]
            error = abs(
                ordered_source[source_index - 1][0]
                - ordered_markers[marker_index - 1][0]
            )
            if error <= tolerance:
                previous = states[source_index - 1][marker_index - 1]
                pair = (
                    ordered_source[source_index - 1][1],
                    ordered_markers[marker_index - 1][1],
                )
                choices.append(
                    (previous[0] + 1, previous[1] + error, previous[2] + (pair,))
                )
            states[source_index][marker_index] = min(
                choices,
                key=lambda state: (-state[0], state[1], state[2]),
            )

    cardinality, total_error, pairs = states[source_count][marker_count]
    used_source = {source_index for source_index, _ in pairs}
    used_markers = {marker_index for _, marker_index in pairs}
    return {
        "cardinality": cardinality,
        "error": total_error,
        "pairs": [(float(source[i]), float(markers[j])) for i, j in pairs],
        "pair_indices": list(pairs),
        "source_indices": [i for i, _ in pairs],
        "marker_indices": [j for _, j in pairs],
        "unmatched_source": [i for i in range(len(source)) if i not in used_source],
        "unmatched_markers": [
            i for i in range(len(markers)) if i not in used_markers
        ],
        "unmatched_source_values": [
            float(source[i]) for i in range(len(source)) if i not in used_source
        ],
        "unmatched_marker_values": [
            float(markers[i]) for i in range(len(markers)) if i not in used_markers
        ],
        "max_error": max(
            (abs(float(source[i]) - float(markers[j])) for i, j in pairs),
            default=None,
        ),
    }


def _verified_gate_index(line: object) -> int | None:
    """Return the exact mapped HKUST stop occurrence on an official line."""
    operator, route, bound = _key(line)
    stop_id: str | None = None
    if operator == "KMB":
        gate = {"outbound": "S", "inbound": "N"}.get(bound.lower())
        stop_id = next(
            (
                str(spec["stop"])
                for spec in KMB_STOPS
                if spec["route"] == route and spec["gate"] == gate
            ),
            None,
        )
    elif operator == "CTB":
        gate = {"outbound": "O", "inbound": "I"}.get(bound.lower())
        stop_id = next(
            (
                str(spec["stop"])
                for spec in CTB_STOPS
                if spec["route"] == route and spec["gate"] == gate
            ),
            None,
        )
    elif operator == "GMB" and bound.lower().startswith("seq-"):
        try:
            sequence = int(bound.removeprefix("seq-"))
        except ValueError:
            return None
        for candidate_stop, mappings in GMB_STOPS.items():
            if any(
                mapped_route == route and mapped_sequence == sequence
                for mapped_route, _destination, _gate, _route_id, mapped_sequence
                in mappings
            ):
                stop_id = str(candidate_stop)
                break
    if stop_id is None:
        return None
    # Circular GMB 104 contains HKUST at both ends. The mapped gate feed is the
    # departure occurrence, so taking the first occurrence prevents wraparound.
    return next(
        (
            index
            for index, stop in enumerate(getattr(line, "stops", ()))
            if str(getattr(stop, "stop_id", "")) == stop_id
        ),
        None,
    )


@dataclass(frozen=True)
class _Evidence:
    row: object
    row_index: int
    checkpoint: int
    minutes: float
    raw_position: float
    scheduled: bool
    timing_tolerance: float | None = None


def _checkpoint_check(
    *,
    key: RouteKey,
    checkpoint: int,
    expected: Sequence[_Evidence],
    excluded: Sequence[_Evidence],
    markers: Sequence[tuple[int, object]],
    kind: str,
    tolerance: float,
    timing_probe_tokens: Mapping[int, set[tuple[str, int]]] | None = None,
) -> tuple[dict[str, Any], set[int]]:
    """Compare one exact official stop occurrence with in-horizon markers."""
    source_rows = [row for row in expected if row.checkpoint == checkpoint]
    excluded_rows = [row for row in excluded if row.checkpoint == checkpoint]
    if not source_rows:
        return (
            {
                "key": key,
                "kind": kind,
                "checkpoint": checkpoint,
                "ok": True,
                "inconclusive": True,
                "reason": (
                    "only future or uncorroborated scheduled rows"
                    if excluded_rows
                    else "no source rows"
                ),
                "excluded_rows": len(excluded_rows),
            },
            set(),
        )

    has_provenance = any(
        bool(getattr(marker, "source_observations", ()))
        for _marker_id, marker in markers
    )
    if has_provenance:
        matched_marker_ids: set[int] = set()
        matched_pairs: list[tuple[int, int]] = []
        unmatched_source: list[int] = []
        duplicate_sources: list[int] = []
        marker_source_counts: dict[int, int] = defaultdict(int)
        timing_deltas: list[float] = []
        timing_outliers: list[int] = []
        superseded_position_observations: list[tuple[str, int]] = []
        passed_checkpoint = 0
        for source_index, row in enumerate(source_rows):
            token = ("probe", row.row_index)
            owners = [
                (marker_id, marker)
                for marker_id, marker in markers
                if token in getattr(marker, "source_observations", ())
            ]
            if len(owners) != 1:
                unmatched_source.append(source_index)
                if len(owners) > 1:
                    duplicate_sources.append(source_index)
                continue
            marker_id, marker = owners[0]
            marker_source_counts[marker_id] += 1
            matched_marker_ids.add(marker_id)
            matched_pairs.append((source_index, marker_id))
            position = float(marker.position)
            if position > checkpoint:
                # A later stop can be fresher than this checkpoint. The row
                # still proves vehicle identity, but its countdown is no
                # longer an independent position measurement.
                passed_checkpoint += 1
            elif (
                timing_probe_tokens is not None
                and token not in timing_probe_tokens.get(marker_id, set())
            ):
                # A probe row owned by the marker is still identity evidence,
                # but only the effective position anchor may be used for
                # timing.  This matters for probe-only ladder markers, whose
                # final position is derived from the freshest row.
                superseded_position_observations.append(token)
            else:
                timing_delta = abs(row.minutes - (checkpoint - position) * 2.0)
                timing_deltas.append(timing_delta)
                allowed_tolerance = (
                    row.timing_tolerance
                    if row.timing_tolerance is not None
                    else tolerance
                )
                if timing_delta > allowed_tolerance:
                    timing_outliers.append(source_index)
        duplicate_markers = sorted(
            marker_id
            for marker_id, count in marker_source_counts.items()
            if count > 1
        )
        ok = (
            not unmatched_source
            and not duplicate_markers
            and not timing_outliers
        )
        match = {
            "cardinality": len(matched_pairs),
            "pair_indices": matched_pairs,
            "unmatched_source": unmatched_source,
            "unmatched_markers": duplicate_markers,
            "unmatched_source_values": [
                source_rows[index].minutes for index in unmatched_source
            ],
            "unmatched_source_observations": [
                ("probe", source_rows[index].row_index)
                for index in unmatched_source
            ],
            "unmatched_marker_values": [
                float(dict(markers)[marker_id].position)
                for marker_id in duplicate_markers
            ],
            "duplicate_sources": duplicate_sources,
            "timing_outliers": timing_outliers,
            "timing_outlier_values": [
                source_rows[index].minutes for index in timing_outliers
            ],
            "superseded_position_observations": superseded_position_observations,
            "superseded_position_count": len(superseded_position_observations),
            "max_error": max(timing_deltas, default=None),
        }
        return (
            {
                "key": key,
                "kind": kind,
                "checkpoint": checkpoint,
                "ok": ok,
                "inconclusive": False,
                "source_count": len(source_rows),
                "marker_count": len(matched_marker_ids),
                "matched_marker_ids": sorted(matched_marker_ids),
                "match": match,
                "excluded_rows": len(excluded_rows),
                "passed_checkpoint": passed_checkpoint,
                "max_timing_delta": max(timing_deltas, default=None),
                "reason": (
                    "source ETA and marker position exceed tolerance"
                    if timing_outliers
                    else None
                ),
            },
            matched_marker_ids,
        )

    horizon = max(row.minutes for row in source_rows)
    eligible: list[tuple[int, object]] = []
    marker_minutes: list[float] = []
    for marker_id, marker in markers:
        position = float(marker.position)
        minutes = (checkpoint - position) * 2.0
        if position <= checkpoint + 1e-9 and -1e-9 <= minutes <= horizon + tolerance:
            eligible.append((marker_id, marker))
            marker_minutes.append(max(0.0, minutes))

    match = _match([row.minutes for row in source_rows], marker_minutes, tolerance)
    matched_marker_ids = {
        eligible[marker_index][0]
        for _source_index, marker_index in match["pair_indices"]
    }
    ok = not match["unmatched_source"] and not match["unmatched_markers"]
    return (
        {
            "key": key,
            "kind": kind,
            "checkpoint": checkpoint,
            "ok": ok,
            "inconclusive": False,
            "source_count": len(source_rows),
            "marker_count": len(eligible),
            "marker_ids": [marker_id for marker_id, _marker in eligible],
            "matched_marker_ids": sorted(matched_marker_ids),
            "match": match,
            "excluded_rows": len(excluded_rows),
        },
        matched_marker_ids,
    )


def audit_marker_positions(
    probe_etas: Sequence[object] = (),
    authoritative_etas: Sequence[object] = (),
    estimates: Sequence[object] = (),
    route_lines: Sequence[object] = (),
    frame_id: int = 0,
    seed: object | None = None,
    tolerance: float = 2.0,
) -> dict[str, Any]:
    """Audit all marker/source relationships represented by one map frame."""
    lines = {_key(line): line for line in route_lines}
    markers_by_key: dict[RouteKey, list[tuple[int, object]]] = defaultdict(list)
    for marker_id, marker in enumerate(estimates):
        key = _key(marker)
        if key in lines and getattr(marker, "position", None) is not None:
            markers_by_key[key].append((marker_id, marker))

    evidence_by_key: dict[RouteKey, list[_Evidence]] = defaultdict(list)
    excluded_by_key: dict[RouteKey, list[_Evidence]] = defaultdict(list)
    probe_rows_by_key: dict[RouteKey, list[_Evidence]] = defaultdict(list)
    invalid_probe_rows = 0
    for row_index, row in enumerate(probe_etas):
        if getattr(row, "minutes", None) is None or _kind(row) == "unavailable":
            continue
        key = _key(row)
        if key not in lines:
            continue
        try:
            evidence = _Evidence(
                row=row,
                row_index=row_index,
                checkpoint=int(row.index),
                minutes=max(0.0, float(row.minutes)),
                raw_position=_raw_position(row),
                scheduled=_kind(row) == "scheduled",
            )
        except (TypeError, ValueError):
            invalid_probe_rows += 1
            continue
        probe_rows_by_key[key].append(evidence)

    authoritative_input_index: dict[int, int] = {}
    indexed_authoritative_by_key: dict[
        RouteKey, list[tuple[int, object]]
    ] = defaultdict(list)
    for row_index, row in enumerate(authoritative_etas):
        if getattr(row, "minutes", None) is None or _key(row) not in lines:
            continue
        authoritative_input_index[id(row)] = row_index
        indexed_authoritative_by_key[_key(row)].append((row_index, row))

    gate_plan = _plan_gate_associations(
        list(probe_etas),
        list(authoritative_etas),
        set(lines),
    )
    departed_gate_inputs = gate_plan.departed_gate_inputs
    undeparted_probe_inputs = gate_plan.undeparted_probe_inputs

    for key, rows in probe_rows_by_key.items():
        gate_index = gate_plan.verified_gate_index.get(key)
        has_gate_rows = key in gate_plan.gate_rows_by_direction
        effective_rows: list[_Evidence] = []
        gate_assigned_rows: set[int] = set()
        for row in rows:
            if row.row_index in undeparted_probe_inputs:
                excluded_by_key[key].append(row)
                continue
            if row.row_index in gate_plan.gate_assignment:
                # A departed authoritative journey owns all of its associated
                # stop observations, even when their coarse two-minute model
                # still lies before the route origin.
                gate_assigned_rows.add(row.row_index)
                timing_tolerance = (
                    GATE_DOWNSTREAM_DRIFT_MINUTES
                    if gate_index is not None and row.checkpoint > gate_index
                    else GATE_UPSTREAM_DRIFT_MINUTES
                )
                effective_rows.append(
                    replace(row, timing_tolerance=timing_tolerance)
                )
                continue
            if has_gate_rows:
                if row.row_index not in gate_plan.passed_probe_rows:
                    excluded_by_key[key].append(row)
                    continue
                effective_position = gate_plan.passed_probe_positions.get(
                    row.row_index, row.raw_position
                )
                if gate_index is None or effective_position <= gate_index:
                    excluded_by_key[key].append(row)
                    continue
                timing_tolerance = (
                    GATE_DOWNSTREAM_DRIFT_MINUTES
                    if row.row_index in gate_plan.passed_probe_positions
                    and row.checkpoint > gate_index
                    else GATE_UPSTREAM_DRIFT_MINUTES
                    if row.row_index in gate_plan.passed_probe_positions
                    else None
                )
                effective_rows.append(
                    replace(
                        row,
                        raw_position=effective_position,
                        timing_tolerance=timing_tolerance,
                    )
                )
                continue
            if row.raw_position < 0:
                excluded_by_key[key].append(row)
                continue
            effective_rows.append(row)

        for row in effective_rows:
            if not row.scheduled:
                evidence_by_key[key].append(row)
                continue
            track_id = gate_plan.passed_track_ids.get(row.row_index)
            corroborated = row.row_index in gate_assigned_rows or any(
                other.checkpoint != row.checkpoint
                and (
                    gate_plan.passed_track_ids.get(other.row_index) == track_id
                    if track_id is not None
                    else gate_plan.passed_track_ids.get(other.row_index) is None
                    and abs(other.raw_position - row.raw_position) <= 2.2
                )
                for other in effective_rows
            )
            if corroborated:
                evidence_by_key[key].append(row)
            else:
                excluded_by_key[key].append(row)

    # Select the one effective probe observation that anchors each marker's
    # final position.  A marker with gate provenance is anchored by its gate;
    # its probes remain identity evidence only.  For probe-only markers, use
    # the closest effective probe position, with token order as a stable tie
    # breaker.
    timing_probe_tokens_by_key: dict[
        RouteKey, dict[int, set[tuple[str, int]]]
    ] = defaultdict(dict)
    effective_probe_by_key: dict[RouteKey, dict[int, _Evidence]] = defaultdict(dict)
    for key, rows in evidence_by_key.items():
        for row in rows:
            effective_probe_by_key[key][row.row_index] = row
    for key, markers in markers_by_key.items():
        available = effective_probe_by_key[key]
        for marker_id, marker in markers:
            observations = getattr(marker, "source_observations", ())
            gate_tokens = {
                int(token[1])
                for token in observations
                if len(token) == 2
                and str(token[0]) == "gate"
                and str(token[1]).lstrip("-").isdigit()
            }
            if gate_tokens and not gate_tokens.intersection(
                gate_plan.passed_gate_inputs
            ):
                timing_probe_tokens_by_key[key][marker_id] = set()
                continue
            candidates = [
                row for token in observations
                if len(token) == 2
                and token[0] == "probe"
                and (row := available.get(int(token[1]))) is not None
            ]
            if candidates:
                anchor = min(
                    candidates,
                    key=lambda row: (
                        abs(row.raw_position - float(marker.position)),
                        row.row_index,
                    ),
                )
                timing_probe_tokens_by_key[key][marker_id] = {
                    ("probe", anchor.row_index)
                }

    authoritative_by_key: dict[RouteKey, list[object]] = defaultdict(list)
    for row in authoritative_etas:
        if getattr(row, "minutes", None) is not None and _key(row) in lines:
            authoritative_by_key[_key(row)].append(row)

    active_keys = sorted(
        set(markers_by_key) | set(probe_rows_by_key) | set(authoritative_by_key)
    )
    checks: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    stats: dict[str, int] = {
        "directions": len(active_keys),
        "markers": len(estimates),
        "matched": 0,
        "excluded_undeparted": 0,
        "excluded_probe_rows": sum(len(rows) for rows in excluded_by_key.values()),
        "invalid_probe_rows": invalid_probe_rows,
        "sampled": 0,
        "inconclusive": 0,
    }

    def record(check: dict[str, Any]) -> None:
        checks.append(check)
        if check.get("inconclusive"):
            stats["inconclusive"] += 1
        if not check.get("ok", False) and not check.get("inconclusive", False):
            issues.append(
                {
                    "key": check["key"],
                    "kind": check["kind"],
                    "detail": {
                        name: value
                        for name, value in check.items()
                        if name not in {"key", "kind"}
                    },
                }
            )

    for key in active_keys:
        line = lines[key]
        markers = markers_by_key[key]
        marker_lookup = dict(markers)
        gate = _verified_gate_index(line)
        authoritative_rows = authoritative_by_key[key]

        if gate is None:
            record(
                {
                    "key": key,
                    "kind": "authoritative",
                    "ok": True,
                    "inconclusive": True,
                    "reason": "no verified gate occurrence in current geometry",
                }
            )
        elif not authoritative_rows:
            record(
                {
                    "key": key,
                    "kind": "authoritative",
                    "gate_index": gate,
                    "ok": True,
                    "inconclusive": True,
                    "reason": "no authoritative HKUST rows",
                    "pre_gate_markers": sum(
                        float(marker.position) <= gate
                        for _marker_id, marker in markers
                    ),
                }
            )
        else:
            inconsistent = [
                int(row.index)
                for row in authoritative_rows
                if int(row.index) != gate
            ]
            if inconsistent:
                record(
                    {
                        "key": key,
                        "kind": "authoritative-index",
                        "gate_index": gate,
                        "actual_indices": inconsistent,
                        "ok": False,
                        "inconclusive": False,
                    }
                )
            valid_gate_rows = [
                row
                for row in authoritative_rows
                if int(row.index) == gate
            ]
            departed_gate_rows = [
                row
                for row in valid_gate_rows
                if authoritative_input_index.get(id(row)) in departed_gate_inputs
            ]
            stats["excluded_undeparted"] += len(valid_gate_rows) - len(
                departed_gate_rows
            )
            pre_gate_markers = [
                (marker_id, marker)
                for marker_id, marker in markers
                if float(marker.position) <= gate + 1e-9
            ]
            departed_gate_pairs = [
                (authoritative_input_index[id(row)], row)
                for row in departed_gate_rows
            ]
            has_gate_provenance = any(
                bool(getattr(marker, "source_observations", ()))
                for _marker_id, marker in markers
            )
            if has_gate_provenance:
                matched_pairs: list[tuple[int, int]] = []
                unmatched_source: list[int] = []
                duplicate_sources: list[int] = []
                marker_gate_counts: dict[int, int] = defaultdict(int)
                timing_deltas: list[float] = []
                for source_index, (gate_input, row) in enumerate(
                    departed_gate_pairs
                ):
                    token = ("gate", gate_input)
                    passed_gate = gate_input in gate_plan.passed_gate_inputs
                    gate_candidates = markers if passed_gate else pre_gate_markers
                    owners = [
                        (marker_id, marker)
                        for marker_id, marker in gate_candidates
                        if token in getattr(marker, "source_observations", ())
                    ]
                    if len(owners) != 1:
                        unmatched_source.append(source_index)
                        if len(owners) > 1:
                            duplicate_sources.append(source_index)
                        continue
                    marker_id, marker = owners[0]
                    marker_gate_counts[marker_id] += 1
                    matched_pairs.append((source_index, marker_id))
                    if not passed_gate:
                        timing_deltas.append(
                            abs(
                                float(row.minutes)
                                - max(
                                    0.0,
                                    (gate - float(marker.position)) * 2.0,
                                )
                            )
                        )
                gate_marker_ids = {
                    marker_id for marker_id, _marker in pre_gate_markers
                }
                for gate_input, _row in departed_gate_pairs:
                    if gate_input not in gate_plan.passed_gate_inputs:
                        continue
                    token = ("gate", gate_input)
                    gate_marker_ids.update(
                        marker_id
                        for marker_id, marker in markers
                        if token in getattr(marker, "source_observations", ())
                    )
                unmatched_markers = sorted(
                    marker_id
                    for marker_id in gate_marker_ids
                    if marker_gate_counts.get(marker_id, 0) != 1
                )
                match = {
                    "cardinality": len(matched_pairs),
                    "pair_indices": matched_pairs,
                    "unmatched_source": unmatched_source,
                    "unmatched_markers": unmatched_markers,
                    "unmatched_source_values": [
                        float(departed_gate_pairs[index][1].minutes)
                        for index in unmatched_source
                    ],
                    "unmatched_source_observations": [
                        ("gate", departed_gate_pairs[index][0])
                        for index in unmatched_source
                    ],
                    "unmatched_marker_values": [
                        float(marker_lookup[marker_id].position)
                        for marker_id in unmatched_markers
                    ],
                    "duplicate_sources": duplicate_sources,
                    "max_error": max(timing_deltas, default=None),
                }
                matched_marker_ids = [
                    marker_id for _source_index, marker_id in matched_pairs
                ]
            else:
                marker_minutes = [
                    max(0.0, (gate - float(marker.position)) * 2.0)
                    for _marker_id, marker in pre_gate_markers
                ]
                match = _match(
                    [float(row.minutes) for row in departed_gate_rows],
                    marker_minutes,
                    tolerance,
                )
                matched_marker_ids = [
                    pre_gate_markers[marker_index][0]
                    for _source_index, marker_index in match["pair_indices"]
                ]
            gate_check = {
                "key": key,
                "kind": "authoritative",
                "gate_index": gate,
                "ok": not match["unmatched_source"]
                and not match["unmatched_markers"],
                "inconclusive": False,
                "source_count": len(departed_gate_rows),
                "marker_count": len(pre_gate_markers),
                "matched_marker_ids": matched_marker_ids,
                "excluded_undeparted": len(valid_gate_rows)
                - len(departed_gate_rows),
                "match": match,
                "max_timing_delta": match.get("max_error"),
            }
            stats["matched"] += match["cardinality"]
            record(gate_check)

        # Every later observed occurrence is independently one-to-one. These
        # checks include incoming and already-past-HKUST vehicles; matched
        # identities then specifically prove each post-HKUST marker.
        covered_post_markers: set[int] = set()
        if gate is not None:
            later_checkpoints = sorted(
                {
                    row.checkpoint
                    for row in probe_rows_by_key[key]
                    if row.checkpoint > gate
                }
            )
            for checkpoint in later_checkpoints:
                check, matched_marker_ids = _checkpoint_check(
                    key=key,
                    checkpoint=checkpoint,
                    expected=evidence_by_key[key],
                    excluded=excluded_by_key[key],
                    markers=markers,
                    kind="checkpoint",
                    tolerance=tolerance,
                    timing_probe_tokens=timing_probe_tokens_by_key[key],
                )
                stats["matched"] += check.get("match", {}).get("cardinality", 0)
                record(check)
                covered_post_markers.update(
                    marker_id
                    for marker_id in matched_marker_ids
                    if float(marker_lookup[marker_id].position) > gate
                )

            for marker_id, marker in markers:
                position = float(marker.position)
                if position <= gate + 1e-9:
                    continue
                if marker_id in covered_post_markers:
                    record(
                        {
                            "key": key,
                            "kind": "downstream",
                            "marker_id": marker_id,
                            "position": position,
                            "ok": True,
                            "inconclusive": False,
                        }
                    )
                    continue
                nearby_excluded = [
                    row
                    for row in excluded_by_key[key]
                    if row.scheduled
                    and row.raw_position >= 0
                    and row.checkpoint > gate
                    and position <= row.checkpoint + 1e-9
                    and abs(row.minutes - (row.checkpoint - position) * 2.0)
                    <= tolerance
                ]
                record(
                    {
                        "key": key,
                        "kind": "downstream",
                        "marker_id": marker_id,
                        "position": position,
                        "ok": bool(nearby_excluded),
                        "inconclusive": bool(nearby_excluded),
                        "reason": (
                            "only uncorroborated scheduled evidence"
                            if nearby_excluded
                            else "no unique later-stop ETA match"
                        ),
                        "source_observations": sorted(
                            getattr(marker, "source_observations", ())
                        ),
                    }
                )

        # One reproducible random non-gate source occurrence per active route.
        sample_candidates = sorted(
            {
                row.checkpoint
                for row in probe_rows_by_key[key]
                if gate is None or row.checkpoint != gate
            }
        )
        if sample_candidates:
            checkpoint = random.Random(
                f"{seed!r}:{frame_id}:{key[0]}:{key[1]}:{key[2]}"
            ).choice(sample_candidates)
            sample, _matched_marker_ids = _checkpoint_check(
                key=key,
                checkpoint=checkpoint,
                expected=evidence_by_key[key],
                excluded=excluded_by_key[key],
                markers=markers,
                kind="sample",
                tolerance=tolerance,
                timing_probe_tokens=timing_probe_tokens_by_key[key],
            )
            stats["sampled"] += 1
            stats["matched"] += sample.get("match", {}).get("cardinality", 0)
            record(sample)
        elif probe_rows_by_key[key]:
            record(
                {
                    "key": key,
                    "kind": "sample",
                    "ok": True,
                    "inconclusive": True,
                    "reason": "no observed non-gate occurrence",
                }
            )

    for issue in issues:
        key = issue["key"]
        detail = issue["detail"]
        checkpoint = detail.get("checkpoint")
        detail["gate_rows"] = [
            (
                input_index,
                int(row.index),
                float(row.minutes),
                _kind(row),
            )
            for input_index, row in indexed_authoritative_by_key[key]
        ]
        detail["checkpoint_rows"] = [
            (
                ("probe", row.row_index),
                row.minutes,
                "scheduled" if row.scheduled else _kind(row.row),
                row.raw_position,
            )
            for row in probe_rows_by_key[key]
            if checkpoint is not None and row.checkpoint == checkpoint
        ]
        detail["route_markers"] = [
            (
                marker_id,
                float(marker.position),
                sorted(
                    getattr(marker, "source_observations", ())
                ),
            )
            for marker_id, marker in markers_by_key[key]
        ]

    gmb_marker_pairs = audit_gmb_marker_pairs(
        probe_etas, authoritative_etas, estimates
    )
    return {
        "checks": checks,
        "issues": issues,
        "gmb_marker_pairs": gmb_marker_pairs,
        "stats": stats,
        "ok": not issues,
    }


__all__ = ["audit_gmb_marker_pairs", "audit_marker_positions"]
