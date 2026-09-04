"""Estimated vehicle positions from stop ETAs on official route geometry.

Each tracked direction is an official ordered stop sequence with a validated
HKeMobility road path. Probe rows first form vehicle ladders around the coarse
``stop index - ETA / 2`` position. Around HKUST, ordered stop arrivals are
associated one-to-one with authoritative gate arrivals so variable real travel
times cannot split one journey into several markers. Unmatched live arrivals
after HKUST remain independent passed vehicles. Final positions are
arclength-interpolated on the matching official direction.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping
from dataclasses import dataclass, replace
from statistics import median

from dashboard.models import EtaKind, Operator


def _passed_row_position(
    raw_position: float,
    gate_index: int,
    probe_input_index: int,
    passed_probe_rows: Collection[int],
    passed_probe_positions: Mapping[int, float],
) -> float | None:
    """Return the corrected position for a probe row known to be past HKUST.

    A row whose coarse ETA already places it beyond the gate is authoritative
    for that fact and keeps its raw position.  Otherwise only rows explicitly
    classified as passed may use the gate-order correction.  Positions that do
    not end up strictly beyond the gate are not eligible for rendering or
    downstream audit proof.
    """
    if raw_position > gate_index:
        return raw_position
    if probe_input_index not in passed_probe_rows:
        return None
    corrected = passed_probe_positions.get(probe_input_index, raw_position)
    return corrected if corrected > gate_index else None


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
    # Official stop occurrences which contributed to this marker.  Separate
    # BusEstimate objects retain vehicle multiplicity when two vehicles share
    # the same occurrence; this set only records each vehicle's evidence.
    source_indices: frozenset[int] = frozenset()
    # Exact rows in the estimator inputs (kind, zero-based input offset).  Stop
    # indices alone cannot distinguish two vehicles reported at one stop.
    source_observations: frozenset[tuple[str, int]] = frozenset()
    # Stable route vocabulary and temporal identity, populated by MarkerTracker.
    operator_code: str = ""
    track_id: int | None = None
    bracket: tuple[float, float] | None = None
    eta_minutes: float | None = None
    eta_arrival_at: object | None = None
    bracket_initial_eta: float | None = None
    boundary_age_seconds: float | None = None
    # Stop occurrences whose current rows form the useful refresh frontier for
    # this exact ETA instance: every due/zero rung plus the first positive rung
    # after it.  MarkerTracker combines these with the physical bracket and
    # terminus so a zero plateau advances promptly without sharing evidence
    # between simultaneous vehicles.
    priority_indices: frozenset[int] = frozenset()

    @property
    def bracket_lower(self):
        return self.bracket[0] if self.bracket else None

    @property
    def bracket_upper(self):
        return self.bracket[1] if self.bracket else None


MINUTES_PER_STOP = 2.0
# Exact timestamps can lead the rounded departure state by a few seconds.
# Keep a narrow grace window so a gate-confirmed vehicle is not hidden at the
# instant it leaves, while a genuinely future terminus ETA remains decisive.
TERMINUS_DEPARTURE_GRACE_MINUTES = 0.5
GATE_DOWNSTREAM_DRIFT_MINUTES = 15.0
GATE_UPSTREAM_DRIFT_MINUTES = 25.0
GATE_PASSAGE_SKEW_MINUTES = 3.0
PROBE_FRESHNESS_MARGIN_SECONDS = 5.0
STALE_SINGLETON_MATCH_TOLERANCE_MINUTES = 2.0

# Route-geometry operator codes -> dashboard Operator enum values.
_OPERATOR_BY_CODE = {
    "KMB": Operator.KMB,
    "CTB": Operator.CITYBUS,
    "GMB": Operator.GMB,
}


def _quantize_position(position: float, gate_index: int | None = None) -> int:
    """Quantize a position while preserving its strict side of HKUST."""
    scaled = round(position * 1000)
    if gate_index is not None:
        gate_scaled = gate_index * 1000
        if position > gate_index and scaled <= gate_scaled:
            return gate_scaled + 1
        if position < gate_index and scaled >= gate_scaled:
            return gate_scaled - 1
    return scaled


def _align_gate_arrivals(
    gate_rows: list[tuple[int, object]],
    probe_rows: list[tuple[int, object]],
    *,
    gate_index: int,
    checkpoint: int,
    rank_first: bool = False,
) -> list[tuple[int, int]]:
    """Associate one stop's ordered arrivals with ordered HKUST arrivals.

    Feeds expose no stable vehicle ID across stops. Arrival order is stable,
    though, and the ETA difference should have the same sign as the stop's
    location relative to HKUST. Dynamic programming therefore maximises
    order-preserving cardinality, then favours travel time nearest the coarse
    two-minutes-per-stop expectation. The returned pairs are
    ``(probe_input_index, gate_input_index)``.
    """
    gates = sorted(
        (
            (max(0.0, float(row.minutes)), input_index)
            for input_index, row in gate_rows
            if row.minutes is not None and row.kind is not EtaKind.UNAVAILABLE
        ),
        key=lambda item: (item[0], item[1]),
    )
    probes = sorted(
        (
            (max(0.0, float(row.minutes)), input_index)
            for input_index, row in probe_rows
            if row.minutes is not None and row.kind is not EtaKind.UNAVAILABLE
        ),
        key=lambda item: (item[0], item[1]),
    )
    if not gates or not probes or checkpoint == gate_index:
        return []

    expected_delta = (checkpoint - gate_index) * MINUTES_PER_STOP
    # Gate rows are integral minutes while probe rows retain fractional source
    # timestamps. Their countdown clocks can differ by rounding, but a later
    # stop must not be associated with an ETA several minutes earlier than
    # HKUST; that is an already-passed vehicle, not clock skew.
    clock_skew_minutes = 1.25
    passage_skew_minutes = GATE_PASSAGE_SKEW_MINUTES

    # State: match count, rank displacement, travel-time error, original pairs.
    states: list[list[tuple[int, int, float, tuple[tuple[int, int], ...]]]] = [
        [(0, 0, 0.0, ()) for _ in range(len(probes) + 1)]
        for _ in range(len(gates) + 1)
    ]
    for gate_offset in range(1, len(gates) + 1):
        for probe_offset in range(1, len(probes) + 1):
            choices = [
                states[gate_offset - 1][probe_offset],
                states[gate_offset][probe_offset - 1],
            ]
            gate_minutes, gate_input_index = gates[gate_offset - 1]
            probe_minutes, probe_input_index = probes[probe_offset - 1]
            delta = probe_minutes - gate_minutes
            if checkpoint > gate_index:
                compatible = (
                    max(
                        -passage_skew_minutes,
                        expected_delta - GATE_DOWNSTREAM_DRIFT_MINUTES,
                    )
                    <= delta
                    <= expected_delta + GATE_DOWNSTREAM_DRIFT_MINUTES
                )
            else:
                # Adjacent stops retain the 1.25-minute cache skew; each
                # additional hop requires another 0.25 minutes of physical
                # travel, preventing impossible long-hop near-zero matches.
                upstream_upper = min(
                    clock_skew_minutes,
                    expected_delta + GATE_UPSTREAM_DRIFT_MINUTES,
                ) - max(0, gate_index - checkpoint - 1) * 0.25
                compatible = (
                    expected_delta - GATE_UPSTREAM_DRIFT_MINUTES
                    <= delta <= upstream_upper
                )
            if compatible:
                previous = states[gate_offset - 1][probe_offset - 1]
                choices.append(
                    (
                        previous[0] + 1,
                        previous[1] + abs((gate_offset - 1) - (probe_offset - 1)),
                        previous[2] + abs(delta - expected_delta),
                        previous[3]
                        + ((probe_input_index, gate_input_index),),
                    )
                )
            states[gate_offset][probe_offset] = min(
                choices,
                key=(
                    (lambda state: (-state[0], state[1], state[2], state[3]))
                    if rank_first and len(gates) > 1
                    else (lambda state: (-state[0], state[2], state[3]))
                ),
            )
    return list(states[-1][-1][3])


@dataclass(frozen=True)
class _GateAssociationPlan:
    gate_assignment: dict[int, int]
    passed_probe_rows: frozenset[int]
    passed_probe_positions: dict[int, float]
    passed_track_ids: dict[int, int]
    superseded_probe_inputs: frozenset[int]
    verified_gate_index: dict[tuple[str, str, str], int]
    gate_rows_by_direction: dict[
        tuple[str, str, str], list[tuple[int, object]]
    ]
    departed_gate_inputs: frozenset[int]
    undeparted_probe_inputs: frozenset[int]
    passed_gate_inputs: frozenset[int]


def _plan_gate_associations(
    probe_inputs: list[object],
    authoritative_inputs: list[object],
    route_keys: set[tuple[str, str, str]],
) -> _GateAssociationPlan:
    """Build the shared source-identity plan used by estimator and auditor."""
    probe_rows_by_occurrence: dict[
        tuple[str, str, str], dict[int, list[tuple[int, object]]]
    ] = {}
    gate_rows_by_direction: dict[
        tuple[str, str, str], list[tuple[int, object]]
    ] = {}
    for input_index, eta in enumerate(probe_inputs):
        key = (str(eta.operator), str(eta.route), str(eta.bound))
        if (
            key in route_keys
            and eta.minutes is not None
            and eta.kind is not EtaKind.UNAVAILABLE
        ):
            probe_rows_by_occurrence.setdefault(key, {}).setdefault(
                int(eta.index), []
            ).append((input_index, eta))
    for input_index, eta in enumerate(authoritative_inputs):
        key = (str(eta.operator), str(eta.route), str(eta.bound))
        if (
            key in route_keys
            and eta.minutes is not None
            and eta.kind is not EtaKind.UNAVAILABLE
        ):
            gate_rows_by_direction.setdefault(key, []).append((input_index, eta))

    gate_assignment: dict[int, int] = {}
    passed_probe_rows: set[int] = set()
    passed_probe_positions: dict[int, float] = {}
    passed_gate_inputs: set[int] = set()
    verified_gate_index: dict[tuple[str, str, str], int] = {}
    for key, gate_rows in gate_rows_by_direction.items():
        gate_indices = {int(row.index) for _input_index, row in gate_rows}
        if len(gate_indices) != 1:
            continue
        gate_index = next(iter(gate_indices))
        verified_gate_index[key] = gate_index
        gate_rows_by_input = dict(gate_rows)
        frontier: list[tuple[int, object]] = []
        frontier_gate_assignments: dict[int, int] = {}
        frontier_checkpoint = gate_index
        for checkpoint in sorted(probe_rows_by_occurrence.get(key, {})):
            checkpoint_rows = probe_rows_by_occurrence[key][checkpoint]
            if checkpoint < gate_index:
                for probe_index, gate_input_index in _align_gate_arrivals(
                    gate_rows,
                    checkpoint_rows,
                    gate_index=gate_index,
                    checkpoint=checkpoint,
                ):
                    gate_assignment[probe_index] = gate_input_index
                    frontier_gate_assignments[probe_index] = gate_input_index
                if frontier_gate_assignments:
                    frontier = [
                        (input_index, row)
                        for input_index, row in checkpoint_rows
                        if input_index in frontier_gate_assignments
                    ]
                    frontier_checkpoint = checkpoint
                continue
            if checkpoint == gate_index:
                continue

            reserved: set[int] = set()
            propagated_gate_inputs: set[int] = set()
            fresh_gate_inputs: set[int] = set()
            fresh_pairs = _align_gate_arrivals(
                gate_rows,
                checkpoint_rows,
                gate_index=gate_index,
                checkpoint=checkpoint,
                rank_first=True,
            )
            frontier_pairs = (
                _align_gate_arrivals(
                    frontier,
                    checkpoint_rows,
                    gate_index=frontier_checkpoint,
                    checkpoint=checkpoint,
                )
                if frontier
                else []
            )
            valid_frontier_gate_pairs = 0
            for current_input, previous_input in frontier_pairs:
                previous_gate = frontier_gate_assignments.get(previous_input)
                if previous_gate is None:
                    continue
                if _align_gate_arrivals(
                    [(previous_gate, gate_rows_by_input[previous_gate])],
                    [(current_input, dict(checkpoint_rows)[current_input])],
                    gate_index=gate_index,
                    checkpoint=checkpoint,
                ):
                    valid_frontier_gate_pairs += 1
            frontier_age = max(
                (
                    float(getattr(row, "cache_age_seconds", 0) or 0)
                    for _input_index, row in frontier
                ),
                default=0.0,
            )
            fresh_age = min(
                (
                    float(getattr(row, "cache_age_seconds", 0) or 0)
                    for _input_index, row in checkpoint_rows
                ),
                default=0.0,
            )
            # A fresh snapshot may repair stale frontier overmatching; on
            # equally fresh ladders retain frontier continuity exactly.
            if (
                fresh_age + PROBE_FRESHNESS_MARGIN_SECONDS < frontier_age
                and len(fresh_pairs) > valid_frontier_gate_pairs
            ):
                for current_input, fresh_gate in fresh_pairs:
                    gate_assignment[current_input] = fresh_gate
                    reserved.add(current_input)
                    fresh_gate_inputs.add(fresh_gate)
            # Carry the combined gate-backed/passed identity frontier forward
            # in ETA order. Treating
            # the prior rows as a synthetic checkpoint keeps the same
            # monotone alignment and tolerance rules as gate matching.
            if frontier:
                propagated = _align_gate_arrivals(
                    frontier,
                    checkpoint_rows,
                    gate_index=frontier_checkpoint,
                    checkpoint=checkpoint,
                )
                for current_input, previous_input in propagated:
                    if current_input in reserved:
                        continue
                    previous_gate = frontier_gate_assignments.get(previous_input)
                    current_row = dict(checkpoint_rows)[current_input]
                    if previous_gate is not None and not _align_gate_arrivals(
                        [
                            (previous_gate, gate_rows_by_input[previous_gate])
                        ],
                        [(current_input, current_row)],
                        gate_index=gate_index,
                        checkpoint=checkpoint,
                    ):
                        # Frontier continuity cannot override a fresh check
                        # against the original gate ETA; let normal matching
                        # reconsider this row when a cache refresh jumps.
                        continue
                    if previous_gate is not None and previous_gate in fresh_gate_inputs:
                        continue
                    reserved.add(current_input)
                    if previous_gate is not None:
                        gate_assignment[current_input] = previous_gate
                        propagated_gate_inputs.add(previous_gate)
                    else:
                        passed_probe_rows.add(current_input)
                    previous_row = dict(frontier)[previous_input]
                    previous_position = passed_probe_positions.get(
                        previous_input,
                        int(previous_row.index)
                        - float(previous_row.minutes) / MINUTES_PER_STOP,
                    )
                    projected = previous_position + (
                        int(current_row.index) - int(previous_row.index)
                    ) - (
                        float(current_row.minutes) - float(previous_row.minutes)
                    ) / MINUTES_PER_STOP
                    # Preserve the passed identity even when rounded ETA
                    # clocks briefly imply a position at/before HKUST.
                    if previous_gate is None:
                        passed_probe_positions[current_input] = max(
                            projected, gate_index + 1e-6
                        )

            # A gate is reserved only after its prior identity actually
            # propagated. If propagation failed due to a transient ETA gap,
            # leave that gate available for a fresh compatible match here.
            available_gate_rows = [
                pair
                for pair in gate_rows
                if pair[0] not in propagated_gate_inputs
                and pair[0] not in fresh_gate_inputs
            ]

            # A raw-past row is seeded as an independent passed vehicle only
            # when it cannot plausibly belong to any stale gate arrival.  Keep
            # gate-compatible raw-past rows available for the gate match.
            for input_index, row in checkpoint_rows:
                if input_index in reserved:
                    continue
                raw_position = int(row.index) - float(row.minutes) / MINUTES_PER_STOP
                compatible = bool(
                    _align_gate_arrivals(
                        available_gate_rows,
                        [(input_index, row)],
                        gate_index=gate_index,
                        checkpoint=checkpoint,
                    )
                )
                if raw_position > gate_index and not compatible:
                    passed_probe_rows.add(input_index)
                    reserved.add(input_index)

            remaining_rows = [
                pair for pair in checkpoint_rows if pair[0] not in reserved
            ]
            pairs = _align_gate_arrivals(
                available_gate_rows,
                remaining_rows,
                gate_index=gate_index,
                checkpoint=checkpoint,
                rank_first=True,
            )
            for probe_index, gate_input_index in pairs:
                gate_assignment[probe_index] = gate_input_index

            # Include propagated gate-backed rows when finding the first gate
            # match: an earlier unreserved row must still be eligible for the
            # existing passed-row inference.
            ordered_probe_rows = sorted(
                checkpoint_rows,
                key=lambda item: (max(0.0, float(item[1].minutes)), item[0]),
            )
            matched_ranks = [
                rank for rank, (input_index, _row) in enumerate(ordered_probe_rows)
                if input_index in gate_assignment
            ]
            if matched_ranks:
                first_matched_rank = min(matched_ranks)
                matched_probe_input = ordered_probe_rows[first_matched_rank][0]
                matched_gate_input = gate_assignment[matched_probe_input]
                matched_probe = probe_inputs[matched_probe_input]
                matched_gate = authoritative_inputs[matched_gate_input]
                for input_index, earlier in ordered_probe_rows[:first_matched_rank]:
                    if input_index in gate_assignment or input_index in reserved:
                        continue
                    projected = (
                        int(matched_gate.index)
                        - float(matched_gate.minutes) / MINUTES_PER_STOP
                        + (float(matched_probe.minutes) - float(earlier.minutes))
                        / MINUTES_PER_STOP
                    )
                    if projected > gate_index:
                        passed_probe_rows.add(input_index)
                        reserved.add(input_index)
                        passed_probe_positions[input_index] = projected

            next_frontier = [
                (input_index, row)
                for input_index, row in checkpoint_rows
                if input_index in passed_probe_rows or input_index in gate_assignment
            ]
            if next_frontier:
                frontier = next_frontier
                frontier_gate_assignments = {
                    input_index: gate_assignment[input_index]
                    for input_index, _row in next_frontier
                    if input_index in gate_assignment
                }
                frontier_checkpoint = checkpoint

        downstream_live_is_passed_gates = {
            gate_input_index
            for gate_input_index, gate_row in gate_rows
            if (gate_index == 0 and float(gate_row.minutes) > 0)
            or (
                int(gate_row.index)
                - float(gate_row.minutes) / MINUTES_PER_STOP
                < 0
                and gate_row.kind is EtaKind.SCHEDULED
            )
        }
        for probe_input_index, gate_input_index in list(gate_assignment.items()):
            if gate_input_index not in downstream_live_is_passed_gates:
                continue
            probe = probe_inputs[probe_input_index]
            if (
                (str(probe.operator), str(probe.route), str(probe.bound)) == key
                and int(probe.index) > gate_index
                and probe.kind is not EtaKind.SCHEDULED
            ):
                del gate_assignment[probe_input_index]
                passed_probe_rows.add(probe_input_index)

    # A live downstream row with a positive implied position and an ETA smaller
    # than its assigned gate ETA proves that the gate vehicle has already
    # passed HKUST. Keep this explicit so the estimator can prefer its
    # freshest downstream rung over the stale direct gate position.
    for probe_input_index, gate_input_index in gate_assignment.items():
        probe = probe_inputs[probe_input_index]
        gate = authoritative_inputs[gate_input_index]
        key = (str(probe.operator), str(probe.route), str(probe.bound))
        gate_index = verified_gate_index.get(key)
        if (
            gate_index is not None
            and int(probe.index) > gate_index
            and probe.kind is not EtaKind.SCHEDULED
            and float(probe.minutes) < float(gate.minutes)
            and int(probe.index) - float(probe.minutes) / MINUTES_PER_STOP
            > gate_index
        ):
            passed_gate_inputs.add(gate_input_index)

    # Rows whose coarse implied position is already beyond HKUST are explicit
    # passed-vehicle evidence even when no gate match was possible.
    for input_index, eta in enumerate(probe_inputs):
        key = (str(eta.operator), str(eta.route), str(eta.bound))
        gate_index = verified_gate_index.get(key)
        if (
            gate_index is not None
            and eta.minutes is not None
            and input_index not in gate_assignment
        ):
            raw_position = int(eta.index) - float(eta.minutes) / MINUTES_PER_STOP
            if raw_position > gate_index:
                passed_probe_rows.add(input_index)

    departed_gate_inputs: set[int] = set()
    undeparted_probe_inputs: set[int] = set()
    for key, gate_rows in gate_rows_by_direction.items():
        gate_index = verified_gate_index.get(key)
        if gate_index is None:
            continue
        for gate_input, gate_row in gate_rows:
            assigned_probe_inputs = [
                probe_input
                for probe_input, assigned_gate in gate_assignment.items()
                if assigned_gate == gate_input
            ]
            future_at_origin = any(
                int(probe_inputs[probe_input].index) == 0
                and int(probe_inputs[probe_input].index)
                - float(probe_inputs[probe_input].minutes) / MINUTES_PER_STOP
                < 0
                and float(probe_inputs[probe_input].minutes)
                > TERMINUS_DEPARTURE_GRACE_MINUTES
                for probe_input in assigned_probe_inputs
            )
            has_departed_probe = any(
                int(probe_inputs[probe_input].index)
                - float(probe_inputs[probe_input].minutes) / MINUTES_PER_STOP
                >= 0
                and probe_inputs[probe_input].kind is not EtaKind.SCHEDULED
                for probe_input in assigned_probe_inputs
            )
            gate_position = (
                int(gate_row.index)
                - float(gate_row.minutes) / MINUTES_PER_STOP
            )
            # A matched positive ETA at stop zero is direct evidence that this
            # journey has not left its terminus.  It must outrank the coarse
            # two-minutes-per-stop projection from a downstream gate: that
            # projection can become nonnegative several minutes before the
            # published origin departure and otherwise creates a premature
            # marker at the terminus.
            departed = not future_at_origin and (
                gate_position >= 0
                or (
                    gate_index > 0
                    and gate_row.kind is not EtaKind.SCHEDULED
                    and has_departed_probe
                )
            )
            if departed:
                departed_gate_inputs.add(gate_input)
            else:
                undeparted_probe_inputs.update(assigned_probe_inputs)

    # Corroborate passed rows across checkpoints using the same ordered ETA
    # matcher used for gate arrivals.  A DSU preserves one-to-one identity
    # without relying on the coarse spatial ladder drift.
    parent = {input_index: input_index for input_index in passed_probe_rows}

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for _key, checkpoints in probe_rows_by_occurrence.items():
        passed_by_checkpoint = {
            checkpoint: [
                (input_index, row)
                for input_index, row in rows
                if input_index in passed_probe_rows
            ]
            for checkpoint, rows in checkpoints.items()
        }
        ordered_checkpoints = sorted(
            checkpoint for checkpoint, rows in passed_by_checkpoint.items() if rows
        )
        if not ordered_checkpoints:
            continue
        pivot = max(
            ordered_checkpoints,
            key=lambda checkpoint: (
                len(passed_by_checkpoint[checkpoint]),
                -min(
                    float(getattr(row, "cache_age_seconds", 0) or 0)
                    for _input_index, row in passed_by_checkpoint[checkpoint]
                ),
                checkpoint,
            ),
        )
        # Use a pivot for high-multiplicity cache-boundary frames, where stale
        # intermediate rows can divert identities. Smaller/uncached ladders
        # retain the established adjacent matching semantics.
        has_stale_cache = any(
            float(getattr(row, "cache_age_seconds", 0) or 0) >= 60
            for rows in passed_by_checkpoint.values()
            for _input_index, row in rows
        )
        # A stale intermediate checkpoint can be the most complete view even
        # with only two rows.  Pivoting such frames prevents an old adjacent
        # row from stealing an identity transitively; multiplicity and
        # freshness choose the pivot deterministically above.
        if len(passed_by_checkpoint[pivot]) >= 2 and has_stale_cache:
            pivot_rows = passed_by_checkpoint[pivot]
            for checkpoint in ordered_checkpoints:
                if checkpoint == pivot:
                    continue
                if checkpoint < pivot:
                    pairs = _align_gate_arrivals(
                        passed_by_checkpoint[checkpoint],
                        pivot_rows,
                        gate_index=checkpoint,
                        checkpoint=pivot,
                    )
                    for earlier_input, pivot_input in pairs:
                        union(earlier_input, pivot_input)
                else:
                    pairs = _align_gate_arrivals(
                        pivot_rows,
                        passed_by_checkpoint[checkpoint],
                        gate_index=pivot,
                        checkpoint=checkpoint,
                    )
                    for pivot_input, later_input in pairs:
                        union(pivot_input, later_input)
        else:
            for earlier, later in zip(ordered_checkpoints, ordered_checkpoints[1:], strict=False):
                pairs = _align_gate_arrivals(
                    passed_by_checkpoint[earlier],
                    passed_by_checkpoint[later],
                    gate_index=earlier,
                    checkpoint=later,
                )
                for earlier_input, later_input in pairs:
                    union(earlier_input, later_input)
    passed_track_ids = {input_index: find(input_index) for input_index in parent}

    # A staggered cache can briefly retain one old realtime row after another
    # stop has replaced it with an established journey.  Such a singleton can
    # otherwise become a second marker beside the established track.  Retire
    # it only when the other row is materially newer, independently
    # timing-compatible, and still ahead of the marker; multi-stop,
    # gate-backed, and equally fresh tracks remain untouched.
    identity_by_input: dict[int, tuple[str, int]] = {
        input_index: ("gate", gate_input)
        for input_index, gate_input in gate_assignment.items()
    }
    identity_by_input.update(
        {
            input_index: ("passed", track_id)
            for input_index, track_id in passed_track_ids.items()
            if input_index not in identity_by_input
        }
    )
    members_by_identity: dict[tuple[str, int], list[int]] = {}
    for input_index, identity in identity_by_input.items():
        members_by_identity.setdefault(identity, []).append(input_index)

    def cache_age(input_index: int) -> float | None:
        value = getattr(probe_inputs[input_index], "cache_age_seconds", None)
        try:
            return max(0.0, float(value)) if value is not None else None
        except (TypeError, ValueError):
            return None

    def established(identity: tuple[str, int]) -> bool:
        members = members_by_identity.get(identity, [])
        if identity[0] == "gate":
            return identity[1] in departed_gate_inputs
        return len({int(probe_inputs[index].index) for index in members}) >= 2

    superseded_probe_inputs: set[int] = set()
    for identity, members in members_by_identity.items():
        if identity[0] != "passed" or len(members) != 1:
            continue
        input_index = members[0]
        row = probe_inputs[input_index]
        if row.kind is EtaKind.SCHEDULED:
            continue
        source_age = cache_age(input_index)
        if source_age is None:
            continue
        key = (str(row.operator), str(row.route), str(row.bound))
        gate_index = verified_gate_index.get(key)
        if gate_index is None:
            continue
        raw_position = int(row.index) - float(row.minutes) / MINUTES_PER_STOP
        position = _passed_row_position(
            raw_position,
            gate_index,
            input_index,
            passed_probe_rows,
            passed_probe_positions,
        )
        if position is None:
            continue
        for checkpoint, checkpoint_rows in probe_rows_by_occurrence.get(key, {}).items():
            # Route order is not refresh order.  Either an upstream or a
            # downstream stop can carry the newer snapshot; it can refute this
            # singleton only while the marker is still before that stop.
            if checkpoint == int(row.index) or position > checkpoint:
                continue
            implied_minutes = (checkpoint - position) * MINUTES_PER_STOP
            compatible_newer_row = False
            for checkpoint_input, checkpoint_row in checkpoint_rows:
                checkpoint_identity = identity_by_input.get(checkpoint_input)
                checkpoint_age = cache_age(checkpoint_input)
                if (
                    checkpoint_identity is None
                    or checkpoint_identity == identity
                    or not established(checkpoint_identity)
                    or checkpoint_input in undeparted_probe_inputs
                    or checkpoint_age is None
                    or source_age - checkpoint_age
                    < PROBE_FRESHNESS_MARGIN_SECONDS
                ):
                    continue
                if (
                    abs(float(checkpoint_row.minutes) - implied_minutes)
                    <= STALE_SINGLETON_MATCH_TOLERANCE_MINUTES
                ):
                    compatible_newer_row = True
                    break
            if compatible_newer_row:
                superseded_probe_inputs.add(input_index)
                break

    return _GateAssociationPlan(
        gate_assignment,
        frozenset(passed_probe_rows),
        passed_probe_positions,
        passed_track_ids,
        frozenset(superseded_probe_inputs),
        verified_gate_index,
        gate_rows_by_direction,
        frozenset(departed_gate_inputs),
        frozenset(undeparted_probe_inputs),
        frozenset(passed_gate_inputs),
    )


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
    # Stop offsets and path arclengths are accumulated through separate loops.
    # Their endpoint can differ by a few floating-point ulps; an exact terminus
    # marker must land on the final path point instead of disappearing.
    if len(path) >= 2 and math.isclose(
        target, travelled, rel_tol=1e-12, abs_tol=1e-6
    ):
        a, b = path[-2], path[-1]
        return b[0], b[1], math.atan2(b[0] - a[0], b[1] - a[1])
    return None


def _separate_common_stop_departures(
    records: list[tuple[tuple[str, str, str], float, bool, frozenset[tuple[str, int]]]],
    evidence: dict[tuple[str, int], tuple[int, float]],
    max_positions: dict[tuple[str, str, str], float],
    min_positions: dict[tuple[str, str, str], float] | None = None,
) -> dict[frozenset[tuple[str, int]], float]:
    """Repair collapsed anchors using large, same-stop ETA headways.

    The ladder model is deliberately coarse.  When two vehicles are both
    reported at a stop, however, their ETA difference is direct evidence of
    their ordering and separation.  Signed pair constraints are projected as
    one acyclic component, so missing rows cannot make three or more vehicles
    invert one another.  Optional route minima keep non-authoritative passed
    components strictly beyond a verified gate; authoritative gate markers
    retain their own incoming positions.  Conflicting cyclic evidence is left
    unchanged.
    """
    baseline = {sources: position for _key, position, _auth, sources in records}
    min_positions = min_positions or {}
    record_by_sources = {sources: record for record in records for sources in [record[3]]}
    edges: dict[frozenset[tuple[str, int]], set[frozenset[tuple[str, int]]]] = {
        sources: set() for _key, _position, _auth, sources in records
    }
    directed: dict[frozenset[tuple[str, int]], dict[frozenset[tuple[str, int]], float]] = {
        sources: {} for _key, _position, _auth, sources in records
    }
    for left_index, left in enumerate(records):
        left_key, _left_position, left_authoritative, left_sources = left
        if left_authoritative:
            continue
        for right in records[left_index + 1 :]:
            right_key, _right_position, right_authoritative, right_sources = right
            if right_authoritative or right_key != left_key:
                continue
            common: list[float] = []
            signed_deltas: list[float] = []
            for left_observation in sorted(left_sources):
                left_evidence = evidence.get(left_observation)
                if left_evidence is None:
                    continue
                left_stop, left_raw_position = left_evidence
                for right_observation in sorted(right_sources):
                    right_evidence = evidence.get(right_observation)
                    if right_evidence is None or right_evidence[0] != left_stop:
                        continue
                    gap = abs(left_raw_position - right_evidence[1])
                    if gap >= 5.0:  # ten minutes at the estimator's 2 min/stop
                        common.append(gap)
                        signed_deltas.append(left_raw_position - right_evidence[1])
            if not common:
                continue
            separation = sorted(common)[len(common) // 2]
            # A larger raw position means an earlier ETA at the same stop and
            # therefore a vehicle farther along the route.  Do not infer this
            # ordering from the already-collapsed anchors.
            left_ahead = sorted(signed_deltas)[len(signed_deltas) // 2] > 0
            edges[left_sources].add(right_sources)
            edges[right_sources].add(left_sources)
            if left_ahead:
                directed[left_sources][right_sources] = separation
            else:
                directed[right_sources][left_sources] = separation
    adjusted: dict[frozenset[tuple[str, int]], float] = {}
    visited: set[frozenset[tuple[str, int]]] = set()
    for root in sorted(edges, key=lambda sources: tuple(sorted(sources))):
        if root in visited:
            continue
        component: list[frozenset[tuple[str, int]]] = []
        stack = [root]
        while stack:
            sources = stack.pop()
            if sources in visited:
                continue
            visited.add(sources)
            component.append(sources)
            stack.extend(edges[sources] - visited)
        if len(component) == 1:
            route_key = record_by_sources[root][0]
            minimum = (
                min_positions.get(route_key, 0.0)
                if not record_by_sources[root][2]
                else 0.0
            )
            adjusted[root] = min(
                max(minimum, baseline[root]),
                max_positions.get(route_key, float("inf")),
            )
            continue
        # Kahn topological sort gives an order satisfying every signed ETA
        # constraint.  Conflicting evidence is cyclic; leave that component
        # at its baseline rather than inventing an ordering.
        indegree = {sources: 0 for sources in component}
        for ahead in component:
            for behind in directed[ahead]:
                indegree[behind] += 1
        ready = sorted(
            (sources for sources in component if indegree[sources] == 0),
            key=lambda sources: tuple(sorted(sources)),
        )
        ordered: list[frozenset[tuple[str, int]]] = []
        while ready:
            sources = ready.pop(0)
            ordered.append(sources)
            for behind in sorted(directed[sources], key=lambda item: tuple(sorted(item))):
                indegree[behind] -= 1
                if indegree[behind] == 0:
                    ready.append(behind)
                    ready.sort(key=lambda item: tuple(sorted(item)))
        if len(ordered) != len(component):
            for sources in component:
                adjusted[sources] = baseline[sources]
            continue
        distance = {sources: 0.0 for sources in component}
        # Directed edges point ahead -> behind.  Walk them backwards so the
        # longest-path coordinate increases toward the route's leading bus.
        for ahead in reversed(ordered):
            for behind, gap in directed[ahead].items():
                distance[ahead] = max(distance[ahead], distance[behind] + gap)
        maximum_distance = max(distance.values())
        component = sorted(
            component,
            key=lambda sources: (-distance[sources], tuple(sorted(sources))),
        )
        route_key = record_by_sources[component[0]][0]
        minimum = min_positions.get(route_key, 0.0)
        maximum = max_positions.get(route_key, float("inf"))
        span = min(maximum_distance, max(0.0, maximum - minimum))
        scale = span / maximum_distance if maximum_distance else 0.0
        scaled_distance = {
            sources: distance[sources] * scale for sources in component
        }
        residuals = sorted(
            baseline[sources] - scaled_distance[sources]
            for sources in component
        )
        # A singleton rung at the leading edge is an independent current
        # position, not merely a collapsed ladder anchor.  Prefer its
        # baseline when available so the projection does not move that
        # source-backed ETA across the audit timing tolerance merely to
        # balance a headway against a multi-rung ladder.  Components with no
        # singleton retain the symmetric residual median (important for the
        # two-track and uneven-three-track headway repairs).
        singleton_anchors = [
            sources for sources in component if len(sources) == 1
        ]
        if singleton_anchors and max(
            baseline[sources] for sources in singleton_anchors
        ) < maximum - 1.0:
            anchor = max(singleton_anchors, key=lambda sources: baseline[sources])
            origin = baseline[anchor] - scaled_distance[anchor]
        else:
            origin = median(residuals)
        origin = min(max(minimum, origin), maximum - span)
        for sources in component:
            position = origin + scaled_distance[sources]
            # The algebra is bounded to [0, maximum], but the leading vehicle
            # can exceed the exact terminus by a few ulps after projection.
            position = max(position, min_positions.get(route_key, 0.0))
            adjusted[sources] = min(max(0.0, position), maximum)
    return adjusted


def estimate_bus_positions(
    probe_etas,
    route_lines,
    destinations: dict[tuple[str, str, str], str] | None = None,
    authoritative_etas=None,
    *,
    observed_checkpoint_indices=None,
) -> list[BusEstimate]:
    """Associate ETA rows into vehicles and interpolate them on route paths.

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

    probe_inputs = list(probe_etas)

    def identity_position(row) -> float:
        """Compare staggered cache rows on one common observation clock.

        Source minutes remain immutable.  Cache age is used only to associate
        observations that can describe the same vehicle; it never becomes a
        displayed countdown or a marker-motion input.
        """
        raw_position, _stop, _scheduled, _authoritative, observation = row
        kind, input_index = observation
        if str(kind).lower() != "probe" or not 0 <= input_index < len(probe_inputs):
            return raw_position
        try:
            age = float(
                getattr(probe_inputs[input_index], "cache_age_seconds", 0.0) or 0.0
            )
        except (TypeError, ValueError):
            age = 0.0
        if not math.isfinite(age) or age < 0.0:
            age = 0.0
        return raw_position + age / (MINUTES_PER_STOP * 60.0)
    observed_by_route = {}
    if observed_checkpoint_indices is not None:
        if isinstance(observed_checkpoint_indices, dict):
            observed_by_route = {tuple(k): frozenset(v)
                                 for k, v in observed_checkpoint_indices.items()}
        else:
            observed_by_route = {key: frozenset(observed_checkpoint_indices)
                                 for key in lines_by_key}
    authoritative_inputs = list(authoritative_etas or [])
    gate_plan = _plan_gate_associations(
        probe_inputs,
        authoritative_inputs,
        set(lines_by_key),
    )
    gate_assignment = gate_plan.gate_assignment
    passed_probe_rows = gate_plan.passed_probe_rows
    passed_probe_positions = gate_plan.passed_probe_positions
    passed_track_ids = gate_plan.passed_track_ids
    verified_gate_index = gate_plan.verified_gate_index
    gate_rows_by_direction = gate_plan.gate_rows_by_direction

    authoritative_keys = {
        (str(eta.operator), str(eta.route), str(eta.bound), int(eta.index))
        for eta in authoritative_inputs
        if eta.minutes is not None
    }
    observation_evidence: dict[tuple[str, int], tuple[int, float]] = {}
    by_direction: dict[
        tuple[str, str, str],
        list[tuple[float, int, bool, bool, tuple[str, int]]],
    ] = {}
    anchored_ladders: dict[
        tuple[str, str, str],
        dict[int, list[tuple[float, int, bool, bool, tuple[str, int]]]],
    ] = {}
    observations = [
        *((eta, False, ("probe", index)) for index, eta in enumerate(probe_inputs)),
        *((eta, True, ("gate", index))
          for index, eta in enumerate(authoritative_inputs)),
    ]
    for eta, is_authoritative, observation in observations:
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
        original_raw_position = raw_position
        if not is_authoritative:
            # Retain the source-implied position even when this probe is
            # later attached to a gate-backed ladder; headway separation must
            # compare raw ETA evidence, not the corrected render position.
            observation_evidence[observation] = (idx, original_raw_position)
        row = (
            raw_position,
            idx,
            eta.kind is EtaKind.SCHEDULED,
            is_authoritative,
            observation,
        )
        if is_authoritative:
            anchored_ladders.setdefault(key, {}).setdefault(
                observation[1], []
            ).append(row)
            continue
        probe_input_index = observation[1]
        if probe_input_index in gate_plan.superseded_probe_inputs:
            continue
        if probe_input_index in gate_assignment:
            anchored_ladders.setdefault(key, {}).setdefault(
                gate_assignment[probe_input_index], []
            ).append(row)
            continue
        if key in gate_rows_by_direction:
            # Keep the exact passed-row rule shared with the frame auditor:
            # coarse positions already beyond the gate win; corrected order
            # positions are accepted only when they remain beyond the gate.
            gate_index = verified_gate_index.get(key)
            if gate_index is None:
                continue
            passed_position = _passed_row_position(
                raw_position,
                gate_index,
                probe_input_index,
                passed_probe_rows,
                passed_probe_positions,
            )
            if passed_position is None:
                continue
            raw_position = passed_position
        # Gate-order proof may have replaced the coarse stop/ETA position;
        # identity normalization must start from that corrected value.
        row = (
            raw_position,
            idx,
            eta.kind is EtaKind.SCHEDULED,
            False,
            observation,
        )
        # Remaining probe-only evidence must imply a position on the route.
        if not is_authoritative and minutes > 0 and identity_position(row) < 0:
            continue
        if raw_position > stops_count - 1:
            continue
        # Keep the source ETA-implied position for same-stop headway
        # constraints; rendering may use a corrected passed-track position.
        by_direction.setdefault(key, []).append(row)

    candidates: dict[
        tuple[str, str, str, int],
        list[
            tuple[
                float,
                int,
                bool,
                frozenset[int],
                bool,
                frozenset[tuple[str, int]],
            ]
        ],
    ] = {}
    for key, rows in sorted(by_direction.items()):
        operator_name, route, bound = key
        stops_count = len(list(lines_by_key[key].stops))
        # Assign each rung deterministically to the nearest compatible ladder.
        # A ladder can contain at most one ETA for a given stop: otherwise
        # same-stop ETAs can chain transitively through neighbouring rows and
        # collapse several actual departures into one vehicle.
        rows.sort(key=lambda row: (identity_position(row), row[1], row[2]))
        ladders: list[
            list[tuple[float, int, bool, bool, tuple[str, int]]]
        ] = []
        for row in rows:
            position, stop_index, _scheduled, _authoritative, _observation = row
            comparable_position = identity_position(row)
            passed_track_id = passed_track_ids.get(_observation[1])
            compatible = []
            for ladder_index, ladder in enumerate(ladders):
                if stop_index in {rung[1] for rung in ladder}:
                    continue
                ladder_track_ids = {
                    passed_track_ids.get(rung[4][1])
                    for rung in ladder
                    if passed_track_ids.get(rung[4][1]) is not None
                }
                if (
                    passed_track_id is not None
                    and ladder_track_ids == {passed_track_id}
                ) or (
                    passed_track_id is None
                    and not ladder_track_ids
                    and abs(comparable_position - identity_position(ladder[-1]))
                    <= LADDER_GAP_STOPS
                ):
                    compatible.append((
                        abs(comparable_position - identity_position(ladder[-1])),
                        ladder_index,
                        ladder,
                    ))
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
                (
                    position,
                    _quantize_position(position, verified_gate_index.get(key)),
                    unreliable,
                    frozenset(rung[1] for rung in ladder),
                    bool(direct),
                    frozenset(rung[4] for rung in ladder),
                )
            )

    # Build one track around every authoritative HKUST arrival. Probe rows
    # were associated by ordered ETA above, so actual travel-time variation
    # cannot split one journey merely because it violates the coarse
    # two-minutes-per-stop position model.
    for key, gate_ladders in sorted(anchored_ladders.items()):
        operator_name, route, bound = key
        stops_count = len(list(lines_by_key[key].stops))
        for gate_input, ladder in gate_ladders.items():
            direct = [rung for rung in ladder if rung[3]]
            if (
                len(direct) != 1
                or gate_input not in gate_plan.departed_gate_inputs
            ):
                continue
            direct_position = direct[0][0]
            unreliable = all(rung[2] for rung in ladder)
            if (
                direct_position >= 0
                and gate_input not in gate_plan.passed_gate_inputs
            ):
                position = direct_position
            else:
                departed_positions = [
                    rung[0]
                    for rung in ladder
                    if not rung[3] and not rung[2] and rung[0] >= 0
                ]
                if not departed_positions:
                    continue
                position = max(departed_positions)
            if position > stops_count - 1:
                continue
            section = min(math.floor(position), stops_count - 2)
            candidates.setdefault(
                (operator_name, route, bound, section), []
            ).append(
                (
                    position,
                    _quantize_position(position, verified_gate_index.get(key)),
                    unreliable,
                    frozenset(rung[1] for rung in ladder),
                    True,
                    frozenset(rung[4] for rung in ladder),
                )
            )

    # A missing middle rung (probe fetch failure) can split one bus's ladder
    # in two, yielding two markers a section apart that alternate between
    # frames. Collapse vehicle anchors that sit within one stop of each other
    # in the same direction, preferring the reliable (realtime-evidenced)
    # anchor and then the earlier position.
    vehicles: dict[
        tuple[str, str, str],
        list[
            tuple[
                float,
                bool,
                frozenset[int],
                bool,
                frozenset[tuple[str, int]],
            ]
        ],
    ] = {}
    for (operator_name, route, bound, _section), entries in sorted(candidates.items()):
        for (
            position,
            _scaled,
            unreliable,
            stop_indices,
            authoritative,
            source_observations,
        ) in entries:
            vehicles.setdefault((operator_name, route, bound), []).append(
                (
                    position,
                    unreliable,
                    stop_indices,
                    authoritative,
                    source_observations,
                )
            )

    candidates2: dict[
        tuple[str, str, str, int],
        list[
            tuple[
                float,
                int,
                bool,
                frozenset[int],
                frozenset[tuple[str, int]],
            ]
        ],
    ] = {}
    for key, anchors in sorted(vehicles.items()):
        # Keep spatial order while forming clusters.  Sorting by reliability
        # first can make a distant reliable anchor absorb an upstream
        # scheduled anchor, and makes the result depend on feed ordering.
        anchors.sort(key=lambda item: item[0])
        clusters: list[
            list[
                tuple[
                    float,
                    bool,
                    frozenset[int],
                    bool,
                    frozenset[tuple[str, int]],
                ]
            ]
        ] = [[anchors[0]]]
        for anchor in anchors[1:]:
            cluster_stops = set().union(*(item[2] for item in clusters[-1]))
            # Shared stop provenance means that one source snapshot exposed
            # both departures simultaneously.  Keep them distinct even when
            # their inferred positions are close: upstream stops stop listing
            # a bus after it passes, so far-away upstream ETAs do not refute a
            # close pair confirmed at downstream stops.
            cluster_is_authoritative = any(item[3] for item in clusters[-1])
            if (
                abs(anchor[0] - clusters[-1][-1][0]) <= 1.0
                and cluster_stops.isdisjoint(anchor[2])
                and not anchor[3]
                and not cluster_is_authoritative
            ):
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
            provenance = frozenset().union(*(anchor[2] for anchor in cluster))
            source_observations = frozenset().union(
                *(anchor[4] for anchor in cluster)
            )
            bucket.append(
                (
                    0.0,
                    _quantize_position(position, verified_gate_index.get(key)),
                    unreliable,
                    provenance,
                    source_observations,
                )
            )

    estimates: list[BusEstimate] = []
    records = [
        (
            (operator_name, route, bound),
            scaled_position / 1000,
            any(kind == "gate" for kind, _index in source_observations),
            source_observations,
        )
        for (operator_name, route, bound, _section), entries in candidates2.items()
        for _best_distance, scaled_position, _unreliable, _provenance, source_observations in entries
    ]
    records.sort(key=lambda item: (item[0], item[1], tuple(sorted(item[3]))))
    max_positions = {
        key: float(len(list(line.stops)) - 1) for key, line in lines_by_key.items()
    }
    min_positions = {
        key: float(gate_index) + 0.001
        for key, gate_index in verified_gate_index.items()
    }
    adjusted_positions = _separate_common_stop_departures(
        records, observation_evidence, max_positions, min_positions
    )
    for (operator_name, route, bound, _section), entries in sorted(candidates2.items()):
        line = lines_by_key[(operator_name, route, bound)]
        stops = list(line.stops)
        stops_count = len(stops)
        path = list(line.path)
        offsets = list(line.stop_offsets)
        for (
            _best_distance,
            scaled_position,
            unreliable,
            provenance,
            source_observations,
        ) in entries:
            position = adjusted_positions.get(source_observations, scaled_position / 1000)
            bracket = None
            eta_minutes = None
            eta_arrival_at = None
            boundary_age_seconds = None
            priority_indices = frozenset()
            observed = observed_by_route.get((operator_name, route, bound))
            if observed is not None and provenance:
                first_present = min(provenance)
                if first_present == 0:
                    bracket = (0.0, 0.0)
                else:
                    absent = [index for index in observed
                              if index < first_present and index not in provenance]
                    if absent:
                        bracket = (float(max(absent)), float(first_present))
                source_rows = [probe_inputs[index] for kind, index in source_observations
                               if str(kind).lower() == "probe"
                               and 0 <= index < len(probe_inputs)]
                source_rows = [row for row in source_rows if getattr(row, "minutes", None) is not None]
                zero_indices = sorted({
                    int(row.index)
                    for row in source_rows
                    if float(row.minutes) <= 0
                })
                positive_indices = sorted({
                    int(row.index)
                    for row in source_rows
                    if float(row.minutes) > 0
                })
                refresh_frontier = set(zero_indices)
                if zero_indices:
                    next_positive = next(
                        (
                            index
                            for index in positive_indices
                            if index > zero_indices[-1]
                        ),
                        None,
                    )
                    if next_positive is not None:
                        refresh_frontier.add(next_positive)
                else:
                    # With no zero plateau yet, keep the first two positive
                    # rungs fresh so the next hand-off already has an upper
                    # observation when the first rung becomes due or vanishes.
                    refresh_frontier.update(positive_indices[:2])
                priority_indices = frozenset(refresh_frontier)
                present_rows = [row for row in source_rows
                                if (str(getattr(row, "operator", "")),
                                    str(getattr(row, "route", "")),
                                    str(getattr(row, "bound", "")))
                                == (operator_name, route, bound)
                                and int(getattr(row, "index", -1)) == first_present]
                if bracket is not None and present_rows:
                    selected_eta = min(
                        present_rows,
                        key=lambda row: (
                            float(getattr(row, "cache_age_seconds", 0.0) or 0.0),
                            float(row.minutes),
                        ),
                    )
                    eta_minutes = float(selected_eta.minutes)
                    eta_arrival_at = getattr(selected_eta, "arrival_at", None)
                    absent_index = int(bracket[0]) if bracket else first_present
                    present_age = float(
                        getattr(selected_eta, "cache_age_seconds", 0.0) or 0.0
                    )
                    absent_ages = [
                        float(getattr(row, "cache_age_seconds", 0.0) or 0.0)
                        for row in probe_inputs
                        if (str(getattr(row, "operator", "")),
                            str(getattr(row, "route", "")),
                            str(getattr(row, "bound", "")))
                        == (operator_name, route, bound)
                        if int(getattr(row, "index", -1)) == absent_index
                    ]
                    if absent_index == first_present:
                        absent_ages.append(present_age)
                    if absent_ages:
                        boundary_age_seconds = max(present_age, min(absent_ages))
                    # The first stop which sees this ETA instance is the
                    # physical upper boundary.  Its source ETA supplies the
                    # requested proportion within the preceding stop span.
                    position = first_present - min(
                        1.0, max(0.0, eta_minutes / MINUTES_PER_STOP)
                    )
                if bracket is not None:
                    position = min(max(position, bracket[0]), bracket[1])
            render_section = min(math.floor(position), stops_count - 2)
            fraction = position - render_section
            target_offset = offsets[render_section] + (offsets[render_section + 1] - offsets[render_section]) * fraction
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
                    operator_code=operator_name,
                    source_indices=provenance,
                    source_observations=source_observations,
                    bracket=bracket,
                    eta_minutes=eta_minutes,
                    eta_arrival_at=eta_arrival_at,
                    bracket_initial_eta=eta_minutes,
                    boundary_age_seconds=boundary_age_seconds,
                    priority_indices=priority_indices,
                )
            )
    return estimates


def reproject_estimate(estimate: BusEstimate, position: float, route_lines) -> BusEstimate:
    """Return ``estimate`` projected at a new scalar position on its official line."""
    key = (estimate.operator_code or str(estimate.operator), estimate.route, estimate.bound)
    line = next((item for item in route_lines if (str(getattr(item, "operator", "")),
                 str(getattr(item, "route", "")), str(getattr(item, "bound", ""))) == key), None)
    if line is None:
        return replace(estimate, position=position)
    stops = list(getattr(line, "stops", ()))
    offsets = list(getattr(line, "stop_offsets", ()))
    path = list(getattr(line, "path", ()))
    if len(offsets) < 2 or not path:
        return replace(estimate, position=position)
    position = max(0.0, min(float(position), max(0.0, len(stops) - 1)))
    section = min(int(math.floor(position)), len(offsets) - 2)
    target = offsets[section] + (offsets[section + 1] - offsets[section]) * (position - section)
    located = _point_at_path_offset(path, target)
    if located is None:
        return replace(estimate, position=position)
    lat, lon, heading = located
    return replace(estimate, lat=lat, lon=lon, heading=heading, position=position,
                   operator_code=estimate.operator_code or str(getattr(line, "operator", "")))


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
