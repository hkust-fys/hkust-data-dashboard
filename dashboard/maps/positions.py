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
from itertools import combinations
from statistics import median

from dashboard.models import EtaKind, Operator

LIVE_PROBE_ETA_KINDS = frozenset({EtaKind.REALTIME, EtaKind.MOVING_SLOWLY, EtaKind.DELAYED})


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
    # Refresh revisions of the two physical boundary observations.  Unlike
    # cache age this is stable across delayed rendering and prevents replaying
    # the same evidence with a different derived position.
    boundary_revision: tuple[int, int] | None = None
    # Signed ETA offsets at the lower/upper bracket observations when the
    # physical boundary is a due-to-future timestamp crossing.
    bracket_eta_offsets: tuple[float, float] | None = None
    # Owned stop occurrences forming the useful refresh frontier for this exact
    # ETA instance: every due/zero rung plus the first future rung.
    priority_indices: frozenset[int] = frozenset()
    # Bounded search hints for a fresh live singleton. MarkerTracker admits
    # these only after recovery, owned frontiers, physical boundaries and its
    # midpoint fallback, so exploration cannot crowd out existing evidence.
    # They are polling metadata, never source ownership or positioning proof.
    exploratory_indices: frozenset[int] = frozenset()
    # Immutable, exact probe checkpoints owned by this ETA ladder.  Entries
    # are (stop index, absolute arrival timestamp, refresh revision); unlike
    # source row offsets they survive staggered probe input rebuilding.
    checkpoint_evidence: tuple[tuple[int, float, int], ...] = ()

    @property
    def bracket_lower(self):
        return self.bracket[0] if self.bracket else None

    @property
    def bracket_upper(self):
        return self.bracket[1] if self.bracket else None


def _checkpoint_evidence(rows):
    """Keep bounded, owned source timestamps; never invent an arrival time."""
    evidence = set()
    for row in rows:
        try:
            index = int(row.index)
            revision = int(row.refresh_generation)
            arrival = float(row.arrival_at.timestamp())
        except (AttributeError, TypeError, ValueError, OverflowError, OSError):
            continue
        if index >= 0 and revision > 0 and math.isfinite(arrival):
            evidence.add((index, arrival, revision))
    return tuple(sorted(evidence)[:64])


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
PROVISIONAL_GATE_PROBE_FRESHNESS_SECONDS = 60.0
ETA_GUIDED_PRIORITY_FRESHNESS_SECONDS = 60.0
ATOMIC_LADDER_FRESHNESS_SECONDS = 60.0
ATOMIC_LADDER_MAX_SECONDS_PER_STOP = 180.0
# Only directions whose rendered geometry is verified to end at the official
# service terminus may use ``len(stops) - 1`` as a route-census sentinel.
# Some KMB routes temporarily expose a validated geometry prefix while their
# operator stop sequence has already gained a trailing extension.
ATOMIC_LADDER_FULL_KMB_ROUTES = frozenset({"91", "91M"})

# Route-geometry operator codes -> dashboard Operator enum values.
_OPERATOR_BY_CODE = {
    "KMB": Operator.KMB,
    "CTB": Operator.CITYBUS,
    "GMB": Operator.GMB,
}


def _eta_guided_priority_indices(row, bracket, stops_count: int) -> frozenset[int]:
    """Return interior probes which can refine one fresh live ETA next cycle.

    The two-minutes-per-stop projection is only a search hint.  The marker keeps
    its conservative observed boundary until these stops return real evidence.
    """
    raw_index = getattr(row, "index", None)
    raw_revision = getattr(row, "refresh_generation", None)
    raw_minutes = getattr(row, "minutes", None)
    raw_age = getattr(row, "cache_age_seconds", None)
    if (
        getattr(row, "kind", None) is not EtaKind.REALTIME
        or isinstance(raw_index, bool)
        or not isinstance(raw_index, int)
        or isinstance(raw_revision, bool)
        or not isinstance(raw_revision, int)
        or isinstance(raw_minutes, bool)
        or isinstance(raw_age, bool)
    ):
        return frozenset()
    try:
        minutes = float(raw_minutes)
        age = float(raw_age)
        lower, upper = map(float, bracket)
    except (TypeError, ValueError, OverflowError):
        return frozenset()
    if (
        not all(math.isfinite(value) for value in (minutes, age, lower, upper))
        or minutes <= 0.0
        or age < 0.0
        or age >= ETA_GUIDED_PRIORITY_FRESHNESS_SECONDS
        or raw_revision <= 0
        or stops_count <= 0
        or not 0 <= raw_index < stops_count
        or not upper.is_integer()
        or raw_index != int(upper)
        or upper - lower <= 1.0
    ):
        return frozenset()
    projected = raw_index - minutes / MINUTES_PER_STOP
    if not math.isfinite(projected) or not lower < projected < upper:
        return frozenset()
    return frozenset(
        index
        for index in {math.floor(projected), math.ceil(projected)}
        if 0 <= index < stops_count and lower < index < upper
    )


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


def _heal_atomic_kmb_ladder_fragments(
    ladders, key, probe_inputs, *, terminal_index: int
):
    """Join one fresh KMB route-response chain split by the coarse ETA model.

    KMB serves every requested stop for one route in a single HTTP response,
    so one positive refresh revision is an atomic observation clock.  When
    that response contains exactly one ETA occurrence per stop and exactly one
    terminal occurrence, a fresh due-to-future monotone chain describes one
    vehicle even if its real travel speed leaves a gap larger than the coarse
    two-minutes-per-stop ladder threshold.

    This intentionally fails closed for staggered rows, competing/equal stop
    occurrences, gate-owned rows, interleaved fragments, stale snapshots, or
    malformed chronology.  Exact source rows are only unioned; none are
    deduplicated, and the normal due/future boundary code still places the
    resulting marker.
    """
    if (
        key[0] != "KMB"
        or key[1] not in ATOMIC_LADDER_FULL_KMB_ROUTES
        or len(ladders) < 2
    ):
        return ladders
    flattened = [rung for ladder in ladders for rung in ladder]
    if len(flattened) < 3 or any(rung[3] for rung in flattened):
        return ladders
    observations = {rung[4] for rung in flattened}
    if len(observations) != len(flattened) or any(
        kind != "probe" for kind, _input_index in observations
    ):
        return ladders

    route_rows = []
    for input_index, row in enumerate(probe_inputs):
        if (
            str(getattr(row, "operator", "")),
            str(getattr(row, "route", "")),
            str(getattr(row, "bound", "")),
        ) != key:
            continue
        if getattr(row, "minutes", None) is None:
            continue
        if getattr(row, "kind", None) is EtaKind.UNAVAILABLE:
            continue
        route_rows.append((input_index, row))
    # If another active row was filtered or assigned elsewhere, this is not a
    # complete one-vehicle census and temporal healing would be ambiguous.
    if observations != {("probe", input_index) for input_index, _row in route_rows}:
        return ladders
    stop_indices = [int(row.index) for _input_index, row in route_rows]
    if len(stop_indices) != len(set(stop_indices)):
        return ladders
    if sum(index == terminal_index for index in stop_indices) != 1:
        return ladders

    revisions = set()
    signed_minutes = []
    timed_rows = []
    for input_index, row in route_rows:
        try:
            revision = int(getattr(row, "refresh_generation", 0) or 0)
            age = float(getattr(row, "cache_age_seconds", 0.0) or 0.0)
            arrival = float(row.arrival_at.timestamp())
            signed = float(
                row.signed_minutes
                if getattr(row, "signed_minutes", None) is not None
                else row.minutes
            )
        except (AttributeError, TypeError, ValueError, OverflowError, OSError):
            return ladders
        if (
            revision <= 0
            or not math.isfinite(age)
            or not 0.0 <= age < ATOMIC_LADDER_FRESHNESS_SECONDS
            or not math.isfinite(arrival)
            or not math.isfinite(signed)
            or row.kind is EtaKind.SCHEDULED
        ):
            return ladders
        revisions.add(revision)
        signed_minutes.append(signed)
        timed_rows.append((int(row.index), arrival, input_index))
    if len(revisions) != 1 or not (
        min(signed_minutes, default=1.0) <= 0.0
        and max(signed_minutes, default=0.0) > 0.0
    ):
        return ladders

    ordered_ladders = sorted(ladders, key=lambda ladder: min(rung[1] for rung in ladder))
    if any(
        max(left_rung[1] for left_rung in left)
        >= min(right_rung[1] for right_rung in right)
        for left, right in zip(ordered_ladders, ordered_ladders[1:], strict=False)
    ):
        return ladders
    timed_rows.sort()
    for (left_stop, left_arrival, _left_input), (
        right_stop,
        right_arrival,
        _right_input,
    ) in zip(timed_rows, timed_rows[1:], strict=False):
        stop_gap = right_stop - left_stop
        travel_seconds = right_arrival - left_arrival
        if (
            stop_gap <= 0
            or travel_seconds <= 0.0
            or travel_seconds
            > ATOMIC_LADDER_MAX_SECONDS_PER_STOP * stop_gap
        ):
            return ladders
    return [[rung for ladder in ordered_ladders for rung in ladder]]


def _align_gate_arrivals(
    gate_rows: list[tuple[int, object]],
    probe_rows: list[tuple[int, object]],
    *,
    gate_index: int,
    checkpoint: int,
    rank_first: bool = False,
    allow_downstream_clock_skew: bool = False,
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
    # HKUST; that is an already-passed vehicle, not clock skew.  Only direct
    # revalidation of an already-carried gate identity may additionally allow
    # the coarse integral gate minute's countdown skew.
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
                        expected_delta - GATE_DOWNSTREAM_DRIFT_MINUTES
                        - (clock_skew_minutes if allow_downstream_clock_skew else 0),
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
    # When the separate gate feed is empty, a fresh probe of the exact mapped
    # gate occurrence may still prove ordered ETA identity.  These IDs never
    # grant authoritative positioning; the ordinary physical boundary remains
    # responsible for marker placement.
    provisional_gate_tracks: dict[int, int]
    provisional_undeparted_inputs: frozenset[int]


def _atomic_kmb_frontier_certificate(
    key, frontier, current, *, previous_rows, gate_index, previous_checkpoint, checkpoint,
    gate_assignments, gate_rows, passed_rows, pairs, carried=False,
):
    """Certify an entire same-response KMB frontier, never one loose tail.

    Admission permits at most one minute below the ordinary direct gate lower
    bound. A carried certificate belongs to this exact atomic response; every
    later transition must preserve its complete ordered frontier and local
    edges. The provider's fixed next-three window may shed only the certified
    trailing root of a one-rank slide. Gate-feed rows are current roots, not
    probe revisions or synthetic clocks.
    """
    if (key[0] != "KMB" or key[1] not in ATOMIC_LADDER_FULL_KMB_ROUTES
            or not gate_index < previous_checkpoint < checkpoint
            or len(previous_rows) != len(frontier)
            or {(source, id(row)) for source, row in previous_rows}
            != {(source, id(row)) for source, row in frontier}
            or len(current) not in {len(frontier), len(frontier) + 1}
            or len(frontier) < 2
            or len(pairs) not in {len(frontier), len(frontier) - 1}):
        return False

    def kind_class(row):
        kind = getattr(row, "kind", None)
        if isinstance(kind, str) and kind in LIVE_PROBE_ETA_KINDS:
            return "live"
        return "scheduled" if kind == EtaKind.SCHEDULED else None

    def number(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) \
            and math.isfinite(value)

    revisions = set()
    for expected_index, rows in ((previous_checkpoint, frontier), (checkpoint, current)):
        for source, row in rows:
            revision = getattr(row, "refresh_generation", None)
            minutes = getattr(row, "minutes", None)
            signed = getattr(row, "signed_minutes", None)
            age = getattr(row, "cache_age_seconds", None)
            try:
                arrival = row.arrival_at.timestamp()
            except (AttributeError, TypeError, ValueError, OverflowError, OSError):
                return False
            if (not isinstance(source, int) or isinstance(source, bool) or source < 0
                    or not isinstance(revision, int) or isinstance(revision, bool) or revision <= 0
                    or not isinstance(getattr(row, "index", None), int)
                    or isinstance(row.index, bool)
                    or getattr(row, "index", None) != expected_index
                    or (str(row.operator), str(row.route), str(row.bound)) != key
                    or not all(number(value) for value in (minutes, signed, age, arrival))
                    or minutes < 0 or abs(minutes - max(0.0, signed)) > 1e-6
                    or not 0 <= age < ATOMIC_LADDER_FRESHNESS_SECONDS
                    or kind_class(row) is None):
                return False
            revisions.add(revision)
    if len(revisions) != 1:
        return False
    ordered_gates = []
    for source, gate in gate_rows.items():
        gate_minutes = getattr(gate, "minutes", None)
        gate_row_index = getattr(gate, "index", None)
        if (
            not isinstance(source, int)
            or isinstance(source, bool)
            or source < 0
            or not number(gate_minutes)
            or gate_minutes < 0
            or not isinstance(gate_row_index, int)
            or isinstance(gate_row_index, bool)
            or gate_row_index != gate_index
            or kind_class(gate) is None
            or (str(gate.operator), str(gate.route), str(gate.bound)) != key
        ):
            return False
        ordered_gates.append((float(gate_minutes), source, gate))
    ordered_gates.sort(key=lambda item: (item[0], item[1]))
    if any(
        left[0] >= right[0]
        for left, right in zip(ordered_gates, ordered_gates[1:], strict=False)
    ):
        return False
    all_rows = [*frontier, *current]
    if (len({source for source, _row in all_rows}) != len(all_rows)
            or len({id(row) for _source, row in all_rows}) != len(all_rows)):
        return False
    ordered_previous = sorted(frontier, key=lambda item: item[1].minutes)
    ordered_current = sorted(current, key=lambda item: item[1].minutes)
    for rows in (ordered_previous, ordered_current):
        if any(left[1].minutes >= right[1].minutes
               or left[1].arrival_at.timestamp() >= right[1].arrival_at.timestamp()
               for left, right in zip(rows, rows[1:], strict=False)):
            return False
    leading_handoff = len(ordered_current) == len(ordered_previous) + 1
    # KMB exposes only the next three arrivals at each stop.  When a bus which
    # has already passed HKUST first appears at a downstream checkpoint, it can
    # enter at the head while the farthest future gate journey falls off the
    # tail in the same atomic response.  That is a one-rank window slide, not a
    # population increase.
    sliding_handoff = (
        len(ordered_previous) == len(ordered_current) == 3
        and len(pairs) == len(ordered_previous) - 1
    )
    continuing_previous = (
        ordered_previous[:-1] if sliding_handoff else ordered_previous
    )
    continuing_current = (
        ordered_current[1:]
        if leading_handoff or sliding_handoff
        else ordered_current
    )
    if leading_handoff or sliding_handoff:
        _leading_source, leading_row = ordered_current[0]
        if (
            kind_class(leading_row) != "live"
            or leading_row.minutes != 0
            or leading_row.signed_minutes > 0
            or checkpoint - leading_row.minutes / MINUTES_PER_STOP <= gate_index
        ):
            return False
    if set(pairs) != {(new[0], old[0]) for old, new in zip(
        continuing_previous, continuing_current, strict=True,
    )}:
        return False

    dropped_root = None
    if sliding_handoff:
        dropped_source, _dropped_row = ordered_previous[-1]
        dropped_root = gate_assignments.get(dropped_source)
        if dropped_root is None or dropped_source in passed_rows:
            return False

    roots = []
    passed = 0
    first_failure = False
    for previous, present in zip(
        continuing_previous, continuing_current, strict=True,
    ):
        previous_source, previous_row = previous
        current_source, current_row = present
        if (kind_class(previous_row) != kind_class(current_row)
                or current_row.arrival_at.timestamp()
                <= previous_row.arrival_at.timestamp()
                or current_row.signed_minutes <= previous_row.signed_minutes
                or not _align_gate_arrivals(
                    [previous], [present], gate_index=previous_checkpoint, checkpoint=checkpoint,
                )):
            return False
        root = gate_assignments.get(previous_source)
        if root is None:
            # Passed vehicles precede the unpassed gate roots in ETA rank.
            if previous_source not in passed_rows or roots:
                return False
            passed += 1
            continue
        gate = gate_rows.get(root)
        gate_kind = kind_class(gate) if gate is not None else None
        current_kind = kind_class(current_row)
        # The root was established on the prior checkpoint by ordinary gate
        # matching.  Preserve the observed live-gate/scheduled-suffix case,
        # but never let a scheduled gate promote live probe motion evidence.
        if (gate is None or root in roots or previous_source in passed_rows
                or not number(getattr(gate, "minutes", None)) or gate.minutes < 0
                or (str(gate.operator), str(gate.route), str(gate.bound)) != key
                or getattr(gate, "index", None) != gate_index
                or not (
                    gate_kind == current_kind
                    or (gate_kind == "live" and current_kind == "scheduled")
                )
                or current_row.minutes - gate.minutes <= 0):
            return False
        if roots and gate_rows[roots[-1]].minutes >= gate.minutes:
            return False
        roots.append(root)
        if not carried and not _align_gate_arrivals(
            [(root, gate)], [present], gate_index=gate_index, checkpoint=checkpoint,
            allow_downstream_clock_skew=True,
        ):
            expected = (checkpoint - gate_index) * MINUTES_PER_STOP
            lower = max(-GATE_PASSAGE_SKEW_MINUTES,
                        expected - GATE_DOWNSTREAM_DRIFT_MINUTES - 1.25)
            delta = current_row.minutes - gate.minutes
            if not lower - 1.0 <= delta < lower:
                return False
            first_failure = True
    # The established production shape is a passed suffix plus at least two
    # gate roots.  A narrow two-rank handoff is also admissible: both rows are
    # still gate-backed, the leading gate is due (using the same departure
    # grace as ordinary origin handling), and the first downstream checkpoint
    # is the existing one-minute deficit admission.  This must remain exact
    # shape so a missing root, a third all-gate journey, or a one-root/passed
    # pair cannot be promoted into an atomic population.
    two_rank_handoff = False
    if (len(frontier) == len(current) == 2 and passed == 0 and len(roots) == 2
            and len(gate_rows) == 2 and set(roots) == set(gate_rows)):
        leading_gate = gate_rows.get(roots[0])
        if leading_gate is not None:
            try:
                leading_minutes = float(leading_gate.minutes)
            except (AttributeError, TypeError, ValueError, OverflowError):
                leading_minutes = float("nan")
            two_rank_handoff = (
                math.isfinite(leading_minutes)
                and 0.0 <= leading_minutes <= TERMINUS_DEPARTURE_GRACE_MINUTES
                and (carried or first_failure)
            )
    ordered_gate_sources = [source for _minutes, source, _gate in ordered_gates]
    represented_root_prefix = roots == ordered_gate_sources[:len(roots)]
    dropped_trailing_root = (
        not sliding_handoff
        or roots + [dropped_root]
        == ordered_gate_sources[: len(roots) + 1]
    )
    expected = (checkpoint - gate_index) * MINUTES_PER_STOP
    relaxed_lower = max(
        -GATE_PASSAGE_SKEW_MINUTES,
        expected - GATE_DOWNSTREAM_DRIFT_MINUTES - 1.25,
    )
    upper = expected + GATE_DOWNSTREAM_DRIFT_MINUTES
    future_gate_suffix_excluded = all(
        not (
            (
                kind_class(gate) == kind_class(row)
                or (
                    kind_class(gate) == "live"
                    and kind_class(row) == "scheduled"
                )
            )
            and float(row.minutes) - gate_minutes > 0
            and relaxed_lower
            <= float(row.minutes) - gate_minutes
            <= upper
        )
        for gate_minutes, _source, gate in ordered_gates[len(roots):]
        for _row_source, row in ordered_current
    )
    return (
        (
            passed > 0
            and not sliding_handoff
            and (len(roots) >= 2 or (carried and len(roots) >= 1))
            and (carried or first_failure)
            and represented_root_prefix
            and future_gate_suffix_excluded
        )
        or two_rank_handoff
        or (
            leading_handoff
            and carried
            and passed == 0
            and len(roots) >= 2
            and represented_root_prefix
            and future_gate_suffix_excluded
        )
        or (
            sliding_handoff
            and (len(roots) >= 2 or (carried and len(roots) >= 1))
            and dropped_trailing_root
            and represented_root_prefix
            and future_gate_suffix_excluded
        )
    )


def _atomic_kmb_fresh_root_pairs(
    key,
    gate_rows,
    current,
    route_rows,
    *,
    gate_index,
    checkpoint,
    prior_gate_assignments=None,
):
    """Return one exact KMB root alignment across rounded-clock skew.

    This is deliberately narrower than ordinary gate matching.  It repairs a
    sparse full-route response only when the entire rank shift has one unique,
    same-kind explanation: already-passed rows form a leading prefix, current
    gate journeys form the matched suffix, and future gates form a trailing
    suffix.  The special path must consume at least one edge in the existing
    1.25-minute integral-clock grace; otherwise ordinary matching remains the
    authority.
    """
    if (
        key[0] != "KMB"
        or key[1] not in ATOMIC_LADDER_FULL_KMB_ROUTES
        or checkpoint <= gate_index
        or len(gate_rows) < 2
        or len(current) < 2
        or len(gate_rows) > 8
        or len(current) > 8
    ):
        return []

    def kind_class(row):
        kind = getattr(row, "kind", None)
        if isinstance(kind, str) and kind in LIVE_PROBE_ETA_KINDS:
            return "live"
        return (
            "scheduled"
            if isinstance(kind, str) and kind == EtaKind.SCHEDULED
            else None
        )

    def number(value):
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        )

    route_revisions = set()
    route_sources = set()
    route_objects = set()
    route_occurrences = set()
    route_by_source = {}
    for source, row in route_rows:
        revision = getattr(row, "refresh_generation", None)
        index = getattr(row, "index", None)
        minutes = getattr(row, "minutes", None)
        signed = getattr(row, "signed_minutes", None)
        age = getattr(row, "cache_age_seconds", None)
        try:
            arrival = row.arrival_at.timestamp()
        except (AttributeError, TypeError, ValueError, OverflowError, OSError):
            return []
        if (
            not isinstance(source, int)
            or isinstance(source, bool)
            or source < 0
            or source in route_sources
            or id(row) in route_objects
            or not isinstance(index, int)
            or isinstance(index, bool)
            or index < 0
            or not isinstance(revision, int)
            or isinstance(revision, bool)
            or revision <= 0
            or not all(number(value) for value in (minutes, signed, age, arrival))
            or minutes < 0
            or abs(minutes - max(0.0, signed)) > 1e-6
            or not 0 <= age < ATOMIC_LADDER_FRESHNESS_SECONDS
            or kind_class(row) is None
            or (str(row.operator), str(row.route), str(row.bound)) != key
        ):
            return []
        route_sources.add(source)
        route_objects.add(id(row))
        route_occurrences.add((source, id(row)))
        route_by_source[source] = row
        route_revisions.add(revision)
    if len(route_revisions) != 1:
        return []

    gates = []
    gate_sources = set()
    for source, row in gate_rows:
        minutes = getattr(row, "minutes", None)
        index = getattr(row, "index", None)
        if (
            not isinstance(source, int)
            or isinstance(source, bool)
            or source < 0
            or source in gate_sources
            or not number(minutes)
            or minutes < 0
            or kind_class(row) is None
            or not isinstance(index, int)
            or isinstance(index, bool)
            or index != gate_index
            or (str(row.operator), str(row.route), str(row.bound)) != key
        ):
            return []
        gate_sources.add(source)
        gates.append((float(minutes), source, row))

    probes = []
    probe_sources = set()
    probe_objects = set()
    for source, row in current:
        minutes = getattr(row, "minutes", None)
        signed = getattr(row, "signed_minutes", None)
        age = getattr(row, "cache_age_seconds", None)
        revision = getattr(row, "refresh_generation", None)
        index = getattr(row, "index", None)
        try:
            arrival = row.arrival_at.timestamp()
        except (AttributeError, TypeError, ValueError, OverflowError, OSError):
            return []
        if (
            not isinstance(source, int)
            or isinstance(source, bool)
            or source < 0
            or source in probe_sources
            or id(row) in probe_objects
            or (source, id(row)) not in route_occurrences
            or not all(number(value) for value in (minutes, signed, age, arrival))
            or minutes < 0
            or abs(minutes - max(0.0, signed)) > 1e-6
            or not 0 <= age < ATOMIC_LADDER_FRESHNESS_SECONDS
            or revision not in route_revisions
            or kind_class(row) is None
            or not isinstance(index, int)
            or isinstance(index, bool)
            or index != checkpoint
            or (str(row.operator), str(row.route), str(row.bound)) != key
        ):
            return []
        probe_sources.add(source)
        probe_objects.add(id(row))
        probes.append((float(minutes), float(arrival), source, row))

    gates.sort(key=lambda item: (item[0], item[1]))
    probes.sort(key=lambda item: (item[0], item[2]))
    if any(
        left[0] >= right[0]
        for left, right in zip(gates, gates[1:], strict=False)
    ) or any(
        left[0] >= right[0] or left[1] >= right[1]
        for left, right in zip(probes, probes[1:], strict=False)
    ):
        return []

    expected = (checkpoint - gate_index) * MINUTES_PER_STOP
    ordinary_lower = max(
        -GATE_PASSAGE_SKEW_MINUTES,
        expected - GATE_DOWNSTREAM_DRIFT_MINUTES,
    )
    lower = ordinary_lower - 1.25
    upper = expected + GATE_DOWNSTREAM_DRIFT_MINUTES
    edges = set()
    same_class_edges = set()
    grace_edges = set()
    for gate_number, (gate_minutes, _gate_source, gate) in enumerate(gates):
        for probe_number, (probe_minutes, _arrival, _probe_source, probe) in enumerate(
            probes
        ):
            delta = probe_minutes - gate_minutes
            gate_class = kind_class(gate)
            probe_class = kind_class(probe)
            if (
                (
                    gate_class == probe_class
                    or (gate_class == "live" and probe_class == "scheduled")
                )
                and delta > 0
                and lower <= delta <= upper
            ):
                pair = (gate_number, probe_number)
                edges.add(pair)
                if gate_class == probe_class:
                    same_class_edges.add(pair)
                    if delta < ordinary_lower:
                        grace_edges.add(pair)

    # Enumerate only maximum-cardinality ordered edge sets and require the
    # maximum itself to be unique. The provider exposes at most three arrivals
    # per stop; the explicit bound above also keeps malformed fixtures cheap.
    best = set()
    for size in range(min(len(gates), len(probes)), 1, -1):
        best = {
            tuple(zip(gate_numbers, probe_numbers, strict=True))
            for gate_numbers in combinations(range(len(gates)), size)
            for probe_numbers in combinations(range(len(probes)), size)
            if all(
                pair in edges
                for pair in zip(gate_numbers, probe_numbers, strict=True)
            )
        }
        if best:
            break
    if len(best) != 1:
        return []
    pairs = next(iter(best))
    gate_numbers = tuple(pair[0] for pair in pairs)
    probe_numbers = tuple(pair[1] for pair in pairs)
    if (
        not all(pair in same_class_edges for pair in pairs)
        or not any(pair in grace_edges for pair in pairs)
        or gate_numbers != tuple(range(len(pairs)))
        or probe_numbers != tuple(range(len(probes) - len(pairs), len(probes)))
        or any(
            checkpoint - probes[index][0] / MINUTES_PER_STOP <= gate_index
            for index in range(len(probes) - len(pairs))
        )
    ):
        return []

    try:
        prior_items = tuple((prior_gate_assignments or {}).items())
    except AttributeError:
        return []
    relevant_prior_items = []
    for previous_source, previous_gate in prior_items:
        if (
            not isinstance(previous_source, int)
            or isinstance(previous_source, bool)
            or not isinstance(previous_gate, int)
            or isinstance(previous_gate, bool)
        ):
            return []
        source_is_current = previous_source in route_by_source
        gate_is_current = previous_gate in gate_sources
        if not source_is_current and not gate_is_current:
            continue
        if source_is_current != gate_is_current:
            return []
        previous_index = getattr(route_by_source[previous_source], "index", None)
        if (
            not isinstance(previous_index, int)
            or isinstance(previous_index, bool)
            or not 0 <= previous_index < checkpoint
        ):
            return []
        relevant_prior_items.append((previous_source, previous_gate))
    represented_gate_sources = {
        gates[gate_number][1] for gate_number, _probe_number in pairs
    }
    if any(
        previous_gate not in represented_gate_sources
        for _previous_source, previous_gate in relevant_prior_items
    ):
        return []
    for gate_number, probe_number in pairs:
        current_source = probes[probe_number][2]
        current_row = probes[probe_number][3]
        gate_source = gates[gate_number][1]
        history = []
        for previous_source, previous_gate in relevant_prior_items:
            if previous_gate != gate_source:
                continue
            previous_row = route_by_source.get(previous_source)
            if previous_row is None or previous_source == current_source:
                return []
            previous_index = getattr(previous_row, "index", None)
            if (
                not isinstance(previous_index, int)
                or isinstance(previous_index, bool)
                or not 0 <= previous_index < checkpoint
                or kind_class(previous_row) != kind_class(current_row)
            ):
                return []
            history.append((previous_index, previous_source, previous_row))
        history.sort(key=lambda item: (item[0], item[1]))
        history.append((checkpoint, current_source, current_row))
        if any(
            left[0] >= right[0]
            or left[2].signed_minutes >= right[2].signed_minutes
            or left[2].arrival_at.timestamp() >= right[2].arrival_at.timestamp()
            for left, right in zip(history, history[1:], strict=False)
        ):
            return []
    return [
        (probes[probe_number][2], gates[gate_number][1])
        for gate_number, probe_number in pairs
    ]


def _plan_gate_associations(
    probe_inputs: list[object],
    authoritative_inputs: list[object],
    route_keys: set[tuple[str, str, str]],
    verified_gate_indices: Mapping[tuple[str, str, str], int] | None = None,
    observed_checkpoint_indices: Mapping[
        tuple[str, str, str], Collection[int]
    ] | None = None,
) -> _GateAssociationPlan:
    """Build the shared source-identity plan used by estimator and auditor."""
    probe_rows_by_occurrence: dict[
        tuple[str, str, str], dict[int, list[tuple[int, object]]]
    ] = {}
    explicit_empty_probe_occurrences: dict[
        tuple[str, str, str], set[int]
    ] = {}
    gate_rows_by_direction: dict[
        tuple[str, str, str], list[tuple[int, object]]
    ] = {}
    for input_index, eta in enumerate(probe_inputs):
        key = (str(eta.operator), str(eta.route), str(eta.bound))
        if key not in route_keys:
            continue
        checkpoint = int(eta.index)
        if eta.minutes is None or eta.kind is EtaKind.UNAVAILABLE:
            explicit_empty_probe_occurrences.setdefault(key, set()).add(checkpoint)
            continue
        probe_rows_by_occurrence.setdefault(key, {}).setdefault(
            checkpoint, []
        ).append((input_index, eta))
    for raw_key, raw_indices in (observed_checkpoint_indices or {}).items():
        key = tuple(str(part) for part in raw_key)
        if len(key) != 3 or key not in route_keys:
            continue
        active = probe_rows_by_occurrence.get(key, {})
        for raw_index in raw_indices:
            try:
                checkpoint = int(raw_index)
            except (TypeError, ValueError):
                continue
            if checkpoint >= 0 and checkpoint not in active:
                explicit_empty_probe_occurrences.setdefault(key, set()).add(checkpoint)
    for input_index, eta in enumerate(authoritative_inputs):
        key = (str(eta.operator), str(eta.route), str(eta.bound))
        if (
            key in route_keys
            and eta.minutes is not None
            and eta.kind is not EtaKind.UNAVAILABLE
        ):
            gate_rows_by_direction.setdefault(key, []).append((input_index, eta))

    configured_gate_indices: dict[tuple[str, str, str], int] = {}
    for raw_key, raw_index in (verified_gate_indices or {}).items():
        try:
            key = tuple(str(part) for part in raw_key)
            gate_index = int(raw_index)
        except (TypeError, ValueError):
            continue
        if len(key) == 3 and key in route_keys and gate_index >= 0:
            configured_gate_indices[key] = gate_index

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
        atomic_frontier = False
        atomic_frontier_broken = False
        route_probe_rows = probe_rows_by_occurrence.get(key, {})
        route_empty_occurrences = explicit_empty_probe_occurrences.get(key, set())
        for checkpoint in sorted(set(route_probe_rows) | route_empty_occurrences):
            checkpoint_rows = route_probe_rows.get(checkpoint, [])
            explicit_empty = checkpoint in route_empty_occurrences
            if explicit_empty and not checkpoint_rows:
                if (checkpoint > gate_index and atomic_frontier and key[0] == "KMB"
                        and key[1] in ATOMIC_LADDER_FULL_KMB_ROUTES):
                    atomic_frontier_broken = True
                    atomic_frontier = False
                    frontier = []
                    frontier_gate_assignments = {}
                    frontier_checkpoint = checkpoint
                continue
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
            if (explicit_empty and atomic_frontier and key[0] == "KMB"
                    and key[1] in ATOMIC_LADDER_FULL_KMB_ROUTES):
                # An explicit successful-empty row is evidence; an absent stop
                # was merely not probed.  A KMB atomic frontier cannot jump the
                # former and later resume within the same route response.
                atomic_frontier_broken = True
                atomic_frontier = False
                frontier = []
                frontier_gate_assignments = {}
                frontier_checkpoint = checkpoint

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
            atomic_fresh_pairs = _atomic_kmb_fresh_root_pairs(
                key,
                gate_rows,
                checkpoint_rows,
                [
                    pair
                    for rows in route_probe_rows.values()
                    for pair in rows
                ],
                gate_index=gate_index,
                checkpoint=checkpoint,
                prior_gate_assignments=gate_assignment,
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
            certificate = not atomic_frontier_broken and _atomic_kmb_frontier_certificate(
                key, frontier, checkpoint_rows, gate_index=gate_index,
                previous_rows=probe_rows_by_occurrence[key].get(frontier_checkpoint, ()),
                previous_checkpoint=frontier_checkpoint, checkpoint=checkpoint,
                gate_assignments=frontier_gate_assignments, gate_rows=gate_rows_by_input,
                passed_rows=passed_probe_rows, pairs=frontier_pairs, carried=atomic_frontier,
            )
            if atomic_frontier and not certificate and not atomic_fresh_pairs:
                atomic_frontier_broken = True
            if atomic_fresh_pairs:
                # This is a new direct root census after the old sparse
                # frontier ended. It may start a new certificate, but it never
                # carries an identity across the missing checkpoint.
                atomic_frontier_broken = False
            atomic_frontier = bool(certificate or atomic_fresh_pairs)
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
                    allow_downstream_clock_skew=True,
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
            if atomic_fresh_pairs:
                for current_input, fresh_gate in atomic_fresh_pairs:
                    gate_assignment[current_input] = fresh_gate
                    reserved.add(current_input)
                    fresh_gate_inputs.add(fresh_gate)
            elif (
                not certificate
                and fresh_age + PROBE_FRESHNESS_MARGIN_SECONDS < frontier_age
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
            if frontier and not atomic_fresh_pairs:
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
                    if previous_gate is not None and not certificate and not _align_gate_arrivals(
                        [
                            (previous_gate, gate_rows_by_input[previous_gate])
                        ],
                        [(current_input, current_row)],
                        gate_index=gate_index,
                        checkpoint=checkpoint,
                        allow_downstream_clock_skew=True,
                    ):
                        # Without a whole atomic frontier certificate, retain
                        # direct-gate revalidation and normal rematching.
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

    # A completed sparse probe generation is not an exhaustive vehicle
    # census, so it cannot safely reunite arbitrary temporal fragments.  The
    # exact configured HKUST occurrence is narrower evidence: while the
    # separate gate feed is empty, each fresh gate-probe occurrence can act as
    # an ordered one-to-one identity pivot.  It remains an ordinary probe for
    # positioning, provenance, freshness, and audit purposes.
    provisional_gate_tracks: dict[int, int] = {}
    provisional_undeparted_inputs: set[int] = set()

    def fresh_provisional_row(row: object) -> bool:
        try:
            age_value = getattr(row, "cache_age_seconds", None)
            revision_value = getattr(row, "refresh_generation", None)
            if isinstance(age_value, bool) or isinstance(revision_value, bool):
                return False
            age = float(age_value)
            revision = int(revision_value)
            minutes = float(row.minutes)
        except (TypeError, ValueError, OverflowError):
            return False
        return (
            math.isfinite(age)
            and 0.0 <= age < PROVISIONAL_GATE_PROBE_FRESHNESS_SECONDS
            and revision > 0
            and math.isfinite(minutes)
            and minutes >= 0.0
        )

    def provisional_raw_position(row: object) -> float:
        return int(row.index) - float(row.minutes) / MINUTES_PER_STOP

    for key, gate_index in configured_gate_indices.items():
        if key[0] != "GMB" or key in gate_rows_by_direction:
            continue
        checkpoints = probe_rows_by_occurrence.get(key, {})
        gate_rows = [
            pair for pair in checkpoints.get(gate_index, ())
            if fresh_provisional_row(pair[1])
            and provisional_raw_position(pair[1]) >= 0.0
        ]
        if not gate_rows:
            continue
        for gate_input, _gate_row in gate_rows:
            provisional_gate_tracks[gate_input] = gate_input
        future_origin_inputs: dict[int, set[int]] = {}
        for checkpoint, checkpoint_rows in sorted(checkpoints.items()):
            if checkpoint == gate_index:
                continue
            fresh_rows = [
                pair for pair in checkpoint_rows
                if fresh_provisional_row(pair[1])
                and (
                    checkpoint == 0
                    or provisional_raw_position(pair[1]) >= 0.0
                )
            ]
            for probe_input, gate_input in _align_gate_arrivals(
                gate_rows,
                fresh_rows,
                gate_index=gate_index,
                checkpoint=checkpoint,
                rank_first=True,
            ):
                row = probe_inputs[probe_input]
                raw_position = provisional_raw_position(row)
                if raw_position >= 0.0:
                    provisional_gate_tracks[probe_input] = gate_input
                elif (
                    checkpoint == 0
                    and float(row.minutes) > TERMINUS_DEPARTURE_GRACE_MINUTES
                ):
                    # A future origin is veto evidence only.  It suppresses
                    # the matched provisional identity but never bypasses the
                    # ordinary nonnegative positioning filter or gains source
                    # ownership in that marker.
                    future_origin_inputs.setdefault(gate_input, set()).add(
                        probe_input
                    )

        members_by_gate: dict[int, set[int]] = {}
        for probe_input, gate_input in provisional_gate_tracks.items():
            probe = probe_inputs[probe_input]
            if (
                str(probe.operator),
                str(probe.route),
                str(probe.bound),
            ) == key:
                members_by_gate.setdefault(gate_input, set()).add(probe_input)
        for gate_input, members in members_by_gate.items():
            future_inputs = future_origin_inputs.get(gate_input, set())
            if future_inputs:
                provisional_undeparted_inputs.update(members | future_inputs)

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
        provisional_gate_tracks,
        frozenset(provisional_undeparted_inputs),
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
    verified_gate_indices: Mapping[tuple[str, str, str], int] | None = None,
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
        if isinstance(observed_checkpoint_indices, Mapping):
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
        verified_gate_indices,
        observed_by_route,
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
    provisional_ladders: dict[
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
        if probe_input_index in gate_plan.provisional_undeparted_inputs:
            continue
        if probe_input_index in gate_assignment:
            anchored_ladders.setdefault(key, {}).setdefault(
                gate_assignment[probe_input_index], []
            ).append(row)
            continue
        provisional_gate = gate_plan.provisional_gate_tracks.get(probe_input_index)
        if provisional_gate is not None:
            provisional_ladders.setdefault(key, {}).setdefault(
                provisional_gate, []
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
    provisional_source_groups: set[frozenset[tuple[str, int]]] = set()
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

        if key not in gate_rows_by_direction and not any(
            passed_track_ids.get(rung[4][1]) is not None
            for ladder in ladders
            for rung in ladder
        ):
            ladders = _heal_atomic_kmb_ladder_fragments(
                ladders,
                key,
                probe_inputs,
                terminal_index=stops_count - 1,
            )

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

    # A fresh probe at the configured HKUST occurrence is an identity pivot,
    # not an authoritative position source.  It reunites only rows paired
    # one-to-one by the shared gate matcher, then uses the same ordinary
    # all-positive or due/future boundary extraction as every probe ladder.
    for key, gate_ladders in sorted(provisional_ladders.items()):
        operator_name, route, bound = key
        stops_count = len(list(lines_by_key[key].stops))
        for gate_input, ladder in gate_ladders.items():
            direct = [rung for rung in ladder if rung[4] == ("probe", gate_input)]
            if len(direct) != 1:
                continue
            unreliable = all(rung[2] for rung in ladder)
            if unreliable and len({rung[1] for rung in ladder}) < 2:
                continue
            position = max(rung[0] for rung in ladder)
            if position < 0 or position > stops_count - 1:
                continue
            section = min(math.floor(position), stops_count - 2)
            source_observations = frozenset(rung[4] for rung in ladder)
            provisional_source_groups.add(source_observations)
            candidates.setdefault(
                (operator_name, route, bound, section), []
            ).append(
                (
                    position,
                    _quantize_position(position),
                    unreliable,
                    frozenset(rung[1] for rung in ladder),
                    False,
                    source_observations,
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
            cluster_is_provisional = any(
                item[4] in provisional_source_groups for item in clusters[-1]
            )
            if (
                abs(anchor[0] - clusters[-1][-1][0]) <= 1.0
                and cluster_stops.isdisjoint(anchor[2])
                and not anchor[3]
                and not cluster_is_authoritative
                and anchor[4] not in provisional_source_groups
                and not cluster_is_provisional
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
            boundary_revision = None
            bracket_eta_offsets = None
            priority_indices = frozenset()
            exploratory_indices = frozenset()
            checkpoint_evidence = ()
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
                checkpoint_evidence = _checkpoint_evidence(source_rows)
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
                next_positive = None
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
                boundary_index = first_present
                if zero_indices:
                    due_index = zero_indices[-1]
                    boundary_index = (
                        next_positive if next_positive is not None else due_index
                    )
                    bracket = (float(due_index), float(boundary_index))
                present_rows = [row for row in source_rows
                                if (str(getattr(row, "operator", "")),
                                    str(getattr(row, "route", "")),
                                    str(getattr(row, "bound", "")))
                                == (operator_name, route, bound)
                                and int(getattr(row, "index", -1)) == boundary_index]
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
                    lower_index = int(bracket[0]) if bracket else boundary_index
                    present_age = float(
                        getattr(selected_eta, "cache_age_seconds", 0.0) or 0.0
                    )
                    lower_ages = [
                        float(getattr(row, "cache_age_seconds", 0.0) or 0.0)
                        for row in probe_inputs
                        if (str(getattr(row, "operator", "")),
                            str(getattr(row, "route", "")),
                            str(getattr(row, "bound", "")))
                        == (operator_name, route, bound)
                        if int(getattr(row, "index", -1)) == lower_index
                    ]
                    if lower_index == boundary_index:
                        lower_ages.append(present_age)
                    if lower_ages:
                        boundary_age_seconds = max(present_age, min(lower_ages))
                    # Every boundary needs an independently refreshed endpoint;
                    # successful-empty rows are retained in probe_inputs and
                    # therefore participate here even though they have no ETA.
                    def _row_revision(row):
                        try:
                            revision = int(getattr(row, "refresh_generation", 0) or 0)
                        except (TypeError, ValueError):
                            return 0
                        return revision if revision > 0 else 0

                    selected_lower = None
                    if zero_indices:
                        lower_rows = [
                            row
                            for row in source_rows
                            if int(getattr(row, "index", -1)) == lower_index
                        ]
                        selected_lower = min(
                            lower_rows,
                            key=lambda row: (
                                float(
                                    getattr(row, "cache_age_seconds", 0.0) or 0.0
                                ),
                                abs(
                                    float(
                                        getattr(row, "signed_minutes", None)
                                        if getattr(row, "signed_minutes", None)
                                        is not None
                                        else row.minutes
                                    )
                                ),
                            ),
                        )
                        lower_eta = float(
                            getattr(selected_lower, "signed_minutes", None)
                            if getattr(selected_lower, "signed_minutes", None)
                            is not None
                            else selected_lower.minutes
                        )
                        upper_eta = float(
                            getattr(selected_eta, "signed_minutes", None)
                            if getattr(selected_eta, "signed_minutes", None)
                            is not None
                            else selected_eta.minutes
                        )
                        bracket_eta_offsets = (lower_eta, upper_eta)
                        if lower_index == boundary_index:
                            position = float(lower_index)
                        elif lower_eta <= 0 < upper_eta:
                            fraction = min(
                                1.0,
                                max(0.0, -lower_eta / (upper_eta - lower_eta)),
                            )
                            position = lower_index + (
                                boundary_index - lower_index
                            ) * fraction
                    else:
                        # With no due row, the first stop which sees this ETA
                        # remains the physical upper boundary.
                        position = boundary_index - min(
                            1.0, max(0.0, eta_minutes / MINUTES_PER_STOP)
                        )
                        # A unique fresh realtime row at the physical upper
                        # boundary is a safe search hint even when coherent
                        # downstream or gate observations also identify this
                        # vehicle. The hint never moves a marker by itself;
                        # only the resulting probe responses can do that.
                        if len(present_rows) == 1:
                            exploratory_indices = _eta_guided_priority_indices(
                                selected_eta, bracket, stops_count
                            )
                    if lower_index == boundary_index:
                        selected_lower = selected_eta
                    elif selected_lower is None:
                        endpoint_rows = [
                            row for row in probe_inputs
                            if (str(getattr(row, "operator", "")),
                                str(getattr(row, "route", "")),
                                str(getattr(row, "bound", "")))
                            == (operator_name, route, bound)
                            and int(getattr(row, "index", -1)) == lower_index
                        ]
                        selected_lower = min(
                            endpoint_rows,
                            key=lambda row: float(
                                getattr(row, "cache_age_seconds", 0.0) or 0.0
                            ),
                            default=None,
                        )
                    selected_lower_revision = _row_revision(selected_lower)
                    selected_upper_revision = _row_revision(selected_eta)
                    selected_lower_age = float(
                        getattr(selected_lower, "cache_age_seconds", 0.0) or 0.0
                    ) if selected_lower is not None else None
                    if selected_lower_age is not None:
                        boundary_age_seconds = max(present_age, selected_lower_age)
                    endpoint_revisions = (selected_lower_revision, selected_upper_revision)
                    if all(endpoint_revisions):
                        boundary_revision = endpoint_revisions
                    elif any(
                        int(getattr(row, "refresh_generation", 0) or 0) > 0
                        for row in probe_inputs
                    ):
                        boundary_revision = (0, 0)
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
                    boundary_revision=boundary_revision,
                    bracket_eta_offsets=bracket_eta_offsets,
                    priority_indices=priority_indices,
                    exploratory_indices=exploratory_indices,
                    checkpoint_evidence=checkpoint_evidence,
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


def rebuild_estimate_from_probe_fragments(
    base: BusEstimate,
    fragments,
    probe_inputs,
    route_lines,
) -> BusEstimate:
    """Rebuild a certified split identity's current due/future boundary.

    The tracker proves physical ownership before calling this helper. This
    function only derives positioning from the union of current probe rows; it
    never uses retained cohort timestamps as motion evidence.
    """
    estimates = [base, *fragments]
    observations = frozenset().union(
        *(estimate.source_observations for estimate in estimates)
    )
    if not observations:
        return base
    key = (
        str(base.operator_code or base.operator),
        str(base.route),
        str(base.bound),
    )
    source_rows = []
    # Validate before attempting any ordering: malformed mixed token types
    # must fail closed without allowing tuple comparison to raise.
    for source in tuple(observations):
        if not isinstance(source, tuple) or len(source) != 2:
            return base
        _kind, input_index = source
        if _kind != "probe":
            return base
        if (isinstance(input_index, bool) or not isinstance(input_index, int)
                or not 0 <= input_index < len(probe_inputs)):
            return base
        row = probe_inputs[input_index]
        if (
            str(getattr(row, "operator", "")),
            str(getattr(row, "route", "")),
            str(getattr(row, "bound", "")),
        ) != key:
            return base
        source_rows.append(row)

    # Route responses not attached to this identity are absence checkpoints.
    # They must nevertheless participate in reconstructing an all-positive
    # boundary, but only for this exact route.
    current_route_rows = [
        row for row in probe_inputs
        if (
            str(getattr(row, "operator", "")),
            str(getattr(row, "route", "")),
            str(getattr(row, "bound", "")),
        ) == key
    ]

    try:
        zero_indices = sorted({
            int(row.index) for row in source_rows
            if row.minutes is not None and float(row.minutes) <= 0.0
        })
        positive_indices = sorted({
            int(row.index) for row in source_rows
            if row.minutes is not None and float(row.minutes) > 0.0
        })
    except (AttributeError, TypeError, ValueError, OverflowError):
        return base
    # Rebuild the same physical frontier as estimate_bus_positions.  Empty
    # rows are successful absence checkpoints, not unusable observations.
    # A positive-only union needs an observed empty rung before its first
    # present stop; otherwise the identity has no trustworthy lower endpoint.
    try:
        current_indices = {int(row.index) for row in current_route_rows}
        active_indices = {int(row.index) for row in source_rows}
    except (AttributeError, TypeError, ValueError, OverflowError):
        return base
    if zero_indices:
        lower_index = zero_indices[-1]
        upper_index = next(
            (index for index in positive_indices if index > lower_index),
            None,
        )
        if upper_index is None:
            # A final due observation is a point bracket, matching the normal
            # estimator's due-only handling.
            upper_index = lower_index
    elif positive_indices:
        upper_index = positive_indices[0]
        # Any currently observed upstream stop not owned by this identity is
        # an absence checkpoint. Its ETA may belong to another bus; use it
        # only for endpoint freshness, never as this identity's ETA.
        lower_candidates = [
            index for index in current_indices
            if index < upper_index and index not in active_indices
        ]
        if not lower_candidates:
            return base
        lower_index = max(lower_candidates)
    else:
        return base

    lower_rows = [
        row for row in (current_route_rows if not zero_indices else source_rows)
        if int(row.index) == lower_index
    ]
    upper_rows = [row for row in source_rows if int(row.index) == upper_index]
    # Due-only boundaries use the same physical endpoint twice.  Keep one
    # selected row and derive a point position; no interpolation is invented.
    if upper_index == lower_index:
        upper_rows = lower_rows
    if not lower_rows or not upper_rows:
        return base
    try:
        selected_lower = min(
            lower_rows,
            key=lambda row: (
                float(getattr(row, "cache_age_seconds", 0.0) or 0.0),
                abs(float(
                    row.signed_minutes
                    if getattr(row, "signed_minutes", None) is not None
                    else (row.minutes if row.minutes is not None else 0.0)
                )),
            ),
        )
        selected_upper = min(
            upper_rows,
            key=lambda row: (
                float(getattr(row, "cache_age_seconds", 0.0) or 0.0),
                float(row.minutes),
            ),
        )
        lower_eta = float(
            selected_lower.signed_minutes
            if getattr(selected_lower, "signed_minutes", None) is not None
            else (selected_lower.minutes if selected_lower.minutes is not None else 0.0)
        )
        upper_eta = float(
            selected_upper.signed_minutes
            if getattr(selected_upper, "signed_minutes", None) is not None
            else selected_upper.minutes
        )
        lower_age = float(
            getattr(selected_lower, "cache_age_seconds", 0.0) or 0.0
        )
        upper_age = float(
            getattr(selected_upper, "cache_age_seconds", 0.0) or 0.0
        )
    except (AttributeError, TypeError, ValueError, OverflowError):
        return base
    if (
        not all(math.isfinite(value) for value in (
            lower_eta, upper_eta, lower_age, upper_age
        ))
        or lower_age < 0.0
        or upper_age < 0.0
        or (zero_indices and upper_index != lower_index
            and not lower_eta <= 0.0 < upper_eta)
        or (upper_index == lower_index and lower_eta > 0.0)
    ):
        return base

    if upper_index == lower_index:
        position = float(lower_index)
    elif not zero_indices:
        # With no due row, the first present stop is the physical upper
        # boundary.  Keep the same one-stop ETA offset used by the estimator.
        position = upper_index - min(1.0, max(0.0, upper_eta / MINUTES_PER_STOP))
    else:
        fraction = min(1.0, max(0.0, -lower_eta / (upper_eta - lower_eta)))
        position = lower_index + (upper_index - lower_index) * fraction
    merged = reproject_estimate(base, position, route_lines)

    def revision(row):
        try:
            value = int(getattr(row, "refresh_generation", 0) or 0)
        except (TypeError, ValueError, OverflowError):
            return 0
        return value if value > 0 else 0

    endpoint_revisions = (
        revision(selected_lower),
        revision(selected_upper),
    )
    if all(endpoint_revisions):
        boundary_revision = endpoint_revisions
    elif any(revision(row) for row in source_rows):
        boundary_revision = (0, 0)
    else:
        boundary_revision = None
    source_indices = frozenset().union(
        *(estimate.source_indices for estimate in estimates)
    )
    source_indices = frozenset({
        *source_indices,
        *(int(row.index) for row in source_rows
          if getattr(row, "minutes", None) is not None),
    })
    return replace(
        merged,
        source_indices=source_indices,
        source_observations=observations,
        bracket=(float(lower_index), float(upper_index)),
        eta_minutes=float(selected_upper.minutes),
        eta_arrival_at=getattr(selected_upper, "arrival_at", None),
        bracket_initial_eta=float(selected_upper.minutes),
        bracket_eta_offsets=(lower_eta, upper_eta) if zero_indices else None,
        boundary_age_seconds=max(lower_age, upper_age),
        boundary_revision=boundary_revision,
        priority_indices=(
            frozenset({*zero_indices, upper_index})
            if zero_indices
            else frozenset(positive_indices[:2])
        ),
        exploratory_indices=frozenset(),
        checkpoint_evidence=_checkpoint_evidence(source_rows),
    )


def rebuild_estimate_from_probe_sources(
    template: BusEstimate,
    selected_source_slots,
    probe_inputs,
    route_lines,
) -> BusEstimate | None:
    """Rebuild one identity from an explicitly owned probe-slot subset.

    Unlike fragment unioning, this deliberately discards the template's
    source/checkpoint metadata and lets the normal boundary derivation rebuild
    it from only the selected raw rows.  Invalid or empty selections fail
    closed instead of returning a contaminated template.
    """
    try:
        slots = tuple(selected_source_slots or ())
        if len(set(slots)) != len(slots):
            return None
    except TypeError:
        return None
    if not slots:
        return None
    key = (str(template.operator_code or template.operator),
           str(template.route), str(template.bound))
    line = next((line for line in route_lines if (
        str(getattr(line, "operator", "")), str(getattr(line, "route", "")),
        str(getattr(line, "bound", "")),
    ) == key), None)
    if (line is None or len(getattr(line, "stops", ())) < 2
            or len(getattr(line, "stop_offsets", ())) != len(line.stops)
            or not getattr(line, "path", ())):
        return None
    selected = []
    for slot in slots:
        if isinstance(slot, bool) or not isinstance(slot, int):
            return None
        if not 0 <= slot < len(probe_inputs):
            return None
        row = probe_inputs[slot]
        if (str(getattr(row, "operator", "")), str(getattr(row, "route", "")),
                str(getattr(row, "bound", ""))) != key:
            return None
        if (isinstance(getattr(row, "minutes", None), bool)
                or isinstance(getattr(row, "cache_age_seconds", None), bool)
                or isinstance(getattr(row, "signed_minutes", None), bool)):
            return None
        try:
            minutes = float(row.minutes)
            age = float(row.cache_age_seconds)
            arrival = float(row.arrival_at.timestamp())
        except (AttributeError, TypeError, ValueError, OverflowError, OSError):
            return None
        if (getattr(row, "kind", None) not in LIVE_PROBE_ETA_KINDS
                or not all(math.isfinite(value) for value in (minutes, age, arrival))
                or not 0 <= age < 900
                or not isinstance(row.index, int) or isinstance(row.index, bool)
                or not 0 <= row.index < len(line.stops)
                or not isinstance(row.refresh_generation, int)
                or isinstance(row.refresh_generation, bool)
                or row.refresh_generation <= 0):
            return None
        selected.append(row)
    if not selected:
        return None
    # Boundary reconstruction may use a successful-empty response or another
    # bus's upstream checkpoint. Reject malformed route context instead of
    # allowing such a row to manufacture a lower endpoint.
    for row in probe_inputs:
        if (str(getattr(row, "operator", "")), str(getattr(row, "route", "")),
                str(getattr(row, "bound", ""))) != key:
            continue
        try:
            valid = (isinstance(row.index, int) and not isinstance(row.index, bool)
                     and 0 <= row.index < len(line.stops)
                     and isinstance(row.refresh_generation, int)
                     and not isinstance(row.refresh_generation, bool)
                     and row.refresh_generation > 0
                     and not isinstance(row.cache_age_seconds, bool)
                     and not isinstance(row.minutes, bool)
                     and not isinstance(getattr(row, "signed_minutes", None), bool)
                     and 0 <= float(row.cache_age_seconds) < 900)
        except (AttributeError, TypeError, ValueError, OverflowError):
            return None
        if not valid:
            return None
    clean = replace(
        template,
        source_observations=frozenset(("probe", slot) for slot in slots),
        source_indices=frozenset(),
        checkpoint_evidence=(),
        boundary_revision=None,
        boundary_age_seconds=None,
    )
    rebuilt = rebuild_estimate_from_probe_fragments(clean, (), probe_inputs, route_lines)
    if rebuilt is clean:
        return None
    return rebuilt


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
