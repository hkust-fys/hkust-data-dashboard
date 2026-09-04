"""Estimated bus position tests: ladder-collapsed vehicle reconstruction."""

from dashboard.maps.marker_audit import audit_marker_positions
from dashboard.maps.positions import (
    BusEstimate,
    _align_gate_arrivals,
    _passed_row_position,
    _path_segment_length,
    _plan_gate_associations,
    _quantize_position,
    _separate_common_stop_departures,
    estimate_bus_positions,
)
from dashboard.models import EtaKind
from dashboard.providers.route_geometry import RouteLine, Stop


class Probe:
    """Minimal probe-ETA stand-in (duck-typed like the provider's ProbeEta)."""

    def __init__(
        self,
        operator,
        route,
        bound,
        index,
        minutes,
        kind=None,
        cache_age_seconds=None,
    ):
        self.operator = operator
        self.route = route
        self.bound = bound
        self.index = index
        self.minutes = minutes
        self.stop_id = "S"
        self.kind = kind or EtaKind.REALTIME
        self.cache_age_seconds = cache_age_seconds


class AuthoritativeProbe(Probe):
    authoritative = True


def _line(operator="KMB", route="X", bound="outbound", stop_count=6):
    stops = [
        Stop(f"{index}", f"Stop {index}", 22.333360, 114.26 + index * 0.001)
        for index in range(stop_count)
    ]
    path = [(stop.lat, stop.lon) for stop in stops]
    offsets = [0.0]
    for first, second in zip(stops, stops[1:], strict=False):
        offsets.append(
            offsets[-1]
            + _path_segment_length((first.lat, first.lon), (second.lat, second.lon))
        )
    return RouteLine(route, operator, bound, stops, path, offsets)


def test_timetable_ladder_collapses_to_one_vehicle():
    """One real bus announced at many stops with timetable-interpolated ETAs
    leaves a ladder of implied positions rising ~1 stop per stop. It must
    collapse to ONE marker anchored at the maximum (closest announcement)."""
    line = _line()
    # Implied positions 1.0 and 2.0 — a rising ladder within merge gap.
    estimates = estimate_bus_positions(
        [
            Probe("KMB", "X", "outbound", 3, 4),
            Probe("KMB", "X", "outbound", 4, 4),
        ],
        [line],
    )
    assert len(estimates) == 1
    estimate = estimates[0]
    assert isinstance(estimate, BusEstimate)
    # Anchored at the maximum implied position (2.0).
    assert abs(estimate.lon - line.stops[2].lon) < 1e-6


def test_long_upstream_hop_rejects_impossible_gate_match():
    gate_rows = [
        (0, AuthoritativeProbe("GMB", "11", "seq-1", 6, 10)),
        (1, AuthoritativeProbe("GMB", "11", "seq-1", 6, 7)),
    ]
    probe_rows = [
        (0, Probe("GMB", "11", "seq-1", 0, 10.1128)),
        (1, Probe("GMB", "11", "seq-1", 0, 0.1128)),
    ]
    pairs = _align_gate_arrivals(
        gate_rows, probe_rows, gate_index=6, checkpoint=0
    )
    assert pairs == [(1, 0)]


def test_impossible_upstream_probe_does_not_move_gate_markers():
    line = _line("GMB", "11", "seq-1", stop_count=20)
    probes = [
        Probe("GMB", "11", "seq-1", 0, 10.1128),
        Probe("GMB", "11", "seq-1", 0, 0.1128),
    ]
    gates = [
        AuthoritativeProbe("GMB", "11", "seq-1", 6, 10),
        AuthoritativeProbe("GMB", "11", "seq-1", 6, 7),
    ]
    estimates = estimate_bus_positions(probes, [line], authoritative_etas=gates)
    assert [round(estimate.position, 3) for estimate in estimates] == [1.0, 2.5]
    assert any(("probe", 1) in estimate.source_observations for estimate in estimates)
    assert not any(("probe", 0) in estimate.source_observations for estimate in estimates)


def test_partial_downstream_frame_keeps_future_gate_journey_separate():
    line = _line("CTB", "792M", "inbound", stop_count=30)
    line.stops[16] = Stop("003130", "HKUST", 22.333360, 114.276)
    values = {
        22: [6.561, 29.461, 54.461],
        24: [10.559, 33.776, 58.776],
        26: [16.411, 39.644],
        28: [18.546, 41.530],
    }
    probes = [
        Probe("CTB", "792M", "inbound", stop, minutes)
        for stop, etas in values.items()
        for minutes in etas
    ]
    gates = [
        AuthoritativeProbe("CTB", "792M", "inbound", 16, 21),
        AuthoritativeProbe("CTB", "792M", "inbound", 16, 46),
    ]
    estimates = estimate_bus_positions(probes, [line], authoritative_etas=gates)
    assert len(estimates) == 2
    positions = sorted(estimate.position for estimate in estimates)
    assert positions[0] < 16.0
    assert positions[1] > 16.0
    assert positions[1] - positions[0] >= 5.0
    audit = audit_marker_positions(probes, gates, estimates, [line])
    assert audit["ok"]
    assert audit["stats"]["uncovered_checkpoints"] == 0
    assert audit["stats"]["uncovered_probe_rows"] == 0
    assert audit["stats"]["observed_checkpoints"] == audit["stats"]["audited_checkpoints"]
    assert audit["stats"]["observed_probe_rows"] == audit["stats"]["audited_probe_rows"]
    assert abs(positions[0] - 5.5) < 1.0
    assert abs(positions[1] - 18.7) < 1.0


def test_fresh_gate_coverage_beats_two_row_stale_frontier():
    line = _line("GMB", "11", "seq-1", stop_count=20)
    probes = [
        Probe("GMB", "11", "seq-1", 9, eta, cache_age_seconds=80.75)
        for eta in (10.277698866, 16.1717822, 31.1717822)
    ] + [
        Probe("GMB", "11", "seq-1", 10, eta, cache_age_seconds=1.11)
        for eta in (6.661798167, 10.759448167, 17.2438315)
    ]
    gates = [
        AuthoritativeProbe("GMB", "11", "seq-1", 6, eta)
        for eta in (2, 7, 12)
    ]
    plan = _plan_gate_associations(probes, gates, {("GMB", "11", "seq-1")})
    assert {3: 0, 4: 1, 5: 2}.items() <= plan.gate_assignment.items()
    assert not ({3, 4, 5} & plan.passed_probe_rows)
    estimates = estimate_bus_positions(probes, [line], authoritative_etas=gates)
    assert len(estimates) == 3
    assert {
        frozenset(estimate.source_observations)
        for estimate in estimates
    } == {
        frozenset({("gate", 0), ("probe", 0), ("probe", 3)}),
        frozenset({("gate", 1), ("probe", 1), ("probe", 4)}),
        frozenset({("gate", 2), ("probe", 2), ("probe", 5)}),
    }
    audit = audit_marker_positions(probes, gates, estimates, [line])
    assert audit["ok"]
    assert not audit["issues"]
    assert audit["stats"]["uncovered_checkpoints"] == 0
    assert audit["stats"]["uncovered_probe_rows"] == 0
    assert audit["stats"]["observed_checkpoints"] == audit["stats"]["audited_checkpoints"]
    assert audit["stats"]["observed_probe_rows"] == audit["stats"]["audited_probe_rows"]


def test_stale_intermediate_checkpoint_does_not_create_extra_passed_track():
    line = _line("GMB", "11", "seq-1", stop_count=25)
    values = {
        15: (9.3791491, 15.3028991),
        16: (7.5386931, 11.499176433),
        17: (3.285067883, 14.239217883, 19.93273455),
    }
    probes = [
        Probe(
            "GMB",
            "11",
            "seq-1",
            stop,
            eta,
            cache_age_seconds=0.828 if stop == 15 else 152.047 if stop == 16 else 41.203,
        )
        for stop, etas in values.items()
        for eta in etas
    ]
    gates = [AuthoritativeProbe("GMB", "11", "seq-1", 6, eta) for eta in (4, 19)]
    estimates = estimate_bus_positions(probes, [line], authoritative_etas=gates)
    assert len(estimates) == 3
    assert {
        frozenset(estimate.source_observations)
        for estimate in estimates
    } == {
        frozenset({("gate", 0), ("probe", 0), ("probe", 3), ("probe", 6)}),
        frozenset({("probe", 1), ("probe", 2), ("probe", 5)}),
        frozenset({("probe", 4)}),
    }
    cp17_inputs = {
        input_index
        for input_index, row in enumerate(probes)
        if row.index == 17
    }
    owners = {
        input_index: estimate
        for estimate in estimates
        for _kind, input_index in estimate.source_observations
        if _kind == "probe" and input_index in cp17_inputs
    }
    assert set(owners) == cp17_inputs
    assert len({id(estimate) for estimate in owners.values()}) == 3
    audit = audit_marker_positions(probes, gates, estimates, [line])
    assert audit["ok"]
    assert not audit["issues"]
    assert audit["stats"]["uncovered_checkpoints"] == 0
    assert audit["stats"]["uncovered_probe_rows"] == 0
    assert audit["stats"]["observed_checkpoints"] == audit["stats"]["audited_checkpoints"]
    assert audit["stats"]["observed_probe_rows"] == audit["stats"]["audited_probe_rows"]


def test_zero_minutes_at_terminus_renders_at_terminus():
    """A bus whose ETA has matured to 0 at the first stop sits ON the terminus
    and must render there (same rule as 0 minutes on top of HKUST)."""
    line = _line()
    estimates = estimate_bus_positions(
        [Probe("KMB", "X", "outbound", 0, 0)],
        [line],
    )
    assert len(estimates) == 1
    assert abs(estimates[0].lon - line.stops[0].lon) < 1e-6


def test_undeparted_terminus_bus_does_not_render():
    """ETA > 0 at the terminus means the bus has NOT left yet — no marker."""
    line = _line()
    assert estimate_bus_positions([Probe("KMB", "X", "outbound", 0, 3)], [line]) == []
    assert estimate_bus_positions([Probe("KMB", "X", "outbound", 0, 1)], [line]) == []


def test_separate_buses_offset_by_headway_stay_separate():
    """Two ladders offset by more than LADDER_GAP_STOPS are two vehicles."""
    line = _line()
    estimates = estimate_bus_positions(
        [
            # Bus A: implied ~1.0.
            Probe("KMB", "X", "outbound", 3, 4),
            Probe("KMB", "X", "outbound", 4, 6),
            # Bus B: implied ~3.5.
            Probe("KMB", "X", "outbound", 4, 1),
        ],
        [line],
    )
    assert len(estimates) == 2


def test_scheduled_rows_render_unreliable():
    """'Scheduled' rows are timetable evidence, not live tracking: once the
    ETA matures the bus is plausibly on the road, so it renders — flagged
    unreliable (paler, dashed outline) rather than silently dropped."""
    line = _line()
    estimates = estimate_bus_positions(
        [
            Probe("KMB", "X", "outbound", 3, 4, kind=EtaKind.SCHEDULED),
            Probe("KMB", "X", "outbound", 4, 6, kind=EtaKind.SCHEDULED),
        ],
        [line],
    )
    assert len(estimates) == 1
    assert estimates[0].unreliable is True


def test_two_stop_scheduled_ladder_is_retained():
    """Two timetable rows at distinct stops corroborate an infrequent route bus."""
    line = _line()
    estimates = estimate_bus_positions(
        [
            Probe("KMB", "X", "outbound", 3, 2, kind=EtaKind.SCHEDULED),
            Probe("KMB", "X", "outbound", 4, 4, kind=EtaKind.SCHEDULED),
        ],
        [line],
    )
    assert len(estimates) == 1
    assert estimates[0].unreliable is True


def test_lone_scheduled_probe_row_is_suppressed():
    """A single timetable row must not create a phantom infrequent-route bus."""
    line = _line()
    assert estimate_bus_positions(
        [Probe("KMB", "X", "outbound", 3, 2, kind=EtaKind.SCHEDULED)],
        [line],
    ) == []


def test_sparse_scheduled_route_artifacts_are_suppressed():
    """Observed long-ETA singleton rows from infrequent routes are phantoms."""
    cases = (
        ("KMB", "91M", "outbound", 21, 41.9, 23),
        ("GMB", "104", "seq-1", 10, 19.9, 24),
    )
    for operator, route, bound, index, minutes, stop_count in cases:
        line = _line(operator, route, bound, stop_count)
        assert estimate_bus_positions(
            [Probe(operator, route, bound, index, minutes, kind=EtaKind.SCHEDULED)],
            [line],
        ) == []


def test_partial_scheduled_sweep_needs_distinct_stop_corroboration():
    line = _line()
    singleton = Probe("KMB", "X", "outbound", 3, 2, kind=EtaKind.SCHEDULED)
    corroborating = Probe("KMB", "X", "outbound", 4, 4, kind=EtaKind.SCHEDULED)
    assert estimate_bus_positions([singleton], [line]) == []
    estimates = estimate_bus_positions([singleton, corroborating], [line])
    assert len(estimates) == 1
    assert estimates[0].unreliable is True


def test_gmb11_close_realtime_convoy_remains_two_vehicles():
    """Shared downstream stop observations prove two close live vehicles."""
    line = _line("GMB", "11", "seq-1", 24)
    estimates = estimate_bus_positions(
        [
            Probe("GMB", "11", "seq-1", 10, 4),
            Probe("GMB", "11", "seq-1", 10, 2),
            Probe("GMB", "11", "seq-1", 11, 4),
            Probe("GMB", "11", "seq-1", 11, 2),
        ],
        [line],
    )
    assert len(estimates) == 2
    assert sorted(estimate.position for estimate in estimates) == [9.0, 10.0]
    assert all(not estimate.unreliable for estimate in estimates)


def test_common_stop_headway_separates_collapsed_gmb11_vehicles():
    """Large ETA gaps at repeated stops must remain visible in map positions."""
    line = _line("GMB", "11", "seq-1", 24)
    rows = [
        (7, 3.208), (8, 4.545), (9, 8.381), (11, 14.849),
        (12, 17.02), (13, 7.957), (13, 21.509), (15, 11.937),
        (15, 25.488), (16, 13.197), (17, 17.626), (17, 31.177),
        (18, 20.035), (19, 26.3), (19, 39.852), (20, 30.402),
        (20, 43.953), (21, 32.722), (21, 46.274), (22, 32.706),
        (22, 46.257), (23, 34.633),
    ]
    estimates = estimate_bus_positions(
        [Probe("GMB", "11", "seq-1", index, minutes) for index, minutes in rows],
        [line],
    )
    assert len(estimates) == 2
    first = next(e for e in estimates if ("probe", 5) in e.source_observations)
    second = next(e for e in estimates if ("probe", 6) in e.source_observations)
    assert first.position > second.position
    assert first.position - second.position >= 5.0


def test_common_stop_headway_correction_is_input_order_independent():
    line = _line("GMB", "11", "seq-1", 24)
    rows = [(13, 7.957), (13, 21.509), (15, 11.937), (15, 25.488)]
    forward = estimate_bus_positions(
        [Probe("GMB", "11", "seq-1", index, minutes) for index, minutes in rows],
        [line],
    )
    reverse = estimate_bus_positions(
        [Probe("GMB", "11", "seq-1", index, minutes) for index, minutes in reversed(rows)],
        [line],
    )
    assert sorted(e.position for e in forward) == sorted(e.position for e in reverse)


def test_close_common_stop_departures_are_not_forced_apart():
    line = _line("GMB", "11", "seq-1", 20)
    estimates = estimate_bus_positions(
        [
            Probe("GMB", "11", "seq-1", 10, 4),
            Probe("GMB", "11", "seq-1", 10, 12),
            Probe("GMB", "11", "seq-1", 11, 6),
            Probe("GMB", "11", "seq-1", 11, 14),
        ],
        [line],
    )
    assert len(estimates) == 2
    assert max(e.position for e in estimates) - min(e.position for e in estimates) < 5.0


def test_signed_common_stop_evidence_overrides_reversed_inferred_order():
    key = ("GMB", "11", "seq-1")
    early = frozenset({("probe", 0)})
    late = frozenset({("probe", 1)})
    corrected = _separate_common_stop_departures(
        [(key, 4.0, False, early), (key, 12.0, False, late)],
        {("probe", 0): (10, 9.0), ("probe", 1): (10, 2.0)},
        {key: 20.0},
    )
    assert corrected[early] > corrected[late]
    assert corrected[early] - corrected[late] >= 3.5


def test_three_vehicle_headway_correction_is_order_independent_and_clamped():
    key = ("GMB", "11", "seq-1")
    records = [
        (key, 19.0, False, frozenset({("probe", 0)})),
        (key, 18.0, False, frozenset({("probe", 1)})),
        (key, 17.0, False, frozenset({("probe", 2)})),
    ]
    evidence = {
        ("probe", 0): (10, 14.0),
        ("probe", 1): (10, 7.0),
        ("probe", 2): (10, 0.0),
    }
    forward = _separate_common_stop_departures(records, evidence, {key: 20.0})
    reverse = _separate_common_stop_departures(
        list(reversed(records)), evidence, {key: 20.0}
    )
    assert forward == reverse
    assert all(0.0 <= position <= 20.0 for position in forward.values())
    assert forward[frozenset({("probe", 0)})] > forward[frozenset({("probe", 1)})]
    assert forward[frozenset({("probe", 1)})] > forward[frozenset({("probe", 2)})]


def test_three_vehicle_signed_order_does_not_invert_after_projection():
    key = ("GMB", "11", "seq-1")
    sources = [frozenset({("probe", index)}) for index in range(3)]
    records = list(zip(
        [3.090377614585228, 19.49097594955635, 17.566816236462124],
        sources,
        strict=True,
    ))
    corrected = _separate_common_stop_departures(
        [(key, position, False, source) for position, source in records],
        {("probe", 0): (10, 10.0), ("probe", 1): (10, 5.0), ("probe", 2): (10, 0.0)},
        {key: 20.0},
    )
    assert corrected[sources[0]] > corrected[sources[1]] > corrected[sources[2]]


def test_uneven_three_track_component_keeps_independent_source_anchors():
    key = ("CTB", "792M", "inbound")
    gate = frozenset({("gate", 0)})
    trailing = frozenset({("probe", 0), ("probe", 4), ("probe", 8), ("probe", 12)})
    middle = frozenset({("probe", 1), ("probe", 5), ("probe", 9), ("probe", 13)})
    leading = frozenset({("probe", 2), ("probe", 6), ("probe", 10), ("probe", 14)})
    records = [
        (key, 13.0, True, gate),
        (key, 17.842, False, trailing),
        (key, 19.239, False, middle),
        (key, 27.856, False, leading),
    ]
    checkpoints = (22, 24, 26, 28)
    evidence = {
        ("probe", offset): (checkpoint, 6.7)
        for offset, checkpoint in zip((0, 4, 8, 12), checkpoints, strict=True)
    }
    evidence.update(
        {
            ("probe", offset): (checkpoint, raw)
            for offsets, raw in (((1, 5, 9, 13), 19.2), ((2, 6, 10, 14), 27.856))
            for offset, checkpoint in zip(offsets, checkpoints, strict=True)
        }
    )
    corrected = _separate_common_stop_departures(records, evidence, {key: 40.0})
    assert abs(corrected[trailing] - 6.7) < 0.25
    assert abs(corrected[middle] - 19.2) < 0.25
    assert abs(corrected[leading] - 27.856) < 0.25
    assert corrected[trailing] < corrected[middle] < corrected[leading]


def test_two_track_residual_median_balances_source_anchor_error():
    key = ("KMB", "91M", "outbound")
    trailing = frozenset({("probe", 0), ("probe", 2)})
    leading = frozenset({("probe", 1), ("probe", 3)})
    records = [
        (key, 18.465, False, trailing),
        (key, 25.2068, False, leading),
    ]
    evidence = {
        ("probe", 0): (27, 18.0),
        ("probe", 1): (27, 26.025),
        ("probe", 2): (28, 18.2),
        ("probe", 3): (28, 25.992),
    }
    corrected = _separate_common_stop_departures(records, evidence, {key: 40.0})
    assert abs(corrected[trailing] - 18.465) < 1.0
    assert abs(corrected[leading] - 25.2068) < 1.0
    assert corrected[leading] > corrected[trailing]
    assert corrected[leading] - corrected[trailing] >= 5.0


def test_passed_component_respects_verified_gate_lower_bound():
    key = ("CTB", "792M", "inbound")
    trailing = frozenset({("probe", 0), ("probe", 2)})
    leading = frozenset({("probe", 1), ("probe", 3)})
    corrected = _separate_common_stop_departures(
        [(key, 19.77, False, trailing), (key, 20.0, False, leading)],
        {
            ("probe", 0): (22, 19.7),
            ("probe", 1): (22, 8.8),
            ("probe", 2): (28, 19.7),
            ("probe", 3): (28, 8.8),
        },
        {key: 29.0},
        {key: 16.001},
    )
    assert corrected[trailing] > 16.0
    assert corrected[leading] > 16.0
    assert corrected[trailing] > corrected[leading]
    assert corrected[trailing] - corrected[leading] >= 5.0


def test_gate_lower_bound_does_not_move_authoritative_singleton():
    key = ("CTB", "792M", "inbound")
    gate = frozenset({("gate", 0)})
    passed_a = frozenset({("probe", 0), ("probe", 2)})
    passed_b = frozenset({("probe", 1), ("probe", 3)})
    corrected = _separate_common_stop_departures(
        [
            (key, 7.5, True, gate),
            (key, 19.77, False, passed_a),
            (key, 20.0, False, passed_b),
        ],
        {
            ("probe", 0): (22, 19.7),
            ("probe", 1): (22, 8.8),
            ("probe", 2): (28, 19.7),
            ("probe", 3): (28, 8.8),
        },
        {key: 29.0},
        {key: 16.001},
    )
    assert corrected[gate] == 7.5
    assert corrected[passed_a] > 16.0
    assert corrected[passed_b] > 16.0


def test_chain_constraints_use_directed_topological_order():
    key = ("GMB", "11", "seq-1")
    sources = [
        frozenset({("probe", 0)}),
        frozenset({("probe", 1), ("probe", 2)}),
        frozenset({("probe", 3)}),
    ]
    records = [(key, 10.0 + index, False, source) for index, source in enumerate(sources)]
    corrected = _separate_common_stop_departures(
        records,
        {
            ("probe", 0): (10, 10.0),
            ("probe", 1): (10, 5.0),
            ("probe", 2): (11, 100.0),
            ("probe", 3): (11, 0.0),
        },
        {key: 200.0},
    )
    # A > B > C despite B's very large raw observation at the second stop.
    assert corrected[sources[0]] > corrected[sources[1]] > corrected[sources[2]]


def test_lone_realtime_probe_row_is_retained():
    """A live one-stop observation remains useful even without corroboration."""
    line = _line()
    estimates = estimate_bus_positions(
        [Probe("KMB", "X", "outbound", 3, 2, kind=EtaKind.REALTIME)],
        [line],
    )
    assert len(estimates) == 1
    assert estimates[0].unreliable is False


def test_lone_scheduled_authoritative_gate_row_is_retained():
    """A direct gate departure is authoritative despite having one stop row."""
    line = _line()
    estimates = estimate_bus_positions(
        [],
        [line],
        authoritative_etas=[
            AuthoritativeProbe(
                "KMB", "X", "outbound", 3, 2, kind=EtaKind.SCHEDULED
            )
        ],
    )
    assert len(estimates) == 1
    assert estimates[0].unreliable is True


def test_realtime_ladder_is_reliable():
    line = _line()
    estimates = estimate_bus_positions(
        [
            Probe("KMB", "X", "outbound", 3, 4, kind=EtaKind.REALTIME),
            Probe("KMB", "X", "outbound", 4, 6, kind=EtaKind.REALTIME),
        ],
        [line],
    )
    assert len(estimates) == 1
    assert estimates[0].unreliable is False


def test_estimate_includes_route_direction_and_position_metadata():
    line = _line()
    estimates = estimate_bus_positions(
        [Probe("KMB", "X", "outbound", 3, 2)],
        [line],
    )
    assert len(estimates) == 1
    estimate = estimates[0]
    assert estimate.route == "X"
    assert estimate.bound == "outbound"
    assert estimate.position == 2.0
    assert estimate.source_observations == frozenset({("probe", 0)})


def test_authoritative_gate_eta_overrides_probe_at_same_stop():
    line = _line(stop_count=15)
    estimates = estimate_bus_positions(
        [Probe("KMB", "X", "outbound", 10, 12)],  # implied position 4
        [line],
        authoritative_etas=[AuthoritativeProbe("KMB", "X", "outbound", 10, 14)],
    )
    assert len(estimates) == 1
    assert estimates[0].position == 3.0


def test_multiple_authoritative_gate_departures_remain_distinct():
    line = _line(stop_count=15)
    estimates = estimate_bus_positions(
        [],
        [line],
        authoritative_etas=[
            AuthoritativeProbe("KMB", "X", "outbound", 10, 12),
            AuthoritativeProbe("KMB", "X", "outbound", 10, 14),
        ],
    )
    assert [estimate.position for estimate in estimates] == [3.0, 4.0]


def test_undeparted_gate_row_absorbs_later_scheduled_ladder():
    """A future HKUST journey must not appear from its downstream timetable."""
    line = _line("GMB", "11", "seq-1", stop_count=20)
    estimates = estimate_bus_positions(
        [
            Probe("GMB", "11", "seq-1", 12, 23.4, EtaKind.SCHEDULED),
            Probe("GMB", "11", "seq-1", 15, 29.0, EtaKind.SCHEDULED),
        ],
        [line],
        authoritative_etas=[
            AuthoritativeProbe(
                "GMB", "11", "seq-1", 6, 16, EtaKind.SCHEDULED
            )
        ],
    )
    assert estimates == []


def test_future_origin_vetoes_nonnegative_coarse_gate_position():
    """A downstream projection cannot launch a journey before stop-zero ETA."""
    line = _line("GMB", "11", "seq-1", stop_count=20)
    estimates = estimate_bus_positions(
        [
            Probe("GMB", "11", "seq-1", 0, 5, EtaKind.SCHEDULED),
            Probe("GMB", "11", "seq-1", 1, 7, EtaKind.SCHEDULED),
        ],
        [line],
        authoritative_etas=[
            # 6 - 11/2 = 0.5: the old coarse gate rule called this departed,
            # despite the same ETA instance still being five minutes from its
            # route-origin departure.
            AuthoritativeProbe(
                "GMB", "11", "seq-1", 6, 11, EtaKind.SCHEDULED
            )
        ],
    )

    assert estimates == []


def test_gate_anchor_reconciles_variable_downstream_travel_times():
    """One later-stop row per gate journey remains exactly three vehicles."""
    line = _line("GMB", "12", "seq-2", stop_count=23)
    estimates = estimate_bus_positions(
        [
            Probe("GMB", "12", "seq-2", 22, 8.3),
            Probe("GMB", "12", "seq-2", 22, 24.3),
            Probe("GMB", "12", "seq-2", 22, 36.1),
        ],
        [line],
        authoritative_etas=[
            AuthoritativeProbe("GMB", "12", "seq-2", 16, 5),
            AuthoritativeProbe("GMB", "12", "seq-2", 16, 17),
            AuthoritativeProbe("GMB", "12", "seq-2", 16, 28),
        ],
    )
    assert [estimate.position for estimate in estimates] == [2.0, 7.5, 13.5]
    assert all(
        {kind for kind, _index in estimate.source_observations}
        == {"gate", "probe"}
        for estimate in estimates
    )


def test_monotone_downstream_frontier_preserves_four_exact_tracks():
    line = _line("GMB", "12", "seq-2", stop_count=23)
    minutes_by_stop = {
        18: [1.29, 13.8],
        19: [1.9, 3.9, 15.94],
        20: [2.97, 5.09],
        21: [0.53, 3.64, 5.76],
        22: [5.51, 8.12, 10.92],
    }
    probes = [
        Probe("GMB", "12", "seq-2", stop, minutes)
        for stop, values in minutes_by_stop.items()
        for minutes in values
    ]
    estimates = estimate_bus_positions(
        probes,
        [line],
        authoritative_etas=[
            AuthoritativeProbe("GMB", "12", "seq-2", 16, 6)
        ],
    )
    groups = {
        frozenset(index for kind, index in estimate.source_observations if kind == "probe")
        for estimate in estimates
    }
    assert groups == {
        frozenset({1, 4}),
        frozenset({0, 3, 6, 9, 12}),
        frozenset({2, 5, 8, 11}),
        frozenset({7, 10}),
    }
    assert len(estimates) == 4
    leading = sorted(
        estimate.position
        for estimate in estimates
        if {0, 2} & {
            index
            for kind, index in estimate.source_observations
            if kind == "probe"
        }
    )
    assert len(leading) == 2
    assert leading[1] - leading[0] <= 2.0
    incoming = next(
        estimate for estimate in estimates
        if {1, 4} <= {index for kind, index in estimate.source_observations if kind == "probe"}
    )
    assert incoming.source_observations & {("gate", 0)}
    assert incoming.position == 13.0


def test_frontier_carries_gate_tracks_before_new_downstream_matching():
    line = _line("GMB", "12", "seq-2", stop_count=23)
    values = {
        13: [11.948, 12],
        14: [0.27, 12.46, 12.51],
        15: [1.27, 12.73, 13.37],
    }
    probes = [
        Probe("GMB", "12", "seq-2", stop, minutes)
        for stop, etas in values.items()
        for minutes in etas
    ]
    estimates = estimate_bus_positions(
        probes,
        [line],
        authoritative_etas=[
            AuthoritativeProbe("GMB", "12", "seq-2", 7, 1),
            AuthoritativeProbe("GMB", "12", "seq-2", 7, 1),
            AuthoritativeProbe("GMB", "12", "seq-2", 7, 21),
        ],
    )
    groups = {
        frozenset(index for kind, index in estimate.source_observations if kind == "probe")
        for estimate in estimates
    }
    assert groups == {
        frozenset({0, 3, 6}),
        frozenset({1, 4, 7}),
        frozenset({2, 5}),
    }
    assert len(estimates) == 3
    assert not any(
        estimate.source_observations == frozenset({("probe", 4)})
        for estimate in estimates
    )


def test_newer_downstream_track_supersedes_stale_realtime_singleton():
    line = _line("GMB", "11M", "seq-2", stop_count=12)
    probes = [
        Probe("GMB", "11M", "seq-2", 3, 2.21, EtaKind.SCHEDULED, 0),
        Probe("GMB", "11M", "seq-2", 4, 4.21, EtaKind.SCHEDULED, 0),
        Probe("GMB", "11M", "seq-2", 5, 6.21, EtaKind.SCHEDULED, 0),
        Probe("GMB", "11M", "seq-2", 4, 3.906, EtaKind.REALTIME, 30),
    ]
    gates = [
        AuthoritativeProbe(
            "GMB", "11M", "seq-2", 0, 100, EtaKind.SCHEDULED
        )
    ]

    plan = _plan_gate_associations(
        probes, gates, {("GMB", "11M", "seq-2")}
    )
    estimates = estimate_bus_positions(
        probes, [line], authoritative_etas=gates
    )

    assert plan.superseded_probe_inputs == {3}
    assert len(estimates) == 1
    assert estimates[0].source_observations == {
        ("probe", 0),
        ("probe", 1),
        ("probe", 2),
    }


def test_newer_upstream_track_supersedes_stale_realtime_singleton():
    line = _line("GMB", "11", "seq-1", stop_count=20)
    probes = [
        Probe("GMB", "11", "seq-1", 11, 0.661, cache_age_seconds=0),
        Probe("GMB", "11", "seq-1", 12, 1.009, cache_age_seconds=0),
        Probe("GMB", "11", "seq-1", 13, 2.110, cache_age_seconds=0),
        Probe("GMB", "11", "seq-1", 13, 5.457, cache_age_seconds=30),
    ]
    gates = [
        AuthoritativeProbe(
            "GMB", "11", "seq-1", 6, 100, EtaKind.SCHEDULED
        )
    ]

    plan = _plan_gate_associations(
        probes, gates, {("GMB", "11", "seq-1")}
    )
    estimates = estimate_bus_positions(
        probes, [line], authoritative_etas=gates
    )

    assert plan.superseded_probe_inputs == {3}
    assert len(estimates) == 1
    assert estimates[0].source_observations == {
        ("probe", 0),
        ("probe", 1),
        ("probe", 2),
    }


def test_newer_upstream_track_does_not_refute_vehicle_that_passed_stop():
    line = _line("GMB", "11", "seq-1", stop_count=20)
    probes = [
        Probe("GMB", "11", "seq-1", 11, 0.661, cache_age_seconds=0),
        Probe("GMB", "11", "seq-1", 12, 1.009, cache_age_seconds=0),
        Probe("GMB", "11", "seq-1", 13, 2.110, cache_age_seconds=0),
        Probe("GMB", "11", "seq-1", 13, 1.0, cache_age_seconds=30),
    ]
    gates = [
        AuthoritativeProbe(
            "GMB", "11", "seq-1", 6, 100, EtaKind.SCHEDULED
        )
    ]

    plan = _plan_gate_associations(
        probes, gates, {("GMB", "11", "seq-1")}
    )
    estimates = estimate_bus_positions(
        probes, [line], authoritative_etas=gates
    )

    assert not plan.superseded_probe_inputs
    assert len(estimates) == 2


def test_newer_realtime_singleton_cannot_refute_another_singleton():
    probes = [
        Probe("GMB", "11M", "seq-2", 11, 2, cache_age_seconds=0),
        # This intervening passed row prevents the two timing-compatible
        # singletons from being assumed to be one monotone track.
        Probe("GMB", "11M", "seq-2", 12, 22, cache_age_seconds=0),
        Probe("GMB", "11M", "seq-2", 13, 6, cache_age_seconds=30),
    ]
    gates = [
        AuthoritativeProbe(
            "GMB", "11M", "seq-2", 0, 100, EtaKind.SCHEDULED
        )
    ]

    plan = _plan_gate_associations(
        probes, gates, {("GMB", "11M", "seq-2")}
    )

    assert len(set(plan.passed_track_ids.values())) == 3
    assert not plan.superseded_probe_inputs


def test_equally_fresh_realtime_singleton_remains_a_distinct_vehicle():
    line = _line("GMB", "11M", "seq-2", stop_count=12)
    probes = [
        Probe("GMB", "11M", "seq-2", 3, 2.21, EtaKind.SCHEDULED, 0),
        Probe("GMB", "11M", "seq-2", 4, 4.21, EtaKind.SCHEDULED, 0),
        Probe("GMB", "11M", "seq-2", 5, 6.21, EtaKind.SCHEDULED, 0),
        Probe("GMB", "11M", "seq-2", 4, 3.906, EtaKind.REALTIME, 2),
    ]
    gates = [
        AuthoritativeProbe(
            "GMB", "11M", "seq-2", 0, 100, EtaKind.SCHEDULED
        )
    ]

    plan = _plan_gate_associations(
        probes, gates, {("GMB", "11M", "seq-2")}
    )
    estimates = estimate_bus_positions(
        probes, [line], authoritative_etas=gates
    )

    assert not plan.superseded_probe_inputs
    assert len(estimates) == 2


def test_stale_realtime_singleton_at_last_stop_is_not_superseded():
    line = _line("KMB", "91", "outbound", stop_count=31)
    probe = Probe(
        "KMB", "91", "outbound", 30, 1.414, EtaKind.REALTIME, 30
    )
    gates = [
        AuthoritativeProbe(
            "KMB", "91", "outbound", 15, 55, EtaKind.SCHEDULED
        )
    ]

    plan = _plan_gate_associations(
        [probe], gates, {("KMB", "91", "outbound")}
    )
    estimates = estimate_bus_positions(
        [probe], [line], authoritative_etas=gates
    )

    assert not plan.superseded_probe_inputs
    assert len(estimates) == 1
    assert estimates[0].source_observations == {("probe", 0)}


def test_headway_projection_keeps_leading_vehicle_at_exact_terminus():
    line = _line("CTB", "792M", "inbound", stop_count=30)
    # Official stop snapping and path arclength accumulation can differ by a
    # few floating-point ulps at the final stop.
    line.stop_offsets[-1] += 1e-8
    probes = [
        Probe("CTB", "792M", "inbound", 23, 1.858936050000342),
        Probe("CTB", "792M", "inbound", 28, 13.155646866667006),
        Probe("CTB", "792M", "inbound", 29, 0.5305527166668642),
        Probe("CTB", "792M", "inbound", 29, 15.197219383333529),
    ]

    estimates = estimate_bus_positions(probes, [line])

    assert [estimate.position for estimate in estimates] == [
        21.666666666666664,
        29.0,
    ]
    assert estimates[1].source_observations == {("probe", 2)}


def test_failed_frontier_propagation_leaves_gate_available_for_fresh_match():
    line = _line("GMB", "12", "seq-2", stop_count=20)
    probes = [
        Probe("GMB", "12", "seq-2", 13, 13),
        Probe("GMB", "12", "seq-2", 14, 25),
    ]
    gates = [AuthoritativeProbe("GMB", "12", "seq-2", 7, 1)]
    plan = _plan_gate_associations(
        probes, gates, {("GMB", "12", "seq-2")}
    )
    assert plan.gate_assignment[1] == 0
    estimates = estimate_bus_positions(probes, [line], authoritative_etas=gates)
    assert any(
        ("probe", 1) in estimate.source_observations
        and ("gate", 0) in estimate.source_observations
        for estimate in estimates
    )


def test_passed_downstream_vehicle_stays_separate_from_gate_journey():
    line = _line("GMB", "11", "seq-1", stop_count=20)
    estimates = estimate_bus_positions(
        [
            # The 9.2-minute row belongs to the HKUST arrival. The earlier
            # 4.16-minute row is a vehicle which has already passed HKUST.
            Probe("GMB", "11", "seq-1", 12, 9.2),
            Probe("GMB", "11", "seq-1", 12, 4.16),
        ],
        [line],
        authoritative_etas=[
            AuthoritativeProbe("GMB", "11", "seq-1", 6, 2),
        ],
    )
    assert [estimate.position for estimate in estimates] == [5.0, 9.92]


def test_raw_passed_position_wins_over_behind_gate_order_correction():
    """A coarse row already past HKUST must not be hidden by stale order data."""
    assert _passed_row_position(
        7.0,
        gate_index=6,
        probe_input_index=0,
        passed_probe_rows={0},
        passed_probe_positions={0: 4.0},
    ) == 7.0


def test_position_quantization_preserves_strict_gate_side():
    assert _quantize_position(6.0004, 6) == 6001
    assert _quantize_position(5.9996, 6) == 5999
    assert _quantize_position(6.0014, 6) == 6001


def test_unmatched_downstream_row_is_not_synthetic_passed_vehicle():
    line = _line("GMB", "11", "seq-1", stop_count=20)
    estimates = estimate_bus_positions(
        [Probe("GMB", "11", "seq-1", 12, 30)],
        [line],
        authoritative_etas=[
            AuthoritativeProbe("GMB", "11", "seq-1", 6, 2),
        ],
    )
    # The downstream ETA is later than the gate ETA and outside the matching
    # drift window; without ordered proof it is unresolved, not passed.
    assert [estimate.position for estimate in estimates] == [5.0]


def test_order_proven_passed_anchor_not_crossing_gate_is_hidden():
    """An order-proven row whose derived anchor is before HKUST is hidden."""
    line = _line("GMB", "11", "seq-1", stop_count=20)
    estimates = estimate_bus_positions(
        [Probe("GMB", "11", "seq-1", 12, 24), Probe("GMB", "11", "seq-1", 12, 50)],
        [line],
        authoritative_etas=[AuthoritativeProbe("GMB", "11", "seq-1", 6, 30)],
    )
    assert estimates == []


def test_two_order_proven_passed_anchors_keep_headway_separation():
    """Two order-proven passed rows retain multiplicity and ETA ordering."""
    line = _line("GMB", "11", "seq-1", stop_count=20)
    estimates = estimate_bus_positions(
        [
            Probe("GMB", "11", "seq-1", 12, 12),
            Probe("GMB", "11", "seq-1", 12, 22),
            Probe("GMB", "11", "seq-1", 12, 57),
        ],
        [line],
        authoritative_etas=[AuthoritativeProbe("GMB", "11", "seq-1", 6, 30)],
    )
    assert len(estimates) == 2
    first = next(e for e in estimates if ("probe", 0) in e.source_observations)
    second = next(e for e in estimates if ("probe", 1) in e.source_observations)
    assert first.position > 6
    assert second.position > 6
    assert first.position > second.position
    assert first.position - second.position >= 5.0


def test_passed_rows_across_checkpoints_form_one_ladder():
    line = _line("GMB", "11", "seq-1", stop_count=20)
    estimates = estimate_bus_positions(
        [
            Probe("GMB", "11", "seq-1", 12, 12),
            Probe("GMB", "11", "seq-1", 12, 57),
            Probe("GMB", "11", "seq-1", 15, 20),
            Probe("GMB", "11", "seq-1", 15, 60),
        ],
        [line],
        authoritative_etas=[AuthoritativeProbe("GMB", "11", "seq-1", 6, 30)],
    )
    passed = [
        estimate for estimate in estimates
        if ("probe", 0) in estimate.source_observations
        or ("probe", 2) in estimate.source_observations
    ]
    assert len(passed) == 1
    assert {("probe", 0), ("probe", 2)} <= passed[0].source_observations


def test_passed_track_survives_large_drift_and_empty_intervening_checkpoint():
    line = _line("GMB", "11", "seq-1", stop_count=20)
    estimates = estimate_bus_positions(
        [
            Probe("GMB", "11", "seq-1", 12, 12),
            Probe("GMB", "11", "seq-1", 12, 57),
            Probe("GMB", "11", "seq-1", 13, 44),
            Probe("GMB", "11", "seq-1", 15, 22),
            Probe("GMB", "11", "seq-1", 15, 60),
        ],
        [line],
        authoritative_etas=[AuthoritativeProbe("GMB", "11", "seq-1", 6, 30)],
    )
    assert len(estimates) == 1
    assert {("probe", 0), ("probe", 3)} <= estimates[0].source_observations


def test_two_cross_checkpoint_passed_tracks_do_not_proximity_merge():
    line = _line("GMB", "11", "seq-1", stop_count=20)
    estimates = estimate_bus_positions(
        [
            Probe("GMB", "11", "seq-1", 12, 2),
            Probe("GMB", "11", "seq-1", 12, 6),
            Probe("GMB", "11", "seq-1", 12, 57),
            Probe("GMB", "11", "seq-1", 15, 10),
            Probe("GMB", "11", "seq-1", 15, 14),
            Probe("GMB", "11", "seq-1", 15, 60),
        ],
        [line],
        authoritative_etas=[AuthoritativeProbe("GMB", "11", "seq-1", 6, 30)],
    )
    assert len(estimates) == 2
    first = next(e for e in estimates if ("probe", 0) in e.source_observations)
    second = next(e for e in estimates if ("probe", 1) in e.source_observations)
    assert {("probe", 0), ("probe", 3)} <= first.source_observations
    assert {("probe", 1), ("probe", 4)} <= second.source_observations


def test_raw_proven_passed_track_matches_across_empty_checkpoint():
    line = _line("GMB", "11", "seq-1", stop_count=20)
    estimates = estimate_bus_positions(
        [
            Probe("GMB", "11", "seq-1", 12, 2),
            Probe("GMB", "11", "seq-1", 13, 34),
            Probe("GMB", "11", "seq-1", 15, 10),
        ],
        [line],
        authoritative_etas=[AuthoritativeProbe("GMB", "11", "seq-1", 6, 20)],
    )
    assert len(estimates) == 1
    assert {("probe", 0), ("probe", 2)} <= estimates[0].source_observations


def test_origin_gate_future_does_not_absorb_vehicle_already_on_route():
    line = _line("GMB", "104", "seq-1", stop_count=20)
    estimates = estimate_bus_positions(
        [Probe("GMB", "104", "seq-1", 13, 7)],
        [line],
        authoritative_etas=[
            AuthoritativeProbe(
                "GMB", "104", "seq-1", 0, 4, EtaKind.SCHEDULED
            )
        ],
    )
    assert len(estimates) == 1
    assert estimates[0].position == 9.5
    assert estimates[0].source_observations == frozenset({("probe", 0)})


def test_live_later_row_is_passed_not_hidden_by_future_scheduled_gate():
    line = _line("GMB", "11", "seq-1", stop_count=20)
    estimates = estimate_bus_positions(
        [Probe("GMB", "11", "seq-1", 15, 12)],
        [line],
        authoritative_etas=[
            AuthoritativeProbe(
                "GMB", "11", "seq-1", 6, 14, EtaKind.SCHEDULED
            )
        ],
    )
    assert len(estimates) == 1
    assert estimates[0].position == 9.0
    assert estimates[0].source_observations == frozenset({("probe", 0)})


def test_later_stop_cannot_shift_to_gate_eta_three_minutes_later():
    gate_rows = [
        (0, AuthoritativeProbe("KMB", "X", "outbound", 12, 3)),
        (1, AuthoritativeProbe("KMB", "X", "outbound", 12, 19)),
        (2, AuthoritativeProbe("KMB", "X", "outbound", 12, 39)),
    ]
    probe_rows = [
        (10, Probe("KMB", "X", "outbound", 22, 3.5, EtaKind.SCHEDULED)),
        (11, Probe("KMB", "X", "outbound", 22, 20)),
        (12, Probe("KMB", "X", "outbound", 22, 36)),
    ]
    assert _align_gate_arrivals(
        gate_rows,
        probe_rows,
        gate_index=12,
        checkpoint=22,
    ) == [(11, 0), (12, 1)]


def test_small_negative_downstream_skew_keeps_passed_gate_journey_on_one_marker():
    line = _line(stop_count=15)
    estimates = estimate_bus_positions(
        [
            Probe("KMB", "X", "outbound", 10, 0.5),
            Probe("KMB", "X", "outbound", 10, 20.5),
        ],
        [line],
        authoritative_etas=[
            AuthoritativeProbe("KMB", "X", "outbound", 9, 2),
            AuthoritativeProbe("KMB", "X", "outbound", 9, 22),
        ],
    )
    assert len(estimates) == 1
    assert estimates[0].position == 9.75
    assert estimates[0].source_observations == frozenset(
        {("gate", 0), ("probe", 0)}
    )


def test_upstream_stop_accepts_fast_but_ordered_gate_journey():
    gate_rows = [
        (0, AuthoritativeProbe("GMB", "12", "seq-2", 16, 10)),
        (1, AuthoritativeProbe("GMB", "12", "seq-2", 16, 30)),
    ]
    probe_rows = [
        (10, Probe("GMB", "12", "seq-2", 4, 2)),
        (11, Probe("GMB", "12", "seq-2", 4, 20)),
    ]
    assert _align_gate_arrivals(
        gate_rows,
        probe_rows,
        gate_index=16,
        checkpoint=4,
    ) == [(10, 0), (11, 1)]


def test_origin_matched_citybus_future_track_does_not_render_downstream():
    line = _line("CTB", "792M", "outbound", stop_count=29)
    estimates = estimate_bus_positions(
        [
            Probe("CTB", "792M", "outbound", 0, 13.5),
            Probe("CTB", "792M", "outbound", 0, 43.5),
            Probe("CTB", "792M", "outbound", 14, 11.8),
            Probe("CTB", "792M", "outbound", 14, 39.2),
        ],
        [line],
        authoritative_etas=[
            AuthoritativeProbe("CTB", "792M", "outbound", 13, 10),
            AuthoritativeProbe("CTB", "792M", "outbound", 13, 38),
        ],
    )
    assert len(estimates) == 1
    assert estimates[0].position == 8.0


def test_same_stop_timetable_etAs_remain_separate_vehicles():
    """Three departures at one stop must not merge transitively into one ladder."""
    line = _line(stop_count=15)
    estimates = estimate_bus_positions(
        [
            Probe("KMB", "X", "outbound", 10, 2),
            Probe("KMB", "X", "outbound", 10, 6),
            Probe("KMB", "X", "outbound", 10, 10),
            # Repeated downstream rungs should reinforce the same three buses.
            Probe("KMB", "X", "outbound", 11, 4),
            Probe("KMB", "X", "outbound", 11, 8),
            Probe("KMB", "X", "outbound", 11, 12),
        ],
        [line],
    )
    assert len(estimates) == 3


def test_same_stop_close_departures_are_not_healed_together():
    line = _line(stop_count=15)
    estimates = estimate_bus_positions(
        [
            Probe("KMB", "X", "outbound", 10, 4),
            Probe("KMB", "X", "outbound", 10, 3),
        ],
        [line],
    )
    assert len(estimates) == 2


def test_anchor_cluster_prefers_earliest_realtime_in_positional_order():
    """A distant scheduled anchor must not pull a realtime vehicle upstream."""
    line = _line(stop_count=15)
    estimates = estimate_bus_positions(
        [
            Probe("KMB", "X", "outbound", 0, 0, kind=EtaKind.SCHEDULED),
            # Corroborate the scheduled departure at a second stop so this
            # fixture continues to exercise spatial-order clustering rather
            # than the sparse-timetable suppression rule.
            Probe("KMB", "X", "outbound", 1, 2, kind=EtaKind.SCHEDULED),
            Probe("KMB", "X", "outbound", 13, 0, kind=EtaKind.REALTIME),
        ],
        [line],
    )
    assert len(estimates) == 2
    assert any(not estimate.unreliable and estimate.lon > line.stops[12].lon for estimate in estimates)
    assert any(estimate.unreliable and estimate.lon == line.stops[0].lon for estimate in estimates)


def test_lone_realtime_row_renders_at_its_implied_position():
    line = _line()
    estimates = estimate_bus_positions([Probe("KMB", "X", "outbound", 3, 2)], [line])
    assert len(estimates) == 1
    assert line.stops[1].lon < estimates[0].lon < line.stops[3].lon


def test_fractional_probe_minute_produces_fractional_marker_position():
    line = _line()
    estimates = estimate_bus_positions(
        [Probe("KMB", "X", "outbound", 3, 2.5)], [line]
    )
    assert len(estimates) == 1
    assert estimates[0].position == 1.75


def test_estimate_without_geometry_is_dropped():
    line = RouteLine("X", "KMB", "outbound", _line().stops)
    assert (
        estimate_bus_positions([Probe("KMB", "X", "outbound", 3, 2)], [line])
        == []
    )


def test_11s_origin_route_gets_marker_from_downstream_probe():
    line = _line(operator="GMB", route="11S", bound="seq-1")
    estimates = estimate_bus_positions(
        [
            Probe("GMB", "11S", "seq-1", 2, 3),
            Probe("GMB", "11S", "seq-1", 3, 5),
        ],
        [line],
    )
    assert len(estimates) == 1
    assert estimates[0].label.startswith("11S")


def test_104_circular_label_switches_at_stop_twelve():
    gate = (22.333360, 114.262881)
    points = [gate]
    points.extend((gate[0], gate[1] - index * 0.001) for index in range(1, 24))
    stops = [
        Stop("G" if index in (0, 23) else str(index + 1), f"Stop {index + 1}", *point)
        for index, point in enumerate(points)
    ]
    offsets = [0.0]
    for _first, _second in zip(points, points[1:], strict=False):
        offsets.append(offsets[-1] + 111.32)
    line = RouteLine("104", "GMB", "seq-1", stops, points, offsets)
    # Two vehicles: one approaching stop 10 (Kwun Tong side), one past stop
    # 15 returning to HKUST — ladders offset beyond the merge gap.
    estimates = estimate_bus_positions(
        [
            Probe("GMB", "104", "seq-1", 10, 4),
            Probe("GMB", "104", "seq-1", 11, 6),
            Probe("GMB", "104", "seq-1", 15, 3),
            Probe("GMB", "104", "seq-1", 16, 5),
        ],
        [line],
    )
    labels = sorted(estimate.label for estimate in estimates)
    assert labels == ["104 HKUST", "104 Kwun Tong"]


def test_destination_map_overrides_terminus_name():
    line = _line()
    estimates = estimate_bus_positions(
        [Probe("KMB", "X", "outbound", 3, 2)],
        [line],
        {("KMB", "X", "outbound"): "Diamond Hill"},
    )
    assert estimates[0].label == "X Diamond Hill"


def test_heading_follows_travel_direction():
    line = _line()
    estimates = estimate_bus_positions([Probe("KMB", "X", "outbound", 3, 2)], [line])
    # path runs east (+lon), so heading is atan2(dlat, dlon) ~ 0
    assert abs(estimates[0].heading) < 0.5


def test_all_stop_boundary_controls_proportion_and_ignores_unrelated_cache_age():
    line = _line(stop_count=7)
    rows = [
        Probe("KMB", "X", "outbound", 2, None, cache_age_seconds=1),
        Probe("KMB", "X", "outbound", 3, 1, cache_age_seconds=1),
        # A stale downstream rung may help identify the same ETA ladder, but
        # it is not one of the two physical boundary observations.
        Probe("KMB", "X", "outbound", 4, 3, cache_age_seconds=120),
        # A coincident index on another route must not contaminate freshness.
        Probe("KMB", "OTHER", "outbound", 2, None, cache_age_seconds=300),
    ]
    estimates = estimate_bus_positions(
        rows,
        [line],
        observed_checkpoint_indices={("KMB", "X", "outbound"): range(7)},
    )
    assert len(estimates) == 1
    assert estimates[0].bracket == (2.0, 3.0)
    assert estimates[0].position == 2.5
    assert estimates[0].eta_minutes == 1
    assert estimates[0].boundary_age_seconds == 1


def test_consecutive_empty_stops_then_downstream_observation_forms_one_boundary_marker():
    line = _line(stop_count=7)
    rows = [
        # Several consecutive checkpoints see zero instances of this vehicle.
        Probe("KMB", "X", "outbound", 1, None, cache_age_seconds=0),
        Probe("KMB", "X", "outbound", 2, None, cache_age_seconds=0),
        Probe("KMB", "X", "outbound", 3, None, cache_age_seconds=0),
        # Stop 4 first sees it; stop 5 corroborates the same ETA ladder.
        Probe("KMB", "X", "outbound", 4, 1, cache_age_seconds=0),
        Probe("KMB", "X", "outbound", 5, 3, cache_age_seconds=0),
    ]

    estimates = estimate_bus_positions(
        rows,
        [line],
        observed_checkpoint_indices={("KMB", "X", "outbound"): range(7)},
    )

    assert len(estimates) == 1
    assert estimates[0].source_indices == frozenset({4, 5})
    assert estimates[0].bracket == (3.0, 4.0)
    assert estimates[0].eta_minutes == 1
    assert estimates[0].position == 3.5


def test_staggered_immutable_minutes_are_normalized_only_for_identity_matching():
    line = _line(stop_count=7)
    rows = [
        Probe("KMB", "X", "outbound", 2, None, cache_age_seconds=0),
        # As fetched six minutes ago this source implied position -1.  On the
        # common identity clock it aligns with the fresh downstream rung at 2.
        Probe("KMB", "X", "outbound", 3, 8, cache_age_seconds=360),
        Probe("KMB", "X", "outbound", 4, 4, cache_age_seconds=0),
    ]
    estimates = estimate_bus_positions(
        rows,
        [line],
        observed_checkpoint_indices={("KMB", "X", "outbound"): range(7)},
    )
    assert len(estimates) == 1
    assert estimates[0].source_indices == frozenset({3, 4})
    assert estimates[0].bracket == (2.0, 3.0)
    # The stale source value itself is unchanged; its age was not converted
    # into displayed motion.
    assert estimates[0].eta_minutes == 8
    assert estimates[0].position == 2.0


def test_partial_observation_without_upstream_absence_has_no_bracket():
    line = _line(stop_count=7)
    estimates = estimate_bus_positions(
        [Probe("KMB", "X", "outbound", 3, 1, cache_age_seconds=0)],
        [line],
        observed_checkpoint_indices={("KMB", "X", "outbound"): {3, 4, 5, 6}},
    )
    assert len(estimates) == 1
    assert estimates[0].bracket is None
    assert estimates[0].boundary_age_seconds is None


def test_probe_selection_uses_bounded_evenly_spaced_anchors():
    from dashboard.providers.route_geometry import select_probe_stops

    line = _line()
    probes = select_probe_stops([line])
    # Every stop of the direction is probed, termini included: the route is
    # just a stop sequence with ETAs — there is no interior/exterior split.
    assert len(probes) == 6
    assert probes[0].index == 0 and probes[-1].index == 5


def test_probe_selection_downsamples_long_routes_and_keeps_mandatory_stop():
    from dashboard.providers.route_geometry import select_probe_stops

    stops = [Stop(str(index), f"Stop {index}", 22.33, 114.26 + index * 0.001) for index in range(31)]
    line = RouteLine("X", "KMB", "outbound", stops)
    probes = select_probe_stops(
        [line], mandatory_stop_ids={"17"}, max_anchors=5
    )
    assert len(probes) == 5
    assert [probe.index for probe in probes] == [0, 10, 17, 20, 30]


def test_probe_selection_mandatory_overflow_keeps_all_occurrences():
    from dashboard.providers.route_geometry import select_probe_stops

    stops = [Stop(str(index), f"Stop {index}", 22.33, 114.26) for index in range(8)]
    probes = select_probe_stops(
        [RouteLine("arbitrary", "KMB", "outbound", stops)],
        mandatory_stop_ids={"1", "3", "5", "6"},
        max_anchors=3,
    )
    assert [probe.index for probe in probes] == [0, 1, 3, 5, 6, 7]


def test_probe_selection_preserves_circular_occurrence_order():
    from dashboard.providers.route_geometry import select_probe_stops

    stops = [Stop("same", "Loop", 22.33, 114.26) for _ in range(6)]
    probes = select_probe_stops([RouteLine("loop", "GMB", "seq-1", stops)], max_anchors=4)
    assert [probe.index for probe in probes] == [0, 2, 3, 5]


def test_fetch_groups_dedupe_shared_physical_stops():
    from dashboard.providers.route_geometry import ProbeStop
    from dashboard.providers.transit import _fetch_group_key

    south_a = ProbeStop("GMB", "11", "seq-1", "20013011", 2004791, 1, 3)
    south_b = ProbeStop("GMB", "11S", "seq-1", "20013011", 2004826, 1, 7)
    north = ProbeStop("GMB", "11", "seq-2", "20012474", 2004791, 2, 5)
    ctb_a = ProbeStop("CTB", "792M", "outbound", "003130", 1616, 1, 2)
    ctb_b = ProbeStop("CTB", "792M", "inbound", "003130", 1616, 2, 2)

    groups = {_fetch_group_key(p) for p in (south_a, south_b, north, ctb_a, ctb_b)}
    # Same physical GMB stop -> one fetch; different stop -> another; CTB
    # needs one fetch per (stop, route) direction pair.
    assert len(groups) == 3
    assert _fetch_group_key(south_a) == _fetch_group_key(south_b)
