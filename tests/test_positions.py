"""Estimated bus position tests: ladder-collapsed vehicle reconstruction."""

from dashboard.maps.positions import (
    BusEstimate,
    _align_gate_arrivals,
    _path_segment_length,
    _separate_common_stop_departures,
    estimate_bus_positions,
)
from dashboard.models import EtaKind
from dashboard.providers.route_geometry import RouteLine, Stop


class Probe:
    """Minimal probe-ETA stand-in (duck-typed like the provider's ProbeEta)."""

    def __init__(self, operator, route, bound, index, minutes, kind=None):
        self.operator = operator
        self.route = route
        self.bound = bound
        self.index = index
        self.minutes = minutes
        self.stop_id = "S"
        self.kind = kind or EtaKind.REALTIME


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


def test_probe_selection_covers_every_stop_including_termini():
    from dashboard.providers.route_geometry import select_probe_stops

    line = _line()
    probes = select_probe_stops([line])
    # Every stop of the direction is probed, termini included: the route is
    # just a stop sequence with ETAs — there is no interior/exterior split.
    assert sorted(probe.index for probe in probes) == [0, 1, 2, 3, 4, 5]


def test_probe_selection_does_not_downsample_long_routes():
    from dashboard.providers.route_geometry import select_probe_stops

    stops = [Stop(str(index), f"Stop {index}", 22.33, 114.26 + index * 0.001) for index in range(31)]
    line = RouteLine("X", "KMB", "outbound", stops)
    probes = select_probe_stops([line])
    assert [probe.index for probe in probes] == list(range(31))


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
