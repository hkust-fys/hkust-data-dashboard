"""Estimated bus position tests: ladder-collapsed vehicle reconstruction."""

from dashboard.maps.positions import (
    BusEstimate,
    _path_segment_length,
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
