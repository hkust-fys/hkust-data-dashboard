"""Estimated bus position tests: ladder-collapsed vehicle reconstruction."""

import math
from collections import Counter
from copy import copy
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from dashboard.maps import positions as positions_module
from dashboard.maps.marker_audit import audit_marker_positions
from dashboard.maps.positions import (
    BusEstimate,
    _align_gate_arrivals,
    _atomic_kmb_frontier_certificate,
    _eta_guided_priority_indices,
    _passed_row_position,
    _path_segment_length,
    _plan_gate_associations,
    _quantize_position,
    _separate_common_stop_departures,
    classify_checkpoint,
    estimate_bus_positions,
    rebuild_estimate_from_probe_fragments,
    rebuild_estimate_from_probe_sources,
)
from dashboard.models import EtaKind, Operator
from dashboard.providers.route_geometry import RouteLine, Stop
from dashboard.providers.transit import ProbeEtaSnapshot, ProbeRouteGeneration


def test_checkpoint_classifier_three_state_matrix():
    arrival = datetime(2026, 1, 1, tzinfo=UTC)
    downstream = SimpleNamespace(
        arrival_at=arrival, minutes=2, kind=EtaKind.REALTIME,
        cache_age_seconds=2, refresh_generation=8,
    )
    empty = SimpleNamespace(index=21, minutes=None, kind=EtaKind.REALTIME,
                            cache_age_seconds=2, refresh_generation=7)
    assert classify_checkpoint(21, [empty], {21}, [downstream]) == "present"
    assert classify_checkpoint(21, [empty], (), [downstream]) == "certified absent"
    unavailable = SimpleNamespace(**{**empty.__dict__, "kind": EtaKind.UNAVAILABLE})
    assert classify_checkpoint(21, [unavailable], (), [downstream]) == "unknown"
    later = SimpleNamespace(index=21, minutes=4, kind=EtaKind.REALTIME,
                            arrival_at=arrival + timedelta(minutes=5),
                            cache_age_seconds=2, refresh_generation=7)
    assert classify_checkpoint(21, [later], (), [downstream]) == "certified absent"
    saturated = [SimpleNamespace(**{**later.__dict__, "refresh_generation": i})
                 for i in (7, 8, 9)]
    assert classify_checkpoint(21, saturated, (), [downstream]) == "unknown"
    assert classify_checkpoint(21, [SimpleNamespace(**{**later.__dict__, "cache_age_seconds": None})], (), [downstream]) == "unknown"
    assert classify_checkpoint(21, [SimpleNamespace(**{**later.__dict__, "refresh_generation": 0})], (), [downstream]) == "unknown"
    assert classify_checkpoint(21, [SimpleNamespace(**{**later.__dict__, "cache_age_seconds": True})], (), [downstream]) == "unknown"
    assert classify_checkpoint(21, [SimpleNamespace(**{**later.__dict__, "refresh_generation": True})], (), [downstream]) == "unknown"
    assert classify_checkpoint(21, [SimpleNamespace(**{**later.__dict__, "cache_age_seconds": 60})], (), [downstream]) == "unknown"
    assert classify_checkpoint(21, [later], (), [SimpleNamespace(**{**downstream.__dict__, "cache_age_seconds": 60})]) == "unknown"
    tied = SimpleNamespace(**{**later.__dict__, "arrival_at": arrival + timedelta(seconds=180)})
    assert classify_checkpoint(21, [tied], (), [downstream]) == "unknown"
    assert classify_checkpoint(21, [SimpleNamespace(**{**later.__dict__, "arrival_at": arrival + timedelta(seconds=179)})], (), [downstream]) == "unknown"
    mixed = [later, SimpleNamespace(**{**later.__dict__, "minutes": None})]
    assert classify_checkpoint(21, mixed, (), [downstream]) == "unknown"
    later_two = SimpleNamespace(**{**later.__dict__, "arrival_at": arrival + timedelta(minutes=6),
                                   "refresh_generation": 7})
    assert classify_checkpoint(21, [later, later_two], (), [downstream]) == "certified absent"
    assert classify_checkpoint(21, [later_two, later], (), [downstream]) == "certified absent"
    mixed_revision = SimpleNamespace(**{**later.__dict__, "refresh_generation": 8})
    assert classify_checkpoint(21, [later, mixed_revision], (), [downstream]) == "unknown"
    for bad_age in (float("nan"), float("inf")):
        bad_downstream = SimpleNamespace(**{**downstream.__dict__, "cache_age_seconds": bad_age})
        assert classify_checkpoint(21, [empty], (), [bad_downstream]) == "unknown"


@pytest.mark.parametrize("kinds", [
    (EtaKind.SCHEDULED,),
    (EtaKind.SCHEDULED, EtaKind.SCHEDULED),
    (EtaKind.REALTIME, EtaKind.SCHEDULED),
    (EtaKind.SCHEDULED, EtaKind.REALTIME),
    (EtaKind.REALTIME, EtaKind.REALTIME, EtaKind.SCHEDULED),
])
def test_scheduled_lower_response_cannot_certify_checkpoint_absence(kinds):
    arrival = datetime(2026, 1, 1, tzinfo=UTC)
    downstream = Probe("GMB", "11S", "seq-1", 5, 2,
                       cache_age_seconds=0, refresh_generation=17, arrival_at=arrival)
    rows = [Probe("GMB", "11S", "seq-1", 4, 20 + index, kind=kind,
                  cache_age_seconds=0, refresh_generation=17,
                  arrival_at=arrival + timedelta(minutes=20 + index))
            for index, kind in enumerate(kinds)]
    assert classify_checkpoint(4, rows, (), [downstream]) == "unknown"


@pytest.mark.parametrize("kind", [EtaKind.REALTIME, EtaKind.MOVING_SLOWLY, EtaKind.DELAYED])
@pytest.mark.parametrize("count", [1, 2, 3])
def test_live_lower_response_certifies_only_unsaturated_checkpoint(kind, count):
    arrival = datetime(2026, 1, 1, tzinfo=UTC)
    downstream = Probe("GMB", "11S", "seq-1", 5, 2,
                       cache_age_seconds=0, refresh_generation=17, arrival_at=arrival)
    rows = [Probe("GMB", "11S", "seq-1", 4, 20 + index, kind=kind,
                  cache_age_seconds=0, refresh_generation=17,
                  arrival_at=arrival + timedelta(minutes=20 + index))
            for index in range(count)]
    assert classify_checkpoint(4, rows, (), [downstream]) == (
        "certified absent" if count < 3 else "unknown"
    )


@pytest.mark.parametrize("nonempty", [False, True])
@pytest.mark.parametrize("downstream_kind", [EtaKind.REALTIME, EtaKind.SCHEDULED])
def test_live_absence_certificate_preserves_empty_and_downstream_kind_rules(
    nonempty, downstream_kind,
):
    arrival = datetime(2026, 1, 1, tzinfo=UTC)
    downstream = Probe("GMB", "11S", "seq-1", 5, 2, kind=downstream_kind,
                       cache_age_seconds=0, refresh_generation=17, arrival_at=arrival)
    lower = Probe("GMB", "11S", "seq-1", 4, 20 if nonempty else None,
                  cache_age_seconds=0, refresh_generation=17,
                  arrival_at=arrival + timedelta(minutes=20) if nonempty else None)
    assert classify_checkpoint(4, [lower], (), [downstream]) == "certified absent"


@pytest.mark.parametrize(("seconds", "lower_age", "downstream_age", "expected"), [
    (180, 0, 0, "unknown"),
    (180.001, 0, 0, "certified absent"),
    (240, 5, 0, "certified absent"),
    (240, 5.001, 0, "unknown"),
    (240, 59.999, 55, "certified absent"),
    (240, 60, 55, "unknown"),
])
def test_live_absence_certificate_keeps_strict_time_and_freshness_bounds(
    seconds, lower_age, downstream_age, expected,
):
    arrival = datetime(2026, 1, 1, tzinfo=UTC)
    downstream = Probe("GMB", "11S", "seq-1", 5, 2,
                       cache_age_seconds=downstream_age, refresh_generation=17,
                       arrival_at=arrival)
    lower = Probe("GMB", "11S", "seq-1", 4, 20,
                  cache_age_seconds=lower_age, refresh_generation=17,
                  arrival_at=arrival + timedelta(seconds=seconds))
    assert classify_checkpoint(4, [lower], (), [downstream]) == expected


@pytest.mark.parametrize("mixed", [False, True])
def test_scheduled_omission_keeps_estimator_and_rebuild_frontiers_unknown(mixed):
    key = ("GMB", "11S", "seq-1")
    line = _line(*key, stop_count=7)
    arrival = datetime(2026, 1, 1, tzinfo=UTC)
    rows = [Probe(*key, 4, 20, kind=EtaKind.SCHEDULED,
                  cache_age_seconds=0, refresh_generation=17,
                  arrival_at=arrival + timedelta(minutes=20))]
    if mixed:
        rows.append(Probe(*key, 4, 21, kind=EtaKind.REALTIME,
                          cache_age_seconds=0, refresh_generation=17,
                          arrival_at=arrival + timedelta(minutes=21)))
    slot = len(rows)
    rows.append(Probe(*key, 5, 0.5, cache_age_seconds=0, refresh_generation=17,
                      arrival_at=arrival + timedelta(minutes=0.5)))
    estimates = estimate_bus_positions(
        rows, [line], observed_checkpoint_indices={key: {4, 5}},
    )
    live = next(estimate for estimate in estimates
                if ("probe", slot) in estimate.source_observations)
    assert live.bracket is None
    assert live.position_authoritative is False
    assert live.boundary_revision is None
    assert live.source_indices == frozenset({5})
    assert live.source_observations == frozenset({("probe", slot)})
    assert live.checkpoint_evidence == ((5, rows[slot].arrival_at.timestamp(), 17),)
    assert rebuild_estimate_from_probe_fragments(live, [], rows, [line]) is live
    rebuilt = rebuild_estimate_from_probe_sources(live, [slot], rows, [line])
    assert rebuilt is not None
    assert rebuilt.bracket is None
    assert rebuilt.position_authoritative is False
    assert rebuilt.checkpoint_evidence == live.checkpoint_evidence
    assert rebuilt.source_observations == live.source_observations


def test_rebuild_probe_sources_clears_stale_position_metadata_on_cold_fallback():
    line = _line(stop_count=10)
    row = Probe("KMB", "X", "outbound", 8, 2, cache_age_seconds=0,
                refresh_generation=9,
                arrival_at=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=16))
    template = BusEstimate(
        "X destination", 22.3, 114.2, Operator.KMB, 0.0,
        route="X", bound="outbound", operator_code="KMB", position=1.5,
        bracket=(1.0, 2.0), eta_minutes=1, bracket_initial_eta=1,
        bracket_eta_offsets=(-1, 1), boundary_age_seconds=4,
        boundary_revision=(3, 4), priority_indices=frozenset({1, 2}),
    )
    rebuilt = rebuild_estimate_from_probe_sources(template, (0,), [row], [line])
    assert rebuilt is not None
    assert rebuilt.position == pytest.approx(7)
    assert rebuilt.bracket is None
    assert rebuilt.eta_minutes is None
    assert rebuilt.boundary_revision is None
    assert rebuilt.priority_indices == frozenset({8})
    assert rebuilt.exploratory_indices == frozenset({7})


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
        signed_minutes=None,
        refresh_generation=0,
        arrival_at=None,
    ):
        self.operator = operator
        self.route = route
        self.bound = bound
        self.index = index
        self.minutes = minutes
        self.stop_id = "S"
        self.kind = kind or EtaKind.REALTIME
        self.cache_age_seconds = cache_age_seconds
        self.signed_minutes = signed_minutes
        self.refresh_generation = refresh_generation
        self.arrival_at = arrival_at


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


def _atomic_gate_frontier_fixture(frame, *, reverse=False):
    key = ("KMB", "91", "inbound")
    base = datetime(2026, 1, 1, tzinfo=UTC)
    if frame == 25:
        clock = 54.94625
        gate_minutes = (10, 27)
        groups = [
            ((0, 0), (3, 542), (4, 768), (6, 990), (7, 1111), (8, 1230), (16, 1787),
             (20, 2139), (21, 2201), (22, 2268), (29, 2543), (30, 2561), (31, 2612),
             (32, 2653), (33, 2757)),
            ((6, -18), (7, 104), (8, 224), (16, 786), (20, 1139), (21, 1199),
             (22, 1264), (29, 1527), (30, 1545), (31, 1588), (32, 1629), (33, 1732)),
            ((20, -29), (21, 30), (22, 98), (29, 373), (30, 391), (31, 441),
             (32, 482), (33, 586)),
        ]
        positions = (0.30413053505535054, 6.597920081967214, 21.36685661764706)
        brackets = ((0, 3), (6, 7), (21, 22))
    else:
        clock = 84.906763
        gate_minutes = (9, 25)
        groups = [
            ((1, 22), (2, 178), (4, 662), (5, 793), (7, 1015), (11, 1285),
             (16, 1696), (22, 2160), (24, 2251), (25, 2277), (26, 2322),
             (31, 2484), (32, 2526), (33, 2629)),
            ((7, 60), (11, 327), (16, 743), (22, 1222), (24, 1312), (25, 1338),
             (26, 1383), (31, 1546), (32, 1588), (33, 1690)),
            ((22, -18), (24, 60), (25, 86), (26, 145), (31, 312), (32, 353), (33, 457)),
        ]
        positions = (1.4032484807692307, 7.373135026217229, 24.95795242307692)
        brackets = ((1, 2), (7, 11), (24, 25))
    probes = []
    expected_rows = []
    group_kinds = (
        (EtaKind.SCHEDULED, EtaKind.REALTIME, EtaKind.REALTIME)
        if frame == 25
        else (EtaKind.REALTIME,) * len(groups)
    )
    for group, kind in zip(groups, group_kinds, strict=True):
        owned = []
        for index, seconds in group:
            signed = (seconds - clock) / 60
            row = Probe(*key, index, max(0, signed), kind=kind, signed_minutes=signed,
                        cache_age_seconds=9.2, refresh_generation=frame,
                        arrival_at=base + timedelta(seconds=seconds))
            owned.append(row)
            probes.append(row)
        expected_rows.append(owned)
    # Successful-empty rows remain full snapshot boundary context, while the
    # 37/33 active source tokens include only live observations and gate roots.
    for index in (0, 15):
        if not any(row.index == index for row in probes):
            probes.append(Probe(*key, index, None, cache_age_seconds=9.2,
                                refresh_generation=frame))
    gates = [Probe(*key, 15, minutes) for minutes in gate_minutes]
    if reverse:
        probes.reverse()
        gates.reverse()
    when = base + timedelta(seconds=clock + 9.2)
    snapshot = ProbeEtaSnapshot((ProbeRouteGeneration(key, tuple(probes), frame, when),), when)
    return snapshot, gates, expected_rows, positions, brackets


def _kmb_91_sparse_gate_partition_fixture(frame, *, reverse=False):
    """Exact active rows from live frames 12/16 of the 2026-09-09 audit."""
    key = ("KMB", "91", "inbound")
    if frame == 12:
        revision = 129
        age = 9.843999999997322
        gate_specs = (
            (4.0, EtaKind.REALTIME),
            (26.0, EtaKind.SCHEDULED),
            (54.0, EtaKind.SCHEDULED),
        )
        row_specs = (
            (1, 0.0, -0.26748093333333334, EtaKind.SCHEDULED,
             "2026-09-09T17:47:13+08:00"),
            (2, 2.3825190666666667, 2.3825190666666667, EtaKind.SCHEDULED,
             "2026-09-09T17:49:52+08:00"),
            (3, 6.349185733333333, 6.349185733333333, EtaKind.SCHEDULED,
             "2026-09-09T17:53:50+08:00"),
            (4, 10.4658524, 10.4658524, EtaKind.SCHEDULED,
             "2026-09-09T17:57:57+08:00"),
            (9, 19.149185733333336, 19.149185733333336, EtaKind.SCHEDULED,
             "2026-09-09T18:06:38+08:00"),
            (10, 19.632519066666667, 19.632519066666667, EtaKind.SCHEDULED,
             "2026-09-09T18:07:07+08:00"),
            (11, 0.0, -1.1674809333333334, EtaKind.REALTIME,
             "2026-09-09T17:46:19+08:00"),
            (11, 20.98251906666667, 20.98251906666667, EtaKind.SCHEDULED,
             "2026-09-09T18:08:28+08:00"),
            (12, 0.0, -0.30081426666666666, EtaKind.REALTIME,
             "2026-09-09T17:47:11+08:00"),
            (12, 21.83251906666667, 21.83251906666667, EtaKind.SCHEDULED,
             "2026-09-09T18:09:19+08:00"),
            (13, 1.2991857333333334, 1.2991857333333334, EtaKind.REALTIME,
             "2026-09-09T17:48:47+08:00"),
            (13, 23.432519066666668, 23.432519066666668, EtaKind.SCHEDULED,
             "2026-09-09T18:10:55+08:00"),
            (16, 5.4658524, 5.4658524, EtaKind.REALTIME,
             "2026-09-09T17:52:57+08:00"),
            (16, 27.58251906666667, 27.58251906666667, EtaKind.SCHEDULED,
             "2026-09-09T18:15:04+08:00"),
            (31, 0.0, -0.9341476, EtaKind.REALTIME,
             "2026-09-09T17:46:33+08:00"),
            (31, 20.58251906666667, 20.58251906666667, EtaKind.REALTIME,
             "2026-09-09T18:08:04+08:00"),
            (31, 42.23251906666667, 42.23251906666667, EtaKind.SCHEDULED,
             "2026-09-09T18:29:43+08:00"),
            (32, 0.0, -0.2508142666666667, EtaKind.REALTIME,
             "2026-09-09T17:47:14+08:00"),
            (32, 21.265852400000004, 21.265852400000004, EtaKind.REALTIME,
             "2026-09-09T18:08:45+08:00"),
            (32, 42.89918573333333, 42.89918573333333, EtaKind.SCHEDULED,
             "2026-09-09T18:30:23+08:00"),
            (33, 1.3325190666666666, 1.3325190666666666, EtaKind.REALTIME,
             "2026-09-09T17:48:49+08:00"),
            (33, 22.8158524, 22.8158524, EtaKind.REALTIME,
             "2026-09-09T18:10:18+08:00"),
            (33, 44.48251906666667, 44.48251906666667, EtaKind.SCHEDULED,
             "2026-09-09T18:31:58+08:00"),
        )
        expected = (1.1009362012578616, 12.188008916666666, 32.15840901052631)
    elif frame == 16:
        revision = 177
        age = 9.26600000000326
        gate_specs = (
            (3.0, EtaKind.REALTIME),
            (25.0, EtaKind.REALTIME),
            (53.0, EtaKind.SCHEDULED),
        )
        row_specs = (
            (1, 0.0, -0.9338833833333333, EtaKind.REALTIME,
             "2026-09-09T17:47:13+08:00"),
            (2, 1.7161166166666668, 1.7161166166666668, EtaKind.REALTIME,
             "2026-09-09T17:49:52+08:00"),
            (11, 20.316116616666665, 20.316116616666665, EtaKind.REALTIME,
             "2026-09-09T18:08:28+08:00"),
            (12, 0.0, -0.95055005, EtaKind.REALTIME,
             "2026-09-09T17:47:12+08:00"),
            (12, 21.166116616666667, 21.166116616666667, EtaKind.REALTIME,
             "2026-09-09T18:09:19+08:00"),
            (13, 0.6327832833333333, 0.6327832833333333, EtaKind.REALTIME,
             "2026-09-09T17:48:47+08:00"),
            (13, 22.766116616666668, 22.766116616666668, EtaKind.REALTIME,
             "2026-09-09T18:10:55+08:00"),
            (16, 4.79944995, 4.79944995, EtaKind.REALTIME,
             "2026-09-09T17:52:57+08:00"),
            (16, 26.916116616666667, 26.916116616666667, EtaKind.REALTIME,
             "2026-09-09T18:15:04+08:00"),
            (31, 19.832783283333335, 19.832783283333335, EtaKind.REALTIME,
             "2026-09-09T18:07:59+08:00"),
            (31, 41.56611661666667, 41.56611661666667, EtaKind.REALTIME,
             "2026-09-09T18:29:43+08:00"),
            (32, 0.0, -1.0838833833333332, EtaKind.REALTIME,
             "2026-09-09T17:47:04+08:00"),
            (32, 20.516116616666668, 20.516116616666668, EtaKind.REALTIME,
             "2026-09-09T18:08:40+08:00"),
            (32, 42.23278328333333, 42.23278328333333, EtaKind.REALTIME,
             "2026-09-09T18:30:23+08:00"),
            (33, 0.49944995, 0.49944995, EtaKind.REALTIME,
             "2026-09-09T17:48:39+08:00"),
            (33, 22.066116616666665, 22.066116616666665, EtaKind.REALTIME,
             "2026-09-09T18:10:13+08:00"),
            (33, 43.81611661666667, 43.81611661666667, EtaKind.REALTIME,
             "2026-09-09T18:31:58+08:00"),
        )
        expected = (1.352408823899371, 12.6003474, 32.68455792631579)
    else:
        raise AssertionError(frame)
    probes = [
        Probe(
            *key,
            index,
            minutes,
            kind=kind,
            cache_age_seconds=age,
            signed_minutes=signed,
            refresh_generation=revision,
            arrival_at=datetime.fromisoformat(arrival),
        )
        for index, minutes, signed, kind, arrival in row_specs
    ]
    gates = [
        AuthoritativeProbe(*key, 15, minutes, kind)
        for minutes, kind in gate_specs
    ]
    if reverse:
        probes.reverse()
        gates.reverse()
    return probes, gates, expected


@pytest.mark.parametrize("frame", [12, 16])
@pytest.mark.parametrize("reverse", [False, True])
def test_sparse_atomic_kmb_gate_partition_does_not_duplicate_one_bus(frame, reverse):
    probes, gates, expected = _kmb_91_sparse_gate_partition_fixture(
        frame, reverse=reverse,
    )
    estimates = estimate_bus_positions(
        probes,
        [_line("KMB", "91", "inbound", stop_count=34)],
        authoritative_etas=gates,
        observed_checkpoint_indices={("KMB", "91", "inbound"): range(34)},
    )

    assert [estimate.position for estimate in estimates] == pytest.approx(expected)
    probe_sources = [
        source
        for estimate in estimates
        for source in estimate.source_observations
        if source[0] == "probe"
    ]
    assert Counter(probe_sources) == Counter(
        ("probe", index) for index in range(len(probes))
    )
    assert len(estimates) == 3
    upstream, middle, front = estimates
    revision = {12: 129, 16: 177}[frame]
    assert (16, datetime.fromisoformat("2026-09-09T18:15:04+08:00").timestamp(), revision) \
        in upstream.checkpoint_evidence
    assert (31, datetime.fromisoformat("2026-09-09T18:29:43+08:00").timestamp(), revision) \
        in upstream.checkpoint_evidence
    assert (16, datetime.fromisoformat("2026-09-09T17:52:57+08:00").timestamp(), revision) \
        in middle.checkpoint_evidence
    middle_tail = "2026-09-09T18:08:04+08:00" if frame == 12 \
        else "2026-09-09T18:07:59+08:00"
    assert (31, datetime.fromisoformat(middle_tail).timestamp(), revision) \
        in middle.checkpoint_evidence
    front_stop = 31 if frame == 12 else 32
    front_arrival = "2026-09-09T17:46:33+08:00" if frame == 12 \
        else "2026-09-09T17:47:04+08:00"
    assert (
        front_stop,
        datetime.fromisoformat(front_arrival).timestamp(),
        revision,
    ) in front.checkpoint_evidence

    def gate_minutes(estimate):
        return {
            gates[source[1]].minutes
            for source in estimate.source_observations
            if source[0] == "gate"
        }

    assert gate_minutes(middle) == {min(gate.minutes for gate in gates)}
    assert gate_minutes(upstream) == {sorted(gate.minutes for gate in gates)[1]}
    assert gate_minutes(front) == set()


def _live_frame_atomic_root_arguments(frame=12, *, reverse=False):
    probes, gates, _expected = _kmb_91_sparse_gate_partition_fixture(
        frame, reverse=reverse,
    )
    checkpoint = 31
    return (
        ("KMB", "91", "inbound"),
        list(enumerate(gates)),
        [(index, row) for index, row in enumerate(probes) if row.index == checkpoint],
        list(enumerate(probes)),
        15,
        checkpoint,
    )


def test_atomic_fresh_root_alignment_is_unique_same_class_rank_shift():
    key, gates, current, route_rows, gate_index, checkpoint = \
        _live_frame_atomic_root_arguments()

    pairs = positions_module._atomic_kmb_fresh_root_pairs(
        key,
        gates,
        current,
        route_rows,
        gate_index=gate_index,
        checkpoint=checkpoint,
    )

    rows = dict(current)
    gate_by_source = dict(gates)
    assert [rows[probe].arrival_at.isoformat() for probe, _gate in pairs] == [
        "2026-09-09T18:08:04+08:00",
        "2026-09-09T18:29:43+08:00",
    ]
    assert [gate_by_source[gate].minutes for _probe, gate in pairs] == [4.0, 26.0]


@pytest.mark.parametrize(
    "invalid",
    [
        "cross_class",
        "beyond_grace",
        "mixed_revision",
        "stale",
        "tie",
        "source_alias",
        "route",
    ],
)
def test_atomic_fresh_root_alignment_fails_closed(invalid):
    key, gates, current, route_rows, gate_index, checkpoint = \
        _live_frame_atomic_root_arguments()
    rows = dict(current)
    middle_source = next(
        source
        for source, row in current
        if row.arrival_at.isoformat() == "2026-09-09T18:08:04+08:00"
    )
    if invalid == "cross_class":
        rows[middle_source].kind = EtaKind.SCHEDULED
    elif invalid == "beyond_grace":
        rows[middle_source].minutes = 19.74
        rows[middle_source].signed_minutes = 19.74
    elif invalid == "mixed_revision":
        rows[middle_source].refresh_generation += 1
    elif invalid == "stale":
        rows[middle_source].cache_age_seconds = 60.0
    elif invalid == "tie":
        final = max(current, key=lambda item: item[1].minutes)[1]
        rows[middle_source].minutes = final.minutes
        rows[middle_source].signed_minutes = final.signed_minutes
    elif invalid == "source_alias":
        current = [
            (source, copy(row) if source == middle_source else row)
            for source, row in current
        ]
    elif invalid == "route":
        key = ("KMB", "91M", "inbound")

    assert positions_module._atomic_kmb_fresh_root_pairs(
        key,
        gates,
        current,
        route_rows,
        gate_index=gate_index,
        checkpoint=checkpoint,
    ) == []


def test_atomic_fresh_root_alignment_rejects_equal_cross_class_competitor():
    key = ("KMB", "91", "inbound")
    base = datetime(2026, 9, 9, 18, 0, tzinfo=UTC)
    gates = [
        (0, AuthoritativeProbe(*key, 15, 4.0, EtaKind.REALTIME)),
        (1, AuthoritativeProbe(*key, 15, 26.0, EtaKind.SCHEDULED)),
    ]
    specs = (
        (21.1, EtaKind.SCHEDULED),
        (22.0, EtaKind.REALTIME),
        (42.2, EtaKind.SCHEDULED),
    )
    current = [
        (
            source,
            Probe(
                *key,
                31,
                minutes,
                kind=kind,
                cache_age_seconds=1.0,
                signed_minutes=minutes,
                refresh_generation=1,
                arrival_at=base + timedelta(minutes=minutes),
            ),
        )
        for source, (minutes, kind) in enumerate(specs)
    ]

    assert positions_module._atomic_kmb_fresh_root_pairs(
        key,
        gates,
        current,
        current,
        gate_index=15,
        checkpoint=31,
        prior_gate_assignments={},
    ) == []

    plan = _plan_gate_associations(
        [row for _source, row in current],
        [row for _source, row in gates],
        {key},
    )
    assert plan.gate_assignment == {0: 0}


def test_atomic_fresh_root_alignment_rejects_backwards_prior_root_chronology():
    key, gates, current, route_rows, gate_index, checkpoint = \
        _live_frame_atomic_root_arguments()
    stop_16_source, stop_16 = next(
        (source, row)
        for source, row in route_rows
        if row.index == 16 and row.kind is EtaKind.REALTIME
    )
    stop_31_source = next(
        source
        for source, row in current
        if row.arrival_at.isoformat() == "2026-09-09T18:08:04+08:00"
    )
    stop_16.minutes = 21.0
    stop_16.signed_minutes = 21.0
    stop_16.arrival_at = datetime.fromisoformat(
        "2026-09-09T18:08:29.048856+08:00"
    )

    assert positions_module._atomic_kmb_fresh_root_pairs(
        key,
        gates,
        current,
        route_rows,
        gate_index=gate_index,
        checkpoint=checkpoint,
        prior_gate_assignments={stop_16_source: 0},
    ) == []

    probes = [row for _source, row in route_rows]
    authoritative = [row for _source, row in gates]
    plan = _plan_gate_associations(probes, authoritative, {key})
    assert plan.gate_assignment.get(stop_31_source) != 0


@pytest.mark.parametrize("reverse", [False, True])
def test_atomic_fresh_root_alignment_does_not_steal_owned_future_gate(reverse):
    key, gates, current, route_rows, gate_index, checkpoint = \
        _live_frame_atomic_root_arguments(frame=16, reverse=reverse)
    future_source, future_gate = max(gates, key=lambda item: item[1].minutes)
    future_gate.minutes = 26.0
    future_gate.kind = EtaKind.REALTIME
    gate_25_source = next(
        source for source, gate in gates if gate.minutes == 25.0
    )
    tail_source = max(current, key=lambda item: item[1].minutes)[0]
    probes = [row for _source, row in route_rows]
    authoritative = [row for _source, row in gates]

    plan = _plan_gate_associations(probes, authoritative, {key})
    assert any(
        gate == future_source and probes[source].index < checkpoint
        for source, gate in plan.gate_assignment.items()
    )
    assert plan.gate_assignment.get(tail_source) != gate_25_source

    prior = {
        source: gate
        for source, gate in plan.gate_assignment.items()
        if probes[source].index < checkpoint
    }
    assert positions_module._atomic_kmb_fresh_root_pairs(
        key,
        gates,
        current,
        route_rows,
        gate_index=gate_index,
        checkpoint=checkpoint,
        prior_gate_assignments=prior,
    ) == []


def test_atomic_fresh_root_alignment_ignores_wholly_unrelated_prior_assignment():
    key, gates, current, route_rows, gate_index, checkpoint = \
        _live_frame_atomic_root_arguments()
    expected = positions_module._atomic_kmb_fresh_root_pairs(
        key,
        gates,
        current,
        route_rows,
        gate_index=gate_index,
        checkpoint=checkpoint,
        prior_gate_assignments={},
    )

    assert expected
    assert positions_module._atomic_kmb_fresh_root_pairs(
        key,
        gates,
        current,
        route_rows,
        gate_index=gate_index,
        checkpoint=checkpoint,
        prior_gate_assignments={10_000: 20_000},
    ) == expected


@pytest.mark.parametrize(
    "prior_factory",
    [
        lambda probe, _gate: {probe: 20_000},
        lambda _probe, gate: {10_000: gate},
        lambda _probe, gate: {True: gate},
        lambda _probe, _gate: {10_000: False},
        lambda probe, gate: {probe: gate},
    ],
)
def test_atomic_fresh_root_alignment_rejects_invalid_prior_assignment_scope(
    prior_factory,
):
    key, gates, current, route_rows, gate_index, checkpoint = \
        _live_frame_atomic_root_arguments()
    probe_source = current[0][0]
    gate_source = gates[0][0]

    assert positions_module._atomic_kmb_fresh_root_pairs(
        key,
        gates,
        current,
        route_rows,
        gate_index=gate_index,
        checkpoint=checkpoint,
        prior_gate_assignments=prior_factory(probe_source, gate_source),
    ) == []


@pytest.mark.parametrize(
    "invalid",
    [
        "probe_kind",
        "gate_kind",
        "probe_bool_index",
        "probe_float_index",
        "gate_bool_index",
        "gate_float_index",
    ],
)
def test_atomic_fresh_root_alignment_rejects_malformed_rows_without_raising(invalid):
    key, gates, current, route_rows, gate_index, checkpoint = \
        _live_frame_atomic_root_arguments()
    target_source, target = sorted(current, key=lambda item: item[1].minutes)[1]
    if invalid == "probe_kind":
        target.kind = []
    elif invalid == "gate_kind":
        gates[0][1].kind = {}
    elif invalid == "probe_bool_index":
        target.index = True
    elif invalid == "probe_float_index":
        target.index = float(checkpoint)
    elif invalid == "gate_bool_index":
        gates[0][1].index = True
    else:
        gates[0][1].index = float(gate_index)

    assert positions_module._atomic_kmb_fresh_root_pairs(
        key,
        gates,
        current,
        route_rows,
        gate_index=gate_index,
        checkpoint=checkpoint,
        prior_gate_assignments={},
    ) == []

    probes = [row for _source, row in route_rows]
    authoritative = [row for _source, row in gates]
    plan = _plan_gate_associations(probes, authoritative, {key})
    assert plan.gate_assignment.get(target_source) != 0
    assert isinstance(
        estimate_bus_positions(
            probes,
            [_line(*key, stop_count=34)],
            authoritative_etas=authoritative,
            observed_checkpoint_indices={key: range(34)},
        ),
        list,
    )


def _leading_due_handoff_arguments(*, reverse=False):
    key, gates, frontier, route_rows, gate_index, previous_checkpoint = \
        _live_frame_atomic_root_arguments(frame=16, reverse=reverse)
    probes = dict(route_rows)
    current = [
        (source, row)
        for source, row in route_rows
        if row.index == previous_checkpoint + 1
    ]
    fresh_pairs = positions_module._atomic_kmb_fresh_root_pairs(
        key,
        gates,
        frontier,
        route_rows,
        gate_index=gate_index,
        checkpoint=previous_checkpoint,
    )
    pairs = _align_gate_arrivals(
        frontier,
        current,
        gate_index=previous_checkpoint,
        checkpoint=previous_checkpoint + 1,
    )
    return key, frontier, current, {
        "previous_rows": list(frontier),
        "gate_index": gate_index,
        "previous_checkpoint": previous_checkpoint,
        "checkpoint": previous_checkpoint + 1,
        "gate_assignments": {source: gate for source, gate in fresh_pairs},
        "gate_rows": dict(gates),
        "passed_rows": set(),
        "pairs": pairs,
        "carried": True,
        "all_rows": probes,
    }


def test_atomic_frontier_carries_across_one_leading_due_vehicle():
    key, frontier, current, args = _leading_due_handoff_arguments()
    assert _atomic_kmb_frontier_certificate(key, frontier, current, **{
        name: value for name, value in args.items() if name != "all_rows"
    })


@pytest.mark.parametrize("invalid", ["future", "scheduled", "two_leading", "root"])
def test_atomic_frontier_leading_due_handoff_fails_closed(invalid):
    key, frontier, current, args = _leading_due_handoff_arguments()
    leading_source, leading = min(current, key=lambda item: item[1].minutes)
    if invalid == "future":
        leading.minutes = 0.1
        leading.signed_minutes = 0.1
    elif invalid == "scheduled":
        leading.kind = EtaKind.SCHEDULED
    elif invalid == "two_leading":
        duplicate = copy(leading)
        duplicate.arrival_at -= timedelta(seconds=1)
        current.append((max(args["all_rows"]) + 1, duplicate))
    elif invalid == "root":
        args["gate_assignments"].pop(next(iter(args["gate_assignments"])))

    assert not _atomic_kmb_frontier_certificate(key, frontier, current, **{
        name: value for name, value in args.items() if name != "all_rows"
    })


def test_atomic_frontier_rejects_compatible_unrepresented_gate_suffix():
    key, frontier, current, args = _leading_due_handoff_arguments()
    future_gate = args["gate_rows"][max(args["gate_rows"])]
    future_gate.kind = EtaKind.REALTIME
    future_gate.minutes = 25.5
    _source, final = max(current, key=lambda item: item[1].minutes)
    final.minutes = 43.5
    final.signed_minutes = 43.5

    assert not _atomic_kmb_frontier_certificate(key, frontier, current, **{
        name: value for name, value in args.items() if name != "all_rows"
    })


@pytest.mark.parametrize("reverse", [False, True])
def test_atomic_frontier_keeps_excluding_future_gate_after_leading_handoff(reverse):
    key, frontier, stop_32, args = _leading_due_handoff_arguments(reverse=reverse)
    future_gate = args["gate_rows"][max(
        args["gate_rows"], key=lambda source: args["gate_rows"][source].minutes,
    )]
    future_gate.kind = EtaKind.REALTIME
    future_gate.minutes = 26.0
    first_args = {
        name: value for name, value in args.items() if name != "all_rows"
    }
    assert _atomic_kmb_frontier_certificate(
        key, frontier, stop_32, **first_args,
    )

    route_rows = args["all_rows"]
    stop_33 = [
        (source, row) for source, row in route_rows.items() if row.index == 33
    ]
    _tail_source, tail = max(stop_33, key=lambda item: item[1].minutes)
    shift = 46.0 - tail.minutes
    tail.minutes = 46.0
    tail.signed_minutes = 46.0
    tail.arrival_at += timedelta(minutes=shift)
    stop_32_gate_assignments = {
        current_source: args["gate_assignments"][previous_source]
        for current_source, previous_source in args["pairs"]
    }
    leading_source = min(stop_32, key=lambda item: item[1].minutes)[0]
    pairs = _align_gate_arrivals(
        stop_32,
        stop_33,
        gate_index=32,
        checkpoint=33,
    )

    assert not _atomic_kmb_frontier_certificate(
        key,
        stop_32,
        stop_33,
        previous_rows=stop_32,
        gate_index=15,
        previous_checkpoint=32,
        checkpoint=33,
        gate_assignments=stop_32_gate_assignments,
        gate_rows=args["gate_rows"],
        passed_rows={leading_source},
        pairs=pairs,
        carried=True,
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_gate_plan_keeps_excluding_future_gate_after_leading_handoff(reverse):
    probes, gates, _expected = _kmb_91_sparse_gate_partition_fixture(
        16, reverse=reverse,
    )
    key = ("KMB", "91", "inbound")
    probes = [row for row in probes if row.index >= 31]
    future_gate = max(gates, key=lambda row: row.minutes)
    future_gate.kind = EtaKind.REALTIME

    stop_32_tail = max(
        (row for row in probes if row.index == 32), key=lambda row: row.minutes,
    )
    stop_32_shift = 58.0 - stop_32_tail.minutes
    stop_32_tail.minutes = 58.0
    stop_32_tail.signed_minutes = 58.0
    stop_32_tail.arrival_at += timedelta(minutes=stop_32_shift)
    stop_33_tail = max(
        (row for row in probes if row.index == 33), key=lambda row: row.minutes,
    )
    stop_33_shift = 74.0 - stop_33_tail.minutes
    stop_33_tail.minutes = 74.0
    stop_33_tail.signed_minutes = 74.0
    stop_33_tail.arrival_at += timedelta(minutes=stop_33_shift)

    plan = _plan_gate_associations(probes, gates, {key})
    gate_3_source = next(
        source for source, gate in enumerate(gates) if gate.minutes == 3.0
    )
    gate_25_source = next(
        source for source, gate in enumerate(gates) if gate.minutes == 25.0
    )
    stop_31 = sorted(
        (pair for pair in enumerate(probes) if pair[1].index == 31),
        key=lambda item: item[1].minutes,
    )
    stop_32 = sorted(
        (pair for pair in enumerate(probes) if pair[1].index == 32),
        key=lambda item: item[1].minutes,
    )
    stop_33 = sorted(
        (pair for pair in enumerate(probes) if pair[1].index == 33),
        key=lambda item: item[1].minutes,
    )

    assert plan.gate_assignment[stop_31[0][0]] == gate_3_source
    assert plan.gate_assignment[stop_31[1][0]] == gate_25_source
    assert plan.gate_assignment[stop_32[1][0]] == gate_3_source
    assert plan.gate_assignment[stop_32[2][0]] == gate_25_source
    assert plan.gate_assignment.get(stop_33[1][0]) != gate_3_source


@pytest.mark.parametrize("frame", [25, 40])
@pytest.mark.parametrize("reverse", [False, True])
def test_atomic_kmb_gate_frontier_preserves_three_full_snapshot_journeys(frame, reverse):
    snapshot, gates, expected_rows, positions, brackets = _atomic_gate_frontier_fixture(
        frame, reverse=reverse,
    )
    key = ("KMB", "91", "inbound")
    probes = list(snapshot.rows)
    route_line = _line(*key, stop_count=34)
    plan = _plan_gate_associations(probes, gates, {key})
    result = estimate_bus_positions(
        probes, [route_line], authoritative_etas=gates,
        observed_checkpoint_indices={key: {row.index for row in probes}},
    )
    assert len(result) == 3
    assert [marker.position for marker in result] == pytest.approx(positions)
    source_multiset = Counter(source for marker in result for source in marker.source_observations)
    expected_sources = Counter({("probe", index): 1 for index, row in enumerate(probes)
                                if row.minutes is not None})
    expected_sources.update(("gate", index) for index in range(len(gates)))
    assert source_multiset == expected_sources
    assert sum(source_multiset.values()) == (37 if frame == 25 else 33)
    for number, (owned_rows, marker, position, bracket) in enumerate(zip(
        expected_rows, result, positions, brackets, strict=True,
    )):
        slots = {index for index, row in enumerate(probes) if any(row is owned for owned in owned_rows)}
        expected = {("probe", index) for index in slots}
        if number < 2:
            root = max(range(2), key=lambda index: gates[index].minutes) if number == 0 else min(
                range(2), key=lambda index: gates[index].minutes,
            )
            expected.add(("gate", root))
            assert all(plan.gate_assignment[index] == root for index in slots)
            assert not slots & plan.passed_probe_rows
        else:
            assert slots <= plan.passed_probe_rows
            assert not slots & plan.gate_assignment.keys()
        assert marker.source_observations == expected
        assert marker.bracket == bracket
        assert marker.lat == pytest.approx(22.333360)
        assert marker.lon == pytest.approx(114.260 + position * 0.001)
        assert marker.position < 30


def _frontier_certificate_arguments():
    snapshot, gates, _owned, _positions, _brackets = _atomic_gate_frontier_fixture(25)
    probes = list(snapshot.rows)
    key = ("KMB", "91", "inbound")
    plan = _plan_gate_associations(probes, gates, {key})
    frontier = [(index, row) for index, row in enumerate(probes) if row.index == 30]
    current = [(index, row) for index, row in enumerate(probes) if row.index == 31]
    pairs = _align_gate_arrivals(frontier, current, gate_index=30, checkpoint=31)
    return key, frontier, current, dict(
        previous_rows=list(frontier), gate_index=15,
        previous_checkpoint=30, checkpoint=31,
        gate_assignments=dict(plan.gate_assignment), gate_rows=dict(enumerate(gates)),
        passed_rows=set(plan.passed_probe_rows), pairs=pairs,
    )


def _fixed_three_eta_horizon_slide_fixture(*, reverse=False, repeated_slide=False):
    """Atomic KMB response where a passed bus displaces the third gate root."""
    key = ("KMB", "91", "inbound")
    base = datetime(2026, 9, 10, 3, 43, 43, tzinfo=UTC)
    revision = 972
    rows_by_stop = {
        16: (
            (17.946430, EtaKind.REALTIME),
            (25.746430, EtaKind.SCHEDULED),
            (46.363097, EtaKind.SCHEDULED),
        ),
        21: (
            (-0.570237, EtaKind.REALTIME),
            (24.746430, EtaKind.REALTIME),
            (32.496430, EtaKind.SCHEDULED),
        ),
        22: (
            (0.529763, EtaKind.REALTIME),
            (25.863097, EtaKind.REALTIME),
            (33.596430, EtaKind.SCHEDULED),
        ),
        24: (
            (1.946430, EtaKind.REALTIME),
            (27.196430, EtaKind.REALTIME),
            (35.013097, EtaKind.SCHEDULED),
        ),
        29: (
            (4.663097, EtaKind.REALTIME),
            (29.863097, EtaKind.REALTIME),
            (37.746430, EtaKind.SCHEDULED),
        ),
        30: (
            (4.979763, EtaKind.REALTIME),
            (30.179763, EtaKind.REALTIME),
            (38.063097, EtaKind.SCHEDULED),
        ),
        31: (
            (5.646430, EtaKind.REALTIME),
            (30.813097, EtaKind.REALTIME),
            (38.729763, EtaKind.SCHEDULED),
        ),
        33: (
            (8.079763, EtaKind.REALTIME),
            (33.213097, EtaKind.REALTIME),
            (41.179763, EtaKind.SCHEDULED),
        ),
    }
    if repeated_slide:
        # A second passed bus enters the next-three window at checkpoint 30,
        # displacing the remaining trailing root. The following checkpoints
        # preserve all three ranks: two passed buses plus the final gate root.
        for checkpoint, leading in ((30, -0.3), (31, 0.366), (33, 2.8)):
            rows_by_stop[checkpoint] = (
                (leading, EtaKind.REALTIME),
                *rows_by_stop[checkpoint][:-1],
            )
    probes = [
        Probe(
            *key,
            index,
            max(0.0, signed),
            kind=kind,
            cache_age_seconds=19.0,
            signed_minutes=signed,
            refresh_generation=revision,
            arrival_at=base + timedelta(minutes=signed),
        )
        for index, rows in rows_by_stop.items()
        for signed, kind in rows
    ]
    gates = [
        AuthoritativeProbe(*key, 15, minutes, kind=kind)
        for minutes, kind in (
            (16.0, EtaKind.REALTIME),
            (24.0, EtaKind.SCHEDULED),
            (45.0, EtaKind.SCHEDULED),
        )
    ]
    if reverse:
        probes.reverse()
        gates.reverse()
    return key, probes, gates, rows_by_stop


def _fixed_three_eta_horizon_slide_certificate_arguments(*, reverse=False):
    key, probes, gates, _rows_by_stop = _fixed_three_eta_horizon_slide_fixture(
        reverse=reverse,
    )
    frontier = [
        (index, row) for index, row in enumerate(probes) if row.index == 16
    ]
    current = [
        (index, row) for index, row in enumerate(probes) if row.index == 21
    ]
    ordered_frontier = sorted(frontier, key=lambda item: item[1].minutes)
    ordered_gates = sorted(enumerate(gates), key=lambda item: item[1].minutes)
    gate_assignments = {
        probe_source: gate_source
        for (probe_source, _probe), (gate_source, _gate) in zip(
            ordered_frontier,
            ordered_gates,
            strict=True,
        )
    }
    return key, frontier, current, {
        "previous_rows": list(frontier),
        "gate_index": 15,
        "previous_checkpoint": 16,
        "checkpoint": 21,
        "gate_assignments": gate_assignments,
        "gate_rows": dict(enumerate(gates)),
        "passed_rows": set(),
        "pairs": _align_gate_arrivals(
            frontier,
            current,
            gate_index=16,
            checkpoint=21,
        ),
        "carried": False,
    }


@pytest.mark.parametrize("reverse", [False, True])
def test_atomic_kmb_certificate_accepts_fixed_three_eta_horizon_slide(reverse):
    key, frontier, current, arguments = \
        _fixed_three_eta_horizon_slide_certificate_arguments(reverse=reverse)

    assert _atomic_kmb_frontier_certificate(
        key,
        frontier,
        current,
        **arguments,
    )


@pytest.mark.parametrize(
    "invalid",
    ["future_leader", "scheduled_leader", "mixed_revision", "dropped_root", "future_gate"],
)
def test_atomic_kmb_fixed_three_eta_horizon_slide_fails_closed(invalid):
    key, frontier, current, arguments = \
        _fixed_three_eta_horizon_slide_certificate_arguments()
    leading_source, leading = min(current, key=lambda item: item[1].minutes)
    if invalid == "future_leader":
        leading.minutes = 0.1
        leading.signed_minutes = 0.1
    elif invalid == "scheduled_leader":
        leading.kind = EtaKind.SCHEDULED
    elif invalid == "mixed_revision":
        leading.refresh_generation += 1
    elif invalid == "dropped_root":
        dropped_source, _row = max(frontier, key=lambda item: item[1].minutes)
        arguments["gate_assignments"].pop(dropped_source)
    else:
        dropped_source, _row = max(frontier, key=lambda item: item[1].minutes)
        dropped_gate = arguments["gate_assignments"][dropped_source]
        arguments["gate_rows"][dropped_gate].minutes = 25.0

    assert not _atomic_kmb_frontier_certificate(
        key,
        frontier,
        current,
        **arguments,
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_atomic_kmb_fixed_three_eta_horizon_does_not_split_fast_journey(reverse):
    key, probes, gates, rows_by_stop = _fixed_three_eta_horizon_slide_fixture(
        reverse=reverse,
    )
    line = _line(*key, stop_count=34)
    plan = _plan_gate_associations(probes, gates, {key})
    estimates = estimate_bus_positions(
        probes,
        [line],
        authoritative_etas=gates,
        observed_checkpoint_indices={key: rows_by_stop},
    )

    gate_by_minutes = {gate.minutes: index for index, gate in enumerate(gates)}
    gate_zero = gate_by_minutes[16.0]
    gate_one = gate_by_minutes[24.0]
    rows_at = {
        checkpoint: sorted(
            (
                (index, row)
                for index, row in enumerate(probes)
                if row.index == checkpoint
            ),
            key=lambda item: item[1].minutes,
        )
        for checkpoint in rows_by_stop
    }
    assert rows_at[21][0][0] in plan.passed_probe_rows
    for checkpoint in (21, 22, 24, 29, 30, 31, 33):
        assert plan.gate_assignment[rows_at[checkpoint][1][0]] == gate_zero
        assert plan.gate_assignment[rows_at[checkpoint][2][0]] == gate_one

    assert len(estimates) == 3
    gate_zero_marker = next(
        marker
        for marker in estimates
        if ("gate", gate_zero) in marker.source_observations
    )
    gate_zero_evidence = {
        (checkpoint, round(arrival, 6), revision)
        for checkpoint, arrival, revision in gate_zero_marker.checkpoint_evidence
    }
    for checkpoint, rank in ((31, 1), (33, 1)):
        expected = rows_at[checkpoint][rank][1]
        assert (
            checkpoint,
            round(expected.arrival_at.timestamp(), 6),
            expected.refresh_generation,
        ) in gate_zero_evidence
    assert not any(
        {checkpoint for checkpoint, _arrival, _revision in marker.checkpoint_evidence}
        <= {31, 33}
        for marker in estimates
    )
    audit = audit_marker_positions(probes, gates, estimates, [line])
    assert audit["ok"], audit["issues"]


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("carried, expected", [(True, True), (False, False)])
def test_atomic_kmb_one_root_frontier_requires_prior_certificate(
    reverse,
    carried,
    expected,
):
    key, probes, gates, _rows_by_stop = _fixed_three_eta_horizon_slide_fixture(
        reverse=reverse,
        repeated_slide=True,
    )
    frontier = [
        (index, row) for index, row in enumerate(probes) if row.index == 30
    ]
    current = [
        (index, row) for index, row in enumerate(probes) if row.index == 31
    ]
    ordered_frontier = sorted(frontier, key=lambda item: item[1].minutes)
    gate_zero = next(
        index for index, gate in enumerate(gates) if gate.minutes == 16.0
    )

    certificate = _atomic_kmb_frontier_certificate(
        key,
        frontier,
        current,
        previous_rows=list(frontier),
        gate_index=15,
        previous_checkpoint=30,
        checkpoint=31,
        gate_assignments={ordered_frontier[-1][0]: gate_zero},
        gate_rows=dict(enumerate(gates)),
        passed_rows={source for source, _row in ordered_frontier[:-1]},
        pairs=_align_gate_arrivals(
            frontier,
            current,
            gate_index=30,
            checkpoint=31,
        ),
        carried=carried,
    )

    assert certificate is expected


@pytest.mark.parametrize("invalid_dropped_root", ["owned", "skipped", "missing"])
def test_atomic_kmb_repeated_slide_still_validates_dropped_root(
    invalid_dropped_root,
):
    key, probes, gates, _rows_by_stop = _fixed_three_eta_horizon_slide_fixture(
        repeated_slide=True,
    )
    frontier = [
        (index, row) for index, row in enumerate(probes) if row.index == 29
    ]
    current = [
        (index, row) for index, row in enumerate(probes) if row.index == 30
    ]
    ordered_frontier = sorted(frontier, key=lambda item: item[1].minutes)
    ordered_gates = sorted(enumerate(gates), key=lambda item: item[1].minutes)
    passed_source = ordered_frontier[0][0]
    retained_source = ordered_frontier[1][0]
    dropped_source = ordered_frontier[2][0]
    retained_root = ordered_gates[0][0]
    invalid_root = {
        "owned": retained_root,
        "skipped": ordered_gates[2][0],
        "missing": 999,
    }[invalid_dropped_root]

    assert not _atomic_kmb_frontier_certificate(
        key,
        frontier,
        current,
        previous_rows=list(frontier),
        gate_index=15,
        previous_checkpoint=29,
        checkpoint=30,
        gate_assignments={
            retained_source: retained_root,
            dropped_source: invalid_root,
        },
        gate_rows=dict(enumerate(gates)),
        passed_rows={passed_source},
        pairs=_align_gate_arrivals(
            frontier,
            current,
            gate_index=29,
            checkpoint=30,
        ),
        carried=True,
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_atomic_kmb_repeated_fixed_three_eta_slides_keep_last_root(reverse):
    key, probes, gates, rows_by_stop = _fixed_three_eta_horizon_slide_fixture(
        reverse=reverse,
        repeated_slide=True,
    )
    line = _line(*key, stop_count=34)
    plan = _plan_gate_associations(probes, gates, {key})
    estimates = estimate_bus_positions(
        probes,
        [line],
        authoritative_etas=gates,
        observed_checkpoint_indices={key: rows_by_stop},
    )

    gate_zero = next(
        index for index, gate in enumerate(gates) if gate.minutes == 16.0
    )
    rows_at = {
        checkpoint: sorted(
            (
                (index, row)
                for index, row in enumerate(probes)
                if row.index == checkpoint
            ),
            key=lambda item: item[1].minutes,
        )
        for checkpoint in rows_by_stop
    }
    for checkpoint in (30, 31, 33):
        assert {source for source, _row in rows_at[checkpoint][:2]} \
            <= plan.passed_probe_rows
        assert plan.gate_assignment[rows_at[checkpoint][2][0]] == gate_zero

    assert len(estimates) == 4
    gate_zero_marker = next(
        marker
        for marker in estimates
        if ("gate", gate_zero) in marker.source_observations
    )
    gate_zero_evidence = {
        (checkpoint, round(arrival, 6), revision)
        for checkpoint, arrival, revision in gate_zero_marker.checkpoint_evidence
    }
    for checkpoint in (31, 33):
        expected = rows_at[checkpoint][2][1]
        assert (
            checkpoint,
            round(expected.arrival_at.timestamp(), 6),
            expected.refresh_generation,
        ) in gate_zero_evidence
    assert not any(
        {checkpoint for checkpoint, _arrival, _revision in marker.checkpoint_evidence}
        <= {31, 33}
        for marker in estimates
    )
    audit = audit_marker_positions(probes, gates, estimates, [line])
    assert audit["ok"], audit["issues"]


def _two_rank_gate_handoff_arguments(*, leading_gate_minutes=0.2, extra=False):
    """Frame-9-shaped live/scheduled gate handoff with exact two-row census."""
    key = ("KMB", "91", "inbound")
    base = datetime(2026, 1, 1, tzinfo=UTC)
    gates = [Probe(*key, 15, leading_gate_minutes, kind=EtaKind.REALTIME),
             Probe(*key, 15, 28.0, kind=EtaKind.SCHEDULED)]
    values = ((16, (0.9705484833, 29.2205484833)),
              (24, (10.0205484833, 38.3705484833)),
              (25, (10.6038818167, 38.9538818167)),
              (29, (12.88721515, 41.2538818167)),
              (30, (13.2205484833, 41.5538818167)),
              (33, (16.28721515, 44.6038818167)))
    all_rows = []
    for index, minutes_by_rank in values:
      for rank, minutes in enumerate(minutes_by_rank):
        kind = EtaKind.REALTIME if rank == 0 else EtaKind.SCHEDULED
        previous = Probe(
            *key, index, minutes, kind=kind, signed_minutes=minutes,
            cache_age_seconds=9.2, refresh_generation=9,
            arrival_at=base + timedelta(minutes=minutes),
        )
        all_rows.append(previous)
    frontier = [(4, all_rows[6]), (5, all_rows[7])]
    current = [(8, all_rows[8]), (9, all_rows[9])]
    if extra:
        row = copy(current[1][1])
        row.minutes = row.signed_minutes = 40.0
        row.arrival_at = base + timedelta(minutes=40)
        current.append((12, row))
    return key, frontier, current, dict(
        previous_rows=list(frontier), gate_index=15,
        previous_checkpoint=29, checkpoint=30,
        gate_assignments={4: 0, 5: 1}, gate_rows=dict(enumerate(gates)),
        passed_rows=set(),
        pairs=[(8, 4), (9, 5)],
        all_rows=all_rows,
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_atomic_frontier_admits_exact_two_rank_due_gate_handoff(reverse):
    key, frontier, current, args = _two_rank_gate_handoff_arguments()
    args.pop("all_rows")
    if reverse:
        frontier.reverse()
        current.reverse()
        args["previous_rows"] = list(frontier)
    assert _atomic_kmb_frontier_certificate(key, frontier, current, **args)


@pytest.mark.parametrize("reverse", [False, True])
def test_two_rank_gate_handoff_renders_two_markers_with_each_source_once(reverse):
    key, frontier, current, _args = _two_rank_gate_handoff_arguments()
    probes = list(_two_rank_gate_handoff_arguments()[3]["all_rows"])
    gates = [Probe(*key, 15, 0.0, kind=EtaKind.REALTIME),
             Probe(*key, 15, 28.0, kind=EtaKind.SCHEDULED)]
    if reverse:
        probes.reverse()
        gates.reverse()
    result = estimate_bus_positions(
        probes, [_line(*key, stop_count=34)], authoritative_etas=gates,
        observed_checkpoint_indices={key: {16, 24, 25, 29, 30, 33}},
    )
    assert len(result) == 2
    tokens_by_kind = {
        EtaKind.REALTIME: {('probe', index) for index, row in enumerate(probes)
                           if row.kind is EtaKind.REALTIME},
        EtaKind.SCHEDULED: {('probe', index) for index, row in enumerate(probes)
                            if row.kind is EtaKind.SCHEDULED},
    }
    live_marker = next(marker for marker in result
                       if next(iter(tokens_by_kind[EtaKind.REALTIME]))
                       in marker.source_observations)
    scheduled_marker = next(marker for marker in result if marker is not live_marker)
    assert {token for token in live_marker.source_observations if token[0] == 'probe'} == tokens_by_kind[EtaKind.REALTIME]
    assert {token for token in scheduled_marker.source_observations if token[0] == 'probe'} == tokens_by_kind[EtaKind.SCHEDULED]
    assert next(token for token in live_marker.source_observations if token[0] == 'gate') == next(
        ('gate', index) for index, gate in enumerate(gates) if gate.kind is EtaKind.REALTIME
    )
    assert next(token for token in scheduled_marker.source_observations if token[0] == 'gate') == next(
        ('gate', index) for index, gate in enumerate(gates) if gate.kind is EtaKind.SCHEDULED
    )
    # Stop 30 is the one-minute first-deficit admission, and stop 33 is the
    # subsequent carried checkpoint; both remain owned by their original rank.
    stop30_live = ('probe', next(i for i, row in enumerate(probes)
                                 if row.index == 30 and row.kind is EtaKind.REALTIME))
    stop33_live = ('probe', next(i for i, row in enumerate(probes)
                                 if row.index == 33 and row.kind is EtaKind.REALTIME))
    stop30_scheduled = ('probe', next(i for i, row in enumerate(probes)
                                      if row.index == 30 and row.kind is EtaKind.SCHEDULED))
    stop33_scheduled = ('probe', next(i for i, row in enumerate(probes)
                                      if row.index == 33 and row.kind is EtaKind.SCHEDULED))
    assert {stop30_live, stop33_live} <= live_marker.source_observations
    assert {stop30_scheduled, stop33_scheduled} <= scheduled_marker.source_observations
    assert {live_marker.position, scheduled_marker.position} == {1.0, 15.0}
    source_multiset = Counter(
        source for marker in result for source in marker.source_observations
    )
    expected_sources = Counter({
        ("probe", index): 1 for index in range(len(probes))
    })
    expected_sources.update(("gate", index) for index in range(len(gates)))
    assert source_multiset == expected_sources


def test_atomic_frontier_two_rank_handoff_requires_due_leading_gate_and_exact_shape():
    key, frontier, current, args = _two_rank_gate_handoff_arguments(
        leading_gate_minutes=0.51,
    )
    args.pop("all_rows")
    assert not _atomic_kmb_frontier_certificate(key, frontier, current, **args)


def test_atomic_gate_partition_preserves_a_genuine_fourth_passed_ladder():
    probes, gates, _expected = _kmb_91_sparse_gate_partition_fixture(12)
    extra = [
        Probe(
            "KMB",
            "91",
            "inbound",
            index,
            minutes,
            kind=EtaKind.REALTIME,
            cache_age_seconds=9.843999999997322,
            signed_minutes=minutes,
            refresh_generation=129,
            arrival_at=datetime.fromisoformat(arrival),
        )
        for index, minutes, arrival in (
            (31, 8.0, "2026-09-09T17:55:31+08:00"),
            (32, 8.7, "2026-09-09T17:56:13+08:00"),
            (33, 10.3, "2026-09-09T17:57:49+08:00"),
        )
    ]
    probes.extend(extra)

    estimates = estimate_bus_positions(
        probes,
        [_line("KMB", "91", "inbound", stop_count=34)],
        authoritative_etas=gates,
        observed_checkpoint_indices={("KMB", "91", "inbound"): range(34)},
    )

    assert len(estimates) == 4
    sources = [
        source
        for estimate in estimates
        for source in estimate.source_observations
        if source[0] == "probe"
    ]
    assert Counter(sources) == Counter(
        ("probe", index) for index in range(len(probes))
    )

    key, frontier, current, args = _two_rank_gate_handoff_arguments(extra=True)
    args.pop("all_rows")
    assert not _atomic_kmb_frontier_certificate(key, frontier, current, **args)

    key, frontier, current, args = _two_rank_gate_handoff_arguments()
    args.pop("all_rows")
    args["gate_rows"][2] = copy(args["gate_rows"][1])
    assert not _atomic_kmb_frontier_certificate(key, frontier, current, **args)

    key, frontier, current, args = _two_rank_gate_handoff_arguments()
    args.pop("all_rows")
    args["gate_assignments"].pop(5)
    assert not _atomic_kmb_frontier_certificate(key, frontier, current, **args)


@pytest.mark.parametrize("invalid", [
    "stale", "revision", "boolean_minutes", "boolean_age", "boolean_signed",
    "nonfinite_minutes", "missing_arrival", "tie", "count", "duplicate_source",
    "incomplete_population", "missing_root", "missing_passed", "root_rank",
    "too_early", "true_passed",
    "scheduled_transition", "unknown_kind", "unavailable", "gmb", "other_kmb",
    "wrong_direction", "zero",
])
def test_atomic_frontier_certificate_fails_closed(invalid):
    key, frontier, current, args = _frontier_certificate_arguments()
    assert _atomic_kmb_frontier_certificate(key, frontier, current, **args)
    row = current[0][1]  # a gate-backed current row
    if invalid == "stale":
        row.cache_age_seconds = 60
    elif invalid == "revision":
        row.refresh_generation += 1
    elif invalid == "boolean_minutes":
        row.minutes = True
    elif invalid == "boolean_age":
        row.cache_age_seconds = False
    elif invalid == "boolean_signed":
        row.signed_minutes = True
    elif invalid == "nonfinite_minutes":
        row.minutes = float("nan")
    elif invalid == "missing_arrival":
        row.arrival_at = None
    elif invalid == "tie":
        row.minutes = row.signed_minutes = current[1][1].minutes
    elif invalid == "count":
        current.pop()
    elif invalid == "duplicate_source":
        current[0] = (frontier[0][0], row)
    elif invalid == "incomplete_population":
        args["previous_rows"].append((999, copy(frontier[0][1])))
    elif invalid == "missing_root":
        args["gate_assignments"].pop(frontier[0][0])
    elif invalid == "missing_passed":
        args["passed_rows"].clear()
    elif invalid == "root_rank":
        roots = args["gate_assignments"]
        roots[frontier[0][0]], roots[frontier[1][0]] = roots[frontier[1][0]], roots[frontier[0][0]]
    elif invalid in {"too_early", "true_passed", "zero"}:
        root = args["gate_rows"][args["gate_assignments"][frontier[0][0]]]
        delta = 14.749 if invalid == "too_early" else -0.1
        row.minutes = row.signed_minutes = root.minutes + delta if invalid != "zero" else 0
    elif invalid == "scheduled_transition":
        next(
            present for source, present in current
            if source in args["gate_assignments"] and present.kind is EtaKind.REALTIME
        ).kind = EtaKind.SCHEDULED
    elif invalid == "unknown_kind":
        row.kind = "unknown"
    elif invalid == "unavailable":
        row.kind = EtaKind.UNAVAILABLE
    elif invalid == "gmb":
        key = ("GMB", "91", "inbound")
    elif invalid == "other_kmb":
        key = ("KMB", "91P", "inbound")
    elif invalid == "wrong_direction":
        row.bound = "outbound"
    assert not _atomic_kmb_frontier_certificate(key, frontier, current, **args)


@pytest.mark.parametrize("scheduled", [False, True])
def test_atomic_frontier_preserves_live_class_or_same_scheduled_journey(scheduled):
    key, frontier, current, args = _frontier_certificate_arguments()
    for previous, present in zip(frontier, current, strict=True):
        if previous[0] in args["gate_assignments"]:
            root = args["gate_rows"][args["gate_assignments"][previous[0]]]
            previous[1].kind = EtaKind.SCHEDULED if scheduled else EtaKind.MOVING_SLOWLY
            present[1].kind = EtaKind.SCHEDULED if scheduled else EtaKind.DELAYED
            root.kind = EtaKind.SCHEDULED if scheduled else EtaKind.REALTIME
    assert _atomic_kmb_frontier_certificate(key, frontier, current, **args)


def test_atomic_frontier_does_not_promote_live_probe_suffix_from_scheduled_gate():
    key, frontier, current, args = _frontier_certificate_arguments()
    source = next(
        source for source, row in frontier
        if source in args["gate_assignments"] and row.kind is EtaKind.REALTIME
    )
    root = args["gate_rows"][args["gate_assignments"][source]]
    root.kind = EtaKind.SCHEDULED
    assert not _atomic_kmb_frontier_certificate(key, frontier, current, **args)


@pytest.mark.parametrize("owner", ["passed", "gate_root"])
@pytest.mark.parametrize("clock", ["arrival", "signed_eta"])
def test_atomic_frontier_rejects_backwards_matched_edge(owner, clock):
    key, frontier, current, args = _frontier_certificate_arguments()
    previous_by_source = dict(frontier)
    current_by_source = dict(current)
    owners = args["passed_rows" if owner == "passed" else "gate_assignments"]
    current_source, previous_source = next(
        pair for pair in args["pairs"] if pair[1] in owners
    )
    previous = previous_by_source[previous_source]
    present = current_by_source[current_source]
    if clock == "arrival":
        present.arrival_at = previous.arrival_at - timedelta(seconds=1)
    else:
        present.signed_minutes = previous.signed_minutes - 0.1
        present.minutes = max(0, present.signed_minutes)
    assert not _atomic_kmb_frontier_certificate(key, frontier, current, **args)


@pytest.mark.parametrize("reverse", [False, True])
def test_atomic_frontier_never_certifies_a_filtered_previous_population(
    monkeypatch, reverse,
):
    snapshot, gates, _owned, _positions, _brackets = _atomic_gate_frontier_fixture(
        25, reverse=reverse,
    )
    probes = list(snapshot.rows)
    template = next(
        row for row in probes
        if row.index == 30 and row.kind is EtaKind.SCHEDULED
    )
    omitted = copy(template)
    omitted.signed_minutes = omitted.minutes = 35.0
    omitted.arrival_at += timedelta(minutes=35.0 - template.signed_minutes)
    probes.append(omitted)
    if reverse:
        probes.reverse()
    checks = []
    original = positions_module._atomic_kmb_frontier_certificate

    def record(*args, **kwargs):
        result = original(*args, **kwargs)
        checks.append((kwargs["previous_checkpoint"], kwargs["checkpoint"], result))
        return result

    monkeypatch.setattr(positions_module, "_atomic_kmb_frontier_certificate", record)
    _plan_gate_associations(probes, gates, {("KMB", "91", "inbound")})

    assert (30, 31, False) in checks
    assert not any(result for previous, _current, result in checks if previous >= 30)


@pytest.mark.parametrize("empty_kind", [EtaKind.REALTIME, EtaKind.UNAVAILABLE])
@pytest.mark.parametrize("reverse", [False, True])
def test_atomic_frontier_cannot_carry_across_explicit_empty_checkpoint(
    monkeypatch, reverse, empty_kind,
):
    snapshot, gates, _owned, _positions, _brackets = _atomic_gate_frontier_fixture(
        25, reverse=reverse,
    )
    key = ("KMB", "91", "inbound")
    probes = [row for row in snapshot.rows if row.index != 32]
    probes.append(
        Probe(
            *key, 32, None, kind=empty_kind, cache_age_seconds=9.2,
            refresh_generation=25,
        )
    )
    checks = []
    original = positions_module._atomic_kmb_frontier_certificate

    def record(*args, **kwargs):
        result = original(*args, **kwargs)
        checks.append((kwargs["previous_checkpoint"], kwargs["checkpoint"], result))
        return result

    monkeypatch.setattr(positions_module, "_atomic_kmb_frontier_certificate", record)
    _plan_gate_associations(probes, gates, {key})

    assert (30, 31, True) in checks
    assert not any(current >= 32 and result for _previous, current, result in checks)


def test_atomic_frontier_cannot_carry_across_rowless_observed_empty_checkpoint(
    monkeypatch,
):
    snapshot, gates, _owned, _positions, _brackets = _atomic_gate_frontier_fixture(25)
    key = ("KMB", "91", "inbound")
    probes = [row for row in snapshot.rows if row.index != 32]
    checks = []
    original = positions_module._atomic_kmb_frontier_certificate

    def record(*args, **kwargs):
        result = original(*args, **kwargs)
        checks.append((kwargs["previous_checkpoint"], kwargs["checkpoint"], result))
        return result

    monkeypatch.setattr(positions_module, "_atomic_kmb_frontier_certificate", record)
    observed = {key: {row.index for row in snapshot.rows}}
    _plan_gate_associations(
        probes, gates, {key}, observed_checkpoint_indices=observed,
    )

    assert (30, 31, True) in checks
    assert not any(current >= 32 and result for _previous, current, result in checks)


def test_observed_checkpoint_collection_is_normalized_for_estimator_and_auditor():
    line = _line("KMB", "91", "inbound", stop_count=34)
    assert estimate_bus_positions(
        [], [line], observed_checkpoint_indices=range(34),
    ) == []
    audit = audit_marker_positions(
        [], [], [], [line], observed_checkpoint_indices=range(34),
    )
    assert audit["ok"]


@pytest.mark.parametrize(
    ("operator", "route", "bound"),
    [
        ("KMB", "91", "inbound"),
        ("KMB", "91P", "inbound"),
        ("GMB", "11B", "seq-1"),
    ],
)
@pytest.mark.parametrize("empty_form", ["sentinel", "metadata"])
def test_empty_only_upstream_checkpoint_preserves_ordinary_frontier(
    operator, route, bound, empty_form,
):
    key = (operator, route, bound)
    gates = [
        Probe(*key, 15, 10),
        Probe(*key, 15, 30),
    ]
    probes = [
        Probe(*key, 13, 9),
        Probe(*key, 13, 29),
        Probe(*key, 16, 10.3),
        Probe(*key, 16, 14),
        Probe(*key, 16, 30.3),
    ]
    baseline = _plan_gate_associations(probes, gates, {key})
    observed = None
    if empty_form == "sentinel":
        probes.append(Probe(*key, 14, None, kind=EtaKind.UNAVAILABLE))
    else:
        observed = {key: {13, 14, 16}}

    guarded = _plan_gate_associations(
        probes, gates, {key}, observed_checkpoint_indices=observed,
    )

    assert guarded == baseline


@pytest.mark.parametrize(
    ("operator", "route", "bound"),
    [
        ("KMB", "91", "inbound"),
        ("KMB", "91P", "inbound"),
        ("GMB", "11B", "seq-1"),
    ],
)
@pytest.mark.parametrize("reverse", [False, True])
def test_empty_sentinel_never_discards_active_checkpoint_siblings(
    operator, route, bound, reverse,
):
    snapshot, template_gates, _owned, _positions, _brackets = (
        _atomic_gate_frontier_fixture(40, reverse=reverse)
    )
    key = (operator, route, bound)
    probes = []
    gates = []
    for source, destination in ((snapshot.rows, probes), (template_gates, gates)):
        for original in source:
            row = copy(original)
            row.operator, row.route, row.bound = key
            destination.append(row)
    baseline = _plan_gate_associations(probes, gates, {key})
    sentinel = copy(next(row for row in probes if row.index == 32))
    sentinel.minutes = sentinel.signed_minutes = None
    sentinel.arrival_at = None
    sentinel.kind = EtaKind.UNAVAILABLE
    probes.append(sentinel)

    guarded = _plan_gate_associations(probes, gates, {key})
    active = {
        index for index, row in enumerate(probes)
        if row.index == 32 and row.minutes is not None
    }
    classified = set(guarded.gate_assignment) | set(guarded.passed_probe_rows)

    assert active <= classified
    assert len(probes) - 1 not in classified
    if key not in {("KMB", "91", "inbound"), ("KMB", "91M", "inbound")}:
        assert guarded == baseline


@pytest.mark.parametrize(
    "break_kind",
    ["revision", "continuity", "order", "arrival_chronology", "signed_chronology"],
)
def test_atomic_frontier_chain_cannot_resume_after_a_later_checkpoint_break(monkeypatch, break_kind):
    snapshot, gates, _owned, _positions, _brackets = _atomic_gate_frontier_fixture(25)
    probes = list(snapshot.rows)
    current = sorted((row for row in probes if row.index == 32), key=lambda row: row.minutes)
    if break_kind == "revision":
        for row in current:
            row.refresh_generation += 1
    elif break_kind in {"continuity", "order"}:
        row = current[0]
        # 24.4 retains the passed rank but exceeds local travel tolerance;
        # crossing the middle root additionally violates the rank invariant.
        minutes = 24.4 if break_kind == "continuity" else current[1].minutes + 1
        row.arrival_at += timedelta(minutes=minutes - row.minutes)
        row.minutes = row.signed_minutes = minutes
    else:
        previous = min(
            (row for row in probes if row.index == 31), key=lambda row: row.minutes,
        )
        row = current[0]
        if break_kind == "arrival_chronology":
            row.arrival_at = previous.arrival_at - timedelta(seconds=1)
        else:
            row.signed_minutes = previous.signed_minutes - 0.1
            row.minutes = max(0, row.signed_minutes)
    checks = []
    original = positions_module._atomic_kmb_frontier_certificate

    def record(*args, **kwargs):
        result = original(*args, **kwargs)
        checks.append((kwargs["checkpoint"], kwargs["carried"], result))
        return result

    monkeypatch.setattr(positions_module, "_atomic_kmb_frontier_certificate", record)
    _plan_gate_associations(probes, gates, {("KMB", "91", "inbound")})
    assert (31, False, True) in checks
    assert (32, True, False) in checks
    assert not any(checkpoint > 32 for checkpoint, _carried, _result in checks)


@pytest.mark.parametrize("operator,route", [("GMB", "11B"), ("KMB", "91P")])
def test_atomic_frontier_leaves_other_routes_and_11b_directions_isolated(monkeypatch, operator, route):
    snapshot, template_gates, _owned, _positions, _brackets = _atomic_gate_frontier_fixture(25)
    probes = []
    gates = []
    keys = set()
    for direction in ("seq-1", "seq-2"):
        keys.add((operator, route, direction))
        for source, destination in ((snapshot.rows, probes), (template_gates, gates)):
            for original in source:
                row = copy(original)
                row.operator, row.route, row.bound = operator, route, direction
                destination.append(row)
    enabled = _plan_gate_associations(probes, gates, keys)
    monkeypatch.setattr(positions_module, "_atomic_kmb_frontier_certificate",
                        lambda *_args, **_kwargs: False)
    ordinary = _plan_gate_associations(probes, gates, keys)
    assert enabled == ordinary
    assert enabled.gate_assignment
    assert all(probes[source].bound == gates[root].bound
               for source, root in enabled.gate_assignment.items())


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


def test_verified_gate_probe_reunites_exact_live_11s_sparse_ladder():
    """A missing gate-feed frame must not split one gate-probed journey."""
    line = _line("GMB", "11S", "seq-1", stop_count=19)
    key = ("GMB", "11S", "seq-1")
    rows = [
        Probe(
            *key,
            0,
            None,
            cache_age_seconds=7.4,
            refresh_generation=2504,
        ),
        Probe(
            *key,
            7,
            10.33589975,
            cache_age_seconds=7.688,
            signed_minutes=10.33589975,
            refresh_generation=2500,
            arrival_at=datetime.fromisoformat("2026-09-09T03:32:07.886+08:00"),
        ),
        Probe(
            *key,
            9,
            14.7188443,
            cache_age_seconds=8.063,
            signed_minutes=14.7188443,
            refresh_generation=2498,
            arrival_at=datetime.fromisoformat("2026-09-09T03:36:30.402+08:00"),
        ),
        Probe(
            *key,
            18,
            24.322681583333335,
            cache_age_seconds=7.86,
            signed_minutes=24.322681583333335,
            refresh_generation=2499,
            arrival_at=datetime.fromisoformat("2026-09-09T03:46:06.887+08:00"),
        ),
    ]

    estimates = estimate_bus_positions(
        rows,
        [line],
        observed_checkpoint_indices={key: {0, 7, 9, 18}},
        verified_gate_indices={key: 7},
    )

    assert len(estimates) == 1
    # The fallback proves identity only. Physical placement projects the
    # first-present ETA across its certified (0, 7) bracket.
    assert estimates[0].position == pytest.approx(1.832050125)
    assert estimates[0].bracket == (0.0, 7.0)
    assert estimates[0].boundary_revision == (2504, 2500)
    assert estimates[0].source_observations == {
        ("probe", 1),
        ("probe", 2),
        ("probe", 3),
    }

    line.stops[7] = Stop("20013011", "HKUST South", 22.333360, 114.267)
    audit = audit_marker_positions(rows, (), estimates, [line])
    assert audit["ok"]
    assert audit["issues"] == []


def test_verified_gate_probe_preserves_two_ordered_occurrences():
    line = _line("GMB", "11S", "seq-1", stop_count=19)
    key = ("GMB", "11S", "seq-1")
    rows = [
        Probe(*key, 0, None, cache_age_seconds=4, refresh_generation=1),
        Probe(*key, 7, 4, cache_age_seconds=4, refresh_generation=2),
        Probe(*key, 7, 10, cache_age_seconds=4, refresh_generation=2),
        Probe(*key, 9, 8, cache_age_seconds=4, refresh_generation=3),
        Probe(*key, 9, 14, cache_age_seconds=4, refresh_generation=3),
        Probe(*key, 18, 26, cache_age_seconds=4, refresh_generation=4),
        Probe(*key, 18, 32, cache_age_seconds=4, refresh_generation=4),
    ]

    estimates = estimate_bus_positions(
        rows,
        [line],
        observed_checkpoint_indices={key: {0, 7, 9, 18}},
        verified_gate_indices={key: 7},
    )

    assert len(estimates) == 2
    assert sorted(
        observation
        for estimate in estimates
        for observation in estimate.source_observations
        if observation in {("probe", 1), ("probe", 2)}
    ) == [("probe", 1), ("probe", 2)]
    assert all(len(estimate.source_observations) == 3 for estimate in estimates)


def test_verified_gate_probe_leaves_unmatched_passed_vehicle_separate():
    line = _line("GMB", "11S", "seq-1", stop_count=19)
    key = ("GMB", "11S", "seq-1")
    rows = [
        Probe(*key, 0, None, cache_age_seconds=4, refresh_generation=1),
        Probe(*key, 7, 10, cache_age_seconds=4, refresh_generation=2),
        Probe(*key, 18, 2, cache_age_seconds=4, refresh_generation=3),
    ]

    estimates = estimate_bus_positions(
        rows,
        [line],
        observed_checkpoint_indices={key: {0, 7, 18}},
        verified_gate_indices={key: 7},
    )

    assert sorted(estimate.position for estimate in estimates) == [2.0, 17.0]
    assert {estimate.source_observations for estimate in estimates} == {
        frozenset({("probe", 1)}),
        frozenset({("probe", 2)}),
    }


def test_live_terminal_singleton_prioritizes_eta_implied_interior_stops():
    """The exact 11S frame-5 shape should need one priority generation."""
    line = _line("GMB", "11S", "seq-1", stop_count=19)
    key = ("GMB", "11S", "seq-1")
    rows = [
        Probe(
            *key, 0, 0.0, kind=EtaKind.SCHEDULED,
            cache_age_seconds=7.1, signed_minutes=-0.149,
            refresh_generation=53,
            arrival_at=datetime.fromisoformat("2026-09-09T04:11:23.709+08:00"),
        ),
        Probe(
            *key, 9, 15.001336, kind=EtaKind.SCHEDULED,
            cache_age_seconds=45.2, refresh_generation=22,
            arrival_at=datetime.fromisoformat("2026-09-09T04:25:54.791+08:00"),
        ),
        Probe(
            *key, 18, 5.874677816666666, kind=EtaKind.REALTIME,
            cache_age_seconds=9.0, refresh_generation=45,
            arrival_at=datetime.fromisoformat("2026-09-09T04:17:23.463+08:00"),
        ),
        Probe(
            *key, 18, 24.78519448333333, kind=EtaKind.SCHEDULED,
            cache_age_seconds=9.0, refresh_generation=45,
            arrival_at=datetime.fromisoformat("2026-09-09T04:36:17.094+08:00"),
        ),
    ]
    gate_rows = [
        AuthoritativeProbe(*key, 7, 11.0, kind=EtaKind.SCHEDULED),
    ]

    estimates = estimate_bus_positions(
        rows,
        [line],
        authoritative_etas=gate_rows,
        observed_checkpoint_indices={key: {0, 9, 18}},
    )

    assert len(estimates) == 2
    live = next(
        estimate for estimate in estimates
        if estimate.source_observations == {("probe", 2)}
    )
    # Search hints do not turn the heuristic projection into motion evidence.
    assert live.position == pytest.approx(15.063)
    assert live.bracket is None
    assert live.priority_indices == frozenset({18})
    assert live.exploratory_indices == frozenset({15, 16})


@pytest.mark.parametrize("rebuild", [False, True])
def test_sparse_empty_future_bracket_projects_across_its_full_span(rebuild):
    line = _line("GMB", "11S", "seq-1", stop_count=19)
    key = ("GMB", "11S", "seq-1")
    arrival = datetime(2026, 9, 11, tzinfo=UTC) + timedelta(minutes=8.2603563)
    rows = [
        Probe(*key, 9, None, cache_age_seconds=0, refresh_generation=10),
        Probe(*key, 18, 8.2603563, kind=EtaKind.REALTIME,
              cache_age_seconds=0, refresh_generation=11, arrival_at=arrival),
    ]
    estimates = estimate_bus_positions(
        rows, [line], observed_checkpoint_indices={key: {9, 18}},
    )
    assert len(estimates) == 1
    estimate = estimates[0]
    if rebuild:
        estimate = rebuild_estimate_from_probe_fragments(estimate, [], rows, [line])
        assert estimate is not estimates[0]
    assert estimate.position == pytest.approx(13.86982185)
    assert estimate.bracket == (9.0, 18.0)
    assert estimate.boundary_revision == (10, 11)
    assert estimate.boundary_age_seconds == 0
    assert estimate.position_authoritative is not False
    assert estimate.source_indices == frozenset({18})
    assert estimate.source_observations == frozenset({("probe", 1)})
    assert estimate.checkpoint_evidence == ((18, arrival.timestamp(), 11),)
    assert estimate.priority_indices == frozenset({18})
    assert estimate.exploratory_indices == (
        frozenset() if rebuild else frozenset({13, 14})
    )

    # A newly observed upper rung consistent with the same two-minutes-per-stop
    # projection narrows the bracket without correcting the position backward.
    narrowed_rows = [
        Probe(*key, 9, None, cache_age_seconds=0, refresh_generation=12),
        Probe(*key, 14, 0.2603563, kind=EtaKind.REALTIME,
              cache_age_seconds=0, refresh_generation=13,
              arrival_at=arrival - timedelta(minutes=8)),
        rows[1],
    ]
    narrowed = estimate_bus_positions(
        narrowed_rows, [line], observed_checkpoint_indices={key: {9, 14, 18}},
    )
    assert len(narrowed) == 1
    assert narrowed[0].bracket == (9.0, 14.0)
    assert narrowed[0].position == pytest.approx(estimate.position)
    assert narrowed[0].position_authoritative is not False
    assert narrowed[0].boundary_revision == (12, 13)
    assert narrowed[0].source_indices == frozenset({14, 18})
    assert narrowed[0].priority_indices == frozenset({14, 18})
    assert narrowed[0].exploratory_indices == frozenset({13})


def test_live_singleton_priority_uses_refreshed_frame_9_eta():
    line = _line("GMB", "11S", "seq-1", stop_count=19)
    key = ("GMB", "11S", "seq-1")
    rows = [
        Probe(*key, 14, None, cache_age_seconds=7.1, refresh_generation=87),
        Probe(
            *key, 18, 4.9704192166666665,
            cache_age_seconds=7.4, refresh_generation=86,
            arrival_at=datetime.fromisoformat("2026-09-09T04:17:10.889+08:00"),
        ),
    ]

    estimates = estimate_bus_positions(
        rows,
        [line],
        observed_checkpoint_indices={key: {14, 18}},
    )

    assert len(estimates) == 1
    assert estimates[0].position == pytest.approx(15.514790391666667)
    assert estimates[0].bracket == (14.0, 18.0)
    assert estimates[0].priority_indices == frozenset({18})
    assert estimates[0].exploratory_indices == frozenset({15, 16})


@pytest.mark.parametrize(
    ("kind", "age", "revision", "minutes"),
    [
        (EtaKind.SCHEDULED, 1.0, 1, 5.0),
        (EtaKind.UNAVAILABLE, 1.0, 1, 5.0),
        (EtaKind.REALTIME, 60.0, 1, 5.0),
        (EtaKind.REALTIME, float("nan"), 1, 5.0),
        (EtaKind.REALTIME, True, 1, 5.0),
        (EtaKind.REALTIME, 1.0, 0, 5.0),
        (EtaKind.REALTIME, 1.0, 1, float("nan")),
        (EtaKind.REALTIME, 1.0, 1, True),
    ],
)
def test_eta_guided_priority_fails_closed_for_noncurrent_rows(
    kind, age, revision, minutes
):
    row = Probe(
        "GMB", "11S", "seq-1", 18, minutes, kind=kind,
        cache_age_seconds=age, refresh_generation=revision,
    )
    assert _eta_guided_priority_indices(row, (9.0, 18.0), 19) == frozenset()


@pytest.mark.parametrize(
    ("minutes", "bracket", "expected"),
    [
        (6.0, (9.0, 18.0), frozenset({15})),
        (5.8, (15.0, 18.0), frozenset({16})),
        (20.0, (9.0, 18.0), frozenset()),
        (2.0, (17.0, 18.0), frozenset()),
    ],
)
def test_eta_guided_priority_is_clamped_to_strict_bracket_interior(
    minutes, bracket, expected
):
    row = Probe(
        "GMB", "11S", "seq-1", 18, minutes,
        cache_age_seconds=1, refresh_generation=1,
    )
    assert _eta_guided_priority_indices(row, bracket, 19) == expected


def test_eta_guided_priority_deduplicates_equal_terminal_occurrences():
    line = _line("GMB", "11S", "seq-1", stop_count=19)
    key = ("GMB", "11S", "seq-1")
    rows = [
        Probe(*key, 9, None, cache_age_seconds=1, refresh_generation=1),
        Probe(*key, 18, 5.0, cache_age_seconds=1, refresh_generation=2),
        Probe(*key, 18, 5.0, cache_age_seconds=1, refresh_generation=2),
    ]

    estimates = estimate_bus_positions(
        rows,
        [line],
        observed_checkpoint_indices={key: {9, 18}},
    )

    assert len(estimates) == 2
    assert {estimate.source_observations for estimate in estimates} == {
        frozenset({("probe", 1)}),
        frozenset({("probe", 2)}),
    }
    assert all(
        estimate.priority_indices == frozenset({18})
        and estimate.exploratory_indices == frozenset({15, 16})
        for estimate in estimates
    )


def test_multirow_realtime_ladder_uses_its_unique_fresh_upper_as_search_hint():
    line = _line("GMB", "11S", "seq-1", stop_count=19)
    key = ("GMB", "11S", "seq-1")
    rows = [
        Probe(*key, 9, None, cache_age_seconds=1, refresh_generation=1),
        Probe(*key, 17, 3.0, cache_age_seconds=1, refresh_generation=2),
        Probe(*key, 18, 5.0, cache_age_seconds=1, refresh_generation=3),
    ]

    estimates = estimate_bus_positions(
        rows,
        [line],
        observed_checkpoint_indices={key: {9, 17, 18}},
    )

    assert len(estimates) == 1
    assert estimates[0].priority_indices == frozenset({17, 18})
    assert estimates[0].exploratory_indices == frozenset({15, 16})


@pytest.mark.parametrize("gate_age", [60.0, None, float("nan")])
def test_stale_or_invalid_gate_probe_cannot_reunite_sparse_fragments(gate_age):
    line = _line("GMB", "11S", "seq-1", stop_count=19)
    key = ("GMB", "11S", "seq-1")
    rows = [
        Probe(*key, 0, None, cache_age_seconds=4, refresh_generation=1),
        Probe(*key, 7, 10, cache_age_seconds=gate_age, refresh_generation=2),
        Probe(*key, 9, 14, cache_age_seconds=4, refresh_generation=3),
        Probe(*key, 18, 24, cache_age_seconds=4, refresh_generation=4),
    ]

    estimates = estimate_bus_positions(
        rows,
        [line],
        observed_checkpoint_indices={key: {0, 7, 9, 18}},
        verified_gate_indices={key: 7},
    )

    assert len(estimates) == 2


def test_negative_gate_probe_cannot_make_filtered_rows_position_evidence():
    line = _line("GMB", "11S", "seq-1", stop_count=19)
    key = ("GMB", "11S", "seq-1")
    rows = [
        Probe(*key, 0, None, cache_age_seconds=4, refresh_generation=1),
        Probe(*key, 7, 20, cache_age_seconds=4, refresh_generation=2),
        Probe(*key, 9, 24, cache_age_seconds=4, refresh_generation=3),
        Probe(*key, 18, 28, cache_age_seconds=4, refresh_generation=4),
    ]

    estimates = estimate_bus_positions(
        rows,
        [line],
        observed_checkpoint_indices={key: {0, 7, 9, 18}},
        verified_gate_indices={key: 7},
    )

    assert len(estimates) == 1
    assert estimates[0].position == pytest.approx(4.0)
    assert estimates[0].bracket is None
    assert estimates[0].position_authoritative is False
    assert estimates[0].exploratory_indices == frozenset({4})
    assert estimates[0].source_observations == {("probe", 3)}


@pytest.mark.parametrize("origin_minutes", [5, 0.25, 0.01])
def test_verified_gate_probe_track_obeys_future_origin_suppression(origin_minutes):
    line = _line("GMB", "11S", "seq-1", stop_count=19)
    key = ("GMB", "11S", "seq-1")
    rows = [
        Probe(*key, 0, origin_minutes, cache_age_seconds=4, refresh_generation=1),
        Probe(*key, 7, 12, cache_age_seconds=4, refresh_generation=2),
        Probe(*key, 9, 16, cache_age_seconds=4, refresh_generation=3),
    ]

    assert estimate_bus_positions(
        rows,
        [line],
        observed_checkpoint_indices={key: {0, 7, 9}},
        verified_gate_indices={key: 7},
    ) == []


def test_verified_gate_probe_keeps_final_zero_first_future_placement():
    line = _line("GMB", "11S", "seq-1", stop_count=19)
    key = ("GMB", "11S", "seq-1")
    rows = [
        Probe(*key, 0, None, cache_age_seconds=4, refresh_generation=1),
        Probe(
            *key,
            7,
            0,
            cache_age_seconds=4,
            signed_minutes=-1.0,
            refresh_generation=2,
        ),
        Probe(
            *key,
            8,
            0,
            cache_age_seconds=4,
            signed_minutes=-0.25,
            refresh_generation=3,
        ),
        Probe(
            *key,
            9,
            1,
            cache_age_seconds=4,
            signed_minutes=1.0,
            refresh_generation=4,
        ),
    ]

    estimates = estimate_bus_positions(
        rows,
        [line],
        observed_checkpoint_indices={key: {0, 7, 8, 9}},
        verified_gate_indices={key: 7},
    )

    assert len(estimates) == 1
    assert estimates[0].bracket == (8.0, 9.0)
    assert estimates[0].position == pytest.approx(8.2)
    assert estimates[0].bracket_eta_offsets == (-0.25, 1.0)


def test_authoritative_gate_rows_disable_probe_gate_fallback():
    line = _line("GMB", "11S", "seq-1", stop_count=19)
    key = ("GMB", "11S", "seq-1")
    probes = [
        Probe(*key, 7, 10, cache_age_seconds=4, refresh_generation=2),
        Probe(*key, 18, 24, cache_age_seconds=4, refresh_generation=3),
    ]
    gates = [AuthoritativeProbe(*key, 7, 10)]

    estimates = estimate_bus_positions(
        probes,
        [line],
        authoritative_etas=gates,
        verified_gate_indices={key: 7},
    )

    assert len(estimates) == 1
    assert ("gate", 0) in estimates[0].source_observations
    assert ("probe", 0) not in estimates[0].source_observations


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


def test_future_origin_suppresses_only_its_matched_gate_marker():
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
    # The near-zero future origin belongs to the ten-minute gate cohort.
    # It remains hidden until due, while the unrelated seven-minute bus stays.
    assert [round(estimate.position, 3) for estimate in estimates] == [2.5]
    assert estimates[0].source_observations == frozenset({("gate", 1)})
    assert not any(("probe", 1) in estimate.source_observations for estimate in estimates)
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


@pytest.mark.parametrize("origin_minutes", [3, 0.25, 0.01])
def test_future_terminus_vetoes_same_cohort_downstream_projection(origin_minutes):
    """A downstream positive rung cannot publish a predeparture bus."""
    line = _line(stop_count=6)
    rows = [
        Probe("KMB", "X", "outbound", 0, origin_minutes),
        Probe("KMB", "X", "outbound", 2, 5),
        # The old maximum-ladder anchor was +0.5 despite the future origin.
        Probe("KMB", "X", "outbound", 4, 7),
    ]

    assert estimate_bus_positions(rows, [line]) == []


@pytest.mark.parametrize("origin_minutes", [3, 0.25, 0.01])
def test_source_rebuild_cannot_resurrect_future_terminus_cohort(origin_minutes):
    line = _line(stop_count=6)
    observed = datetime(2026, 1, 1, tzinfo=UTC)
    rows = [
        Probe("KMB", "X", "outbound", index, minutes,
              cache_age_seconds=0, refresh_generation=1,
              arrival_at=observed + timedelta(minutes=minutes))
        for index, minutes in ((0, origin_minutes), (2, 5), (4, 7))
    ]
    template = BusEstimate(
        "X destination", 22.3, 114.2, Operator.KMB, 0.0,
        route="X", bound="outbound", operator_code="KMB", position=0.5,
    )

    assert rebuild_estimate_from_probe_sources(
        template, (0, 1, 2), rows, [line],
    ) is None


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


def _kmb91m_singleton_boundary_case():
    key = ("KMB", "91M", "outbound")
    singleton = frozenset({("probe", 0)})
    trailing = frozenset({("probe", 1), ("probe", 2)})
    middle = frozenset({("probe", 3), ("probe", 4)})
    records = [
        (key, 22.637016375, False, singleton),
        (key, 8.978683041666667, False, trailing),
        (key, 18.137016375, False, middle),
    ]
    evidence = {
        ("probe", 0): (28, 22.637016375),
        ("probe", 1): (28, 6.295349708333333),
        ("probe", 2): (28, 6.295349708333333),
        ("probe", 3): (28, 15.453683041666668),
        ("probe", 4): (28, 15.453683041666668),
    }
    return key, records, evidence, singleton, trailing, middle


def _frame60_kmb91m_estimator_fixture():
    """Exact distilled frame-60 rows; all three separator components are live."""
    key = ("KMB", "91M", "outbound")
    line = _line(*key, stop_count=29)
    collected_at = datetime(2026, 1, 1, 13, 0, tzinfo=UTC)
    scheduled = EtaKind.SCHEDULED
    triples = {
        0: ((4.109300583333333, scheduled), (19.109300583333333, scheduled), (34.10930058333334, scheduled)),
        12: ((8.77596725, EtaKind.REALTIME), (22.709300583333334, scheduled), (37.70930058333333, scheduled)),
        14: ((13.959300583333334, EtaKind.REALTIME), (27.859300583333333, scheduled), (42.85930058333334, scheduled)),
        28: ((10.72596725, EtaKind.REALTIME), (25.092633916666667, EtaKind.REALTIME), (43.409300583333334, EtaKind.REALTIME)),
        16: ((0.0, EtaKind.REALTIME), (16.92596725, EtaKind.REALTIME), (30.809300583333332, scheduled)),
        19: ((1.9093005833333334, EtaKind.REALTIME), (20.392633916666668, EtaKind.REALTIME), (34.259300583333335, scheduled)),
        22: ((7.72596725, EtaKind.REALTIME), (26.042633916666666, EtaKind.REALTIME), (39.99263391666667, scheduled)),
        23: ((10.37596725, EtaKind.REALTIME), (28.69263391666667, EtaKind.REALTIME), (42.64263391666667, scheduled)),
    }
    rows = []
    for index in (0, 12, 14, 28, 16, 19, 22, 23):
        for minutes, kind in triples[index]:
            signed = -1.67403275 if index == 16 and minutes == 0.0 else minutes
            rows.append(Probe(*key, index, minutes, kind=kind,
                              signed_minutes=signed,
                              arrival_at=collected_at - timedelta(seconds=9.547)
                              + timedelta(minutes=signed),
                              cache_age_seconds=9.547, refresh_generation=648))
    gates = [
        AuthoritativeProbe(*key, 12, 9, kind=EtaKind.REALTIME,
                           cache_age_seconds=9.547, refresh_generation=648),
        AuthoritativeProbe(*key, 12, 23, kind=scheduled,
                           cache_age_seconds=9.547, refresh_generation=648),
        AuthoritativeProbe(*key, 12, 38, kind=scheduled,
                           cache_age_seconds=9.547, refresh_generation=648),
    ]
    return key, line, rows, gates, collected_at


def test_singleton_anchor_keeps_interior_eta_when_headway_span_is_infeasible():
    key, records, evidence, singleton, trailing, middle = _kmb91m_singleton_boundary_case()
    corrected = _separate_common_stop_departures(
        records, evidence, {key: 28.0}, {key: 12.001}
    )
    assert corrected[singleton] == 22.637016375
    assert corrected[middle] == 18.137016375
    assert corrected[trailing] == 12.001


def test_singleton_boundary_guard_is_input_order_independent():
    key, records, evidence, singleton, _trailing, _middle = _kmb91m_singleton_boundary_case()
    forward = _separate_common_stop_departures(
        records, evidence, {key: 28.0}, {key: 12.001}
    )
    reverse = _separate_common_stop_departures(
        list(reversed(records)), evidence, {key: 28.0}, {key: 12.001}
    )
    assert forward == reverse
    assert forward[singleton] == 22.637016375


def test_singleton_boundary_guard_does_not_accept_reversed_baselines():
    key, records, evidence, singleton, trailing, middle = _kmb91m_singleton_boundary_case()
    reversed_baselines = [
        records[0],
        (key, 18.0, False, trailing),
        (key, 12.0, False, middle),
    ]
    corrected = _separate_common_stop_departures(
        reversed_baselines, evidence, {key: 28.0}, {key: 12.001}
    )
    assert corrected[singleton] != 22.637016375
    assert corrected[middle] > corrected[trailing]
    assert corrected[singleton] <= 28.0


@pytest.mark.parametrize(
    ("endpoint", "delta_ulps", "expected_projection"),
    [("lower", 0, True), ("lower", -1, True), ("lower", -32, False),
     ("upper", 0, True), ("upper", 1, True), ("upper", 16, False)],
)
def test_singleton_boundary_tolerance_selects_projection_or_bounded_baselines(
    endpoint, delta_ulps, expected_projection,
):
    key = ("KMB", "91M", "outbound")
    singleton = frozenset({("probe", 0)})
    ahead = frozenset({("probe", 1), ("probe", 3)})
    behind = frozenset({("probe", 2), ("probe", 4)})
    if endpoint == "lower":
        singleton_position = 5.0
        baselines = [(ahead, 12.0), (singleton, singleton_position), (behind, 1.0)]
    else:
        singleton_position = 15.0
        baselines = [(ahead, 17.0), (singleton, singleton_position), (behind, 1.0)]
    singleton_position += delta_ulps * math.ulp(singleton_position)
    records = [(key, position, False, sources) for sources, position in baselines]
    evidence = {
        ("probe", 0): (28, 5.0),
        ("probe", 1): (28, 10.0),
        ("probe", 2): (28, 0.0),
        ("probe", 3): (28, 10.0),
        ("probe", 4): (28, 0.0),
    }
    records = [
        (route, singleton_position if sources == singleton else position, auth, sources)
        for route, position, auth, sources in records
    ]
    corrected = _separate_common_stop_departures(
        records, evidence, {key: 20.0}, {key: 0.0}
    )
    if endpoint == "lower":
        assert corrected[ahead] == pytest.approx(10.0 if expected_projection else 12.0)
        assert corrected[behind] == pytest.approx(0.0 if expected_projection else 1.0)
    else:
        assert corrected[ahead] == pytest.approx(20.0 if expected_projection else 17.0)
        assert corrected[behind] == pytest.approx(10.0 if expected_projection else 1.0)


def test_full_span_zero_origin_nextafter_stays_on_projection_path():
    key = ("KMB", "91M", "outbound")
    ahead = frozenset({("probe", 1), ("probe", 3)})
    singleton = frozenset({("probe", 0)})
    behind = frozenset({("probe", 2), ("probe", 4)})
    singleton_baseline = math.nextafter(22.0, math.inf)
    corrected = _separate_common_stop_departures(
        [
            (key, 30.0, False, ahead),
            (key, singleton_baseline, False, singleton),
            (key, 1.0, False, behind),
        ],
        {
            ("probe", 0): (28, 22.0),
            ("probe", 1): (28, 30.0),
            ("probe", 2): (28, 0.0),
            ("probe", 3): (28, 30.0),
            ("probe", 4): (28, 0.0),
        },
        {key: 30.0},
        {key: 0.0},
    )
    assert corrected[ahead] == pytest.approx(30.0)
    assert corrected[singleton] == pytest.approx(singleton_baseline)
    assert corrected[behind] == pytest.approx(0.0)


def test_frame60_trace_keeps_kmb91m_terminal_singleton_at_eta_position():
    key, line, rows, gates, _collected_at = _frame60_kmb91m_estimator_fixture()
    estimates = estimate_bus_positions(
        rows,
        [line],
        authoritative_etas=gates,
        observed_checkpoint_indices={key: {0, 12, 14, 16, 19, 22, 23, 28}},
        verified_gate_indices={key: 12},
    )
    assert len(estimates) == 3
    terminal = next(estimate for estimate in estimates if ("probe", 9) in estimate.source_observations)
    assert terminal.position == pytest.approx(28 - rows[9].minutes / 2, abs=1e-9)
    assert terminal.position != 28.0
    assert terminal.position_authoritative is not True
    assert terminal.source_observations == frozenset({("probe", 9)})


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


@pytest.mark.parametrize("origin_minutes", [5, 0.25, 0.01])
def test_future_origin_vetoes_nonnegative_coarse_gate_position(origin_minutes):
    """A downstream projection cannot launch a journey before stop-zero ETA."""
    line = _line("GMB", "11", "seq-1", stop_count=20)
    estimates = estimate_bus_positions(
        [
            Probe("GMB", "11", "seq-1", 0, origin_minutes, EtaKind.SCHEDULED),
            Probe("GMB", "11", "seq-1", 1, 7, EtaKind.SCHEDULED),
        ],
        [line],
        authoritative_etas=[
            # 6 - 11/2 = 0.5: the old coarse gate rule called this departed,
            # despite the same ETA instance still being in advance of its
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


def test_gate_downstream_clock_skew_keeps_live_terminal_ladder_attached():
    """A fractional downstream countdown must survive the integral gate edge."""
    line = _line("KMB", "91M", "outbound", stop_count=29)
    probes = [
        Probe("KMB", "91M", "outbound", index, minutes)
        for index, minutes in (
            (13, 19.084),
            (14, 20.384),
            (20, 27.501),
            (21, 29.951),
            (28, 31.801),
        )
    ]
    estimates = estimate_bus_positions(
        probes,
        [line],
        authoritative_etas=[
            AuthoritativeProbe("KMB", "91M", "outbound", 12, 15)
        ],
    )

    assert len(estimates) == 1
    assert estimates[0].source_observations == frozenset(
        {("gate", 0), *(('probe', index) for index in range(5))}
    )


def test_frame27_91m_mixed_population_preserves_three_vehicle_identity_ladders():
    line = _line("KMB", "91M", "outbound", stop_count=29)
    values = [
        (3, 0.0),
        (4, 0.4341865833),
        (5, 2.3675199167),
        (13, 0.7175199167),
        (13, 19.0841865833),
        (14, 2.0175199167),
        (14, 20.3841865833),
        (20, 9.1341865833),
        (20, 27.50085325),
        (21, 0.0),
        (21, 11.5841865833),
        (21, 29.95085325),
        (28, 9.9675199167),
        (28, 19.8175199167),
        (28, 31.80085325),
    ]
    estimates = estimate_bus_positions(
        [Probe("KMB", "91M", "outbound", index, minutes) for index, minutes in values],
        [line],
        authoritative_etas=[
            AuthoritativeProbe("KMB", "91M", "outbound", 12, 15)
        ],
    )

    assert len(estimates) == 3
    probe_tokens = {
        ("probe", index)
        for index in range(len(values))
    }
    owned_tokens = [
        token
        for estimate in estimates
        for token in estimate.source_observations
        if token[0] == "probe"
    ]
    assert set(owned_tokens) == probe_tokens
    assert len(owned_tokens) == len(set(owned_tokens)) == len(values)
    gate_backed = next(
        estimate
        for estimate in estimates
        if ("gate", 0) in estimate.source_observations
    )
    assert ("probe", 14) in gate_backed.source_observations
    terminal_estimates = [
        estimate
        for estimate in estimates
        if ("probe", 12) in estimate.source_observations
        or ("probe", 13) in estimate.source_observations
    ]
    assert len(terminal_estimates) == 2
    assert all(estimate is not gate_backed for estimate in terminal_estimates)
    assert terminal_estimates[0].source_observations.isdisjoint(
        terminal_estimates[1].source_observations
    )


def test_equal_raw_terminal_occurrence_remains_a_second_vehicle():
    """Repeated terminal evidence remains a distinct second vehicle."""
    line = _line("KMB", "91M", "outbound", stop_count=29)
    probes = [
        Probe("KMB", "91M", "outbound", 20, 27.501),
        Probe("KMB", "91M", "outbound", 28, 31.801),
        Probe("KMB", "91M", "outbound", 28, 31.801),
    ]
    estimates = estimate_bus_positions(
        probes,
        [line],
        authoritative_etas=[
            AuthoritativeProbe("KMB", "91M", "outbound", 12, 15)
        ],
    )

    assert len(estimates) == 2
    assert any(
        estimate.source_observations
        == frozenset({("gate", 0), ("probe", 0), ("probe", 1)})
        for estimate in estimates
    )
    assert any(
        estimate.source_observations == frozenset({("probe", 2)})
        for estimate in estimates
    )


def test_fresh_downstream_gate_matching_keeps_existing_drift_boundary():
    gate_rows = [
        (0, AuthoritativeProbe("KMB", "91M", "outbound", 12, 15)),
        (1, AuthoritativeProbe("KMB", "91M", "outbound", 12, 35)),
    ]
    probe_rows = [
        (0, Probe("KMB", "91M", "outbound", 28, 31)),
        (1, Probe("KMB", "91M", "outbound", 28, 51)),
        (2, Probe("KMB", "91M", "outbound", 28, 71)),
    ]

    assert _align_gate_arrivals(
        gate_rows, probe_rows, gate_index=12, checkpoint=28
    ) == [(1, 0), (2, 1)]


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


def test_atomic_kmb_sparse_fast_ladder_does_not_split_terminal_marker():
    """One route-response ETA chain remains one bus across a wide stop gap."""
    line = _line("KMB", "91", "outbound", stop_count=31)
    rows = [
        Probe(
            "KMB", "91", "outbound", index, minutes,
            cache_age_seconds=9.391,
            signed_minutes=signed_minutes,
            refresh_generation=305,
            arrival_at=datetime.fromisoformat(arrival_at),
        )
        for index, minutes, signed_minutes, arrival_at in (
            (17, 0.0, -1.90, "2026-09-08T23:46:30+08:00"),
            (18, 0.0, -0.55, "2026-09-08T23:47:51+08:00"),
            (30, 16.412, 16.412, "2026-09-09T00:04:39+08:00"),
        )
    ]

    estimates = estimate_bus_positions(
        rows,
        [line],
        observed_checkpoint_indices={
            ("KMB", "91", "outbound"): range(31),
        },
    )

    assert len(estimates) == 1
    assert estimates[0].source_observations == frozenset(
        {("probe", 0), ("probe", 1), ("probe", 2)}
    )
    # Preserve the established final-zero/first-future boundary. The distant
    # terminal proves identity but does not place a second bus at stop 29.
    assert estimates[0].bracket == (18.0, 30.0)
    assert 18.0 < estimates[0].position < 19.0
    assert estimates[0].priority_indices == frozenset({17, 18, 30})


def test_atomic_kmb_91m_sparse_terminal_ladder_remains_one_marker():
    """91M has a verified full line and uses the same atomic KMB response."""
    line = _line("KMB", "91M", "inbound", stop_count=28)
    rows = [
        Probe(
            "KMB", "91M", "inbound", index, minutes,
            cache_age_seconds=28.843,
            signed_minutes=signed_minutes,
            refresh_generation=976,
            arrival_at=datetime.fromisoformat(arrival_at),
        )
        for index, minutes, signed_minutes, arrival_at in (
            (17, 0.0, -0.858, "2026-09-09T00:38:07+08:00"),
            (18, 0.609, 0.609, "2026-09-09T00:39:35+08:00"),
            (27, 11.942, 11.942, "2026-09-09T00:50:55+08:00"),
        )
    ]

    estimates = estimate_bus_positions(
        rows,
        [line],
        observed_checkpoint_indices={
            ("KMB", "91M", "inbound"): range(28),
        },
    )

    assert len(estimates) == 1
    assert estimates[0].source_observations == frozenset(
        {("probe", 0), ("probe", 1), ("probe", 2)}
    )
    assert estimates[0].bracket == (17.0, 18.0)
    assert 17.0 < estimates[0].position < 18.0


def test_kmb_91_gate_does_not_cross_backward_arrival_into_passed_cohort():
    """Captured 91 rows must keep the gate bus ahead of three bunched buses."""
    line = _line("KMB", "91", "outbound", stop_count=31)
    rows = [
        Probe(
            "KMB", "91", "outbound", index, minutes,
            cache_age_seconds=0.125,
            signed_minutes=minutes,
            refresh_generation=1,
            arrival_at=datetime.fromisoformat(arrival_at),
        )
        for index, minutes, arrival_at in (
            (19, 9.214818566666667, "2026-09-16T17:06:03+08:00"),
            (20, 10.414818566666666, "2026-09-16T17:07:15+08:00"),
            (21, 0.0, "2026-09-16T16:55:28+08:00"),
            (21, 0.0, "2026-09-16T16:55:33+08:00"),
            (21, 11.4481519, "2026-09-16T17:08:17+08:00"),
            (22, 0.0, "2026-09-16T16:55:57+08:00"),
            (22, 0.0, "2026-09-16T16:56:07+08:00"),
            (22, 0.0, "2026-09-16T16:56:15+08:00"),
            (23, 0.0, "2026-09-16T16:56:33+08:00"),
            (23, 0.0, "2026-09-16T16:56:43+08:00"),
            (23, 0.0, "2026-09-16T16:56:50+08:00"),
            (24, 2.3648185666666666, "2026-09-16T16:59:12+08:00"),
            (24, 2.5314852333333335, "2026-09-16T16:59:22+08:00"),
            (24, 2.664818566666667, "2026-09-16T16:59:30+08:00"),
            (25, 4.414818566666667, "2026-09-16T17:01:15+08:00"),
            (25, 4.581485233333334, "2026-09-16T17:01:25+08:00"),
            (25, 4.781485233333333, "2026-09-16T17:01:37+08:00"),
            (26, 7.031485233333333, "2026-09-16T17:04:32+08:00"),
            (26, 7.214818566666667, "2026-09-16T17:04:43+08:00"),
            (26, 7.5481519, "2026-09-16T17:05:03+08:00"),
            (27, 10.781485233333333, "2026-09-16T17:08:17+08:00"),
            (27, 10.9481519, "2026-09-16T17:08:27+08:00"),
            (27, 11.298151899999999, "2026-09-16T17:08:48+08:00"),
            (28, 11.564818566666666, "2026-09-16T17:09:04+08:00"),
            (28, 11.7481519, "2026-09-16T17:09:15+08:00"),
            (28, 12.081485233333332, "2026-09-16T17:09:35+08:00"),
        )
    ]
    gate = AuthoritativeProbe(
        "KMB", "91", "outbound", 15, 0, EtaKind.REALTIME
    )

    estimates = estimate_bus_positions(
        rows,
        [line],
        authoritative_etas=[gate],
        observed_checkpoint_indices={
            ("KMB", "91", "outbound"): range(31),
        },
    )

    gate_estimate = next(
        estimate for estimate in estimates
        if ("gate", 0) in estimate.source_observations
    )
    downstream = [estimate for estimate in estimates if estimate.position > 15]
    assert len(estimates) == 4
    assert gate_estimate.position == 15
    gate_arrivals = [item[1] for item in gate_estimate.checkpoint_evidence]
    assert all(
        earlier < later
        for earlier, later in zip(gate_arrivals, gate_arrivals[1:], strict=False)
    )
    # The three exact stop-24 rows are a census of three genuinely bunched
    # buses.  Each stays owned by one separate downstream marker.
    stop_24_inputs = {11, 12, 13}
    assert {
        next(
            input_index for source, input_index in estimate.source_observations
            if source == "probe" and input_index in stop_24_inputs
        )
        for estimate in downstream
    } == stop_24_inputs
    assert all(("gate", 0) not in estimate.source_observations for estimate in downstream)


def test_kmb_91m_close_same_stop_arrivals_remain_separate():
    line = _line("KMB", "91M", "inbound", stop_count=28)
    rows = [
        Probe(
            "KMB", "91M", "inbound", 20, minutes,
            cache_age_seconds=0.2,
            signed_minutes=minutes,
            refresh_generation=8,
            arrival_at=datetime.fromisoformat(arrival_at),
        )
        for minutes, arrival_at in (
            (2.0, "2026-09-16T17:00:00+08:00"),
            (2.1666666667, "2026-09-16T17:00:10+08:00"),
            (2.3333333333, "2026-09-16T17:00:20+08:00"),
        )
    ]

    estimates = estimate_bus_positions(rows, [line])

    assert len(estimates) == 3
    assert {next(iter(estimate.source_observations)) for estimate in estimates} == {
        ("probe", 0),
        ("probe", 1),
        ("probe", 2),
    }


def test_kmb_91m_future_terminal_cohorts_keep_count_and_local_eta_bounds():
    """Captured Diamond Hill rows keep three buses without a terminus jump."""
    line = _line("KMB", "91M", "outbound", stop_count=29)
    rows = [
        Probe(
            "KMB", "91M", "outbound", index, minutes,
            cache_age_seconds=0.125,
            signed_minutes=signed_minutes,
            refresh_generation=1,
            arrival_at=datetime.fromisoformat(arrival_at),
        )
        for index, minutes, signed_minutes, arrival_at in (
            (22, 0.0, -0.9596931, "2026-09-16T17:26:20+08:00"),
            (23, 2.1236402333333335, 2.1236402333333335,
             "2026-09-16T17:29:25+08:00"),
            (24, 6.123640233333333, 6.123640233333333,
             "2026-09-16T17:33:25+08:00"),
            (25, 6.956973566666666, 6.956973566666666,
             "2026-09-16T17:34:15+08:00"),
            (26, 10.056973566666667, 10.056973566666667,
             "2026-09-16T17:37:21+08:00"),
            (27, 0.9736402333333333, 0.9736402333333333,
             "2026-09-16T17:28:16+08:00"),
            (27, 15.323640233333332, 15.323640233333332,
             "2026-09-16T17:42:37+08:00"),
            (28, 0.3236402333333333, 0.3236402333333333,
             "2026-09-16T17:27:37+08:00"),
            (28, 4.173640233333334, 4.173640233333334,
             "2026-09-16T17:31:28+08:00"),
            (28, 18.756973566666666, 18.756973566666666,
             "2026-09-16T17:46:03+08:00"),
        )
    ]

    estimates = estimate_bus_positions(
        rows,
        [line],
        observed_checkpoint_indices={
            ("KMB", "91M", "outbound"): range(29),
        },
    )

    earliest_terminal = next(
        estimate for estimate in estimates
        if ("probe", 7) in estimate.source_observations
    )
    second_terminal = next(
        estimate for estimate in estimates
        if ("probe", 8) in estimate.source_observations
    )
    trailing = next(
        estimate for estimate in estimates
        if ("probe", 9) in estimate.source_observations
    )
    assert len(estimates) == 3
    assert earliest_terminal.position == pytest.approx(
        28 - rows[7].minutes / 2.0
    )
    assert second_terminal.position == pytest.approx(
        27 - rows[5].minutes / 2.0
    )
    assert second_terminal.position < 27
    assert earliest_terminal.position < 28
    assert earliest_terminal.position_authoritative is False
    assert ("probe", 9) in trailing.source_observations


def test_atomic_kmb_temporal_heal_requires_one_route_response_revision():
    """Staggered disjoint rows are ambiguous and must remain separate."""
    line = _line("KMB", "91", "outbound", stop_count=31)
    rows = [
        Probe(
            "KMB", "91", "outbound", index, minutes,
            cache_age_seconds=9.391,
            signed_minutes=signed_minutes,
            refresh_generation=revision,
            arrival_at=datetime.fromisoformat(arrival_at),
        )
        for index, minutes, signed_minutes, revision, arrival_at in (
            (17, 0.0, -1.90, 304, "2026-09-08T23:46:30+08:00"),
            (18, 0.0, -0.55, 304, "2026-09-08T23:47:51+08:00"),
            (30, 16.412, 16.412, 305, "2026-09-09T00:04:39+08:00"),
        )
    ]

    estimates = estimate_bus_positions(rows, [line])

    assert len(estimates) == 2


def test_atomic_kmb_temporal_heal_rejects_geometry_prefix_end():
    """A rendered prefix endpoint is not an authoritative route terminus."""
    line = _line("KMB", "91P", "outbound", stop_count=11)
    rows = [
        Probe(
            "KMB", "91P", "outbound", index, minutes,
            cache_age_seconds=5.0,
            signed_minutes=signed_minutes,
            refresh_generation=305,
            arrival_at=datetime.fromisoformat(arrival_at),
        )
        for index, minutes, signed_minutes, arrival_at in (
            (4, 0.0, -1.0, "2026-09-09T00:00:00+08:00"),
            (5, 0.0, -0.2, "2026-09-09T00:00:48+08:00"),
            (10, 4.0, 4.0, "2026-09-09T00:04:00+08:00"),
        )
    ]

    estimates = estimate_bus_positions(rows, [line])

    assert len(estimates) == 2


def test_atomic_kmb_temporal_heal_preserves_equal_terminal_multiplicity():
    """Two equal terminal occurrences are still evidence for two vehicles."""
    line = _line("KMB", "91", "outbound", stop_count=31)
    rows = [
        Probe(
            "KMB", "91", "outbound", index, minutes,
            cache_age_seconds=9.391,
            signed_minutes=signed_minutes,
            refresh_generation=305,
            arrival_at=datetime.fromisoformat(arrival_at),
        )
        for index, minutes, signed_minutes, arrival_at in (
            (17, 0.0, -1.90, "2026-09-08T23:46:30+08:00"),
            (18, 0.0, -0.55, "2026-09-08T23:47:51+08:00"),
            (30, 16.412, 16.412, "2026-09-09T00:04:39+08:00"),
            (30, 16.412, 16.412, "2026-09-09T00:04:39+08:00"),
        )
    ]

    estimates = estimate_bus_positions(rows, [line])

    # Ambiguous equal occurrences are never consumed merely by input order.
    assert len(estimates) == 3
    terminal_tokens = [
        token
        for estimate in estimates
        for token in estimate.source_observations
        if token in {("probe", 2), ("probe", 3)}
    ]
    assert sorted(terminal_tokens) == [("probe", 2), ("probe", 3)]


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
        Probe("KMB", "X", "outbound", 2, None, cache_age_seconds=1, refresh_generation=1),
        Probe("KMB", "X", "outbound", 3, 1, cache_age_seconds=1, refresh_generation=2,
              arrival_at=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=6)),
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
    assert estimates[0].priority_indices == frozenset({3, 4})


def test_boundary_revision_includes_empty_lower_endpoint():
    line = _line(stop_count=7)
    rows = [
        Probe("KMB", "X", "outbound", 2, None, cache_age_seconds=9.0,
              refresh_generation=101),
        Probe("KMB", "X", "outbound", 3, 1, cache_age_seconds=8.4,
              refresh_generation=102,
              arrival_at=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=6)),
    ]
    estimates = estimate_bus_positions(
        rows, [line],
        observed_checkpoint_indices={("KMB", "X", "outbound"): range(7)},
    )
    assert len(estimates) == 1
    assert estimates[0].bracket == (2.0, 3.0)
    assert estimates[0].boundary_revision == (101, 102)
    assert estimates[0].position_authoritative is not False


def test_priority_indices_cover_full_zero_plateau_and_next_positive_stop():
    line = _line(stop_count=9)
    rows = [
        Probe("KMB", "X", "outbound", 2, None, cache_age_seconds=0),
        Probe("KMB", "X", "outbound", 3, 0, cache_age_seconds=0),
        Probe("KMB", "X", "outbound", 4, 0, cache_age_seconds=0),
        Probe("KMB", "X", "outbound", 5, 0, cache_age_seconds=0),
        Probe("KMB", "X", "outbound", 6, 2, cache_age_seconds=0),
        Probe("KMB", "X", "outbound", 7, 4, cache_age_seconds=0),
    ]

    estimates = estimate_bus_positions(
        rows,
        [line],
        observed_checkpoint_indices={("KMB", "X", "outbound"): range(9)},
    )

    assert len(estimates) == 1
    assert estimates[0].bracket == (5.0, 6.0)
    # Internal zero is a clamped due/overdue ETA. The last zero and first
    # positive ETA form the physical frontier, even without signed metadata.
    assert estimates[0].position == 5.0
    assert estimates[0].priority_indices == frozenset({3, 4, 5, 6})


def test_due_future_frontier_interpolates_preserved_signed_eta_offsets():
    line = _line(stop_count=7)
    rows = [
        Probe("KMB", "X", "outbound", 2, None, cache_age_seconds=0),
        Probe(
            "KMB", "X", "outbound", 3, 0,
            cache_age_seconds=0, signed_minutes=-0.6,
        ),
        Probe(
            "KMB", "X", "outbound", 4, 0.2,
            cache_age_seconds=0, signed_minutes=0.2,
        ),
        Probe("KMB", "X", "outbound", 5, 2.2, cache_age_seconds=0),
    ]

    estimates = estimate_bus_positions(
        rows,
        [line],
        observed_checkpoint_indices={("KMB", "X", "outbound"): range(7)},
    )

    assert len(estimates) == 1
    assert estimates[0].bracket == (3.0, 4.0)
    assert estimates[0].bracket_eta_offsets == pytest.approx((-0.6, 0.2))
    assert estimates[0].position == pytest.approx(3.75)


def test_consecutive_empty_stops_then_downstream_observation_forms_one_boundary_marker():
    line = _line(stop_count=7)
    rows = [
        # Several consecutive checkpoints see zero instances of this vehicle.
        Probe("KMB", "X", "outbound", 1, None, cache_age_seconds=0, refresh_generation=1),
        Probe("KMB", "X", "outbound", 2, None, cache_age_seconds=0, refresh_generation=1),
        Probe("KMB", "X", "outbound", 3, None, cache_age_seconds=0, refresh_generation=1),
        # Stop 4 first sees it; stop 5 corroborates the same ETA ladder.
        Probe("KMB", "X", "outbound", 4, 1, cache_age_seconds=0, refresh_generation=2,
              arrival_at=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=8)),
        Probe("KMB", "X", "outbound", 5, 3, cache_age_seconds=0, refresh_generation=2,
              arrival_at=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=10)),
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
        Probe("KMB", "X", "outbound", 2, None, cache_age_seconds=0, refresh_generation=1),
        # As fetched six minutes ago this source implied position -1.  On the
        # common identity clock it aligns with the fresh downstream rung at 2.
        Probe("KMB", "X", "outbound", 3, 8, cache_age_seconds=360, refresh_generation=2),
        Probe("KMB", "X", "outbound", 4, 4, cache_age_seconds=0, refresh_generation=3),
    ]
    estimates = estimate_bus_positions(
        rows,
        [line],
        observed_checkpoint_indices={("KMB", "X", "outbound"): range(7)},
    )
    assert len(estimates) == 1
    assert estimates[0].source_indices == frozenset({3, 4})
    assert estimates[0].bracket is None
    # The stale source value itself is unchanged; its age was not converted
    # into displayed motion.
    assert estimates[0].eta_minutes is None
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


def test_rebuild_positive_union_uses_upstream_empty_checkpoint():
    line = _line(stop_count=10)
    rows = [
        Probe("KMB", "X", "outbound", 0, None, cache_age_seconds=0, refresh_generation=11),
        Probe("KMB", "X", "outbound", 4, 0.2, cache_age_seconds=0,
              refresh_generation=11,
              arrival_at=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=8)),
        Probe("KMB", "X", "outbound", 6, None, cache_age_seconds=0, refresh_generation=11),
        Probe("KMB", "X", "outbound", 8, 0.7, cache_age_seconds=0, refresh_generation=11,
              arrival_at=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=16)),
        Probe("KMB", "X", "outbound", 9, 0.8, cache_age_seconds=0, refresh_generation=11,
              arrival_at=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=18)),
    ]
    base = BusEstimate(
        "X destination", 22.3, 114.2, Operator.KMB, 0.0,
        route="X", bound="outbound", operator_code="KMB",
        source_observations=frozenset({("probe", 1)}),
    )
    fragments = [
        BusEstimate(
            "X destination", 22.3, 114.2, Operator.KMB, 0.0,
            route="X", bound="outbound", operator_code="KMB",
            source_observations=frozenset({("probe", index)}),
        )
        for index in (3, 4)
    ]
    rebuilt = rebuild_estimate_from_probe_fragments(
        base, fragments, rows, [line]
    )
    assert rebuilt.bracket == (0.0, 4.0)
    assert rebuilt.position == pytest.approx(3.9)
    assert rebuilt.boundary_revision == (11, 11)


def test_rebuild_positive_union_rejects_unrelated_upstream_eta_checkpoint():
    line = _line(stop_count=8)
    rows = [
        Probe("KMB", "X", "outbound", 2, 5.0, refresh_generation=12),
        Probe("KMB", "X", "outbound", 4, 0.2, refresh_generation=12),
    ]
    base = BusEstimate(
        "X destination", 22.3, 114.2, Operator.KMB, 0.0,
        route="X", bound="outbound", operator_code="KMB",
        source_observations=frozenset({("probe", 1)}),
    )
    rebuilt = rebuild_estimate_from_probe_fragments(base, (), rows, [line])
    assert rebuilt.bracket is None
    assert rebuilt.position_authoritative is None
    assert rebuilt.position is None
    assert rebuilt.boundary_revision is None


def test_rebuild_due_only_union_is_a_point_boundary():
    line = _line(stop_count=8)
    rows = [
        Probe("KMB", "X", "outbound", 3, 0.0, refresh_generation=7),
        Probe("KMB", "X", "outbound", 4, 0.0, refresh_generation=7),
    ]
    base = BusEstimate(
        "X destination", 22.3, 114.2, Operator.KMB, 0.0,
        route="X", bound="outbound", operator_code="KMB",
        source_observations=frozenset({("probe", 0), ("probe", 1)}),
    )
    rebuilt = rebuild_estimate_from_probe_fragments(base, (), rows, [line])
    assert rebuilt.bracket == (4.0, 4.0)
    assert rebuilt.position == pytest.approx(4.0)


@pytest.mark.parametrize("kind", [EtaKind.REALTIME, EtaKind.MOVING_SLOWLY, EtaKind.DELAYED])
def test_owned_subset_accepts_confirmed_live_kinds(kind):
    rows = [
        Probe("KMB", "X", "outbound", 2, None, cache_age_seconds=0, refresh_generation=12),
        Probe("KMB", "X", "outbound", 3, 0, kind=kind, cache_age_seconds=0,
              signed_minutes=-0.25, refresh_generation=12,
              arrival_at=datetime.fromisoformat("2026-01-01T00:00:00+00:00")),
        Probe("KMB", "X", "outbound", 4, 0.75, kind=kind, cache_age_seconds=0,
              signed_minutes=0.75, refresh_generation=12,
              arrival_at=datetime.fromisoformat("2026-01-01T00:01:00+00:00")),
    ]
    template = BusEstimate("X destination", 0, 0, Operator.KMB, 0,
                           route="X", bound="outbound", operator_code="KMB")
    rebuilt = rebuild_estimate_from_probe_sources(template, (1, 2), rows, [_line()])
    assert rebuilt is not None
    assert rebuilt.position == pytest.approx(3.25)
    assert rebuilt.bracket == (3, 4)
    assert rebuilt.lon == pytest.approx(114.26325)


@pytest.mark.parametrize("field,value", [
    ("minutes", True), ("cache_age_seconds", False), ("signed_minutes", True),
    ("kind", EtaKind.SCHEDULED), ("kind", EtaKind.UNAVAILABLE), ("kind", "unknown"),
])
def test_owned_subset_rejects_boolean_numbers_and_non_live_kinds(field, value):
    rows = [
        Probe("KMB", "X", "outbound", 2, None, cache_age_seconds=0, refresh_generation=12),
        Probe("KMB", "X", "outbound", 3, 1, cache_age_seconds=0, refresh_generation=12,
              arrival_at=datetime.fromisoformat("2026-01-01T00:01:00+00:00")),
    ]
    setattr(rows[1], field, value)
    template = BusEstimate("X destination", 0, 0, Operator.KMB, 0,
                           route="X", bound="outbound", operator_code="KMB")
    assert rebuild_estimate_from_probe_sources(template, (1,), rows, [_line()]) is None


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
