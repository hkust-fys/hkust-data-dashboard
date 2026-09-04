"""End-to-end boundary revision checks across estimates and marker tracking."""

from datetime import UTC, datetime

import pytest

from dashboard.maps.positions import BusEstimate, _path_segment_length, estimate_bus_positions
from dashboard.maps.tracker import MarkerTracker
from dashboard.models import EtaKind, Operator
from dashboard.providers.route_geometry import RouteLine, Stop
from dashboard.providers.transit import ProbeEtaSnapshot, ProbeRouteGeneration

KEY = ("KMB", "X", "outbound")
BASE = datetime(2026, 1, 1, tzinfo=UTC)


class Probe:
    def __init__(self, index, minutes, age, revision):
        self.operator = "KMB"
        self.route = "X"
        self.bound = "outbound"
        self.index = index
        self.minutes = minutes
        self.signed_minutes = None
        self.cache_age_seconds = age
        self.refresh_generation = revision
        self.kind = EtaKind.REALTIME
        self.stop_id = f"S{index}"


def line():
    stops = tuple(Stop(str(i), f"Stop {i}", 22.333360, 114.260 + i * 0.001)
                  for i in range(7))
    path = [(stop.lat, stop.lon) for stop in stops]
    offsets = [0.0]
    for first, second in zip(stops, stops[1:], strict=False):
        offsets.append(offsets[-1] + _path_segment_length(
            (first.lat, first.lon), (second.lat, second.lon)))
    return RouteLine("X", "KMB", "outbound", stops, path, offsets)


def estimates(rows):
    return estimate_bus_positions(
        rows, [line()],
        observed_checkpoint_indices={KEY: range(7)},
    )


def snapshot(generation, when, rows=()):
    route = ProbeRouteGeneration(KEY, tuple(rows), generation, when)
    return ProbeEtaSnapshot((route,), when)


@pytest.mark.asyncio
async def test_estimate_revisions_move_delayed_and_hold_replayed_or_partial():
    route_line = line()
    initial_rows = [Probe(2, None, 0.0, 101), Probe(3, 1, 0.0, 102)]
    first = estimates(initial_rows)
    assert len(first) == 1 and first[0].boundary_revision == (101, 102)
    tracker = MarkerTracker()
    first_output = await tracker.update(snapshot(1, BASE, initial_rows), first, [route_line])
    assert first_output[0].position == pytest.approx(first[0].position)

    delayed_rows = [Probe(2, None, 38.0, 103), Probe(3, 4, 8.4, 104)]
    delayed = estimates(delayed_rows)
    moved = await tracker.update(
        snapshot(1, BASE.replace(second=20), delayed_rows), delayed, [route_line]
    )
    assert moved[0].position == pytest.approx(delayed[0].position)
    assert moved[0].position != pytest.approx(first_output[0].position)

    replay_rows = [Probe(2, None, 38.0, 103), Probe(3, 1, 8.4, 104)]
    replay = await tracker.update(
        snapshot(1, BASE.replace(second=30), replay_rows), estimates(replay_rows), [route_line]
    )
    assert replay[0].position == pytest.approx(moved[0].position)

    partial_rows = [Probe(2, None, 38.0, 105), Probe(3, 1, 8.4, 104)]
    partial = await tracker.update(
        snapshot(1, BASE.replace(second=40), partial_rows), estimates(partial_rows), [route_line]
    )
    assert partial[0].position == pytest.approx(replay[0].position)


@pytest.mark.asyncio
async def test_same_generation_replay_cannot_move_the_wrong_marker():
    tracker = MarkerTracker()

    def candidate(position, bracket, revision, source):
        return BusEstimate(
            "X", 22.3, 114.2, Operator.KMB, 0.0, route="X", bound="outbound",
            position=position, operator_code="KMB", bracket=bracket,
            eta_minutes=1.0, boundary_age_seconds=0.0,
            boundary_revision=revision,
            source_observations=frozenset({("probe", source)}),
        )

    first = [candidate(1.5, (1, 2), (201, 202), 1),
             candidate(5.5, (5, 6), (301, 302), 2)]
    await tracker.update(snapshot(1, BASE), first)
    replay = [candidate(5.5, (5, 6), (201, 202), 1),
              candidate(6.5, (6, 7), (302, 303), 2)]
    output = await tracker.update(snapshot(1, BASE.replace(second=10)), replay)
    by_id = {item.track_id: item.position for item in output}
    assert sorted(by_id.values()) == pytest.approx([1.5, 6.5])


@pytest.mark.asyncio
async def test_same_generation_alignment_does_not_pair_ineligible_nearest_track():
    tracker = MarkerTracker()

    def candidate(position, bracket, revision, source):
        return BusEstimate(
            "X", 22.3, 114.2, Operator.KMB, 0.0, route="X", bound="outbound",
            position=position, operator_code="KMB", bracket=bracket,
            eta_minutes=1.0, boundary_age_seconds=8.4,
            boundary_revision=revision,
            source_observations=frozenset({("probe", source)}),
        )

    first = [candidate(1.5, (1, 2), (10, 11), 1),
             candidate(3.5, (3, 4), (20, 21), 2)]
    await tracker.update(snapshot(1, BASE), first)
    # The candidate is physically nearest track 2, but its consumed endpoint
    # revisions can only advance track 1.  Alignment must apply that gate
    # before choosing the nearest pair.
    eligible = [candidate(3.0, (2, 3), (15, 16), 3)]
    output = await tracker.update(snapshot(1, BASE.replace(second=10)), eligible)
    by_id = {item.track_id: item.position for item in output}
    assert by_id[1] == pytest.approx(3.0)
    assert by_id[2] == pytest.approx(3.5)
