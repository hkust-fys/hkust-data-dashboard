"""Current ETA evidence controls the map even while identity is uncertain."""

from dataclasses import replace
from datetime import UTC, datetime

import pytest

from dashboard.maps.positions import BusEstimate
from dashboard.maps.tracker import MarkerTracker
from dashboard.models import Operator
from dashboard.providers.route_geometry import RouteLine, Stop
from dashboard.providers.transit import ProbeEtaSnapshot, ProbeRouteGeneration

NOW = datetime(2026, 9, 19, 10, 50, tzinfo=UTC)
KEY = ("KMB", "91", "inbound")
LINE = RouteLine("91", "KMB", "inbound", [
    Stop(str(index), str(index), 22.3 + index / 10000, 114.2)
    for index in range(34)
])


def snapshot(generation=None, *, attempt=0, attempted=()):
    routes = () if generation is None else (ProbeRouteGeneration(
        route_key=KEY, rows=(), generation=generation, collected_at=NOW,
        observed_checkpoint_indices=frozenset(range(34)),
    ),)
    return ProbeEtaSnapshot(
        routes=routes, collected_at=NOW, positioning_rows=(),
        probe_attempt_generation=attempt, attempted_checkpoints=frozenset(attempted),
    )


def marker(position, revision=1):
    lower = int(position)
    return BusEstimate(
        "91 HKUST", 22.3 + position / 10000, 114.2, Operator.KMB, 0,
        route="91", bound="inbound", operator_code="KMB", position=position,
        bracket=(lower, lower + 1), boundary_revision=(revision, revision),
        boundary_age_seconds=0,
        checkpoint_evidence=((lower, NOW.timestamp() + lower * 120, revision),),
        source_observations=frozenset({("probe", lower)}),
        priority_indices=frozenset({lower, lower + 1}),
    )


def without_id(items):
    return [replace(item, track_id=None) for item in items]


@pytest.mark.asyncio
async def test_presentation_follows_current_count_and_local_positions_between_generations():
    tracker = MarkerTracker()
    # Public 91 inbound observation: the next response had three vehicles
    # while the historical tracker retained four at the previous positions.
    old = [marker(p) for p in (5.0, 12.521, 24.931, 31.852)]
    await tracker.present(snapshot(42), old, [LINE])
    current = [marker(p, 2) for p in (3.021, 13.351, 26.689)]
    displayed = await tracker.present(snapshot(42), current, [LINE])
    assert without_id(displayed) == current
    assert len(displayed) == 3

    advanced = [marker(p, 3) for p in (3.229, 14.298, 27.346)]
    displayed = await tracker.present(snapshot(42), advanced, [LINE])
    assert without_id(displayed) == advanced
    # A cached presentation does not extrapolate motion from wall-clock age.
    assert without_id(await tracker.present(snapshot(42), advanced, [LINE])) == advanced
    # Empty current evidence cannot leave historical ghost markers behind.
    assert await tracker.present(snapshot(42), [], [LINE]) == []


@pytest.mark.asyncio
async def test_current_evidence_is_visible_and_probed_before_complete_publication():
    tracker = MarkerTracker()
    current = [marker(6.25), marker(12.0)]
    displayed = await tracker.present(snapshot(), current, [LINE])
    assert without_id(displayed) == current
    priorities = tracker.poll_priorities()
    assert {6, 7, 11, 12, 13} <= priorities[KEY]

    # A new location must select its own neighbours after the old page was
    # attempted; an old temporal identity must not keep polling its old stop.
    attempted = {(*KEY, index) for index in priorities[KEY]}
    newer = [marker(8.75, 2)]
    await tracker.present(snapshot(attempt=1, attempted=attempted), newer, [LINE])
    assert tracker.poll_priorities()[KEY] == {8, 9}
    tracker.clear()
    assert tracker.poll_priorities() == {}


@pytest.mark.asyncio
async def test_same_stop_distinct_arrivals_keep_multiplicity_and_scheduled_style():
    tracker = MarkerTracker()
    first = marker(4.1)
    second = replace(marker(4.2), unreliable=True, checkpoint_evidence=(
        (4, NOW.timestamp() + 720, 1),
    ), source_observations=frozenset({("probe", 99)}))
    shown = await tracker.present(snapshot(1), [first, second], [LINE])
    assert without_id(shown) == [first, second]
    assert len(shown) == 2
    assert shown[1].unreliable
