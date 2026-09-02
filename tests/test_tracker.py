from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from dashboard.maps.positions import BusEstimate
from dashboard.maps.tracker import MarkerTracker
from dashboard.models import Operator
from dashboard.providers.transit import ProbeEtaSnapshot, ProbeRouteGeneration

BASE_TIME = datetime(2026, 1, 1, tzinfo=UTC)


def _candidate(position, *, gate=False, route="R", operator=Operator.KMB,
               bound="out", unreliable=False, scheduled=False):
    code = {Operator.KMB: "KMB", Operator.CITYBUS: "CTB", Operator.GMB: "GMB"}[operator]
    return BusEstimate(
        f"{route} destination", 22.3, 114.2, operator, 0.0,
        unreliable=unreliable, route=route, bound=bound, position=position,
        operator_code=code,
        source_observations=frozenset({(
            "gate" if gate else ("scheduled" if scheduled else "probe"),
            int(position * 10),
        )}),
    )


def _snapshot(generation, rows=(), *, collected_at=BASE_TIME, route_key=None):
    key = route_key or ("KMB", "R", "out")
    route = ProbeRouteGeneration(key, tuple(rows), generation, collected_at)
    return ProbeEtaSnapshot((route,), collected_at)


def _omitted(collected_at):
    return ProbeEtaSnapshot((), collected_at)


def _line(*, stops=3, route="R", bound="out", operator="KMB"):
    return SimpleNamespace(operator=operator, route=route, bound=bound,
                           stops=tuple(range(stops)))


@pytest.mark.asyncio
async def test_first_generation_seed_and_unchanged_generation_ages_stably():
    tracker = MarkerTracker()
    first = await tracker.update(_snapshot(1), [_candidate(2.0)])
    second = await tracker.update(
        _snapshot(1, collected_at=BASE_TIME + timedelta(seconds=60)), [_candidate(2.5)]
    )
    assert len(first) == len(second) == 1
    assert first[0].track_id == second[0].track_id
    assert second[0].position == pytest.approx(2.5, abs=0.2)


@pytest.mark.asyncio
async def test_complete_empty_first_miss_coasts_once_then_second_miss_removes():
    tracker = MarkerTracker()
    visible = await tracker.update(_snapshot(1), [_candidate(2.0)])
    coast = await tracker.update(_snapshot(2), [])
    gone = await tracker.update(_snapshot(3), [])
    assert len(visible) == len(coast) == 1
    assert coast[0].track_id == visible[0].track_id
    assert gone == []


@pytest.mark.asyncio
async def test_omitted_or_partial_routes_coast_without_changing_cardinality():
    tracker = MarkerTracker()
    a, b = ("KMB", "A", "out"), ("KMB", "B", "out")
    seeded = await tracker.update(
        ProbeEtaSnapshot((ProbeRouteGeneration(a, (), 1, BASE_TIME),
                          ProbeRouteGeneration(b, (), 1, BASE_TIME)), BASE_TIME),
        [_candidate(1.0, route="A"), _candidate(3.0, route="B")],
    )
    partial = await tracker.update(_snapshot(2, route_key=a), [_candidate(1.2, route="A")])
    assert len(seeded) == len(partial) == 2
    assert {marker.route for marker in partial} == {"A", "B"}


@pytest.mark.asyncio
async def test_two_hit_birth_and_reliability_gate():
    tracker = MarkerTracker()
    await tracker.update(_snapshot(1), [_candidate(1.0)])
    assert len(await tracker.update(_snapshot(2), [_candidate(1.0), _candidate(4.0)])) == 1
    assert len(await tracker.update(_snapshot(3), [_candidate(1.1), _candidate(4.1)])) == 2

    reliable = MarkerTracker()
    assert len(await reliable.update(_snapshot(1), [_candidate(2.0, gate=True)])) == 1
    unreliable = MarkerTracker()
    tentative = _candidate(2.0, unreliable=True)
    assert await unreliable.update(_snapshot(1), [tentative]) == []
    assert len(await unreliable.update(_snapshot(2), [tentative])) == 1


@pytest.mark.asyncio
async def test_permutation_stable_ids_separation_and_no_crossing_backward():
    tracker = MarkerTracker()
    first = await tracker.update(_snapshot(1), [_candidate(2.0), _candidate(2.6)])
    second = await tracker.update(
        _snapshot(2, collected_at=BASE_TIME + timedelta(seconds=60)),
        [_candidate(2.6), _candidate(2.0)],
    )
    assert [m.track_id for m in first] == [m.track_id for m in second]
    assert second[0].position < second[1].position
    assert second[1].position - second[0].position > 0.1

    third = await tracker.update(_snapshot(3, collected_at=BASE_TIME + timedelta(seconds=120)),
                                 [_candidate(5.0), _candidate(1.0)])
    assert third[0].position <= third[1].position
    assert all(b.position >= a.position for a, b in zip(second, third, strict=True))


@pytest.mark.asyncio
async def test_generation_rollback_reseeds_and_clear_resets_identity():
    tracker = MarkerTracker()
    first = await tracker.update(_snapshot(5), [_candidate(2.0)])
    rolled = await tracker.update(_snapshot(4), [_candidate(1.0)])
    assert rolled[0].track_id != first[0].track_id
    tracker.clear()
    reset = await tracker.update(_snapshot(1), [_candidate(1.0)])
    assert reset[0].track_id != rolled[0].track_id


@pytest.mark.asyncio
async def test_route_bound_eviction_allows_same_generation_to_seed_again():
    tracker = MarkerTracker(max_routes=1)
    first = await tracker.update(_snapshot(1, route_key=("KMB", "A", "out")),
                                 [_candidate(2.0, route="A")])
    await tracker.update(_snapshot(1, route_key=("KMB", "B", "out")),
                         [_candidate(2.0, route="B")])
    returned = await tracker.update(_snapshot(1, route_key=("KMB", "A", "out")),
                                    [_candidate(2.0, route="A")])
    assert returned[0].track_id != first[0].track_id


@pytest.mark.asyncio
async def test_arbitrary_route_and_operator_labels_are_tracking_keys():
    tracker = MarkerTracker()
    key = ("GMB", "custom-route", "seq-9")
    candidate = _candidate(2.0, route="custom-route", operator=Operator.GMB, bound="seq-9")
    visible = await tracker.update(_snapshot(1, route_key=key), [candidate])
    assert visible[0].route == "custom-route"
    assert visible[0].operator_code == "GMB"


def test_constructor_coerces_and_validates_positive_bounds():
    tracker = MarkerTracker(max_routes="2", max_tracks_per_route=3.0)
    assert (tracker.max_routes, tracker.max_tracks_per_route) == (2, 3)
    with pytest.raises(ValueError):
        MarkerTracker(max_routes=0)
    with pytest.raises(ValueError):
        MarkerTracker(max_tracks_per_route="nope")
    for value in (-1, float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            MarkerTracker(evidence_ttl_seconds=value)


@pytest.mark.asyncio
async def test_omitted_terminal_route_before_ttl_retains_and_clamps():
    tracker = MarkerTracker(evidence_ttl_seconds=240)
    line = _line(stops=3)
    visible = await tracker.update(_snapshot(1, [_candidate(2.0)]),
                                   [_candidate(2.0)], [line])
    retained = await tracker.update(_omitted(BASE_TIME + timedelta(seconds=239)),
                                    [], [line])
    assert retained[0].track_id == visible[0].track_id
    assert retained[0].position == pytest.approx(2.0)


@pytest.mark.asyncio
async def test_omitted_terminal_route_at_ttl_retires():
    tracker = MarkerTracker(evidence_ttl_seconds=240)
    line = _line(stops=3)
    await tracker.update(_snapshot(1, [_candidate(2.0)]), [_candidate(2.0)], [line])
    assert await tracker.update(_omitted(BASE_TIME + timedelta(seconds=240)), [], [line]) == []


@pytest.mark.asyncio
async def test_omitted_route_at_ttl_mid_route_retains():
    tracker = MarkerTracker(evidence_ttl_seconds=240)
    line = _line(stops=5)
    await tracker.update(_snapshot(1, [_candidate(1.0)]), [_candidate(1.0)], [line])
    retained = await tracker.update(_omitted(BASE_TIME + timedelta(seconds=240)), [], [line])
    assert len(retained) == 1
    assert retained[0].position == pytest.approx(3.0)


@pytest.mark.asyncio
async def test_omitted_terminal_route_without_geometry_retains():
    tracker = MarkerTracker(evidence_ttl_seconds=0)
    await tracker.update(_snapshot(1, [_candidate(2.0)]), [_candidate(2.0)])
    retained = await tracker.update(_omitted(BASE_TIME + timedelta(seconds=900)), [])
    assert len(retained) == 1


@pytest.mark.asyncio
async def test_matched_generation_refreshes_terminal_evidence_age():
    tracker = MarkerTracker(evidence_ttl_seconds=240)
    line = _line(stops=3)
    candidate = _candidate(2.0)
    await tracker.update(_snapshot(1, [candidate]), [candidate], [line])
    refreshed_at = BASE_TIME + timedelta(seconds=100)
    await tracker.update(_snapshot(2, [candidate], collected_at=refreshed_at),
                         [candidate], [line])
    retained = await tracker.update(_omitted(refreshed_at + timedelta(seconds=239)),
                                    [], [line])
    assert len(retained) == 1


@pytest.mark.asyncio
async def test_configurable_track_bound_and_route_line_clamp():
    tracker = MarkerTracker(max_tracks_per_route=1)
    rows = [_candidate(1.0), _candidate(2.0)]
    line = SimpleNamespace(operator="KMB", route="R", bound="out", stops=(1, 2))
    result = await tracker.update(_snapshot(1, rows=rows), rows, [line])
    assert len(result) == 1
    assert result[0].position <= 1


@pytest.mark.asyncio
async def test_scheduled_first_generation_is_tentative_then_confirms():
    tracker = MarkerTracker()
    scheduled = _candidate(2.0, scheduled=True)
    assert await tracker.update(_snapshot(1), [scheduled]) == []
    visible = await tracker.update(_snapshot(2), [scheduled])
    assert len(visible) == 1
