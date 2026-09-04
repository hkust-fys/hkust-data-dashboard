from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from dashboard.maps.positions import BusEstimate
from dashboard.maps.tracker import MarkerTracker
from dashboard.models import Operator
from dashboard.providers.transit import ProbeEtaSnapshot, ProbeRouteGeneration

BASE_TIME = datetime(2026, 1, 1, tzinfo=UTC)


def _candidate(position, *, gate=False, route="R", operator=Operator.KMB,
               bound="out", unreliable=False, scheduled=False, bracket=None,
               boundary_age=None, arrival_at=None, priority_indices=(),
               boundary_revision=None):
    code = {Operator.KMB: "KMB", Operator.CITYBUS: "CTB", Operator.GMB: "GMB"}[operator]
    return BusEstimate(
        f"{route} destination", 22.3, 114.2, operator, 0.0,
        unreliable=unreliable, route=route, bound=bound, position=position,
        operator_code=code,
        source_observations=frozenset({(
            "gate" if gate else ("scheduled" if scheduled else "probe"),
            int(position * 10),
        )}),
        bracket=bracket,
        boundary_age_seconds=boundary_age,
        boundary_revision=boundary_revision,
        eta_arrival_at=arrival_at,
        priority_indices=frozenset(priority_indices),
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
async def test_new_boundary_revision_moves_after_delayed_render_and_replay_does_not():
    tracker = MarkerTracker()
    first = _candidate(5.5, bracket=(5, 6), boundary_age=30,
                       boundary_revision=(10, 10))
    await tracker.update(_snapshot(1, first and [first]), [first], [_line(stops=10)])

    delayed = _candidate(7.5, bracket=(7, 8), boundary_age=30,
                         boundary_revision=(11, 11))
    moved = await tracker.update(_snapshot(1), [delayed], [_line(stops=10)])
    assert moved[0].position == pytest.approx(7.5)

    replay = _candidate(8.0, bracket=(7, 8), boundary_age=0,
                        boundary_revision=(11, 11))
    held = await tracker.update(_snapshot(1), [replay], [_line(stops=10)])
    assert held[0].position == pytest.approx(7.5)


@pytest.mark.asyncio
async def test_one_sided_boundary_revision_is_not_actionable():
    tracker = MarkerTracker()
    initial = _candidate(5.5, bracket=(5, 6), boundary_age=0,
                         boundary_revision=(20, 20))
    await tracker.update(_snapshot(1, [initial]), [initial], [_line(stops=10)])
    partial = _candidate(6.5, bracket=(6, 7), boundary_age=0,
                         boundary_revision=(21, 20))
    held = await tracker.update(_snapshot(1), [partial], [_line(stops=10)])
    assert held[0].position == pytest.approx(5.5)


@pytest.mark.asyncio
async def test_unchanged_generation_without_fresh_boundary_holds_stably():
    tracker = MarkerTracker()
    first = await tracker.update(_snapshot(1), [_candidate(2.0)])
    second = await tracker.update(
        _snapshot(1, collected_at=BASE_TIME + timedelta(seconds=60)), [_candidate(2.5)]
    )
    assert len(first) == len(second) == 1
    assert first[0].track_id == second[0].track_id
    assert second[0].position == pytest.approx(2.0)


@pytest.mark.asyncio
async def test_complete_empty_generation_removes_immediately():
    tracker = MarkerTracker()
    visible = await tracker.update(_snapshot(1), [_candidate(2.0)])
    gone = await tracker.update(_snapshot(2), [])
    assert len(visible) == 1
    assert gone == []


@pytest.mark.asyncio
async def test_omitted_or_partial_routes_hold_without_changing_cardinality():
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
async def test_complete_generation_births_follow_eta_population_immediately():
    tracker = MarkerTracker()
    await tracker.update(_snapshot(1), [_candidate(1.0)])
    assert len(await tracker.update(
        _snapshot(2), [_candidate(1.0), _candidate(4.0)]
    )) == 2

    reliable = MarkerTracker()
    assert len(await reliable.update(_snapshot(1), [_candidate(2.0, gate=True)])) == 1
    unreliable = MarkerTracker()
    tentative = _candidate(2.0, unreliable=True)
    assert len(await unreliable.update(_snapshot(1), [tentative])) == 1


@pytest.mark.asyncio
async def test_complete_generation_drops_unmatched_stale_bracket_ghost():
    tracker = MarkerTracker()
    first = await tracker.update(
        _snapshot(1),
        [
            _candidate(1.5, bracket=(1.0, 2.0), boundary_age=0),
            _candidate(2.5, bracket=(2.0, 3.0), boundary_age=0),
            _candidate(5.5, bracket=(5.0, 6.0), boundary_age=0),
        ],
    )
    current = await tracker.update(
        _snapshot(2, collected_at=BASE_TIME + timedelta(seconds=60)),
        [
            _candidate(1.7, bracket=(1.0, 2.0), boundary_age=30),
            _candidate(2.7, bracket=(2.0, 3.0), boundary_age=30),
        ],
    )
    assert len(current) == 2
    assert [marker.track_id for marker in current] == [
        first[0].track_id,
        first[1].track_id,
    ]
    assert [marker.position for marker in current] == [1.5, 2.5]


@pytest.mark.asyncio
async def test_unbracketed_gate_refresh_cannot_move_or_coast_track():
    tracker = MarkerTracker()
    first = await tracker.update(
        _snapshot(1),
        [_candidate(8.0, gate=True, bracket=(7.0, 8.0), boundary_age=0)],
    )
    refreshed = await tracker.update(
        _snapshot(1, collected_at=BASE_TIME + timedelta(seconds=30)),
        [_candidate(9.256, gate=True)],
    )
    omitted = await tracker.update(
        _omitted(BASE_TIME + timedelta(seconds=120)), []
    )
    assert refreshed[0].track_id == omitted[0].track_id == first[0].track_id
    assert refreshed[0].position == omitted[0].position == first[0].position
    assert refreshed[0].bracket == omitted[0].bracket == (7.0, 8.0)
    assert refreshed[0].boundary_age_seconds == pytest.approx(30.0)
    assert omitted[0].boundary_age_seconds == pytest.approx(120.0)


@pytest.mark.asyncio
async def test_same_generation_fresh_boundaries_are_not_starved_by_stale_candidates():
    tracker = MarkerTracker()
    initial = [
        _candidate(6.5, bracket=(6.0, 7.0), boundary_age=0,
                   arrival_at=BASE_TIME + timedelta(seconds=100)),
        _candidate(7.90, bracket=(7.0, 8.0), boundary_age=0,
                   arrival_at=BASE_TIME + timedelta(seconds=0)),
        _candidate(7.93, bracket=(7.0, 8.0), boundary_age=0,
                   arrival_at=BASE_TIME + timedelta(seconds=3)),
    ]
    first = await tracker.update(_snapshot(1), initial)
    refreshed = [
        _candidate(6.6, bracket=(6.0, 7.0), boundary_age=0,
                   arrival_at=BASE_TIME + timedelta(seconds=100)),
        _candidate(7.16, bracket=(7.0, 8.0), boundary_age=0,
                   arrival_at=BASE_TIME + timedelta(seconds=120)),
        _candidate(8.0, bracket=(7.0, 8.0), boundary_age=0,
                   arrival_at=BASE_TIME + timedelta(seconds=0)),
        _candidate(8.2, bracket=(8.0, 9.0), boundary_age=30,
                   arrival_at=BASE_TIME + timedelta(seconds=3)),
    ]
    current = await tracker.update(
        _snapshot(1, collected_at=BASE_TIME + timedelta(seconds=30)), refreshed
    )

    assert [marker.track_id for marker in current] == [
        marker.track_id for marker in first
    ]
    assert [marker.position for marker in current] == pytest.approx(
        [6.6, 7.16, 8.0]
    )
    assert all(marker.boundary_age_seconds == pytest.approx(0.0)
               for marker in current)


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
    assert retained[0].position == pytest.approx(1.0)


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
async def test_scheduled_first_complete_generation_is_visible():
    tracker = MarkerTracker()
    scheduled = _candidate(2.0, scheduled=True)
    visible = await tracker.update(_snapshot(1), [scheduled])
    assert len(visible) == 1


@pytest.mark.asyncio
async def test_priority_poll_covers_every_active_marker_boundary():
    tracker = MarkerTracker()
    candidates = [
        _candidate(1.5, bracket=(1.0, 2.0), boundary_age=0),
        _candidate(4.5, bracket=(4.0, 5.0), boundary_age=0),
        _candidate(7.5, bracket=(7.0, 8.0), boundary_age=0),
    ]
    await tracker.update(_snapshot(1), candidates)
    assert tracker.poll_priorities() == {
        ("KMB", "R", "out"): frozenset({1, 2, 4, 5, 7, 8})
    }


@pytest.mark.asyncio
async def test_priority_poll_includes_route_terminus_with_marker_boundaries():
    tracker = MarkerTracker()
    candidates = [_candidate(1.5, bracket=(1.0, 2.0), boundary_age=0)]
    await tracker.update(_snapshot(1), candidates, [_line(stops=6)])
    assert tracker.poll_priorities() == {
        ("KMB", "R", "out"): frozenset({1, 2, 5})
    }


@pytest.mark.asyncio
async def test_priority_poll_bisects_a_coarse_marker_bracket():
    tracker = MarkerTracker()
    candidates = [_candidate(4.5, bracket=(1.0, 8.0), boundary_age=0)]
    await tracker.update(_snapshot(1), candidates, [_line(stops=10)])

    assert tracker.poll_priorities() == {
        ("KMB", "R", "out"): frozenset({1, 4, 5, 8, 9})
    }


@pytest.mark.asyncio
async def test_priority_poll_covers_owned_zero_plateau_and_forward_rung():
    tracker = MarkerTracker()
    candidate = _candidate(
        3.0,
        bracket=(2.0, 3.0),
        boundary_age=0,
        priority_indices={3, 4, 5, 6},
    )
    await tracker.update(_snapshot(1), [candidate], [_line(stops=9)])
    assert tracker.poll_priorities() == {
        ("KMB", "R", "out"): frozenset({2, 3, 4, 5, 6, 8})
    }


@pytest.mark.asyncio
async def test_priority_poll_ignores_terminus_without_an_active_marker():
    tracker = MarkerTracker()
    await tracker.update(_snapshot(1), [], [_line(stops=6)])
    assert tracker.poll_priorities() == {}


@pytest.mark.asyncio
async def test_priority_poll_keeps_terminals_separate_for_each_direction():
    tracker = MarkerTracker()
    first_key = ("KMB", "A", "out")
    second_key = ("KMB", "B", "in")
    snapshot = ProbeEtaSnapshot(
        (ProbeRouteGeneration(first_key, (), 1, BASE_TIME),
         ProbeRouteGeneration(second_key, (), 1, BASE_TIME)),
        BASE_TIME,
    )
    await tracker.update(
        snapshot,
        [_candidate(1.5, route="A", bound="out", bracket=(1.0, 2.0), boundary_age=0),
         _candidate(2.5, route="B", bound="in", bracket=(2.0, 3.0), boundary_age=0)],
        [_line(route="A", bound="out", stops=6),
         _line(route="B", bound="in", stops=9)],
    )
    assert tracker.poll_priorities() == {
        first_key: frozenset({1, 2, 5}),
        second_key: frozenset({2, 3, 8}),
    }


@pytest.mark.asyncio
async def test_priority_poll_cap_retains_route_terminus():
    tracker = MarkerTracker()
    candidates = [
        _candidate(index * 2 + 0.5, bracket=(index * 2, index * 2 + 1), boundary_age=0)
        for index in range(20)
    ]
    await tracker.update(_snapshot(1), candidates, [_line(stops=1000)])
    priorities = tracker.poll_priorities()[("KMB", "R", "out")]
    assert len(priorities) == 32
    assert 999 in priorities


@pytest.mark.asyncio
async def test_clear_removes_remembered_route_terminus():
    tracker = MarkerTracker()
    await tracker.update(_snapshot(1), [_candidate(1.5, bracket=(1, 2), boundary_age=0)],
                         [_line(stops=6)])
    tracker.clear()
    assert tracker.poll_priorities() == {}


@pytest.mark.asyncio
async def test_stale_bracket_holds_exactly_across_generation_change():
    tracker = MarkerTracker()
    fresh = _candidate(3.5, bracket=(3.0, 4.0), boundary_age=0)
    first = await tracker.update(_snapshot(1), [fresh])
    stale = _candidate(4.5, bracket=(4.0, 5.0), boundary_age=30)
    second = await tracker.update(
        _snapshot(2, collected_at=BASE_TIME + timedelta(seconds=60)), [stale]
    )
    assert second[0].track_id == first[0].track_id
    assert second[0].position == first[0].position
    assert second[0].bracket == first[0].bracket


@pytest.mark.asyncio
async def test_fresh_same_generation_boundary_can_snap_backward_without_birth():
    tracker = MarkerTracker()
    first = await tracker.update(
        _snapshot(1), [_candidate(5.5, bracket=(5.0, 6.0), boundary_age=0)]
    )
    corrected = await tracker.update(
        _snapshot(1, collected_at=BASE_TIME + timedelta(seconds=30)),
        [_candidate(4.5, bracket=(4.0, 5.0), boundary_age=0)],
    )
    assert len(corrected) == 1
    assert corrected[0].track_id == first[0].track_id
    assert corrected[0].position == 4.5


@pytest.mark.asyncio
async def test_unbracketed_partial_probe_holds_last_real_boundary():
    tracker = MarkerTracker()
    first = await tracker.update(
        _snapshot(1), [_candidate(5.5, bracket=(5.0, 6.0), boundary_age=0)]
    )
    partial = await tracker.update(
        _snapshot(1, collected_at=BASE_TIME + timedelta(seconds=30)),
        [_candidate(3.0)],
    )
    assert partial[0].track_id == first[0].track_id
    assert partial[0].position == first[0].position
    assert partial[0].bracket == (5.0, 6.0)


@pytest.mark.asyncio
async def test_ambiguous_stale_generation_match_holds_exactly_while_fresh_tracks_advance():
    tracker = MarkerTracker()
    old_rows = [
        _candidate(10.0, bracket=(9.0, 10.0), boundary_age=0),
        _candidate(10.0, bracket=(9.0, 10.0), boundary_age=0),
        _candidate(22.0, bracket=(21.0, 22.0), boundary_age=0),
    ]
    first = await tracker.update(_snapshot(670), old_rows)
    assert len(first) == 3

    stale = replace(
        _candidate(12.0, bracket=(11.0, 12.0), boundary_age=None),
        source_observations=old_rows[1].source_observations,
    )
    fresh = _candidate(11.0, bracket=(10.0, 11.0), boundary_age=0)
    later_fresh = _candidate(22.0, bracket=(21.0, 22.0), boundary_age=0)
    second = await tracker.update(
        _snapshot(672, collected_at=BASE_TIME + timedelta(seconds=60)),
        [fresh, stale, later_fresh],
    )

    held = next(marker for marker in second if marker.bracket == (9.0, 10.0))
    advanced = next(marker for marker in second if marker.bracket == (10.0, 11.0))
    assert held.position == pytest.approx(10.0)
    assert held.bracket == (9.0, 10.0)
    assert advanced.position == pytest.approx(11.0)
    assert len(second) == 3

    recovered = await tracker.update(
        _snapshot(673, collected_at=BASE_TIME + timedelta(seconds=120)),
        [_candidate(12.0, bracket=(11.0, 12.0), boundary_age=0),
         _candidate(13.0, bracket=(12.0, 13.0), boundary_age=0),
         _candidate(23.0, bracket=(22.0, 23.0), boundary_age=0)],
    )
    recovered_held = next(marker for marker in recovered if marker.track_id == held.track_id)
    assert recovered_held.position == pytest.approx(12.0)
    assert recovered_held.bracket == (11.0, 12.0)


@pytest.mark.asyncio
async def test_tied_component_preserves_strict_predecessor_and_successor_boundaries():
    tracker = MarkerTracker()
    initial = await tracker.update(
        _snapshot(1),
        [_candidate(9.0, bracket=(8.0, 9.0), boundary_age=0),
         _candidate(10.0, bracket=(9.0, 10.0), boundary_age=0),
         _candidate(10.0, bracket=(9.0, 10.0), boundary_age=0)],
    )
    updated = await tracker.update(
        _snapshot(2, collected_at=BASE_TIME + timedelta(seconds=60)),
        [_candidate(11.0, bracket=(10.0, 11.0), boundary_age=0),
         _candidate(12.0, bracket=(11.0, 12.0), boundary_age=None),
         _candidate(12.0, bracket=(11.0, 12.0), boundary_age=None)],
    )
    assert [marker.position for marker in updated] == [9.0, 10.0, 10.0]
    assert [marker.track_id for marker in updated] == [initial[0].track_id,
                                                        initial[1].track_id,
                                                        initial[2].track_id]

    tracker = MarkerTracker()
    initial = await tracker.update(
        _snapshot(1),
        [_candidate(10.0, bracket=(9.0, 10.0), boundary_age=0),
         _candidate(10.0, bracket=(9.0, 10.0), boundary_age=0),
         _candidate(11.0, bracket=(10.0, 11.0), boundary_age=0)],
    )
    updated = await tracker.update(
        _snapshot(2, collected_at=BASE_TIME + timedelta(seconds=60)),
        [_candidate(9.0, bracket=(8.0, 9.0), boundary_age=None),
         _candidate(9.0, bracket=(8.0, 9.0), boundary_age=None),
         _candidate(12.0, bracket=(11.0, 12.0), boundary_age=0)],
    )
    assert [marker.position for marker in updated] == [10.0, 10.0, 12.0]
    assert {marker.track_id for marker in updated} == {marker.track_id for marker in initial}


@pytest.mark.asyncio
async def test_long_strict_chain_is_not_crossed_by_ahead_correction():
    tracker = MarkerTracker()
    initial = await tracker.update(
        _snapshot(1),
        [_candidate(8.0, bracket=(7.0, 8.0), boundary_age=0),
         _candidate(9.0, bracket=(8.0, 9.0), boundary_age=0),
         _candidate(10.0, bracket=(9.0, 10.0), boundary_age=0),
         _candidate(10.0, bracket=(9.0, 10.0), boundary_age=0)],
    )
    updated = await tracker.update(
        _snapshot(2, collected_at=BASE_TIME + timedelta(seconds=60)),
        [_candidate(10.5, bracket=(10.0, 11.0), boundary_age=0),
         _candidate(11.0, bracket=(10.0, 11.0), boundary_age=0),
         _candidate(12.0, bracket=(11.0, 12.0), boundary_age=None),
         _candidate(12.0, bracket=(11.0, 12.0), boundary_age=None)],
    )
    by_id = {marker.track_id: marker.position for marker in updated}
    assert [by_id[marker.track_id] for marker in initial] == sorted(
        by_id[marker.track_id] for marker in initial
    )


@pytest.mark.asyncio
async def test_same_generation_correction_preserves_original_tie_chain_boundaries():
    tracker = MarkerTracker()
    initial = await tracker.update(
        _snapshot(1),
        [_candidate(8.0, bracket=(7.0, 8.0), boundary_age=0),
         _candidate(9.0, bracket=(8.0, 9.0), boundary_age=0),
         _candidate(10.0, bracket=(9.0, 10.0), boundary_age=0),
         _candidate(10.0, bracket=(9.0, 10.0), boundary_age=0)],
    )
    corrected = await tracker.update(
        _snapshot(1, collected_at=BASE_TIME + timedelta(seconds=60)),
        [_candidate(10.5, bracket=(10.0, 11.0), boundary_age=0),
         _candidate(11.0, bracket=(10.0, 11.0), boundary_age=0),
         _candidate(12.0, bracket=(11.0, 12.0), boundary_age=None),
         _candidate(12.0, bracket=(11.0, 12.0), boundary_age=None)],
    )
    positions = {marker.track_id: marker.position for marker in corrected}
    assert [positions[marker.track_id] for marker in initial] == sorted(
        positions[marker.track_id] for marker in initial
    )


@pytest.mark.asyncio
async def test_omitted_route_holds_tie_and_strict_neighbor():
    tracker = MarkerTracker()
    initial = await tracker.update(
        _snapshot(1),
        [_candidate(9.0), _candidate(10.0), _candidate(10.0)],
    )
    omitted = await tracker.update(
        _omitted(BASE_TIME + timedelta(seconds=60)), []
    )
    by_id = {marker.track_id: marker.position for marker in omitted}
    assert [by_id[marker.track_id] for marker in initial] == [9.0, 10.0, 10.0]


@pytest.mark.asyncio
async def test_complete_turnover_prefers_eta_anchors_over_unanchored_distance():
    tracker = MarkerTracker()

    def candidate(position, token, **kwargs):
        return replace(
            _candidate(position, **kwargs),
            source_observations=frozenset({("probe", token)}),
        )

    old_arrival = BASE_TIME + timedelta(minutes=10)
    initial = await tracker.update(
        _snapshot(1),
        [
            candidate(0.0, 10, bracket=(0.0, 0.0), boundary_age=0),
            candidate(0.0, 20, bracket=(0.0, 0.0), boundary_age=0),
            candidate(
                7.9, 30, bracket=(7.0, 8.0), boundary_age=0,
                arrival_at=old_arrival,
            ),
            candidate(
                7.9, 40, bracket=(7.0, 8.0), boundary_age=0,
                arrival_at=old_arrival,
            ),
        ],
    )
    downstream_ids = {
        next(iter(marker.source_observations))[1]: marker.track_id
        for marker in initial
        if marker.position > 0
    }

    updated = await tracker.update(
        _snapshot(2, collected_at=BASE_TIME + timedelta(seconds=60)),
        [
            # This newly observed gate candidate is spatially matchable to an
            # old downstream track but has no ETA anchor or fresh bracket.
            candidate(5.0, 10),
            candidate(
                8.0, 30, bracket=(7.0, 8.0), boundary_age=0,
                arrival_at=old_arrival + timedelta(seconds=24),
            ),
            candidate(
                8.0, 40, bracket=(7.0, 8.0), boundary_age=0,
                arrival_at=old_arrival + timedelta(seconds=24),
            ),
        ],
    )

    assert [marker.position for marker in updated] == [5.0, 8.0, 8.0]
    assert {
        next(iter(marker.source_observations))[1]: marker.track_id
        for marker in updated
        if marker.position == 8.0
    } == downstream_ids
    assert len({marker.source_observations for marker in updated}) == len(updated)
