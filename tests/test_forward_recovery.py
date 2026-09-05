"""Focused regression coverage for forward recovery after a probe disappears."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from dashboard.maps.positions import BusEstimate, _path_segment_length, estimate_bus_positions
from dashboard.maps.tracker import MarkerTracker
from dashboard.models import EtaKind, Operator
from dashboard.providers.route_geometry import RouteLine, Stop
from dashboard.providers.transit import ProbeEta, ProbeEtaSnapshot, ProbeRouteGeneration

KEY = ("KMB", "X", "outbound")
BASE = datetime(2026, 1, 1, tzinfo=UTC)


def _line(stops=18, *, route="X", bound="outbound", operator="KMB"):
    points = tuple(
        Stop(f"S{i}", f"Stop {i}", 22.333360, 114.260000 + i * 0.001)
        for i in range(stops)
    )
    path = [(stop.lat, stop.lon) for stop in points]
    offsets = [0.0]
    for first, second in zip(points, points[1:], strict=False):
        offsets.append(offsets[-1] + _path_segment_length(
            (first.lat, first.lon), (second.lat, second.lon)
        ))
    return RouteLine(route, operator, bound, points, path, offsets)


def _probe(index, minutes, revision, *, arrival=True, route="X", bound="outbound",
           age=0.0):
    return ProbeEta(
        "KMB", route, bound, f"S{index}", index, minutes,
        kind=EtaKind.REALTIME,
        cache_age_seconds=age,
        arrival_at=(arrival if isinstance(arrival, datetime)
                     else (BASE + timedelta(minutes=index + 10) if arrival else None)),
        observed_at=BASE,
        refresh_generation=revision,
        signed_minutes=minutes,
    )


def _snapshot(generation, rows, *, when=BASE, key=KEY, observed=None):
    route = ProbeRouteGeneration(
        key, tuple(rows), generation, when,
        frozenset(range(18) if observed is None else observed),
    )
    return ProbeEtaSnapshot((route,), when)


def _estimates(rows, *, observed=range(9), route_line=None):
    return estimate_bus_positions(
        rows, [route_line or _line()],
        observed_checkpoint_indices={KEY: observed},
    )


def _initial_rows():
    # A due lower rung and a positive upper rung form one owned two-sided ETA.
    return [_probe(2, 0.0, 10), _probe(3, 1.0, 11)]


def _forward_rows():
    return [
        _probe(2, None, 12, arrival=False),
        _probe(3, None, 13, arrival=False),
        _probe(4, 0.0, 14),
        _probe(5, 1.0, 15),
    ]


@pytest.mark.asyncio
async def test_real_probe_recovery_keeps_id_and_moves_past_disappeared_anchor():
    route_line = _line()
    initial_rows = _initial_rows()
    initial = _estimates(initial_rows, route_line=route_line)
    assert len(initial) == 1
    assert initial[0].bracket == (2.0, 3.0)
    assert initial[0].checkpoint_evidence == (
        (2, initial_rows[0].arrival_at.timestamp(), 10),
        (3, initial_rows[1].arrival_at.timestamp(), 11),
    )

    tracker = MarkerTracker()
    first = await tracker.update(_snapshot(1, initial_rows), initial, [route_line])
    recovered_rows = _forward_rows()
    recovered = _estimates(recovered_rows, route_line=route_line)
    second = await tracker.update(
        _snapshot(1, recovered_rows, when=BASE + timedelta(minutes=1)),
        recovered,
        [route_line],
    )

    assert len(second) == 1
    assert second[0].track_id == first[0].track_id
    assert second[0].position > 3.5
    assert second[0].bracket == (4.0, 5.0)


@pytest.mark.asyncio
async def test_fast_multi_stop_recovery_uses_sparse_requested_7_to_9_corridor():
    """A 2.0 marker can survive a rapid jump to the fresh 8.7 estimate."""
    route_line = _line()
    initial_rows = _initial_rows()
    tracker = MarkerTracker()
    first = await tracker.update(
        _snapshot(1, initial_rows), _estimates(initial_rows, route_line=route_line),
        [route_line],
    )

    empty = [
        _probe(2, None, 12, arrival=False),
        _probe(3, None, 13, arrival=False),
    ]
    await tracker.update(
        _snapshot(1, empty), _estimates(empty, route_line=route_line), [route_line],
    )
    assert 9 in tracker.poll_priorities()[KEY]

    jumped = [
        _probe(7, 0.0, 20),
        _probe(9, 1.0, 21),
    ]
    recovered = _manual(
        8.7,
        bracket=(7.0, 9.0),
        evidence=(
            (7, _probe(7, 0.0, 20).arrival_at.timestamp(), 20),
            (9, _probe(9, 1.0, 21).arrival_at.timestamp(), 21),
        ),
        revision=(20, 21),
    )
    output = await tracker.update(
        _snapshot(1, jumped), [recovered], [route_line],
    )
    assert len(output) == 1
    assert output[0].track_id == first[0].track_id
    assert output[0].position == pytest.approx(8.7)
    assert output[0].bracket == (7.0, 9.0)


@pytest.mark.asyncio
async def test_same_checkpoint_replacement_does_not_trigger_owned_forward_search():
    rows = _initial_rows()
    route_line = _line()

    # A newer ETA at the old stop belongs to another vehicle, while unchanged,
    # failed, cached, and revision-rollback rows must not be interpreted as a
    # successful disappearance of the tracked vehicle.
    cases = (
        # A different vehicle still reporting the old stop has a different
        # absolute arrival anchor and must not steal the owned track.
        ([_probe(2, 0.0, 12, arrival=BASE + timedelta(hours=1)),
          _probe(3, 1.0, 11)], "replacement"),
        (_initial_rows(), "unchanged"),
        ([_probe(3, 1.0, 11)], "failed"),
        ([_probe(2, None, 12, arrival=False, age=9.0),
          _probe(3, 1.0, 11)], "cached"),
        ([_probe(2, None, 9, arrival=False), _probe(3, 1.0, 11)], "rollback"),
    )
    for replacement, _case in cases:
        tracker = MarkerTracker()
        first = await tracker.update(
            _snapshot(1, rows), _estimates(rows, route_line=route_line), [route_line]
        )
        output = await tracker.update(
            _snapshot(1, replacement, when=BASE + timedelta(seconds=12)),
            _estimates(replacement, route_line=route_line), [route_line],
        )
        assert output[0].track_id == first[0].track_id
        assert output[0].position <= first[0].position
        assert not ({4, 5, 7} & tracker.poll_priorities()[KEY]), _case


@pytest.mark.asyncio
async def test_forward_empty_samples_advance_bounded_poll_without_all_stop_scan():
    route_line = _line(stops=18)
    first_rows = _initial_rows()
    tracker = MarkerTracker()
    seeded = await tracker.update(
        _snapshot(1, first_rows), _estimates(first_rows, observed=range(18), route_line=route_line),
        [route_line],
    )
    first_empty = _forward_rows()[:2] + [_probe(4, None, 16, arrival=False)]
    await tracker.update(
        _snapshot(1, first_empty, when=BASE + timedelta(minutes=1)),
        _estimates(first_empty, observed=range(18), route_line=route_line), [route_line],
    )
    priorities = tracker.poll_priorities()[KEY]
    assert 4 in priorities and max(priorities) <= 17
    assert len(priorities) <= 8

    # A repeated cached empty response does not repeatedly push the frontier.
    repeated = [_probe(2, None, 12, arrival=False), _probe(3, None, 13, arrival=False),
                _probe(4, None, 16, arrival=False)]
    await tracker.update(
        _snapshot(1, repeated, when=BASE + timedelta(minutes=2)),
        _estimates(repeated, observed=range(18), route_line=route_line), [route_line],
    )
    assert tracker.poll_priorities()[KEY] == priorities

    # A genuinely newer sampled pair beyond the disappeared stop reacquires
    # the same marker; a later narrow pair continues the same identity.
    for lower, upper, revisions in ((7, 10, (20, 21)), (10, 11, (22, 23))):
        forward = [
            _probe(2, None, 12, arrival=False), _probe(3, None, 13, arrival=False),
            _probe(lower, 0.0, revisions[0]), _probe(upper, 1.0, revisions[1]),
        ]
        before = await tracker.update(
            _snapshot(1, forward, when=BASE + timedelta(minutes=3)),
            _estimates(forward, observed=range(18), route_line=route_line), [route_line],
        )
        assert len(before) == 1
        assert before[0].track_id == seeded[0].track_id
        assert before[0].position >= lower


@pytest.mark.asyncio
async def test_ambiguous_old_tracks_do_not_cross_recover_or_change_cardinality():
    route_line = _line()
    rows = _initial_rows()
    tracker = MarkerTracker()
    candidates = _estimates(rows, route_line=route_line)
    # Two old tracks share the same owned bracket; only a candidate with a
    # genuinely newer two-rung ladder may recover, and it must not duplicate.
    candidates = [candidates[0], replace(candidates[0], source_observations=frozenset({("probe", 99)}))]
    first = await tracker.update(_snapshot(1, rows), candidates, [route_line])
    assert len(first) == 2
    forward = [
        _probe(2, None, 12, arrival=False), _probe(3, None, 13, arrival=False),
        _probe(7, 0.0, 20), _probe(8, 1.0, 21),
    ]
    recovered = _estimates(forward, route_line=route_line)
    second = await tracker.update(_snapshot(1, forward), recovered, [route_line])
    assert len(second) == 2
    assert {item.track_id for item in second} == {item.track_id for item in first}


@pytest.mark.asyncio
async def test_wrong_route_or_direction_cannot_reacquire_forward_identity():
    route_line = _line()
    rows = _initial_rows()
    tracker = MarkerTracker()
    first = await tracker.update(_snapshot(1, rows), _estimates(rows, route_line=route_line), [route_line])
    wrong = [_probe(4, 0.0, 14, route="Y", bound="inbound"),
             _probe(5, 1.0, 15, route="Y", bound="inbound")]
    wrong_line = _line(route="Y", bound="inbound")
    output = await tracker.update(
        _snapshot(1, wrong, key=("KMB", "Y", "inbound")),
        estimate_bus_positions(wrong, [wrong_line], observed_checkpoint_indices={
            ("KMB", "Y", "inbound"): range(18)
        }),
        [route_line, wrong_line],
    )
    assert len(output) == 2  # the wrong route may be born, but cannot reuse X's ID
    original = next(item for item in output if item.route == "X")
    foreign = next(item for item in output if item.route == "Y")
    assert original.track_id == first[0].track_id
    assert original.bound == "outbound"
    assert foreign.track_id != original.track_id


@pytest.mark.asyncio
async def test_missing_owned_upper_replaced_by_other_eta_starts_forward_search():
    route_line = _line()
    old = _initial_rows()
    tracker = MarkerTracker()
    await tracker.update(_snapshot(1, old), _estimates(old, route_line=route_line), [route_line])

    replacement = [
        _probe(2, None, 12, arrival=False),
        _probe(3, 1.0, 13, arrival=BASE + timedelta(hours=1)),
    ]
    output = await tracker.update(
        _snapshot(1, replacement, when=BASE + timedelta(minutes=1)),
        _estimates(replacement, route_line=route_line), [route_line],
    )
    assert len(output) == 1
    assert output[0].position == pytest.approx(2.0)
    assert {4, 5, 7} & tracker.poll_priorities()[KEY]


@pytest.mark.asyncio
async def test_cached_lower_empty_with_owned_upper_does_not_start_search():
    route_line = _line()
    old = _initial_rows()
    tracker = MarkerTracker()
    await tracker.update(_snapshot(1, old), _estimates(old, route_line=route_line), [route_line])
    cached = [_probe(2, None, 12, arrival=False, age=9.0), _probe(3, 1.0, 11)]
    output = await tracker.update(
        _snapshot(1, cached, when=BASE + timedelta(minutes=1)),
        _estimates(cached, route_line=route_line), [route_line],
    )
    assert len(output) == 1
    assert not ({4, 5, 7} & tracker.poll_priorities()[KEY])


def _manual(position, *, bracket=(2.0, 3.0), evidence=(), priorities=(), revision=(10, 11), unreliable=False):
    return BusEstimate(
        "X destination", 22.333, 114.262, Operator.KMB, 0.0,
        unreliable=unreliable,
        route="X", bound="outbound", position=position, operator_code="KMB",
        bracket=bracket, boundary_age_seconds=0.0, boundary_revision=revision,
        priority_indices=frozenset(priorities), checkpoint_evidence=tuple(evidence),
    )


@pytest.mark.asyncio
async def test_owned_terminal_evidence_is_preserved_while_search_starts_after_upper():
    rows = _initial_rows() + [_probe(17, 30.0, 17)]
    candidate = _estimates(rows, route_line=_line())[0]
    assert {2, 3, 17}.issubset({row[0] for row in candidate.checkpoint_evidence})
    tracker = MarkerTracker()
    await tracker.update(_snapshot(1, rows), [candidate], [_line()])
    empty = [_probe(2, None, 12, arrival=False), _probe(3, None, 13, arrival=False)]
    await tracker.update(_snapshot(1, empty), [], [_line()])
    priorities = tracker.poll_priorities()[KEY]
    assert 4 in priorities and 17 in priorities


@pytest.mark.asyncio
async def test_timestamp_free_estimate_still_starts_explicit_empty_upper_search():
    line = _line()
    candidate = _manual(2.5, evidence=(), priorities=(3,))
    tracker = MarkerTracker()
    await tracker.update(_snapshot(1, [_probe(2, 0.0, 10), _probe(3, 1.0, 11)]),
                         [candidate], [line])
    empty = [_probe(2, 0.0, 12), _probe(3, None, 13, arrival=False)]
    await tracker.update(_snapshot(1, empty), [], [line])
    assert 4 in tracker.poll_priorities()[KEY]


@pytest.mark.asyncio
async def test_new_complete_empty_generation_and_clear_remove_recovery_state():
    line = _line()
    tracker = MarkerTracker()
    candidate = _manual(2.5, evidence=((2, BASE.timestamp(), 10), (3, (BASE + timedelta(minutes=1)).timestamp(), 11)))
    await tracker.update(_snapshot(1, _initial_rows()), [candidate], [line])
    await tracker.update(_snapshot(1, [_probe(3, None, 13, arrival=False)]), [], [line])
    assert 4 in tracker.poll_priorities()[KEY]
    assert tracker.poll_priorities()[KEY]
    assert await tracker.update(_snapshot(2, []), [], [line]) == []
    assert tracker.poll_priorities() == {}
    tracker.clear()
    assert tracker.state_size == 0 and tracker.poll_priorities() == {}


@pytest.mark.asyncio
async def test_far_relative_candidate_holds_without_requested_upper_then_accepts_fresh_narrow_pair():
    line = _line()
    old = _manual(2.5, evidence=((2, BASE.timestamp(), 10), (3, (BASE + timedelta(minutes=1)).timestamp(), 11)), priorities=(10,))
    tracker = MarkerTracker()
    first = await tracker.update(_snapshot(1, _initial_rows()), [old], [line])
    unrequested = _manual(8.0, bracket=(8.0, 9.0), evidence=(), revision=(20, 21))
    held = await tracker.update(
        _snapshot(1, [_probe(3, None, 13, arrival=False), _probe(9, 1.0, 21)]),
        [unrequested], [line]
    )
    assert held[0].track_id == first[0].track_id and held[0].position == pytest.approx(2.5)
    fresh = _manual(7.0, bracket=(7.0, 10.0), evidence=((7, (BASE + timedelta(minutes=7)).timestamp(), 20), (10, (BASE + timedelta(minutes=10)).timestamp(), 21)), revision=(20, 21))
    accepted = await tracker.update(
        _snapshot(1, [_probe(7, 0.0, 20), _probe(10, 1.0, 21)]), [fresh], [line]
    )
    assert accepted[0].track_id == first[0].track_id and accepted[0].position == pytest.approx(7.0)


@pytest.mark.asyncio
async def test_one_sided_and_replayed_boundary_candidates_cannot_reacquire():
    line = _line()
    old = _manual(2.5, evidence=((2, BASE.timestamp(), 10), (3, (BASE + timedelta(minutes=1)).timestamp(), 11)))
    tracker = MarkerTracker()
    first = await tracker.update(_snapshot(1, _initial_rows()), [old], [line])
    shared = ((3, (BASE + timedelta(minutes=1)).timestamp(), 20),)
    one_sided = _manual(7.0, bracket=(7.0, 8.0), evidence=shared, revision=(20, 11))
    replay = _manual(7.0, bracket=(7.0, 8.0), evidence=shared, revision=(10, 11))
    for candidate in (one_sided, replay):
        output = await tracker.update(_snapshot(1, []), [candidate], [line])
        assert output[0].track_id == first[0].track_id
        assert output[0].position == pytest.approx(2.5)


@pytest.mark.asyncio
async def test_priority_cap_keeps_forward_scouts_and_terminal_with_legacy_low_priorities():
    line = _line(stops=60)
    candidate = _manual(2.5, evidence=((2, BASE.timestamp(), 10), (3, (BASE + timedelta(minutes=1)).timestamp(), 11)), priorities=range(41))
    tracker = MarkerTracker()
    await tracker.update(_snapshot(1, _initial_rows()), [candidate], [line])
    empty = [_probe(2, None, 12, arrival=False), _probe(3, None, 13, arrival=False)]
    await tracker.update(_snapshot(1, empty), [], [line])
    priorities = tracker.poll_priorities()[KEY]
    assert len(priorities) == 32
    assert {4, 5, 7, 59}.issubset(priorities)


@pytest.mark.asyncio
async def test_complete_replacement_cannot_keep_the_searching_departed_bus_as_a_ghost():
    line = _line()
    tracker = MarkerTracker()
    old = _manual(2.5)
    first = await tracker.update(_snapshot(1, _initial_rows()), [old], [line])
    await tracker.update(_snapshot(1, [_probe(3, None, 13, arrival=False)]), [], [line])
    replacement = _manual(2.0, revision=(20, 21))
    output = await tracker.update(_snapshot(2, []), [replacement], [line])
    assert len(output) == 1
    assert output[0].track_id != first[0].track_id
    assert output[0].position == replacement.position


@pytest.mark.asyncio
async def test_coarse_complete_forward_candidate_holds_then_narrow_corridor_moves_same_id():
    line = _line(stops=18)
    tracker = MarkerTracker()
    old = _manual(
        7.84, bracket=(5.0, 8.0),
        evidence=((5, BASE.timestamp(), 10),
                  (8, (BASE + timedelta(minutes=1)).timestamp(), 11),
                  (17, (BASE + timedelta(minutes=20)).timestamp(), 456)),
        revision=(10, 11),
    )
    first = await tracker.update(_snapshot(1, _initial_rows()), [old], [line])
    await tracker.update(
        _snapshot(1, [_probe(5, None, 12, arrival=False), _probe(8, None, 13, arrival=False)]),
        [], [line],
    )
    coarse = _manual(
        16.0, bracket=(9.0, 17.0), revision=(20, 21),
        evidence=((17, (BASE + timedelta(minutes=20, seconds=20)).timestamp(), 550),),
    )
    held = await tracker.update(_snapshot(2, []), [coarse], [line])
    assert held[0].track_id == first[0].track_id
    assert held[0].position == pytest.approx(7.84)

    narrow_rows = [_probe(9, 0.0, 22), _probe(10, 1.0, 23)]
    narrow = _manual(
        9.7, bracket=(9.0, 10.0),
        evidence=((9, _probe(9, 0.0, 22).arrival_at.timestamp(), 22),
                  (10, _probe(10, 1.0, 23).arrival_at.timestamp(), 23)),
        revision=(22, 23),
    )
    moved = await tracker.update(_snapshot(3, narrow_rows), [narrow], [line])
    assert moved[0].track_id == first[0].track_id
    assert moved[0].position == pytest.approx(9.7)


@pytest.mark.asyncio
async def test_coarse_forward_later_departure_does_not_hold_old_identity():
    line = _line(stops=18)
    tracker = MarkerTracker()
    old = _manual(
        7.84, bracket=(5.0, 8.0),
        evidence=((17, (BASE + timedelta(minutes=20)).timestamp(), 456),),
        revision=(10, 11),
    )
    first = await tracker.update(_snapshot(1, _initial_rows()), [old], [line])
    await tracker.update(
        _snapshot(1, [_probe(5, None, 12, arrival=False), _probe(8, None, 13, arrival=False)]),
        [], [line],
    )
    later = _manual(
        16.0, bracket=(9.0, 17.0), revision=(20, 21),
        evidence=((17, (BASE + timedelta(minutes=30)).timestamp(), 550),),
    )
    output = await tracker.update(_snapshot(2, []), [later], [line])
    assert len(output) == 1
    assert output[0].track_id != first[0].track_id
    assert output[0].position == pytest.approx(16.0)


@pytest.mark.parametrize(
    ("position", "bracket", "unreliable", "correction_position"),
    (
        (0.1868, (0.0, 8.0), True, 0.066),
        (0.1868, (0.0, 8.0), False, 0.066),
        (7.84, (5.0, 8.0), True, 7.9),
    ),
)
@pytest.mark.asyncio
async def test_coarse_marker_guards_do_not_start_forward_search(
    position, bracket, unreliable, correction_position,
):
    line = _line(stops=18)
    tracker = MarkerTracker()
    old = _manual(
        position, bracket=bracket, unreliable=unreliable,
        evidence=((8, (BASE + timedelta(minutes=20)).timestamp(), 33),),
        revision=(102, 33),
    )
    first = await tracker.update(_snapshot(1, _initial_rows()), [old], [line])
    missing_upper = [_probe(8, None, 34, arrival=False)]
    await tracker.update(
        _snapshot(1, missing_upper), [], [line],
    )
    priorities = tracker.poll_priorities()[KEY]
    assert not ({9, 10, 12, 14} & priorities)

    correction = _manual(
        correction_position, bracket=bracket, revision=(151, 173)
    )
    output = await tracker.update(_snapshot(2, []), [correction], [line])
    assert output[0].track_id == first[0].track_id
    assert output[0].position == pytest.approx(correction_position)


@pytest.mark.asyncio
async def test_terminal_upper_disappearance_does_not_start_forward_search():
    line = _line(stops=19)
    tracker = MarkerTracker()
    old = _manual(
        17.0, bracket=(9.0, 18.0),
        evidence=((18, (BASE + timedelta(minutes=20)).timestamp(), 54),),
        revision=(53, 54),
    )
    first = await tracker.update(_snapshot(1, _initial_rows()), [old], [line])
    shifted_terminal = [_probe(18, None, 195, arrival=False)]
    await tracker.update(_snapshot(1, shifted_terminal), [], [line])
    state = next(iter(tracker._routes[KEY].values()))
    assert state.forward_after is None
    assert state.forward_frontier == ()

    correction = _manual(
        14.383, bracket=(14.0, 18.0), revision=(194, 195)
    )
    output = await tracker.update(_snapshot(2, []), [correction], [line])
    assert output[0].track_id == first[0].track_id
    assert output[0].position == pytest.approx(14.383)


@pytest.mark.asyncio
async def test_coarse_forward_candidate_reseeds_frontier_then_narrow_candidate_recovers():
    line = _line(stops=18)
    tracker = MarkerTracker()
    old = _manual(
        7.405, bracket=(5.0, 8.0),
        evidence=((8, (BASE + timedelta(minutes=20)).timestamp(), 527),
                  (16, (BASE + timedelta(minutes=31)).timestamp(), 500),
                  (17, (BASE + timedelta(minutes=32)).timestamp(), 501)),
        revision=(526, 527),
    )
    first = await tracker.update(_snapshot(1, _initial_rows()), [old], [line])
    await tracker.update(
        _snapshot(1, [_probe(8, None, 528, arrival=False)]), [], [line]
    )
    coarse = _manual(
        16.0, bracket=(9.0, 17.0),
        evidence=((16, (BASE + timedelta(minutes=31, seconds=20)).timestamp(), 758),
                  (17, (BASE + timedelta(minutes=32, seconds=30)).timestamp(), 641)),
        revision=(758, 641),
    )
    held = await tracker.update(
        _snapshot(1, [_probe(9, 5.0, 758,
                             arrival=BASE + timedelta(hours=1)),
                      _probe(17, 30.0, 641,
                             arrival=BASE + timedelta(minutes=32, seconds=30))]),
        [coarse], [line]
    )
    state = next(iter(tracker._routes[KEY].values()))
    assert held[0].track_id == first[0].track_id
    assert held[0].position == pytest.approx(7.405)
    assert state.forward_after == 9
    assert {10, 11}.issubset(state.forward_frontier)

    narrow_rows = [_probe(10, 0.0, 760), _probe(11, 1.0, 761)]
    narrow = _manual(
        10.4, bracket=(10.0, 11.0),
        evidence=((10, (BASE + timedelta(minutes=30)).timestamp(), 760),
                  (11, (BASE + timedelta(minutes=31)).timestamp(), 761),
                  (17, (BASE + timedelta(minutes=32, seconds=40)).timestamp(), 642)),
        revision=(760, 761),
    )
    moved = await tracker.update(_snapshot(1, narrow_rows), [narrow], [line])
    assert moved[0].track_id == first[0].track_id
    assert moved[0].position == pytest.approx(10.4)
