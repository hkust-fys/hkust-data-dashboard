from collections import Counter
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from math import inf, nextafter
from random import Random
from time import perf_counter, process_time
from types import SimpleNamespace

import pytest

import dashboard.maps.tracker as tracker_module
from dashboard.maps.positions import BusEstimate, estimate_bus_positions
from dashboard.maps.tracker import (
    MarkerTracker,
    _certified_first_boundary_reseed_position,
    _checkpoint_ownership_profile,
    _commit_boundary_evidence,
    _first_boundary_reseed_context,
    _matching_track,
    _missing_instance,
    _next_poll_checkpoints,
    _ordered_pairs,
    _position_order_authoritative,
    _retain_current_checkpoint_capacity,
    _same_generation_actionable,
    _select_valid_partial_transaction,
    _Track,
    _unique_tied_checkpoint_assignment,
)
from dashboard.models import EtaKind, Operator
from dashboard.providers.route_geometry import RouteLine, Stop
from dashboard.providers.transit import ProbeEta, ProbeEtaSnapshot, ProbeRouteGeneration

BASE_TIME = datetime(2026, 1, 1, tzinfo=UTC)


def _route11_identity_case():
    """Public ETA evidence distilled from the last clean cohort and frames 84/85."""
    epoch = BASE_TIME.timestamp() - 41.1721

    def evidence(values):
        return tuple((stop, epoch + arrival, revision) for stop, arrival, revision in values)

    def candidate(position, values, slots, bracket, revision=None, age=None):
        return replace(
            _candidate(position, route="11", operator=Operator.GMB, bound="seq-1",
                       bracket=bracket, boundary_revision=revision, boundary_age=age),
            checkpoint_evidence=evidence(values),
            source_indices=frozenset(stop for stop, _, _ in values),
            source_observations=frozenset(("probe", slot) for slot in slots),
        )

    cohort = [
        candidate(5.0, ((7, 285.550, 771), (8, 350.320, 805), (17, 896.660, 806)),
                  (66, 69, 178), (0.0, 6.0)),
        candidate(6.0, ((7, 62.067, 771), (8, 121.739, 805), (17, 792.473, 806)),
                  (65, 68, 177), (0.0, 6.0)),
        candidate(7.961849366666667, ((8, 18.257, 805), (17, 397.550, 806)),
                  (64, 67), (7.0, 8.0), (842, 805), 46.219),
    ]
    frame84 = [
        replace(candidate(0.0, ((0, 100.0, 881), (3, 368.190, 850), (4, 390.155, 848)),
                          (58, 163, 166), (0.0, 0.0), (881, 881), 0), unreliable=True),
        candidate(5.0, ((7, 254.986, 842), (8, 309.757, 882), (17, 912.678, 883)),
                  (66, 69, 170), (3.0, 6.0)),
        candidate(6.0, ((7, 89.854, 842), (8, 147.855, 882), (17, 849.473, 883)),
                  (65, 68, 169), (3.0, 6.0)),
        candidate(7.883513016666667, ((8, 99.257, 882), (17, 402.859, 883)),
                  (64, 67), (7.0, 8.0), (842, 882), 0),
    ]
    frame85 = [
        frame84[0], frame84[1],
        candidate(7.069589714922744, ((7, 92.617, 888), (8, 147.855, 882), (17, 849.473, 883)),
                  (65, 68, 169), (7.0, 8.0), (888, 882), 0),
        replace(frame84[3], boundary_revision=(888, 882)),
    ]
    return cohort, frame84, frame85


@pytest.mark.asyncio
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("renumber", [False, True])
async def test_same_generation_route11_reserves_held_identity_before_frames84_85(reverse, renumber):
    cohort, frame84, frame85 = _route11_identity_case()
    key = ("GMB", "11", "seq-1")
    tracker = MarkerTracker()
    initial = await tracker.update(_snapshot(817, route_key=key), cohort)
    old = list(tracker._routes[key].values())
    frozen = [track.cohort_evidence for track in old]
    held = old[0]
    held.forward_after = 5
    held.forward_frontier = (6, 7)
    held.forward_revision = held.forward_started_revision = 805
    held.forward_baselines = {6: 805, 7: 771}
    invariants = {name: getattr(held, name) for name in (
        "position", "motion_bracket", "display_bracket", "boundary_revision",
        "boundary_observed_at", "committed_boundary_evidence", "position_authoritative",
        "forward_frontier", "forward_after", "forward_revision", "forward_started_revision",
        "forward_baselines", "cohort_evidence", "cohort_observed_at", "last_evidence_at",
    )}
    for number, candidates in enumerate((frame84, frame85), 1):
        if renumber:
            candidates = [replace(candidate, source_observations=frozenset(
                (kind, 1000 * number - slot) for kind, slot in candidate.source_observations
            )) for candidate in candidates]
        current = await tracker.update(
            _snapshot(817, collected_at=BASE_TIME + timedelta(seconds=40 + 10 * number),
                      route_key=key),
            list(reversed(candidates)) if reverse else candidates,
        )
        assert [item.track_id for item in current] == [item.track_id for item in initial]
        assert [item.position for item in current] == pytest.approx(
            [5.0, 6.0, cohort[2].position] if number == 1 else [5.0, 7.069589714922744, 7.883513016666667]
        )
        assert [track.cohort_evidence for track in old] == frozen
        assert all(getattr(held, name) == value for name, value in invariants.items())
        assert [item.checkpoint_evidence for item in current] == [
            item.checkpoint_evidence for item in candidates[1:]
        ]
        assert all(count == 1 for count in Counter(
            row for item in current for row in item.source_observations
        ).values())
        assert all(count == 1 for count in Counter(
            row for item in current for row in item.checkpoint_evidence
        ).values())
    # Cached repetition cannot renew a reservation's evidence lifetime.
    await tracker.update(_snapshot(817, collected_at=BASE_TIME + timedelta(seconds=70),
                                   route_key=key), frame85)
    assert all(getattr(held, name) == value for name, value in invariants.items())
    atomic = await tracker.update(
        _snapshot(894, collected_at=BASE_TIME + timedelta(seconds=80), route_key=key), frame85,
    )
    departed = [item for item in atomic if not item.unreliable]
    assert [item.track_id for item in departed] == [item.track_id for item in initial]
    assert [item.position for item in departed] == pytest.approx([5.0, 7.069589714922744, 7.883513016666667])


@pytest.mark.parametrize("variant", ["cold", "scheduled", "equal", "malformed"])
def test_same_generation_identity_census_keeps_nonmoving_competitors(variant):
    cohort, _frame84, frame85 = _route11_identity_case()
    old = [_Track(index + 1, item, item.position, 817,
                   cohort_evidence=item.checkpoint_evidence,
                   cohort_observed_at=BASE_TIME.timestamp()) for index, item in enumerate(cohort)]
    duplicate = frame85[2]
    if variant == "cold":
        duplicate = replace(duplicate, bracket=None, position_authoritative=False)
    elif variant == "scheduled":
        duplicate = replace(duplicate, unreliable=True)
    elif variant == "malformed":
        duplicate = replace(duplicate, checkpoint_evidence=duplicate.checkpoint_evidence + ((-1, 0, 0),))
    reservations, blocked_old, blocked_new = tracker_module._same_generation_identity_plan(
        old, [*frame85, duplicate], BASE_TIME.timestamp() + 60,
    )
    assert (1, 2) not in reservations
    assert 1 in blocked_old and {2, 4} <= blocked_new
    if variant != "malformed":
        assert reservations == {(0, 1), (2, 3)}


@pytest.mark.asyncio
async def test_same_generation_ambiguous_component_cannot_fall_through_to_motion():
    cohort, _frame84, frame85 = _route11_identity_case()
    key = ("GMB", "11", "seq-1")
    tracker = MarkerTracker()
    initial = await tracker.update(_snapshot(817, route_key=key), cohort)
    candidate = replace(frame85[2], position=6.1, bracket=None, position_authoritative=False)
    result = await tracker.update(
        _snapshot(817, collected_at=BASE_TIME + timedelta(seconds=60), route_key=key),
        [*frame85, candidate],
    )
    assert [marker.track_id for marker in result] == [marker.track_id for marker in initial]
    assert [marker.position for marker in result] == pytest.approx([5.0, 6.0, 7.883513016666667])
    assert result[1].checkpoint_evidence == initial[1].checkpoint_evidence
    assert tracker.poll_lifecycle_routes() == {key}


@pytest.mark.parametrize(("stops", "revisions", "elapsed", "expected"), [
    ((7, 8), (11, 11), 30, True),
    ((7, 7), (11, 11), 30, False),
    ((7, 8), (10, 11), 30, False),
    ((7, 8), (10, 10), 30, False),
    ((7, 8), (11, 11), 121, False),
])
def test_same_generation_tolerant_identity_requires_distinct_newer_checkpoints(stops, revisions, elapsed, expected):
    rows = tuple((stop, BASE_TIME.timestamp() + 120, 10) for stop in stops)
    initial = replace(_candidate(5.0), checkpoint_evidence=rows)
    candidate = replace(initial, checkpoint_evidence=tuple(
        (stop, arrival + 20, revision)
        for (stop, arrival, _), revision in zip(rows, revisions, strict=True)
    ))
    old = [_Track(1, initial, 5.0, 1, cohort_evidence=rows,
                   cohort_observed_at=BASE_TIME.timestamp())]
    reservations, _blocked_old, _blocked_new = tracker_module._same_generation_identity_plan(
        old, [candidate], BASE_TIME.timestamp() + elapsed,
    )
    assert bool(reservations) is expected


@pytest.mark.parametrize("distinct", [False, True])
def test_same_generation_identity_plan_is_bounded_for_full_tied_route(distinct):
    initial = _candidate(5.0)
    old = [_Track(index, replace(initial, checkpoint_evidence=tuple(
        (stop, 1000.0 + stop * 120 + (index * 0.1 if distinct else 0), 10)
        for stop in range(64)
    )), float(index), 1) for index in range(128)]
    candidates = [replace(initial, checkpoint_evidence=tuple(
        (stop, arrival + 30, 11) for stop, arrival, _ in track.estimate.checkpoint_evidence
    )) for track in old]
    started = process_time()
    reservations, blocked_old, blocked_new = tracker_module._same_generation_identity_plan(
        old, candidates, BASE_TIME.timestamp(),
    )
    assert not reservations
    assert blocked_old == blocked_new == set(range(128))
    # Bound algorithmic work, excluding scheduling delays on shared CI hosts.
    assert process_time() - started < 5.0


@pytest.mark.parametrize("spacing", [1.0, 10.0])
def test_full_identity_census_is_bounded_with_32_occurrences_per_stop(spacing):
    initial = _candidate(5.0)
    old = [_Track(index, replace(initial, checkpoint_evidence=tuple(
        (stop, 1000.0 + stop * 1000 + occurrence * spacing + index * 0.001, 10)
        for stop in range(2) for occurrence in range(32)
    )), float(index), 1) for index in range(128)]
    candidates = [replace(initial, checkpoint_evidence=tuple(
        (stop, arrival + 40, 11) for stop, arrival, _ in track.estimate.checkpoint_evidence
    )) for track in old]
    started = process_time()
    reservations, blocked_old, blocked_new = tracker_module._same_generation_identity_plan(
        old, candidates, BASE_TIME.timestamp(),
    )
    assert not reservations
    assert blocked_old == blocked_new == set(range(128))
    # Both all-to-all and partially overlapping arrival windows remain bounded.
    # CPU time catches the original 15-second scan without host-scheduling noise.
    assert process_time() - started < 5.0


def test_repeated_checkpoint_masks_preserve_occurrence_and_revision_matching():
    random = Random(93218)

    def indexed(rows):
        result = {}
        for number, (stop, arrival, revision) in enumerate(rows):
            result.setdefault(stop, []).append((arrival, revision, number))
        return {stop: sorted(values, key=lambda row: (row[0], row[2]))
                for stop, values in result.items()}

    def reference(owned, current):
        exact = set(tracker_module._checkpoint_overlap(owned, current))
        temporal, unique, newer_stops = set(), Counter(), set()
        for old_stop, old_arrival, old_revision in owned:
            matches = set()
            for number, (stop, arrival, revision) in enumerate(current):
                if stop != old_stop:
                    continue
                drift = abs(arrival - old_arrival)
                newer = revision > old_revision and drift <= 90.0
                if newer:
                    newer_stops.add(stop)
                if drift <= 0.5 or newer:
                    matches.add(number)
            temporal.update(matches)
            if len(matches) == 1:
                unique.update(matches)
        eligible = {row for row, count in unique.items() if count == 1}
        return exact, temporal, eligible, len(newer_stops) >= 2, bool(exact)

    for case in range(160):
        current = tuple(
            (random.choice((7, 8)), 1000.0 + random.randrange(12) * 30, random.randrange(1, 6))
            for _ in range(random.randrange(1, 65))
        )
        current_index = indexed(current)
        range_cache = {}
        # Reuse the current index with different old revision populations, as
        # the full census does; cached masks must remain strictly revision-local.
        for _ in range(3):
            owned = tuple(
                (stop, arrival + random.choice((-90.001, -90.0, -0.5, 0.0, 0.5, 90.0, 90.001)),
                 random.randrange(1, 6))
                for stop, arrival, _revision in random.choices(current, k=random.randrange(1, 65))
            )
            actual = tracker_module._indexed_checkpoint_links(indexed(owned), current_index, range_cache)
            assert actual == reference(owned, current), case


@pytest.mark.parametrize(("prior", "arrival", "tolerance", "included"), [
    (100.1, 10.099999999999993, 90.0, True),
    (150.76253236201435, 60.76253236201435, 90.0, True),
    (nextafter(float(2**31), -inf), nextafter(float(2**31), -inf) + 90.0, 90.0, False),
    (-0.5, 1e-18, 0.5, True),
    (nextafter(float(2**31), -inf), nextafter(float(2**31), -inf) + 0.5, 0.5, False),
])
@pytest.mark.parametrize("multiplicity", [1, 2, 32])
def test_repeated_checkpoint_windows_match_original_arrival_predicate(prior, arrival, tolerance, included, multiplicity):
    assert (abs(arrival - prior) <= tolerance) is included
    revision = 11 if tolerance == 90.0 else 10
    owned = {stop: [(prior, 10, group * multiplicity + number) for number in range(multiplicity)]
             for group, stop in enumerate((7, 8))}
    current = {
        stop: [(arrival, revision, group * (multiplicity + 1) + number) for number in range(multiplicity)]
        + [(prior + 1000.0, revision, group * (multiplicity + 1) + multiplicity)]
        for group, stop in enumerate((7, 8))
    }
    exact, temporal, eligible, strong, exact_proof = tracker_module._indexed_checkpoint_links(owned, current)
    expected = {group * (multiplicity + 1) + number for group in range(2)
                for number in range(multiplicity)} if included else set()
    assert exact == (expected if tolerance == 0.5 else set())
    assert exact_proof is (included and tolerance == 0.5)
    assert temporal == expected
    assert eligible == (expected if multiplicity == 1 else set())
    assert strong is (included and tolerance == 90.0)


@pytest.mark.parametrize("center", [
    100.1, 150.76253236201435, nextafter(float(2**31), -inf),
    -90.0, -0.5, 1e-308, 1e308, -1e308,
])
@pytest.mark.parametrize("tolerance", [0.5, 90.0])
def test_checkpoint_range_mask_preserves_subtraction_rounding(center, tolerance):
    arrivals = tuple(sorted([
        value for edge in (center - tolerance, center + tolerance)
        for value in (nextafter(edge, -inf), edge, nextafter(edge, inf))
    ] + [-1e-15, 1e-15]))
    expected = sum(1 << index for index, arrival in enumerate(arrivals)
                   if abs(arrival - center) <= tolerance)
    assert tracker_module._checkpoint_arrival_range_mask(arrivals, center, tolerance) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("reverse", [False, True])
async def test_exact_census_owner_outranks_neighbour_temporal_continuation(reverse):
    first = replace(_candidate(5.0, bracket=(4.0, 5.0), boundary_age=0,
                               boundary_revision=(9, 9)),
                    checkpoint_evidence=((7, 100.0, 9), (8, 200.0, 9)))
    second = replace(_candidate(6.0, bracket=(5.0, 6.0), boundary_age=0,
                                boundary_revision=(10, 10)),
                     checkpoint_evidence=((7, 160.0, 10), (8, 260.0, 10)))
    tracker = MarkerTracker()
    initial = await tracker.update(_snapshot(1), [second, first] if reverse else [first, second])
    candidate = replace(second, position=6.5, bracket=(6.0, 7.0), boundary_revision=(11, 11))
    old = list(tracker._routes[("KMB", "R", "out")].values())
    reservations, _, _ = tracker_module._same_generation_identity_plan(old, [candidate], BASE_TIME.timestamp())
    assert reservations == {(1, 0)}
    current = await tracker.update(_snapshot(1), [candidate])
    assert [marker.track_id for marker in current] == [marker.track_id for marker in initial]
    assert [marker.position for marker in current] == [5.0, 6.5]
    assert current[0].checkpoint_evidence == first.checkpoint_evidence
    assert current[1].checkpoint_evidence == second.checkpoint_evidence


def test_exact_census_composes_shared_checkpoint_owners_before_temporal_claims():
    shared = ((7, 100.0, 10), (8, 200.0, 10))
    first = replace(_candidate(5.0), checkpoint_evidence=shared)
    second = replace(_candidate(6.0), checkpoint_evidence=shared + ((9, 300.0, 10),))
    old = [_Track(1, first, 5.0, 1), _Track(2, second, 6.0, 1)]
    exact_first = first
    exact_second = replace(second, checkpoint_evidence=((9, 300.0, 10),))
    temporal = replace(first, checkpoint_evidence=((7, 120.0, 11), (8, 220.0, 11)))
    reservations, _, _ = tracker_module._same_generation_identity_plan(
        old, [exact_first, exact_second, temporal], BASE_TIME.timestamp(),
    )
    assert reservations == {(0, 0), (1, 1)}


def test_ambiguous_exact_candidate_cannot_be_reserved_to_temporal_outsider():
    temporal_owner = replace(_candidate(5.0), checkpoint_evidence=((7, 100.0, 9), (8, 200.0, 9)))
    exact_owner = replace(_candidate(6.0), checkpoint_evidence=((7, 160.0, 10), (8, 260.0, 10)))
    old = [_Track(1, temporal_owner, 5.0, 1), _Track(2, exact_owner, 6.0, 1),
           _Track(3, exact_owner, 7.0, 1)]
    reservations, blocked_old, blocked_new = tracker_module._same_generation_identity_plan(
        old, [exact_owner], BASE_TIME.timestamp(),
    )
    assert not reservations
    # The shared exact owners keep their existing motion-eligible fallback;
    # tolerant evidence cannot manufacture a unique reservation for the outsider.
    assert not blocked_old and not blocked_new


@pytest.mark.parametrize("kind", ["exact", "temporal"])
def test_invalid_candidate_cannot_force_another_edge_in_its_component(kind):
    def candidate(rows):
        return replace(_candidate(5.0), checkpoint_evidence=rows)

    if kind == "exact":
        initial = [candidate(((7, 100.0, 10), (8, 200.0, 10))),
                   candidate(((7, 100.0, 10), (8, 300.0, 10)))]
        shared = candidate(((7, 100.0, 10),))
        invalid = candidate(((8, 200.0, 10), (9, 150.0, 10)))
    else:
        initial = [candidate(((7, 100.0, 10), (8, 200.0, 10))),
                   candidate(((7, 230.0, 10), (8, 330.0, 10)))]
        shared = candidate(((7, 170.0, 11), (8, 270.0, 11)))
        invalid = candidate(((7, 120.0, 11), (8, 220.0, 11), (9, 50.0, 11)))
    initial.append(candidate(((15, 600.0, 10), (16, 700.0, 10))))
    unrelated = candidate(((15, 620.0, 11), (16, 720.0, 11)))
    old = [_Track(index + 1, item, float(index), 1) for index, item in enumerate(initial)]
    reservations, blocked_old, blocked_new = tracker_module._same_generation_identity_plan(
        old, [shared, invalid, unrelated], BASE_TIME.timestamp(),
    )
    assert reservations == {(2, 2)}
    assert blocked_old == blocked_new == {0, 1}


@pytest.mark.parametrize("oversized_old", [False, True])
def test_identity_census_rejects_ledgers_above_supported_checkpoint_bound(oversized_old):
    valid = replace(_candidate(5.0), checkpoint_evidence=((7, 100.0, 10),))
    oversized = replace(valid, checkpoint_evidence=tuple((stop, 100.0 + stop, 10) for stop in range(65)))
    old = [_Track(1, oversized if oversized_old else valid, 5.0, 1)]
    reservations, blocked_old, blocked_new = tracker_module._same_generation_identity_plan(
        old, [valid if oversized_old else oversized], BASE_TIME.timestamp(),
    )
    assert not reservations
    assert blocked_old == blocked_new == {0}


@pytest.mark.asyncio
async def test_public_partial_update_preserves_repeated_candidate_object_occurrences(monkeypatch):
    first = replace(_candidate(4.0, bracket=(3.0, 4.0), boundary_age=0,
                               boundary_revision=(10, 10)), checkpoint_evidence=((7, 100.0, 10),))
    second = replace(first, position=5.0, bracket=(4.0, 5.0))
    tracker = MarkerTracker()
    initial = await tracker.update(_snapshot(1), [first, second])
    current = replace(second, position=5.5, bracket=(5.0, 6.0), boundary_revision=(11, 11))
    selected_pairs = []
    transaction = tracker_module._select_valid_partial_transaction

    def capture(old, population, updates, pairs, proposed, **kwargs):
        selected_pairs.extend(pairs)
        return transaction(old, population, updates, pairs, proposed, **kwargs)

    monkeypatch.setattr(tracker_module, "_select_valid_partial_transaction", capture)
    result = await tracker.update(_snapshot(1), [current, current])
    assert {index for _, index in selected_pairs} == {0, 1}
    assert {item.track_id for item in result} == {item.track_id for item in initial}
    assert len(result) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("omit_after_acceptance", [False, True])
async def test_held_metadata_rejects_revisions_between_frozen_and_accepted(omit_after_acceptance):
    initial = replace(_candidate(5.5, bracket=(5.0, 6.0), boundary_age=0,
                                 boundary_revision=(10, 10)),
                      checkpoint_evidence=((7, 100.0, 10), (8, 200.0, 10), (9, 300.0, 10)))
    tracker = MarkerTracker()
    seeded = await tracker.update(_snapshot(1), [initial])
    track = tracker._routes[("KMB", "R", "out")][seeded[0].track_id]
    current = replace(initial, position=8.0, bracket=None, boundary_revision=None,
                      position_authoritative=False, eta_arrival_at=BASE_TIME + timedelta(seconds=120),
                      eta_minutes=2.0, source_observations=frozenset({("probe", 30)}),
                      source_indices=frozenset({7, 8}), priority_indices=frozenset({7}),
                      checkpoint_evidence=((7, 120.0, 30), (8, 220.0, 30), (9, 320.0, 30)))
    await tracker.update(_snapshot(1, collected_at=BASE_TIME + timedelta(seconds=30)), [current])
    assert track.estimate.checkpoint_evidence == current.checkpoint_evidence
    if omit_after_acceptance:
        current = replace(current, checkpoint_evidence=current.checkpoint_evidence[1:])
        await tracker.update(_snapshot(1, collected_at=BASE_TIME + timedelta(seconds=40)), [current])
        assert track.estimate.checkpoint_evidence == current.checkpoint_evidence
    accepted_bundle = track.estimate
    stale = replace(current, eta_arrival_at=BASE_TIME + timedelta(seconds=130), eta_minutes=1.0,
                    source_observations=frozenset({("probe", 20)}),
                    source_indices=frozenset({7}), priority_indices=frozenset({8}),
                    checkpoint_evidence=((7, 130.0, 20), (8, 230.0, 30), (9, 330.0, 30)))
    result = await tracker.update(_snapshot(1, collected_at=BASE_TIME + timedelta(seconds=60)), [stale])
    assert track.estimate == accepted_bundle
    assert result[0].track_id == seeded[0].track_id
    assert result[0].position == seeded[0].position
    assert track.cohort_evidence == initial.checkpoint_evidence
    assert track.last_evidence_at == BASE_TIME.timestamp()
    if omit_after_acceptance:
        complete = replace(current, checkpoint_evidence=((8, 220.0, 40), (9, 320.0, 40)))
        await tracker.update(_snapshot(2, collected_at=BASE_TIME + timedelta(seconds=90)), [complete])
        assert track.metadata_revision_floors == {8: 40, 9: 40}


def test_regressing_metadata_keeps_retained_claim_in_birth_capacity_transaction():
    retained = (7, 100.0, 30)
    stale = (7, 100.0, 20)
    old = [_capacity_track(1, retained, 4.0)]
    candidate, birth = _capacity_candidate(stale, 4.0), _capacity_candidate(retained, 5.0)
    accepted, birth_allowed, motion = _select_valid_partial_transaction(
        old, [candidate, birth], [candidate], [(0, 0)], {}, metadata_indices={0},
        birth_candidate=birth,
    )
    assert accepted == motion == set()
    assert not birth_allowed


def _revision_candidate(position, rows, revision=10):
    return replace(_candidate(position, bracket=(float(int(position)), float(int(position) + 1)),
                              boundary_age=0, boundary_revision=(revision, revision)),
                   checkpoint_evidence=rows)


@pytest.mark.asyncio
async def test_regressing_exact_census_row_fences_frozen_owner_after_metadata_refresh():
    tracker = MarkerTracker()
    first = _revision_candidate(4.0, ((7, 100.0, 10), (8, 200.0, 10)))
    second = _revision_candidate(6.0, ((7, 400.0, 10), (8, 500.0, 10)))
    initial = await tracker.update(_snapshot(1), [first, second])
    accepted = replace(second, bracket=None, boundary_revision=None, position_authoritative=False,
                       checkpoint_evidence=((7, 420.0, 30), (8, 520.0, 30)),
                       source_observations=frozenset({("probe", 30)}))
    await tracker.update(_snapshot(1, collected_at=BASE_TIME + timedelta(seconds=30)), [accepted])
    old = list(tracker._routes[("KMB", "R", "out")].values())
    held_bundle = old[1].estimate
    stale = replace(_revision_candidate(4.5, ((7, 400.0, 9), (8, 500.0, 9)), 40),
                    source_observations=frozenset({("probe", 9)}))
    unrelated = _revision_candidate(4.25, ((7, 120.0, 31), (8, 220.0, 31)), 31)
    reservations, blocked_old, blocked_new = tracker_module._same_generation_identity_plan(
        old, [stale, unrelated], BASE_TIME.timestamp() + 60,
    )
    assert reservations == {(0, 1)}
    assert blocked_old == {1} and blocked_new == {0}
    current = await tracker.update(
        _snapshot(1, collected_at=BASE_TIME + timedelta(seconds=60)), [stale, unrelated],
    )
    assert [item.track_id for item in current] == [item.track_id for item in initial]
    assert [item.position for item in current] == [4.25, 6.0]
    assert old[1].estimate == held_bundle


@pytest.mark.asyncio
async def test_post_cohort_ttl_fallback_cannot_publish_regressing_response():
    tracker = MarkerTracker()
    initial = _revision_candidate(5.0, ((7, 100.0, 10), (8, 200.0, 10)))
    seeded = await tracker.update(_snapshot(1), [initial])
    accepted = replace(initial, bracket=None, boundary_revision=None, position_authoritative=False,
                       checkpoint_evidence=((7, 120.0, 30), (8, 220.0, 30)),
                       source_observations=frozenset({("probe", 30)}))
    await tracker.update(_snapshot(1, collected_at=BASE_TIME + timedelta(seconds=30)), [accepted])
    track = tracker._routes[("KMB", "R", "out")][seeded[0].track_id]
    accepted_bundle = track.estimate
    stale = _revision_candidate(5.5, ((7, 130.0, 20), (8, 230.0, 20)), 40)
    held = await tracker.update(
        _snapshot(1, collected_at=BASE_TIME + timedelta(seconds=121)), [stale],
    )
    assert held[0].position == 5.0
    assert track.estimate == accepted_bundle
    assert track.boundary_revision == (10, 10)
    assert track.last_evidence_at == BASE_TIME.timestamp()
    fresh = replace(stale, checkpoint_evidence=((7, 130.0, 40), (8, 230.0, 40)))
    moved = await tracker.update(
        _snapshot(1, collected_at=BASE_TIME + timedelta(seconds=130)), [fresh],
    )
    assert moved[0].track_id == seeded[0].track_id
    assert moved[0].position == 5.5
    assert track.estimate.checkpoint_evidence == fresh.checkpoint_evidence


@pytest.mark.asyncio
@pytest.mark.parametrize("reverse", [False, True])
async def test_stale_extra_candidate_cannot_force_neighbour_motion(reverse):
    tracker = MarkerTracker()
    initial_candidates = [
        _revision_candidate(4.0, ((7, 100.0, 10), (8, 200.0, 10))),
        _revision_candidate(6.0, ((7, 230.0, 10), (8, 330.0, 10))),
        _revision_candidate(10.0, ((15, 600.0, 10), (16, 700.0, 10))),
    ]
    initial = await tracker.update(_snapshot(1), initial_candidates)
    accepted = replace(initial_candidates[0], bracket=None, boundary_revision=None,
                       position_authoritative=False,
                       checkpoint_evidence=((7, 120.0, 30), (8, 220.0, 30)))
    await tracker.update(_snapshot(1, collected_at=BASE_TIME + timedelta(seconds=30)), [accepted])
    old = list(tracker._routes[("KMB", "R", "out")].values())
    held_bundles = [track.estimate for track in old[:2]]
    candidates = [
        _revision_candidate(4.5, ((7, 120.0, 20), (8, 220.0, 20)), 40),
        _revision_candidate(6.5, ((7, 170.0, 40), (8, 270.0, 40)), 40),
        _revision_candidate(10.5, ((15, 620.0, 40), (16, 720.0, 40)), 40),
    ]
    reservations, blocked_old, blocked_new = tracker_module._same_generation_identity_plan(
        old, candidates, BASE_TIME.timestamp() + 60,
    )
    assert reservations == {(2, 2)}
    assert blocked_old == blocked_new == {0, 1}
    current = await tracker.update(
        _snapshot(1, collected_at=BASE_TIME + timedelta(seconds=60)),
        candidates[::-1] if reverse else candidates,
    )
    assert [item.track_id for item in current] == [item.track_id for item in initial]
    assert [item.position for item in current] == [4.0, 6.0, 10.5]
    assert [track.estimate for track in old[:2]] == held_bundles


def test_nonreserved_fallback_transaction_obeys_other_tracks_response_revision_floor():
    old = [_capacity_track(1, (7, 100.0, 10), 4.0),
           _capacity_track(2, (7, 400.0, 30), 6.0)]
    stale = _capacity_candidate((7, 130.0, 20), 4.5)
    accepted, birth_allowed = _select_valid_partial_transaction(
        old, [stale], [stale], [(0, 0)], {0: 4.5},
    )
    assert not accepted
    assert birth_allowed


def test_partial_metadata_transaction_rolls_back_claims_and_blocks_retained_owner_birth():
    first, second, third = ((index, BASE_TIME.timestamp() + index, 10) for index in range(3))
    old = [_capacity_track(1, first, 4.0), _capacity_track(2, second, 5.0)]
    candidate = _capacity_candidate(second, 4.0)
    birth = _capacity_candidate(third, 6.0)
    accepted, birth_allowed, motion = _select_valid_partial_transaction(
        old, [candidate, birth], [candidate], [(0, 0)], {}, metadata_indices={0},
        birth_candidate=birth,
    )
    assert accepted == motion == set()
    assert birth_allowed
    conflicting_birth = _capacity_candidate(second, 6.0)
    accepted, birth_allowed, motion = _select_valid_partial_transaction(
        old, [conflicting_birth], [], [], {}, metadata_indices=set(),
        birth_candidate=conflicting_birth,
    )
    assert accepted == motion == set()
    assert not birth_allowed


def test_unknown_candidate_cannot_demote_sticky_position_authority():
    estimate = BusEstimate(
        label="x", lat=0, lon=0, operator=Operator.KMB, heading=0,
        position=4.5, boundary_revision=(2, 2),
        position_authoritative=False,
    )
    track = _Track(1, estimate, 4.5, 1, position_authoritative=True,
                   boundary_revision=(2, 2))
    unknown = replace(estimate, position_authoritative=False,
                      boundary_revision=None)
    assert _position_order_authoritative(track, unknown)


@pytest.mark.asyncio
async def test_same_generation_gmb12_unknown_frontier_holds_track_and_adopts_search_hints():
    evidence = ((23, BASE_TIME.timestamp(), 1324),
                (24, (BASE_TIME + timedelta(minutes=2)).timestamp(), 1324))
    initial = BusEstimate(
        "12 destination", 22.3, 114.2, Operator.GMB, 0.0,
        route="12", bound="seq-1", operator_code="GMB", position=4.5,
        source_observations=frozenset({("probe", 0)}),
        bracket=(2.0, 7.0), boundary_revision=(1324, 1324),
        source_indices=frozenset({2, 7}),
        checkpoint_evidence=evidence, priority_indices=frozenset({23, 24}),
    )
    tracker = MarkerTracker()
    first = await tracker.update(
        _snapshot(1324, route_key=("GMB", "12", "seq-1")), [initial]
    )
    track_id = first[0].track_id
    cold = replace(
        initial, position=11.2, bracket=None, boundary_revision=None,
        position_authoritative=False,
        source_indices=frozenset({23, 24}),
        source_observations=frozenset({("probe", 1)}),
        priority_indices=frozenset({23, 24}),
        exploratory_indices=frozenset({11, 12}),
    )
    result = await tracker.update(
        _snapshot(1324, route_key=("GMB", "12", "seq-1")), [cold]
    )
    assert len(result) == 1
    assert result[0].track_id == track_id
    assert result[0].position == pytest.approx(4.5)
    assert result[0].bracket == (2.0, 7.0)
    assert result[0].boundary_revision == (1324, 1324)
    assert result[0].checkpoint_evidence == evidence
    assert tracker._routes[("GMB", "12", "seq-1")][track_id].committed_boundary_evidence == evidence
    assert result[0].source_observations == cold.source_observations
    assert result[0].source_indices == frozenset({23, 24})
    assert result[0].exploratory_indices == frozenset({11, 12})
    assert tracker.poll_priorities()[("GMB", "12", "seq-1")] == frozenset({11, 12, 23})


@pytest.mark.asyncio
@pytest.mark.parametrize(("lower", "upper"), [(4, 5), (0, 4)])
async def test_scheduled_lower_omission_holds_existing_physical_boundary(lower, upper):
    tracker = MarkerTracker()
    line = _geometry_line(stops=7)
    key = ("KMB", "R", "out")
    initial_rows = [
        _fresh_probe_row(upper - 2, BASE_TIME - timedelta(minutes=1), 12, BASE_TIME),
        _fresh_probe_row(upper, BASE_TIME + timedelta(minutes=1), 12, BASE_TIME),
    ]
    candidates = estimate_bus_positions(
        initial_rows, [line], observed_checkpoint_indices={key: {upper - 2, upper}},
    )
    assert len(candidates) == 1
    initial = await tracker.update(_snapshot(12, initial_rows), candidates, [line])
    original = initial[0]
    track = tracker._routes[key][original.track_id]
    committed = track.committed_boundary_evidence
    boundary_time = track.boundary_observed_at
    now = BASE_TIME + timedelta(seconds=30)
    current_rows = [
        _fresh_probe_row(lower, now + timedelta(minutes=20), 17, now,
                         kind=EtaKind.SCHEDULED),
        _fresh_probe_row(upper, BASE_TIME + timedelta(minutes=1), 17, now),
    ]
    candidates = estimate_bus_positions(
        current_rows, [line], observed_checkpoint_indices={key: {lower, upper}},
    )
    live = next(candidate for candidate in candidates
                if ("probe", 1) in candidate.source_observations)
    assert live.bracket is None
    assert live.position_authoritative is False
    current = await tracker.update(
        _snapshot(17, current_rows, collected_at=now), candidates, [line],
    )
    held = next(marker for marker in current if marker.track_id == original.track_id)
    assert held.position == pytest.approx(original.position)
    assert held.bracket == original.bracket
    assert held.bracket != (float(lower), float(upper))
    assert held.boundary_revision == original.boundary_revision
    assert held.position_authoritative is False
    assert track.display_bracket == track.motion_bracket == original.bracket
    assert track.boundary_observed_at == boundary_time
    assert track.committed_boundary_evidence == committed
    for name in ("source_indices", "source_observations", "checkpoint_evidence",
                 "priority_indices", "exploratory_indices"):
        assert getattr(held, name) == getattr(live, name)
    assert track.cohort_evidence == live.checkpoint_evidence


def test_no_bracket_poll_prefers_cold_exploratory_hints():
    estimate = BusEstimate(
        label="x", lat=0, lon=0, operator=Operator.KMB, heading=0,
        position=11.4, priority_indices=frozenset({23, 24}),
        exploratory_indices=frozenset({11, 12}), position_authoritative=False,
    )
    track = _Track(1, estimate, 4.5, 1)
    assert _next_poll_checkpoints(track, 30) == (11, 12, 23)


def test_certified_due_frontier_clears_cold_hints_and_restores_bracket_poll():
    estimate = replace(
        _candidate(3.4, bracket=(3.0, 8.0), priority_indices=(3, 8),
                   exploratory_indices=(), boundary_revision=(9, 10),
                   boundary_age=0),
        position_authoritative=True,
    )
    track = _Track(1, estimate, 3.4, 1)
    assert _next_poll_checkpoints(track, 30) == (3, 5, 8)


def test_matching_view_strips_untrusted_authority_but_preserves_legacy_tracks():
    arrival = BASE_TIME + timedelta(minutes=3)
    estimate = replace(
        _candidate(4.0, arrival_at=arrival),
        checkpoint_evidence=((5, arrival.timestamp(), 10),),
        source_indices=frozenset({5}),
    )
    legacy = _Track(1, estimate, 4.0, 1)
    assert _matching_track(legacy) is legacy
    anchor_only = _Track(2, _candidate(4.0, arrival_at=arrival), 4.0, 1,
                         cohort_observed_at=BASE_TIME.timestamp())
    assert _matching_track(anchor_only) is anchor_only
    untrusted = _Track(3, estimate, 4.0, 1, cohort_trusted=False,
                       cohort_observed_at=BASE_TIME.timestamp(),
                       committed_boundary_evidence=estimate.checkpoint_evidence)
    view = _matching_track(untrusted)
    assert view.position == untrusted.position
    assert view.estimate.checkpoint_evidence == ()
    assert view.estimate.eta_arrival_at is None
    assert view.estimate.source_observations == frozenset()
    assert view.estimate.source_indices == frozenset()
    assert view.committed_boundary_evidence == ()
    assert untrusted.estimate is estimate
    assert untrusted.committed_boundary_evidence == estimate.checkpoint_evidence


def _checkpoint_row(index, arrival, revision, minutes=3):
    return SimpleNamespace(
        index=index, arrival_at=arrival, refresh_generation=revision,
        minutes=minutes, cache_age_seconds=0,
    )


def test_missing_instance_uses_injective_committed_witness_multiplicity():
    first = BASE_TIME + timedelta(minutes=3)
    second = first + timedelta(seconds=30)
    track = _Track(
        1, _candidate(4.0), 4.0, 1,
        committed_boundary_evidence=(
            (8, first.timestamp(), 10), (8, second.timestamp(), 10),
        ),
    )
    both = [_checkpoint_row(8, first, 11), _checkpoint_row(8, second, 11)]
    one = [_checkpoint_row(8, first, 11),
           _checkpoint_row(8, first + timedelta(minutes=5), 11)]
    assert _missing_instance(track, 8, both) is False
    assert _missing_instance(track, 8, one) is True


def test_accepted_timestamp_free_boundary_clears_old_committed_witness():
    arrival = BASE_TIME + timedelta(minutes=3)
    track = _Track(
        1, _candidate(4.0), 4.0, 1,
        committed_boundary_evidence=((8, arrival.timestamp(), 10),),
    )
    _commit_boundary_evidence(track, _candidate(5.0))
    assert track.committed_boundary_evidence == ()


def _capacity_track(track_id, row, position):
    estimate = replace(_candidate(position), checkpoint_evidence=(row,))
    return _Track(track_id, estimate, position, 1)


def _capacity_candidate(row, position):
    return replace(_candidate(position), checkpoint_evidence=(row,))


def test_partial_capacity_allows_two_legitimate_owners_of_one_occurrence():
    row = (8, (BASE_TIME + timedelta(minutes=3)).timestamp(), 10)
    old = [_capacity_track(1, row, 4.0), _capacity_track(2, row, 5.0)]
    current = [_capacity_candidate(row, 4.1), _capacity_candidate(row, 5.1)]

    accepted = _retain_current_checkpoint_capacity(
        old, current, current, ((0, 0), (1, 1)), {0, 1}
    )

    assert accepted == {0, 1}


def test_partial_capacity_accepts_simultaneous_safe_swaps_transactionally():
    first = (8, (BASE_TIME + timedelta(minutes=3)).timestamp(), 10)
    second = (12, (BASE_TIME + timedelta(minutes=4)).timestamp(), 10)
    old = [_capacity_track(1, first, 4.0), _capacity_track(2, second, 5.0)]
    candidates = [_capacity_candidate(second, 4.1), _capacity_candidate(first, 5.1)]

    accepted = _retain_current_checkpoint_capacity(
        old, candidates, candidates, ((0, 0), (1, 1)), {0, 1}
    )

    assert accepted == {0, 1}


def test_partial_capacity_is_invariant_to_candidate_input_order():
    first = (8, (BASE_TIME + timedelta(minutes=3)).timestamp(), 10)
    second = (12, (BASE_TIME + timedelta(minutes=4)).timestamp(), 10)
    third = (20, (BASE_TIME + timedelta(minutes=5)).timestamp(), 10)
    old = [
        _capacity_track(1, first, 4.0),
        _capacity_track(2, second, 5.0),
        _capacity_track(3, third, 6.0),
    ]
    # The first claimant would duplicate the retained holder's occurrence;
    # the unrelated second update remains safe and must still be accepted.
    candidates = [
        _capacity_candidate(third, 6.1),
        _capacity_candidate(first, 5.1),
    ]
    reversed_candidates = list(reversed(candidates))

    forward = _retain_current_checkpoint_capacity(
        old, candidates, candidates, ((1, 1), (2, 0)), {1, 2}
    )
    reverse = _retain_current_checkpoint_capacity(
        old, reversed_candidates, reversed_candidates, ((1, 0), (2, 1)), {1, 2}
    )

    assert forward == reverse == {2}


def test_partial_capacity_reselects_order_after_rejecting_dependent_claimant():
    first = (8, (BASE_TIME + timedelta(minutes=3)).timestamp(), 10)
    second = (12, (BASE_TIME + timedelta(minutes=4)).timestamp(), 10)
    third = (20, (BASE_TIME + timedelta(minutes=5)).timestamp(), 10)
    old = [
        _capacity_track(1, first, 4.0),
        _capacity_track(2, second, 5.0),
        _capacity_track(3, third, 8.0),
    ]
    candidates = [_capacity_candidate(first, 6.0), _capacity_candidate(third, 7.0)]

    accepted, birth_allowed = _select_valid_partial_transaction(
        old, candidates, candidates, ((0, 0), (1, 1)),
        {0: 6.0, 1: 7.0},
    )

    assert accepted == set()
    assert birth_allowed is True


def test_partial_birth_does_not_block_on_preexisting_unrelated_overcapacity():
    first = (8, (BASE_TIME + timedelta(minutes=3)).timestamp(), 10)
    second = (12, (BASE_TIME + timedelta(minutes=4)).timestamp(), 10)
    birth = (20, (BASE_TIME + timedelta(minutes=5)).timestamp(), 10)
    old = [_capacity_track(1, first, 4.0), _capacity_track(2, first, 5.0)]
    birth_candidate = _capacity_candidate(birth, 6.0)
    current = [
        _capacity_candidate(first, 4.0),
        _capacity_candidate(second, 5.0),
        birth_candidate,
    ]

    accepted, birth_allowed = _select_valid_partial_transaction(
        old, current, current, (), {}, birth_candidate=birth_candidate
    )

    assert accepted == set()
    assert birth_allowed is True


def test_missing_instance_unrelated_later_bus_is_disappearance():
    arrival = BASE_TIME + timedelta(minutes=3)
    track = _Track(
        1, _candidate(4.0), 4.0, 1,
        committed_boundary_evidence=((8, arrival.timestamp(), 10),),
    )
    current = [_checkpoint_row(8, arrival + timedelta(minutes=4), 11)]
    assert _missing_instance(track, 8, current) is True


def test_temporal_owner_ambiguity_does_not_create_exclusive_owner():
    arrival = BASE_TIME + timedelta(minutes=3)
    shared = replace(
        _candidate(5.0),
        checkpoint_evidence=((8, arrival.timestamp(), 10),),
    )
    candidate = replace(
        _candidate(5.5),
        checkpoint_evidence=((8, (arrival + timedelta(seconds=20)).timestamp(), 11),),
    )
    profile = _checkpoint_ownership_profile(
        [_Track(1, shared, 5.0, 1), _Track(2, shared, 5.0, 1)], candidate
    )
    assert not profile[-1]


def test_temporal_owner_degree_two_does_not_infer_exclusive_mixed_owner():
    arrival = BASE_TIME + timedelta(minutes=3)
    owner_a = replace(
        _candidate(4.0),
        checkpoint_evidence=((2, arrival.timestamp(), 10),
                             (8, (arrival + timedelta(seconds=20)).timestamp(), 10)),
    )
    owner_b = replace(
        _candidate(8.0),
        checkpoint_evidence=((8, arrival.timestamp(), 10),
                             (8, (arrival + timedelta(seconds=40)).timestamp(), 10)),
    )
    candidate = replace(
        _candidate(4.5),
        checkpoint_evidence=((2, arrival.timestamp(), 10),
                             (8, (arrival + timedelta(seconds=20)).timestamp(), 11)),
    )
    profile = _checkpoint_ownership_profile(
        [_Track(1, owner_a, 4.0, 1), _Track(2, owner_b, 8.0, 1)], candidate
    )
    assert not profile[-1]


def test_temporal_hall_assignment_blocks_false_mixed_ownership():
    base = BASE_TIME
    def est(position, arrivals):
        return replace(_candidate(position), checkpoint_evidence=tuple(
            (8, (base + timedelta(seconds=offset)).timestamp(), 10)
            for offset in arrivals
        ))
    owner_a = est(4.0, (-10,))
    owner_b = est(8.0, (50, 110))
    candidate = replace(_candidate(6.0), checkpoint_evidence=(
        (8, base.timestamp(), 11),
        (8, (base + timedelta(seconds=100)).timestamp(), 11),
    ))
    profile = _checkpoint_ownership_profile(
        [_Track(1, owner_a, 4.0, 1), _Track(2, owner_b, 8.0, 1)], candidate
    )
    assert not profile[-1]


def test_temporal_ownership_profile_is_order_invariant_and_row_local():
    base = BASE_TIME
    def est(position, rows):
        return replace(_candidate(position), checkpoint_evidence=tuple(
            (index, (base + timedelta(seconds=offset)).timestamp(), 10)
            for index, offset in rows
        ))
    owner_a = est(4.0, ((2, 0), (8, 200)))
    owner_b = est(8.0, ((8, 50), (8, 100), (9, 100)))
    candidate = replace(_candidate(6.0), checkpoint_evidence=(
        (2, base.timestamp(), 10),
        (8, (base + timedelta(seconds=110)).timestamp(), 11),
        (8, (base + timedelta(seconds=200)).timestamp(), 11),
        (9, (base + timedelta(seconds=100)).timestamp(), 11),
    ))
    tracks = [_Track(1, owner_a, 4.0, 1), _Track(2, owner_b, 8.0, 1)]
    forward = _checkpoint_ownership_profile(tracks, candidate)
    reverse = _checkpoint_ownership_profile(tuple(reversed(tracks)), candidate)
    assert forward == reverse
    assert forward[-1] == frozenset({1, 2})


@pytest.mark.asyncio
async def test_held_complete_replacement_keeps_boundary_witness_for_forward_search():
    arrival = BASE_TIME + timedelta(minutes=3)
    initial = replace(
        _candidate(5.0, bracket=(3.0, 8.0), boundary_age=0,
                   boundary_revision=(10, 10), arrival_at=arrival),
        checkpoint_evidence=((8, arrival.timestamp(), 10),),
    )
    tracker = MarkerTracker()
    await tracker.update(_snapshot(1), [initial], [_line(stops=12)])
    replacement = replace(
        _candidate(8.0, bracket=None, boundary_age=30,
                   boundary_revision=(10, 10), arrival_at=arrival),
        checkpoint_evidence=((8, (arrival + timedelta(minutes=4)).timestamp(), 11),),
    )
    newer = ProbeEta(
        "KMB", "R", "out", "8", 8, 3,
        arrival_at=arrival + timedelta(minutes=4), refresh_generation=11,
    )
    snapshot = ProbeEtaSnapshot(
        (ProbeRouteGeneration(("KMB", "R", "out"), (newer,), 2,
                               BASE_TIME + timedelta(seconds=30)),),
        BASE_TIME + timedelta(seconds=30),
    )
    current = await tracker.update(snapshot, [replacement], [_line(stops=12)])
    assert current[0].position == pytest.approx(5.0)
    track = next(iter(tracker._routes[("KMB", "R", "out")].values()))
    # The committed witness is retained even though the complete candidate's
    # checkpoint population belongs to another vehicle.
    assert track.committed_boundary_evidence == ((8, arrival.timestamp(), 10),)
    assert track.forward_after == 8


@pytest.mark.asyncio
async def test_same_generation_mixed_exact_and_temporal_ownership_holds_both_tracks():
    arrival = BASE_TIME + timedelta(minutes=3)
    a = replace(_candidate(4.0, bracket=(3.0, 4.0), boundary_age=0,
                          boundary_revision=(10, 10), arrival_at=arrival),
                checkpoint_evidence=((2, arrival.timestamp(), 10),
                                     (3, (arrival + timedelta(seconds=30)).timestamp(), 10)))
    b = replace(_candidate(8.0, bracket=(7.0, 8.0), boundary_age=0,
                          boundary_revision=(10, 10), arrival_at=arrival + timedelta(minutes=1)),
                    checkpoint_evidence=((13, (arrival + timedelta(minutes=1)).timestamp(), 10),
                                         (14, (arrival + timedelta(minutes=1, seconds=30)).timestamp(), 10)))
    tracker = MarkerTracker()
    initial = await tracker.update(_snapshot(1, [a, b]), [a, b])
    mixed = replace(
        _candidate(4.5, bracket=(3.5, 4.5), boundary_age=0,
                   boundary_revision=(11, 11), arrival_at=arrival),
        checkpoint_evidence=(
            (2, arrival.timestamp(), 10),
            (3, (arrival + timedelta(seconds=30)).timestamp(), 10),
            (13, (arrival + timedelta(minutes=1, seconds=20)).timestamp(), 11),
            (14, (arrival + timedelta(minutes=1, seconds=50)).timestamp(), 11),
        ),
    )
    partial = await tracker.update(
        _snapshot(1, [mixed], collected_at=BASE_TIME + timedelta(seconds=30)),
        [mixed],
    )
    assert [marker.track_id for marker in partial] == [marker.track_id for marker in initial]
    assert [marker.position for marker in partial] == pytest.approx([4.0, 8.0])
    assert tracker.poll_lifecycle_routes() == {("KMB", "R", "out")}
    a2 = replace(a, position=4.4, bracket=(3.4, 4.4),
                 boundary_revision=(12, 12),
                 checkpoint_evidence=((2, (arrival + timedelta(seconds=5)).timestamp(), 12),))
    b2 = replace(b, position=8.4, bracket=(7.4, 8.4),
                 boundary_revision=(12, 12),
                 checkpoint_evidence=((13, (arrival + timedelta(minutes=1, seconds=5)).timestamp(), 12),))
    final = await tracker.update(_snapshot(2, [a2, b2]), [a2, b2])
    by_id = {marker.track_id: marker for marker in final}
    assert set(by_id) == {marker.track_id for marker in initial}
    assert len(final) == 2
    assert by_id[initial[0].track_id].position == pytest.approx(4.4)
    assert by_id[initial[1].track_id].position == pytest.approx(8.4)
    assert (2, (arrival + timedelta(seconds=5)).timestamp(), 12) in \
        by_id[initial[0].track_id].checkpoint_evidence
    assert (13, (arrival + timedelta(minutes=1, seconds=5)).timestamp(), 12) in \
        by_id[initial[1].track_id].checkpoint_evidence


@pytest.mark.asyncio
async def test_same_generation_partial_update_cannot_duplicate_retained_eta_occurrence():
    tracker = MarkerTracker()
    route_key = ("GMB", "11", "seq-1")
    shared_16 = (16, (BASE_TIME + timedelta(minutes=10)).timestamp(), 325)
    shared_17 = (17, (BASE_TIME + timedelta(minutes=11)).timestamp(), 362)
    retained_mixed = replace(
        _candidate(
            9.0,
            route="11",
            operator=Operator.GMB,
            bound="seq-1",
            bracket=(8.0, 10.0),
            boundary_age=0,
            boundary_revision=(321, 396),
        ),
        checkpoint_evidence=(
            (10, (BASE_TIME + timedelta(minutes=20)).timestamp(), 396),
            shared_16,
            shared_17,
        ),
    )
    neighbour = replace(
        _candidate(
            15.0,
            route="11",
            operator=Operator.GMB,
            bound="seq-1",
            bracket=(15.0, 16.0),
            boundary_age=0,
            boundary_revision=(322, 325),
        ),
        checkpoint_evidence=(
            (15, (BASE_TIME + timedelta(minutes=2)).timestamp(), 322),
            (16, (BASE_TIME + timedelta(minutes=3)).timestamp(), 325),
            (17, (BASE_TIME + timedelta(minutes=4)).timestamp(), 362),
        ),
    )
    initial = await tracker.update(
        _snapshot(401, route_key=route_key),
        [retained_mixed, neighbour],
    )
    assert not tracker._routes[route_key][initial[0].track_id].cohort_trusted

    claimant = replace(
        _candidate(
            14.0,
            route="11",
            operator=Operator.GMB,
            bound="seq-1",
            bracket=(10.0, 15.0),
            boundary_age=0,
            boundary_revision=(396, 433),
        ),
        checkpoint_evidence=(
            (15, (BASE_TIME + timedelta(minutes=20)).timestamp(), 433),
            shared_16,
            shared_17,
        ),
    )
    current = await tracker.update(
        _snapshot(
            401,
            collected_at=BASE_TIME + timedelta(seconds=30),
            route_key=route_key,
        ),
        [claimant],
    )

    current_capacity = Counter(claimant.checkpoint_evidence)
    displayed = Counter(
        row for marker in current for row in marker.checkpoint_evidence
        if row in current_capacity
    )
    assert all(
        count <= current_capacity[row] for row, count in displayed.items()
    )
    assert [marker.position for marker in current] == pytest.approx([9.0, 15.0])
    assert tracker.poll_lifecycle_routes() == {route_key}


@pytest.mark.asyncio
async def test_same_position_boundary_refresh_advances_committed_witness():
    tracker = MarkerTracker()
    candidates = [
        replace(_candidate(5.0, bracket=(0.0, 6.0), boundary_age=0,
                           boundary_revision=(revision, revision),
                           arrival_at=BASE_TIME + timedelta(seconds=seconds)),
                 checkpoint_evidence=((6, (BASE_TIME + timedelta(seconds=seconds)).timestamp(), revision),))
        for revision, seconds in ((1, 300), (2, 360), (3, 420))
    ]
    first = await tracker.update(_snapshot(1), [candidates[0]])
    second = await tracker.update(_snapshot(2), [candidates[1]])
    third = await tracker.update(_snapshot(3), [candidates[2]])
    assert [marker.track_id for marker in (first[0], second[0], third[0])] == [1, 1, 1]
    assert third[0].boundary_revision == (3, 3)
    track = next(iter(tracker._routes[("KMB", "R", "out")].values()))
    assert track.committed_boundary_evidence == ((6, (BASE_TIME + timedelta(seconds=420)).timestamp(), 3),)
    assert track.forward_after is None


@pytest.mark.asyncio
async def test_stable_gate_move_keeps_motion_boundary_witness_for_recovery():
    arrival = BASE_TIME + timedelta(minutes=3)
    old = replace(
        _candidate(2.0, gate=True, bracket=(0.0, 3.0), boundary_age=0,
                   boundary_revision=(10, 10), arrival_at=arrival),
        checkpoint_evidence=((2, arrival.timestamp(), 10),
                             (3, (arrival + timedelta(seconds=30)).timestamp(), 10)),
    )
    tracker = MarkerTracker()
    first = await tracker.update(_snapshot(1), [old], [_line(stops=9)])
    gate = replace(
        _candidate(5.0, gate=True, bracket=(4.0, 6.0), boundary_age=None,
                   boundary_revision=None, arrival_at=arrival),
        checkpoint_evidence=((2, arrival.timestamp(), 11),
                             (3, (arrival + timedelta(seconds=30)).timestamp(), 11)),
    )
    second = await tracker.update(_snapshot(2), [gate], [_line(stops=9)])
    track = next(iter(tracker._routes[("KMB", "R", "out")].values()))
    assert second[0].track_id == first[0].track_id
    assert second[0].position == pytest.approx(5.0)
    assert second[0].bracket == (4.0, 6.0)
    assert track.motion_bracket == (0.0, 3.0)
    assert track.boundary_revision == (10, 10)
    assert track.committed_boundary_evidence == (
        (2, arrival.timestamp(), 10),
        (3, (arrival + timedelta(seconds=30)).timestamp(), 10),
    )
    unrelated = ProbeEta(
        "KMB", "R", "out", "3", 3, 3,
        arrival_at=arrival + timedelta(minutes=4), refresh_generation=12,
    )
    recovery = ProbeEtaSnapshot((), BASE_TIME + timedelta(seconds=60),
                                positioning_rows=(unrelated,))
    await tracker.update(recovery, [], [_line(stops=9)])
    assert track.forward_after == 3


@pytest.mark.asyncio
async def test_equal_count_mixed_candidates_request_lifecycle_refresh():
    arrival = BASE_TIME + timedelta(minutes=3)
    def make(position, stops, revision=10):
        return replace(
            _candidate(position, bracket=(position - 1, position), boundary_age=0,
                       boundary_revision=(revision, revision), arrival_at=arrival),
            checkpoint_evidence=tuple(
                (stop, (arrival + timedelta(seconds=offset)).timestamp(), revision)
                for stop, offset in stops
            ),
        )
    old = [make(3.0, ((2, 0), (3, 20))),
           make(7.0, ((8, 60), (9, 80))),
           make(11.0, ((13, 120), (14, 140)))]
    tracker = MarkerTracker()
    initial = await tracker.update(_snapshot(1), old)
    mixed = [
        make(3.2, ((2, 0), (3, 20), (8, 70), (9, 90)), 11),
        make(7.2, ((8, 60), (9, 80), (13, 130), (14, 150)), 11),
        make(11.2, ((13, 120), (14, 140), (2, 10), (3, 30)), 11),
    ]
    partial = await tracker.update(_snapshot(1), mixed)
    key = ("KMB", "R", "out")
    assert [marker.track_id for marker in partial] == [marker.track_id for marker in initial]
    assert [marker.position for marker in partial] == pytest.approx([3.0, 7.0, 11.0])
    assert key in tracker.poll_lifecycle_routes()
    final = await tracker.update(_snapshot(2), old)
    assert key not in tracker.poll_lifecycle_routes()
    assert {marker.track_id for marker in final} == {marker.track_id for marker in initial}
    assert len(final) == 3


def test_large_connected_tie_uses_unique_global_weighted_assignment():
    size = 13
    overlap = [[1 for _column in range(size)] for _row in range(size)]
    overlap[0][0], overlap[0][1] = 3, 4
    overlap[1][0], overlap[1][1] = 4, 4
    for index in range(2, size):
        overlap[index][index] = 5

    pairs = _unique_tied_checkpoint_assignment(
        overlap, list(range(size)), list(range(size))
    )

    assert set(pairs) == {
        (0, 1),
        (1, 0),
        *((index, index) for index in range(2, size)),
    }
    assert _unique_tied_checkpoint_assignment(
        [[1 for _column in range(size)] for _row in range(size)],
        list(range(size)),
        list(range(size)),
    ) == ()


def _candidate(position, *, gate=False, route="R", operator=Operator.KMB,
               bound="out", unreliable=False, scheduled=False, bracket=None,
               boundary_age=None, arrival_at=None, priority_indices=(),
               exploratory_indices=(), boundary_revision=None):
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
        exploratory_indices=frozenset(exploratory_indices),
    )


def _snapshot(generation, rows=(), *, collected_at=BASE_TIME, route_key=None):
    key = route_key or ("KMB", "R", "out")
    route = ProbeRouteGeneration(key, tuple(rows), generation, collected_at)
    return ProbeEtaSnapshot((route,), collected_at)


def _atomic_snapshot(generation, rows, *, collected_at=BASE_TIME, route_key=None,
                     observed_indices=()):
    key = route_key or ("KMB", "R", "out")
    rows = tuple(rows)
    revisions = {}
    for row in rows:
        revisions.setdefault(row.index, row.refresh_generation)
        assert revisions[row.index] == row.refresh_generation
    observed = frozenset(observed_indices) | frozenset(revisions)
    for index in observed:
        revisions.setdefault(index, next(iter(revisions.values())))
    route = ProbeRouteGeneration(
        key,
        rows,
        generation,
        collected_at,
        observed_checkpoint_indices=observed,
        checkpoint_revisions=tuple(sorted(revisions.items())),
    )
    return ProbeEtaSnapshot((route,), collected_at)


def _fresh_probe_row(index, arrival, revision, collected_at, *, minutes=None,
                     kind=EtaKind.REALTIME, age=0.0, operator="KMB", route="R",
                     bound="out"):
    signed = (
        (arrival - collected_at).total_seconds() / 60.0
        if minutes is None else minutes
    )
    return ProbeEta(
        operator,
        route,
        bound,
        f"s{index}",
        index,
        max(0.0, signed),
        kind=kind,
        cache_age_seconds=age,
        arrival_at=arrival,
        observed_at=collected_at - timedelta(seconds=age),
        refresh_generation=revision,
        signed_minutes=signed,
    )


def _candidate_from_rows(position, bracket, revision, rows, slots, *,
                         arrival_slot=None, unreliable=False, route="R",
                         operator=Operator.KMB, bound="out"):
    selected = [rows[slot] for slot in slots]
    arrival_slot = slots[0] if arrival_slot is None else arrival_slot
    return replace(
        _candidate(
            position,
            unreliable=unreliable,
            bracket=bracket,
            boundary_age=selected[0].cache_age_seconds,
            boundary_revision=revision,
            arrival_at=rows[arrival_slot].arrival_at,
            route=route,
            operator=operator,
            bound=bound,
        ),
        source_indices=frozenset(row.index for row in selected),
        source_observations=frozenset(("probe", slot) for slot in slots),
        checkpoint_evidence=tuple(
            (row.index, row.arrival_at.timestamp(), row.refresh_generation)
            for row in selected
        ),
    )


def _omitted(collected_at):
    return ProbeEtaSnapshot((), collected_at)


def _line(*, stops=3, route="R", bound="out", operator="KMB"):
    return SimpleNamespace(operator=operator, route=route, bound=bound,
                           stops=tuple(range(stops)))


def _geometry_line(*, stops=11, route="R", bound="out", operator="KMB"):
    route_stops = [
        Stop(str(index), f"Stop {index}", 22.3, 114.2 + index * 0.001)
        for index in range(stops)
    ]
    return RouteLine(
        route,
        operator,
        bound,
        route_stops,
        [(stop.lat, stop.lon) for stop in route_stops],
        [index * 100.0 for index in range(stops)],
    )


def _crossed_first_boundary_frames():
    first_at = BASE_TIME
    second_at = BASE_TIME + timedelta(seconds=30)
    third_at = BASE_TIME + timedelta(minutes=1)

    def add_ladder(rows, collected_at, revision, age, values):
        slots = []
        for index, signed_minutes in values:
            slots.append(len(rows))
            rows.append(_fresh_probe_row(
                index,
                collected_at + timedelta(minutes=signed_minutes),
                revision,
                collected_at,
                age=age,
            ))
        return tuple(slots)

    old_rows = []
    old_a = add_ladder(old_rows, second_at, 8, 19.188, (
        (13, 13.947986633333333),
        (27, 37.73131996666667),
    ))
    old_b = add_ladder(old_rows, second_at, 8, 19.188, (
        (27, 14.631319966666666),
    ))
    old_c = add_ladder(old_rows, second_at, 8, 19.188, (
        (27, 20.88131996666667),
    ))
    old_candidates = [
        replace(
            _candidate_from_rows(
                12.0, (0.0, 13.0), (8, 8), old_rows, old_a,
            ),
            source_observations=frozenset({("gate", 4), *(
                ("probe", slot) for slot in old_a
            )}),
        ),
        replace(
            _candidate_from_rows(
                15.0, (13.0, 16.0), None, old_rows, old_b,
            ),
            source_observations=frozenset({
                ("gate", 3), *(("probe", slot) for slot in old_b),
            }),
            eta_arrival_at=None,
            boundary_age_seconds=None,
        ),
        _candidate_from_rows(
            26.0, (16.0, 27.0), (8, 8), old_rows, old_c,
        ),
    ]

    current_rows = []
    current_a = add_ladder(current_rows, second_at, 37, 9.704, (
        (6, 4.431319966666666),
        (7, 5.981319966666666),
        (13, 13.947986633333333),
        (14, 15.781319966666667),
        (15, 16.781319966666665),
        (17, 20.247986633333333),
        (21, 27.297986633333334),
        (22, 29.647986633333336),
        (27, 37.73131996666667),
    ))
    current_c = add_ladder(current_rows, second_at, 37, 9.704, (
        (14, -1.1186800333333333),
        (15, -0.10201336666666667),
        (17, 3.3479866333333335),
        (21, 10.397986633333334),
        (22, 12.764653299999999),
        (27, 20.83131996666667),
    ))
    current_b = add_ladder(current_rows, second_at, 37, 9.704, (
        (21, 4.2146533),
        (22, 6.5646533),
        (27, 14.631319966666666),
    ))
    current_candidates = [
        replace(
            _candidate_from_rows(
                5.0, (0.0, 6.0), (37, 37), current_rows, current_a,
            ),
            source_observations=frozenset({("gate", 4), *(
                ("probe", slot) for slot in current_a
            )}),
        ),
        replace(
            _candidate_from_rows(
                15.059138183574879,
                (15.0, 17.0),
                (37, 37),
                current_rows,
                current_c,
                arrival_slot=current_c[2],
            ),
            source_observations=frozenset({("gate", 3), *(
                ("probe", slot) for slot in current_c
            )}),
        ),
        _candidate_from_rows(
            20.0, (17.0, 21.0), (37, 37), current_rows, current_b,
        ),
    ]

    final_rows = []
    final_a = add_ladder(final_rows, third_at, 85, 9.766, (
        (3, 5.3305557666666665),
        (4, 8.9138891),
        (6, 3.9305557666666666),
        (7, 5.480555766666667),
        (13, 13.447222433333334),
        (14, 15.280555766666668),
        (15, 16.280555766666666),
        (17, 19.74722243333333),
        (19, 24.74722243333333),
        (20, 25.9138891),
        (21, 26.79722243333333),
        (22, 29.147222433333333),
        (27, 37.23055576666666),
    ))
    final_c = add_ladder(final_rows, third_at, 85, 9.766, (
        (14, -1.5861109),
        (15, -0.2861109),
        (17, 3.1805557666666666),
        (19, 8.180555766666666),
        (20, 9.347222433333334),
        (21, 10.230555766666667),
        (22, 12.580555766666667),
        (27, 20.6638891),
    ))
    final_b = add_ladder(final_rows, third_at, 85, 9.766, (
        (19, 1.6805557666666668),
        (20, 2.830555766666667),
        (21, 3.730555766666667),
        (22, 6.0805557666666665),
        (27, 14.147222433333333),
    ))
    final_candidates = [
        replace(
            _candidate_from_rows(
                2.0, (0.0, 3.0), (85, 85), final_rows, final_a,
            ),
            source_observations=frozenset({("gate", 4), *(
                ("probe", slot) for slot in final_a
            )}),
        ),
        replace(
            _candidate_from_rows(
                15.165063980769231,
                (15.0, 17.0),
                (85, 85),
                final_rows,
                final_c,
                arrival_slot=final_c[2],
            ),
            source_observations=frozenset({("gate", 3), *(
                ("probe", slot) for slot in final_c
            )}),
        ),
        _candidate_from_rows(
            18.159722116666668,
            (17.0, 19.0),
            (85, 85),
            final_rows,
            final_b,
        ),
    ]
    current_observed = {0, 6, 7, 13, 14, 15, 16, 17, 21, 22, 27}
    final_observed = {0, 3, 4, 6, 7, 13, 14, 15, 16, 17, 19, 20, 21, 22, 27}
    return (
        _atomic_snapshot(1, old_rows, collected_at=first_at),
        old_candidates,
        _atomic_snapshot(
            2,
            current_rows,
            collected_at=second_at,
            observed_indices=current_observed,
        ),
        current_candidates,
        _atomic_snapshot(
            3,
            final_rows,
            collected_at=third_at,
            observed_indices=final_observed,
        ),
        final_candidates,
    )


def _citybus_fast_first_boundary_frames():
    """Rebuild the live CTB 792M frame 9 -> 12 -> 15 ETA ladders."""
    key = ("CTB", "792M", "inbound")
    first_at = datetime(2026, 9, 9, 9, 2, 23, 241366, tzinfo=UTC)
    crossed_at = datetime(2026, 9, 9, 9, 2, 53, 218595, tzinfo=UTC)
    final_at = datetime(2026, 9, 9, 9, 3, 23, 230359, tzinfo=UTC)

    def population(collected_at, specs, empty=()):
        rows = []
        candidates = []
        for spec in specs:
            slots = []
            for index, arrival, revision in spec["rows"]:
                slots.append(len(rows))
                rows.append(_fresh_probe_row(
                    index,
                    datetime.fromtimestamp(arrival, UTC),
                    revision,
                    collected_at,
                    age=9.0,
                    operator=key[0],
                    route=key[1],
                    bound=key[2],
                ))
            upper = int(spec["bracket"][1])
            arrival_slot = next(
                (slot for slot in slots if rows[slot].index == upper),
                slots[0],
            )
            candidate = _candidate_from_rows(
                spec["position"],
                spec["bracket"],
                spec["revision"],
                rows,
                slots,
                arrival_slot=arrival_slot,
                route=key[1],
                operator=Operator.CITYBUS,
                bound=key[2],
            )
            if spec.get("gate") is not None:
                candidate = replace(
                    candidate,
                    source_observations=frozenset({
                        ("gate", spec["gate"]),
                        *(("probe", slot) for slot in slots),
                    }),
                )
            if spec["revision"] is None:
                candidate = replace(
                    candidate,
                    boundary_age_seconds=None,
                    eta_arrival_at=None,
                )
            candidates.append(candidate)
        for index, revision in empty:
            rows.append(replace(
                _fresh_probe_row(
                    index,
                    collected_at,
                    revision,
                    collected_at,
                    age=9.0,
                    operator=key[0],
                    route=key[1],
                    bound=key[2],
                ),
                minutes=None,
                signed_minutes=None,
                arrival_at=None,
            ))
        return tuple(rows), candidates

    def snapshot(generation, collected_at, positioning_rows):
        revisions = {}
        for row in positioning_rows:
            if row.index in revisions:
                assert revisions[row.index] == row.refresh_generation
            revisions[row.index] = row.refresh_generation
        observed = frozenset(revisions)
        route = ProbeRouteGeneration(
            key,
            tuple(row for row in positioning_rows if row.minutes is not None),
            generation,
            collected_at,
            observed_checkpoint_indices=observed,
            checkpoint_revisions=tuple(sorted(revisions.items())),
        )
        return ProbeEtaSnapshot(
            (route,),
            collected_at,
            positioning_rows=tuple(positioning_rows),
            positioning_checkpoints=frozenset(
                (*key, index) for index in observed
            ),
        )

    old_rows, old_candidates = population(first_at, (
        {
            "position": 6.593515725,
            "bracket": (4.0, 7.0),
            "revision": (99, 100),
            "gate": 8,
            "rows": (
                (7, 1788944582.0, 100),
                (8, 1788944648.0, 102),
                (14, 1788944962.0, 101),
                (15, 1788945128.0, 103),
                (22, 1788945890.0, 85),
                (23, 1788945997.0, 104),
                (24, 1788946180.0, 86),
                (25, 1788946493.0, 105),
                (29, 1788946916.0, 106),
            ),
        },
        {
            "position": 16.0,
            "bracket": (15.0, 16.0),
            "revision": None,
            "gate": 7,
            "rows": (
                (22, 1788945145.0, 85),
                (23, 1788945222.0, 104),
                (24, 1788945429.0, 86),
                (25, 1788945717.0, 105),
                (29, 1788946140.0, 106),
            ),
        },
        {
            "position": 25.202725684210527,
            "bracket": (25.0, 29.0),
            "revision": (105, 106),
            "rows": (
                (25, 1788944513.0, 105),
                (29, 1788944912.0, 106),
            ),
        },
    ), empty=((4, 99),))
    crossed_rows, crossed_candidates = population(crossed_at, (
        {
            "position": 6.843633808333333,
            "bracket": (6.0, 7.0),
            "revision": (137, 136),
            "gate": 7,
            "rows": (
                (7, 1788944582.0, 136),
                (8, 1788944625.0, 138),
                (14, 1788944942.0, 139),
                (15, 1788945108.0, 140),
                (22, 1788945890.0, 123),
                (23, 1788945972.0, 141),
                (25, 1788946467.0, 143),
                (27, 1788946682.0, 145),
                (28, 1788946744.0, 127),
                (29, 1788946890.0, 142),
            ),
        },
        {
            "position": 21.0,
            "bracket": (16.0, 22.0),
            "revision": (144, 123),
            "rows": (
                (22, 1788945145.0, 123),
                (23, 1788945221.0, 141),
                (25, 1788945716.0, 143),
                (27, 1788945931.0, 145),
                (28, 1788945994.0, 127),
                (29, 1788946140.0, 142),
            ),
        },
        {
            "position": 25.241493368421054,
            "bracket": (25.0, 27.0),
            "revision": (143, 145),
            "rows": (
                (25, 1788944538.0, 143),
                (27, 1788944747.0, 145),
                (28, 1788944804.0, 127),
                (29, 1788944939.0, 142),
            ),
        },
    ), empty=((6, 137), (16, 144)))
    final_rows, final_candidates = population(final_at, (
        {
            "position": 7.979774486486487,
            "bracket": (7.0, 8.0),
            "revision": (178, 179),
            "gate": 7,
            "rows": (
                (7, 1788944557.0, 178),
                (8, 1788944594.0, 179),
                (14, 1788944918.0, 182),
                (19, 1788945583.0, 185),
                (20, 1788945648.0, 184),
                (22, 1788945859.0, 183),
                (23, 1788945941.0, 186),
                (25, 1788946436.0, 188),
                (26, 1788946546.0, 168),
                (27, 1788946651.0, 190),
                (29, 1788946859.0, 189),
            ),
        },
        {
            "position": 18.0,
            "bracket": (16.0, 19.0),
            "revision": (187, 185),
            "rows": (
                (19, 1788944895.0, 185),
                (20, 1788944955.0, 184),
                (22, 1788945150.0, 183),
                (23, 1788945225.0, 186),
                (25, 1788945716.0, 188),
                (26, 1788945826.0, 168),
                (27, 1788945931.0, 190),
                (29, 1788946139.0, 189),
            ),
        },
        {
            "position": 25.535422125,
            "bracket": (25.0, 26.0),
            "revision": (188, 168),
            "rows": (
                (26, 1788944649.0, 168),
                (27, 1788944752.0, 190),
                (29, 1788944945.0, 189),
            ),
        },
    ), empty=((16, 187), (25, 188)))
    initial_snapshot = snapshot(121, first_at, old_rows)
    crossed_snapshot = snapshot(159, crossed_at, crossed_rows)
    partial_snapshot = replace(
        initial_snapshot,
        collected_at=crossed_at,
        positioning_rows=crossed_snapshot.positioning_rows,
        positioning_checkpoints=crossed_snapshot.positioning_checkpoints,
    )
    return (
        initial_snapshot,
        old_candidates,
        partial_snapshot,
        crossed_snapshot,
        crossed_candidates,
        snapshot(204, final_at, final_rows),
        final_candidates,
    )


@pytest.mark.asyncio
async def test_complete_census_atomically_reseeds_crossed_first_boundary_track():
    (
        initial_snapshot,
        initial_candidates,
        crossed_snapshot,
        crossed_candidates,
        final_snapshot,
        final_candidates,
    ) = _crossed_first_boundary_frames()
    tracker = MarkerTracker(max_tracks_per_route=3)
    line = _line(stops=28)

    initial = await tracker.update(initial_snapshot, initial_candidates, [line])
    initial_ids = [marker.track_id for marker in initial]
    assert [marker.position for marker in initial] == pytest.approx([12.0, 15.0, 26.0])

    crossed = await tracker.update(crossed_snapshot, crossed_candidates, [line])
    assert [marker.position for marker in crossed] == pytest.approx(
        [5.0, 15.059138183574879, 20.0]
    )
    crossed_by_position = sorted(crossed, key=lambda marker: marker.position)
    assert crossed_by_position[0].track_id == initial_ids[0]
    assert initial_ids[1] not in {marker.track_id for marker in crossed}
    assert initial_ids[2] not in {marker.track_id for marker in crossed}
    target_id = crossed_by_position[2].track_id
    assert target_id not in initial_ids
    assert crossed_by_position[2].checkpoint_evidence == \
        crossed_candidates[2].checkpoint_evidence
    assert Counter(
        row
        for marker in crossed
        for row in marker.checkpoint_evidence
    ) == Counter(
        row
        for candidate in crossed_candidates
        for row in candidate.checkpoint_evidence
    )
    middle_id = crossed_by_position[1].track_id

    replay = await tracker.update(crossed_snapshot, crossed_candidates, [line])
    assert [marker.track_id for marker in replay] == [
        marker.track_id for marker in crossed
    ]
    assert [marker.position for marker in replay] == pytest.approx(
        [marker.position for marker in crossed]
    )

    final = await tracker.update(final_snapshot, final_candidates, [line])
    assert [marker.position for marker in final] == pytest.approx(
        [2.0, 15.165063980769231, 18.159722116666668]
    )
    final_by_position = sorted(final, key=lambda marker: marker.position)
    assert final_by_position[0].track_id not in {
        marker.track_id for marker in crossed
    }
    assert final_by_position[1].track_id == middle_id
    assert final_by_position[2].track_id == target_id
    assert final_by_position[2].position == pytest.approx(18.159722116666668)
    assert final_by_position[2].bracket == (17.0, 19.0)
    assert Counter(
        row
        for marker in final
        for row in marker.checkpoint_evidence
    ) == Counter(
        row
        for candidate in final_candidates
        for row in candidate.checkpoint_evidence
    )
    route_tracks = tracker._routes[("KMB", "R", "out")]
    assert route_tracks[final_by_position[0].track_id].cohort_trusted is False


@pytest.mark.asyncio
async def test_citybus_per_stop_revisions_reseed_fast_first_boundary_track():
    (
        initial_snapshot,
        initial_candidates,
        partial_snapshot,
        crossed_snapshot,
        crossed_candidates,
        final_snapshot,
        final_candidates,
    ) = _citybus_fast_first_boundary_frames()
    tracker = MarkerTracker(max_tracks_per_route=3)
    line = _line(
        stops=30,
        route="792M",
        bound="inbound",
        operator="CTB",
    )

    initial = await tracker.update(initial_snapshot, initial_candidates, [line])
    target_old_id = initial[1].track_id
    for _ in range(2):
        partial = await tracker.update(
            partial_snapshot,
            [crossed_candidates[1]],
            [line],
        )
        target = next(
            marker for marker in partial if marker.track_id == target_old_id
        )
        assert target.position == 16.0

    crossed = await tracker.update(
        crossed_snapshot,
        crossed_candidates,
        [line],
    )

    assert [marker.position for marker in crossed] == pytest.approx([
        6.843633808333333,
        21.0,
        25.241493368421054,
    ])
    assert len(crossed) == 3
    assert target_old_id not in {marker.track_id for marker in crossed}
    replacement = next(marker for marker in crossed if marker.position == 21.0)
    assert replacement.boundary_revision == (144, 123)
    assert Counter(
        row
        for marker in crossed
        for row in marker.checkpoint_evidence
    ) == Counter(
        row
        for candidate in crossed_candidates
        for row in candidate.checkpoint_evidence
    )

    final = await tracker.update(final_snapshot, final_candidates, [line])

    continued = next(
        marker for marker in final if marker.track_id == replacement.track_id
    )
    assert continued.position == 18.0
    assert continued.bracket == (16.0, 19.0)
    assert len(final) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("segment_seconds", "expected"),
    [(360.0, 21.0), (361.0, None)],
)
async def test_first_boundary_long_segment_requires_two_historical_anchors(
    segment_seconds, expected,
):
    (
        initial_snapshot,
        initial_candidates,
        _partial_snapshot,
        crossed_snapshot,
        crossed_candidates,
        _final_snapshot,
        _final_candidates,
    ) = _citybus_fast_first_boundary_frames()
    tracker = MarkerTracker()
    line = _line(
        stops=30,
        route="792M",
        bound="inbound",
        operator="CTB",
    )
    await tracker.update(initial_snapshot, initial_candidates, [line])
    old = sorted(
        tracker._routes[("CTB", "792M", "inbound")].values(),
        key=lambda track: (track.position, track.track_id),
    )
    rows = list(crossed_snapshot.rows)
    candidates = list(crossed_candidates)
    target_slots = {
        rows[source[1]].index: source[1]
        for source in candidates[1].source_observations
        if source[0] == "probe"
    }
    arrival = rows[target_slots[23]].arrival_at + timedelta(
        seconds=segment_seconds,
    )
    minutes = (arrival - crossed_snapshot.collected_at).total_seconds() / 60.0
    rows[target_slots[25]] = replace(
        rows[target_slots[25]],
        arrival_at=arrival,
        minutes=minutes,
        signed_minutes=minutes,
    )
    candidates[1] = replace(
        candidates[1],
        checkpoint_evidence=tuple(
            (index, arrival.timestamp(), revision)
            if index == 25 else (index, stamp, revision)
            for index, stamp, revision in candidates[1].checkpoint_evidence
        ),
    )

    position = _certified_first_boundary_reseed_position(
        old,
        candidates,
        1,
        1,
        tuple(rows),
        crossed_snapshot.complete_routes[0],
        crossed_snapshot.collected_at.timestamp(),
    )

    assert position == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("empty_corruption", ["missing", "stale"])
async def test_citybus_reseed_requires_fresh_successful_empty_lower_endpoint(
    empty_corruption,
):
    (
        initial_snapshot,
        initial_candidates,
        _partial_snapshot,
        crossed_snapshot,
        crossed_candidates,
        _final_snapshot,
        _final_candidates,
    ) = _citybus_fast_first_boundary_frames()
    tracker = MarkerTracker()
    line = _line(
        stops=30,
        route="792M",
        bound="inbound",
        operator="CTB",
    )
    await tracker.update(initial_snapshot, initial_candidates, [line])
    old = sorted(
        tracker._routes[("CTB", "792M", "inbound")].values(),
        key=lambda track: (track.position, track.track_id),
    )
    rows = list(crossed_snapshot.rows)
    lower_slot = next(
        slot for slot, row in enumerate(rows)
        if row.index == 16 and row.minutes is None
    )
    if empty_corruption == "missing":
        rows.pop(lower_slot)
    else:
        rows[lower_slot] = replace(rows[lower_slot], cache_age_seconds=60.0)

    assert _certified_first_boundary_reseed_position(
        old,
        crossed_candidates,
        1,
        1,
        tuple(rows),
        crossed_snapshot.complete_routes[0],
        crossed_snapshot.collected_at.timestamp(),
    ) is None


@pytest.mark.asyncio
async def test_citybus_reseed_rejects_mixed_sibling_revision_at_one_stop():
    (
        initial_snapshot,
        initial_candidates,
        _partial_snapshot,
        crossed_snapshot,
        crossed_candidates,
        _final_snapshot,
        _final_candidates,
    ) = _citybus_fast_first_boundary_frames()
    tracker = MarkerTracker()
    line = _line(
        stops=30,
        route="792M",
        bound="inbound",
        operator="CTB",
    )
    await tracker.update(initial_snapshot, initial_candidates, [line])
    old = sorted(
        tracker._routes[("CTB", "792M", "inbound")].values(),
        key=lambda track: (track.position, track.track_id),
    )
    rows = list(crossed_snapshot.rows)
    candidates = list(crossed_candidates)
    upstream_slot = next(
        source[1]
        for source in candidates[0].source_observations
        if source[0] == "probe" and rows[source[1]].index == 22
    )
    rows[upstream_slot] = replace(
        rows[upstream_slot],
        refresh_generation=124,
    )
    candidates[0] = replace(
        candidates[0],
        checkpoint_evidence=tuple(
            (index, arrival, 124)
            if index == 22 else (index, arrival, revision)
            for index, arrival, revision in candidates[0].checkpoint_evidence
        ),
    )

    assert _first_boundary_reseed_context(
        old,
        candidates,
        tuple(rows),
        crossed_snapshot.complete_routes[0],
        crossed_snapshot.collected_at.timestamp(),
    ) is None


@pytest.mark.asyncio
async def test_citybus_long_segment_cannot_bridge_two_historical_owners():
    (
        initial_snapshot,
        initial_candidates,
        _partial_snapshot,
        crossed_snapshot,
        crossed_candidates,
        _final_snapshot,
        _final_candidates,
    ) = _citybus_fast_first_boundary_frames()
    tracker = MarkerTracker()
    line = _line(
        stops=30,
        route="792M",
        bound="inbound",
        operator="CTB",
    )
    await tracker.update(initial_snapshot, initial_candidates, [line])
    old = sorted(
        tracker._routes[("CTB", "792M", "inbound")].values(),
        key=lambda track: (track.position, track.track_id),
    )
    target_arrival = next(
        arrival
        for index, arrival, _revision
        in crossed_candidates[1].checkpoint_evidence
        if index == 25
    )
    old[1].cohort_evidence = tuple(
        row for row in old[1].cohort_evidence if row[0] != 25
    )
    old[2].cohort_evidence = tuple(
        (index, target_arrival, revision)
        if index == 25 else (index, arrival, revision)
        for index, arrival, revision in old[2].cohort_evidence
    )

    assert _first_boundary_reseed_context(
        old,
        crossed_candidates,
        crossed_snapshot.rows,
        crossed_snapshot.complete_routes[0],
        crossed_snapshot.collected_at.timestamp(),
    ) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "corruption",
    [
        "stale",
        "scheduled",
        "signed_mismatch",
        "boolean_revision",
        "revision_replay",
        "revision_rollback",
        "duplicate_occurrence",
        "duplicate_source_only",
        "duplicate_source_and_ledger",
        "foreign_temporal_owner",
        "frozen_revision_rollback",
        "unreliable",
        "prior_boundary",
        "forward_search",
        "backward",
        "coordinate_mismatch",
        "missing_complete_census",
    ],
)
async def test_first_boundary_reseed_proof_fails_closed(corruption):
    (
        initial_snapshot,
        initial_candidates,
        crossed_snapshot,
        crossed_candidates,
        _final_snapshot,
        _final_candidates,
    ) = _crossed_first_boundary_frames()
    tracker = MarkerTracker()
    await tracker.update(initial_snapshot, initial_candidates, [_line(stops=28)])
    old = sorted(
        tracker._routes[("KMB", "R", "out")].values(),
        key=lambda track: (track.position, track.track_id),
    )
    rows = list(crossed_snapshot.rows)
    candidates = list(crossed_candidates)
    complete_route = crossed_snapshot.complete_routes[0]
    target_probe_slots = sorted(
        source[1]
        for source in candidates[2].source_observations
        if source[0] == "probe"
    )
    target_upper_slot = target_probe_slots[0]

    if corruption == "stale":
        rows = [replace(row, cache_age_seconds=60.0) for row in rows]
    elif corruption == "scheduled":
        rows[target_upper_slot] = replace(
            rows[target_upper_slot], kind=EtaKind.SCHEDULED,
        )
    elif corruption == "signed_mismatch":
        rows[target_upper_slot] = replace(
            rows[target_upper_slot], signed_minutes=-1.0,
        )
    elif corruption == "boolean_revision":
        candidates[2] = replace(candidates[2], boundary_revision=(True, True))
    elif corruption in {"revision_replay", "revision_rollback"}:
        revision = 8 if corruption == "revision_replay" else 7
        rows = [replace(row, refresh_generation=revision) for row in rows]
        candidates = [replace(
            candidate,
            boundary_revision=(revision, revision),
            checkpoint_evidence=tuple(
                (index, arrival, revision)
                for index, arrival, _old_revision in candidate.checkpoint_evidence
            ),
        ) for candidate in candidates]
    elif corruption == "duplicate_occurrence":
        candidates[1] = replace(
            candidates[1],
            checkpoint_evidence=(
                *candidates[1].checkpoint_evidence,
                candidates[2].checkpoint_evidence[-1],
            ),
        )
    elif corruption in {"duplicate_source_only", "duplicate_source_and_ledger"}:
        candidates[1] = replace(
            candidates[1],
            source_observations=frozenset({
                *candidates[1].source_observations,
                ("probe", target_upper_slot),
            }),
            checkpoint_evidence=(
                candidates[1].checkpoint_evidence
                if corruption == "duplicate_source_only"
                else (
                    *candidates[1].checkpoint_evidence,
                    candidates[2].checkpoint_evidence[0],
                )
            ),
        )
    elif corruption == "foreign_temporal_owner":
        foreign = (
            22,
            candidates[2].checkpoint_evidence[1][1] + 3.0,
            8,
        )
        old[2].cohort_evidence = tuple(sorted((*old[2].cohort_evidence, foreign)))
        old[2].estimate = replace(
            old[2].estimate,
            checkpoint_evidence=old[2].cohort_evidence,
        )
    elif corruption == "frozen_revision_rollback":
        rolled_back = (
            21,
            candidates[2].checkpoint_evidence[0][1] + 3.0,
            38,
        )
        old[1].cohort_evidence = tuple(sorted((*old[1].cohort_evidence, rolled_back)))
        old[1].estimate = replace(
            old[1].estimate,
            checkpoint_evidence=old[1].cohort_evidence,
        )
    elif corruption == "unreliable":
        candidates[2] = replace(candidates[2], unreliable=True)
    elif corruption == "prior_boundary":
        old[1].boundary_revision = (8, 8)
    elif corruption == "forward_search":
        old[1].forward_after = 16
    elif corruption == "backward":
        candidates[2] = replace(
            candidates[2], position=14.0, bracket=(13.0, 17.0),
        )
    elif corruption == "coordinate_mismatch":
        candidates[2] = replace(candidates[2], position=19.0)
    elif corruption == "missing_complete_census":
        complete_route = None

    if corruption in {
        "stale", "scheduled", "signed_mismatch",
        "revision_replay", "revision_rollback",
    }:
        complete_route = _atomic_snapshot(
            2,
            rows,
            collected_at=crossed_snapshot.collected_at,
            observed_indices={0, 6, 13, 16, 17, 21, 22, 27},
        ).complete_routes[0]

    assert _certified_first_boundary_reseed_position(
        old,
        candidates,
        1,
        2,
        tuple(rows),
        complete_route,
        crossed_snapshot.collected_at.timestamp(),
    ) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_index", [[], None, -1, 1.5, True])
async def test_first_boundary_reseed_rejects_malformed_empty_checkpoint(bad_index):
    (
        initial_snapshot,
        initial_candidates,
        crossed_snapshot,
        crossed_candidates,
        _final_snapshot,
        _final_candidates,
    ) = _crossed_first_boundary_frames()
    tracker = MarkerTracker()
    initial = await tracker.update(
        initial_snapshot, initial_candidates, [_line(stops=28)],
    )
    malformed = replace(
        crossed_snapshot.rows[0],
        index=bad_index,
        minutes=None,
        signed_minutes=None,
        arrival_at=None,
    )
    snapshot = replace(
        crossed_snapshot,
        positioning_rows=(*crossed_snapshot.rows, malformed),
    )

    current = await tracker.update(snapshot, crossed_candidates, [_line(stops=28)])

    assert len(current) == len(initial) == 3
    target = next(
        marker for marker in current if marker.track_id == initial[1].track_id
    )
    assert target.position == 15.0
    assert all(marker.position != 20.0 for marker in current)


@pytest.mark.asyncio
async def test_first_boundary_reseed_uses_final_consecutive_due_stop():
    (
        initial_snapshot,
        initial_candidates,
        crossed_snapshot,
        crossed_candidates,
        _final_snapshot,
        _final_candidates,
    ) = _crossed_first_boundary_frames()
    tracker = MarkerTracker()
    await tracker.update(initial_snapshot, initial_candidates, [_line(stops=28)])
    old = sorted(
        tracker._routes[("KMB", "R", "out")].values(),
        key=lambda track: (track.position, track.track_id),
    )
    old[1].position = 12.0
    rows = list(crossed_snapshot.rows)
    due_slots = (len(rows), len(rows) + 1)
    rows.extend((
        _fresh_probe_row(
            16,
            crossed_snapshot.collected_at - timedelta(seconds=30),
            37,
            crossed_snapshot.collected_at,
            age=9.704,
        ),
        _fresh_probe_row(
            17,
            crossed_snapshot.collected_at,
            37,
            crossed_snapshot.collected_at,
            age=9.704,
        ),
    ))
    candidates = list(crossed_candidates)
    target_slots = tuple(sorted(
        source[1]
        for source in candidates[2].source_observations
        if source[0] == "probe"
    ))
    candidates[2] = _candidate_from_rows(
        17.0,
        (17.0, 21.0),
        (37, 37),
        rows,
        (*due_slots, *target_slots),
        arrival_slot=target_slots[0],
    )
    complete_route = _atomic_snapshot(
        2,
        rows,
        collected_at=crossed_snapshot.collected_at,
        observed_indices={0, 6, 13, 16, 17, 21, 22, 27},
    ).complete_routes[0]

    assert _certified_first_boundary_reseed_position(
        old,
        candidates,
        1,
        2,
        tuple(rows),
        complete_route,
        crossed_snapshot.collected_at.timestamp(),
    ) == pytest.approx(17.0)
    candidates[2] = replace(candidates[2], bracket=(16.0, 21.0))
    assert _certified_first_boundary_reseed_position(
        old,
        candidates,
        1,
        2,
        tuple(rows),
        complete_route,
        crossed_snapshot.collected_at.timestamp(),
    ) is None


def test_first_boundary_reseed_context_is_bounded_at_route_capacity():
    collected_at = BASE_TIME + timedelta(seconds=30)
    old = []
    rows = []
    candidate_slots = []
    for owner in range(128):
        evidence = tuple(
            (
                stop,
                (BASE_TIME + timedelta(
                    minutes=10 + owner * 5 + stop,
                )).timestamp(),
                10,
            )
            for stop in range(64)
        )
        estimate = replace(
            _candidate(float(owner)), checkpoint_evidence=evidence,
        )
        old.append(_Track(
            owner + 1,
            estimate,
            float(owner),
            1,
            cohort_evidence=evidence,
            cohort_observed_at=BASE_TIME.timestamp(),
            cohort_trusted=True,
        ))
        slots = []
        for stop, arrival, _revision in evidence:
            slots.append(len(rows))
            rows.append(_fresh_probe_row(
                stop,
                datetime.fromtimestamp(arrival, tz=UTC),
                11,
                collected_at,
            ))
        candidate_slots.append(tuple(slots))
    candidates = [
        _candidate_from_rows(
            float(owner),
            (float(owner), float(owner + 1)),
            (11, 11),
            rows,
            candidate_slots[owner],
        )
        for owner in range(128)
    ]
    complete = _atomic_snapshot(
        2, rows, collected_at=collected_at,
    ).complete_routes[0]

    started = perf_counter()
    context = _first_boundary_reseed_context(
        old, candidates, tuple(rows), complete, collected_at.timestamp(),
    )
    elapsed = perf_counter() - started

    assert context is not None
    assert len(context["reserved"]) == 128 * 64
    assert elapsed < 1.0


def test_first_boundary_reseed_context_declines_dense_ambiguity():
    collected_at = BASE_TIME + timedelta(seconds=30)
    old = []
    rows = []
    candidates = []
    for owner in range(33):
        arrival = BASE_TIME + timedelta(minutes=10, milliseconds=owner * 10)
        evidence = ((8, arrival.timestamp(), 10),)
        estimate = replace(
            _candidate(float(owner)), checkpoint_evidence=evidence,
        )
        old.append(_Track(
            owner + 1,
            estimate,
            float(owner),
            1,
            cohort_evidence=evidence,
            cohort_observed_at=BASE_TIME.timestamp(),
            cohort_trusted=True,
        ))
        slot = len(rows)
        rows.append(_fresh_probe_row(8, arrival, 11, collected_at))
        candidates.append(_candidate_from_rows(
            float(owner),
            (8.0, 9.0),
            (11, 11),
            rows,
            (slot,),
        ))
    complete = _atomic_snapshot(
        2, rows, collected_at=collected_at,
    ).complete_routes[0]

    assert _first_boundary_reseed_context(
        old, candidates, tuple(rows), complete, collected_at.timestamp(),
    ) is None


@pytest.mark.asyncio
async def test_same_generation_cannot_reseed_a_crossed_first_boundary_track():
    (
        initial_snapshot,
        initial_candidates,
        crossed_snapshot,
        crossed_candidates,
        _final_snapshot,
        _final_candidates,
    ) = _crossed_first_boundary_frames()
    tracker = MarkerTracker()
    initial = await tracker.update(
        initial_snapshot, initial_candidates, [_line(stops=28)]
    )
    same_generation = replace(
        crossed_snapshot.complete_routes[0], generation=1,
    )

    current = await tracker.update(
        ProbeEtaSnapshot((same_generation,), crossed_snapshot.collected_at),
        crossed_candidates,
        [_line(stops=28)],
    )

    assert {marker.track_id for marker in current} == {
        marker.track_id for marker in initial
    }
    assert next(
        marker for marker in current if marker.track_id == initial[1].track_id
    ).position == pytest.approx(15.0)
    assert all(marker.position != pytest.approx(20.0) for marker in current)


@pytest.mark.asyncio
async def test_boundary_reseed_retiree_does_not_block_survivor_ordered_motion():
    first_at = BASE_TIME
    second_at = BASE_TIME + timedelta(minutes=1)
    old_rows = [
        _fresh_probe_row(27, BASE_TIME + timedelta(minutes=30), 8, first_at),
        _fresh_probe_row(27, BASE_TIME + timedelta(minutes=20), 8, first_at),
        _fresh_probe_row(27, BASE_TIME + timedelta(minutes=25), 8, first_at),
    ]
    old_candidates = [
        _candidate_from_rows(12.0, (0.0, 13.0), (8, 8), old_rows, (0,)),
        _candidate_from_rows(15.0, (13.0, 16.0), None, old_rows, (1,)),
        _candidate_from_rows(26.0, (16.0, 27.0), (8, 8), old_rows, (2,)),
    ]
    current_rows = [
        _fresh_probe_row(17, second_at + timedelta(minutes=3), 37, second_at),
        _fresh_probe_row(
            27, BASE_TIME + timedelta(minutes=25, seconds=3), 37, second_at,
        ),
        _fresh_probe_row(19, second_at + timedelta(minutes=5), 37, second_at),
        _fresh_probe_row(27, BASE_TIME + timedelta(minutes=30), 37, second_at),
        _fresh_probe_row(21, second_at + timedelta(minutes=4), 37, second_at),
        _fresh_probe_row(27, BASE_TIME + timedelta(minutes=20), 37, second_at),
    ]
    current_candidates = [
        _candidate_from_rows(
            15.059138183574879,
            (15.0, 17.0),
            (37, 37),
            current_rows,
            (0, 1),
        ),
        _candidate_from_rows(
            18.0, (17.0, 19.0), (37, 37), current_rows, (2, 3),
        ),
        _candidate_from_rows(
            20.0, (17.0, 21.0), (37, 37), current_rows, (4, 5),
        ),
    ]
    tracker = MarkerTracker(max_tracks_per_route=3)
    line = _line(stops=28)
    initial = await tracker.update(
        _atomic_snapshot(1, old_rows, collected_at=first_at),
        old_candidates,
        [line],
    )

    current = await tracker.update(
        _atomic_snapshot(
            2,
            current_rows,
            collected_at=second_at,
            observed_indices={0, 13, 16, 17, 19, 21, 27},
        ),
        current_candidates,
        [line],
    )

    assert [marker.position for marker in current] == pytest.approx(
        [15.059138183574879, 18.0, 20.0]
    )
    survivor = next(marker for marker in current if marker.track_id == initial[0].track_id)
    assert survivor.position == pytest.approx(18.0)
    assert survivor.checkpoint_evidence == current_candidates[1].checkpoint_evidence
    assert initial[1].track_id not in {marker.track_id for marker in current}
    assert initial[2].track_id not in {marker.track_id for marker in current}
    assert len({marker.track_id for marker in current}) == 3


@pytest.mark.asyncio
async def test_order_rejected_motion_is_not_converted_to_boundary_reseed(monkeypatch):
    first_at = BASE_TIME
    second_at = BASE_TIME + timedelta(seconds=30)
    arrival = BASE_TIME + timedelta(minutes=10)
    old_row = _fresh_probe_row(8, arrival, 10, first_at)
    candidate_row = _fresh_probe_row(8, arrival, 11, second_at)
    old_candidate = _candidate_from_rows(
        5.0, (4.0, 5.0), None, [old_row], (0,),
    )
    candidate = _candidate_from_rows(
        7.0, (6.0, 7.0), (11, 11), [candidate_row], (0,),
    )
    tracker = MarkerTracker()
    initial = await tracker.update(
        _atomic_snapshot(1, [old_row], collected_at=first_at),
        [old_candidate],
        [_line(stops=12)],
    )
    proof_calls = []

    def unexpected_reseed(*args, **kwargs):
        proof_calls.append((args, kwargs))
        return 7.0

    monkeypatch.setattr(
        tracker_module, "_certified_first_boundary_reseed_position", unexpected_reseed,
    )
    monkeypatch.setattr(tracker_module, "_select_ordered_updates", lambda *args, **kwargs: set())

    current = await tracker.update(
        _atomic_snapshot(2, [candidate_row], collected_at=second_at),
        [candidate],
        [_line(stops=12)],
    )

    assert proof_calls == []
    assert len(current) == 1
    assert current[0].track_id == initial[0].track_id
    assert current[0].position == pytest.approx(5.0)


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
async def test_same_generation_replay_cannot_overwrite_neighbour_with_same_eta():
    tracker = MarkerTracker()
    arrival = BASE_TIME + timedelta(minutes=3)
    owned = replace(
        _candidate(
            7.0,
            bracket=(0.0, 8.0),
            boundary_age=0,
            boundary_revision=(12, 20),
            arrival_at=arrival,
        ),
        source_observations=frozenset({("gate", 11), ("probe", 89), ("probe", 96)}),
        checkpoint_evidence=(
            (8, arrival.timestamp(), 20),
            (16, (arrival + timedelta(minutes=3)).timestamp(), 21),
        ),
    )
    neighbour = replace(
        _candidate(9.0, bracket=(8.0, 9.0), boundary_age=0),
        source_observations=frozenset({("gate", 10), ("probe", 97)}),
        checkpoint_evidence=(
            (16, (arrival + timedelta(minutes=1)).timestamp(), 22),
        ),
    )
    initial = await tracker.update(_snapshot(1), [owned, neighbour])

    # A partial render of the same complete generation retains only the first
    # candidate. Its revisions are unchanged for its actual owner but appear
    # new to the neighbouring track, which must not consume the same ETA.
    current = await tracker.update(
        _snapshot(1, collected_at=BASE_TIME + timedelta(seconds=30)),
        [replace(owned, boundary_age_seconds=30)],
    )

    assert [marker.track_id for marker in current] == [
        marker.track_id for marker in initial
    ]
    assert [marker.position for marker in current] == [7.0, 9.0]
    assert current[0].source_observations == owned.source_observations
    assert current[1].source_observations == neighbour.source_observations
    assert sum(
        (8, arrival.timestamp(), 20) in marker.checkpoint_evidence
        for marker in current
    ) == 1


@pytest.mark.asyncio
async def test_merged_checkpoint_replay_cannot_overwrite_unrelated_track():
    tracker = MarkerTracker()
    arrival = BASE_TIME + timedelta(minutes=3)

    def observed(position, revision, eta_offset, evidence):
        return replace(
            _candidate(
                position,
                bracket=(float(int(position)), float(int(position) + 1)),
                boundary_age=0,
                boundary_revision=revision,
                arrival_at=arrival + timedelta(seconds=eta_offset),
            ),
            checkpoint_evidence=tuple(evidence),
        )

    leading = observed(3.058, (98, 100), 84, (
        (4, (arrival + timedelta(seconds=84)).timestamp(), 100),
        (8, (arrival + timedelta(seconds=220)).timestamp(), 103),
    ))
    replay_owner = observed(4.408, (100, 101), 0, (
        (4, (arrival - timedelta(seconds=50)).timestamp(), 100),
        (5, arrival.timestamp(), 101),
        (8, (arrival + timedelta(seconds=178)).timestamp(), 103),
    ))
    downstream = observed(12.397, (102, 99), 42, (
        (13, (arrival + timedelta(seconds=42)).timestamp(), 99),
        (16, (arrival + timedelta(seconds=232)).timestamp(), 93),
    ))
    trailing = observed(15.656, (99, 93), 10, (
        (16, (arrival + timedelta(seconds=10)).timestamp(), 93),
    ))
    initial = await tracker.update(
        _snapshot(1), [leading, replay_owner, downstream, trailing]
    )

    # A staggered refresh can merge downstream corroboration from another bus
    # into an otherwise unchanged replay. Both prior tracks then own some exact
    # checkpoint evidence; neither fact makes it evidence for the leading bus.
    merged_replay = replace(
        replay_owner,
        checkpoint_evidence=(
            *replay_owner.checkpoint_evidence,
            *downstream.checkpoint_evidence,
        ),
    )
    current = await tracker.update(
        _snapshot(1, collected_at=BASE_TIME + timedelta(seconds=30)),
        [leading, merged_replay, trailing],
    )

    assert [marker.track_id for marker in current] == [
        marker.track_id for marker in initial
    ]
    assert [marker.position for marker in current] == pytest.approx(
        [3.058, 4.408, 12.397, 15.656]
    )
    assert sum(
        marker.eta_arrival_at == replay_owner.eta_arrival_at
        for marker in current
    ) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("replay_bracket", "replay_position", "replay_revision"),
    (
        ((2.0, 5.0), 4.4, (102, 101)),
        # Revisions are local to a physical stop. A newly narrowed lower stop
        # may have an older revision number than the lower stop it replaces.
        ((4.0, 5.0), 4.7, (99, 101)),
    ),
)
async def test_unchanged_boundary_replay_cannot_overwrite_checkpoint_coowner(
    replay_bracket, replay_position, replay_revision
):
    tracker = MarkerTracker()
    arrival = BASE_TIME + timedelta(minutes=3)
    replay_owner = replace(
        _candidate(
            4.4,
            bracket=(2.0, 5.0),
            boundary_age=0,
            boundary_revision=(100, 101),
            arrival_at=arrival,
        ),
        checkpoint_evidence=((5, arrival.timestamp(), 101),),
    )
    nearby = replace(
        _candidate(
            6.3,
            bracket=(6.0, 7.0),
            boundary_age=0,
            boundary_revision=(98, 99),
            arrival_at=arrival + timedelta(minutes=1),
        ),
        checkpoint_evidence=(
            (7, (arrival + timedelta(minutes=1)).timestamp(), 99),
        ),
    )
    initial = await tracker.update(_snapshot(1), [replay_owner, nearby])
    merged_replay = replace(
        replay_owner,
        # The lower boundary and derived position may refine while the upper
        # ETA observation remains unchanged.
        position=replay_position,
        bracket=replay_bracket,
        boundary_revision=replay_revision,
        checkpoint_evidence=(
            *replay_owner.checkpoint_evidence,
            *nearby.checkpoint_evidence,
        ),
    )

    current = await tracker.update(
        _snapshot(1, collected_at=BASE_TIME + timedelta(seconds=30)),
        [merged_replay],
    )

    assert [marker.track_id for marker in current] == [
        marker.track_id for marker in initial
    ]
    assert [marker.position for marker in current] == [4.4, 6.3]
    assert [marker.eta_arrival_at for marker in current] == [
        replay_owner.eta_arrival_at,
        nearby.eta_arrival_at,
    ]


@pytest.mark.asyncio
async def test_upper_only_boundary_refresh_cannot_overwrite_checkpoint_coowner():
    tracker = MarkerTracker()
    arrival = BASE_TIME + timedelta(minutes=3)
    replay_owner = replace(
        _candidate(
            4.4,
            bracket=(2.0, 5.0),
            boundary_age=0,
            boundary_revision=(100, 101),
            arrival_at=arrival,
        ),
        checkpoint_evidence=(
            (5, arrival.timestamp(), 101),
            (8, (arrival + timedelta(seconds=100)).timestamp(), 103),
        ),
    )
    nearby = replace(
        _candidate(
            6.3,
            bracket=(6.0, 7.0),
            boundary_age=0,
            boundary_revision=(98, 99),
            arrival_at=arrival + timedelta(minutes=1),
        ),
        checkpoint_evidence=(
            (7, (arrival + timedelta(minutes=1)).timestamp(), 99),
        ),
    )
    initial = await tracker.update(_snapshot(1), [replay_owner, nearby])
    merged_replay = replace(
        replay_owner,
        # The provider refreshed only the upper endpoint while retaining the
        # same absolute ETA and an exact downstream checkpoint owned by this
        # track. The stale lower endpoint still makes this non-actionable.
        boundary_revision=(100, 104),
        checkpoint_evidence=(
            (5, arrival.timestamp(), 104),
            replay_owner.checkpoint_evidence[1],
            *nearby.checkpoint_evidence,
        ),
    )

    current = await tracker.update(
        _snapshot(1, collected_at=BASE_TIME + timedelta(seconds=30)),
        [merged_replay],
    )

    assert [marker.track_id for marker in current] == [
        marker.track_id for marker in initial
    ]
    assert [marker.position for marker in current] == [4.4, 6.3]
    assert [marker.eta_arrival_at for marker in current] == [
        replay_owner.eta_arrival_at,
        nearby.eta_arrival_at,
    ]


@pytest.mark.asyncio
async def test_shared_checkpoint_evidence_still_allows_a_fresh_owner_update():
    tracker = MarkerTracker()
    shared_arrival = (BASE_TIME + timedelta(minutes=5)).timestamp()
    blocking = replace(
        _candidate(
            5.0,
            bracket=(4.0, 5.0),
            boundary_age=0,
            boundary_revision=(20, 20),
        ),
        checkpoint_evidence=((10, shared_arrival, 30),),
    )
    moving = replace(
        _candidate(
            7.0,
            bracket=(6.0, 7.0),
            boundary_age=0,
            boundary_revision=(10, 10),
        ),
        checkpoint_evidence=(
            (10, shared_arrival, 30),
            (12, shared_arrival + 60, 31),
        ),
    )
    initial = await tracker.update(_snapshot(1), [blocking, moving])
    refreshed = replace(
        moving,
        position=8.0,
        bracket=(7.0, 8.0),
        boundary_revision=(11, 11),
    )

    current = await tracker.update(
        _snapshot(1, collected_at=BASE_TIME + timedelta(seconds=30)),
        [refreshed],
    )

    # A shared checkpoint is still safe when the moving owner's complete
    # evidence covers every prior-owned row in the candidate.
    assert [marker.track_id for marker in current] == [
        marker.track_id for marker in initial
    ]
    assert [marker.position for marker in current] == [5.0, 8.0]


@pytest.mark.asyncio
async def test_complete_generation_retires_old_order_barrier_before_selecting_updates():
    tracker = MarkerTracker()
    old_arrival = BASE_TIME + timedelta(minutes=5)

    def with_checkpoint(candidate, index, arrival, revision):
        return replace(
            candidate,
            checkpoint_evidence=((index, arrival.timestamp(), revision),),
        )

    old = [
        with_checkpoint(_candidate(0.55, bracket=(0.0, 1.0), boundary_age=0,
                                  boundary_revision=(10, 10)),
                        2, old_arrival, 10),
        _candidate(8.25, bracket=(8.0, 9.0), boundary_age=0,
                   boundary_revision=(10, 10)),
        with_checkpoint(_candidate(20.0, bracket=(20.0, 21.0), boundary_age=0,
                                  boundary_revision=(10, 10)),
                        20, old_arrival + timedelta(minutes=2), 10),
        _candidate(28.0, bracket=(28.0, 29.0), boundary_age=0,
                   boundary_revision=(10, 10)),
    ]
    initial = await tracker.update(_snapshot(1), old)

    recovered = with_checkpoint(
        _candidate(10.0, bracket=(9.0, 10.0), boundary_age=0,
                   boundary_revision=(11, 11)),
        2, old_arrival, 10,
    )
    downstream = with_checkpoint(
        _candidate(16.7, bracket=(16.0, 17.0), boundary_age=0,
                   boundary_revision=(11, 11)),
        20, old_arrival + timedelta(minutes=2), 10,
    )
    current = await tracker.update(
        _snapshot(2),
        [_candidate(8.79, bracket=(8.0, 9.0), boundary_age=0,
                    boundary_revision=(11, 11)), recovered, downstream],
    )

    assert [marker.position for marker in current] == pytest.approx(
        [8.79, 10.0, 16.7]
    )
    assert next(marker for marker in current if marker.position == 10.0).track_id == initial[0].track_id


@pytest.mark.asyncio
async def test_complete_reliable_motion_ignores_tentative_order_barrier():
    tracker = MarkerTracker()
    reliable_arrival = BASE_TIME + timedelta(minutes=10)
    unreliable_arrival = BASE_TIME + timedelta(minutes=15)
    reliable_evidence = ((15, reliable_arrival.timestamp(), 10),)
    unreliable_evidence = ((18, unreliable_arrival.timestamp(), 10),)
    initial_candidates = [
        replace(
            _candidate(
                12.737,
                bracket=(12.0, 13.0),
                boundary_age=0,
                boundary_revision=(10, 10),
                arrival_at=reliable_arrival,
            ),
            checkpoint_evidence=reliable_evidence,
        ),
        replace(
            _candidate(
                13.0,
                unreliable=True,
                bracket=(13.0, 14.0),
                arrival_at=unreliable_arrival,
            ),
            checkpoint_evidence=unreliable_evidence,
        ),
    ]
    initial = await tracker.update(_snapshot(1), initial_candidates)

    reliable_candidate = replace(
        initial_candidates[0],
        position=14.219,
        bracket=(14.0, 15.0),
        boundary_revision=(11, 11),
    )
    current_unreliable_evidence = ((
        18,
        (unreliable_arrival + timedelta(seconds=30)).timestamp(),
        11,
    ),)
    unreliable_candidate = replace(
        initial_candidates[1],
        position=15.0,
        bracket=(15.0, 16.0),
        checkpoint_evidence=current_unreliable_evidence,
    )
    current = await tracker.update(
        _snapshot(2, collected_at=BASE_TIME + timedelta(seconds=30)),
        [unreliable_candidate, reliable_candidate],
    )

    assert len(current) == len(initial) == 2
    reliable = next(
        marker for marker in current if marker.track_id == initial[0].track_id
    )
    assert reliable.position == pytest.approx(14.219)
    assert reliable.bracket == (14.0, 15.0)
    assert reliable.checkpoint_evidence == reliable_evidence
    tentative = next(
        marker for marker in current if marker.track_id == initial[1].track_id
    )
    assert tentative.unreliable is True
    assert tentative.position == pytest.approx(13.0)
    assert tentative.bracket == (13.0, 14.0)
    assert tentative.checkpoint_evidence == current_unreliable_evidence
    route = tracker._routes[("KMB", "R", "out")]  # noqa: SLF001
    assert route[tentative.track_id].cohort_evidence == current_unreliable_evidence

    continued = await tracker.update(
        _snapshot(3, collected_at=BASE_TIME + timedelta(seconds=60)),
        [
            replace(
                reliable_candidate,
                position=14.6,
                bracket=(14.0, 15.0),
                boundary_revision=(12, 12),
            ),
            replace(unreliable_candidate, position=15.2),
        ],
    )

    continued_by_id = {marker.track_id: marker for marker in continued}
    assert set(continued_by_id) == {marker.track_id for marker in initial}
    assert continued_by_id[reliable.track_id].position == pytest.approx(14.6)
    assert continued_by_id[tentative.track_id].position == pytest.approx(13.0)


@pytest.mark.asyncio
async def test_retained_boundary_history_remains_an_order_barrier_when_unreliable():
    tracker = MarkerTracker()
    initial_candidates = [
        _candidate(
            12.737,
            bracket=(12.0, 13.0),
            boundary_age=0,
            boundary_revision=(10, 10),
        ),
        _candidate(
            13.0,
            unreliable=True,
            bracket=(13.0, 14.0),
            boundary_age=0,
            boundary_revision=(10, 10),
        ),
    ]
    initial = await tracker.update(_snapshot(1), initial_candidates)
    moving = replace(
        initial_candidates[0],
        position=14.219,
        bracket=(14.0, 15.0),
        boundary_revision=(11, 11),
    )
    held_barrier = replace(
        initial_candidates[1],
        position=15.0,
        bracket=(15.0, 16.0),
        boundary_age_seconds=None,
        boundary_revision=None,
    )

    current = await tracker.update(
        _snapshot(2, collected_at=BASE_TIME + timedelta(seconds=30)),
        [moving, held_barrier],
    )

    assert [marker.track_id for marker in current] == [
        marker.track_id for marker in initial
    ]
    assert [marker.position for marker in current] == pytest.approx([12.737, 13.0])
    assert [marker.bracket for marker in current] == [
        (12.0, 13.0),
        (13.0, 14.0),
    ]
    assert current[0].boundary_revision == (10, 10)
    assert current[1].boundary_revision == (10, 10)


@pytest.mark.asyncio
async def test_reliable_position_history_survives_unreliable_complete_holds():
    tracker = MarkerTracker()
    moving = _candidate(
        10.0,
        bracket=(9.0, 10.0),
        boundary_age=0,
        boundary_revision=(10, 10),
    )
    prior_reliable_barrier = _candidate(13.0, bracket=(13.0, 14.0))
    initial = await tracker.update(
        _snapshot(1), [moving, prior_reliable_barrier]
    )
    unreliable_barrier = replace(
        prior_reliable_barrier,
        position=15.0,
        bracket=(15.0, 16.0),
        unreliable=True,
    )

    held = await tracker.update(
        _snapshot(2, collected_at=BASE_TIME + timedelta(seconds=30)),
        [moving, unreliable_barrier],
    )
    barrier_id = initial[1].track_id
    barrier_track = tracker._routes[("KMB", "R", "out")][barrier_id]  # noqa: SLF001
    assert next(marker for marker in held if marker.track_id == barrier_id).position == 13.0
    assert barrier_track.estimate.unreliable is True
    assert barrier_track.position_authoritative is True

    advanced_candidate = replace(
        moving,
        position=13.219,
        bracket=(13.0, 14.0),
        boundary_revision=(11, 11),
    )
    current = await tracker.update(
        _snapshot(3, collected_at=BASE_TIME + timedelta(seconds=60)),
        [advanced_candidate, unreliable_barrier],
    )

    assert [marker.track_id for marker in current] == [
        marker.track_id for marker in initial
    ]
    assert [marker.position for marker in current] == pytest.approx([10.0, 13.0])


@pytest.mark.asyncio
async def test_complete_reliable_motion_can_pass_multiple_tentative_barriers():
    tracker = MarkerTracker()
    arrivals = [BASE_TIME + timedelta(minutes=value) for value in (10, 15, 20)]
    initial_candidates = [
        _candidate(
            10.0,
            bracket=(9.0, 10.0),
            boundary_age=0,
            boundary_revision=(10, 10),
            arrival_at=arrivals[0],
        ),
        _candidate(
            11.0,
            unreliable=True,
            bracket=(11.0, 12.0),
            arrival_at=arrivals[1],
        ),
        _candidate(
            12.0,
            unreliable=True,
            bracket=(12.0, 13.0),
            arrival_at=arrivals[2],
        ),
    ]
    initial = await tracker.update(_snapshot(1), initial_candidates)
    candidates = [
        replace(
            initial_candidates[0],
            position=13.2,
            bracket=(13.0, 14.0),
            boundary_revision=(11, 11),
        ),
        replace(initial_candidates[1], position=14.0, bracket=(14.0, 15.0)),
        replace(initial_candidates[2], position=15.0, bracket=(15.0, 16.0)),
    ]

    current = await tracker.update(
        _snapshot(2, collected_at=BASE_TIME + timedelta(seconds=30)),
        list(reversed(candidates)),
    )

    by_id = {marker.track_id: marker for marker in current}
    assert set(by_id) == {marker.track_id for marker in initial}
    assert by_id[initial[0].track_id].position == pytest.approx(13.2)
    assert by_id[initial[1].track_id].position == pytest.approx(11.0)
    assert by_id[initial[2].track_id].position == pytest.approx(12.0)


@pytest.mark.asyncio
async def test_same_generation_tentative_barrier_keeps_strict_order():
    tracker = MarkerTracker()
    reliable_evidence = ((
        15,
        (BASE_TIME + timedelta(minutes=10)).timestamp(),
        10,
    ),)
    tentative_evidence = ((
        18,
        (BASE_TIME + timedelta(minutes=15)).timestamp(),
        10,
    ),)
    initial_candidates = [
        replace(
            _candidate(
                12.737,
                bracket=(12.0, 13.0),
                boundary_age=0,
                boundary_revision=(10, 10),
            ),
            checkpoint_evidence=reliable_evidence,
        ),
        replace(
            _candidate(
                13.0,
                unreliable=True,
                bracket=(13.0, 14.0),
            ),
            checkpoint_evidence=tentative_evidence,
        ),
    ]
    initial = await tracker.update(_snapshot(1), initial_candidates)

    current = await tracker.update(
        _snapshot(1, collected_at=BASE_TIME + timedelta(seconds=30)),
        [
            replace(
                initial_candidates[0],
                position=14.219,
                bracket=(14.0, 15.0),
                boundary_revision=(11, 11),
            ),
            replace(
                initial_candidates[1],
                position=15.0,
                bracket=(15.0, 16.0),
            ),
        ],
    )

    assert [marker.track_id for marker in current] == [
        marker.track_id for marker in initial
    ]
    assert [marker.position for marker in current] == pytest.approx([12.737, 13.0])
    assert [marker.bracket for marker in current] == [
        (12.0, 13.0),
        (13.0, 14.0),
    ]


@pytest.mark.asyncio
async def test_same_generation_gmb11_staggered_candidate_holds_multiple_checkpoint_owners():
    tracker = MarkerTracker()
    t39 = BASE_TIME + timedelta(minutes=4)
    t30 = BASE_TIME + timedelta(minutes=5)
    first = replace(
        _candidate(7.0, route="11B", operator=Operator.GMB, bracket=(7.0, 8.0),
                   boundary_age=0, boundary_revision=(39, 39)),
        checkpoint_evidence=((9, t39.timestamp(), 39),),
    )
    second = replace(
        _candidate(8.023, route="11B", operator=Operator.GMB, bracket=(8.0, 9.0),
                   boundary_age=0, boundary_revision=(30, 30)),
        checkpoint_evidence=((8, t30.timestamp(), 30),),
    )
    third = _candidate(8.033, route="11B", operator=Operator.GMB,
                       bracket=(8.0, 9.0), boundary_age=0,
                       boundary_revision=(31, 31))
    initial = await tracker.update(_snapshot(1, route_key=("GMB", "11B", "out")),
                                   [first, second, third])
    staggered = replace(
        _candidate(8.006, route="11B", operator=Operator.GMB,
                   bracket=(8.0, 9.0), boundary_age=0,
                   boundary_revision=(40, 40)),
        checkpoint_evidence=(first.checkpoint_evidence[0],
                             second.checkpoint_evidence[0]),
    )
    current = await tracker.update(
        _snapshot(1, collected_at=BASE_TIME + timedelta(seconds=30),
                  route_key=("GMB", "11B", "out")),
        [staggered],
    )

    assert [marker.position for marker in current] == pytest.approx(
        [7.0, 8.023, 8.033]
    )
    assert [marker.track_id for marker in current] == [marker.track_id for marker in initial]
    signatures = [row for marker in current for row in marker.checkpoint_evidence]
    assert len(signatures) == len(set(signatures))


def test_unique_covering_owner_cannot_be_overridden_by_positioning_owner():
    arrival = BASE_TIME + timedelta(minutes=4)
    shared = (8, arrival.timestamp(), 10)
    exclusive = (9, (arrival + timedelta(minutes=1)).timestamp(), 10)
    positioning = replace(
        _candidate(7.0, bracket=(7.0, 8.0), boundary_age=0,
                   boundary_revision=(10, 10), arrival_at=arrival),
        checkpoint_evidence=(shared,),
    )
    covering = replace(
        _candidate(8.0, bracket=(8.0, 9.0), boundary_age=0,
                   boundary_revision=(10, 10), arrival_at=arrival),
        checkpoint_evidence=(shared, exclusive),
    )
    candidate = replace(
        positioning,
        boundary_revision=(11, 11),
        checkpoint_evidence=(shared, exclusive, (8, arrival.timestamp(), 11)),
    )
    tracks = [
        _Track(1, positioning, 7.0, 1, boundary_revision=(10, 10)),
        _Track(2, covering, 8.0, 1, boundary_revision=(10, 10)),
    ]

    # The first track is a positioning match, but the second uniquely covers
    # every prior-owned candidate row. The non-positioning covering owner is
    # still eligible; the positioning-only owner is rejected.
    assert not _same_generation_actionable(tracks, tracks[0], candidate)
    assert _same_generation_actionable(tracks, tracks[1], candidate)


def test_multiple_covering_owners_reject_closer_subset_owner():
    arrival = BASE_TIME + timedelta(minutes=4)
    shared = (8, arrival.timestamp(), 10)
    exclusive = (9, (arrival + timedelta(minutes=1)).timestamp(), 10)
    full = replace(
        _candidate(8.0, bracket=(8.0, 9.0), boundary_age=0,
                   boundary_revision=(10, 10), arrival_at=arrival),
        checkpoint_evidence=(shared, exclusive),
    )
    subset = replace(
        _candidate(7.0, bracket=(7.0, 8.0), boundary_age=0,
                   boundary_revision=(10, 10), arrival_at=arrival),
        checkpoint_evidence=(shared,),
    )
    candidate = replace(
        subset,
        position=7.1,
        boundary_revision=(11, 11),
        checkpoint_evidence=(shared, exclusive, (8, arrival.timestamp(), 11)),
    )
    tracks = [
        _Track(1, full, 8.0, 1, boundary_revision=(10, 10)),
        _Track(2, replace(full), 8.1, 1, boundary_revision=(10, 10)),
        _Track(3, subset, 7.0, 1, boundary_revision=(10, 10)),
    ]

    # Two tracks cover the complete immutable signature set. The closer third
    # track owns only the shared row and cannot absorb the exclusive row.
    assert not _same_generation_actionable(tracks, tracks[2], candidate)


@pytest.mark.asyncio
async def test_complete_held_gmb12_candidate_refreshes_identity_and_priorities():
    tracker = MarkerTracker()
    old = replace(
        _candidate(4.0, route="12", operator=Operator.GMB,
                   bracket=(3.0, 4.0), boundary_age=0,
                   boundary_revision=(20, 20)),
        source_observations=frozenset({("probe", 40)}),
        checkpoint_evidence=((5, BASE_TIME.timestamp(), 20),),
    )
    await tracker.update(_snapshot(1, route_key=("GMB", "12", "out")), [old])
    wide = replace(
        _candidate(10.0, route="12", operator=Operator.GMB,
                   bracket=(7.0, 12.0), boundary_age=0,
                   boundary_revision=(21, 20)),
        source_observations=frozenset({("probe", 70), ("probe", 120)}),
        checkpoint_evidence=((5, BASE_TIME.timestamp(), 21),),
    )
    current = await tracker.update(
        _snapshot(2, route_key=("GMB", "12", "out")), [wide]
    )

    assert current[0].position == pytest.approx(4.0)
    assert current[0].source_observations == wide.source_observations
    assert current[0].checkpoint_evidence == wide.checkpoint_evidence
    assert current[0].bracket == (3.0, 4.0)
    assert tracker.poll_priorities()[("GMB", "12", "out")] == frozenset(
        {4, 7, 9, 12}
    )


def _complete_cold_fixture():
    first = replace(
        _candidate(4.5, bracket=(4.0, 5.0), boundary_age=0,
                   boundary_revision=(20, 20)),
        position_authoritative=True,
        source_indices=frozenset({4, 5}),
        checkpoint_evidence=((23, BASE_TIME.timestamp() + 300, 20),
                             (24, BASE_TIME.timestamp() + 360, 20)),
    )
    second = replace(
        _candidate(8.0, bracket=(7.0, 9.0), boundary_age=0,
                   boundary_revision=(20, 20)),
        position_authoritative=True,
        source_indices=frozenset({25, 26}),
        checkpoint_evidence=((25, BASE_TIME.timestamp() + 420, 20),
                             (26, BASE_TIME.timestamp() + 480, 20)),
    )
    cold = replace(
        first, position=11.2, lat=22.4, lon=114.3,
        bracket=None, boundary_revision=None, position_authoritative=False,
        source_indices=frozenset({23, 24}),
        source_observations=frozenset({("probe", 100), ("probe", 101)}),
        checkpoint_evidence=tuple((index, arrival, 21)
                                  for index, arrival, _revision in first.checkpoint_evidence),
        priority_indices=frozenset({23, 24}),
        exploratory_indices=frozenset({11, 12}),
    )
    return first, second, cold


@pytest.mark.asyncio
@pytest.mark.parametrize("reverse_candidates", [False, True])
@pytest.mark.parametrize("reverse_insertion", [False, True])
async def test_complete_cold_crossover_preserves_ids_and_later_certified_motion(
    reverse_candidates, reverse_insertion,
):
    first, second, cold = _complete_cold_fixture()
    tracker = MarkerTracker()
    initial = await tracker.update(_snapshot(1), [first, second])
    first_id, second_id = [marker.track_id for marker in initial]
    tracks = tracker._routes[("KMB", "R", "out")]
    original = tracks[first_id]
    boundary_time = original.boundary_observed_at
    next_id = tracker._next_id
    if reverse_insertion:
        tracker._routes[("KMB", "R", "out")] = dict(reversed(list(tracks.items())))
    candidates = [cold, second] if reverse_candidates else [second, cold]
    current = await tracker.update(_snapshot(2), candidates)
    assert len(current) == 2
    by_id = {marker.track_id: marker for marker in current}
    assert set(by_id) == {first_id, second_id}
    assert tracker._next_id == next_id
    held = by_id[first_id]
    assert held.position == pytest.approx(4.5)
    assert (held.lat, held.lon) == (first.lat, first.lon)
    assert held.bracket == original.display_bracket == original.motion_bracket == first.bracket
    assert held.boundary_revision == original.boundary_revision == first.boundary_revision
    assert original.boundary_observed_at == boundary_time
    assert original.committed_boundary_evidence == first.checkpoint_evidence
    assert original.position_authoritative is True
    assert held.position_authoritative is False
    for name in ("source_indices", "source_observations", "checkpoint_evidence",
                 "priority_indices", "exploratory_indices"):
        assert getattr(held, name) == getattr(cold, name)
    assert original.cohort_evidence == cold.checkpoint_evidence
    assert original.cohort_trusted is True
    assert original.generation == 2
    assert by_id[second_id].position == pytest.approx(8.0)

    certified = replace(
        cold, position=5.5, bracket=(5.0, 6.0), boundary_revision=(22, 22),
        position_authoritative=True, priority_indices=frozenset({5, 6}),
        exploratory_indices=frozenset(),
        source_indices=frozenset({5, 6, 23, 24}),
        source_observations=frozenset({("probe", 200), ("probe", 201)}),
        checkpoint_evidence=((5, BASE_TIME.timestamp(), 22),
                             (6, BASE_TIME.timestamp() + 60, 22),
                             *cold.checkpoint_evidence),
    )
    final = await tracker.update(_snapshot(3), [second, certified])
    assert len(final) == 2
    by_id = {marker.track_id: marker for marker in final}
    assert set(by_id) == {first_id, second_id}
    moved = by_id[first_id]
    assert moved.position == pytest.approx(5.5)
    assert moved.bracket == (5.0, 6.0)
    assert moved.boundary_revision == (22, 22)
    assert moved.position_authoritative is True
    assert moved.exploratory_indices == frozenset()
    assert moved.priority_indices == frozenset({5, 6})
    assert original.committed_boundary_evidence == certified.checkpoint_evidence
    assert by_id[second_id].position == pytest.approx(8.0)
    assert tracker._next_id == next_id


@pytest.mark.parametrize("competition", ["cold_claimant", "authoritative_claimant", "shared_owner"])
def test_complete_cold_reservation_ambiguous_witness_fails_closed(competition):
    first, _second, cold = _complete_cold_fixture()
    old = [_Track(1, first, first.position, 1, position_authoritative=True,
                  cohort_observed_at=BASE_TIME.timestamp(),
                  cohort_evidence=first.checkpoint_evidence)]
    candidates = [cold]
    if competition == "shared_owner":
        old.append(replace(old[0], track_id=2, position=8.0))
    else:
        candidates.append(replace(
            cold, position=8.0,
            position_authoritative=competition == "authoritative_claimant",
        ))
    assert tracker_module._complete_cold_reservations(old, candidates) == set()
    assert tracker_module._complete_pair_plan(old, candidates, {})[3] == set()


@pytest.mark.parametrize("malformed_competitor", ["candidate", "owner"])
def test_complete_cold_reservation_malformed_competitor_cannot_create_uniqueness(
    malformed_competitor,
):
    first, _second, cold = _complete_cold_fixture()
    old = [_Track(1, first, first.position, 1, position_authoritative=True,
                  cohort_observed_at=BASE_TIME.timestamp(),
                  cohort_evidence=first.checkpoint_evidence)]
    candidates = [cold]
    if malformed_competitor == "candidate":
        candidates.append(replace(cold))
    else:
        old.append(replace(old[0], track_id=2))
    assert tracker_module._complete_cold_reservations(old, candidates) == set()
    malformed = ((True, 100.0, 21),)
    if malformed_competitor == "candidate":
        candidates[1] = replace(cold, checkpoint_evidence=cold.checkpoint_evidence + malformed)
    else:
        old[1] = replace(old[1], cohort_evidence=first.checkpoint_evidence + malformed)
    assert tracker_module._complete_cold_reservations(old, candidates) == set()


def test_complete_cold_reservation_counts_raw_and_certified_competitors_together():
    first, _second, cold = _complete_cold_fixture()
    old = [_Track(1, first, first.position, 1, position_authoritative=True)]
    certified = replace(cold, checkpoint_evidence=tuple(
        (index, arrival + 10, revision)
        for index, arrival, revision in cold.checkpoint_evidence
    ))
    assert tracker_module._complete_cold_reservations(
        old, [certified], reconciled_owners={id(certified): 1},
    ) == {(0, 0)}
    assert tracker_module._complete_cold_reservations(
        old, [certified, cold], reconciled_owners={id(certified): 1},
    ) == set()


@pytest.mark.asyncio
async def test_complete_cold_crossover_retains_atomic_reconciliation_with_arrival_drift():
    tracker = MarkerTracker()
    rows = tuple(_fresh_probe_row(
        index, BASE_TIME + timedelta(seconds=seconds), 20, BASE_TIME,
    ) for index, seconds in ((23, 300), (24, 360), (25, 420), (26, 480)))
    first = replace(_candidate_from_rows(4.5, (4.0, 5.0), (20, 20), rows, (0, 1)),
                    position_authoritative=True)
    second = replace(_candidate_from_rows(8.0, (7.0, 9.0), (20, 20), rows, (2, 3)),
                     position_authoritative=True)
    initial = await tracker.update(_snapshot(1, rows), [first, second])
    first_id, second_id = [marker.track_id for marker in initial]
    next_id = tracker._next_id
    now = BASE_TIME + timedelta(seconds=30)
    current_rows = tuple(_fresh_probe_row(
        row.index, row.arrival_at + timedelta(seconds=10 if row.index < 25 else 0),
        21, now,
    ) for row in rows)
    cold = replace(
        _candidate_from_rows(11.2, None, None, current_rows, (0, 1)),
        position_authoritative=False, priority_indices=frozenset({23, 24}),
        exploratory_indices=frozenset({11, 12}),
    )
    current_second = replace(
        _candidate_from_rows(8.0, (7.0, 9.0), (21, 21), current_rows, (2, 3)),
        position_authoritative=True,
    )
    tracks = list(tracker._routes[("KMB", "R", "out")].values())
    candidates, certificates = tracker_module._reconcile_complete_probe_ownership(
        tracks, [current_second, cold], now.timestamp(), current_rows, (),
    )
    assert certificates == {id(cold): first_id, id(current_second): second_id}
    assert tracker_module._complete_cold_reservations(tracks, candidates) == set()
    result = await tracker.update(_snapshot(2, current_rows, collected_at=now), candidates)
    assert len(result) == 2
    by_id = {marker.track_id: marker for marker in result}
    assert set(by_id) == {first_id, second_id}
    assert tracker._next_id == next_id
    held = by_id[first_id]
    assert held.position == pytest.approx(4.5)
    assert held.bracket == first.bracket
    assert held.boundary_revision == first.boundary_revision
    assert held.checkpoint_evidence == cold.checkpoint_evidence
    assert held.source_observations == cold.source_observations
    assert held.exploratory_indices == cold.exploratory_indices
    assert held.position_authoritative is False
    assert tracks[0].committed_boundary_evidence == first.checkpoint_evidence
    assert tracks[0].cohort_evidence == cold.checkpoint_evidence
    assert by_id[second_id].position == pytest.approx(8.0)


@pytest.mark.parametrize("ledger", [
    ((True, 100.0, 20),), ((23, "100", 20),), ((23, float("nan"), 20),),
    ((23, 100.0, True),), ((23, 100.0, "20"),), ((23, 100.0, 20, 1),),
    [(23, 100.0, 20)],
])
@pytest.mark.parametrize("malformed_old", [False, True])
def test_complete_cold_reservation_rejects_malformed_ledgers(ledger, malformed_old):
    first, _second, cold = _complete_cold_fixture()
    valid = ((23, 100.0, 20),)
    track = _Track(
        1, first, first.position, 1, position_authoritative=True,
        cohort_observed_at=BASE_TIME.timestamp(),
        cohort_evidence=ledger if malformed_old else valid,
    )
    candidate = replace(cold, checkpoint_evidence=valid if malformed_old else ledger)
    assert tracker_module._complete_cold_reservations([track], [candidate]) == set()


@pytest.mark.parametrize("revision,drift,expected", [(19, 0, False), (20, 0.5, True),
                                                    (21, 0.51, False)])
def test_complete_cold_reservation_requires_exact_nonregressing_witness(revision, drift, expected):
    first, _second, cold = _complete_cold_fixture()
    track = _Track(1, first, first.position, 1, position_authoritative=True)
    candidate = replace(cold, checkpoint_evidence=tuple(
        (index, arrival + drift, revision)
        for index, arrival, _revision in first.checkpoint_evidence
    ))
    assert bool(tracker_module._complete_cold_reservations([track], [candidate])) is expected


@pytest.mark.asyncio
async def test_complete_cold_hold_remains_authoritative_order_barrier():
    first, second, cold = _complete_cold_fixture()
    second = replace(second, position=6.0, bracket=(5.0, 7.0))
    tracker = MarkerTracker()
    initial = await tracker.update(_snapshot(1), [first, second])
    crossing = replace(second, position=3.5, bracket=(3.0, 4.0), boundary_revision=(21, 21))
    moving_track = tracker._routes[("KMB", "R", "out")][initial[1].track_id]
    assert tracker_module._paired_position(moving_track, crossing, False) == 3.5
    current = await tracker.update(_snapshot(2), [crossing, cold])
    assert [marker.track_id for marker in current] == [marker.track_id for marker in initial]
    assert [marker.position for marker in current] == pytest.approx([4.5, 6.0])
    held = tracker._routes[("KMB", "R", "out")][initial[0].track_id]
    assert held.position_authoritative is True
    assert held.estimate.position_authoritative is False


@pytest.mark.asyncio
async def test_complete_exact_cold_continuation_behind_forward_after_retains_id():
    first, _second, cold = _complete_cold_fixture()
    tracker = MarkerTracker()
    initial = await tracker.update(_snapshot(1), [first])
    held = tracker._routes[("KMB", "R", "out")][initial[0].track_id]
    held.forward_after = 15
    held.forward_frontier = (16, 17)
    held.forward_started_revision = 20
    current = await tracker.update(_snapshot(2), [cold])
    assert len(current) == 1
    assert current[0].track_id == initial[0].track_id
    assert current[0].position == pytest.approx(4.5)
    assert current[0].checkpoint_evidence == cold.checkpoint_evidence
    assert held.forward_after == 15
    assert held.forward_frontier == (16, 17)


@pytest.mark.asyncio
async def test_complete_gate_continuation_uses_physical_identity_across_revisions():
    tracker = MarkerTracker()
    arrival = BASE_TIME + timedelta(minutes=8)
    initial = replace(
        _candidate(10.5, gate=True, bracket=(10.0, 11.0), arrival_at=arrival,
                   boundary_revision=(30, 30)),
        checkpoint_evidence=((10, arrival.timestamp(), 30),
                             (11, (arrival + timedelta(seconds=60)).timestamp(), 30)),
    )
    first = await tracker.update(_snapshot(1), [initial])
    tracked = tracker._routes[("KMB", "R", "out")][first[0].track_id]
    tracked.forward_after = 10
    tracked.forward_revision = 30
    tracked.forward_frontier = (11, 12)
    tracked.forward_baselines = {11: 30, 12: 30}
    gate = replace(
        _candidate(11.5, gate=True, bracket=(11.0, 12.0), arrival_at=arrival,
                   boundary_revision=None),
        checkpoint_evidence=((10, arrival.timestamp(), 31),
                             (11, (arrival + timedelta(seconds=60)).timestamp(), 31)),
    )
    current = await tracker.update(_snapshot(2), [gate])

    assert current[0].track_id == first[0].track_id
    assert current[0].position == pytest.approx(11.5)
    assert current[0].boundary_revision == initial.boundary_revision
    assert current[0].bracket == (11.0, 12.0)
    assert tracked.forward_frontier == (11, 12)


@pytest.mark.asyncio
async def test_revisionless_stale_gate_candidate_with_unchanged_rows_holds():
    tracker = MarkerTracker()
    arrival = BASE_TIME + timedelta(minutes=8)
    initial = replace(
        _candidate(10.5, gate=True, bracket=(10.0, 11.0), boundary_age=0,
                   boundary_revision=(30, 30), arrival_at=arrival),
        checkpoint_evidence=((10, arrival.timestamp(), 30),
                             (11, (arrival + timedelta(seconds=60)).timestamp(), 30)),
    )
    await tracker.update(_snapshot(1), [initial])
    stale = replace(
        initial, position=11.5, bracket=(11.0, 12.0),
        boundary_age_seconds=800, boundary_revision=None,
    )
    current = await tracker.update(_snapshot(2), [stale])
    assert current[0].position == pytest.approx(10.5)


@pytest.mark.asyncio
async def test_complete_gmb11_cohort_rejoins_split_checkpoint_evidence():
    tracker = MarkerTracker()
    arrival = BASE_TIME + timedelta(minutes=9)
    guard = replace(
        _candidate(10.0, route="11", operator=Operator.GMB,
                   bracket=(9.0, 10.0), boundary_age=0,
                   boundary_revision=(20, 20)),
        source_observations=frozenset({("probe", 100)}),
        checkpoint_evidence=((10, (arrival - timedelta(minutes=3)).timestamp(), 20),),
    )
    target = replace(
        _candidate(14.0, route="11", operator=Operator.GMB,
                   bracket=(13.0, 14.0), boundary_age=0,
                   boundary_revision=(20, 20)),
        source_observations=frozenset({("probe", 130), ("probe", 140),
                                       ("probe", 150)}),
        checkpoint_evidence=((13, arrival.timestamp(), 20),
                             (14, (arrival + timedelta(seconds=60)).timestamp(), 20),
                             (15, (arrival + timedelta(seconds=120)).timestamp(), 20)),
    )
    initial = await tracker.update(
        _snapshot(1, route_key=("GMB", "11", "out")), [guard, target]
    )
    moved_guard = replace(
        guard, position=13.5, bracket=(13.0, 14.0),
        boundary_revision=(21, 21),
        source_observations=frozenset({("probe", 0)}),
        checkpoint_evidence=((10, (arrival - timedelta(minutes=3)).timestamp(), 21),),
    )
    downstream = replace(
        target, position=15.0, bracket=(14.0, 15.0),
        boundary_revision=(21, 20),
        source_observations=frozenset({("probe", 3)}),
        checkpoint_evidence=((15, (arrival + timedelta(seconds=120)).timestamp(), 21),),
    )
    newborn = replace(
        _candidate(12.0, route="11", operator=Operator.GMB,
                   bracket=(12.0, 13.0), boundary_age=0,
                   boundary_revision=(21, 21)),
        source_observations=frozenset({("probe", 1), ("probe", 2)}),
        checkpoint_evidence=((13, arrival.timestamp(), 21),
                             (14, (arrival + timedelta(seconds=60)).timestamp(), 21)),
    )
    response_rows = [
        ProbeEta(
            "GMB", "11", "out", f"stop-{index}", index, minutes,
            arrival_at=arrival_at, refresh_generation=21,
        )
        for index, minutes, arrival_at in (
            (10, 6.0, arrival - timedelta(minutes=3)),
            (13, 9.0, arrival),
            (14, 10.0, arrival + timedelta(seconds=60)),
            (15, 11.0, arrival + timedelta(seconds=120)),
        )
    ]
    current = await tracker.update(
        _snapshot(2, response_rows, route_key=("GMB", "11", "out")),
        [newborn, moved_guard, downstream],
    )
    assert len(current) == 2
    by_id = {marker.track_id: marker for marker in current}
    assert set(by_id) == {marker.track_id for marker in initial}
    # Identity-only history retains the split rows for the next complete
    # publication without rendering a third marker.
    stored = tracker._routes[("GMB", "11", "out")][initial[1].track_id]
    assert {row[0] for row in stored.cohort_evidence} == {13, 14, 15}


@pytest.mark.asyncio
async def test_complete_91m_split_terminal_reuses_retained_cohort_owner():
    """The live frame-96/97/98 shape cannot birth a terminal satellite."""
    tracker = MarkerTracker()
    route_key = ("KMB", "91M", "inbound")

    def observed(position, bracket, revision, evidence, token):
        return replace(
            _candidate(
                position,
                route="91M",
                bound="inbound",
                bracket=bracket,
                boundary_age=0,
                boundary_revision=revision,
            ),
            source_observations=frozenset({("probe", token)}),
            checkpoint_evidence=tuple(evidence),
        )

    arrivals = {
        16: (BASE_TIME + timedelta(seconds=10)).timestamp(),
        18: (BASE_TIME + timedelta(minutes=1)).timestamp(),
        19: (BASE_TIME + timedelta(minutes=3)).timestamp(),
        21: (BASE_TIME + timedelta(minutes=5)).timestamp(),
        27: (BASE_TIME + timedelta(minutes=16)).timestamp(),
    }
    complete = observed(
        16.0,
        (15.0, 16.0),
        (900, 900),
        [(index, arrivals[index], 906) for index in (18, 19, 21, 27)],
        10,
    )
    initial = await tracker.update(
        _snapshot(1, route_key=route_key), [complete],
        [_line(stops=28, route="91M", bound="inbound")],
    )

    upstream = observed(
        16.814,
        (16.0, 18.0),
        (906, 906),
        [(index, arrivals[index], 906) for index in (16, 18, 19, 21)],
        11,
    )
    terminal = observed(
        26.0, (21.0, 27.0), (906, 906), [(27, arrivals[27], 906)], 12
    )
    partial = await tracker.update(
        _snapshot(1, route_key=route_key), [upstream, terminal]
    )
    assert len(partial) == 1
    assert partial[0].position == pytest.approx(16.814)
    assert {row[0] for row in partial[0].checkpoint_evidence} == {16, 18, 19, 21}

    later = {index: arrival + 9 for index, arrival in arrivals.items()}
    base = observed(
        16.0,
        (16.0, 17.0),
        (734, 941),
        [(17, (BASE_TIME + timedelta(seconds=30)).timestamp(), 734),
         (18, later[18], 941)],
        0,
    )
    satellite = observed(
        26.0, (18.0, 27.0), (941, 941), [(27, later[27], 941)], 1
    )
    response_rows = [
        ProbeEta(
            "KMB", "91M", "inbound", f"s{index}", index, 1.0,
            arrival_at=datetime.fromtimestamp(later[index], tz=UTC),
            refresh_generation=941,
        )
        for index in (18, 27)
    ]
    reconciled = await tracker.update(
        _snapshot(
            2,
            response_rows,
            collected_at=BASE_TIME + timedelta(seconds=30),
            route_key=route_key,
        ),
        [base, satellite],
    )

    assert len(reconciled) == 1
    assert reconciled[0].track_id == initial[0].track_id
    assert reconciled[0].position == pytest.approx(16.814)
    stored = tracker._routes[route_key][initial[0].track_id]
    assert {row[0] for row in stored.cohort_evidence} == {17, 18, 27}


@pytest.mark.asyncio
@pytest.mark.parametrize("reverse", [False, True])
async def test_all_positive_cold_fragment_union_holds_and_conserves_current_metadata(reverse):
    tracker = MarkerTracker()
    line = _geometry_line(stops=10)
    route_key = ("KMB", "R", "out")
    initial_rows = [_fresh_probe_row(
        index, BASE_TIME + timedelta(minutes=offset), 10, BASE_TIME,
    ) for index, offset in ((4, -1 / 3), (6, 1 / 3), (8, 1.0), (9, 7 / 6))]
    candidates = estimate_bus_positions(
        initial_rows, [line],
        observed_checkpoint_indices={route_key: {0, 4, 6, 8, 9}},
    )
    assert len(candidates) == 1
    assert candidates[0].position == pytest.approx(5.0)
    assert candidates[0].bracket == (4.0, 6.0)
    initial = await tracker.update(_snapshot(1, initial_rows), candidates, [line])
    track = tracker._routes[route_key][initial[0].track_id]
    committed = track.committed_boundary_evidence
    boundary_time = track.boundary_observed_at
    next_id = tracker._next_id
    now = BASE_TIME + timedelta(seconds=30)
    current_rows = [_fresh_probe_row(
        index, now + timedelta(minutes=offset), 11, now,
    ) for index, offset in ((4, 0.2), (8, 0.7), (9, 0.8))]
    if reverse:
        current_rows.reverse()
    split = estimate_bus_positions(
        current_rows, [line], observed_checkpoint_indices={route_key: {4, 8, 9}},
    )
    assert [candidate.position for candidate in split] == pytest.approx([3.9, 8.6])
    assert all(candidate.position_authoritative is False for candidate in split)
    sources = Counter(source for candidate in split for source in candidate.source_observations)
    checkpoints = Counter(row for candidate in split for row in candidate.checkpoint_evidence)
    priorities = frozenset().union(*(candidate.priority_indices for candidate in split))
    exploratory = frozenset().union(*(candidate.exploratory_indices for candidate in split))
    if reverse:
        split.reverse()
    current = await tracker.update(_snapshot(2, current_rows, collected_at=now), split, [line])
    assert len(current) == 1
    held = current[0]
    assert held.track_id == initial[0].track_id
    assert tracker._next_id == next_id
    assert held.position == pytest.approx(5.0)
    assert held.bracket == initial[0].bracket
    assert held.boundary_revision == initial[0].boundary_revision
    assert held.position_authoritative is False
    assert track.position_authoritative is True
    assert track.boundary_observed_at == boundary_time
    assert track.committed_boundary_evidence == committed
    assert held.source_indices == frozenset({4, 8, 9})
    assert Counter(held.source_observations) == sources
    assert Counter(held.checkpoint_evidence) == checkpoints
    assert Counter(track.cohort_evidence) == checkpoints
    assert held.priority_indices == priorities
    assert held.exploratory_indices == exploratory


@pytest.mark.asyncio
@pytest.mark.parametrize("reverse", [False, True])
async def test_authoritative_base_with_cold_fragment_holds_failed_union(reverse):
    tracker = MarkerTracker()
    line = _geometry_line(stops=10)
    route_key = ("KMB", "R", "out")
    initial_rows = [_fresh_probe_row(
        index, BASE_TIME + timedelta(minutes=offset), 10, BASE_TIME,
    ) for index, offset in ((4, -0.7), (6, -0.5), (8, 1 / 6), (9, 4 / 15))]
    candidates = estimate_bus_positions(
        initial_rows, [line],
        observed_checkpoint_indices={route_key: {4, 6, 8, 9}},
    )
    assert len(candidates) == 1
    assert candidates[0].position == pytest.approx(7.5)
    initial = await tracker.update(_snapshot(1, initial_rows), candidates, [line])
    track = tracker._routes[route_key][initial[0].track_id]
    committed = track.committed_boundary_evidence
    boundary_time = track.boundary_observed_at
    next_id = tracker._next_id
    now = BASE_TIME + timedelta(seconds=30)
    current_rows = [_fresh_probe_row(
        index, now + timedelta(minutes=offset), 11, now,
    ) for index, offset in ((4, 0.2), (8, 0.7), (9, 0.8))]
    current_rows.insert(1, ProbeEta(
        "KMB", "R", "out", "s6", 6, None,
        cache_age_seconds=0.0, refresh_generation=11,
    ))
    if reverse:
        current_rows.reverse()
    split = estimate_bus_positions(
        current_rows, [line],
        observed_checkpoint_indices={route_key: {4, 6, 8, 9}},
    )
    assert [candidate.position for candidate in split] == pytest.approx([3.9, 7.65])
    fragment, base = split
    assert fragment.position_authoritative is False
    assert _position_order_authoritative(track, base)
    assert base.bracket == (6.0, 8.0)
    # The union's first present stop is 4, whose upstream frontier is unknown.
    assert tracker_module.rebuild_estimate_from_probe_fragments(
        base, [fragment], current_rows, [line],
    ) is base
    sources = Counter(source for candidate in split for source in candidate.source_observations)
    checkpoints = Counter(row for candidate in split for row in candidate.checkpoint_evidence)
    priorities = frozenset().union(*(candidate.priority_indices for candidate in split))
    exploratory = frozenset().union(*(candidate.exploratory_indices for candidate in split))
    if reverse:
        split.reverse()
    current = await tracker.update(_snapshot(2, current_rows, collected_at=now), split, [line])
    assert len(current) == 1
    held = current[0]
    assert held.track_id == initial[0].track_id
    assert tracker._next_id == next_id
    assert held.position == pytest.approx(7.5)
    assert held.bracket == initial[0].bracket
    assert held.boundary_revision == initial[0].boundary_revision
    assert held.position_authoritative is False
    assert track.position_authoritative is True
    assert track.boundary_observed_at == boundary_time
    assert track.committed_boundary_evidence == committed
    assert held.source_indices == frozenset({4, 8, 9})
    assert Counter(held.source_observations) == sources
    assert Counter(held.checkpoint_evidence) == checkpoints
    assert Counter(track.cohort_evidence) == checkpoints
    assert held.priority_indices == priorities
    assert held.exploratory_indices == exploratory


@pytest.mark.asyncio
async def test_all_positive_fragment_union_moves_to_first_present_boundary():
    """A split all-positive ladder rejoins at its upstream physical boundary."""
    tracker = MarkerTracker()
    line = _geometry_line(stops=10)
    route_key = ("KMB", "R", "out")

    def observed(index, signed_minutes, revision, collected_at):
        return ProbeEta(
            "KMB",
            "R",
            "out",
            f"s{index}",
            index,
            max(0.0, signed_minutes),
            cache_age_seconds=0.0,
            arrival_at=collected_at + timedelta(minutes=signed_minutes),
            observed_at=collected_at,
            refresh_generation=revision,
            signed_minutes=signed_minutes,
        )

    initial_rows = [
        observed(index, offset, 10, BASE_TIME)
        for index, offset in ((4, -1 / 3), (6, 1 / 3), (8, 1.0), (9, 7 / 6))
    ]
    initial_candidates = estimate_bus_positions(
        initial_rows,
        [line],
        observed_checkpoint_indices={route_key: {0, 4, 6, 8, 9}},
    )
    assert len(initial_candidates) == 1
    assert initial_candidates[0].position == pytest.approx(5.0)
    assert initial_candidates[0].bracket == (4.0, 6.0)
    initial = await tracker.update(
        _snapshot(1, initial_rows), initial_candidates, [line]
    )

    collected_at = BASE_TIME + timedelta(seconds=30)
    current_rows = [
        ProbeEta(
            "KMB", "R", "out", "s0", 0, None,
            cache_age_seconds=0.0, refresh_generation=11,
        ),
        observed(4, 0.2, 11, collected_at),
        observed(8, 0.7, 11, collected_at),
        observed(9, 0.8, 11, collected_at),
    ]
    split = estimate_bus_positions(
        current_rows,
        [line],
        observed_checkpoint_indices={route_key: {0, 4, 8, 9}},
    )
    assert [candidate.position for candidate in split] == pytest.approx([3.9, 8.6])

    current = await tracker.update(
        _snapshot(2, current_rows, collected_at=collected_at),
        split,
        [line],
    )
    assert len(current) == 1
    assert current[0].track_id == initial[0].track_id
    assert current[0].position == pytest.approx(3.9)
    assert current[0].bracket == (0.0, 4.0)
    assert current[0].source_indices == frozenset({4, 8, 9})
    assert {row[0] for row in current[0].checkpoint_evidence} == {4, 8, 9}
    assert current[0].priority_indices == frozenset({4, 8})
    assert current[0].boundary_revision == (11, 11)
    assert current[0].bracket_eta_offsets is None


@pytest.mark.asyncio
async def test_passed_checkpoint_keeps_upstream_fragment_as_new_departure():
    tracker = MarkerTracker()
    line = _geometry_line(stops=10)
    route_key = ("KMB", "R", "out")

    def observed(index, signed_minutes, revision, collected_at):
        return ProbeEta(
            "KMB",
            "R",
            "out",
            f"s{index}",
            index,
            max(0.0, signed_minutes),
            cache_age_seconds=0.0,
            arrival_at=collected_at + timedelta(minutes=signed_minutes),
            observed_at=collected_at,
            refresh_generation=revision,
            signed_minutes=signed_minutes,
        )

    initial_rows = [
        observed(index, offset, 10, BASE_TIME)
        for index, offset in ((4, -1 / 3), (6, 1 / 3), (8, 1.0), (9, 7 / 6))
    ]
    initial_candidates = estimate_bus_positions(
        initial_rows,
        [line],
        observed_checkpoint_indices={route_key: {0, 4, 6, 8, 9}},
    )
    initial = await tracker.update(
        _snapshot(1, initial_rows), initial_candidates, [line]
    )

    collected_at = BASE_TIME + timedelta(seconds=30)
    current_rows = [
        ProbeEta(
            "KMB", "R", "out", "s0", 0, None,
            cache_age_seconds=0.0, refresh_generation=11,
        ),
        observed(4, 0.2, 11, collected_at),
        ProbeEta(
            "KMB", "R", "out", "s6", 6, None,
            cache_age_seconds=0.0, refresh_generation=11,
        ),
        observed(8, 0.7, 11, collected_at),
        observed(9, 0.8, 11, collected_at),
    ]
    split = estimate_bus_positions(
        current_rows,
        [line],
        observed_checkpoint_indices={route_key: {0, 4, 6, 8, 9}},
    )
    current = await tracker.update(
        _snapshot(2, current_rows, collected_at=collected_at),
        split,
        [line],
    )

    assert [marker.position for marker in current] == pytest.approx([3.9, 7.65])
    by_id = {marker.track_id: marker for marker in current}
    assert by_id[initial[0].track_id].position == pytest.approx(7.65)
    assert next(
        marker for marker in current if marker.track_id != initial[0].track_id
    ).position == pytest.approx(3.9)


@pytest.mark.asyncio
async def test_rebuilt_fragment_cannot_reuse_stale_long_jump_authority():
    tracker = MarkerTracker()
    line = _geometry_line(stops=11)
    collected_at = BASE_TIME + timedelta(seconds=30)

    def current_row(index, signed_minutes):
        return ProbeEta(
            "KMB",
            "R",
            "out",
            f"s{index}",
            index,
            max(0.0, signed_minutes),
            cache_age_seconds=0.0,
            arrival_at=collected_at + timedelta(minutes=signed_minutes),
            observed_at=collected_at,
            refresh_generation=11,
            signed_minutes=signed_minutes,
        )

    current_rows = [
        current_row(4, -3.0),
        current_row(5, -2.0),
        current_row(10, 0.1),
    ]
    previous_evidence = tuple(
        (
            row.index,
            (row.arrival_at - timedelta(seconds=9)).timestamp(),
            10,
        )
        for row in current_rows
    )
    old = replace(
        _candidate(
            5.0,
            bracket=(4.0, 5.0),
            boundary_age=0.0,
            boundary_revision=(10, 10),
        ),
        checkpoint_evidence=previous_evidence,
    )
    initial = await tracker.update(_snapshot(1), [old], [line])
    split = estimate_bus_positions(
        current_rows,
        [line],
        observed_checkpoint_indices={("KMB", "R", "out"): {4, 5, 10}},
    )
    assert len(split) == 2
    assert [candidate.position for candidate in split] == pytest.approx([5.0, 9.95])

    held = await tracker.update(
        _snapshot(2, current_rows, collected_at=collected_at),
        split,
        [line],
    )
    assert len(held) == 1
    assert held[0].track_id == initial[0].track_id
    assert held[0].position == pytest.approx(5.0)
    assert held[0].bracket == (4.0, 5.0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "malformed_token",
    [("probe",), ("probe", True), ("probe", -1), ("probe", 999)],
)
async def test_malformed_probe_token_keeps_fragment_cardinality(malformed_token):
    tracker = MarkerTracker()
    early = BASE_TIME + timedelta(minutes=5)
    late = BASE_TIME + timedelta(minutes=10)
    initial_candidate = replace(
        _candidate(5.0),
        checkpoint_evidence=(
            (5, early.timestamp(), 10),
            (10, late.timestamp(), 10),
        ),
    )
    initial = await tracker.update(_snapshot(1), [initial_candidate])
    rows = [
        ProbeEta(
            "KMB", "R", "out", "s5", 5, 0.0,
            arrival_at=early + timedelta(seconds=9), refresh_generation=11,
        ),
        ProbeEta(
            "KMB", "R", "out", "s10", 10, 1.0,
            arrival_at=late + timedelta(seconds=9), refresh_generation=11,
        ),
    ]
    base = replace(
        _candidate(5.1),
        source_observations=frozenset({("probe", 0)}),
        checkpoint_evidence=((5, rows[0].arrival_at.timestamp(), 11),),
    )
    fragment = replace(
        _candidate(9.0),
        source_observations=frozenset({malformed_token}),
        checkpoint_evidence=((10, rows[1].arrival_at.timestamp(), 11),),
    )

    current = await tracker.update(_snapshot(2, rows), [base, fragment])
    assert len(current) == 2
    assert initial[0].track_id in {marker.track_id for marker in current}


@pytest.mark.asyncio
async def test_new_base_row_participates_in_cohort_chronology():
    tracker = MarkerTracker()
    epoch = BASE_TIME.timestamp()
    initial_candidate = replace(
        _candidate(5.0),
        checkpoint_evidence=((5, epoch + 300.0, 10), (10, epoch + 600.0, 10)),
    )
    initial = await tracker.update(_snapshot(1), [initial_candidate])
    rows = [
        ProbeEta(
            "KMB", "R", "out", "s5", 5, 0.0,
            arrival_at=datetime.fromtimestamp(epoch + 309.0, tz=UTC),
            refresh_generation=11,
        ),
        ProbeEta(
            "KMB", "R", "out", "s7", 7, 10.0,
            arrival_at=datetime.fromtimestamp(epoch + 1000.0, tz=UTC),
            refresh_generation=11,
        ),
        ProbeEta(
            "KMB", "R", "out", "s10", 10, 1.0,
            arrival_at=datetime.fromtimestamp(epoch + 609.0, tz=UTC),
            refresh_generation=11,
        ),
    ]
    base = replace(
        _candidate(5.1),
        source_observations=frozenset({("probe", 0), ("probe", 1)}),
        checkpoint_evidence=((5, epoch + 309.0, 11), (7, epoch + 1000.0, 11)),
    )
    fragment = replace(
        _candidate(9.0),
        source_observations=frozenset({("probe", 2)}),
        checkpoint_evidence=((10, epoch + 609.0, 11),),
    )

    current = await tracker.update(_snapshot(2, rows), [base, fragment])
    assert len(current) == 2
    assert initial[0].track_id in {marker.track_id for marker in current}


def _cohort_split_candidates(*, extra_rows, extra_tokens=(2,)):
    early = (BASE_TIME + timedelta(minutes=5)).timestamp()
    late = (BASE_TIME + timedelta(minutes=10)).timestamp()
    initial = replace(
        _candidate(5.0),
        checkpoint_evidence=((5, early, 10), (10, late, 10)),
    )
    base = replace(
        _candidate(5.1),
        source_observations=frozenset({("probe", 1)}),
        checkpoint_evidence=((5, early + 9, 11),),
    )
    extras = [
        replace(
            _candidate(9.0),
            source_observations=frozenset({("probe", token)}),
            checkpoint_evidence=tuple(rows),
        )
        for token, rows in zip(extra_tokens, extra_rows, strict=True)
    ]
    return initial, base, extras, late


@pytest.mark.asyncio
async def test_cohort_fragment_with_unowned_row_retains_normal_birth():
    late = (BASE_TIME + timedelta(minutes=10)).timestamp()
    initial, base, extras, _late = _cohort_split_candidates(
        extra_rows=[((10, late + 9, 11), (11, late + 69, 11))]
    )
    tracker = MarkerTracker()
    await tracker.update(_snapshot(1), [initial])

    current = await tracker.update(_snapshot(2), [base, *extras])

    assert len(current) == 2


@pytest.mark.asyncio
async def test_equal_terminal_fragments_cannot_spend_one_cohort_occurrence():
    late = (BASE_TIME + timedelta(minutes=10)).timestamp()
    continuation = ((10, late + 9, 11),)
    initial, base, extras, _late = _cohort_split_candidates(
        extra_rows=[continuation, continuation], extra_tokens=(2, 3)
    )
    tracker = MarkerTracker()
    await tracker.update(_snapshot(1), [initial])

    current = await tracker.update(_snapshot(2), [base, *extras])

    assert len(current) == 3
    assert len({marker.track_id for marker in current}) == 3


@pytest.mark.asyncio
async def test_unowned_base_occurrence_prevents_same_stop_fragment_collapse():
    early = (BASE_TIME + timedelta(minutes=5)).timestamp()
    late = (BASE_TIME + timedelta(minutes=10)).timestamp()
    initial = replace(
        _candidate(5.0),
        checkpoint_evidence=((5, early, 10), (10, late, 10)),
    )
    base = replace(
        _candidate(5.1),
        source_observations=frozenset({("probe", 1)}),
        checkpoint_evidence=(
            (5, early + 9, 11),
            (10, late + 600, 11),
        ),
    )
    fragment = replace(
        _candidate(9.0),
        source_observations=frozenset({("probe", 2)}),
        checkpoint_evidence=((10, late + 9, 11),),
    )
    tracker = MarkerTracker()
    await tracker.update(_snapshot(1), [initial])

    current = await tracker.update(_snapshot(2), [base, fragment])

    assert len(current) == 2
    assert sum(
        row[0] == 10
        for marker in current
        for row in marker.checkpoint_evidence
    ) == 2


@pytest.mark.asyncio
async def test_gate_backed_fragment_retains_authoritative_cardinality():
    initial, base, extras, _late = _cohort_split_candidates(
        extra_rows=[((
            10,
            (BASE_TIME + timedelta(minutes=10)).timestamp() + 9,
            11,
        ),)]
    )
    base = replace(
        base,
        source_observations=frozenset({("gate", 0), ("probe", 1)}),
    )
    extra = replace(
        extras[0],
        source_observations=frozenset({("gate", 1), ("probe", 2)}),
    )
    tracker = MarkerTracker()
    await tracker.update(_snapshot(1), [initial])

    current = await tracker.update(_snapshot(2), [base, extra])

    assert len(current) == 2


@pytest.mark.asyncio
async def test_fragment_occurrence_reserved_by_other_base_is_not_consumed():
    early = (BASE_TIME + timedelta(minutes=5)).timestamp()
    middle = (BASE_TIME + timedelta(minutes=7)).timestamp()
    late = (BASE_TIME + timedelta(minutes=10)).timestamp()
    first = replace(
        _candidate(5.0),
        checkpoint_evidence=((5, early, 10), (10, late, 10)),
    )
    second = replace(
        _candidate(6.0),
        checkpoint_evidence=((9, middle, 10),),
    )
    base_first = replace(
        _candidate(5.1),
        source_observations=frozenset({("probe", 1)}),
        checkpoint_evidence=((5, early + 9, 11),),
    )
    base_second = replace(
        _candidate(6.1),
        source_observations=frozenset({("probe", 2)}),
        checkpoint_evidence=((9, middle + 9, 11), (10, late + 9, 11)),
    )
    fragment = replace(
        _candidate(9.0),
        source_observations=frozenset({("probe", 3)}),
        checkpoint_evidence=((10, late + 9, 11),),
    )
    tracker = MarkerTracker()
    await tracker.update(_snapshot(1), [first, second])

    current = await tracker.update(
        _snapshot(2), [base_first, base_second, fragment]
    )

    assert len(current) == 3


@pytest.mark.asyncio
async def test_cohort_fragment_rebuilds_final_due_first_future_boundary():
    tracker = MarkerTracker()
    lower_at = BASE_TIME + timedelta(minutes=5)
    upper_at = BASE_TIME + timedelta(minutes=10)
    initial = replace(
        _candidate(
            5.0,
            bracket=(5.0, 5.0),
            boundary_age=0,
            boundary_revision=(10, 10),
        ),
        checkpoint_evidence=(
            (5, lower_at.timestamp(), 10),
            (10, upper_at.timestamp(), 10),
        ),
    )
    await tracker.update(_snapshot(1), [initial])
    lower = ProbeEta(
        "KMB", "R", "out", "s5", 5, 0.0,
        cache_age_seconds=1.0,
        arrival_at=lower_at + timedelta(seconds=9),
        refresh_generation=11,
        signed_minutes=-0.2,
    )
    upper = ProbeEta(
        "KMB", "R", "out", "s10", 10, 2.0,
        cache_age_seconds=1.0,
        arrival_at=upper_at + timedelta(seconds=9),
        refresh_generation=11,
        signed_minutes=2.0,
    )
    base = replace(
        _candidate(
            5.0,
            bracket=(5.0, 5.0),
            boundary_age=1.0,
            boundary_revision=(11, 11),
        ),
        source_observations=frozenset({("probe", 0)}),
        checkpoint_evidence=((5, lower.arrival_at.timestamp(), 11),),
    )
    fragment = replace(
        _candidate(
            9.0,
            bracket=(9.0, 10.0),
            boundary_age=1.0,
            boundary_revision=(11, 11),
        ),
        source_observations=frozenset({("probe", 1)}),
        checkpoint_evidence=((10, upper.arrival_at.timestamp(), 11),),
    )

    current = await tracker.update(
        _snapshot(
            2,
            [lower, upper],
            collected_at=BASE_TIME + timedelta(seconds=30),
        ),
        [base, fragment],
        [_line(stops=11)],
    )

    assert len(current) == 1
    assert current[0].bracket == (5.0, 10.0)
    assert current[0].position == pytest.approx(5.0 + 5.0 * (0.2 / 2.2))
    assert current[0].priority_indices == frozenset({5, 10})
    assert current[0].source_observations == frozenset({
        ("probe", 0), ("probe", 1),
    })


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("arrival_drift", "revision", "elapsed"),
    [(120, 11, 30), (9, 10, 30), (0, 10, 121)],
)
async def test_unproved_or_expired_cohort_fragment_retains_normal_birth(
    arrival_drift, revision, elapsed
):
    late = (BASE_TIME + timedelta(minutes=10)).timestamp()
    initial, base, extras, _late = _cohort_split_candidates(
        extra_rows=[((10, late + arrival_drift, revision),)]
    )
    tracker = MarkerTracker()
    await tracker.update(_snapshot(1), [initial])

    current = await tracker.update(
        _snapshot(2, collected_at=BASE_TIME + timedelta(seconds=elapsed)),
        [base, *extras],
    )

    assert len(current) == 2


@pytest.mark.asyncio
async def test_complete_same_physical_checkpoint_multiplicity_keeps_tied_tracks():
    tracker = MarkerTracker()
    arrival = BASE_TIME + timedelta(minutes=6)
    first = replace(
        _candidate(8.0, bracket=(8.0, 9.0), boundary_age=0,
                   boundary_revision=(30, 30)),
        source_observations=frozenset({("probe", 801)}),
        checkpoint_evidence=((12, arrival.timestamp(), 30),),
    )
    second = replace(
        _candidate(8.0, bracket=(8.0, 9.0), boundary_age=0,
                   boundary_revision=(31, 31)),
        source_observations=frozenset({("probe", 802)}),
        checkpoint_evidence=((12, arrival.timestamp(), 31),),
    )
    initial = await tracker.update(_snapshot(1), [first, second])
    refreshed = [
        replace(first, position=8.4, boundary_revision=(40, 40),
                checkpoint_evidence=((12, arrival.timestamp(), 40),)),
        replace(second, position=8.6, boundary_revision=(41, 41),
                checkpoint_evidence=((12, arrival.timestamp(), 41),)),
    ]
    current = await tracker.update(_snapshot(2), refreshed)

    assert len(current) == 2
    assert {marker.track_id for marker in current} == {
        marker.track_id for marker in initial
    }
    assert [marker.position for marker in current] == pytest.approx([8.4, 8.6])


@pytest.mark.asyncio
async def test_complete_tied_tracks_keep_crossed_physical_checkpoint_ids():
    tracker = MarkerTracker()
    arrival = BASE_TIME + timedelta(minutes=7)
    first = replace(_candidate(8.0, bracket=(8.0, 9.0), boundary_age=0,
                               boundary_revision=(20, 20)),
                    checkpoint_evidence=((12, arrival.timestamp(), 20),))
    second = replace(_candidate(8.0, bracket=(8.0, 9.0), boundary_age=0,
                                boundary_revision=(21, 21)),
                     checkpoint_evidence=((13, arrival.timestamp(), 21),))
    initial = await tracker.update(_snapshot(1), [first, second])
    crossed = [
        replace(second, position=8.2, boundary_revision=(30, 30)),
        replace(first, position=8.4, boundary_revision=(31, 31)),
    ]
    current = await tracker.update(_snapshot(2), crossed)
    by_position = {marker.position: marker.track_id for marker in current}
    assert by_position[8.2] == initial[1].track_id
    assert by_position[8.4] == initial[0].track_id


@pytest.mark.asyncio
async def test_complete_crossed_tie_component_preserves_strict_neighbor_ids():
    tracker = MarkerTracker()
    at = BASE_TIME + timedelta(minutes=7)
    predecessor = replace(_candidate(6.0, bracket=(6.0, 7.0), boundary_age=0,
                                     boundary_revision=(20, 20)),
                          checkpoint_evidence=((6, at.timestamp(), 20),))
    first = replace(_candidate(8.0, bracket=(8.0, 9.0), boundary_age=0,
                               boundary_revision=(21, 21)),
                    checkpoint_evidence=((12, at.timestamp(), 21),))
    second = replace(_candidate(8.0, bracket=(8.0, 9.0), boundary_age=0,
                                boundary_revision=(22, 22)),
                     checkpoint_evidence=((13, at.timestamp(), 22),))
    successor = replace(_candidate(10.0, bracket=(10.0, 11.0), boundary_age=0,
                                   boundary_revision=(23, 23)),
                        checkpoint_evidence=((15, at.timestamp(), 23),))
    initial = await tracker.update(_snapshot(1), [predecessor, first, second, successor])
    refreshed = [
        replace(predecessor, boundary_revision=(30, 30)),
        replace(second, position=8.2, boundary_revision=(31, 31)),
        replace(first, position=9.2, bracket=(9.0, 10.0),
                boundary_revision=(32, 32)),
        replace(successor, boundary_revision=(33, 33)),
    ]
    current = await tracker.update(_snapshot(2), refreshed)
    by_position = {marker.position: marker.track_id for marker in current}
    assert by_position[6.0] == initial[0].track_id
    assert by_position[8.2] == initial[2].track_id
    assert by_position[9.2] == initial[1].track_id
    assert by_position[10.0] == initial[3].track_id


@pytest.mark.asyncio
async def test_complete_two_crossed_tie_components_preserve_all_physical_ids():
    tracker = MarkerTracker()
    at = BASE_TIME + timedelta(minutes=7)

    def tied(position, stop, revision):
        return replace(
            _candidate(position, bracket=(position, position + 1.0),
                       boundary_age=0, boundary_revision=(revision, revision)),
            checkpoint_evidence=((stop, at.timestamp(), revision),),
        )

    first, second = tied(8.0, 12, 20), tied(8.0, 13, 21)
    third, fourth = tied(12.0, 18, 22), tied(12.0, 19, 23)
    initial = await tracker.update(
        _snapshot(1), [first, second, third, fourth]
    )
    current = await tracker.update(_snapshot(2), [
        replace(second, position=8.2, boundary_revision=(31, 31)),
        replace(first, position=8.4, boundary_revision=(32, 32)),
        replace(fourth, position=12.2, boundary_revision=(33, 33)),
        replace(third, position=12.4, boundary_revision=(34, 34)),
    ])

    by_position = {marker.position: marker.track_id for marker in current}
    assert by_position == {
        8.2: initial[1].track_id,
        8.4: initial[0].track_id,
        12.2: initial[3].track_id,
        12.4: initial[2].track_id,
    }


@pytest.mark.asyncio
async def test_crossed_tie_survivors_are_sorted_before_merging_middle_birth():
    tracker = MarkerTracker()
    at = BASE_TIME + timedelta(minutes=7)
    first = replace(
        _candidate(4.0, bracket=(4.0, 5.0), boundary_age=0,
                   boundary_revision=(20, 20)),
        checkpoint_evidence=((8, at.timestamp(), 20),),
    )
    second = replace(
        _candidate(4.0, bracket=(4.0, 5.0), boundary_age=0,
                   boundary_revision=(21, 21)),
        checkpoint_evidence=((9, (at + timedelta(minutes=1)).timestamp(), 21),),
    )
    initial = await tracker.update(_snapshot(1), [first, second])
    newborn = replace(
        _candidate(4.2, bracket=(4.2, 5.2), boundary_age=0,
                   boundary_revision=(32, 32)),
        checkpoint_evidence=((30, (at + timedelta(minutes=5)).timestamp(), 32),),
    )
    current = await tracker.update(_snapshot(2), [
        replace(second, position=4.1, boundary_revision=(30, 30)),
        newborn,
        replace(first, position=4.3, boundary_revision=(31, 31)),
    ])

    assert [marker.position for marker in current] == pytest.approx([4.1, 4.2, 4.3])
    assert current[0].track_id == initial[1].track_id
    assert current[2].track_id == initial[0].track_id
    assert current[1].track_id not in {marker.track_id for marker in initial}
    assert current[1].bracket == newborn.bracket


@pytest.mark.asyncio
async def test_large_tied_component_uses_unique_checkpoint_permutation():
    tracker = MarkerTracker()
    at = BASE_TIME + timedelta(minutes=8)
    tied = [
        replace(
            _candidate(8.0, bracket=(8.0, 9.0), boundary_age=0,
                       boundary_revision=(20 + index, 20 + index)),
            source_observations=frozenset({("probe", 800 + index)}),
            checkpoint_evidence=((20 + index, at.timestamp(), 20 + index),),
        )
        for index in range(13)
    ]
    initial = await tracker.update(_snapshot(1), tied)
    refreshed = [
        replace(
            tied[12 - index],
            position=8.01 + index * 0.01,
            boundary_revision=(50 + index, 50 + index),
        )
        for index in range(13)
    ]
    current = await tracker.update(_snapshot(2), refreshed)

    assert [marker.track_id for marker in current] == [
        initial[12 - index].track_id for index in range(13)
    ]
    assert [marker.position for marker in current] == pytest.approx(
        [8.01 + index * 0.01 for index in range(13)]
    )


@pytest.mark.asyncio
async def test_large_ambiguous_tie_does_not_authorize_exact_long_jumps():
    tracker = MarkerTracker()
    at = BASE_TIME + timedelta(minutes=8)
    tied = [
        replace(
            _candidate(8.0, bracket=(8.0, 9.0), boundary_age=0,
                       boundary_revision=(20 + index, 20 + index)),
            source_observations=frozenset({("probe", 800 + index)}),
            checkpoint_evidence=((20, at.timestamp(), 20 + index),),
        )
        for index in range(13)
    ]
    successor = replace(
        _candidate(10.0, bracket=(9.0, 10.0), boundary_age=0,
                   boundary_revision=(40, 40)),
        checkpoint_evidence=((30, (at + timedelta(minutes=5)).timestamp(), 40),),
    )
    initial = await tracker.update(_snapshot(1), [*tied, successor])
    refreshed = [
        replace(
            tied[index],
            position=20.01 + index * 0.01,
            bracket=(20.0, 21.0),
            boundary_revision=(50 + index, 50 + index),
            source_observations=frozenset({("probe", 2000 + index)}),
            checkpoint_evidence=((20, at.timestamp(), 50 + index),),
        )
        for index in range(13)
    ]
    moved_successor = replace(
        successor,
        position=30.0,
        bracket=(29.0, 30.0),
        boundary_revision=(80, 80),
        checkpoint_evidence=((30, (at + timedelta(minutes=5)).timestamp(), 80),),
    )
    current = await tracker.update(_snapshot(2), [*refreshed, moved_successor])

    assert {marker.track_id for marker in current[:-1]}.isdisjoint(
        marker.track_id for marker in initial[:-1]
    )
    assert [marker.position for marker in current[:-1]] == pytest.approx(
        [20.01 + index * 0.01 for index in range(13)]
    )
    assert current[-1].track_id == initial[-1].track_id
    assert current[-1].position == pytest.approx(30.0)


@pytest.mark.asyncio
async def test_crossed_tie_block_composes_with_later_exact_long_jump_anchor():
    tracker = MarkerTracker()
    at = BASE_TIME + timedelta(minutes=8)
    first = replace(
        _candidate(8.0, bracket=(8.0, 9.0), boundary_age=0,
                   boundary_revision=(20, 20)),
        checkpoint_evidence=((12, at.timestamp(), 20),),
    )
    second = replace(
        _candidate(8.0, bracket=(8.0, 9.0), boundary_age=0,
                   boundary_revision=(21, 21)),
        checkpoint_evidence=((13, (at + timedelta(minutes=1)).timestamp(), 21),),
    )
    anchor = replace(
        _candidate(20.0, bracket=(19.0, 20.0), boundary_age=0,
                   boundary_revision=(22, 22)),
        checkpoint_evidence=((20, (at + timedelta(minutes=2)).timestamp(), 22),),
    )
    initial = await tracker.update(_snapshot(1), [first, second, anchor])
    current = await tracker.update(_snapshot(2), [
        replace(second, position=8.2, boundary_revision=(31, 31)),
        replace(first, position=8.4, boundary_revision=(32, 32)),
        replace(anchor, position=26.0, bracket=(25.0, 26.0),
                boundary_revision=(33, 33),
                checkpoint_evidence=((20, (at + timedelta(minutes=2)).timestamp(), 33),)),
    ])

    by_position = {marker.position: marker.track_id for marker in current}
    assert len(current) == 3
    assert by_position[8.2] == initial[1].track_id
    assert by_position[8.4] == initial[0].track_id
    assert by_position[26.0] == initial[2].track_id


def test_crossed_tie_block_preserves_adjacent_recovery_edge():
    old = [
        _Track(1, _candidate(0.0), 0.0, 1),
        _Track(2, _candidate(8.0), 8.0, 1),
        _Track(3, _candidate(8.0), 8.0, 1),
        _Track(4, _candidate(10.0), 10.0, 1),
    ]
    new = [_candidate(20.0), _candidate(8.2), _candidate(8.4), _candidate(10.2)]
    pairs = _ordered_pairs(
        old, new, recoveries={(0, 0)}, fixed={(1, 2), (2, 1)}
    )
    assert (0, 0) in pairs
    assert {(1, 2), (2, 1)}.issubset(pairs)


@pytest.mark.asyncio
async def test_same_generation_identity_filter_also_constrains_recovery_pairs():
    tracker = MarkerTracker()
    shared = BASE_TIME + timedelta(minutes=5)
    distant = replace(
        _candidate(
            0.0,
            bracket=(0.0, 1.0),
            boundary_age=0,
            boundary_revision=(10, 10),
        ),
        checkpoint_evidence=((20, (shared + timedelta(seconds=100)).timestamp(), 10),),
    )
    owner = replace(
        _candidate(
            5.0,
            bracket=(4.0, 5.0),
            boundary_age=0,
            boundary_revision=(10, 10),
        ),
        checkpoint_evidence=((10, shared.timestamp(), 30),),
    )
    initial = await tracker.update(_snapshot(1), [distant, owner])
    refreshed = replace(
        owner,
        position=6.0,
        bracket=(5.0, 6.0),
        boundary_revision=(11, 11),
        checkpoint_evidence=(
            *owner.checkpoint_evidence,
            (20, (shared + timedelta(seconds=100)).timestamp(), 11),
        ),
    )

    current = await tracker.update(
        _snapshot(1, collected_at=BASE_TIME + timedelta(seconds=30)),
        [refreshed],
    )

    assert [marker.track_id for marker in current] == [
        marker.track_id for marker in initial
    ]
    assert [marker.position for marker in current] == [0.0, 5.0]


@pytest.mark.asyncio
async def test_exact_recovery_uniqueness_excludes_incompatible_stale_owner():
    tracker = MarkerTracker()
    shared_arrival = (BASE_TIME + timedelta(minutes=5)).timestamp()
    stale = replace(
        _candidate(
            0.5,
            bracket=(0.0, 1.0),
            boundary_age=0,
            boundary_revision=(12, 12),
        ),
        checkpoint_evidence=((5, shared_arrival, 30),),
    )
    moving = replace(
        _candidate(
            1.5,
            bracket=(1.0, 2.0),
            boundary_age=0,
            boundary_revision=(10, 10),
        ),
        checkpoint_evidence=((5, shared_arrival, 30),),
    )
    initial = await tracker.update(_snapshot(1), [stale, moving])
    jumped = replace(
        moving,
        position=7.0,
        bracket=(6.0, 7.0),
        boundary_revision=(11, 11),
    )

    current = await tracker.update(_snapshot(1), [jumped])

    assert [marker.track_id for marker in current] == [
        marker.track_id for marker in initial
    ]
    assert [marker.position for marker in current] == [0.5, 7.0]


@pytest.mark.asyncio
async def test_render_local_source_collision_does_not_reserve_wrong_track():
    tracker = MarkerTracker()
    arrival = BASE_TIME + timedelta(minutes=3)
    owner = replace(
        _candidate(
            5.0,
            bracket=(4.0, 5.0),
            boundary_age=0,
            boundary_revision=(20, 20),
            arrival_at=arrival,
        ),
        source_observations=frozenset({("probe", 7)}),
        checkpoint_evidence=((4, arrival.timestamp(), 20),),
    )
    moving = replace(
        _candidate(
            7.0,
            bracket=(6.0, 7.0),
            boundary_age=0,
            boundary_revision=(10, 10),
            arrival_at=arrival + timedelta(minutes=1),
        ),
        source_observations=frozenset({("probe", 8)}),
        checkpoint_evidence=(
            (7, (arrival + timedelta(minutes=1)).timestamp(), 10),
        ),
    )
    initial = await tracker.update(_snapshot(1), [owner, moving])
    candidate = replace(
        _candidate(
            8.0,
            bracket=(7.0, 8.0),
            boundary_age=0,
            boundary_revision=(19, 21),
            arrival_at=arrival,
        ),
        # Input offsets and an ETA anchor can collide across renders. The
        # changed physical checkpoint proves this is not the owner's replay.
        source_observations=owner.source_observations,
        checkpoint_evidence=(
            (8, (arrival + timedelta(minutes=2)).timestamp(), 21),
        ),
    )

    current = await tracker.update(
        _snapshot(1, collected_at=BASE_TIME + timedelta(seconds=30)),
        [candidate],
    )

    by_id = {marker.track_id: marker for marker in current}
    assert by_id[initial[0].track_id].position == 5.0
    assert by_id[initial[1].track_id].position == 8.0
    assert by_id[initial[1].track_id].checkpoint_evidence == candidate.checkpoint_evidence


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


def _partial_birth_case(
    *,
    old_times=None,
    current_times=None,
    include_empty_lower=True,
    due_lower=False,
    birth_unreliable=False,
    birth_kind=EtaKind.REALTIME,
    ages=(1.0, 1.0),
    witness_revisions=(32, 33),
    mixed_revision=False,
    drop_birth_stop20=False,
    extra_unowned=False,
    later_due_stop=False,
    earlier_future_stop=False,
    bad_lower_signed=False,
    negative_birth_witness=False,
    single_old_fragment=False,
    single_old_fragment_revision=34,
    single_old_fragment_seconds=1500,
    boundary_row_ages=(1.0, 1.0),
    birth_bracket=None,
    lower_revision=30,
    baseline_lower_revision=30,
    complete_checkpoint_revisions=None,
    complete_row_mode=None,
):
    route_key = ("KMB", "R", "out")
    old_times = old_times or {"b": (120, 180), "a": (720, 780)}
    current_times = current_times or {
        "b": old_times["b"],
        "a": old_times["a"],
        "birth": (1320, 1380),
    }

    def evidence(label, revisions=(10, 11)):
        return tuple(
            (
                stop,
                (BASE_TIME + timedelta(seconds=current_times[label][number])).timestamp(),
                revisions[number],
            )
            for number, stop in enumerate((19, 20))
        )

    initial = [
        replace(
            _candidate(10.0),
            source_observations=frozenset({("probe", 900)}),
            checkpoint_evidence=tuple(
                (
                    stop,
                    (BASE_TIME + timedelta(seconds=old_times["a"][number])).timestamp(),
                    (10, 11)[number],
                )
                for number, stop in enumerate((19, 20))
            ),
        ),
        replace(
            _candidate(18.0),
            source_observations=frozenset({("probe", 901)}),
            checkpoint_evidence=tuple(
                (
                    stop,
                    (BASE_TIME + timedelta(seconds=old_times["b"][number])).timestamp(),
                    (10, 11)[number],
                )
                for number, stop in enumerate((19, 20))
            ),
        ),
    ]
    if single_old_fragment:
        initial[0] = replace(
            initial[0],
            checkpoint_evidence=initial[0].checkpoint_evidence + ((
                25,
                (BASE_TIME + timedelta(seconds=1500)).timestamp(),
                12,
            ),),
        )
    rows = []
    slots = {label: [] for label in ("a", "b", "birth")}

    def append_row(label, stop, seconds, revision, minutes, *, kind=EtaKind.REALTIME,
                   age=1.0, signed=None):
        slot = len(rows)
        rows.append(ProbeEta(
            "KMB", "R", "out", str(stop), stop, minutes,
            kind=kind,
            cache_age_seconds=age,
            arrival_at=(
                None if minutes is None
                else BASE_TIME + timedelta(seconds=seconds)
            ),
            refresh_generation=revision,
            signed_minutes=minutes if signed is None else signed,
        ))
        if label is not None:
            slots[label].append(slot)

    if due_lower:
        append_row("birth", 3, 430, 29, 0, kind=birth_kind)
        append_row(
            "birth", 4, 450, lower_revision, 0, kind=birth_kind,
            age=boundary_row_ages[0],
            signed=3 if bad_lower_signed else None,
        )
    elif include_empty_lower:
        append_row(
            None, 0, 0, lower_revision, None, age=boundary_row_ages[0],
        )
    if later_due_stop:
        append_row("birth", 5, 460, 30, 0, kind=birth_kind)
    if earlier_future_stop:
        append_row("birth", 5, 470, 31, 0.5, kind=birth_kind)
    append_row(
        "birth", 6, 480, 31, 1, kind=birth_kind,
        age=boundary_row_ages[1],
    )
    for number, stop in enumerate((19, 20)):
        revision = witness_revisions[number]
        age = ages[number]
        for label, minutes in (("b", 2), ("a", 12), ("birth", 22)):
            if negative_birth_witness and label == "birth" and stop == 19:
                minutes = -1
            append_row(
                label,
                stop,
                current_times[label][number],
                revision + int(mixed_revision and stop == 20 and label == "a"),
                minutes,
                kind=birth_kind if label == "birth" else EtaKind.REALTIME,
                age=age,
            )
    if single_old_fragment:
        append_row(
            "birth", 25, single_old_fragment_seconds,
            single_old_fragment_revision, 25,
        )
    if extra_unowned:
        append_row(None, 19, 1500, witness_revisions[0], 25, age=ages[0])

    if drop_birth_stop20:
        slots["birth"] = [
            slot for slot in slots["birth"] if rows[slot].index != 20
        ]

    def current_candidate(label, position):
        source_slots = slots[label]
        checkpoint_rows = tuple(
            (
                rows[slot].index,
                rows[slot].arrival_at.timestamp(),
                rows[slot].refresh_generation,
            )
            for slot in source_slots
        )
        kwargs = {}
        sources = frozenset(("probe", slot) for slot in source_slots)
        if label == "birth":
            kwargs = {
                "gate": True,
                "unreliable": birth_unreliable,
                "bracket": (
                    birth_bracket if birth_bracket is not None
                    else (4.0, 6.0) if due_lower else (0.0, 6.0)
                ),
                "boundary_age": max(ages),
                "boundary_revision": (lower_revision, 31),
                "arrival_at": BASE_TIME + timedelta(seconds=480),
            }
            sources = sources | {("gate", 20)}
        return replace(
            _candidate(position, **kwargs),
            source_observations=frozenset(sources),
            checkpoint_evidence=checkpoint_rows,
        )

    candidates = [
        current_candidate("birth", 4.0 if due_lower else 5.0),
        current_candidate("a", 10.0),
        current_candidate("b", 18.0),
    ]
    collected_at = BASE_TIME + timedelta(seconds=30)
    complete_rows = ()
    complete_observed = {0}
    default_complete_revisions = [(0, baseline_lower_revision)]
    if complete_row_mode is not None:
        complete_rows = (ProbeEta(
            "KMB", "R", "out", "6", 6, 1,
            cache_age_seconds=1.0,
            arrival_at=BASE_TIME + timedelta(seconds=480),
            refresh_generation=31,
            signed_minutes=1,
        ),)
        if complete_row_mode != "unobserved":
            complete_observed.add(6)
            default_complete_revisions.append((
                6, 30 if complete_row_mode == "mismatched" else 31,
            ))
    snapshot = ProbeEtaSnapshot(
        (ProbeRouteGeneration(
            route_key,
            complete_rows,
            1,
            collected_at,
            observed_checkpoint_indices=frozenset(complete_observed),
            checkpoint_revisions=(
                tuple(default_complete_revisions)
                if complete_checkpoint_revisions is None
                else complete_checkpoint_revisions
            ),
        ),),
        collected_at,
        positioning_rows=tuple(rows),
        positioning_checkpoints=frozenset(
            ("KMB", "R", "out", row.index) for row in rows
        ),
    )
    return initial, snapshot, candidates


@pytest.mark.asyncio
async def test_two_stop_census_births_one_partial_surplus_then_atomic_promotes_it():
    initial_candidates, snapshot, candidates = _partial_birth_case()
    tracker = MarkerTracker()
    initial = await tracker.update(_snapshot(1), initial_candidates)

    partial = await tracker.update(snapshot, candidates)

    complete_route = snapshot.complete_routes[0]
    assert complete_route.rows == ()
    assert complete_route.observed_checkpoint_indices == frozenset({0})
    assert complete_route.checkpoint_revisions == ((0, 30),)
    assert [marker.position for marker in partial] == pytest.approx([5.0, 10.0, 18.0])
    assert {marker.track_id for marker in initial} < {
        marker.track_id for marker in partial
    }
    newborn_id = next(
        marker.track_id for marker in partial
        if marker.track_id not in {item.track_id for item in initial}
    )
    route_key = ("KMB", "R", "out")
    newborn = tracker._routes[route_key][newborn_id]  # noqa: SLF001
    assert newborn.cohort_observed_at > 0
    assert newborn.cohort_trusted is False
    assert tracker.poll_lifecycle_routes() == frozenset({route_key})

    repeated = await tracker.update(snapshot, candidates)
    assert {marker.track_id for marker in repeated} == {
        marker.track_id for marker in partial
    }
    assert tracker._routes[route_key][newborn_id].position == 5.0  # noqa: SLF001

    complete = await tracker.update(
        _snapshot(2, collected_at=BASE_TIME + timedelta(seconds=60)),
        candidates,
    )
    assert {marker.track_id for marker in complete} == {
        marker.track_id for marker in partial
    }
    assert tracker._routes[route_key][newborn_id].cohort_trusted is True  # noqa: SLF001
    assert tracker.poll_lifecycle_routes() == frozenset()
    assert tracker.poll_lifecycle_requests() == {}


@pytest.mark.asyncio
async def test_partial_capacity_public_update_holds_dependent_move_after_claimant_rejection(monkeypatch):
    first = (8, (BASE_TIME + timedelta(minutes=3)).timestamp(), 10)
    second = (12, (BASE_TIME + timedelta(minutes=4)).timestamp(), 10)
    third = (20, (BASE_TIME + timedelta(minutes=5)).timestamp(), 10)
    def fresh(candidate, position, revision):
        return replace(
            candidate, position=position, bracket=(position, position + 1),
            boundary_age_seconds=0.0, boundary_revision=(revision, revision),
        )
    incoherent = (third, (21, (BASE_TIME + timedelta(minutes=4)).timestamp(), 10))
    initial_candidates = [
        fresh(_capacity_candidate(first, 4.0), 4.0, 10),
        fresh(_capacity_candidate(second, 5.0), 5.0, 10),
        fresh(replace(_capacity_candidate(third, 8.0), checkpoint_evidence=incoherent), 8.0, 10),
    ]
    tracker = MarkerTracker()
    complete_rows = tuple(
        _fresh_probe_row(index, BASE_TIME + timedelta(minutes=minutes), 10, BASE_TIME)
        for index, minutes in ((8, 3), (12, 4), (20, 5), (21, 4))
    )
    initial = await tracker.update(_snapshot(1), initial_candidates[:2])
    initial = await tracker.update(
        _atomic_snapshot(2, complete_rows), initial_candidates
    )
    route_key = ("KMB", "R", "out")
    third_track = tracker._routes[route_key][initial[2].track_id]  # noqa: SLF001
    assert third_track.cohort_trusted is False
    calls = []
    original_transaction = tracker_module._select_valid_partial_transaction

    def capture_transaction(*args, **kwargs):
        calls.append((args[3], args[4]))
        return original_transaction(*args, **kwargs)

    monkeypatch.setattr(
        tracker_module, "_select_valid_partial_transaction", capture_transaction
    )
    current = await tracker.update(
        _snapshot(2, collected_at=BASE_TIME + timedelta(seconds=30)),
        [fresh(_capacity_candidate(first, 6.0), 6.0, 11),
         fresh(replace(_capacity_candidate(third, 7.0),
                       checkpoint_evidence=(second, third)), 7.0, 11)],
    )

    assert calls and calls[-1][1] == {0: 6.0, 1: 7.0}
    retained = next(marker for marker in current if marker.track_id == initial[0].track_id)
    assert retained.position == pytest.approx(4.0)
    assert retained.checkpoint_evidence == initial[0].checkpoint_evidence
    for marker in current:
        if marker.bracket is not None:
            assert marker.bracket[0] <= marker.position <= marker.bracket[1]


@pytest.mark.asyncio
async def test_partial_birth_capacity_checks_final_public_population():
    initial_candidates, snapshot, candidates = _partial_birth_case()
    tracker = MarkerTracker()
    initial = await tracker.update(_snapshot(1), initial_candidates)
    enriched = replace(
        initial_candidates[0],
        bracket=(9.0, 10.0), boundary_age_seconds=0.0,
        boundary_revision=(34, 34),
        checkpoint_evidence=initial_candidates[0].checkpoint_evidence + (
            (6, 1767226080.0, 31),
        ),
    )
    enriched_current = await tracker.update(
        _snapshot(1), [enriched, initial_candidates[1]]
    )
    assert any(
        (6, 1767226080.0, 31) in marker.checkpoint_evidence
        for marker in enriched_current
    )
    current = await tracker.update(snapshot, candidates)
    route_key = ("KMB", "R", "out")
    multiplicity = Counter(
        row for marker in current for row in marker.checkpoint_evidence
    )
    candidate_multiplicity = Counter(
        row for candidate in candidates for row in candidate.checkpoint_evidence
    )
    # The reserved held continuation releases the stale checkpoint-6 claim
    # transactionally, leaving capacity for the already-certified surplus.
    assert len(current) == len(initial) + 1 == 3
    assert multiplicity[(6, 1767226080.0, 31)] <= candidate_multiplicity[
        (6, 1767226080.0, 31)
    ]
    assert tracker.poll_lifecycle_routes() == {route_key}
    assert route_key in tracker._partial_birth_generations  # noqa: SLF001

    ordinary_tracker = MarkerTracker()
    ordinary_initial = await ordinary_tracker.update(_snapshot(1), initial_candidates)
    ordinary = await ordinary_tracker.update(snapshot, candidates)
    assert len(ordinary) == len(ordinary_initial) + 1


@pytest.mark.asyncio
async def test_partial_birth_accepts_consistent_nonempty_complete_revision_metadata():
    initial_candidates, snapshot, candidates = _partial_birth_case(
        complete_row_mode="consistent",
    )
    tracker = MarkerTracker()
    initial = await tracker.update(_snapshot(1), initial_candidates)

    partial = await tracker.update(snapshot, candidates)

    assert len(partial) == len(initial) + 1
    assert snapshot.complete_routes[0].checkpoint_revisions == ((0, 30), (6, 31))


@pytest.mark.asyncio
async def test_partial_birth_at_capacity_keeps_established_tracks():
    initial_candidates, snapshot, candidates = _partial_birth_case()
    initial_candidates = [
        replace(initial_candidates[0], position=1.0),
        replace(initial_candidates[1], position=2.0),
    ]
    tracker = MarkerTracker(max_tracks_per_route=2)
    initial = await tracker.update(_snapshot(1), initial_candidates)

    partial = await tracker.update(snapshot, candidates)

    route_key = ("KMB", "R", "out")
    assert {marker.track_id for marker in partial} == {
        marker.track_id for marker in initial
    }
    assert [marker.position for marker in partial] == pytest.approx([1.0, 2.0])
    assert [marker.checkpoint_evidence for marker in partial] == [
        candidate.checkpoint_evidence for candidate in candidates[1:]
    ]
    assert [track.committed_boundary_evidence for track in tracker._routes[route_key].values()] == [
        marker.checkpoint_evidence for marker in initial
    ]
    assert route_key not in tracker._partial_birth_generations  # noqa: SLF001
    assert tracker.poll_lifecycle_routes() == frozenset({route_key})
    assert tracker.poll_lifecycle_requests() == {route_key: 1}


@pytest.mark.asyncio
async def test_partial_birth_preserves_consecutive_zero_lower_endpoint_placement():
    initial_candidates, snapshot, candidates = _partial_birth_case(due_lower=True)
    tracker = MarkerTracker()
    await tracker.update(_snapshot(1), initial_candidates)

    partial = await tracker.update(snapshot, candidates)

    birth_slots = {
        slot for source, slot in candidates[0].source_observations
        if source == "probe"
    }
    assert [
        (row.index, row.minutes)
        for slot, row in enumerate(snapshot.rows)
        if slot in birth_slots and row.minutes == 0
    ] == [(3, 0), (4, 0)]
    assert [marker.position for marker in partial] == pytest.approx([4.0, 10.0, 18.0])


@pytest.mark.parametrize(
    "case_kwargs",
    [
        {"include_empty_lower": False},
        {"birth_unreliable": True},
        {"birth_kind": EtaKind.SCHEDULED},
        {"ages": (60.0, 60.0)},
        {"ages": (1.0, 10.0)},
        {"witness_revisions": (9, 10)},
        {"mixed_revision": True},
        {"drop_birth_stop20": True},
        {"extra_unowned": True},
        {"due_lower": True, "later_due_stop": True},
        {"earlier_future_stop": True},
        {"due_lower": True, "bad_lower_signed": True},
        {"negative_birth_witness": True},
        {"single_old_fragment": True},
        {
            "single_old_fragment": True,
            "single_old_fragment_revision": 11,
        },
        {
            "single_old_fragment": True,
            "single_old_fragment_revision": 11,
            "single_old_fragment_seconds": 1600,
        },
        {"due_lower": True, "boundary_row_ages": (60.0, 1.0)},
        {"boundary_row_ages": (1.0, 60.0)},
        {"birth_bracket": (False, 6.0)},
        {"lower_revision": 29, "baseline_lower_revision": 30},
        {"complete_checkpoint_revisions": ()},
        {"complete_checkpoint_revisions": ((0, 30), (0, 30))},
        {"complete_checkpoint_revisions": ((False, 30),)},
        {"complete_checkpoint_revisions": ((0, False),)},
        {"complete_checkpoint_revisions": ((1, 30),)},
        {"complete_row_mode": "unobserved"},
        {"complete_row_mode": "mismatched"},
        {
            "old_times": {"b": (100, 150), "a": (600, 650)},
            "current_times": {
                "b": (100, 150), "a": (620, 670), "birth": (640, 690),
            },
        },
        {
            "old_times": {"b": (600, 660), "a": (640, 700)},
            "current_times": {
                "b": (610, 670), "a": (630, 690), "birth": (1320, 1380),
            },
        },
        {
            "old_times": {"b": (100, 160), "a": (600, 770)},
            "current_times": {
                "b": (100, 160), "a": (600, 770), "birth": (720, 750),
            },
        },
        {
            "old_times": {"b": (100, 160), "a": (600, 660)},
            "current_times": {
                "b": (100, 160), "a": (600, 660), "birth": (600, 720),
            },
        },
    ],
    ids=(
        "missing-empty-lower",
        "unreliable-birth",
        "scheduled-birth",
        "stale-census",
        "cross-response-age-skew",
        "checkpoint-rollback",
        "mixed-response-revision",
        "split-fragment",
        "unassigned-sibling",
        "declared-lower-is-not-final-due-stop",
        "declared-upper-is-not-first-future-stop",
        "clamped-minutes-disagree-with-signed-offset",
        "negative-live-minutes",
        "single-old-owned-fragment",
        "same-arrival-noncensus-revision-rollback",
        "changed-arrival-noncensus-revision-rollback",
        "stale-lower-boundary-row",
        "stale-upper-boundary-row",
        "boolean-bracket-component",
        "successful-empty-boundary-revision-rollback",
        "missing-complete-checkpoint-revision",
        "duplicate-complete-checkpoint-revision",
        "boolean-complete-checkpoint-index",
        "boolean-complete-checkpoint-revision",
        "incomplete-checkpoint-revision-coverage",
        "complete-row-missing-from-checkpoint-ledger",
        "complete-row-revision-disagrees-with-ledger",
        "globally-ambiguous-surplus",
        "alternating-owner-cycle",
        "rank-crossing",
        "tied-rank",
    ),
)
@pytest.mark.asyncio
async def test_partial_birth_certificate_fails_closed(case_kwargs):
    initial_candidates, snapshot, candidates = _partial_birth_case(**case_kwargs)
    tracker = MarkerTracker()
    initial = await tracker.update(_snapshot(1), initial_candidates)

    partial = await tracker.update(snapshot, candidates)

    assert {marker.track_id for marker in partial} == {
        marker.track_id for marker in initial
    }
    assert tracker.poll_lifecycle_routes() == frozenset({("KMB", "R", "out")})


@pytest.mark.asyncio
async def test_partial_population_requests_lifecycle_refresh_until_new_generation():
    tracker = MarkerTracker()
    route_key = ("KMB", "R", "out")
    initial = await tracker.update(_snapshot(1), [_candidate(1.0)])
    assert len(initial) == 1
    assert tracker.poll_lifecycle_routes() == frozenset()

    held = await tracker.update(
        _snapshot(1), [_candidate(1.0), _candidate(4.0)]
    )
    assert len(held) == 1
    assert tracker.poll_lifecycle_routes() == frozenset({route_key})

    # A later partial render that happens to match the old population cannot
    # cancel the request; only a different complete generation is authority.
    await tracker.update(_snapshot(1), [_candidate(1.0)])
    assert tracker.poll_lifecycle_routes() == frozenset({route_key})
    assert tracker.poll_lifecycle_requests() == {route_key: 1}

    refreshed = await tracker.update(
        _snapshot(2), [_candidate(1.0), _candidate(4.0)]
    )
    assert len(refreshed) == 2
    assert tracker.poll_lifecycle_routes() == frozenset()
    assert tracker.poll_lifecycle_requests() == {}


@pytest.mark.asyncio
async def test_equal_count_turnover_requests_refresh_from_physical_occurrences():
    tracker = MarkerTracker()
    route_key = ("KMB", "R", "out")
    arrivals = [BASE_TIME + timedelta(minutes=value) for value in (1, 3, 5, 7)]
    old = [
        replace(
            _candidate(position),
            checkpoint_evidence=((index, arrival.timestamp(), 10),),
        )
        for position, index, arrival in zip(
            (1.0, 5.0, 10.0, 16.0), (1, 5, 10, 16), arrivals, strict=True
        )
    ]
    initial = await tracker.update(_snapshot(1), old)
    replacement_at_terminal = BASE_TIME + timedelta(minutes=15)
    current = [
        replace(old[index], position=old[index].position + 0.1)
        for index in range(3)
    ] + [
        replace(
            _candidate(12.0),
            checkpoint_evidence=((
                16, replacement_at_terminal.timestamp(), 20,
            ),),
        )
    ]
    terminal_response = ProbeEta(
        "KMB", "R", "out", "terminal", 16, 10.0,
        arrival_at=replacement_at_terminal, refresh_generation=20,
    )

    held = await tracker.update(_snapshot(1, [terminal_response]), current)

    assert [marker.track_id for marker in held] == [
        marker.track_id for marker in initial
    ]
    assert tracker.poll_lifecycle_routes() == frozenset({route_key})

    refreshed = await tracker.update(_snapshot(2, [terminal_response]), current)
    assert len(refreshed) == 4
    assert initial[-1].track_id not in {marker.track_id for marker in refreshed}
    assert tracker.poll_lifecycle_routes() == frozenset()


@pytest.mark.asyncio
async def test_missing_checkpoint_with_downstream_continuity_is_not_turnover():
    tracker = MarkerTracker()
    departure = BASE_TIME + timedelta(minutes=2)
    downstream = BASE_TIME + timedelta(minutes=8)
    initial_candidate = replace(
        _candidate(2.0),
        checkpoint_evidence=(
            (2, departure.timestamp(), 10),
            (8, downstream.timestamp(), 11),
        ),
    )
    await tracker.update(_snapshot(1), [initial_candidate])
    later_departure = BASE_TIME + timedelta(minutes=20)
    stop_response = ProbeEta(
        "KMB", "R", "out", "s2", 2, 18.0,
        arrival_at=later_departure, refresh_generation=20,
    )
    downstream_candidate = replace(
        _candidate(8.0),
        checkpoint_evidence=((8, downstream.timestamp(), 11),),
    )

    await tracker.update(
        _snapshot(1, [stop_response]), [downstream_candidate]
    )

    assert tracker.poll_lifecycle_routes() == frozenset()


@pytest.mark.asyncio
async def test_equal_arrival_multiplicity_detects_one_for_one_turnover():
    tracker = MarkerTracker()
    route_key = ("KMB", "R", "out")
    shared = BASE_TIME + timedelta(minutes=5)
    initial_candidates = [
        replace(
            _candidate(position),
            checkpoint_evidence=((16, shared.timestamp(), 10),),
        )
        for position in (15.5, 16.0)
    ]
    await tracker.update(_snapshot(1), initial_candidates)
    replacement = BASE_TIME + timedelta(minutes=15)
    current = [
        initial_candidates[0],
        replace(
            _candidate(14.0),
            checkpoint_evidence=((16, replacement.timestamp(), 20),),
        ),
    ]
    response_rows = [
        ProbeEta(
            "KMB", "R", "out", "terminal", 16, minutes,
            arrival_at=arrival, refresh_generation=20,
        )
        for minutes, arrival in ((3.0, shared), (13.0, replacement))
    ]

    held = await tracker.update(_snapshot(1, response_rows), current)

    assert len(held) == 2
    assert tracker.poll_lifecycle_routes() == frozenset({route_key})


@pytest.mark.asyncio
async def test_empty_baseline_can_request_lifecycle_refresh_without_early_birth():
    tracker = MarkerTracker()
    route_key = ("KMB", "R", "out")
    assert await tracker.update(_snapshot(1), []) == []

    partial = await tracker.update(_snapshot(1), [_candidate(2.0)])

    assert partial == []
    assert tracker.poll_lifecycle_routes() == frozenset({route_key})
    tracker.clear()
    assert tracker.poll_lifecycle_routes() == frozenset()


@pytest.mark.asyncio
async def test_lifecycle_refresh_signal_is_pruned_with_route_eviction():
    tracker = MarkerTracker(max_routes=1)
    route_a = ("KMB", "A", "out")
    route_b = ("KMB", "B", "out")
    await tracker.update(_snapshot(1, route_key=route_a), [])
    await tracker.update(
        _snapshot(1, route_key=route_a),
        [_candidate(1.0, route="A")],
    )
    assert tracker.poll_lifecycle_routes() == frozenset({route_a})

    await tracker.update(
        _snapshot(1, route_key=route_b),
        [_candidate(1.0, route="B")],
    )

    assert tracker.poll_lifecycle_routes() == frozenset()


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
async def test_complete_generation_recovers_clean_fast_jump_and_rejects_mixed_near():
    tracker = MarkerTracker()
    jumped_arrival = BASE_TIME + timedelta(minutes=1)
    # The retained passing ladder is chronological: its upstream checkpoint
    # precedes the downstream one, so it is trusted recovery evidence.
    near_arrival = jumped_arrival - timedelta(seconds=89)
    future_arrival = jumped_arrival + timedelta(minutes=15)
    future = replace(
        _candidate(
            2.0,
            bracket=(2.0, 3.0),
            boundary_age=0,
            boundary_revision=(298, 335),
            arrival_at=future_arrival,
        ),
        checkpoint_evidence=(
            (2, (future_arrival - timedelta(minutes=2)).timestamp(), 298),
            (3, future_arrival.timestamp(), 335),
            (9, (future_arrival + timedelta(minutes=8)).timestamp(), 336),
        ),
    )
    passing = replace(
        _candidate(
            2.08,
            bracket=(2.0, 3.0),
            boundary_age=0,
            boundary_revision=(298, 335),
            arrival_at=near_arrival,
        ),
        checkpoint_evidence=(
            (3, near_arrival.timestamp(), 335),
            (9, jumped_arrival.timestamp(), 336),
        ),
    )
    initial = await tracker.update(_snapshot(1), [future, passing])

    # A complete staggered rebuild merges one checkpoint from the passing bus
    # into the near candidate, making its ETA anchor look like the closer
    # identity. The other candidate has jumped over several stops but retains
    # that bus's exact checkpoint. Recover the clean fast jump; without raw
    # rows to repartition, the mixed near candidate cannot continue an owner.
    rebuilt_near = replace(
        future,
        position=2.001,
        eta_arrival_at=near_arrival,
        checkpoint_evidence=(
            future.checkpoint_evidence[0],
            passing.checkpoint_evidence[0],
            future.checkpoint_evidence[2],
        ),
    )
    jumped = replace(
        passing,
        position=8.827,
        bracket=(8.0, 9.0),
        boundary_revision=(335, 336),
        eta_arrival_at=jumped_arrival,
        checkpoint_evidence=(
            passing.checkpoint_evidence[1],
            (11, (jumped_arrival + timedelta(minutes=2)).timestamp(), 439),
        ),
    )
    current = await tracker.update(_snapshot(2), [rebuilt_near, jumped])

    assert len(current) == 2
    assert current[1].track_id == initial[1].track_id
    assert current[0].track_id not in {marker.track_id for marker in initial}
    assert tracker._routes[("KMB", "R", "out")][current[0].track_id].cohort_evidence == ()
    assert [marker.position for marker in current] == pytest.approx([2.001, 8.827])
    assert sum(
        passing.checkpoint_evidence[1] in marker.checkpoint_evidence
        for marker in current
    ) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("trusted_history", [False, True])
async def test_same_generation_recovery_requires_trusted_dominant_evidence(trusted_history):
    """Only trusted coherent history can authorize a dominant recovery."""
    tracker = MarkerTracker()
    route_key = ("GMB", "11", "seq-2")
    later_at_8 = BASE_TIME + timedelta(minutes=7)
    earlier_at_8 = BASE_TIME + timedelta(minutes=1)
    later_terminal = BASE_TIME + timedelta(minutes=15 if trusted_history else 13)
    earlier_terminal = BASE_TIME + timedelta(minutes=9 if trusted_history else 1)
    later = replace(
        _candidate(
            7.0,
            route="11",
            operator=Operator.GMB,
            bound="seq-2",
            bracket=(0.0, 8.0),
            boundary_age=0,
            boundary_revision=(154, 426),
            arrival_at=later_at_8,
        ),
        checkpoint_evidence=(
            (8, later_at_8.timestamp(), 426),
            (13, (BASE_TIME + timedelta(minutes=14)).timestamp(), 417),
            (16, later_terminal.timestamp(), 418),
        ),
    )
    earlier = replace(
        _candidate(
            7.195,
            route="11",
            operator=Operator.GMB,
            bound="seq-2",
            bracket=(0.0, 8.0),
            boundary_age=0,
            boundary_revision=(154, 426),
            arrival_at=earlier_at_8,
        ),
        checkpoint_evidence=(
            (8, earlier_at_8.timestamp(), 426),
            (13, (BASE_TIME + timedelta(minutes=8)).timestamp(), 417),
            (16, earlier_terminal.timestamp(), 418),
        ),
    )
    initial = await tracker.update(
        _snapshot(430, route_key=route_key),
        [later, earlier],
        [_line(stops=17, route="11", bound="seq-2", operator="GMB")],
    )

    corrected_later = replace(
        later,
        position=0.255,
        bracket=(0.0, 4.0),
        boundary_revision=(460, 507),
        eta_arrival_at=BASE_TIME + timedelta(minutes=4),
        checkpoint_evidence=(
            (4, (BASE_TIME + timedelta(minutes=4)).timestamp(), 507),
            (8, (later_at_8 + timedelta(seconds=14)).timestamp(), 512),
            (13, (BASE_TIME + timedelta(minutes=14, seconds=10)).timestamp(), 506),
            (16, (BASE_TIME + timedelta(minutes=19)).timestamp(), 513),
        ),
    )
    refreshed_earlier = replace(
        earlier,
        position=7.526,
        bracket=(5.0, 8.0),
        boundary_revision=(508, 512),
        eta_arrival_at=earlier_at_8 + timedelta(seconds=28),
        checkpoint_evidence=(
            (8, (earlier_at_8 + timedelta(seconds=28)).timestamp(), 512),
            (13, (BASE_TIME + timedelta(minutes=8, seconds=32)).timestamp(), 506),
            # One weak accidental overlap with the later track's old terminal.
            (16, (later_terminal + timedelta(seconds=37)).timestamp(), 513),
        ),
    )
    passed_earlier = replace(
        earlier,
        position=15.523,
        bracket=(13.0, 16.0),
        boundary_revision=(506, 513),
        eta_arrival_at=earlier_terminal + timedelta(seconds=36),
        checkpoint_evidence=(
            (16, (earlier_terminal + timedelta(seconds=36)).timestamp(), 513),
        ),
    )

    current = await tracker.update(
        _snapshot(
            430,
            collected_at=BASE_TIME + timedelta(seconds=30),
            route_key=route_key,
        ),
        [corrected_later, refreshed_earlier, passed_earlier],
        [_line(stops=17, route="11", bound="seq-2", operator="GMB")],
    )

    # Supplemental evidence cannot birth the passed vehicle. The unique
    # two-track optimum uses trusted dominant continuations; backwards source
    # ladders remain untrusted displays and cannot move on apparent ownership.
    assert [marker.track_id for marker in current] == [
        initial[0].track_id,
        initial[1].track_id,
    ]
    assert [marker.position for marker in current] == pytest.approx(
        [0.255, 7.526] if trusted_history else [7.0, 7.195]
    )
    if not trusted_history:
        assert all(not track.cohort_trusted and not track.cohort_evidence
                   for track in tracker._routes[route_key].values())
    expected_evidence = [corrected_later, refreshed_earlier] if trusted_history else initial
    assert [marker.checkpoint_evidence for marker in current] == [
        marker.checkpoint_evidence for marker in expected_evidence
    ]


@pytest.mark.asyncio
async def test_exact_checkpoint_assignment_cannot_authorize_mixed_complete_owner():
    tracker = MarkerTracker()
    anchor = BASE_TIME + timedelta(minutes=1)
    first = replace(
        _candidate(
            0.0,
            bracket=(0.0, 1.0),
            boundary_age=0,
            boundary_revision=(10, 10),
        ),
        checkpoint_evidence=(
            (10, anchor.timestamp(), 10),
            (11, (anchor + timedelta(seconds=1000)).timestamp(), 10),
        ),
    )
    second = replace(
        _candidate(
            1.0,
            bracket=(1.0, 2.0),
            boundary_age=0,
            boundary_revision=(10, 10),
        ),
        checkpoint_evidence=(
            (12, (anchor + timedelta(seconds=2000)).timestamp(), 10),
        ),
    )
    initial = await tracker.update(_snapshot(1), [first, second])
    weaker_for_first = replace(
        first,
        position=6.0,
        bracket=(6.0, 7.0),
        boundary_revision=(11, 11),
        checkpoint_evidence=(
            (11, (anchor + timedelta(seconds=1000)).timestamp(), 11),
        ),
    )
    exact_for_first = replace(
        second,
        position=8.0,
        bracket=(8.0, 9.0),
        boundary_revision=(11, 11),
        checkpoint_evidence=(
            first.checkpoint_evidence[0],
            (12, (anchor + timedelta(seconds=2000)).timestamp(), 11),
        ),
    )

    current = await tracker.update(
        _snapshot(2), [weaker_for_first, exact_for_first]
    )
    by_position = {marker.position: marker for marker in current}

    # The clean candidate retains its owner, while a candidate with positively
    # owned rows from both buses births without a trusted identity ledger.
    assert by_position[6.0].track_id == initial[0].track_id
    assert by_position[8.0].track_id not in {marker.track_id for marker in initial}
    assert tracker._routes[("KMB", "R", "out")][
        by_position[8.0].track_id
    ].cohort_evidence == ()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("old_position", "old_bracket", "new_position", "new_bracket"),
    (
        (5.5, (5.0, 6.0), 15.0, (10.0, 20.0)),
        (10.5, (10.0, 11.0), 1.5, (1.0, 2.0)),
    ),
)
async def test_exact_checkpoint_identity_does_not_authorize_unsafe_motion(
    old_position, old_bracket, new_position, new_bracket
):
    tracker = MarkerTracker()
    shared_arrival = (BASE_TIME + timedelta(minutes=10)).timestamp()
    initial_candidate = replace(
        _candidate(
            old_position,
            bracket=old_bracket,
            boundary_age=0,
            boundary_revision=(10, 10),
        ),
        checkpoint_evidence=((30, shared_arrival, 20),),
    )
    initial = await tracker.update(_snapshot(1), [initial_candidate])
    candidate = replace(
        initial_candidate,
        position=new_position,
        bracket=new_bracket,
        boundary_revision=(11, 11),
    )

    current = await tracker.update(_snapshot(1), [candidate])

    assert current[0].track_id == initial[0].track_id
    assert current[0].position == old_position
    assert current[0].checkpoint_evidence == initial_candidate.checkpoint_evidence


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
async def test_complete_generation_retires_extra_terminal_marker_on_four_to_three_refresh():
    tracker = MarkerTracker()
    route_key = ("KMB", "91", "inbound")
    initial_candidates = [
        _candidate(0.304130535055, route="91", bound="inbound", bracket=(0, 3),
                   boundary_age=0, boundary_revision=(25, 25)),
        _candidate(6.597920081967, route="91", bound="inbound", bracket=(6, 7),
                   boundary_age=0, boundary_revision=(25, 25)),
        _candidate(21.366856617647, route="91", bound="inbound", bracket=(21, 22),
                   boundary_age=0, boundary_revision=(25, 25)),
        _candidate(30.0, route="91", bound="inbound", bracket=(30, 31),
                   boundary_age=0, boundary_revision=(25, 25)),
    ]
    first = await tracker.update(
        _snapshot(25, route_key=route_key), initial_candidates,
    )
    current_candidates = [
        replace(candidate, boundary_revision=(26, 26))
        for candidate in initial_candidates[:3]
    ]

    current = await tracker.update(
        _snapshot(
            26,
            collected_at=BASE_TIME + timedelta(seconds=10),
            route_key=route_key,
        ),
        current_candidates,
    )

    assert len(first) == 4
    assert len(current) == 3
    assert [marker.position for marker in current] == pytest.approx(
        [0.304130535055, 6.597920081967, 21.366856617647]
    )


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
async def test_priority_poll_splits_a_coarse_marker_boundary():
    tracker = MarkerTracker()
    candidates = [_candidate(4.5, bracket=(1.0, 8.0), boundary_age=0)]
    await tracker.update(_snapshot(1), candidates, [_line(stops=10)])

    assert tracker.poll_priorities() == {
        ("KMB", "R", "out"): frozenset({1, 4, 8, 9})
    }


def test_priority_poll_recomputes_binary_split_for_gate_linked_coarse_track():
    track = _Track(1, _candidate(7.5, bracket=(0.0, 12.0)), 7.5, 1)

    assert tracker_module._next_poll_checkpoints(track, 28) == (0, 6, 12)
    track.estimate = _candidate(9.0, bracket=(6.0, 12.0))
    assert tracker_module._next_poll_checkpoints(track, 28) == (6, 9, 12)
    track.estimate = _candidate(7.5, bracket=(6.0, 9.0))
    assert tracker_module._next_poll_checkpoints(track, 28) == (6, 7, 9)


@pytest.mark.parametrize("attempt_order", [(0, 6, 12), (12, 6, 0)])
def test_under_cap_priority_page_stays_full_until_every_group_is_attempted(
    attempt_order,
):
    tracker = MarkerTracker()
    key = ("GMB", "11S", "seq-1")
    track = _Track(
        1,
        _candidate(
            7.5,
            route="11S",
            operator=Operator.GMB,
            bound="seq-1",
            bracket=(0.0, 12.0),
        ),
        7.5,
        1,
    )
    tracker._routes[key] = {1: track}
    tracker._terminal_indices[key] = 28

    first = tracker.poll_priorities()[key]
    assert first == frozenset({0, 6, 12, 28})
    track.estimate = replace(track.estimate, bracket=(6.0, 12.0))

    for generation, checkpoint in enumerate(attempt_order, start=1):
        tracker._ack_probe_attempts(SimpleNamespace(
            probe_attempt_generation=generation,
            attempted_checkpoints=frozenset({(*key, checkpoint)}),
        ))
        current = tracker.poll_priorities()[key]
        if generation < len(attempt_order):
            assert current == first
        else:
            assert current == frozenset({6, 9, 12, 28})


def test_cache_presentation_cannot_replace_unserved_midpoint_page():
    tracker = MarkerTracker()
    key = ("GMB", "11S", "seq-1")
    track = _Track(
        1,
        _candidate(
            7.5,
            route="11S",
            operator=Operator.GMB,
            bound="seq-1",
            bracket=(0.0, 12.0),
        ),
        7.5,
        1,
    )
    tracker._routes[key] = {1: track}
    tracker._terminal_indices[key] = 28

    assert tracker.poll_priorities()[key] == frozenset({0, 6, 12, 28})
    tracker._ack_probe_attempts(SimpleNamespace(
        probe_attempt_generation=1,
        attempted_checkpoints=frozenset({(*key, 0), (*key, 6), (*key, 12)}),
    ))
    track.estimate = replace(track.estimate, bracket=(6.0, 12.0))
    narrowed = tracker.poll_priorities()[key]
    assert narrowed == frozenset({6, 9, 12, 28})

    # The 15-second cache presentation may reconstruct the baseline bracket
    # before the 30-second network sweep.  It cannot cancel midpoint 9.
    track.estimate = replace(track.estimate, bracket=(0.0, 12.0))
    assert tracker.poll_priorities()[key] == narrowed
    tracker._ack_probe_attempts(SimpleNamespace(
        probe_attempt_generation=2,
        attempted_checkpoints=frozenset({(*key, 12)}),
    ))
    assert tracker.poll_priorities()[key] == narrowed


def test_under_cap_page_defers_owed_group_when_terminus_appears():
    tracker = MarkerTracker()
    key = ("KMB", "R", "out")
    tracker._routes[key] = {
        index: _Track(index, _candidate(index, bracket=(index, index)), index, 1)
        for index in range(32)
    }
    first = tracker.poll_priorities()[key]
    assert len(first) == 32
    tracker._ack_probe_attempts(SimpleNamespace(
        probe_attempt_generation=1,
        attempted_checkpoints=frozenset((*key, index) for index in range(31)),
    ))

    tracker._terminal_indices[key] = 59
    deferred = tracker.poll_priorities()[key]
    assert {31, 59} <= deferred
    assert len(deferred) <= 32


def test_latched_page_prunes_checkpoint_beyond_new_route_terminus():
    tracker = MarkerTracker()
    key = ("KMB", "R", "out")
    tracker._routes[key] = {
        index: _Track(index, _candidate(index, bracket=(index, index)), index, 1)
        for index in (5, 8)
    }
    assert tracker.poll_priorities()[key] == frozenset({5, 8})
    tracker._ack_probe_attempts(SimpleNamespace(
        probe_attempt_generation=1,
        attempted_checkpoints=frozenset({(*key, 5)}),
    ))

    tracker._terminal_indices[key] = 6
    current = tracker.poll_priorities()[key]

    assert current == frozenset({5, 6})
    assert 8 not in tracker._priority_queue[key]
    assert tracker._priority_pending[key][1] == {5}


def test_priority_poll_splits_wide_zero_plateau_without_all_due_stops():
    track = _Track(
        1,
        _candidate(
            5.0,
            bracket=(5.0, 8.0),
            priority_indices={3, 4, 5, 8},
        ),
        5.0,
        1,
    )

    assert tracker_module._next_poll_checkpoints(track, 17) == (5, 6, 8)


@pytest.mark.asyncio
async def test_priority_poll_combines_eta_guided_stops_with_safe_bracket_fallback():
    tracker = MarkerTracker()
    candidate = _candidate(
        17.0,
        route="11S",
        operator=Operator.GMB,
        bound="seq-1",
        bracket=(9.0, 18.0),
        boundary_age=0,
        priority_indices={18},
        exploratory_indices={15, 16},
    )
    await tracker.update(
        _snapshot(1, route_key=("GMB", "11S", "seq-1")),
        [candidate],
        [_line(stops=19, route="11S", bound="seq-1", operator="GMB")],
    )

    # ETA-guided stops replace the stale coarse bracket; the route terminus is
    # the one high-value population sentinel retained alongside them.
    assert tracker.poll_priorities() == {
        ("GMB", "11S", "seq-1"): frozenset({15, 16, 18})
    }


@pytest.mark.asyncio
async def test_minimal_guided_plan_keeps_each_marker_boundary():
    tracker = MarkerTracker()
    singleton = _candidate(
        15.0,
        bracket=(0.0, 16.0),
        boundary_age=0,
        priority_indices={16},
        exploratory_indices={13, 14},
    )
    established = _candidate(
        44.2,
        bracket=(44.0, 45.0),
        boundary_age=0,
        priority_indices=set(range(19, 46)),
    )
    await tracker.update(
        _snapshot(1),
        [singleton, established],
        [_line(stops=50)],
    )

    assert tracker.poll_priorities() == {
        ("KMB", "R", "out"): frozenset({13, 14, 16, 44, 45, 49})
    }


@pytest.mark.asyncio
async def test_priority_poll_covers_owned_zero_plateau_and_forward_rung():
    tracker = MarkerTracker()
    candidate = _candidate(
        5.0,
        bracket=(5.0, 6.0),
        boundary_age=0,
        priority_indices={3, 4, 5, 6},
    )
    await tracker.update(_snapshot(1), [candidate], [_line(stops=9)])
    assert tracker.poll_priorities() == {
        ("KMB", "R", "out"): frozenset({5, 6, 8})
    }


@pytest.mark.asyncio
async def test_forward_search_replaces_obsolete_marker_probes():
    tracker = MarkerTracker()
    candidate = _candidate(
        14.257,
        route="104",
        operator=Operator.GMB,
        bound="seq-1",
        bracket=(8.0, 15.0),
        boundary_age=0,
        priority_indices={15, 23},
        exploratory_indices={11, 12},
    )
    key = ("GMB", "104", "seq-1")
    await tracker.update(
        _snapshot(1, route_key=key),
        [candidate],
        [_line(stops=24, route="104", bound="seq-1", operator="GMB")],
    )
    track = next(iter(tracker._routes[key].values()))
    track.motion_bracket = (8.0, 15.0)
    track.forward_after = 15
    track.forward_frontier = (16, 17, 19, 21, 23)

    assert tracker.poll_priorities() == {
        key: frozenset({16, 19, 23})
    }


@pytest.mark.asyncio
async def test_unbracketed_marker_keeps_only_terminus_fallback():
    tracker = MarkerTracker()
    await tracker.update(_snapshot(1), [_candidate(1.0)], [_line(stops=6)])
    assert tracker.poll_priorities() == {
        ("KMB", "R", "out"): frozenset({5})
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


def test_priority_poll_cap_prioritizes_recovery_over_low_index_refinement():
    tracker = MarkerTracker()
    key = ("KMB", "R", "out")
    ordinary = {
        index: _Track(index, _candidate(index + 0.5,
                                        bracket=(index, index + 1)),
                      index + 0.5, 1)
        for index in range(40)
    }
    recovery = _Track(100, _candidate(50.5, bracket=(50, 51)), 50.5, 1,
                      forward_after=50, forward_frontier=(51, 55, 58))
    tracker._routes[key] = {**ordinary, 100: recovery}
    tracker._terminal_indices[key] = 59

    priorities = tracker.poll_priorities()[key]
    assert {51, 55, 58, 59}.issubset(priorities)
    assert len(priorities) == 32


def test_priority_poll_rotates_oversized_recovery_population():
    tracker = MarkerTracker()
    key = ("KMB", "R", "out")
    tracks = {
        index: _Track(index, _candidate(index + 0.5), index + 0.5, 1,
                      forward_after=0,
                      forward_frontier=(1 + index,
                                        20 + index,
                                        40 + index))
        for index in range(12)
    }
    tracker._routes[key] = tracks
    tracker._terminal_indices[key] = 59

    first = tracker.poll_priorities()[key]
    tracker._ack_probe_attempts(SimpleNamespace(
        probe_attempt_generation=1,
        attempted_checkpoints=frozenset((*key, index)
                                         for index in first if index != 59),
    ))
    second = tracker.poll_priorities()[key]
    assert len(first) == len(second) == 32
    assert first != second
    assert 49 in first | second


def test_priority_poll_rotates_oversized_ordinary_population():
    tracker = MarkerTracker()
    key = ("KMB", "R", "out")
    tracks = {
        index: _Track(index, _candidate(index + 0.5,
                                        bracket=(index, index + 1)),
                      index + 0.5, 1)
        for index in range(40)
    }
    tracker._routes[key] = tracks
    tracker._terminal_indices[key] = 59

    first = tracker.poll_priorities()[key]
    tracker._ack_probe_attempts(SimpleNamespace(
        probe_attempt_generation=1,
        attempted_checkpoints=frozenset((*key, index)
                                         for index in first if index != 59),
    ))
    second = tracker.poll_priorities()[key]
    assert len(first) == len(second) == 32
    assert first != second
    assert 30 in first
    assert 40 in first | second


@pytest.mark.asyncio
async def test_priority_queue_trims_page_when_terminus_appears():
    tracker = MarkerTracker()
    key = ("KMB", "R", "out")
    tracker._routes[key] = {
        index: _Track(index, _candidate(index, bracket=(index, index)), index, 1)
        for index in range(33)
    }
    first = tracker.poll_priorities()[key]
    tracker._ack_probe_attempts(SimpleNamespace(
        probe_attempt_generation=1,
        attempted_checkpoints=frozenset((*key, index)
                                         for index in first if index < 31),
    ))
    # Supplying the route line updates the reserved terminus in the live
    # tracker; use the direct state hook here to keep the population intact.
    tracker._terminal_indices[key] = 59
    page = tracker.poll_priorities()[key]
    assert len(page) <= 32
    assert 31 in page


def test_priority_queue_serves_stable_tail_across_alternating_populations():
    tracker = MarkerTracker()
    key = ("KMB", "R", "out")
    tracks = {
        index: _Track(index, _candidate(index, bracket=(index, index)), index, 1)
        for index in range(62)
    }
    tracks[62] = _Track(62, _candidate(69, bracket=(69, 69)), 69, 1)
    tracker._routes[key] = tracks
    tracker._terminal_indices[key] = 100
    observed = set()
    for generation in range(1, 8):
        page = tracker.poll_priorities()[key]
        observed.update(page)
        tracker._ack_probe_attempts(SimpleNamespace(
            probe_attempt_generation=generation,
            attempted_checkpoints=frozenset((*key, index)
                                             for index in page if index != 100),
        ))
        tracks[61].estimate = _candidate(
            62 if generation % 2 else 61,
            bracket=(62, 62) if generation % 2 else (61, 61),
        )
    assert 69 in observed


def test_priority_queue_survives_under_cap_cache_read_between_overflow_pages():
    tracker = MarkerTracker()
    key = ("KMB", "R", "out")
    tracks = {
        index: _Track(index, _candidate(index, bracket=(index, index)), index, 1)
        for index in range(40)
    }
    tracker._routes[key] = tracks
    tracker._terminal_indices[key] = 59
    first = tracker.poll_priorities()[key]
    tracker._ack_probe_attempts(SimpleNamespace(
        probe_attempt_generation=1,
        attempted_checkpoints=frozenset((*key, index)
                                         for index in range(10)),
    ))
    tracker._routes[key] = dict(list(tracks.items())[:20])
    assert tracker.poll_priorities()[key] == first
    tracker._routes[key] = tracks
    assert tracker.poll_priorities()[key] == first
    pending = tracker._priority_pending[key][1]
    tracker._ack_probe_attempts(SimpleNamespace(
        probe_attempt_generation=2,
        attempted_checkpoints=frozenset((*key, index) for index in pending),
    ))
    next_page = tracker.poll_priorities()[key]
    assert next_page != first
    assert 20 in next_page


def test_priority_queue_reconciles_when_population_fits_then_overflows():
    tracker = MarkerTracker()
    key = ("KMB", "R", "out")
    tracker._priority_queue[key] = [17]
    tracker._routes[key] = {
        1: _Track(1, _candidate(1.5, bracket=(1, 2)), 1.5, 1)
    }
    tracker._terminal_indices[key] = 59
    initial = tracker.poll_priorities()[key]
    assert tracker._priority_queue[key] == [1, 2]
    tracker._ack_probe_attempts(SimpleNamespace(
        probe_attempt_generation=1,
        attempted_checkpoints=frozenset(
            (*key, index) for index in initial if index != 59
        ),
    ))

    tracker._routes[key] = {
        index: _Track(index, _candidate(index + 0.5,
                                        bracket=(index, index + 1)),
                      index + 0.5, 1)
        for index in range(40)
    }
    assert 0 in tracker.poll_priorities()[key]


def test_priority_poll_keeps_motion_bridge_for_held_wide_candidate():
    tracker = MarkerTracker()
    key = ("KMB", "R", "out")
    held = _Track(1, _candidate(4.0, bracket=(6.0, 11.0)), 4.0, 1,
                  motion_bracket=(3.0, 4.0))
    tracker._routes[key] = {1: held}
    tracker._terminal_indices[key] = 17

    assert tracker.poll_priorities()[key] == frozenset({4, 6, 8, 11, 17})


def test_priority_poll_keeps_overlapping_motion_bridge_for_wide_candidate():
    track = _Track(1, _candidate(8.0, bracket=(6.0, 11.0)), 8.0, 1,
                   motion_bracket=(3.0, 8.0))
    assert tracker_module._next_poll_checkpoints(track, 17) == (6, 8, 11)


def test_priority_poll_keeps_degenerate_motion_bridge_for_wide_candidate():
    track = _Track(1, _candidate(8.0, bracket=(6.0, 11.0)), 8.0, 1,
                   motion_bracket=(4.0, 4.0))
    assert tracker_module._next_poll_checkpoints(track, 17) == (6, 8, 11, 4)


def test_probe_ack_epoch_is_monotonic_and_indices_are_normalized():
    tracker = MarkerTracker()
    key = ("KMB", "R", "out")
    tracker._priority_pending[key] = ((1, 2), {1, 2})
    tracker._ack_probe_attempts(SimpleNamespace(
        probe_attempt_generation=3,
        attempted_checkpoints=frozenset({(*key, 1.0)}),
    ))
    tracker._ack_probe_attempts(SimpleNamespace(
        probe_attempt_generation=2,
        attempted_checkpoints=frozenset({(*key, 2)}),
    ))
    assert tracker._priority_pending[key][1] == {2}


def test_priority_pending_reconciles_completed_items_across_tail_mutation():
    tracker = MarkerTracker()
    key = ("KMB", "R", "out")
    tracker._routes[key] = {
        index: _Track(index, _candidate(index + 0.5,
                                        bracket=(index, index + 1)),
                      index + 0.5, 1)
        for index in range(40)
    }
    tracker._terminal_indices[key] = 59
    first = tracker.poll_priorities()[key]
    acknowledged = set(range(20))
    tracker._ack_probe_attempts(SimpleNamespace(
        probe_attempt_generation=1,
        attempted_checkpoints=frozenset((*key, index) for index in acknowledged),
    ))

    # An unselected tail change rebuilds the population but must not make the
    # already serviced selected items due again.
    tracker._routes[key][39].estimate = _candidate(39.5, bracket=(39, 41))
    second = tracker.poll_priorities()[key]
    assert second == first
    pending = tracker._priority_pending[key]
    assert pending[1] == set(range(20, 31))

    tracker._ack_probe_attempts(SimpleNamespace(
        probe_attempt_generation=2,
        attempted_checkpoints=frozenset((*key, index)
                                         for index in pending[1]),
    ))
    assert tracker._priority_pending[key][1] == set()
    tracker.poll_priorities()
    assert tracker._priority_queue[key][0] == 31


def test_priority_pending_episode_blocks_volatile_leave_and_reentry():
    tracker = MarkerTracker()
    key = ("KMB", "R", "out")
    volatile = _Track(0, _candidate(0.5, bracket=(0, 1)), 0.5, 1)
    stable = {
        index: _Track(index, _candidate(10 + (index - 1) * 2 + 0.5,
                                        bracket=(10 + (index - 1) * 2,
                                                 11 + (index - 1) * 2)),
                      10 + (index - 1) * 2 + 0.5, 1)
        for index in range(1, 21)
    }
    tracker._routes[key] = {0: volatile, **stable}
    tracker._terminal_indices[key] = 100
    initial = tracker.poll_priorities()[key]
    tracker._ack_probe_attempts(SimpleNamespace(
        probe_attempt_generation=1,
        attempted_checkpoints=frozenset(
            (*key, index)
            for index in sorted(initial - {100, 0, 1})[:3]
        ),
    ))
    before = set(tracker._priority_pending[key][1])

    volatile.estimate = _candidate(2.5, bracket=(2, 3))
    departed = tracker.poll_priorities()[key]
    assert departed == initial
    after_departure = set(tracker._priority_pending[key][1])
    assert after_departure == before

    volatile.estimate = _candidate(0.5, bracket=(0, 1))
    returned = tracker.poll_priorities()[key]
    assert returned == initial
    assert set(tracker._priority_pending[key][1]) == after_departure

    tracker._ack_probe_attempts(SimpleNamespace(
        probe_attempt_generation=2,
        attempted_checkpoints=frozenset((*key, index)
        for index in after_departure),
    ))
    assert tracker._priority_pending[key][1] == set()
    next_page = tracker.poll_priorities()[key]
    assert next_page
    tracker._ack_probe_attempts(SimpleNamespace(
        probe_attempt_generation=3,
        attempted_checkpoints=frozenset((*key, index)
                                         for index in next_page if index != 100),
    ))
    assert tracker.poll_priorities()[key]


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

    held = next(marker for marker in second if marker.track_id == first[1].track_id)
    advanced = next(marker for marker in second if marker.track_id == first[0].track_id)
    assert held.position == pytest.approx(10.0)
    assert held.bracket == first[1].bracket
    assert advanced.position == pytest.approx(11.0)
    assert advanced.bracket == fresh.bracket
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
