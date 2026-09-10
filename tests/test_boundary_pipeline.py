"""End-to-end boundary revision checks across estimates and marker tracking."""

from collections import Counter
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from dashboard.maps.positions import (
    BusEstimate,
    _path_segment_length,
    estimate_bus_positions,
    rebuild_estimate_from_probe_sources,
)
from dashboard.maps.tracker import MarkerTracker, _reconcile_complete_probe_ownership
from dashboard.models import EtaKind, Operator
from dashboard.providers.route_geometry import RouteLine, Stop
from dashboard.providers.transit import ProbeEtaSnapshot, ProbeRouteGeneration

KEY = ("KMB", "X", "outbound")
BASE = datetime(2026, 1, 1, tzinfo=UTC)


def _complete_ownership_fixture(groups, when, revision, *, reverse=False, kind=EtaKind.REALTIME):
    """Full raw snapshots, including empty checkpoints omitted by audit logs."""
    route_line = line(stop_count=30)
    rows = []
    owned_slots = []
    for group in groups:
        slots = []
        for index, seconds in group:
            arrival = BASE + timedelta(seconds=seconds)
            signed = (arrival - when).total_seconds() / 60
            value = Probe(index, max(0, signed), 0, revision, arrival_at=arrival)
            value.signed_minutes = signed
            value.kind = kind
            slots.append(len(rows))
            rows.append(value)
        owned_slots.append(slots)
    present = {row.index for row in rows}
    rows.extend(Probe(index, None, 0, revision)
                for index in (0, 12, 23) if index not in present)
    if reverse:
        owned_slots = [[len(rows) - 1 - slot for slot in slots] for slots in owned_slots]
        rows.reverse()
    template = BusEstimate("X terminus", 0, 0, Operator.KMB, 0,
                           route="X", bound="outbound", operator_code="KMB")
    candidates = [rebuild_estimate_from_probe_sources(template, slots, rows, [route_line])
                  for slots in owned_slots]
    assert all(candidate is not None for candidate in candidates)
    return rows, candidates, route_line


@pytest.mark.asyncio
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("case", ["frame15", "frame22"])
@pytest.mark.parametrize("partial", [False, True])
async def test_complete_snapshot_repartitions_already_paired_rows(case, reverse, partial):
    # Synthetic relative times preserve the exact occurrence/delta shapes of
    # the two failures without requiring private captures or a live provider.
    if case == "frame15":
        historical = [
            ((23, 6408), (24, 6591), (25, 6634), (26, 6765), (27, 7059), (28, 7258)),
            ((24, 6461), (25, 6504), (26, 6635), (27, 6929), (28, 7129)),
            ((26, 6390), (27, 6683), (28, 6883)),
        ]
        mixed = [
            ((24, 6591), (25, 6634), (26, 6765), (27, 6968), (28, 7168)),
            ((24, 6500), (25, 6543), (26, 6674), (27, 6711), (28, 6910)),
            ((27, 7059), (28, 7258)),
        ]
        expected = [mixed[0][:3] + mixed[2], mixed[1][:3] + mixed[0][3:], mixed[1][3:]]
        when = BASE + timedelta(seconds=6540)
    else:
        historical = [
            ((13, 6657), (14, 6732), (23, 7636), (24, 7788), (25, 7830), (26, 7961)),
            ((13, 6627), (14, 6702), (23, 7606)),
        ]
        mixed = [historical[0][:2], historical[1][:2] + historical[0][3:]]
        expected = [historical[0][:2] + historical[0][3:], historical[1][:2]]
        # Three other physical buses, including a nonmonotonic prior ladder,
        # remain independent components of the same complete route snapshot.
        others = [
            ((24, 6546), (25, 6588), (26, 6719), (27, 6986), (28, 7186)),
            ((24, 6519), (25, 6561), (26, 6692), (27, 6665), (28, 6865)),
            ((27, 7013), (28, 7213)),
        ]
        historical.extend(others)
        mixed.extend(others)
        expected.extend(others)
        # Actual frame22 due crossing: the earlier bus's stop13 signed ETA is
        # -0.12477 minutes; MOVING_SLOWLY is live operator evidence.
        when = BASE + timedelta(seconds=6604.4862)
    kind = EtaKind.MOVING_SLOWLY if case == "frame22" else EtaKind.REALTIME
    rows, initial, route_line = _complete_ownership_fixture(historical, when, 10, kind=kind)
    tracker = MarkerTracker()
    previous = await tracker.update(snapshot(1, when, rows), initial, [route_line])
    owner_ids = [next(marker.track_id for marker in previous
                      if marker.checkpoint_evidence == candidate.checkpoint_evidence)
                 for candidate in initial]
    current_time = when + timedelta(seconds=30)
    rows, candidates, _ = _complete_ownership_fixture(
        mixed, current_time, 11, reverse=reverse, kind=kind,
    )
    if reverse:
        candidates.reverse()
        tracker._routes[KEY] = dict(reversed(list(tracker._routes[KEY].items())))
    original_sources = Counter(source for candidate in candidates
                               for source in candidate.source_observations)
    if partial:
        prior_cohorts = {track.track_id: track.cohort_evidence
                         for track in tracker._routes[KEY].values()}
        await tracker.update(snapshot(1, current_time, rows), candidates, [route_line])
        assert {track.track_id: track.cohort_evidence
                for track in tracker._routes[KEY].values()} == prior_cohorts
    result = await tracker.update(snapshot(2, current_time, rows), candidates, [route_line])
    assert len(result) == len(candidates) == len(previous)
    trusted_ids = set(owner_ids) - ({owner_ids[3]} if case == "frame22" else set())
    assert trusted_ids <= {marker.track_id for marker in result}
    assert Counter(source for marker in result for source in marker.source_observations) \
        == original_sources
    for number, (owner_id, group) in enumerate(zip(owner_ids, expected, strict=True)):
        evidence = tuple((stop, (BASE + timedelta(seconds=seconds)).timestamp(), 11)
                         for stop, seconds in group)
        marker = next(marker for marker in result if marker.checkpoint_evidence == evidence)
        if case == "frame22" and number == 3:
            # The unrelated backwards ladder cannot seed trusted history.
            assert tracker._routes[KEY][marker.track_id].cohort_evidence == ()
            continue
        assert marker.track_id == owner_id
        assert marker.checkpoint_evidence == evidence
        assert marker.source_indices == {stop for stop, _seconds in group}
        actual_rows = [rows[slot] for kind, slot in marker.source_observations if kind == "probe"]
        assert Counter((row.index, row.arrival_at.timestamp(), row.refresh_generation)
                       for row in actual_rows) == Counter(evidence)
        assert tracker._routes[KEY][owner_id].cohort_evidence == evidence
        assert marker.boundary_revision == (11, 11)
        assert marker.exploratory_indices == frozenset()
        if case == "frame22" and number < 2:
            position = 12.812385 if number == 0 else 13.099816
            assert marker.position == pytest.approx(position)
            assert marker.bracket == ((12.0, 13.0) if number == 0 else (13.0, 14.0))
            assert marker.lat == pytest.approx(22.333360)
            assert marker.lon == pytest.approx(114.260 + position * 0.001)
            assert marker.eta_arrival_at == BASE + timedelta(seconds=6657 if number == 0 else 6702)
            assert marker.priority_indices == frozenset({13, 14})
            if number == 1:
                due = next(row for row in actual_rows if row.index == 13)
                assert due.signed_minutes == pytest.approx(-0.12477)
                assert due.kind is EtaKind.MOVING_SLOWLY


@pytest.mark.asyncio
@pytest.mark.parametrize("exclusion", [
    "expired", "malformed_ledger", "gate", "scheduled", "unreliable",
    "malformed_slot", "missing_geometry", "missing_boundary", "old_revision",
    "regressed_revision", "boolean_minutes", "boolean_age", "boolean_signed", "unknown_kind",
])
async def test_complete_repartition_exclusions_are_atomic(exclusion):
    historical = [((13, 6657), (14, 6732), (24, 7788), (25, 7830)),
                  ((13, 6627), (14, 6702))]
    mixed = [historical[0][:2], historical[1] + historical[0][2:]]
    when = BASE + timedelta(seconds=6580)
    rows, initial, route_line = _complete_ownership_fixture(historical, when, 10)
    tracker = MarkerTracker()
    await tracker.update(snapshot(1, when, rows), initial, [route_line])
    current_time = when + timedelta(seconds=30)
    rows, candidates, _ = _complete_ownership_fixture(mixed, current_time, 11)
    tracks = list(tracker._routes[KEY].values())
    lines = [route_line]
    if exclusion == "expired":
        current_time += timedelta(seconds=121)
    elif exclusion == "malformed_ledger":
        tracks[0].cohort_evidence += (("bad", 1.0, 10),)
    elif exclusion == "gate":
        candidates[1] = replace(candidates[1], source_observations=(
            candidates[1].source_observations | {("gate", 0)}))
    elif exclusion == "scheduled":
        rows[2].kind = EtaKind.SCHEDULED
    elif exclusion == "unreliable":
        candidates[1] = replace(candidates[1], unreliable=True)
    elif exclusion == "malformed_slot":
        candidates[1] = replace(candidates[1], source_observations=(
            candidates[1].source_observations | {("probe", "invalid")}))
    elif exclusion == "missing_geometry":
        lines = []
    elif exclusion == "missing_boundary":
        rows = rows[:6]  # no successful-empty upstream responses
    elif exclusion == "old_revision":
        for row in rows:
            if row.minutes is not None:
                row.arrival_at += timedelta(seconds=5)
                row.refresh_generation = 10
        candidates = [replace(candidate, checkpoint_evidence=tuple(
            (stop, arrival + 5, 10) for stop, arrival, _revision in candidate.checkpoint_evidence
        )) for candidate in candidates]
    elif exclusion == "regressed_revision":
        for row in rows:
            row.refresh_generation = 9
        candidates = [replace(candidate, checkpoint_evidence=tuple(
            (stop, arrival, 9) for stop, arrival, _revision in candidate.checkpoint_evidence
        )) for candidate in candidates]
    elif exclusion == "boolean_minutes":
        rows[2].minutes = True
    elif exclusion == "boolean_age":
        rows[2].cache_age_seconds = False
    elif exclusion == "boolean_signed":
        rows[2].signed_minutes = True
    elif exclusion == "unknown_kind":
        rows[2].kind = "not-a-live-kind"
    ledger_before = [(track.cohort_evidence, track.cohort_observed_at) for track in tracks]
    result, owners = _reconcile_complete_probe_ownership(
        tracks, candidates, current_time.timestamp(), rows, lines,
    )
    assert not owners
    assert set(map(id, result)) == set(map(id, candidates))
    assert [(track.cohort_evidence, track.cohort_observed_at) for track in tracks] == ledger_before


@pytest.mark.asyncio
@pytest.mark.parametrize("ambiguity", ["alternate_owner", "hall_deficient", "equal_arrival"])
async def test_complete_repartition_requires_unique_injective_owners(ambiguity):
    when = BASE + timedelta(seconds=6500)
    historical = [((13, 6600), (14, 6720)), ((13, 6630), (14, 6750))]
    mixed = [((13, 6610), (14, 6740)), ((13, 6620), (14, 6730))]
    if ambiguity == "hall_deficient":
        mixed[0] += ((13, 6625),)
    elif ambiguity == "equal_arrival":
        historical = [historical[0], historical[0]]
        mixed = [historical[0], historical[0]]
    rows, initial, route_line = _complete_ownership_fixture(historical, when, 10)
    tracker = MarkerTracker()
    await tracker.update(snapshot(1, when, rows), initial, [route_line])
    current_time = when + timedelta(seconds=30)
    rows, candidates, _ = _complete_ownership_fixture(mixed, current_time, 11)
    result, owners = _reconcile_complete_probe_ownership(
        list(tracker._routes[KEY].values()), candidates, current_time.timestamp(), rows, [route_line],
    )
    assert not owners
    assert len(result) == len(candidates) == 2
    assert set(map(id, result)) == set(map(id, candidates))
    assert Counter(source for candidate in result for source in candidate.source_observations) \
        == Counter(source for candidate in candidates for source in candidate.source_observations)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["missing_geometry", "hall_deficient"])
async def test_failed_positive_mixed_repartition_cannot_seed_trusted_history(failure):
    historical = [((13, 6600), (14, 6720), (24, 7700), (25, 7820)),
                  ((13, 6640), (14, 6760))]
    mixed = [historical[0][:2], historical[1] + historical[0][2:]]
    if failure == "hall_deficient":
        mixed[1] += ((24, 7710),)
    when = BASE + timedelta(seconds=6500)
    rows, initial, route_line = _complete_ownership_fixture(historical, when, 10)
    tracker = MarkerTracker()
    previous = await tracker.update(snapshot(1, when, rows), initial, [route_line])
    old_ids = {marker.track_id for marker in previous}
    current_time = when + timedelta(seconds=30)
    rows, candidates, _ = _complete_ownership_fixture(mixed, current_time, 11)
    mixed_sources = candidates[1].source_observations
    result = await tracker.update(snapshot(2, current_time, rows), candidates,
                                  [] if failure == "missing_geometry" else [route_line])
    assert len(result) == len(candidates) == 2
    assert Counter(source for marker in result for source in marker.source_observations) \
        == Counter(source for candidate in candidates for source in candidate.source_observations)
    mixed_birth = next(marker for marker in result if marker.source_observations == mixed_sources)
    assert mixed_birth.track_id not in old_ids
    assert mixed_birth.bracket == candidates[1].bracket
    assert mixed_birth.boundary_revision == candidates[1].boundary_revision
    assert mixed_birth.checkpoint_evidence == candidates[1].checkpoint_evidence
    assert tracker._routes[KEY][mixed_birth.track_id].cohort_evidence == ()
    for track in tracker._routes[KEY].values():
        assert track.cohort_evidence != candidates[1].checkpoint_evidence

    # The untrusted display population cannot claim both later clean buses.
    final_time = current_time + timedelta(seconds=30)
    rows, clean, _ = _complete_ownership_fixture(historical, final_time, 12)
    recovered = await tracker.update(snapshot(3, final_time, rows), clean, [route_line])
    assert len(recovered) == 2
    assert {marker.checkpoint_evidence for marker in recovered} == {
        candidate.checkpoint_evidence for candidate in clean
    }
    assert {track.cohort_evidence for track in tracker._routes[KEY].values()} == {
        candidate.checkpoint_evidence for candidate in clean
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("corroborated", [False, True])
@pytest.mark.parametrize("partial", [False, True])
async def test_untrusted_display_cannot_steal_clean_owner_in_third_generation(corroborated, partial):
    historical = [((13, 6600), (14, 6720)), ((24, 7800), (25, 7920))]
    clean = ((13, 6630), (14, 6750))
    when = BASE + timedelta(seconds=6500)
    rows, candidates, route_line = _complete_ownership_fixture(historical, when, 10)
    tracker = MarkerTracker()
    initial = await tracker.update(snapshot(1, when, rows), candidates, [route_line])
    a_id = next(marker.track_id for marker in initial
                if marker.checkpoint_evidence == candidates[0].checkpoint_evidence)
    b_id = next(marker.track_id for marker in initial
                if marker.checkpoint_evidence == candidates[1].checkpoint_evidence)

    when += timedelta(seconds=30)
    rows, candidates, _ = _complete_ownership_fixture(
        [historical[0], clean + historical[1]], when, 11,
    )
    second = await tracker.update(snapshot(2, when, rows), candidates, [route_line])
    assert len(second) == 2
    assert a_id in {marker.track_id for marker in second}
    assert b_id not in {marker.track_id for marker in second}
    untrusted = next(marker for marker in second if marker.track_id != a_id)
    assert untrusted.checkpoint_evidence == candidates[1].checkpoint_evidence
    assert tracker._routes[KEY][untrusted.track_id].cohort_evidence == ()
    assert not tracker._routes[KEY][untrusted.track_id].cohort_trusted
    assert tracker._routes[KEY][a_id].position == pytest.approx(12.4166666667)
    assert untrusted.position == pytest.approx(12.1666666667)

    when += timedelta(seconds=30)
    rows, candidates, _ = _complete_ownership_fixture(
        [clean if corroborated else clean[:1]], when, 12,
    )
    if partial:
        prior_cohort = tracker._routes[KEY][a_id].cohort_evidence
        original_untrusted_evidence = untrusted.checkpoint_evidence
        # The first partial uses temporal continuity for A; the second repeats
        # its now-exact rows under a newer revision. Neither fixed/recovery nor
        # ordered fallback may spend U's displayed exact ownership.
        for revision in (12, 13):
            if revision == 13:
                when += timedelta(seconds=10)
                rows, candidates, _ = _complete_ownership_fixture(
                    [clean if corroborated else clean[:1]], when, revision,
                )
            refreshed = await tracker.update(snapshot(2, when, rows), candidates, [route_line])
            assert {marker.track_id for marker in refreshed} == {a_id, untrusted.track_id}
            a = next(marker for marker in refreshed if marker.track_id == a_id)
            held = next(marker for marker in refreshed if marker.track_id == untrusted.track_id)
            assert a.position == pytest.approx(candidates[0].position)
            assert a.checkpoint_evidence == candidates[0].checkpoint_evidence
            assert a.boundary_revision == (revision, revision)
            assert held.position == pytest.approx(untrusted.position)
            assert held.position < a.position
            assert held.lat == untrusted.lat
            assert held.lon == untrusted.lon
            assert held.checkpoint_evidence == original_untrusted_evidence
            assert held.boundary_revision == untrusted.boundary_revision
            assert tracker._routes[KEY][a_id].cohort_evidence == prior_cohort
            assert not tracker._routes[KEY][untrusted.track_id].cohort_trusted
            assert tracker._routes[KEY][untrusted.track_id].cohort_evidence == ()
        when += timedelta(seconds=10)
        rows, candidates, _ = _complete_ownership_fixture(
            [clean if corroborated else clean[:1]], when, 14,
        )
    unchanged, certificates = _reconcile_complete_probe_ownership(
        list(tracker._routes[KEY].values()), candidates, when.timestamp(), rows, [route_line],
    )
    assert unchanged[0] is candidates[0]
    # No repartition is needed, but its injective ownership still constrains
    # all later complete pairing paths. A singleton exercises the uncertified
    # fallback separately: U's displayed exact rows must have no weight there.
    assert certificates == ({id(candidates[0]): a_id} if corroborated else {})
    third = await tracker.update(snapshot(3, when, rows), candidates, [route_line])
    assert len(third) == 1
    assert third[0].track_id == a_id
    assert third[0].checkpoint_evidence == candidates[0].checkpoint_evidence
    assert Counter(third[0].source_observations) == Counter(candidates[0].source_observations)
    assert set(tracker._routes[KEY]) == {a_id}
    assert tracker._routes[KEY][a_id].cohort_trusted
    assert tracker._routes[KEY][a_id].cohort_evidence == candidates[0].checkpoint_evidence


@pytest.mark.asyncio
async def test_complete_sequence_does_not_publish_backwards_cohorts_or_churn_clean_frame29_ids():
    """Follow repaired ownership through ambiguous ladders and due crossings."""
    groups14 = [
        ((13, 6598), (14, 6673), (23, 7578), (24, 7697), (25, 7739)),
        ((13, 6534), (14, 6609), (23, 7513)),
        ((23, 6408), (24, 6591), (25, 6634), (26, 6765), (27, 7059), (28, 7258)),
        ((24, 6461), (25, 6504), (26, 6635), (27, 6929), (28, 7129)),
        ((26, 6390), (27, 6683), (28, 6883)),
    ]
    groups15 = [
        ((13, 6624), (14, 6700), (23, 7604), (24, 7757), (25, 7799), (26, 7931)),
        ((13, 6594), (14, 6669), (23, 7573)),
        ((24, 6591), (25, 6634), (26, 6765), (27, 6968), (28, 7168)),
        ((24, 6500), (25, 6543), (26, 6674), (27, 6711), (28, 6910)),
        ((27, 7059), (28, 7258)),
    ]
    groups19 = [
        ((13, 6657), (14, 6732), (23, 7636), (24, 7788), (25, 7830), (26, 7961)),
        ((13, 6627), (14, 6702), (23, 7606)),
        ((24, 6546), (25, 6588), (26, 6719), (27, 6986), (28, 7186)),
        ((24, 6519), (25, 6561), (26, 6692), (27, 6665), (28, 6865)),
        ((27, 7013), (28, 7213)),
    ]
    groups22 = [groups19[0][:2], groups19[1][:2] + groups19[0][3:], *groups19[2:]]
    groups29 = [
        ((13, 6716), (14, 6791), (24, 7876)),
        ((13, 6679), (14, 6754), (24, 7839), (25, 7883), (26, 8013)),
        ((25, 6596), (26, 6726), (27, 6982), (28, 7182)),
        ((24, 6574), (25, 6567), (26, 6688), (27, 6685), (28, 6885)),
        ((27, 7020), (28, 7220)),
    ]
    tracker = MarkerTracker()
    generations = [
        (14, 1, 132, 6534.4336, groups14),
        (15, 2, 174, 6564.47304, groups15),
        (19, 3, 230, 6604.335151, groups19),
        (22, 4, 270, 6634.486234, groups22),
        (26, 5, 321, 6674.421286, groups29),
        (29, 6, 365, 6704.441705, groups29),
    ]
    markers_by_frame = {}
    for frame, generation, revision, seconds, groups in generations:
        when = BASE + timedelta(seconds=seconds)
        rows, candidates, route_line = _complete_ownership_fixture(
            groups, when, revision, kind=EtaKind.MOVING_SLOWLY,
        )
        if frame != 14:
            # The complete cache is preceded by a same-generation presentation
            # of the refreshed source slots, as in the production poll cadence.
            prior = {track.track_id: track.cohort_evidence
                     for track in tracker._routes[KEY].values()}
            await tracker.update(snapshot(generation - 1, when, rows), candidates, [route_line])
            assert {track.track_id: track.cohort_evidence
                    for track in tracker._routes[KEY].values()} == prior
        result = await tracker.update(snapshot(generation, when, rows), candidates, [route_line])
        markers_by_frame[frame] = result
        assert len(result) == len(candidates) == 5
        assert Counter(source for marker in result for source in marker.source_observations) \
            == Counter(source for candidate in candidates for source in candidate.source_observations)
        for track in tracker._routes[KEY].values():
            evidence = sorted(track.cohort_evidence)
            assert all(right[1] >= left[1] for left, right in zip(
                evidence, evidence[1:], strict=False,
            )), (
                frame, track.track_id, evidence,
            )
        if frame == 15:
            # The exact frame15 component is repaired before any ambiguity.
            expected = groups15[2][:3] + groups15[4]
            marker = next(marker for marker in result if tuple(
                (stop, round(arrival - BASE.timestamp()))
                for stop, arrival, _revision in marker.checkpoint_evidence
            ) == expected)
            assert marker.track_id in {marker.track_id for marker in markers_by_frame[14]}
    # The backwards ladder's final due stop changes at frame29. That is not a
    # reason to retire either independently continued clean downstream bus.
    for group in (groups29[2], groups29[4]):
        signature = tuple((stop, (BASE + timedelta(seconds=seconds)).timestamp())
                          for stop, seconds in group)
        previous = next(marker for marker in markers_by_frame[26] if tuple(
            (stop, arrival) for stop, arrival, _revision in marker.checkpoint_evidence
        ) == signature)
        current = next(marker for marker in markers_by_frame[29] if tuple(
            (stop, arrival) for stop, arrival, _revision in marker.checkpoint_evidence
        ) == signature)
        assert current.track_id == previous.track_id


class Probe:
    def __init__(self, index, minutes, age, revision, *, arrival_at=None,
                 route="X"):
        self.operator = "KMB"
        self.route = route
        self.bound = "outbound"
        self.index = index
        self.minutes = minutes
        self.signed_minutes = None
        self.arrival_at = arrival_at
        self.cache_age_seconds = age
        self.refresh_generation = revision
        self.kind = EtaKind.REALTIME
        self.stop_id = f"S{index}"


def line(route="X", stop_count=7):
    stops = tuple(Stop(str(i), f"Stop {i}", 22.333360, 114.260 + i * 0.001)
                  for i in range(stop_count))
    path = [(stop.lat, stop.lon) for stop in stops]
    offsets = [0.0]
    for first, second in zip(stops, stops[1:], strict=False):
        offsets.append(offsets[-1] + _path_segment_length(
            (first.lat, first.lon), (second.lat, second.lon)))
    return RouteLine(route, "KMB", "outbound", stops, path, offsets)


def estimates(rows):
    return estimate_bus_positions(
        rows, [line()],
        observed_checkpoint_indices={KEY: range(7)},
    )


def snapshot(generation, when, rows=(), key=KEY):
    route = ProbeRouteGeneration(key, tuple(rows), generation, when)
    return ProbeEtaSnapshot((route,), when)


@pytest.mark.asyncio
async def test_provisional_11s_gate_handoff_never_creates_second_track():
    key = ("GMB", "11S", "seq-1")
    stops = [
        Stop(str(index), f"Stop {index}", 22.333360, 114.260 + index * 0.001)
        for index in range(19)
    ]
    stops[7] = Stop("20013011", "HKUST South", 22.333360, 114.267)
    route_line = RouteLine(
        "11S",
        "GMB",
        "seq-1",
        stops,
        [(stop.lat, stop.lon) for stop in stops],
        [float(index * 100) for index in range(19)],
    )

    def row(index, minutes, age, revision, arrival=None):
        value = Probe(index, minutes, age, revision, arrival_at=arrival, route="11S")
        value.operator = "GMB"
        value.bound = "seq-1"
        value.stop_id = "20013011" if index == 7 else str(index)
        value.signed_minutes = minutes
        return value

    def current_snapshot(generation, when, baseline_rows, positioning_rows):
        route = ProbeRouteGeneration(
            key,
            tuple(baseline_rows),
            generation,
            when,
            frozenset({0, 7, 9, 18}),
        )
        checkpoints = frozenset(
            (*key, int(value.index)) for value in positioning_rows
        )
        return ProbeEtaSnapshot(
            (route,), when, tuple(positioning_rows), checkpoints
        )

    first_rows = [
        row(0, None, 7.4, 2504),
        row(
            7,
            10.33589975,
            7.688,
            2500,
            datetime.fromisoformat("2026-09-09T03:32:07.886+08:00"),
        ),
        row(
            9,
            14.7188443,
            8.063,
            2498,
            datetime.fromisoformat("2026-09-09T03:36:30.402+08:00"),
        ),
        row(
            18,
            24.322681583333335,
            7.86,
            2499,
            datetime.fromisoformat("2026-09-09T03:46:06.887+08:00"),
        ),
    ]
    first = estimate_bus_positions(
        first_rows,
        [route_line],
        observed_checkpoint_indices={key: {0, 7, 9, 18}},
        verified_gate_indices={key: 7},
    )
    assert len(first) == 1

    tracker = MarkerTracker()
    first_output = await tracker.update(
        current_snapshot(1, BASE, first_rows[1:], first_rows),
        first,
        [route_line],
    )
    assert len(first_output) == 1
    track_id = first_output[0].track_id

    handoff_rows = [first_rows[0], first_rows[2], first_rows[3]]
    gate = row(7, 10, 0.0, 0)
    handoff = estimate_bus_positions(
        handoff_rows,
        [route_line],
        authoritative_etas=[gate],
        observed_checkpoint_indices={key: {0, 7, 9, 18}},
        verified_gate_indices={key: 7},
    )
    assert len(handoff) == 1
    handoff_output = await tracker.update(
        current_snapshot(
            1,
            BASE + timedelta(seconds=10),
            first_rows[1:],
            handoff_rows,
        ),
        handoff,
        [route_line],
    )
    assert len(handoff_output) == 1
    assert handoff_output[0].track_id == track_id

    refreshed_rows = [
        row(0, None, 8.4, 2532),
        row(
            3,
            4.400660966666666,
            8.078,
            2533,
            datetime.fromisoformat("2026-09-09T03:26:41.251+08:00"),
        ),
        row(
            4,
            5.204058683333333,
            7.953,
            2534,
            datetime.fromisoformat("2026-09-09T03:27:29.838+08:00"),
        ),
        row(
            9,
            14.1683765,
            7.766,
            2535,
            datetime.fromisoformat("2026-09-09T03:36:27.821+08:00"),
        ),
        row(
            13,
            17.57462513333333,
            8.516,
            2531,
            datetime.fromisoformat("2026-09-09T03:39:51.228+08:00"),
        ),
        row(
            14,
            18.20001285,
            8.922,
            2530,
            datetime.fromisoformat("2026-09-09T03:40:28.470+08:00"),
        ),
        row(
            18,
            24.925031216666664,
            7.531,
            2536,
            datetime.fromisoformat("2026-09-09T03:47:13.402+08:00"),
        ),
    ]
    refreshed = estimate_bus_positions(
        refreshed_rows,
        [route_line],
        authoritative_etas=[gate],
        observed_checkpoint_indices={key: {0, 3, 4, 7, 9, 13, 14, 18}},
        verified_gate_indices={key: 7},
    )
    assert len(refreshed) == 1
    refreshed_output = await tracker.update(
        current_snapshot(
            2,
            BASE + timedelta(seconds=20),
            [refreshed_rows[3], refreshed_rows[-1]],
            refreshed_rows,
        ),
        refreshed,
        [route_line],
    )
    assert len(refreshed_output) == 1
    assert refreshed_output[0].track_id == track_id


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


@pytest.mark.asyncio
async def test_91m_complete_generations_preserve_terminal_physical_evidence_owner():
    route_key = ("KMB", "91M", "outbound")
    route_line = line("91M", 29)
    times = [
        "2026-09-08T19:04:52+08:00",
        "2026-09-08T19:07:11+08:00",
        "2026-09-08T19:09:07+08:00",
        "2026-09-08T19:07:28+08:00",
        "2026-09-08T19:25:50+08:00",
        "2026-09-08T19:08:46+08:00",
        "2026-09-08T19:27:08+08:00",
        "2026-09-08T19:15:53+08:00",
        "2026-09-08T19:34:15+08:00",
        "2026-09-08T19:06:22+08:00",
        "2026-09-08T19:18:20+08:00",
        "2026-09-08T19:36:42+08:00",
        "2026-09-08T19:16:43+08:00",
        "2026-09-08T19:26:34+08:00",
        "2026-09-08T19:38:33+08:00",
    ]
    values = [
        (3, 0.0), (4, 0.4341865833), (5, 2.3675199167),
        (13, 0.7175199167), (13, 19.0841865833),
        (14, 2.0175199167), (14, 20.3841865833),
        (20, 9.1341865833), (20, 27.50085325),
        (21, 0.0), (21, 11.5841865833), (21, 29.95085325),
        (28, 9.9675199167), (28, 19.8175199167),
    ]

    def rows(terminal_minutes, revision_base):
        entries = values + [(28, terminal_minutes)]
        return [
            Probe(
                index, minutes, 0.0, revision_base + input_index,
                arrival_at=datetime.fromisoformat(times[input_index]),
                route="91M",
            )
            for input_index, (index, minutes) in enumerate(entries)
        ]

    frame24 = rows(32.3011385167, 256)
    frame24[-1].refresh_generation = 270
    frame27 = rows(31.80085325, 299)
    frame27[-1].refresh_generation = 313
    first = estimate_bus_positions(
        frame24, [route_line],
        observed_checkpoint_indices={route_key: range(29)},
        authoritative_etas=[
            Probe(12, 15, 0.0, 270, route="91M")
        ],
    )
    second = estimate_bus_positions(
        frame27, [route_line],
        observed_checkpoint_indices={route_key: range(29)},
        authoritative_etas=[
            Probe(12, 15, 0.0, 313, route="91M")
        ],
    )
    assert len(first) == len(second) == 3

    tracker = MarkerTracker()
    first_output = await tracker.update(
        snapshot(24, BASE, frame24, route_key), first, [route_line]
    )
    second_output = await tracker.update(
        snapshot(27, BASE.replace(day=8), frame27, route_key), second, [route_line]
    )
    first_ids = {item.track_id for item in first_output}
    assert {item.track_id for item in second_output} == first_ids
    terminal_evidence = (28, datetime.fromisoformat(times[-1]).timestamp(), 313)
    owners = [
        item.track_id
        for item in second_output
        if terminal_evidence in item.checkpoint_evidence
    ]
    assert len(owners) == 1
    prior_terminal_owners = [
        item.track_id
        for item in first_output
        if (28, datetime.fromisoformat(times[-1]).timestamp(), 270)
        in item.checkpoint_evidence
    ]
    assert owners == prior_terminal_owners


@pytest.mark.asyncio
async def test_partial_positioning_candidate_does_not_create_marker_until_complete_generation():
    """MarkerTracker consumes atomic generations, not the live positioning view."""
    route_line = line()
    tracker = MarkerTracker()

    empty = await tracker.update(snapshot(1, BASE), [], [route_line])
    assert empty == []

    # A departed/advanced candidate is useful for positioning, but it belongs
    # to the old empty generation and must not bootstrap a marker by itself.
    partial_probe = Probe(0, None, 0.0, 10)
    partial = estimates([partial_probe, Probe(1, 1, 0.0, 11)])
    held = await tracker.update(snapshot(1, BASE.replace(second=10)), partial,
                                [route_line])
    assert held == []

    # Once the provider publishes a newer complete generation containing the
    # departed terminal evidence, exactly one marker is admitted.
    complete_rows = [Probe(0, None, 0.0, 12), Probe(1, 1, 0.0, 13)]
    complete = estimates(complete_rows)
    output = await tracker.update(
        snapshot(2, BASE.replace(second=20), complete_rows), complete,
        [route_line],
    )
    assert len(output) == 1
