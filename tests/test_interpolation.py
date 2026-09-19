"""Measured adjacent-stop travel times determine marker fractions."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from dashboard.maps.interpolation import refine_adjacent_positions
from dashboard.maps.positions import BusEstimate, reproject_estimate
from dashboard.models import Operator
from dashboard.providers.route_geometry import RouteLine, Stop
from dashboard.providers.transit import ProbeEta

NOW = datetime(2026, 9, 20, 0, 0, tzinfo=UTC)


def row(index, minutes, *, operator="GMB", rounded=False, age=0):
    return ProbeEta(
        operator, "X", "outbound", str(index), index, max(0, minutes),
        arrival_at=None if rounded else NOW + timedelta(minutes=minutes),
        observed_at=NOW, signed_minutes=None if rounded else minutes,
        countdown_rounded=rounded, refresh_generation=index + 1, cache_age_seconds=age,
    )


def marker(indices, *, position=3.5, operator="GMB", scheduled=False):
    return BusEstimate(
        "X destination", 22.3, 114.2, Operator.KMB if operator == "KMB" else Operator.GMB, 0,
        route="X", bound="outbound", operator_code=operator, position=position,
        source_observations=frozenset(("probe", index) for index in indices),
        unreliable=scheduled,
    )


def refine(markers, rows):
    operator = markers[0].operator_code
    stops = [Stop(str(i), str(i), 22.3, 114.2 + i / 1000) for i in range(8)]
    line = RouteLine("X", operator, "outbound", stops,
                     [(stop.lat, stop.lon) for stop in stops], list(range(0, 800, 100)))
    return refine_adjacent_positions(markers, rows, [line], reproject=reproject_estimate)


@pytest.mark.parametrize("operator", ["KMB", "CTB", "GMB"])
def test_shifted_lists_keep_two_markers_and_use_each_countdown(operator):
    rows = [row(3, minutes, operator=operator) for minutes in (8, 18)]
    rows += [row(4, minutes, operator=operator) for minutes in (1, 3, 12, 22)]
    markers = [marker([2], operator=operator), marker([3], operator=operator, scheduled=True)]
    shown = refine(markers, rows)
    assert len(shown) == 2
    assert [item.position for item in shown] == pytest.approx([3.75, 3.25])
    assert [item.lon for item in shown] == pytest.approx([114.20375, 114.20325])
    assert all(item.segment_minutes == 4 and item.query_pair == (3, 4) for item in shown)
    assert [item.unreliable for item in shown] == [False, True]
    assert [item.source_observations for item in shown] == [item.source_observations for item in markers]


def test_denominator_uses_all_matched_arrivals_at_the_two_stops():
    rows = [row(3, value) for value in (8, 18, 28)]
    rows += [row(4, value) for value in (1, 10, 22, 34)]
    shown = refine([marker([3])], rows)[0]
    # Matched travel times are 2, 4 and 6 minutes; the central estimate is 4.
    assert shown.segment_minutes == 4
    assert shown.position == 3.75


def test_staggered_response_countdowns_use_absolute_arrival_differences():
    rows = [replace(row(3, value, age=3), minutes=value + .05,
                    signed_minutes=value + .05) for value in (5, 15)]
    rows += [row(4, value) for value in (.25, 6, 16)]
    shown = refine([marker([2])], rows)[0]
    assert shown.segment_minutes == 1
    assert shown.position == 3.75
    older = [replace(item, cache_age_seconds=item.cache_age_seconds + 10) for item in rows]
    assert refine([marker([2])], older)[0].position == shown.position


def test_precise_future_eta_displayed_as_zero_stays_before_the_stop():
    rows = [row(3, value) for value in (8, 18)]
    rows += [replace(row(4, .25), minutes=0), row(4, 12), row(4, 22)]
    shown = refine([marker([2], position=4)], rows)[0]
    assert shown.position == pytest.approx(3.9375)


def test_exact_signed_zero_crossing_keeps_its_own_measured_interval():
    rows = [row(3, -.6), row(3, 8), row(4, .2), row(4, 12)]
    initial = replace(marker([0, 2]), bracket=(3, 4), bracket_eta_offsets=(-.6, .2))
    shown = refine([initial], rows)[0]
    assert shown.position == 3.75
    assert shown.segment_minutes == .8


def test_rounded_zero_never_certifies_passage_and_refreshes_both_sides():
    rows = [row(3, 8, rounded=True), row(4, 0, rounded=True), row(4, 12, rounded=True)]
    shown = refine([marker([1], position=4.5)], rows)[0]
    assert shown.position == 4
    assert shown.position_authoritative is False
    assert shown.priority_indices == {3, 4, 5}


def test_unresolved_zero_plateau_uses_the_midpoint_until_more_queries_disambiguate():
    rows = [row(3, 0, rounded=True), row(4, 0, rounded=True), row(5, 2, rounded=True)]
    shown = refine([marker([0, 1, 2], position=4.5)], rows)[0]
    assert shown.position == 3.5
    assert shown.query_stops == (2, 3, 4, 5)
    assert shown.position_authoritative is False
    assert shown.segment_minutes is None


def test_stale_pair_requests_actual_stops_without_inventing_speed():
    rows = [row(3, 8, age=61), row(4, 1), row(4, 12)]
    shown = refine([marker([1])], rows)[0]
    assert shown.query_pair == (3, 4)
    assert shown.segment_minutes is None
    assert shown.position == 3.5


def test_new_downstream_response_aligns_an_earlier_positive_upstream_eta():
    rows = [row(3, .2, age=30),
            replace(row(4, 1), minutes=.5, signed_minutes=.5,
                    observed_at=NOW + timedelta(seconds=30))]
    shown = refine([marker([0, 1])], rows)[0]
    # At the new observation: -0.3 minutes at stop 3 and +0.5 at stop 4.
    assert shown.segment_minutes == pytest.approx(.8)
    assert shown.position == pytest.approx(3.375)
    assert shown.position_authoritative is True


def test_full_responses_walk_adjacent_intervals_until_the_countdown_fits():
    rows = [row(2, 8), row(3, 1), row(3, 10), row(4, 5), row(4, 14)]
    shown = refine([marker([3], position=1.5)], rows)[0]
    # Five minutes to stop 4: four minutes from 3 to 4, then one of two to 3.
    assert shown.position == 2.5
    assert shown.query_pair == (2, 3)
    assert shown.segment_minutes == 2


def test_other_stops_disambiguate_consecutive_zeroes_before_midpoint_fallback():
    rows = [row(3, 0, rounded=True), row(3, 8), row(4, 0, rounded=True), row(4, 8.4),
            row(5, .5), row(5, 8.8)]
    shown = refine([marker([0, 2, 4], position=4.5)], rows)[0]
    # Peer vehicles measure 0.4 minutes per interval. The precise 0.5 at stop 5
    # places this bus at -0.3 and +0.1 at stops 3/4, inside both rounding bands.
    assert shown.position == pytest.approx(3.75)
    assert shown.segment_minutes == pytest.approx(.4)
    assert shown.position_authoritative is True
    assert shown.bracket == (3, 4)


def test_three_unresolved_zero_stops_use_the_middle_stop():
    rows = [row(index, 0, rounded=True) for index in (3, 4, 5)]
    shown = refine([marker([0, 1, 2], position=5)], rows)[0]
    assert shown.position == 4
    assert shown.position_authoritative is False
    assert shown.query_stops == (2, 3, 4, 5, 6)


def test_hidden_fourth_upstream_row_does_not_certify_passage():
    rows = [row(3, value) for value in (1, 2, 3)]
    rows += [row(4, value) for value in (2, 3, 8)]
    shown = refine([marker([5])], rows)[0]
    # The matched row at stop 3 is positive: still approach it, never claim the
    # bus has passed merely because its rank/countdown changed between stops.
    assert shown.segment_minutes is None
    assert shown.query_pair == (2, 3)
    assert shown.position <= 3


def test_leading_row_absent_upstream_stays_inside_the_confirmed_interval_in_a_delay():
    rows = [row(3, 10), row(4, 5), row(4, 14)]
    shown = refine([marker([1])], rows)[0]
    assert shown.query_pair == (3, 4)
    assert shown.bracket == (3, 4)
    assert shown.position == 3


def test_local_matching_cannot_collapse_distinct_cohorts_onto_another_markers_rows():
    rows = [row(2, -.5), row(3, .5), row(4, 2.5), row(5, 3.5)]
    shown = refine([marker([0, 1, 2], position=2.5), marker([3], position=4.2)], rows)
    assert [item.position for item in shown] == [2.5, 4.2]
    assert shown[0].segment_minutes == 1
    assert shown[1].segment_minutes is None
    assert shown[1].query_pair == (4, 5)


def test_wrong_route_and_direction_cannot_supply_the_denominator():
    rows = [replace(row(3, 8), bound="inbound"), row(4, 1), row(4, 12)]
    shown = refine([marker([1])], rows)[0]
    assert shown.segment_minutes is None
    assert shown.query_pair == (3, 4)
