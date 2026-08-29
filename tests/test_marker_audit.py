"""Frame-level source-to-marker correspondence tests."""

from types import SimpleNamespace as Obj

from dashboard.maps.marker_audit import (
    _checkpoint_check,
    _Evidence,
    _match,
    _verified_gate_index,
    audit_gmb_marker_pairs,
    audit_marker_positions,
)
from dashboard.maps.positions import BusEstimate, estimate_bus_positions
from dashboard.models import EtaKind, Operator

KMB_SOUTH = "B002CEF0DBC568F5"
KMB_91P_SOUTH = "E9018F8A7E096544"


def _line(
    *,
    operator: str = "KMB",
    route: str = "91",
    bound: str = "outbound",
    gate_index: int | None = 2,
    count: int = 8,
    gate_stop: str = KMB_SOUTH,
    repeat_gate_at_end: bool = False,
):
    stops = [
        Obj(stop_id=f"stop-{index}", name=f"Stop {index}")
        for index in range(count)
    ]
    if gate_index is not None:
        stops[gate_index] = Obj(stop_id=gate_stop, name="mapped gate")
    if repeat_gate_at_end:
        stops[-1] = Obj(stop_id=gate_stop, name="mapped gate again")
    return Obj(operator=operator, route=route, bound=bound, stops=stops)


def _row(
    index: int,
    minutes: float,
    *,
    operator: str = "KMB",
    route: str = "91",
    bound: str = "outbound",
    kind: EtaKind = EtaKind.REALTIME,
    cache_age_seconds: float | None = None,
):
    return Obj(
        operator=operator,
        route=route,
        bound=bound,
        index=index,
        minutes=minutes,
        kind=kind,
        cache_age_seconds=cache_age_seconds,
    )


def _marker(
    position: float,
    *,
    operator: Operator = Operator.KMB,
    route: str = "91",
    bound: str = "outbound",
    observations: frozenset[tuple[str, int]] = frozenset(),
):
    return BusEstimate(
        route,
        0.0,
        0.0,
        operator,
        0.0,
        route=route,
        bound=bound,
        position=position,
        source_observations=observations,
    )


def _checks(result, kind: str):
    return [check for check in result["checks"] if check["kind"] == kind]


def _issues(result, kind: str):
    return [issue for issue in result["issues"] if issue["kind"] == kind]


def _gmb_row(index: int, minutes: float):
    return _row(index, minutes, operator="GMB", route="11", bound="seq-1")


def _gmb_marker(lat: float, lon: float, observations):
    return BusEstimate(
        "11", lat, lon, Operator.GMB, 0.0, route="11", bound="seq-1",
        position=1.0, source_observations=frozenset(observations),
    )


def test_gmb_marker_pair_diagnostic_classifies_stacked_and_includes_eta_context():
    result = audit_gmb_marker_pairs(
        [_gmb_row(4, 5), _gmb_row(4, 25)],
        [],
        [_gmb_marker(22.334, 114.230, {('probe', 0)}),
         _gmb_marker(22.334, 114.230, {('probe', 1)})],
    )
    assert len(result) == 1
    assert result[0]["classification"] == "stacked"
    assert result[0]["common_stops"][0]["index"] == 4
    assert result[0]["common_stops"][0]["eta_delta"] == 20


def test_gmb_marker_pair_diagnostic_classifies_nearby_over_eight_pixels():
    result = audit_gmb_marker_pairs(
        [_gmb_row(4, 5), _gmb_row(4, 25)], [],
        [_gmb_marker(22.334, 114.230, {('probe', 0)}),
         _gmb_marker(22.334, 114.231, {('probe', 1)})],
    )
    assert result and result[0]["classification"] == "nearby"
    assert result[0]["pixel_distance"] > 8


def test_gmb_marker_pair_diagnostic_finds_two_departures_merged_into_one_marker():
    result = audit_gmb_marker_pairs(
        [_gmb_row(4, 5), _gmb_row(4, 25)], [],
        [_gmb_marker(22.334, 114.230, {('probe', 0), ('probe', 1)})],
    )
    assert len(result) == 1
    assert result[0]["marker_ids"] == [0, 0]
    assert result[0]["classification"] == "stacked"
    assert result[0]["source_observations"] == [('probe', 0), ('probe', 1)]


def test_gmb_marker_pair_diagnostic_ignores_close_etas_and_missing_common_stop():
    close = audit_gmb_marker_pairs(
        [_gmb_row(4, 5), _gmb_row(4, 14)], [],
        [_gmb_marker(22.334, 114.230, {('probe', 0)}),
         _gmb_marker(22.334, 114.230, {('probe', 1)})],
    )
    missing = audit_gmb_marker_pairs(
        [_gmb_row(4, 5), _gmb_row(5, 25)], [],
        [_gmb_marker(22.334, 114.230, {('probe', 0)}),
         _gmb_marker(22.334, 114.230, {('probe', 1)})],
    )
    assert close == []
    assert missing == []


def test_audit_accepts_gate_crossing_position_derived_from_passed_probe_row():
    line = _line(operator="GMB", route="2004791", bound="seq-1", gate_index=None, count=15)
    line.stops[6] = Obj(stop_id="20013010", name="mapped gate")
    line.path = [(22.3, 114.2), (22.3, 114.22)]
    line.stop_offsets = [index * 100.0 for index in range(15)]
    gate = _row(6, 30, operator="GMB", route="2004791", bound="seq-1")
    probes = [
        _row(12, 25, operator="GMB", route="2004791", bound="seq-1"),
        _row(12, 57, operator="GMB", route="2004791", bound="seq-1"),
    ]
    estimates = estimate_bus_positions(probes, [line], authoritative_etas=[gate])
    assert len(estimates) == 1
    assert estimates[0].position == 7
    assert estimates[0].source_observations == frozenset({("probe", 0)})
    result = audit_marker_positions(probes, [gate], estimates, [line])
    assert result["ok"]


def test_audit_excludes_unproven_downstream_row_before_gate():
    line = _line(operator="GMB", route="2004791", bound="seq-1", gate_index=None, count=15)
    line.stops[6] = Obj(stop_id="20013010", name="mapped gate")
    gate = _row(6, 30, operator="GMB", route="2004791", bound="seq-1")
    probe = _row(12, 24, operator="GMB", route="2004791", bound="seq-1")
    result = audit_marker_positions([probe], [gate], [], [line])
    assert result["ok"]
    assert result["stats"]["excluded_probe_rows"] == 1


def test_audit_keeps_negative_raw_row_assigned_to_departed_gate_journey():
    line = _line(operator="GMB", route="2004791", bound="seq-1", gate_index=None, count=15)
    line.stops[6] = Obj(stop_id="20013010", name="mapped gate")
    line.path = [(22.3, 114.2), (22.3, 114.22)]
    line.stop_offsets = [index * 100.0 for index in range(15)]
    gate = _row(6, 2, operator="GMB", route="2004791", bound="seq-1")
    probe = _row(12, 25, operator="GMB", route="2004791", bound="seq-1")
    estimates = estimate_bus_positions([probe], [line], authoritative_etas=[gate])
    assert len(estimates) == 1
    assert {("gate", 0), ("probe", 0)} == estimates[0].source_observations
    result = audit_marker_positions([probe], [gate], estimates, [line])
    assert result["ok"]


def test_match_preserves_original_identities_after_value_sorting():
    result = _match([20, 3], [3, 20], tolerance=0)
    assert result["pair_indices"] == [(1, 0), (0, 1)]
    assert result["cardinality"] == 2


def test_match_beats_naive_zip_and_keeps_the_unmatched_original_marker():
    result = _match([20, 3], [20, 2, 4], tolerance=1)
    assert result["cardinality"] == 2
    assert (0, 0) in result["pair_indices"]
    assert result["unmatched_markers"] in ([1], [2])


def test_authoritative_gate_matches_exact_departed_vehicle():
    result = audit_marker_positions([], [_row(2, 2)], [_marker(1)], [_line()])
    gate = _checks(result, "authoritative")[0]
    assert gate["ok"] and gate["match"]["cardinality"] == 1
    assert not result["issues"]


def test_authoritative_gate_preserves_two_vehicle_multiplicity():
    result = audit_marker_positions(
        [], [_row(2, 2), _row(2, 4)], [_marker(1), _marker(0)], [_line()]
    )
    assert _checks(result, "authoritative")[0]["match"]["cardinality"] == 2
    assert result["ok"]


def test_gate_provenance_is_identity_exact_despite_coarse_timing_delta():
    marker = _marker(0, observations=frozenset({("gate", 0)}))
    result = audit_marker_positions([], [_row(2, 0)], [marker], [_line()])
    gate = _checks(result, "authoritative")[0]
    assert gate["ok"]
    assert gate["max_timing_delta"] == 4.0


def test_probe_only_pre_gate_marker_is_extra_with_gate_provenance():
    markers = [
        _marker(1, observations=frozenset({("gate", 0)})),
        _marker(0, observations=frozenset({("probe", 99)})),
    ]
    result = audit_marker_positions([], [_row(2, 2)], markers, [_line()])
    gate = _checks(result, "authoritative")[0]
    assert not gate["ok"]
    assert gate["match"]["unmatched_markers"] == [1]


def test_future_terminus_departure_is_excluded_without_a_phantom():
    line = _line(route="291P", gate_index=0, gate_stop=KMB_91P_SOUTH)
    result = audit_marker_positions([], [_row(0, 3, route="291P")], [], [line])
    gate = _checks(result, "authoritative")[0]
    assert gate["ok"] and gate["excluded_undeparted"] == 1
    assert result["stats"]["excluded_undeparted"] == 1


def test_future_terminus_departure_does_not_excuse_an_extra_marker():
    line = _line(route="291P", gate_index=0, gate_stop=KMB_91P_SOUTH)
    result = audit_marker_positions(
        [], [_row(0, 3, route="291P")], [_marker(0, route="291P")], [line]
    )
    assert _issues(result, "authoritative")


def test_authoritative_index_mismatch_is_an_issue():
    result = audit_marker_positions([], [_row(1, 0)], [_marker(2)], [_line()])
    assert _issues(result, "authoritative-index")


def test_empty_gate_feed_is_explicitly_inconclusive():
    result = audit_marker_positions([_row(6, 4)], [], [_marker(4)], [_line()])
    gate = _checks(result, "authoritative")[0]
    assert gate["inconclusive"]
    assert gate["reason"] == "no authoritative HKUST rows"


def test_extra_pre_gate_marker_fails_the_one_to_one_gate_check():
    result = audit_marker_positions(
        [], [_row(2, 2)], [_marker(1), _marker(0)], [_line()]
    )
    gate = _checks(result, "authoritative")[0]
    assert gate["match"]["unmatched_markers"] == [1]
    assert _issues(result, "authoritative")


def test_post_hkust_marker_is_proved_at_a_later_stop():
    result = audit_marker_positions([_row(6, 4)], [], [_marker(4)], [_line()])
    assert _checks(result, "downstream")[0]["ok"]
    assert not _issues(result, "downstream")


def test_raw_passed_probe_is_owned_at_checkpoint_and_proves_downstream():
    """A passed probe remains visible even if its order correction lags."""
    line = _line()
    gate = _row(2, 2)
    probe = _row(6, 4)
    markers = [
        _marker(1, observations=frozenset({("gate", 0)})),
        _marker(4, observations=frozenset({("probe", 0)})),
    ]
    result = audit_marker_positions([probe], [gate], markers, [line])
    checkpoint = next(
        check for check in _checks(result, "checkpoint") if check["checkpoint"] == 6
    )
    assert _checks(result, "authoritative")[0]["ok"]
    assert checkpoint["ok"]
    assert checkpoint["matched_marker_ids"] == [1]
    assert _checks(result, "downstream")[0]["ok"]


def test_post_hkust_marker_without_matching_later_eta_fails():
    result = audit_marker_positions([_row(6, 1)], [], [_marker(4)], [_line()])
    assert not _checks(result, "downstream")[0]["ok"]
    assert _issues(result, "downstream")


def test_two_post_hkust_markers_require_two_later_eta_rows():
    result = audit_marker_positions(
        [_row(6, 4), _row(6, 2)], [], [_marker(4), _marker(5)], [_line()]
    )
    assert sum(check["ok"] for check in _checks(result, "downstream")) == 2
    assert not _issues(result, "checkpoint")


def test_one_later_eta_cannot_support_two_post_hkust_markers():
    result = audit_marker_positions(
        [_row(6, 2)], [], [_marker(4), _marker(5)], [_line()]
    )
    assert sum(check["ok"] for check in _checks(result, "downstream")) == 1
    assert _issues(result, "checkpoint")
    assert _issues(result, "downstream")


def test_future_probe_row_is_excluded_and_checkpoint_is_inconclusive():
    result = audit_marker_positions([_row(1, 4)], [], [], [_line()])
    assert result["stats"]["excluded_probe_rows"] == 1
    assert _checks(result, "checkpoint")[0]["inconclusive"]
    assert result["ok"]


def test_scheduled_singleton_is_inconclusive_not_vehicle_evidence():
    result = audit_marker_positions(
        [_row(6, 4, kind=EtaKind.SCHEDULED)], [], [_marker(4)], [_line()]
    )
    downstream = _checks(result, "downstream")[0]
    assert downstream["inconclusive"]
    assert not _issues(result, "downstream")


def test_scheduled_rows_at_two_occurrences_corroborate_one_vehicle():
    result = audit_marker_positions(
        [
            _row(6, 4, kind=EtaKind.SCHEDULED),
            _row(7, 6, kind=EtaKind.SCHEDULED),
        ],
        [],
        [_marker(4)],
        [_line()],
    )
    assert _checks(result, "downstream")[0]["ok"]
    assert not _issues(result, "checkpoint")


def test_realtime_row_corroborates_nearby_scheduled_occurrence():
    result = audit_marker_positions(
        [_row(6, 4, kind=EtaKind.SCHEDULED), _row(7, 6)],
        [],
        [_marker(4)],
        [_line()],
    )
    assert result["stats"]["excluded_probe_rows"] == 0
    assert result["ok"]


def test_observed_checkpoint_passes_same_one_to_one_rule():
    result = audit_marker_positions(
        [_row(6, 4)], [], [_marker(4)], [_line()], frame_id=3
    )
    check = _checks(result, "checkpoint")[0]
    assert check["checkpoint"] == 6 and check["ok"]


def test_observed_checkpoint_reports_a_mismatch():
    result = audit_marker_positions(
        [_row(6, 4)], [], [_marker(1)], [_line()], frame_id=3
    )
    assert not _checks(result, "checkpoint")[0]["ok"]
    assert _issues(result, "checkpoint")


def test_eligible_upstream_marker_without_source_journey_is_rejected():
    probes = [_row(6, 4)]
    markers = [
        _marker(4, observations=frozenset({("probe", 0)})),
        _marker(3, observations=frozenset({("probe", 99)})),
    ]
    result = audit_marker_positions(probes, [], markers, [_line()])
    check = _checks(result, "checkpoint")[0]
    assert not check["ok"]
    assert check["match"]["unowned_markers"] == [1]


def _evidence(index, minutes, checkpoint):
    row = _row(checkpoint, minutes)
    return _Evidence(
        row=row,
        row_index=index,
        checkpoint=checkpoint,
        minutes=float(minutes),
        raw_position=float(checkpoint) - float(minutes) / 2.0,
        scheduled=False,
    )


def test_validated_gate_boundary_marker_is_not_unowned_at_any_offset():
    expected = [_evidence(0, 2, 8)]
    exact = _checkpoint_check(
        key=("KMB", "91", "outbound"), checkpoint=8,
        expected=expected, excluded=[],
        markers=[
            (0, _marker(7, observations=frozenset({("probe", 0)}))),
            (1, _marker(8, observations=frozenset({("gate", 0)}))),
        ],
        kind="checkpoint", tolerance=2, all_probe_rows=expected,
        valid_gate_tokens={0},
    )[0]
    offset = _checkpoint_check(
        key=("KMB", "91", "outbound"), checkpoint=8,
        expected=expected, excluded=[],
        markers=[
            (0, _marker(7, observations=frozenset({("probe", 0)}))),
            (1, _marker(7.9, observations=frozenset({("gate", 0)}))),
        ],
        kind="checkpoint", tolerance=2, all_probe_rows=expected,
        valid_gate_tokens={0},
    )[0]
    assert exact["ok"]
    assert exact["match"]["boundary_markers"] == [1]
    assert offset["ok"]
    assert offset["match"]["boundary_markers"] == [1]


def test_later_probe_boundary_is_narrow_and_gate_token_disqualifies_it():
    current = _evidence(1, 4, 8)
    later = _evidence(0, 2, 9)
    latest = _evidence(2, 2, 10)
    base_markers = [
        _marker(6, observations=frozenset({("probe", 1)})),
        _marker(7.9628875, observations=frozenset({("probe", 0)})),
    ]
    boundary = _checkpoint_check(
        key=("KMB", "91", "outbound"), checkpoint=8,
        expected=[current], excluded=[], markers=list(enumerate(base_markers)),
        kind="checkpoint", tolerance=2,
            all_probe_rows=[current, later, latest],
        valid_gate_tokens={2},
    )[0]
    offset = _checkpoint_check(
        key=("KMB", "91", "outbound"), checkpoint=8,
        expected=[current], excluded=[],
        markers=[(0, base_markers[0]), (1, _marker(6.9, observations=frozenset({("probe", 0)})))],
        kind="checkpoint", tolerance=2,
            all_probe_rows=[current, later, latest],
        valid_gate_tokens={2},
    )[0]
    with_gate = _checkpoint_check(
        key=("KMB", "91", "outbound"), checkpoint=8,
        expected=[current], excluded=[],
        markers=[(0, base_markers[0]), (1, _marker(7.9628875, observations=frozenset({("probe", 0), ("gate", 99)})))],
        kind="checkpoint", tolerance=2,
            all_probe_rows=[current, later, latest],
        valid_gate_tokens={2},
    )[0]
    assert boundary["ok"] and boundary["match"]["boundary_markers"] == [1]
    assert not offset["ok"]
    assert not with_gate["ok"]


def test_validated_gate_proven_marker_is_source_backed_at_other_checkpoints():
    current = _evidence(1, 4, 8)
    result = _checkpoint_check(
        key=("KMB", "91", "outbound"), checkpoint=8,
        expected=[current], excluded=[],
        markers=[
            (0, _marker(6, observations=frozenset({("probe", 1)}))),
            (1, _marker(5, observations=frozenset({("gate", 0)}))),
        ],
        kind="checkpoint", tolerance=2,
        all_probe_rows=[current], valid_gate_tokens={0},
    )[0]
    assert result["ok"]
    assert result["match"]["boundary_markers"] == [1]


def test_terminal_probe_singleton_is_allowed_at_refresh_frontier():
    terminal = _evidence(0, 20, 10)
    current = _evidence(1, 20, 10)
    result = _checkpoint_check(
        key=("KMB", "91", "outbound"), checkpoint=10,
        expected=[current], excluded=[],
        markers=[
            (0, _marker(0, observations=frozenset({("probe", 1)}))),
            (1, _marker(2, observations=frozenset({("probe", 0)}))),
        ],
        kind="checkpoint", tolerance=2,
        all_probe_rows=[terminal, current],
    )[0]
    assert result["ok"]
    assert result["match"]["boundary_markers"] == [1]


def test_interior_probe_singleton_is_not_refresh_frontier_evidence():
    interior = _evidence(0, 20, 10)
    current = _evidence(1, 20, 8)
    later = _evidence(2, 4, 11)
    result = _checkpoint_check(
        key=("KMB", "91", "outbound"), checkpoint=8,
        expected=[current], excluded=[],
        markers=[
            (0, _marker(-2, observations=frozenset({("probe", 1)}))),
            (1, _marker(2, observations=frozenset({("probe", 0)}))),
        ],
        kind="checkpoint", tolerance=2,
        all_probe_rows=[interior, current, later],
    )[0]
    assert not result["ok"]


def test_distinct_source_journey_tokens_allow_same_stop_multiplicity():
    probes = [_row(6, 4), _row(6, 6)]
    markers = [
        _marker(4, observations=frozenset({("probe", 0)})),
        _marker(3, observations=frozenset({("probe", 1)})),
    ]
    result = audit_marker_positions(probes, [], markers, [_line()])
    check = _checks(result, "checkpoint")[0]
    assert check["ok"]
    assert check["match"]["cardinality"] == 2


def test_all_observed_checkpoints_are_audited_deterministically():
    probes = [_row(4, 2), _row(5, 4), _row(6, 6)]
    first = audit_marker_positions(probes, [], [_marker(3)], [_line()], frame_id=7)
    again = audit_marker_positions(probes, [], [_marker(3)], [_line()], frame_id=7)
    assert [c["checkpoint"] for c in _checks(first, "checkpoint")] == [4, 5, 6]
    assert [c["checkpoint"] for c in _checks(again, "checkpoint")] == [4, 5, 6]
    assert first["stats"]["observed_checkpoints"] == 3
    assert first["stats"]["audited_checkpoints"] == 3
    assert first["stats"]["uncovered_checkpoints"] == 0
    assert first["stats"]["uncovered_probe_rows"] == 0


def test_route_without_current_gate_geometry_audits_all_checkpoints():
    line = _line(route="91P", gate_index=None)
    result = audit_marker_positions(
        [_row(6, 4, route="91P")],
        [],
        [_marker(4, route="91P")],
        [line],
    )
    assert _checks(result, "authoritative")[0]["inconclusive"]
    assert _checks(result, "checkpoint")[0]["ok"]
    assert not _checks(result, "downstream")


def test_citybus_operator_normalization_reaches_verified_gate():
    line = _line(operator="CTB", route="792M", gate_stop="003130")
    result = audit_marker_positions(
        [_row(6, 4, operator="CTB", route="792M")],
        [_row(2, 2, operator="CTB", route="792M")],
        [
            _marker(1, operator=Operator.CITYBUS, route="792M"),
            _marker(4, operator=Operator.CITYBUS, route="792M"),
        ],
        [line],
    )
    assert _checks(result, "authoritative")[0]["ok"]
    assert _checks(result, "downstream")[0]["ok"]


def test_circular_104_first_gate_and_later_repeat_is_downstream():
    line = _line(
        operator="GMB",
        route="104",
        bound="seq-1",
        gate_index=0,
        gate_stop="20015226",
        repeat_gate_at_end=True,
    )
    marker = _marker(5, operator=Operator.GMB, route="104", bound="seq-1")
    probe = _row(7, 4, operator="GMB", route="104", bound="seq-1")
    result = audit_marker_positions([probe], [], [marker], [line])
    assert _verified_gate_index(line) == 0
    assert _checks(result, "downstream")[0]["ok"]
    assert _checks(result, "checkpoint")[0]["checkpoint"] == 7


def test_checkpoint_audit_never_uses_another_route_stop_index():
    first = _line()
    second = _line(route="91M", bound="inbound", gate_index=2)
    result = audit_marker_positions(
        [_row(6, 4), _row(4, 2, route="91M", bound="inbound")],
        [],
        [_marker(4), _marker(3, route="91M", bound="inbound")],
        [first, second],
        frame_id=4,
    )
    checkpoints = {
        check["key"]: check["checkpoint"]
        for check in _checks(result, "checkpoint")
    }
    assert checkpoints[("KMB", "91", "outbound")] == 6
    assert checkpoints[("KMB", "91M", "inbound")] == 4
    assert result["stats"]["uncovered_checkpoints"] == 0


def test_exact_probe_provenance_preserves_two_rows_as_two_vehicles():
    probes = [_row(6, 4), _row(6, 2)]
    markers = [
        _marker(4, observations=frozenset({("probe", 0)})),
        _marker(5, observations=frozenset({("probe", 1)})),
    ]
    result = audit_marker_positions(probes, [], markers, [_line()])
    checkpoint = _checks(result, "checkpoint")[0]
    assert checkpoint["ok"]
    assert checkpoint["matched_marker_ids"] == [0, 1]


def test_two_same_stop_rows_cannot_share_one_provenance_marker():
    probes = [_row(6, 4), _row(6, 2)]
    marker = _marker(
        5,
        observations=frozenset({("probe", 0), ("probe", 1)}),
    )
    result = audit_marker_positions(probes, [], [marker], [_line()])
    checkpoint = _checks(result, "checkpoint")[0]
    assert not checkpoint["ok"]
    assert checkpoint["match"]["unmatched_markers"] == [0]


def test_probe_provenance_remains_valid_after_marker_passes_checkpoint():
    marker = _marker(6.2, observations=frozenset({("probe", 0)}))
    result = audit_marker_positions([_row(6, 0.2)], [], [marker], [_line()])
    checkpoint = _checks(result, "checkpoint")[0]
    assert checkpoint["ok"]
    assert checkpoint["passed_checkpoint"] == 1


def test_probe_provenance_fails_when_marker_timing_exceeds_tolerance():
    marker = _marker(0.2, observations=frozenset({("probe", 0)}))
    result = audit_marker_positions([_row(6, 4)], [], [marker], [_line()])
    checkpoint = _checks(result, "checkpoint")[0]
    assert not checkpoint["ok"]
    assert checkpoint["match"]["timing_outliers"] == [0]
    assert checkpoint["max_timing_delta"] == 7.6
    assert _issues(result, "checkpoint")


def test_probe_only_marker_checks_only_effective_position_anchor_timing():
    """Older ladder probes prove identity without invalidating the final anchor."""
    line = _line(count=26)
    earlier = _row(21, 4.45)
    anchor = _row(25, 9.72)
    marker = _marker(
        20.14,
        observations=frozenset({("probe", 0), ("probe", 1)}),
    )

    result = audit_marker_positions([earlier, anchor], [], [marker], [line])

    checkpoint = next(
        check for check in _checks(result, "checkpoint")
        if check["checkpoint"] == 21
    )
    assert checkpoint["ok"]
    assert checkpoint["match"]["superseded_position_observations"] == [
        ("probe", 0)
    ]
    anchor_check = next(
        check for check in _checks(result, "checkpoint")
        if check["checkpoint"] == 25
    )
    assert anchor_check["ok"]


def test_probe_only_marker_anchor_still_fails_when_anchor_timing_is_wrong():
    line = _line(count=26)
    # Keep the earlier observation well behind the final marker so the later
    # row is deterministically selected as the position anchor.
    probes = [_row(21, 30.0), _row(25, 20.0)]
    marker = _marker(
        20.14,
        observations=frozenset({("probe", 0), ("probe", 1)}),
    )

    result = audit_marker_positions(probes, [], [marker], [line])

    anchor_check = next(
        check for check in _checks(result, "checkpoint")
        if check["checkpoint"] == 25
    )
    assert not anchor_check["ok"]
    assert anchor_check["match"]["timing_outliers"] == [0]


def test_passed_gate_marker_uses_downstream_probe_anchor_for_audit():
    line = _line(count=15, gate_index=9)
    line.path = [(22.3, 114.2), (22.3, 114.22)]
    line.stop_offsets = [index * 100.0 for index in range(15)]
    gate_rows = [_row(9, 2), _row(9, 22)]
    probes = [_row(10, 0.5), _row(10, 20.5)]
    estimates = estimate_bus_positions(
        probes, [line], authoritative_etas=gate_rows
    )

    assert len(estimates) == 1
    assert estimates[0].position == 9.75
    result = audit_marker_positions(probes, gate_rows, estimates, [line])

    assert result["ok"]
    checkpoint = _checks(result, "checkpoint")[0]
    assert checkpoint["ok"]


def test_gate_contract_tolerance_is_not_widened_by_caller():
    evidence = _Evidence(
        row=_row(20, 0),
        row_index=0,
        checkpoint=20,
        minutes=0,
        raw_position=20,
        scheduled=False,
        timing_tolerance=25,
    )
    marker = _marker(0, observations=frozenset({("probe", 0)}))
    check, _matched = _checkpoint_check(
        key=("KMB", "91", "outbound"),
        checkpoint=20,
        expected=[evidence],
        excluded=[],
        markers=[(0, marker)],
        kind="checkpoint",
        tolerance=100,
    )
    assert not check["ok"]
    assert check["max_timing_delta"] == 40


def test_origin_gate_future_is_excluded_but_later_vehicle_is_checked():
    line = _line(
        operator="KMB",
        route="291P",
        bound="outbound",
        gate_index=0,
        gate_stop=KMB_91P_SOUTH,
    )
    probe = _row(6, 4, route="291P")
    marker = _marker(
        4,
        route="291P",
        observations=frozenset({("probe", 0)}),
    )
    result = audit_marker_positions(
        [probe],
        [_row(0, 3, route="291P", kind=EtaKind.SCHEDULED)],
        [marker],
        [line],
    )
    assert _checks(result, "authoritative")[0]["excluded_undeparted"] == 1
    assert _checks(result, "downstream")[0]["ok"]
    assert result["ok"]


def test_live_later_row_is_not_excluded_with_future_scheduled_gate():
    line = _line(
        operator="GMB",
        route="11",
        bound="seq-1",
        gate_index=6,
        count=20,
        gate_stop="20013010",
    )
    probe = _row(
        15,
        12,
        operator="GMB",
        route="11",
        bound="seq-1",
    )
    marker = _marker(
        9,
        operator=Operator.GMB,
        route="11",
        bound="seq-1",
        observations=frozenset({("probe", 0)}),
    )
    gate = _row(
        6,
        14,
        operator="GMB",
        route="11",
        bound="seq-1",
        kind=EtaKind.SCHEDULED,
    )
    result = audit_marker_positions([probe], [gate], [marker], [line])
    assert _checks(result, "authoritative")[0]["excluded_undeparted"] == 1
    assert _checks(result, "downstream")[0]["ok"]
    assert result["ok"]


def test_expected_probe_row_missing_from_all_marker_provenance_fails():
    marker = _marker(4, observations=frozenset({("probe", 99)}))
    result = audit_marker_positions([_row(6, 4)], [], [marker], [_line()])
    checkpoint = _checks(result, "checkpoint")[0]
    assert not checkpoint["ok"]
    assert checkpoint["match"]["unmatched_source_values"] == [4.0]


def test_audit_excludes_realtime_singleton_superseded_by_newer_track():
    line = _line(
        operator="GMB",
        route="11M",
        bound="seq-2",
        gate_index=0,
        count=12,
        gate_stop="20012474",
    )
    probes = [
        _row(
            3,
            2.21,
            operator="GMB",
            route="11M",
            bound="seq-2",
            kind=EtaKind.SCHEDULED,
            cache_age_seconds=0,
        ),
        _row(
            4,
            4.21,
            operator="GMB",
            route="11M",
            bound="seq-2",
            kind=EtaKind.SCHEDULED,
            cache_age_seconds=0,
        ),
        _row(
            5,
            6.21,
            operator="GMB",
            route="11M",
            bound="seq-2",
            kind=EtaKind.SCHEDULED,
            cache_age_seconds=0,
        ),
        _row(
            4,
            3.906,
            operator="GMB",
            route="11M",
            bound="seq-2",
            cache_age_seconds=30,
        ),
    ]
    gate = _row(
        0,
        100,
        operator="GMB",
        route="11M",
        bound="seq-2",
        kind=EtaKind.SCHEDULED,
    )
    marker = _marker(
        1.895,
        operator=Operator.GMB,
        route="11M",
        bound="seq-2",
        observations=frozenset(
            {("probe", 0), ("probe", 1), ("probe", 2)}
        ),
    )

    result = audit_marker_positions(probes, [gate], [marker], [line])

    assert result["ok"]
    checkpoint = next(
        check
        for check in _checks(result, "checkpoint")
        if check["checkpoint"] == 4
    )
    assert checkpoint["excluded_rows"] == 1
    assert checkpoint["ok"]
