from datetime import datetime
from types import SimpleNamespace

from scripts.verify_marker_tracker_live import (
    _json_safe_record,
    check_minute_baselines,
    compare_adjacent,
    evaluate_run,
    frame_record,
    fresh_routes,
    minute_checks,
)

KEY = ("KMB", "A", "in")


def item(position, track_id=None):
    return SimpleNamespace(
        operator="KMB", route="A", bound="in", position=position, track_id=track_id
    )


def frame(generation=1, tracks=((1, 2.0),), candidates=(2.0,), stamp="2026-01-01T00:00:00+00:00"):
    route = SimpleNamespace(
        route_key=KEY, generation=generation, collected_at=datetime.fromisoformat(stamp), rows=()
    )
    snapshot = SimpleNamespace(complete_routes=(route,), rows=())
    return frame_record(
        snapshot, [item(p) for p in candidates], [item(p, i) for i, p in tracks], {KEY: 20.0}, stamp
    )


def test_identity_and_cardinality_require_generation():
    issues, _ = compare_adjacent(frame(), frame(1, ((1, 2.0), (2, 3.0)), (2.0, 3.0)))
    assert {x["kind"] for x in issues} == {
        "identity_change_without_generation",
        "cardinality_without_generation",
    }


def test_complete_generation_requires_exact_candidate_cardinality():
    initial = frame(1, ((1, 2.0), (2, 3.0)), (2.0,))
    initial_issues, _ = compare_adjacent(None, initial)
    mismatch = next(
        issue for issue in initial_issues
        if issue["kind"] == "cardinality_mismatch_at_complete_generation"
    )
    assert (mismatch["candidate_count"], mismatch["track_count"]) == (1, 2)

    old = frame(1, ((1, 2.0),), (2.0,))
    current = frame(2, ((1, 2.0), (2, 3.0)), (2.0,))
    issues, _ = compare_adjacent(old, current, {})
    assert any(
        issue["kind"] == "cardinality_mismatch_at_complete_generation"
        for issue in issues
    )


def test_spacing_requires_equal_complete_cardinality():
    old = frame(1, ((1, 1.0), (2, 5.0)), (1.0, 5.0))
    bad = frame(1, ((1, 1.0), (2, 2.0)), (1.0, 5.0))
    assert any(x["kind"] == "spacing_mismatch" for x in compare_adjacent(old, bad)[0])
    unequal = frame(1, ((1, 1.0),), (1.0, 5.0))
    assert not any(x["kind"] == "spacing_mismatch" for x in compare_adjacent(old, unequal)[0])


def provenance_frame(candidate_positions, track_positions, *, generation=1,
                     provenance=((1, (1, 2), (3,)), (2, (3, 4), (5,))),
                     track_provenance=None,
                     stamp="2026-01-01T00:00:00+00:00"):
    route = SimpleNamespace(route_key=KEY, generation=generation,
                            collected_at=datetime.fromisoformat(stamp), rows=())

    def evidence(position, item_provenance):
        eta, bracket, sources = item_provenance
        return SimpleNamespace(operator="KMB", route="A", bound="in",
                               position=position, bracket=bracket,
                               eta_minutes=eta, eta_arrival_at=None,
                               boundary_age_seconds=1.0, source_indices=sources)

    candidates = [evidence(position, provenance[index])
                  for index, position in enumerate(candidate_positions)]
    track_provenance = provenance if track_provenance is None else track_provenance
    tracks = [evidence(position, track_provenance[index])
              for index, position in enumerate(track_positions)]
    for index, track in enumerate(tracks):
        track.track_id = index + 1
    return frame_record(SimpleNamespace(complete_routes=(route,), rows=()),
                        candidates, tracks, {KEY: 20}, stamp)


def test_spacing_turnover_with_equal_cardinality_is_inconclusive_without_provenance_bijection():
    old = provenance_frame((1.0, 5.0), (1.0, 5.0), generation=1)
    new = provenance_frame((1.0, 2.0), (1.0, 5.0), generation=2,
                           provenance=((3, (1, 2), (7,)), (4, (3, 4), (8,))),
                           track_provenance=((1, (1, 2), (3,)), (2, (3, 4), (5,))))
    state = {}
    issues, checks = compare_adjacent(old, new, state)
    assert not any(x["kind"] == "spacing_mismatch" for x in issues)
    assert checks == 0 and state["gap_inconclusive"][KEY] == 1
    repeated = provenance_frame((1.0, 2.0), (1.0, 5.0), generation=2,
                                provenance=((3, (1, 2), (7,)), (4, (3, 4), (8,))),
                                track_provenance=((1, (1, 2), (3,)), (2, (3, 4), (5,))))
    issues, checks = compare_adjacent(new, repeated, state)
    assert not any(x["kind"] == "spacing_mismatch" for x in issues)
    assert checks == 0 and state["gap_inconclusive"][KEY] == 2


def test_spacing_provenance_uses_duplicate_multisets():
    same = ((1, (1, 2), (3,)), (1, (1, 2), (3,)))
    old = provenance_frame((1.0, 5.0), (1.0, 5.0), provenance=same)
    matching = provenance_frame((1.0, 5.0), (1.0, 5.0), provenance=same)
    assert compare_adjacent(old, matching)[1] == 1
    different = provenance_frame((1.0, 2.0), (1.0, 5.0), provenance=same)
    different["candidate_evidence"][KEY][1]["source_indices"] = [9]
    state = {}
    issues, checks = compare_adjacent(old, different, state)
    assert not any(x["kind"] == "spacing_mismatch" for x in issues)
    assert checks == 0 and state["gap_inconclusive"][KEY] == 1


def test_duplicate_track_evidence_is_a_hard_failure():
    same = ((1, (1, 2), (3,)), (1, (1, 2), (3,)))
    duplicated = provenance_frame(
        (1.0, 1.0), (1.0, 1.0), provenance=same
    )
    issues, _checks = compare_adjacent(duplicated, duplicated)
    duplicate = next(
        issue for issue in issues
        if issue["kind"] == "duplicate_track_evidence"
    )
    assert {duplicate["track_id"], duplicate["other_track_id"]} == {1, 2}


def test_reused_source_ladder_is_duplicate_even_when_eta_timestamp_changed():
    record = provenance_frame((1.0, 1.0), (1.0, 1.0))
    evidence = record["track_evidence"][KEY]
    evidence[1]["source_observations"] = [["probe", 42]]
    evidence[2]["source_observations"] = [["probe", 42]]
    evidence[1]["eta_arrival_at"] = "2026-01-01T00:01:00+00:00"
    evidence[2]["eta_arrival_at"] = "2026-01-01T00:02:00+00:00"

    issues, _checks = compare_adjacent(record, record)

    duplicate = next(
        issue for issue in issues
        if issue["kind"] == "duplicate_track_evidence"
    )
    assert {duplicate["track_id"], duplicate["other_track_id"]} == {1, 2}


def test_matching_provenance_still_reports_real_spacing_error():
    old = provenance_frame((1.0, 5.0), (1.0, 5.0))
    bad = provenance_frame((1.0, 2.0), (1.0, 5.0))
    issues, checks = compare_adjacent(old, bad)
    assert any(x["kind"] == "spacing_mismatch" for x in issues)
    assert checks == 1


def test_spacing_provenance_accepts_json_style_observation_lists():
    old = provenance_frame((1.0, 5.0), (1.0, 5.0))
    replayed = provenance_frame((1.0, 5.0), (1.0, 5.0))
    replayed["candidate_evidence"][KEY][0]["source_observations"] = [
        ["probe", 1]
    ]
    replayed["track_evidence"][KEY][1]["source_observations"] = [
        ["probe", 1]
    ]

    issues, checks = compare_adjacent(old, replayed)

    assert not any(x["kind"] == "spacing_mismatch" for x in issues)
    assert checks == 1


def test_backward_and_crossing_violations():
    old = frame(1, ((1, 2.0), (2, 5.0)), (2.0, 5.0))
    new = frame(1, ((2, 1.0), (1, 4.0)), (1.0, 4.0))
    kinds = {x["kind"] for x in compare_adjacent(old, new)[0]}
    assert {"backward_without_eta_evidence", "identity_order_crossing"} <= kinds


def test_equal_position_identity_reordering_has_no_verifier_issues():
    old = frame(1, ((21, 10.0), (19, 10.0)), (10.0, 10.0))
    new = frame(2, ((19, 10.0), (21, 10.0)), (10.0, 10.0))
    issues, _ = compare_adjacent(old, new)
    assert not issues


def test_strict_position_inversion_remains_a_crossing():
    old = frame(1, ((19, 10.0), (21, 11.0)), (10.0, 11.0))
    new = frame(1, ((21, 9.0), (19, 12.0)), (9.0, 12.0))
    issues, _ = compare_adjacent(old, new)
    assert any(issue["kind"] == "identity_order_crossing" for issue in issues)


def test_strict_pair_collapsing_to_reordered_tie_remains_a_crossing():
    old = frame(1, ((19, 10.0), (21, 11.0)), (10.0, 11.0))
    new = frame(2, ((21, 10.0), (19, 10.0)), (10.0, 10.0))
    issues, _ = compare_adjacent(old, new)
    assert any(issue["kind"] == "identity_order_crossing" for issue in issues)


def test_strict_pair_collapsing_to_same_order_tie_is_allowed():
    old = frame(1, ((19, 10.0), (21, 11.0)), (10.0, 11.0))
    new = frame(2, ((19, 10.0), (21, 10.0)), (10.0, 10.0))
    issues, _ = compare_adjacent(old, new)
    assert not any(issue["kind"] == "identity_order_crossing" for issue in issues)


def test_near_tie_is_strict_for_identity_order_crossing():
    old = frame(1, ((19, 10.00), (21, 10.01)), (10.00, 10.01))
    new = frame(2, ((21, 10.00), (19, 10.00)), (10.00, 10.00))
    issues, _ = compare_adjacent(old, new)
    assert any(issue["kind"] == "identity_order_crossing" for issue in issues)


def evidence_frame(pos=(3.5, 7.5), *, age=1.0, eta=(1, 1), brackets=((3, 4), (7, 8))):
    tracks = [SimpleNamespace(operator="KMB", route="A", bound="in", position=p, track_id=i + 1,
                              bracket=brackets[i], eta_minutes=eta[i], eta_arrival_at=None,
                              boundary_age_seconds=age, source_indices=(int(brackets[i][1]),))
              for i, p in enumerate(pos)]
    candidates = [SimpleNamespace(operator="KMB", route="A", bound="in", position=p,
                                  bracket=brackets[i], eta_minutes=eta[i], boundary_age_seconds=age)
                 for i, p in enumerate(pos)]
    return frame_record(SimpleNamespace(
        complete_routes=(SimpleNamespace(
            route_key=KEY, generation=1,
            collected_at=datetime.fromisoformat("2026-01-01T00:00:00+00:00"),
        ),),
        positioning_checkpoints=frozenset(
            ("KMB", "A", "in", index) for index in range(9)
        ),
    ),
        candidates, tracks, {KEY: 20}, "2026-01-01T00:00:00+00:00")


def test_fresh_changed_backward_and_bracket_pairs_are_evidence():
    old, new = evidence_frame(), evidence_frame(pos=(3.0, 7.0), eta=(2, 2))
    assert not compare_adjacent(old, new)[0]
    state = {}
    compare_adjacent(old, new, state)
    assert state["bracket_checks"][KEY] == 1


def test_stale_or_unchanged_motion_is_rejected_and_hold_passes():
    old = evidence_frame()
    stale = evidence_frame(pos=(3.0, 7.5), age=6.0, eta=(2, 1))
    unchanged = evidence_frame(pos=(4.0, 7.5), age=2.0)
    assert any(x["kind"] == "backward_without_eta_evidence" for x in compare_adjacent(old, stale)[0])
    assert any(x["kind"] == "movement_without_eta_evidence" for x in compare_adjacent(old, unchanged)[0])
    assert compare_adjacent(old, evidence_frame())[0] == []


def test_minute_baseline_allows_fresh_eta_snap_but_rejects_stale_snap():
    old = evidence_frame()
    fresh = evidence_frame(pos=(3.0, 7.0), eta=(2, 2), age=1.0)
    fresh["utc"] = "2026-01-01T00:01:00+00:00"
    assert minute_checks(old, fresh) == ([], 2)

    stale = evidence_frame(pos=(3.0, 7.0), eta=(2, 2), age=6.0)
    stale["utc"] = "2026-01-01T00:01:00+00:00"
    assert {issue["kind"] for issue in minute_checks(old, stale)[0]} == {
        "minute_backward"
    }


def test_minute_baseline_restarts_at_intermediate_fresh_backward_snap():
    initial = evidence_frame()
    corrected = evidence_frame(pos=(3.0, 7.0), eta=(2, 2), age=1.0)
    corrected["utc"] = "2026-01-01T00:00:30+00:00"
    cached = evidence_frame(pos=(3.0, 7.0), eta=(2, 2), age=31.0)
    cached["utc"] = "2026-01-01T00:01:00+00:00"
    later = evidence_frame(pos=(3.0, 7.0), eta=(2, 2), age=61.0)
    later["utc"] = "2026-01-01T00:01:30+00:00"
    state = {"last_generation_by_route": {KEY: 1}, "minute_checks": {}}
    baselines = {
        (KEY, track_id): (initial["utc"], position, initial, 1)
        for track_id, position in initial["tracks"][KEY]
    }

    baselines, issues, checks = check_minute_baselines(
        baselines, corrected, evidence_state=state
    )
    assert issues == [] and checks == 0
    baselines, issues, checks = check_minute_baselines(
        baselines, cached, evidence_state=state
    )
    assert issues == [] and checks == 0
    _baselines, issues, checks = check_minute_baselines(
        baselines, later, evidence_state=state
    )
    assert issues == [] and checks == 2


def test_bracket_qualification_rejects_marker_outside_its_boundary():
    invalid = evidence_frame(pos=(2.5, 7.5))
    state = {}
    issues, _ = compare_adjacent(invalid, invalid, state)
    assert any(issue["kind"] == "invalid_bracket_evidence" for issue in issues)
    assert state.get("bracket_checks", {}).get(KEY, 0) == 0
    assert state["bracket_inconclusive"][KEY] == 1


def test_checkpoint_set_is_grouped():
    snap = SimpleNamespace(positioning_checkpoints=frozenset({("KMB", "A", "in", 2)}), complete_routes=())
    from scripts.verify_marker_tracker_live import _observed_checkpoint_map
    assert _observed_checkpoint_map(snap) == {KEY: frozenset({2})}


def test_evaluate_run_requires_freshness_and_actual_evidence():
    assert evaluate_run({KEY}, set(), True, 2, 2, 0) == 2
    assert evaluate_run({KEY}, {KEY}, True, 0, 2, 0) == 2
    assert evaluate_run({KEY}, {KEY}, True, 1, 1, 0, bracket_count=1) == 0
    assert evaluate_run({KEY}, {KEY}, True, 1, 1, 1) == 1
    assert evaluate_run({KEY}, {KEY}, True, 1, 1, 0, {"transit"}) == 2


def test_omission_preserves_generation_identity_state():
    state = {}
    first = frame(1, ((1, 2.0),), (2.0,))
    omitted = frame(1, ((1, 2.0),), (), "2026-01-01T00:00:10+00:00")
    omitted["generations"] = {}
    changed = frame(1, ((2, 2.0),), (), "2026-01-01T00:00:20+00:00")
    compare_adjacent(None, first, state)
    assert compare_adjacent(first, omitted, state)[0] == []
    assert any(x["kind"] in {"identity_change_during_omission", "identity_change_without_generation"}
               for x in compare_adjacent(omitted, changed, state)[0])


def test_route_change_is_never_hidden_by_generation_change():
    old = frame(1, ((1, 2.0),), (2.0,))
    other = SimpleNamespace(route_key=("KMB", "B", "in"), generation=2,
                            collected_at=datetime.fromisoformat("2026-01-01T00:01:00+00:00"))
    current = frame_record(SimpleNamespace(complete_routes=(other,)),
                           [], [SimpleNamespace(operator="KMB", route="B", bound="in", position=2.0, track_id=1)], {})
    assert any(x["kind"] == "identity_route_change" for x in compare_adjacent(old, current)[0])


def test_minute_check_allows_a_legitimate_hold_and_uses_terminus_exemption():
    old = frame(1, ((1, 2.0),), (2.0,), "2026-01-01T00:00:00+00:00")
    stalled = frame(1, ((1, 2.0),), (2.0,), "2026-01-01T00:01:00+00:00")
    assert minute_checks(old, stalled)[0] == []
    at_end = frame_record(SimpleNamespace(complete_routes=(SimpleNamespace(
        route_key=KEY, generation=1, collected_at=datetime.fromisoformat("2026-01-01T00:01:00+00:00")),)),
        [item(20.0)], [item(20.0, 1)], {KEY: 20.0}, "2026-01-01T00:01:00+00:00")
    assert minute_checks(old, at_end)[0] == []


def test_json_safe_counters_and_global_pass_threshold():
    state = {"gap_checks": {KEY: 1}, "minute_checks": {KEY: 1}, "gap_inconclusive": {}}
    assert _json_safe_record({"counters": state})["counters"]["gap_checks"]["KMB/A/in"] == 1
    assert _json_safe_record({"seen": frozenset({2, 1})})["seen"] == [1, 2]
    assert evaluate_run({KEY, ("KMB", "B", "in")}, {KEY}, True, {KEY: 1}, {KEY: 1}, 0) == 2


def test_frame_evidence_records_scheduled_marker_reliability():
    scheduled = item(2.0, 1)
    scheduled.unreliable = True
    record = frame_record(
        SimpleNamespace(complete_routes=()), [scheduled], [scheduled], {KEY: 20.0}
    )
    assert record["candidate_evidence"][KEY][0]["unreliable"] is True
    assert record["track_evidence"][KEY][1]["unreliable"] is True


def test_frame_records_requested_priorities_and_checkpoint_cache_ages():
    rows = (
        SimpleNamespace(
            operator="KMB", route="A", bound="in", index=2,
            cache_age_seconds=3.5,
        ),
        SimpleNamespace(
            operator="KMB", route="A", bound="in", index=3,
            cache_age_seconds=7.0,
        ),
    )
    record = frame_record(
        SimpleNamespace(complete_routes=(), positioning_rows=rows),
        [],
        [],
        priorities={KEY: {2, 3}},
    )
    assert record["priority_checkpoints"][KEY] == [2, 3]
    assert record["checkpoint_ages"][KEY] == {2: 3.5, 3: 7.0}


def test_each_requested_route_needs_active_track_and_direct_evidence():
    other = ("KMB", "B", "in")
    assert evaluate_run({KEY, other}, {KEY, other}, {KEY: 1, other: 0},
                        {KEY: 1, other: 0}, {KEY: 1, other: 1}, 0) == 2


def test_freshness_is_exactly_process_start_and_omitted_baseline_coasts():
    from datetime import timedelta
    start = datetime.fromisoformat("2026-01-01T00:00:00+00:00")
    assert fresh_routes({KEY: start - timedelta(microseconds=1)}, start, start) == set()
    state = {"last_generation_by_route": {KEY: 1}}
    baseline = frame(1, ((1, 2.0),), (2.0,))
    omitted = frame(1, ((1, 2.0),), (), "2026-01-01T00:01:00+00:00")
    omitted["generations"] = {}
    baselines = {(KEY, 1): (baseline["utc"], 2.0, baseline, 1)}
    baselines, _, checks = check_minute_baselines(baselines, omitted, evidence_state=state)
    later = frame(1, ((1, 2.0),), (), "2026-01-01T00:02:00+00:00")
    later["generations"] = {}
    baselines, _, checks2 = check_minute_baselines(baselines, later, evidence_state=state)
    assert checks == 1 and checks2 == 1 and state["minute_checks"][KEY] == 2


def test_omitted_nonterminal_removal_is_a_violation():
    old = frame(1, ((1, 2.0),), (2.0,))
    current = frame(1, (), (), "2026-01-01T00:01:00+00:00")
    current["generations"] = {}
    assert any(x["kind"] == "identity_change_during_omission"
               for x in compare_adjacent(old, current, {} )[0])


def test_stale_terminal_removal_is_allowed_but_recent_is_inconclusive():
    old = frame(1, ((1, 20.0),), (20.0,))
    current = frame(1, (), (), "2026-01-01T00:16:00+00:00")
    current["generations"] = {}
    state = {}
    assert compare_adjacent(old, current, state)[0] == []
    recent = frame(1, (), (), "2026-01-01T00:01:00+00:00")
    recent["generations"] = {}
    state = {}
    assert compare_adjacent(old, recent, state)[0] == []
    assert state["lifecycle_inconclusive"][KEY] == 1


def test_lifecycle_inconclusive_blocks_final_pass():
    assert evaluate_run({KEY}, {KEY}, {KEY: 1}, {KEY: 1}, {KEY: 1}, 0,
                        lifecycle_inconclusive={KEY: 1}) == 2


def test_motion_baseline_survives_generation_turnover_for_same_id():
    old = frame(1, ((1, 2.0),), (2.0,), "2026-01-01T00:00:00+00:00")
    new = frame(2, ((1, 3.0),), (3.0,), "2026-01-01T00:01:00+00:00")
    assert minute_checks(old, new)[1] == 1
    replacement = frame(3, ((9, 3.0),), (3.0,), "2026-01-01T00:01:00+00:00")
    assert minute_checks(old, replacement)[1] == 0
    state = {"last_generation_by_route": {KEY: 1}, "minute_checks": {}}
    baselines = {(KEY, 1): (old["utc"], 2.0, old, 1)}
    for stamp in ("2026-01-01T00:01:00+00:00", "2026-01-01T00:02:00+00:00"):
        current = frame(2 if stamp.endswith("01:00+00:00") else 3,
                        ((1, 3.0),), (), stamp)
        current["generations"] = {KEY: (2 if stamp.endswith("01:00+00:00") else 3, stamp)}
        baselines, _, _ = check_minute_baselines(baselines, current, evidence_state=state)
    assert state["minute_checks"][KEY] == 2
