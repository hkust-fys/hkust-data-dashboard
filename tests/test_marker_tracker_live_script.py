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


def test_spacing_requires_equal_complete_cardinality():
    old = frame(1, ((1, 1.0), (2, 5.0)), (1.0, 5.0))
    bad = frame(1, ((1, 1.0), (2, 2.0)), (1.0, 5.0))
    assert any(x["kind"] == "spacing_mismatch" for x in compare_adjacent(old, bad)[0])
    unequal = frame(1, ((1, 1.0),), (1.0, 5.0))
    assert not any(x["kind"] == "spacing_mismatch" for x in compare_adjacent(old, unequal)[0])


def test_backward_and_crossing_violations():
    old = frame(1, ((1, 2.0), (2, 5.0)), (2.0, 5.0))
    new = frame(1, ((2, 1.0), (1, 4.0)), (1.0, 4.0))
    kinds = {x["kind"] for x in compare_adjacent(old, new)[0]}
    assert {"backward", "identity_order_crossing"} <= kinds


def test_evaluate_run_requires_freshness_and_actual_evidence():
    assert evaluate_run({KEY}, set(), True, 2, 2, 0) == 2
    assert evaluate_run({KEY}, {KEY}, True, 0, 2, 0) == 2
    assert evaluate_run({KEY}, {KEY}, True, 1, 1, 0) == 0
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


def test_minute_check_requires_mature_observation_and_uses_terminus_exemption():
    old = frame(1, ((1, 2.0),), (2.0,), "2026-01-01T00:00:00+00:00")
    stalled = frame(1, ((1, 2.0),), (2.0,), "2026-01-01T00:01:00+00:00")
    assert minute_checks(old, stalled)[0][0]["kind"] == "minute_stalled"
    at_end = frame_record(SimpleNamespace(complete_routes=(SimpleNamespace(
        route_key=KEY, generation=1, collected_at=datetime.fromisoformat("2026-01-01T00:01:00+00:00")),)),
        [item(20.0)], [item(20.0, 1)], {KEY: 20.0}, "2026-01-01T00:01:00+00:00")
    assert minute_checks(old, at_end)[0] == []


def test_json_safe_counters_and_global_pass_threshold():
    state = {"gap_checks": {KEY: 1}, "minute_checks": {KEY: 1}, "gap_inconclusive": {}}
    assert _json_safe_record({"counters": state})["counters"]["gap_checks"]["KMB/A/in"] == 1
    assert evaluate_run({KEY, ("KMB", "B", "in")}, {KEY}, True, {KEY: 1}, {KEY: 1}, 0) == 2


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
