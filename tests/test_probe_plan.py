"""Focused coverage for the pure adaptive probe sampling planner."""

from dataclasses import replace

import pytest

from dashboard.providers.probe_plan import adaptive_query_units
from dashboard.providers.route_geometry import ProbeStop
from dashboard.providers.transit import ProbeEta


def _probes(operator="GMB", route="X", bound="outbound", count=7):
    return [
        ProbeStop(operator, route, bound, f"{route}-{bound}-{index}", 1, 1, index)
        for index in range(count)
    ]


def _key(probe):
    return (probe.operator, probe.route, probe.bound, probe.index)


def _route(probe):
    return (probe.operator, probe.route, probe.bound)


def _group(probe):
    if probe.operator == "KMB":
        return f"KMB:route:{probe.route}"
    if probe.operator == "CTB":
        return f"CTB:{probe.stop_id}:{probe.route}"
    return f"{probe.operator}:stop:{probe.stop_id}"


def _row(probe, minutes, *, age=0.0):
    return ProbeEta(
        probe.operator,
        probe.route,
        probe.bound,
        probe.stop_id,
        probe.index,
        minutes,
        cache_age_seconds=age,
    )


def _plan(probes, rows=(), pairs=None, attempted=()):
    return adaptive_query_units(
        probes,
        dict(rows),
        pairs,
        set(attempted),
        group_key=_group,
    )


def test_termini_are_the_only_initial_units():
    ctb = _probes("CTB", "A", "outbound", 5)
    gmb = _probes("GMB", "B", "seq-1", 6)
    pairs = {_route(ctb[0]): ((1, 2),), _route(gmb[0]): ((2, 3),)}

    assert _plan([*ctb, *gmb], pairs=pairs) == [
        (_group(ctb[-1]),),
        (_group(gmb[-1]),),
    ]


def test_third_terminal_row_starts_upstream_search_after_terminus_response():
    probes = _probes(count=9)
    terminal = probes[-1]
    pairs = {_route(terminal): ((0, 1),)}
    initial_attempts = {_group(probes[0]), _group(probes[1])}

    assert _plan(probes, pairs=pairs, attempted=initial_attempts) == [
        (_group(terminal),)
    ]

    rows = {
        _key(terminal): (_row(terminal, 1), _row(terminal, 3), _row(terminal, 7))
    }
    assert _plan(
        probes,
        rows.items(),
        pairs,
        [*initial_attempts, _group(terminal)],
    ) == [(_group(probes[4]),)]


@pytest.mark.parametrize(
    ("target_state", "target_attempted", "expected"),
    [
        ("missing", False, True),
        ("unsaturated", False, False),
        ("successful-empty", False, False),
        ("missing", True, False),
    ],
    ids=("missing", "unsaturated", "successful-empty", "attempted-failure"),
)
def test_upstream_search_distinguishes_missing_empty_unsaturated_and_failed(
    target_state, target_attempted, expected
):
    probes = _probes(count=7)
    terminal = probes[-1]
    target = probes[3]
    pairs = {_route(terminal): ((0, 1),)}
    rows = {
        _key(terminal): (_row(terminal, 1), _row(terminal, 2), _row(terminal, 6))
    }
    if target_state == "unsaturated":
        rows[_key(target)] = (_row(target, 2), _row(target, 4))
    elif target_state == "successful-empty":
        rows[_key(target)] = ()
    attempted = {_group(terminal), _group(probes[0]), _group(probes[1])}
    if target_attempted:
        attempted.add(_group(target))

    result = _plan(probes, rows.items(), pairs, attempted)
    assert result == ([(_group(target),)] if expected else [])


def test_repeated_saturated_checkpoints_walk_to_the_next_missing_stop():
    probes = _probes(count=11)
    terminal = probes[-1]
    pairs = {_route(terminal): ((0, 1),)}
    rows = {
        _key(terminal): (_row(terminal, 1), _row(terminal, 2), _row(terminal, 4)),
        _key(probes[8]): (_row(probes[8], 1), _row(probes[8], 2), _row(probes[8], 2)),
        _key(probes[7]): (_row(probes[7], 0), _row(probes[7], 1), _row(probes[7], 0)),
    }
    attempted = {
        _group(terminal),
        _group(probes[8]),
        _group(probes[7]),
        _group(probes[0]),
        _group(probes[1]),
    }

    assert _plan(probes, rows.items(), pairs, attempted) == [(_group(probes[6]),)]


def test_current_two_and_three_stop_neighbour_units_stay_intact():
    probes = _probes(count=8)
    terminal = probes[-1]
    pairs = {_route(terminal): ((2, 3), (4, 5, 6))}

    assert _plan(
        probes,
        [(_key(terminal), ())],
        pairs,
        [_group(terminal)],
    ) == [
        (_group(probes[2]), _group(probes[3])),
        (_group(probes[4]), _group(probes[5]), _group(probes[6])),
    ]


def test_partially_attempted_neighbour_unit_returns_only_remaining_groups():
    probes = _probes(count=8)
    terminal = probes[-1]
    pairs = {_route(terminal): ((2, 3, 4),)}

    assert _plan(
        probes,
        [(_key(terminal), ())],
        pairs,
        [_group(terminal), _group(probes[3])],
    ) == [(_group(probes[2]), _group(probes[4]))]


def test_exact_coarse_terminal_projection_seeds_a_three_stop_bracket():
    probes = _probes(count=8)
    terminal = probes[-1]

    assert _plan(
        probes,
        [(_key(terminal), (_row(terminal, 6),))],
        attempted=[_group(terminal)],
    ) == [(_group(probes[3]), _group(probes[4]), _group(probes[5]))]


def test_kmb_route_group_is_deduplicated_across_directions_and_already_served():
    outbound = _probes("KMB", "91", "outbound", 5)
    inbound = _probes("KMB", "91", "inbound", 5)
    group = _group(outbound[0])
    pairs = {
        _route(outbound[0]): ((1, 2),),
        _route(inbound[0]): ((1, 2),),
    }

    assert _plan([*outbound, *inbound], pairs=pairs) == [(group,)]

    rows = {
        _key(outbound[-1]): (_row(outbound[-1], 1), _row(outbound[-1], 2), _row(outbound[-1], 4)),
        _key(inbound[-1]): (_row(inbound[-1], 1), _row(inbound[-1], 2), _row(inbound[-1], 4)),
    }
    assert _plan([*outbound, *inbound], rows.items(), pairs, [group]) == []


def test_route_direction_and_stop_occurrence_rows_do_not_cross():
    outbound = _probes("CTB", "X", "outbound", 3)
    inbound = _probes("CTB", "X", "inbound", 3)
    pairs = {
        _route(outbound[0]): ((1, 2),),
        _route(inbound[0]): ((1, 2),),
    }
    attempted = {
        _group(outbound[1]), _group(outbound[2]),
        _group(inbound[1]), _group(inbound[2]),
    }
    rows = {
        _key(outbound[-1]): (),
        _key(inbound[-1]): (_row(inbound[-1], 1), _row(inbound[-1], 2), _row(inbound[-1], 4)),
    }

    assert _plan([*outbound, *inbound], rows.items(), pairs, attempted) == [
        (_group(inbound[0]),)
    ]


@pytest.mark.parametrize(
    "target_rows",
    [
        lambda probe: (_row(probe, 2, age=60),),
        lambda probe: (replace(_row(probe, 2), minutes=None),),
    ],
    ids=("stale", "invalid"),
)
def test_stale_or_invalid_checkpoint_is_refreshed_instead_of_treated_as_empty(target_rows):
    probes = _probes(count=7)
    terminal = probes[-1]
    target = probes[3]
    pairs = {_route(terminal): ((0, 1),)}
    rows = {
        _key(terminal): (_row(terminal, 1), _row(terminal, 2), _row(terminal, 6)),
        _key(target): target_rows(target),
    }
    attempted = {_group(terminal), _group(probes[0]), _group(probes[1])}

    assert _plan(probes, rows.items(), pairs, attempted) == [(_group(target),)]


def test_discovery_and_current_neighbours_are_deterministically_interleaved():
    first = _probes("GMB", "A", "seq-1", 7)
    second = _probes("GMB", "B", "seq-1", 7)
    pairs = {
        _route(first[0]): ((1, 2),),
        _route(second[0]): ((1, 2),),
    }
    rows = {
        _key(first[-1]): (_row(first[-1], 1), _row(first[-1], 2), _row(first[-1], 12)),
        _key(second[-1]): (_row(second[-1], 1), _row(second[-1], 2), _row(second[-1], 12)),
    }
    attempted = {_group(first[-1]), _group(second[-1])}

    assert _plan([*second, *first], rows.items(), pairs, attempted) == [
        (_group(first[0]),),
        (_group(first[1]), _group(first[2])),
        (_group(second[0]),),
        (_group(second[1]), _group(second[2])),
    ]
