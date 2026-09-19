"""Pure adaptive planning for bounded probe-ETA sweeps.

The transport provider owns request accounting and budgets.  This module only
turns already-observed checkpoint rows into ordered physical request units.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from math import ceil, floor, isfinite
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from dashboard.providers.route_geometry import ProbeStop
    from dashboard.providers.transit import ProbeEta


RouteKey: TypeAlias = tuple[str, str, str]
CheckpointKey: TypeAlias = tuple[str, str, str, int]
QueryUnit: TypeAlias = tuple[str, ...]

_FRESH_SECONDS = 60.0


def adaptive_query_units(
    probes: Sequence[ProbeStop],
    rows_by_checkpoint: Mapping[CheckpointKey, Sequence[ProbeEta]],
    neighbour_pairs: Mapping[RouteKey, Sequence[Sequence[int]]] | None,
    attempted_groups: Collection[str],
    *,
    group_key: Callable[[ProbeStop], str],
) -> list[QueryUnit]:
    """Return unattempted physical probe units in deterministic order.

    A sweep starts with its final termini.  Once every terminal physical group
    was attempted, fresh saturated ETA lists can search upstream for the next
    missing or stale checkpoint.  Current marker-bracketing units are then
    interleaved with that search work; this function deliberately does not
    apply any request budget.
    """
    groups_by_route = _groups_by_route(probes, group_key)
    if not groups_by_route:
        return []

    attempted = {str(group) for group in attempted_groups}
    routes = sorted(groups_by_route)
    terminal_indices = {
        route: max(groups_by_route[route])
        for route in routes
        if groups_by_route[route]
    }

    terminal_units = []
    for route in routes:
        terminal = terminal_indices.get(route)
        if terminal is None:
            continue
        remaining = _unattempted(
            groups_by_route[route][terminal], attempted
        )
        if remaining:
            terminal_units.append(remaining)
    if terminal_units:
        return _unique_units(terminal_units)

    caller_units, caller_routes = _caller_neighbour_units(
        groups_by_route, neighbour_pairs, attempted
    )
    seed_units = _seed_neighbour_units(
        groups_by_route,
        terminal_indices,
        rows_by_checkpoint,
        attempted,
        caller_routes,
    )
    discovery_units = [
        unit
        for route in routes
        if (unit := _upstream_discovery_unit(
            route,
            groups_by_route[route],
            terminal_indices[route],
            rows_by_checkpoint,
            attempted,
        ))
    ]

    return _interleave_unique(discovery_units, [*caller_units, *seed_units])


def _groups_by_route(
    probes: Sequence[ProbeStop], group_key: Callable[[ProbeStop], str]
) -> dict[RouteKey, dict[int, QueryUnit]]:
    grouped: dict[RouteKey, dict[int, list[str]]] = {}
    for probe in probes:
        route = _route_key(probe)
        index = _integer(getattr(probe, "index", None))
        if index is None or index < 0:
            continue
        group = str(group_key(probe))
        by_index = grouped.setdefault(route, {}).setdefault(index, [])
        if group not in by_index:
            by_index.append(group)
    return {
        route: {
            index: tuple(sorted(groups))
            for index, groups in by_index.items()
        }
        for route, by_index in grouped.items()
    }


def _route_key(probe: ProbeStop) -> RouteKey:
    return (
        str(getattr(probe, "operator", "")),
        str(getattr(probe, "route", "")),
        str(getattr(probe, "bound", "")),
    )


def _caller_neighbour_units(
    groups_by_route: Mapping[RouteKey, Mapping[int, QueryUnit]],
    neighbour_pairs: Mapping[RouteKey, Sequence[Sequence[int]]] | None,
    attempted: Collection[str],
) -> tuple[list[QueryUnit], set[RouteKey]]:
    supplied: dict[RouteKey, list[Sequence[int]]] = {}
    for raw_route, raw_units in (neighbour_pairs or {}).items():
        route = _normalised_route_key(raw_route)
        if route is None or route not in groups_by_route:
            continue
        try:
            supplied.setdefault(route, []).extend(raw_units)
        except TypeError:
            continue

    units: list[QueryUnit] = []
    caller_routes: set[RouteKey] = set()
    for route in sorted(supplied):
        for raw_unit in supplied[route]:
            indices = _neighbour_indices(raw_unit)
            if not indices:
                continue
            full_unit = _unit_for_indices(groups_by_route[route], indices)
            if not full_unit:
                continue
            caller_routes.add(route)
            remaining = _unattempted(full_unit, attempted)
            if remaining:
                units.append(remaining)
    return _unique_units(units), caller_routes


def _seed_neighbour_units(
    groups_by_route: Mapping[RouteKey, Mapping[int, QueryUnit]],
    terminal_indices: Mapping[RouteKey, int],
    rows_by_checkpoint: Mapping[CheckpointKey, Sequence[ProbeEta]],
    attempted: Collection[str],
    caller_routes: Collection[RouteKey],
) -> list[QueryUnit]:
    units: list[QueryUnit] = []
    for route in sorted(groups_by_route):
        if route in caller_routes:
            continue
        terminal = terminal_indices[route]
        state, minutes = _checkpoint_state(
            rows_by_checkpoint, (*route, terminal)
        )
        if state not in {"unsaturated", "saturated"}:
            continue
        for minute in minutes:
            projected = terminal - max(0.0, minute) / 2.0
            lower = max(0, min(terminal, floor(projected)))
            upper = max(0, min(terminal, ceil(projected)))
            indices = (
                tuple(index for index in (lower - 1, lower, lower + 1)
                      if 0 <= index <= terminal)
                if lower == upper else (lower, upper)
            )
            full_unit = _unit_for_indices(
                groups_by_route[route], indices
            )
            if (remaining := _unattempted(full_unit, attempted)):
                units.append(remaining)
    return _unique_units(units)


def _upstream_discovery_unit(
    route: RouteKey,
    groups_by_index: Mapping[int, QueryUnit],
    terminal: int,
    rows_by_checkpoint: Mapping[CheckpointKey, Sequence[ProbeEta]],
    attempted: Collection[str],
) -> QueryUnit | None:
    state, minutes = _checkpoint_state(rows_by_checkpoint, (*route, terminal))
    if state != "saturated":
        return None

    current = terminal
    current_minutes = minutes
    visited: set[int] = set()
    while current > 0:
        distance = ceil(max(0.0, current_minutes[-1]) / 2.0)
        target = max(0, min(current - 1, current - distance))
        if target in visited or target not in groups_by_index:
            return None
        visited.add(target)

        full_unit = _unit_for_indices(groups_by_index, (target,))
        state, minutes = _checkpoint_state(rows_by_checkpoint, (*route, target))
        if state == "saturated":
            current = target
            current_minutes = minutes
            continue
        if state in {"missing", "invalid"}:
            return _unattempted(full_unit, attempted) or None
        # An observed empty response and an unsaturated valid list both stop
        # the search.  Neither tells us to fabricate a further checkpoint.
        return None
    return None


def _checkpoint_state(
    rows_by_checkpoint: Mapping[CheckpointKey, Sequence[ProbeEta]],
    key: CheckpointKey,
) -> tuple[str, tuple[float, ...]]:
    if key not in rows_by_checkpoint:
        return "missing", ()
    rows = rows_by_checkpoint[key]
    if not isinstance(rows, (list, tuple)):
        return "invalid", ()
    if not rows:
        return "empty", ()

    minutes: list[float] = []
    for row in rows:
        minute = _valid_minutes(row, key)
        if minute is None:
            return "invalid", ()
        minutes.append(minute)
    return ("saturated" if len(minutes) >= 3 else "unsaturated"), tuple(sorted(minutes))


def _valid_minutes(row: ProbeEta, key: CheckpointKey) -> float | None:
    operator, route, bound, index = key
    if _route_key(row) != (operator, route, bound):
        return None
    if _integer(getattr(row, "index", None)) != index:
        return None
    minutes = _finite_number(getattr(row, "minutes", None))
    age = _finite_number(getattr(row, "cache_age_seconds", 0.0))
    if minutes is None or age is None or not 0.0 <= age < _FRESH_SECONDS:
        return None
    return minutes


def _unit_for_indices(
    groups_by_index: Mapping[int, QueryUnit], indices: Sequence[int]
) -> QueryUnit:
    groups: list[str] = []
    for index in indices:
        indexed_groups = groups_by_index.get(index)
        if not indexed_groups:
            return ()
        for group in indexed_groups:
            if group not in groups:
                groups.append(group)
    return tuple(groups)


def _unattempted(unit: QueryUnit, attempted: Collection[str]) -> QueryUnit:
    return tuple(group for group in unit if group not in attempted)


def _interleave_unique(
    discovery: Sequence[QueryUnit], neighbours: Sequence[QueryUnit]
) -> list[QueryUnit]:
    ordered: list[QueryUnit] = []
    seen: set[frozenset[str]] = set()
    for index in range(max(len(discovery), len(neighbours))):
        if index < len(discovery):
            _append_unique(ordered, seen, discovery[index])
        if index < len(neighbours):
            _append_unique(ordered, seen, neighbours[index])
    return ordered


def _unique_units(units: Sequence[QueryUnit]) -> list[QueryUnit]:
    ordered: list[QueryUnit] = []
    seen: set[frozenset[str]] = set()
    for unit in units:
        _append_unique(ordered, seen, unit)
    return ordered


def _append_unique(
    ordered: list[QueryUnit], seen: set[frozenset[str]], unit: QueryUnit
) -> None:
    if not unit:
        return
    identity = frozenset(unit)
    if identity not in seen:
        seen.add(identity)
        ordered.append(unit)


def _normalised_route_key(value: object) -> RouteKey | None:
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        return None
    return tuple(str(part) for part in value)  # type: ignore[return-value]


def _neighbour_indices(value: object) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)):
        return ()
    try:
        raw_indices = tuple(value)  # type: ignore[arg-type]
    except TypeError:
        return ()
    indices: list[int] = []
    for raw_index in raw_indices:
        index = _integer(raw_index)
        if index is None or index < 0:
            return ()
        if index not in indices:
            indices.append(index)
    return tuple(indices) if len(indices) >= 2 else ()


def _integer(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    number = _finite_number(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return None
    return number if isfinite(number) else None
