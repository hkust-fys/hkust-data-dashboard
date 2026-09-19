"""Local ETA interpolation, after vehicle cohorts have been associated.

One marker's countdown is a distance fraction only when divided by the travel
time of the same adjacent stop interval. Other arrivals at those two stops can
measure that interval even after this vehicle disappears from the upstream list.
"""

from __future__ import annotations

import math
from dataclasses import replace
from itertools import combinations
from statistics import median

FRESH_SECONDS = 60.0


def marker_query_units(estimates, route_lines):
    """Actual local boundaries and unresolved zero spans, kept as query units."""
    terminals = {_key(line): len(line.stops) - 1 for line in route_lines}
    requested = {}
    for marker in estimates:
        key = _key(marker)
        terminal = terminals.get(key)
        units = requested.setdefault(key, [])
        if marker.query_stops:
            units.append(tuple(marker.query_stops))
        elif marker.query_pair:
            units.append(tuple(marker.query_pair))
        if not marker.query_stops and marker.position is not None and math.isfinite(marker.position):
            lower, upper = math.floor(marker.position), math.ceil(marker.position)
            nearby = (((lower, upper),) if lower != upper
                      else ((lower - 1, lower), (lower, lower + 1)))
            if marker.segment_minutes is None:
                # Test the coarse search location before walking every stop
                # backwards from a distant terminus observation.
                units[0:0] = nearby
            else:
                units.extend(nearby)
        requested[key] = list(dict.fromkeys(
            unit for unit in units if len(set(unit)) > 1 and min(unit) >= 0
            and (terminal is None or max(unit) <= terminal)
        ))[:32]
    return {key: tuple(units) for key, units in requested.items() if units}


def _key(row):
    return (str(getattr(row, "operator_code", "") or row.operator),
            str(row.route), str(row.bound))


def _arrival(row):
    """Compare absolute arrivals, including staggered rounded-only responses."""
    arrival = getattr(row, "arrival_at", None)
    if arrival is not None:
        return arrival.timestamp()
    observed = getattr(row, "observed_at", None)
    if observed is not None and row.minutes is not None:
        return observed.timestamp() + float(row.minutes) * 60
    return None


def _fresh(row):
    try:
        age = float(row.cache_age_seconds)
        return (0 <= age < FRESH_SECONDS and int(row.refresh_generation) > 0
                and (row.minutes is None or math.isfinite(float(row.minutes))))
    except (AttributeError, TypeError, ValueError, OverflowError):
        return False


def _countdown(row):
    signed = getattr(row, "signed_minutes", None)
    return float(row.minutes if signed is None else signed)


def _observation_clock(row):
    return _arrival(row) - _countdown(row) * 60


def _match_arrivals(lower, upper):
    """Align whole ETA lists; ranks can shift when a bus passes the lower stop.

    Maximize matched arrivals, then minimize forward travel time. Using one
    ordered match per row prevents comparing a leading bus to the next bus's
    headway. The median uses all matched differences without one outlier
    dominating every marker. ETA lists are small; cap malformed long lists.
    """
    def prepared(rows):
        return sorted(
            ((value, index, row) for index, row in enumerate(rows)
             if row.minutes is not None and (value := _arrival(row)) is not None
             and math.isfinite(value)), key=lambda item: item[:2],
        )[:8]

    left, right = prepared(lower), prepared(upper)
    for count in range(min(len(left), len(right)), 0, -1):
        matches = []
        for a in combinations(left, count):
            for b in combinations(right, count):
                deltas = tuple((y[0] - x[0]) / 60 for x, y in zip(a, b, strict=True))
                if all(delta > 1e-6 for delta in deltas):
                    matches.append((deltas, a, b))
        if matches:
            deltas, a, b = min(matches, key=lambda item: (sum(item[0]), item[0]))
            # Rounded zeroes are interval observations, not travel-time samples.
            measured = [delta for delta, x, y in zip(deltas, a, b, strict=True)
                        if not any(_rounded(row) and _countdown(row) == 0
                                   for row in (x[2], y[2]))]
            return (median(measured) if measured else None,
                    {id(y[2]): x[2] for x, y in zip(a, b, strict=True)})
    return None, {}


def _rounded(row):
    return (getattr(row, "signed_minutes", None) is None
            or bool(getattr(row, "countdown_rounded", False)))


def _zero_span(estimate, owned, checkpoints, terminal, route_lines, reproject):
    """Fit rounded-zero intervals against travel times and nearby nonzero ETAs."""
    zeroes = sorted({int(row.index) for row in owned
                     if _rounded(row) and _countdown(row) == 0})
    if not zeroes:
        return None
    start, end = zeroes[0], zeroes[0]
    for index in zeroes[1:]:
        if index != end + 1:
            break
        end = index
    by_index = {int(row.index): row for row in owned}
    first = start - 1 if start - 1 in by_index else start
    last = end + 1 if end + 1 in by_index else end
    offsets = {first: 0.0}
    for index in range(first, last):
        lower, upper = checkpoints.get(index), checkpoints.get(index + 1)
        if not lower or not upper:
            break
        travel, _matches = _match_arrivals(lower, upper)
        if travel is None:
            break
        offsets[index + 1] = offsets[index] + travel
    position = (start + end) / 2
    bracket, segment, remaining = (float(start), float(end)), None, None
    confirmed = False
    if last > first and last in offsets:
        reference = max(_arrival(row) - _countdown(row) * 60
                        for row in owned if int(row.index) in offsets)
        constraints = []
        for index, row in by_index.items():
            if index not in offsets:
                continue
            value = (_arrival(row) - reference) / 60 - offsets[index]
            radius = .5 if _rounded(row) else 0.0
            constraints.append((value - radius, value + radius))
        lo = max(item[0] for item in constraints)
        hi = min(item[1] for item in constraints)
        if lo <= hi + 1e-6:
            centre = (lo + hi) / 2
            for upper in range(first + 1, last + 1):
                lower = upper - 1
                a, b = centre + offsets[lower], centre + offsets[upper]
                if a <= 0 <= b and b > a:
                    segment, remaining = b - a, b
                    position = upper - b / segment
                    bracket = (float(lower), float(upper))
                    confirmed = hi + offsets[lower] <= 0 <= lo + offsets[upper]
                    break
    wanted = tuple(range(max(0, min(first, start - 1)), min(terminal, max(last, end + 1)) + 1))
    updated = reproject(estimate, position, route_lines)
    return replace(updated, bracket=bracket, query_stops=wanted,
                   query_pair=None, segment_minutes=segment, eta_minutes=remaining,
                   eta_arrival_at=None, bracket_initial_eta=remaining,
                   bracket_eta_offsets=None, boundary_revision=None,
                   position_authoritative=confirmed, priority_indices=frozenset(wanted))


def refine_adjacent_positions(estimates, rows, route_lines, *, reproject):
    """Keep the population intact and refine from queried adjacent stops only.

    A missing pair becomes an explicit polling request. Full-route responses
    can walk upstream through already queried pairs; per-stop feeds acquire
    the next needed pair on their bounded refresh queue. No wall-clock aging
    or fixed minutes-per-stop value supplies an interpolation denominator.
    """
    by_route = {}
    for row in rows:
        if _fresh(row):
            by_route.setdefault(_key(row), {}).setdefault(int(row.index), []).append(row)
    terminals = {_key(line): len(line.stops) - 1 for line in route_lines}
    travel_cache = {}
    owners = {}
    for marker_index, marker in enumerate(estimates):
        for kind, row_index in marker.source_observations:
            if kind == "probe" and 0 <= row_index < len(rows):
                owners.setdefault(id(rows[row_index]), set()).add(marker_index)
    output = []
    for marker_index, estimate in enumerate(estimates):
        key = _key(estimate)
        checkpoints = by_route.get(key, {})
        owned = [rows[index] for kind, index in estimate.source_observations
                 if kind == "probe" and 0 <= index < len(rows)
                 and rows[index].minutes is not None and _fresh(rows[index])
                 and _key(rows[index]) == key]
        # Old/injected observations without clocks retain their coarse result.
        # They cannot manufacture a measured local travel time.
        owned = [row for row in owned if _arrival(row) is not None]
        if not owned:
            output.append(estimate)
            continue
        zero_fit = _zero_span(estimate, owned, checkpoints, terminals.get(key, 0),
                             route_lines, reproject)
        if zero_fit is not None:
            output.append(zero_fit)
            continue
        clocks = [_observation_clock(row) for row in owned]
        reference = max(clocks)
        staggered = reference - min(clocks) > 1.0

        def at_reference(row, reference=reference, staggered=staggered):
            return (_arrival(row) - reference) / 60 if staggered else _countdown(row)

        last_passed = max((int(row.index) for row in owned
                           if at_reference(row) < 0), default=-1)
        future = [row for row in owned
                  if int(row.index) > last_passed and at_reference(row) >= 0]
        if not future:
            terminal = terminals.get(key, last_passed)
            pair = (last_passed, min(last_passed + 1, terminal))
            output.append(replace(estimate, query_pair=pair))
            continue
        anchor = min(future, key=lambda row: (int(row.index), row.cache_age_seconds))
        upper = int(anchor.index)
        remaining = at_reference(anchor)
        rounded = _rounded(anchor)
        refined = estimate
        walked = False
        interpolated = False
        while upper > 0:
            lower = upper - 1
            pair = (lower, upper)
            refined = replace(refined, query_pair=pair)
            lower_rows, upper_rows = checkpoints.get(lower), checkpoints.get(upper)
            if not lower_rows or not upper_rows:
                break
            cache_key = (*key, lower, upper)
            if cache_key not in travel_cache:
                travel_cache[cache_key] = _match_arrivals(lower_rows, upper_rows)
            travel, matched = travel_cache[cache_key]
            # An exact signed crossing for this vehicle is stronger than an
            # average of later vehicles' travel times. Preserve that zero rule.
            crossing = estimate.bracket_eta_offsets
            exact_crossing = (estimate.bracket == (lower, upper) and crossing is not None
                              and crossing[0] <= 0 < crossing[1] and not staggered)
            upstream = matched.get(id(anchor))
            if (upstream is not None and not exact_crossing
                    and owners.get(id(upstream), set()) - {marker_index}):
                # Peer arrivals measure speed but cannot transfer another
                # current cohort's position/identity to this marker.
                break
            upstream_offset = (remaining - (_arrival(anchor) - _arrival(upstream)) / 60
                               if upstream is not None else None)
            if upstream is not None and upstream_offset > 0 and not exact_crossing:
                # This exact ETA is still listed ahead of the upstream stop.
                # Walk the observed match, not an assumed distance or ETA rank.
                anchor = upstream
                upper = lower
                remaining = upstream_offset
                rounded = _rounded(anchor)
                walked = True
                continue
            if exact_crossing:
                travel = crossing[1] - crossing[0]
                remaining = crossing[1]
            elif upstream is not None and not _rounded(upstream) and upstream_offset <= 0:
                travel = (_arrival(anchor) - _arrival(upstream)) / 60
                crossing = (remaining - travel, remaining)
                exact_crossing = travel > 0 and crossing[0] <= 0
            lower_arrivals = [_arrival(row) for row in lower_rows
                              if row.minutes is not None and _arrival(row) is not None]
            leading = (upstream is None and (not lower_arrivals
                       or _arrival(anchor) < min(lower_arrivals)))
            # An unlisted later row can be hidden behind a three-row response.
            # Only a leading absent row or a signed crossing delimits this bus.
            if not exact_crossing and not leading:
                break
            if travel is None:
                break
            if travel > 0:
                position = upper - min(1.0, max(0.0, remaining) / travel)
                # Rounded zero is centred on the stop, not evidence that the
                # vehicle passed it. Keep both sides in the refresh frontier.
                frontier = {lower, upper}
                if rounded and remaining == 0 and upper < terminals.get(key, upper):
                    frontier.add(upper + 1)
                refined = reproject(refined, position, route_lines)
                refined = replace(
                    refined, bracket=(float(lower), float(upper)),
                    eta_minutes=remaining, bracket_initial_eta=remaining,
                    eta_arrival_at=anchor.arrival_at if upper == anchor.index else None,
                    bracket_eta_offsets=crossing if exact_crossing else None,
                    boundary_age_seconds=max(row.cache_age_seconds
                                             for row in (*lower_rows, *upper_rows)),
                    boundary_revision=(max(row.refresh_generation for row in lower_rows),
                                       max(row.refresh_generation for row in upper_rows)),
                    priority_indices=frozenset(frontier), exploratory_indices=frozenset(),
                    position_authoritative=not rounded,
                    segment_minutes=travel,
                )
                interpolated = True
                break
            break
        else:
            refined = replace(refined, query_pair=(0, min(1, terminals.get(key, 1))))
        if not interpolated and (walked or float(estimate.position) > upper
                                 or (rounded and remaining == 0)):
            # Evidence says the bus is upstream of this stop, but the next
            # interval is still missing. Do not extrapolate its speed across it.
            position = float(upper) if rounded and remaining == 0 else min(
                float(estimate.position), float(upper))
            refined = reproject(refined, position, route_lines)
            refined = replace(refined, bracket=None, boundary_revision=None,
                              bracket_eta_offsets=None, position_authoritative=False)
        output.append(refined)
    return output
