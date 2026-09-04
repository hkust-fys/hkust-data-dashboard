"""Route-generic temporal identity for estimated map markers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from math import inf, isfinite

from dashboard.maps.positions import reproject_estimate

MAX_ROUTES = 64
MAX_TRACKS_PER_ROUTE = 128
MATCH_DISTANCE = 3.5
MAX_PRIORITY_ENDPOINTS_PER_ROUTE = 32


@dataclass
class _Track:
    track_id: int
    estimate: object
    position: float
    generation: int
    last_evidence_at: float = 0.0
    boundary_observed_at: float | None = None
    boundary_revision: tuple[int, int] | None = None


def _key(item):
    return (
        str(getattr(item, "operator_code", "") or getattr(item, "operator", "")),
        str(getattr(item, "route", "")),
        str(getattr(item, "bound", "")),
    )


class MarkerTracker:
    """Maintain bounded, ordered identities over complete probe generations."""

    def __init__(self, *, max_routes=MAX_ROUTES, max_tracks_per_route=MAX_TRACKS_PER_ROUTE,
                 evidence_ttl_seconds=900):
        self.max_routes = _positive_int(max_routes, "max_routes")
        self.max_tracks_per_route = _positive_int(
            max_tracks_per_route, "max_tracks_per_route"
        )
        self.evidence_ttl_seconds = _nonnegative_finite(
            evidence_ttl_seconds, "evidence_ttl_seconds"
        )
        self._routes = {}
        self._generations = {}
        self._terminal_indices = {}
        self._next_id = 1
        self._lock = asyncio.Lock()

    async def update(self, snapshot, candidates, route_lines=()):
        async with self._lock:
            return self._update(
                snapshot, list(candidates or ()), list(route_lines or ())
            )

    async def track(self, snapshot, candidates, route_lines=()):
        return await self.update(snapshot, candidates, route_lines)

    def clear(self):
        self._routes.clear()
        self._generations.clear()
        self._terminal_indices.clear()
        self._next_id = 1

    @property
    def state_size(self):
        return sum(len(tracks) for tracks in self._routes.values())

    def poll_priorities(self):
        """Return the bounded route/checkpoint hints needed by the next poll."""
        priorities = {}
        for key, tracks in self._routes.items():
            if not tracks:
                continue
            endpoints = set()
            for track in tracks.values():
                for endpoint in (
                    getattr(track.estimate, "priority_indices", None) or ()
                ):
                    if isinstance(endpoint, int) and endpoint >= 0:
                        endpoints.add(endpoint)
                    elif (
                        isinstance(endpoint, float)
                        and endpoint.is_integer()
                        and endpoint >= 0
                    ):
                        endpoints.add(int(endpoint))
                for endpoint in getattr(track.estimate, "bracket", None) or ():
                    if isinstance(endpoint, int) and endpoint >= 0:
                        endpoints.add(endpoint)
                    elif isinstance(endpoint, float) and endpoint.is_integer() and endpoint >= 0:
                        endpoints.add(int(endpoint))
                bracket = getattr(track.estimate, "bracket", None) or ()
                if len(bracket) == 2:
                    try:
                        lower, upper = sorted((int(bracket[0]), int(bracket[1])))
                    except (TypeError, ValueError):
                        lower = upper = 0
                    if upper - lower > 1:
                        midpoint = (lower + upper) // 2
                        endpoints.add(midpoint)
                        if upper - lower > 2:
                            endpoints.add(midpoint + 1)
            terminal = self._terminal_indices.get(key)
            if terminal is not None:
                endpoints.add(terminal)
            if endpoints:
                selected = sorted(endpoints)
                if terminal is not None and terminal in endpoints:
                    selected = sorted(
                        set(sorted(endpoints - {terminal})[
                            :MAX_PRIORITY_ENDPOINTS_PER_ROUTE - 1
                        ]) | {terminal}
                    )
                priorities[key] = frozenset(
                    selected[:MAX_PRIORITY_ENDPOINTS_PER_ROUTE]
                )
        return priorities

    def _update(self, snapshot, candidates, route_lines):
        now = _timestamp(getattr(snapshot, "collected_at", 0.0))
        route_terminals = _route_terminals(route_lines)
        for key in set(self._routes) & route_terminals.keys():
            self._terminal_indices[key] = route_terminals[key]
        complete = {
            tuple(route.route_key): route
            for route in getattr(snapshot, "complete_routes", ())
        }
        grouped = _group(candidates)
        keys = set(complete)
        keys.update(key for key in grouped if key in self._routes)
        # Capture every stored route before any complete or same-generation
        # branch can mutate positions. Omitted routes still reach prediction.
        split_ties = {
            key: _tie_components(tracks)
            for key, tracks in self._routes.items()
        }
        for key in sorted(keys):
            if key in route_terminals:
                self._terminal_indices[key] = route_terminals[key]
            rows = sorted(grouped.get(key, ()), key=_candidate_sort_key)
            generation = getattr(complete.get(key), "generation", None)
            tracks = self._routes.get(key)
            if tracks is None:
                if generation is None:
                    continue
                tracks = self._routes.setdefault(key, {})
            split_ties.setdefault(key, _tie_components(tracks))
            if generation is not None and generation != self._generations.get(key):
                old_generation = self._generations.get(key)
                rollback = old_generation is not None and generation < old_generation
                if rollback:
                    tracks.clear()
                    old_generation = None
                self._generations[key] = generation
                old = list(tracks.values())
                held_positions = {
                    track.track_id: track.position for track in old
                }
                self._predict(old, now, route_lines)
                pairs = _ordered_pairs(old, rows)
                proposed_positions = {
                    old_index: position
                    for old_index, new_index in pairs
                    if (position := _bracket_position(
                        old[old_index], rows[new_index]
                    )) is not None
                }
                accepted_updates = _select_ordered_updates(
                    old, proposed_positions
                )
                used = set()
                for old_index, new_index in pairs:
                    track = old[old_index]
                    candidate = rows[new_index]
                    used.add(new_index)
                    if old_index not in accepted_updates:
                        # Complete lifecycle evidence can confirm an identity,
                        # but stale, unbracketed, or order-crossing positioning
                        # evidence must retain the last exact boundary.
                        _hold_track(track, held_positions[track.track_id])
                        track.generation = generation
                        track.last_evidence_at = now
                        continue
                    track.position = proposed_positions[old_index]
                    track.estimate = replace(candidate, track_id=track.track_id, operator_code=key[0])
                    track.generation = generation
                    track.last_evidence_at = now
                    track.boundary_observed_at = _boundary_observed_at(candidate, now)
                    track.boundary_revision = _candidate_revision(candidate)
                births = []
                for index, candidate in enumerate(rows):
                    if index in used:
                        continue
                    track_id = self._next_id
                    self._next_id += 1
                    births.append(_Track(
                        track_id=track_id,
                        estimate=replace(
                            candidate, track_id=track_id, operator_code=key[0]
                        ),
                        position=float(candidate.position or 0.0),
                        generation=generation,
                        last_evidence_at=now,
                        boundary_observed_at=_boundary_observed_at(candidate, now),
                        boundary_revision=_candidate_revision(candidate),
                    ))
                matched_old = {old_index for old_index, _ in pairs}
                # A complete all-stop generation is the lifecycle authority.
                # Keeping an unmatched prior track for another generation
                # renders a ghost alongside the replacement ETA instance.
                old_survivors = [
                    track for index, track in enumerate(old)
                    if index in matched_old
                ]
                merged = _merge_tracks(old_survivors, births)
                tracks.clear()
                tracks.update((track.track_id, track) for track in merged)
            elif generation is not None and tracks:
                # A publication may be re-rendered with aged/corrected ETA
                # rows. Refresh matched tracks, but never alter cardinality
                # until a newer complete generation arrives.
                old = list(tracks.values())
                fresh_rows = [
                    candidate for candidate in rows
                    if _fresh_bracket_position(candidate) is not None
                    and any(_candidate_actionable(candidate, track) for track in old)
                ]
                pairs = _ordered_pairs(
                    old, fresh_rows,
                    compatible=lambda track, candidate: _candidate_actionable(
                        candidate, track
                    ),
                )
                proposed_positions = {
                    old_index: _bracket_position(
                        old[old_index], fresh_rows[new_index]
                    )
                    for old_index, new_index in pairs
                    if _bracket_position(old[old_index], fresh_rows[new_index]) is not None
                }
                accepted_updates = _select_ordered_updates(
                    old, proposed_positions
                )
                for old_index, new_index in pairs:
                    if old_index not in accepted_updates:
                        continue
                    track = old[old_index]
                    candidate = fresh_rows[new_index]
                    track.position = proposed_positions[old_index]
                    track.estimate = replace(candidate, track_id=track.track_id,
                                             operator_code=key[0])
                    track.last_evidence_at = now
                    track.boundary_observed_at = _boundary_observed_at(candidate, now)
                    track.boundary_revision = _candidate_revision(candidate)
            self._bound()

        output = []
        for key in sorted(self._routes):
            tracks = self._routes[key]
            ordered = list(tracks.values())
            self._predict(ordered, now, route_lines, split_ties.get(key, ()))
            self._routes[key] = tracks = self._sort_tracks(tracks)
            ordered = list(tracks.values())
            if key not in complete:
                stale = {
                    track.track_id
                    for track in ordered
                    if self._terminal_stale(track, now, route_lines)
                }
                if stale:
                    for track_id in stale:
                        self._routes[key].pop(track_id, None)
                    ordered = [track for track in ordered if track.track_id not in stale]
                    if not self._routes[key]:
                        self._routes.pop(key, None)
                        self._generations.pop(key, None)
                        self._terminal_indices.pop(key, None)
            output.extend(
                _output_estimate(track, key[0], now)
                for track in ordered
            )
        return output

    def _terminal_stale(self, track, now, route_lines):
        maximum = _route_max(track.estimate, route_lines)
        if maximum == inf:
            return False
        return (track.position >= maximum
                and now - track.last_evidence_at >= self.evidence_ttl_seconds)

    def _predict(self, tracks, now, route_lines, split_ties=()):
        del now
        split_ties = dict(split_ties)
        input_positions = [track.position for track in tracks]
        components = [
            split_ties.get(track.track_id, ("track", track.track_id))
            for track in tracks
        ]
        positions = []
        for track in tracks:
            # Every estimate holds until a fresh two-sided boundary poll
            # updates it. Cached ETA age and wall time are identity metadata,
            # never synthetic marker motion.
            position = track.position
            maximum = _route_max(track.estimate, route_lines)
            if maximum != inf:
                position = min(position, maximum)
            positions.append(position)

        # Preserve the order of distinct prior-position components globally.
        # Members of one exact prior tie may split, but a correction at one
        # boundary must be allowed to propagate through an arbitrarily long
        # chain of strict components.
        for _ in range(len(positions)):
            changed = False
            for left in range(len(positions) - 1):
                for right in range(left + 1, len(positions)):
                    if (components[left] == components[right]
                            or positions[left] <= positions[right]):
                        continue
                    if (positions[left] > input_positions[right]
                            and positions[right] <= input_positions[right]):
                        positions[left] = positions[right]
                    else:
                        positions[right] = positions[left]
                    changed = True
            if not changed:
                break

        for track, position in zip(tracks, positions, strict=True):
            track.position = position
            track.estimate = reproject_estimate(track.estimate, position, route_lines)

    @staticmethod
    def _sort_tracks(tracks):
        # Python's sort is stable, so a strict prior ordering remains intact
        # when two identities collapse to the same position.
        return dict(sorted(tracks.items(), key=lambda item: item[1].position))

    def _bound(self):
        for key in sorted(self._routes):
            tracks = self._routes[key]
            while len(tracks) > self.max_tracks_per_route:
                tracks.pop(next(iter(tracks)))
        while len(self._routes) > self.max_routes:
            key = next(iter(self._routes))
            self._routes.pop(key, None)
            self._generations.pop(key, None)
            self._terminal_indices.pop(key, None)
        active_keys = set(self._routes)
        for key in list(self._terminal_indices):
            if key not in active_keys:
                self._terminal_indices.pop(key, None)


def _group(items):
    grouped = {}
    for item in items:
        grouped.setdefault(_key(item), []).append(item)
    return grouped


def _candidate_sort_key(item):
    """Keep equal-position departures in stable ETA order."""
    arrival = getattr(item, "eta_arrival_at", None)
    try:
        arrival_key = _timestamp(arrival) if arrival is not None else inf
    except (TypeError, ValueError, OverflowError):
        arrival_key = inf
    observations = tuple(sorted(getattr(item, "source_observations", ()) or ()))
    return (float(getattr(item, "position", 0.0) or 0.0), arrival_key, observations)


def _tie_components(tracks):
    positions = [track.position for track in tracks.values()]
    return {
        track.track_id: track.position
        for track in tracks.values()
        if positions.count(track.position) > 1
    }


def _merge_tracks(old, new):
    """Position merge preserving the order of both identity subsequences."""
    merged = []
    old_index = new_index = 0
    while old_index < len(old) and new_index < len(new):
        if old[old_index].position <= new[new_index].position:
            merged.append(old[old_index])
            old_index += 1
        else:
            merged.append(new[new_index])
            new_index += 1
    merged.extend(old[old_index:])
    merged.extend(new[new_index:])
    return merged


def _ordered_pairs(old, new, compatible=None):
    # Score an ordered alignment by retained cardinality first, then by ETA-
    # anchor presence continuity. ETA timestamps can drift by tens of seconds
    # between provider generations, while an unbracketed turnover candidate
    # has no timestamp at all. Letting plain distance make that mixed-anchor
    # pair look cheaper can assign the turnover candidate to a surviving
    # downstream track and then birth a second marker from its source ladder.
    # Source-row slots can shift after a departure, so overlap remains only a
    # deterministic final tie-breaker rather than an identity authority.
    dp = [[(0, 0, 0.0, ()) for _ in range(len(new) + 1)]
          for _ in range(len(old) + 1)]
    for i in range(1, len(old) + 1):
        for j in range(1, len(new) + 1):
            choices = [dp[i - 1][j], dp[i][j - 1]]
            old_anchor = getattr(old[i - 1].estimate, "eta_arrival_at", None)
            new_anchor = getattr(new[j - 1], "eta_arrival_at", None)
            anchor_distance = 0.0
            if old_anchor is not None and new_anchor is not None:
                try:
                    anchor_distance = abs(_timestamp(old_anchor) - _timestamp(new_anchor)) / 60.0
                except (TypeError, ValueError):
                    anchor_distance = 0.0
            distance = abs(old[i - 1].position - float(new[j - 1].position or 0.0))
            if distance <= MATCH_DISTANCE and (
                compatible is None or compatible(old[i - 1], new[j - 1])
            ):
                previous = dp[i - 1][j - 1]
                overlap = bool(
                    old[i - 1].estimate.source_observations
                    & new[j - 1].source_observations
                )
                anchor_mismatch = (old_anchor is None) != (new_anchor is None)
                choices.append(
                    (
                        previous[0] + 1,
                        previous[1] + int(anchor_mismatch),
                        previous[2] + (
                            anchor_distance * 10.0 + distance * 0.01
                            if old_anchor is not None and new_anchor is not None
                            else distance
                        ),
                        previous[3] + ((0 if overlap else 1, i - 1, j - 1),),
                    )
                )
            dp[i][j] = min(
                choices,
                key=lambda item: (-item[0], item[1], item[2], item[3]),
            )
    return [(item[1], item[2]) for item in dp[-1][-1][3]]


def _bracket_position(track, candidate):
    """Use only a freshly observed boundary to reposition a marker."""
    if not _candidate_actionable(candidate, track):
        return None
    return _fresh_bracket_position(candidate)


def _fresh_bracket_position(candidate):
    bracket = getattr(candidate, "bracket", None)
    if not bracket:
        return None
    try:
        lower, upper = map(float, bracket)
    except (TypeError, ValueError):
        return None
    if not isfinite(lower) or not isfinite(upper) or lower > upper:
        return None
    age = getattr(candidate, "boundary_age_seconds", None)
    try:
        age = float(age)
    except (TypeError, ValueError):
        return None
    if not isfinite(age) or age < 0.0:
        return None
    # Provider revisions are durable evidence; render delay must not turn
    # otherwise valid observations into synthetic motion or a held marker.
    if _candidate_revision(candidate) is not None:
        position = float(getattr(candidate, "position", lower) or lower)
        if not isfinite(position):
            return None
        return min(upper, max(lower, position))
    if age > 5.0:
        return None
    return min(upper, max(lower, float(candidate.position or lower)))


def _candidate_revision(candidate):
    value = getattr(candidate, "boundary_revision", None)
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        return None
    try:
        revision = (int(value[0]), int(value[1]))
    except (TypeError, ValueError):
        return None
    return revision if all(item > 0 for item in revision) else None


def _candidate_actionable(candidate, track):
    """Require unseen complete boundary evidence in production snapshots."""
    raw_revision = getattr(candidate, "boundary_revision", None)
    revision = _candidate_revision(candidate)
    if raw_revision is not None and revision is None:
        return False
    previous = track.boundary_revision
    if revision is not None:
        if previous is None:
            return True
        # Both physical endpoints must advance; this rejects one-sided stale
        # snapshots and ensures a cached replay cannot move a marker.
        return revision[0] > previous[0] and revision[1] > previous[1]
    # Hand-built/legacy estimates have no stable provider revision.
    return previous is None


def _select_ordered_updates(old, proposed_positions):
    """Keep the largest exact-update subset that cannot reorder identities.

    Tracks tied at the prior position are one unordered component and may
    split when fresh evidence distinguishes them. Distinct prior-position
    components retain their global order. Rejected proposals keep both their
    old position and old evidence instead of relabelling an adjusted point as
    the fresh ETA-proportionate position.
    """
    if not proposed_positions:
        return set()

    components = []
    for index, track in enumerate(old):
        if not components or old[components[-1][0]].position != track.position:
            components.append([index])
        else:
            components[-1].append(index)

    def score(value):
        count, movement, selected = value
        return (-count, movement, selected)

    component_options = []
    for component in components:
        states = {None: (0, 0.0, ())}
        for index in component:
            choices = [(float(old[index].position), False)]
            if index in proposed_positions:
                choices.append((float(proposed_positions[index]), True))
            next_states = {}
            for bounds, value in states.items():
                for position, selected in choices:
                    lower = position if bounds is None else min(bounds[0], position)
                    upper = position if bounds is None else max(bounds[1], position)
                    candidate = (
                        value[0] + int(selected),
                        value[1] + (
                            abs(position - float(old[index].position))
                            if selected else 0.0
                        ),
                        value[2] + ((index,) if selected else ()),
                    )
                    key = (lower, upper)
                    if key not in next_states or score(candidate) < score(next_states[key]):
                        next_states[key] = candidate
            states = next_states
        component_options.append([
            (bounds[0], bounds[1], value)
            for bounds, value in states.items()
        ])

    states = {None: (0, 0.0, ())}
    for options in component_options:
        next_states = {}
        for previous_upper, previous in states.items():
            for lower, upper, option in options:
                if previous_upper is not None and previous_upper > lower:
                    continue
                candidate = (
                    previous[0] + option[0],
                    previous[1] + option[1],
                    previous[2] + option[2],
                )
                if upper not in next_states or score(candidate) < score(next_states[upper]):
                    next_states[upper] = candidate
        states = next_states

    best = min(states.values(), key=score)
    return set(best[2])


def _boundary_observed_at(candidate, now):
    age = getattr(candidate, "boundary_age_seconds", None)
    try:
        age = float(age)
    except (TypeError, ValueError):
        return None
    if not isfinite(age) or age < 0.0:
        return None
    return now - age


def _output_estimate(track, operator_code, now):
    age = None
    if track.boundary_observed_at is not None:
        age = max(0.0, now - track.boundary_observed_at)
    return replace(
        track.estimate,
        track_id=track.track_id,
        operator_code=operator_code,
        boundary_age_seconds=age,
    )


def _hold_track(track, position):
    """Keep a generation-confirmed track fixed until fresh evidence arrives."""
    track.position = position


def _timestamp(value):
    return float(value.timestamp()) if hasattr(value, "timestamp") else float(value)


def _positive_int(value, name):
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a positive integer") from None
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _nonnegative_finite(value, name):
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a nonnegative finite number") from None
    if not isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a nonnegative finite number")
    return result


def _route_max(estimate, route_lines):
    key = (estimate.operator_code or str(estimate.operator), estimate.route, estimate.bound)
    for line in route_lines:
        line_key = (
            str(getattr(line, "operator", "")),
            str(getattr(line, "route", "")),
            str(getattr(line, "bound", "")),
        )
        if line_key == key:
            return max(0.0, float(len(list(getattr(line, "stops", ()))) - 1))
    return inf


def _route_terminals(route_lines):
    terminals = {}
    for line in route_lines:
        stops = list(getattr(line, "stops", ()) or ())
        if stops:
            key = (
                str(getattr(line, "operator", "")),
                str(getattr(line, "route", "")),
                str(getattr(line, "bound", "")),
            )
            terminals[key] = len(stops) - 1
    return terminals


__all__ = ["MarkerTracker"]
