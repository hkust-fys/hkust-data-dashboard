"""Route-generic temporal identity for estimated map markers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
from math import inf, isfinite

from dashboard.maps.positions import reproject_estimate

MAX_ROUTES = 64
MAX_TRACKS_PER_ROUTE = 128
MATCH_DISTANCE = 3.5
MAX_PRIORITY_ENDPOINTS_PER_ROUTE = 32
RECOVERY_ARRIVAL_TOLERANCE_SECONDS = 90.0
MAX_CHECKPOINT_EVIDENCE = 64


@dataclass
class _Track:
    track_id: int
    estimate: object
    position: float
    generation: int
    last_evidence_at: float = 0.0
    boundary_observed_at: float | None = None
    boundary_revision: tuple[int, int] | None = None
    forward_frontier: tuple[int, ...] = ()
    forward_after: int | None = None
    forward_revision: int = 0
    forward_started_revision: int = 0
    forward_baselines: dict[int, int] = field(default_factory=dict)


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
            recovery_endpoints = set()
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
                recovery_endpoints.update(track.forward_frontier)
                if track.forward_after is not None:
                    recovery_endpoints.add(track.forward_after)
            terminal = self._terminal_indices.get(key)
            if terminal is not None:
                endpoints.add(terminal)
            if endpoints or recovery_endpoints:
                # A held marker's old, low-index rungs must not crowd the
                # downstream search out of the bounded priority tier.
                selected = sorted(recovery_endpoints - {terminal})
                selected.extend(sorted(endpoints - recovery_endpoints - {terminal}))
                selected = selected[:MAX_PRIORITY_ENDPOINTS_PER_ROUTE - 1]
                if terminal is not None:
                    selected.append(terminal)
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
        keys.update(self._routes)
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
            route_rows = tuple(row for row in getattr(snapshot, "rows", ())
                               if _key(row) == key)
            checkpoints = _successful_checkpoints(route_rows)
            for track in tracks.values():
                _refresh_forward_search(
                    track, checkpoints, self._terminal_indices.get(key)
                )
            _advance_forward_search_from_candidates(tracks, rows, checkpoints,
                                                     self._terminal_indices.get(key))
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
                pairs = _ordered_pairs(
                    old, rows, compatible=_search_compatible,
                    recoveries=_recovery_pairs(old, rows, checkpoints),
                )
                # A complete publication can contain one newly discovered,
                # coarse downstream bracket while a held marker is already
                # searching forward.  Consume that candidate for lifecycle
                # cardinality, but defer movement until a narrow requested
                # corridor arrives.  Restrict this to an unambiguous 1:1
                # route association; competing tracks fail closed.
                hold_pairs = _forward_hold_pairs(old, rows)
                if hold_pairs:
                    pairs = sorted(set(pairs) | hold_pairs)
                proposed_positions = {
                    old_index: position
                    for old_index, new_index in pairs
                    if (old_index, new_index) not in hold_pairs
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
                    _clear_forward_search(track)
                    track.forward_frontier = ()
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
                    recoveries=_recovery_pairs(old, fresh_rows, checkpoints),
                )
                proposed_positions = {
                    old_index: position
                    for old_index, new_index in pairs
                    if (position := _bracket_position(
                        old[old_index], fresh_rows[new_index]
                    )) is not None
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
                    _clear_forward_search(track)
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


def _checkpoint_rows(estimate):
    rows = getattr(estimate, "checkpoint_evidence", ()) or ()
    out = []
    for row in rows:
        try:
            index, arrival, revision = int(row[0]), float(row[1]), int(row[2])
        except (TypeError, ValueError, IndexError, OverflowError):
            continue
        if index >= 0 and isfinite(arrival) and revision > 0:
            out.append((index, arrival, revision))
    return tuple(out[:MAX_CHECKPOINT_EVIDENCE])


def _arrival_timestamp(value):
    if value is None:
        return None
    try:
        stamp = _timestamp(value)
    except (TypeError, ValueError, OverflowError, OSError):
        return None
    return stamp if isfinite(stamp) else None


def _successful_checkpoints(rows):
    """Index exact cached responses, never treating a missing response as empty."""
    grouped = {}
    invalid = set()
    for row in rows:
        try:
            index = int(row.index)
            revision = int(row.refresh_generation)
            age = float(row.cache_age_seconds)
        except (AttributeError, TypeError, ValueError, OverflowError):
            continue
        if index < 0:
            continue
        if revision <= 0 or not isfinite(age) or not 0 <= age < 900:
            invalid.add(index)
            continue
        grouped.setdefault(index, []).append(row)
    return {
        index: (int(values[0].refresh_generation), tuple(values))
        for index, values in grouped.items()
        if index not in invalid
        and len({int(row.refresh_generation) for row in values}) == 1
    }


def _missing_instance(track, index, rows):
    if all(getattr(row, "minutes", None) is None for row in rows):
        return True
    prior = [arrival for stop, arrival, _revision in _checkpoint_rows(track.estimate)
             if stop == index]
    current = [_arrival_timestamp(getattr(row, "arrival_at", None))
               for row in rows if getattr(row, "minutes", None) is not None]
    # Without comparable timestamps, a nonempty response is inconclusive.
    return bool(prior and current and all(value is not None for value in current)) and not any(
        abs(before - after) <= RECOVERY_ARRIVAL_TOLERANCE_SECONDS
        for before in prior for after in current
    )


def _search_anchors(after, terminal):
    # A few forward scouts plus a distant sentinel; do not walk every stop.
    return tuple(sorted({index for index in (
        after + 1, after + 2, after + 4, after + 6,
        after + max(1, (terminal - after) // 2), terminal,
    ) if after < index <= terminal}))


def _clear_forward_search(track):
    track.forward_after = None
    track.forward_revision = 0
    track.forward_started_revision = 0
    track.forward_frontier = ()
    track.forward_baselines.clear()


def _refresh_forward_search(track, checkpoints, terminal):
    if terminal is None:
        return
    try:
        upper = float(track.estimate.bracket[1])
    except (AttributeError, TypeError, ValueError, IndexError):
        return
    if not isfinite(upper) or not upper.is_integer() or not 0 <= upper <= terminal:
        return
    previous_revision = track.forward_revision
    requested = track.forward_frontier
    advanced = False
    if track.forward_after is None:
        index = int(upper)
        response = checkpoints.get(index)
        if (response is None or track.boundary_revision is None
                or bool(getattr(track.estimate, "unreliable", False))
                or index >= terminal
                or abs(float(track.position) - upper) > MATCH_DISTANCE):
            return
        revision, rows = response
        if revision <= track.boundary_revision[1] or not _missing_instance(track, index, rows):
            return
        track.forward_after = index
        track.forward_revision = revision
        track.forward_started_revision = revision
        advanced = True
    else:
        # Only a requested downstream checkpoint can extend the search.  The
        # consumed watermark prevents the same empty response moving it again.
        missing = [
            (index, revision)
            for index in track.forward_frontier
            if (response := checkpoints.get(index)) is not None
            for revision, rows in (response,)
            if revision > max(track.forward_revision, track.forward_baselines.get(index, 0))
            and _missing_instance(track, index, rows)
        ]
        if missing:
            track.forward_after = max(index for index, _revision in missing)
            track.forward_revision = max(revision for _index, revision in missing)
            advanced = True
    present = [index for index in requested if index > track.forward_after
               and (response := checkpoints.get(index)) is not None
               and response[0] > max(previous_revision, track.forward_baselines.get(index, 0))
               and any(isinstance(getattr(row, "minutes", None), (int, float))
                       and row.minutes > 0 for row in response[1])]
    if not advanced and not present:
        return
    frontier = set(_search_anchors(track.forward_after, terminal))
    if present:
        # Keep the just-found upper boundary eligible for matching even when
        # a later-arriving empty lower response advances the search watermark.
        upper = min(present)
        frontier = {index for index in frontier if index <= upper or index == terminal}
        frontier.add(upper)
    track.forward_frontier = tuple(sorted(frontier))
    track.forward_baselines = {
        index: track.forward_baselines.get(index, checkpoints.get(index, (0, ()))[0])
        for index in track.forward_frontier
    }


def _common_checkpoint(evidence, current_by_index):
    return any(
        new_revision > revision
        and abs(new_arrival - arrival) <= RECOVERY_ARRIVAL_TOLERANCE_SECONDS
        for index, arrival, revision in evidence
        for new_arrival, new_revision in current_by_index.get(index, ())
    )


def _recovery_pairs(old, new, checkpoints):
    """Uniquely evidenced long jumps, not a wider nearest-neighbour radius."""
    current = []
    for candidate in new:
        checkpoints_by_index = {}
        for index, arrival, revision in _checkpoint_rows(candidate):
            checkpoints_by_index.setdefault(index, []).append((arrival, revision))
        current.append(checkpoints_by_index)
    anchors = {(i, j) for i, track in enumerate(old)
               for evidence in (_checkpoint_rows(track.estimate),)
               for j, checkpoints_by_index in enumerate(current)
               if _common_checkpoint(evidence, checkpoints_by_index)}
    old_degrees, new_degrees = {}, {}
    for i, j in anchors:
        old_degrees[i] = old_degrees.get(i, 0) + 1
        new_degrees[j] = new_degrees.get(j, 0) + 1
    possible = set()
    for i, track in enumerate(old):
        for j, candidate in enumerate(new):
            position = _bracket_position(track, candidate)
            if position is None or abs(position - track.position) <= MATCH_DISTANCE:
                continue
            lower, upper = map(float, candidate.bracket)
            # First narrow a coarse search interval. This also avoids leaping
            # to its far end and then losing the necessary backward refinement.
            if upper - lower > MATCH_DISTANCE:
                continue
            unique_anchor = ((i, j) in anchors
                             and old_degrees[i] == new_degrees[j] == 1)
            nested = (getattr(track.estimate, "bracket", None) is not None
                      and track.estimate.bracket[0] <= lower <= upper <= track.estimate.bracket[1])
            if unique_anchor and (position > track.position or nested):
                possible.add((i, j))
                continue
            # Timestamp-free providers can recover one isolated track from a
            # newly probed, narrow corridor. Any competing vehicle fails closed.
            response = checkpoints.get(int(upper)) if upper.is_integer() else None
            if (len(old) == len(new) == 1 and track.forward_after is not None
                    and position > track.position and lower >= track.forward_after
                    and int(upper) in track.forward_frontier and response is not None
                    and response[0] > max(track.forward_started_revision,
                                          track.forward_baselines.get(int(upper), 0))
                    and any(getattr(row, "minutes", None) is not None for row in response[1])):
                possible.add((i, j))
    return possible


def _forward_hold_pairs(old, new):
    """Associate one coarse forward candidate without moving the marker."""
    if len(old) != len(new) or len(old) != 1:
        return set()
    track, candidate = old[0], new[0]
    after = track.forward_after
    bracket = getattr(candidate, "bracket", None)
    if after is None or not bracket or len(bracket) != 2:
        return set()
    try:
        lower, upper = map(float, bracket)
        position = float(getattr(candidate, "position", 0.0) or 0.0)
    except (TypeError, ValueError):
        return set()
    if (not all(isfinite(value) for value in (lower, upper, position))
            or lower > upper or upper - lower <= MATCH_DISTANCE
            or lower < after or position <= track.position
            or _bracket_position(track, candidate) is None):
        return set()
    owned = _checkpoint_rows(track.estimate)
    current = _checkpoint_rows(candidate)
    common = any(
        new_index == old_index
        and new_revision > old_revision
        and abs(new_arrival - old_arrival) <= RECOVERY_ARRIVAL_TOLERANCE_SECONDS
        for old_index, old_arrival, old_revision in owned
        for new_index, new_arrival, new_revision in current
    )
    if not common:
        return set()
    return {(0, 0)}


def _advance_forward_search_from_candidates(tracks, candidates, checkpoints, terminal):
    """Use one coarse candidate's fresh lower rung to reseed forward scouts."""
    if len(tracks) != 1 or len(candidates) != 1 or terminal is None:
        return
    track = next(iter(tracks.values()))
    if track.forward_after is None:
        return
    candidate = candidates[0]
    bracket = getattr(candidate, "bracket", None)
    if not bracket or len(bracket) != 2:
        return
    try:
        lower, upper = map(float, bracket)
    except (TypeError, ValueError):
        return
    if (not all(isfinite(value) for value in (lower, upper))
            or not lower.is_integer() or lower <= track.forward_after
            or upper - lower <= MATCH_DISTANCE
            or lower >= terminal):
        return
    lower = int(lower)
    response = checkpoints.get(lower)
    if response is None or lower not in track.forward_frontier:
        return
    revision, _rows = response
    baseline = max(
        track.forward_revision,
        track.forward_started_revision,
        track.forward_baselines.get(lower, 0),
    )
    if revision <= baseline:
        return
    owned = _checkpoint_rows(track.estimate)
    current = _checkpoint_rows(candidate)
    matches = {
        old_index
        for old_index, old_arrival, old_revision in owned
        for new_index, new_arrival, new_revision in current
        if old_index == new_index
        and new_revision > old_revision
        and abs(new_arrival - old_arrival) <= RECOVERY_ARRIVAL_TOLERANCE_SECONDS
    }
    if not matches:
        return
    track.forward_after = lower
    track.forward_revision = revision
    track.forward_frontier = _search_anchors(lower, terminal)
    track.forward_baselines = {
        index: checkpoints.get(index, (0, ()))[0]
        for index in track.forward_frontier
    }


def _ordered_pairs(old, new, compatible=None, recoveries=()):
    # Score an ordered alignment by retained cardinality first, then by ETA-
    # anchor presence continuity. ETA timestamps can drift by tens of seconds
    # between provider generations, while an unbracketed turnover candidate
    # has no timestamp at all. Letting plain distance make that mixed-anchor
    # pair look cheaper can assign the turnover candidate to a surviving
    # downstream track and then birth a second marker from its source ladder.
    # Source-row slots can shift after a departure, so overlap remains only a
    # deterministic final tie-breaker rather than an identity authority.
    reserved_old = {i for i, _j in recoveries}
    reserved_new = {j for _i, j in recoveries}
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
            pair = (i - 1, j - 1)
            recovery = pair in recoveries
            reserved = not recovery and (pair[0] in reserved_old or pair[1] in reserved_new)
            if not reserved and (distance <= MATCH_DISTANCE or recovery) and (
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


def _search_compatible(track, candidate):
    return (track.forward_after is None
            or float(getattr(candidate, "position", 0.0) or 0.0) >= track.forward_after)


def _candidate_actionable(candidate, track):
    """Require unseen complete boundary evidence in production snapshots."""
    if not _search_compatible(track, candidate):
        # A later departure at the vacated stop is not the bus being searched
        # for. Absence changes probe selection, not this marker's identity.
        return False
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
