"""Route-generic temporal identity for estimated map markers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from math import inf, isfinite

from dashboard.maps.positions import reproject_estimate

MAX_ROUTES = 64
MAX_TRACKS_PER_ROUTE = 128
MATCH_DISTANCE = 3.5
SECONDS_PER_STOP = 120.0
SMOOTHING_ALPHA = 0.35
MAX_PRIORITY_ENDPOINTS_PER_ROUTE = 32


@dataclass
class _Track:
    track_id: int
    estimate: object
    phase: float
    position: float
    generation: int
    hits: int = 1
    misses: int = 0
    confirmed: bool = True
    last_evidence_at: float = 0.0


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
                for endpoint in getattr(track.estimate, "bracket", None) or ():
                    if isinstance(endpoint, int) and endpoint >= 0:
                        endpoints.add(endpoint)
                    elif isinstance(endpoint, float) and endpoint.is_integer() and endpoint >= 0:
                        endpoints.add(int(endpoint))
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
        for key in sorted(keys):
            if key in route_terminals:
                self._terminal_indices[key] = route_terminals[key]
            rows = sorted(grouped.get(key, ()), key=lambda item: float(item.position or 0.0))
            generation = getattr(complete.get(key), "generation", None)
            tracks = self._routes.get(key)
            if tracks is None:
                if generation is None:
                    continue
                tracks = self._routes.setdefault(key, {})
            if generation is not None and generation != self._generations.get(key):
                old_generation = self._generations.get(key)
                rollback = old_generation is not None and generation < old_generation
                if rollback:
                    tracks.clear()
                    old_generation = None
                self._generations[key] = generation
                old = list(tracks.values())
                self._predict(old, now, route_lines)
                pairs = _ordered_pairs(old, rows)
                used = set()
                for old_index, new_index in pairs:
                    track = old[old_index]
                    candidate = rows[new_index]
                    used.add(new_index)
                    candidate_bracket = getattr(candidate, "bracket", None)
                    bracket_position = _bracket_position(track, candidate)
                    if candidate_bracket is not None and bracket_position is None:
                        # Complete lifecycle evidence can advance generation,
                        # but stale positioning evidence must hold the track.
                        track.generation = generation
                        track.hits += 1
                        track.misses = 0
                        track.confirmed = track.confirmed or track.hits >= 2
                        track.last_evidence_at = now
                        continue
                    if candidate_bracket is None and not _authoritative_evidence(candidate):
                        # A complete generation may confirm that this identity
                        # still exists, but incomplete probe geometry cannot
                        # replace the last real positioning boundary.
                        track.generation = generation
                        track.hits += 1
                        track.misses = 0
                        track.confirmed = track.confirmed or track.hits >= 2
                        track.last_evidence_at = now
                        continue
                    target_phase = now - float(candidate.position or track.position) * SECONDS_PER_STOP
                    track.phase += SMOOTHING_ALPHA * (target_phase - track.phase)
                    track.position = (bracket_position if bracket_position is not None
                                      else max(track.position, (now - track.phase) / SECONDS_PER_STOP))
                    track.estimate = replace(candidate, track_id=track.track_id, operator_code=key[0])
                    track.generation = generation
                    track.last_evidence_at = now
                    track.hits += 1
                    track.misses = 0
                    track.confirmed = track.confirmed or track.hits >= 2
                matched_ids = {old[index].track_id for index, _ in pairs}
                births = []
                for index, candidate in enumerate(rows):
                    if index in used:
                        continue
                    track_id = self._next_id
                    self._next_id += 1
                    births.append(_Track(track_id,
                        replace(candidate, track_id=track_id, operator_code=key[0]),
                        now - float(candidate.position or 0.0) * SECONDS_PER_STOP,
                        float(candidate.position or 0.0),
                        generation,
                        confirmed=(_reliable(candidate) or (old_generation is None and not _tentative(candidate))),
                        last_evidence_at=now,
                    ))
                for track_id, track in list(tracks.items()):
                    if track.generation != generation and track_id not in matched_ids:
                        track.misses += 1
                        if track.misses >= 2:
                            tracks.pop(track_id, None)
                old_survivors = [track for track in old
                                 if track.track_id in tracks]
                merged = _merge_tracks(old_survivors, births)
                tracks.clear()
                tracks.update((track.track_id, track) for track in merged)
            elif generation is not None and tracks:
                # A publication may be re-rendered with aged/corrected ETA
                # rows. Refresh matched tracks, but never alter cardinality
                # until a newer complete generation arrives.
                old = list(tracks.values())
                pairs = _ordered_pairs(old, rows)
                for old_index, new_index in pairs:
                    track = old[old_index]
                    candidate = rows[new_index]
                    position = _bracket_position(track, candidate)
                    if (
                        position is None
                        and getattr(candidate, "bracket", None) is None
                        and _authoritative_evidence(candidate)
                    ):
                        position = float(candidate.position or track.position)
                    if position is None:
                        continue
                    track.position = position
                    track.estimate = replace(candidate, track_id=track.track_id,
                                             operator_code=key[0])
                    track.last_evidence_at = now
                    track.misses = 0
            self._bound()

        output = []
        for key in sorted(self._routes):
            tracks = self._routes[key]
            ordered = list(tracks.values())
            self._predict(ordered, now, route_lines)
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
                replace(track.estimate, track_id=track.track_id, operator_code=key[0])
                for track in ordered
                if track.confirmed
            )
        return output

    def _terminal_stale(self, track, now, route_lines):
        maximum = _route_max(track.estimate, route_lines)
        if maximum == inf:
            return False
        raw_position = max(track.position, (now - track.phase) / SECONDS_PER_STOP)
        return raw_position >= maximum and now - track.last_evidence_at >= self.evidence_ttl_seconds

    def _predict(self, tracks, now, route_lines):
        prior = -inf
        for track in tracks:
            # Bracketed estimates move only when a fresh boundary poll updates
            # them.  Cached ETA age must never become synthetic motion.
            if getattr(track.estimate, "bracket", None) is not None:
                position = track.position
            else:
                position = max(track.position, (now - track.phase) / SECONDS_PER_STOP)
            maximum = _route_max(track.estimate, route_lines)
            if maximum != inf:
                position = min(position, maximum)
            # Preserve order without inventing a positive separation.
            position = max(position, prior)
            track.position = position
            prior = position
            track.estimate = reproject_estimate(track.estimate, position, route_lines)

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


def _ordered_pairs(old, new):
    dp = [[(0, 0.0, ()) for _ in range(len(new) + 1)] for _ in range(len(old) + 1)]
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
            if distance <= MATCH_DISTANCE:
                previous = dp[i - 1][j - 1]
                overlap = bool(
                    old[i - 1].estimate.source_observations
                    & new[j - 1].source_observations
                )
                choices.append(
                    (
                        previous[0] + 1,
                        previous[1] + (
                            anchor_distance * 10.0 + distance * 0.01
                            if old_anchor is not None and new_anchor is not None
                            else distance
                        ),
                        previous[2] + ((0 if overlap else 1, i - 1, j - 1),),
                    )
                )
            dp[i][j] = min(choices, key=lambda item: (-item[0], item[1], item[2]))
    return [(item[1], item[2]) for item in dp[-1][-1][2]]


def _bracket_position(track, candidate):
    """Use only a freshly observed boundary to reposition a marker."""
    bracket = getattr(candidate, "bracket", None)
    if not bracket:
        return None
    lower, upper = map(float, bracket)
    age = getattr(candidate, "boundary_age_seconds", None)
    if age is None:
        return None
    try:
        age = float(age)
    except (TypeError, ValueError):
        return None
    if not isfinite(age) or age < 0.0 or age > 5.0:
        return None
    return min(upper, max(lower, float(candidate.position or lower)))


def _timestamp(value):
    return float(value.timestamp()) if hasattr(value, "timestamp") else float(value)


def _reliable(candidate):
    return not bool(getattr(candidate, "unreliable", False)) and any(
        str(kind).lower() in {"gate", "authoritative"}
        for kind, _ in candidate.source_observations
    )


def _authoritative_evidence(candidate):
    return any(
        str(kind).lower() in {"gate", "authoritative"}
        for kind, _ in getattr(candidate, "source_observations", ())
    )


def _tentative(candidate):
    if bool(getattr(candidate, "unreliable", False)):
        return True
    return any(str(kind).lower() in {"scheduled", "schedule"}
               for kind, _ in candidate.source_observations)


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
