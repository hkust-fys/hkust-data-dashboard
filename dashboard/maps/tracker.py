"""Route-generic temporal identity for estimated map markers."""

from __future__ import annotations

import asyncio
from bisect import bisect_left, bisect_right
from collections import Counter
from dataclasses import dataclass, field, replace
from math import ceil, floor, inf, isfinite, nextafter

from dashboard.maps.positions import (
    LIVE_PROBE_ETA_KINDS,
    MINUTES_PER_STOP,
    rebuild_estimate_from_probe_fragments,
    rebuild_estimate_from_probe_sources,
    reproject_estimate,
)

MAX_ROUTES = 64
MAX_TRACKS_PER_ROUTE = 128
MATCH_DISTANCE = 3.5
MAX_PRIORITY_ENDPOINTS_PER_ROUTE = 32
RECOVERY_ARRIVAL_TOLERANCE_SECONDS = 90.0
MAX_CHECKPOINT_EVIDENCE = 64
COHORT_EVIDENCE_TTL_SECONDS = 120.0
COHORT_MAX_SECONDS_PER_STOP = 180.0
PARTIAL_BIRTH_FRESHNESS_SECONDS = 60.0
PARTIAL_BIRTH_MAX_AGE_SKEW_SECONDS = 5.0
MAX_RESEED_TEMPORAL_NEIGHBOURS = 32
MAX_RESEED_TEMPORAL_EDGES = 32768


@dataclass
class _Track:
    track_id: int
    estimate: object
    position: float
    generation: int
    last_evidence_at: float = 0.0
    boundary_observed_at: float | None = None
    boundary_revision: tuple[int, int] | None = None
    motion_bracket: tuple[float, float] | None = None
    display_bracket: tuple[float, float] | None = None
    forward_frontier: tuple[int, ...] = ()
    forward_after: int | None = None
    forward_revision: int = 0
    forward_started_revision: int = 0
    forward_baselines: dict[int, int] = field(default_factory=dict)
    # The last atomically reconciled checkpoint population is identity
    # evidence, distinct from the latest partial candidate used for position.
    cohort_evidence: tuple[tuple[int, float, int], ...] = ()
    cohort_observed_at: float = 0.0
    # False only for positively mixed/incoherent display births. A legitimate
    # timestamp-free or gate-only track may have an empty checkpoint ledger
    # without losing its established ETA-anchor matching contract.
    cohort_trusted: bool = True
    # Exact occurrences which justified the last committed motion boundary.
    # This survives replacement of ``estimate`` by a held complete-generation
    # candidate (which may contain a different bus' checkpoint population).
    committed_boundary_evidence: tuple[tuple[int, float, int], ...] = ()
    # Sticky once any reliable display or fresh motion boundary establishes
    # this identity's route order. A later timetable-only cohort cannot erase
    # that historical position authority.
    position_authoritative: bool = False
    # Current-source metadata may refresh without committing a motion boundary.
    # Keep its per-stop high-water marks even when a later bundle omits a stop.
    metadata_revision_floors: dict[int, int] = field(default_factory=dict)


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
        # Every route exposes one immutable physical service page.  The queue
        # supplies fair pages only when the population exceeds the cap; the
        # same acknowledgement contract protects small plans from cache-only
        # presentations replacing a probe before it was actually attempted.
        self._priority_queue = {}
        self._priority_pending = {}
        self._probe_ack_generation = None
        self._lifecycle_refresh_routes = set()
        self._partial_birth_generations = {}
        self._next_id = 1
        self._lock = asyncio.Lock()
        self._presented = None

    async def update(self, snapshot, candidates, route_lines=()):
        async with self._lock:
            self._ack_probe_attempts(snapshot)
            return self._update(
                snapshot, list(candidates or ()), list(route_lines or ())
            )

    async def track(self, snapshot, candidates, route_lines=()):
        return await self.update(snapshot, candidates, route_lines)

    async def present(self, snapshot, candidates, route_lines=()):
        """Display the current ETA population, with identity only as a hint.

        A stop's ETA rank is not a vehicle identifier. The temporal matcher
        may retain an uncertain identity while requesting more evidence, but
        that must not retain an extra visible bus or veto a corrected local
        interpolation. Counts, coordinates and provenance come from this
        frame's estimator; historical tracks only supply unambiguous IDs.
        """
        grouped = _group(candidates or ())
        candidates = [candidate for key in list(grouped)[:self.max_routes]
                      for candidate in grouped[key][:self.max_tracks_per_route]]
        route_lines = list(route_lines or ())
        async with self._lock:
            self._ack_probe_attempts(snapshot)
            tracked = self._update(snapshot, candidates, route_lines)
            self._terminal_indices.update(_route_terminals(route_lines))
            by_evidence = {}
            for marker in tracked:
                evidence = tuple(getattr(marker, "checkpoint_evidence", ()) or ())
                if evidence:
                    by_evidence.setdefault((_key(marker), evidence), []).append(marker.track_id)
            used_ids = set()
            presented = []
            for candidate in candidates:
                evidence = tuple(getattr(candidate, "checkpoint_evidence", ()) or ())
                matches = by_evidence.get((_key(candidate), evidence), ())
                track_id = matches[0] if len(matches) == 1 else None
                if track_id in used_ids:
                    track_id = None
                used_ids.add(track_id)
                presented.append(replace(candidate, track_id=track_id,
                                         operator_code=_key(candidate)[0]))
            self._presented = _group(presented)
            return presented

    def clear(self):
        self._routes.clear()
        self._generations.clear()
        self._terminal_indices.clear()
        self._priority_queue.clear()
        self._priority_pending.clear()
        self._probe_ack_generation = None
        self._lifecycle_refresh_routes.clear()
        self._partial_birth_generations.clear()
        self._next_id = 1
        self._presented = None

    @property
    def state_size(self):
        return sum(len(tracks) for tracks in self._routes.values())

    def poll_priorities(self):
        """Return only the physical checkpoints needed by the next refinement."""
        if self._presented is not None:
            return self._presented_priorities()
        priorities = {}
        for key, tracks in self._routes.items():
            if not tracks:
                self._priority_queue.pop(key, None)
                self._priority_pending.pop(key, None)
                continue
            terminal = self._terminal_indices.get(key)
            latched = self._latched_priority_page(key, terminal)
            if latched is not None:
                selected = set(latched)
                if terminal is not None:
                    selected.add(terminal)
                priorities[key] = frozenset(selected)
                continue
            recovery_plans = []
            ordinary_plans = []
            for track in tracks.values():
                checkpoints = _next_poll_checkpoints(track, terminal)
                if terminal is not None:
                    checkpoints = tuple(index for index in checkpoints
                                        if index != terminal)
                if track.forward_after is not None:
                    recovery_plans.append((track.track_id, checkpoints))
                else:
                    ordinary_plans.append((track.track_id, checkpoints))
            budget = MAX_PRIORITY_ENDPOINTS_PER_ROUTE - (1 if terminal is not None else 0)
            population = _priority_population(recovery_plans, ordinary_plans)
            queue = self._priority_queue.setdefault(key, [])
            current = set(population)
            queue[:] = [item for item in queue if item in current]
            queue.extend(item for item in population if item not in queue)
            window = tuple(queue[:budget])
            selected = set(window)
            if window:
                self._priority_pending[key] = (window, set(window))
            if terminal is not None:
                selected.add(terminal)
            if selected:
                priorities[key] = frozenset(
                    selected
                )
        return priorities

    def _presented_priorities(self):
        """Refine the immediate neighbours of the markers actually on the map."""
        priorities = {}
        for key, candidates in self._presented.items():
            terminal = self._terminal_indices.get(key)
            latched = self._latched_priority_page(key, terminal)
            if latched is not None:
                priorities[key] = latched
                continue
            neighbours = []
            evidence = []
            for candidate in candidates:
                position = getattr(candidate, "position", None)
                if position is not None and isfinite(position):
                    lower, upper = floor(position), ceil(position)
                    # At an exact stop, include the stops on both sides too.
                    neighbours.extend((lower, upper) if lower != upper
                                      else (lower - 1, lower, lower + 1))
                evidence.extend(sorted(candidate.priority_indices))
                evidence.extend(sorted(candidate.exploratory_indices))
            population = list(dict.fromkeys(
                index for index in (*neighbours, *evidence)
                if index >= 0 and (terminal is None or index <= terminal)
            ))
            queue = self._priority_queue.setdefault(key, [])
            current = set(population)
            queue[:] = [index for index in queue if index in current]
            queue.extend(index for index in population if index not in queue)
            window = tuple(queue[:MAX_PRIORITY_ENDPOINTS_PER_ROUTE])
            if window:
                self._priority_pending[key] = (window, set(window))
                priorities[key] = frozenset(window)
        for key in set(self._priority_queue) - self._presented.keys():
            self._priority_queue.pop(key, None)
            self._priority_pending.pop(key, None)
        return priorities

    def _latched_priority_page(self, key, terminal):
        """Return one full physical page until every valid group was attempted."""
        while True:
            page = self._priority_pending.get(key)
            if page is None:
                return None

            window, remaining = page
            queue = self._priority_queue.setdefault(key, [])
            retained = [
                index for index in window
                if terminal is None or index < terminal
            ]
            invalid = set(window) - set(retained)
            if invalid:
                queue[:] = [index for index in queue if index not in invalid]
            remaining.intersection_update(retained)
            limit = MAX_PRIORITY_ENDPOINTS_PER_ROUTE - (
                1 if terminal is not None else 0
            )
            while len(retained) > limit:
                # Keep every unattempted obligation if an already serviced
                # endpoint can make room for a newly reserved terminus.
                displaced = next(
                    (index for index in reversed(retained) if index not in remaining),
                    retained[-1],
                )
                retained.remove(displaced)
                if displaced in remaining:
                    remaining.remove(displaced)
                    if displaced in queue:
                        queue.remove(displaced)
                    queue.insert(0, displaced)
                elif displaced in queue:
                    queue.remove(displaced)
            if not remaining:
                self._priority_pending.pop(key, None)
                serviced = set(retained)
                queue[:] = [index for index in queue if index not in serviced]
                continue
            normalized = (tuple(retained), remaining)
            self._priority_pending[key] = normalized
            return frozenset(retained)

    def _ack_probe_attempts(self, snapshot):
        generation = getattr(snapshot, "probe_attempt_generation", None)
        if (not isinstance(generation, int) or isinstance(generation, bool)
                or generation <= 0
                or (self._probe_ack_generation is not None
                    and generation <= self._probe_ack_generation)):
            return
        self._probe_ack_generation = generation
        attempted = getattr(snapshot, "attempted_checkpoints", ()) or ()
        by_route = {}
        for item in attempted:
            if not isinstance(item, (tuple, list)) or len(item) != 4:
                continue
            operator, route, bound, index = item
            physical = _physical_indices((index,))
            if not physical:
                continue
            by_route.setdefault(
                (str(operator), str(route), str(bound)), set()
            ).add(physical[0])
        for key, pending in list(self._priority_pending.items()):
            _window, remaining = pending
            remaining.difference_update(by_route.get(key, ()))

    def poll_lifecycle_routes(self):
        """Return routes whose partial ETA population needs a full refresh."""
        return frozenset(self._lifecycle_refresh_routes)

    def poll_lifecycle_requests(self):
        """Return refresh routes fenced to their last reconciled generation."""
        return {
            key: self._generations.get(key)
            for key in self._lifecycle_refresh_routes
        }

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
            else:
                self._routes[key] = tracks = self._sort_tracks(tracks)
            split_ties.setdefault(key, _tie_components(tracks))
            route_rows = tuple(row for row in getattr(snapshot, "rows", ())
                               if _key(row) == key)
            checkpoints = _successful_checkpoints(route_rows)
            for track in tracks.values():
                # Partial responses refine motion/disappearance witnesses;
                # only an accepted complete population replaces the cohort.
                # Row-local pruning here destroys exact reservations before
                # the next complete generation can account for all owners.
                _refresh_forward_search(
                    track, checkpoints, self._terminal_indices.get(key)
                )
            _advance_forward_search_from_candidates(tracks, rows, checkpoints,
                                                     self._terminal_indices.get(key))
            if generation is not None and generation != self._generations.get(key):
                self._lifecycle_refresh_routes.discard(key)
                self._partial_birth_generations.pop(key, None)
                old_generation = self._generations.get(key)
                rollback = old_generation is not None and generation < old_generation
                if rollback:
                    tracks.clear()
                    old_generation = None
                self._generations[key] = generation
                old = sorted(tracks.values(), key=lambda track: (track.position, track.track_id))
                held_positions = {
                    track.track_id: track.position for track in old
                }
                self._predict(old, now, route_lines)
                rows, reconciled_owners = _reconcile_complete_probe_ownership(
                    old, rows, now, tuple(getattr(snapshot, "rows", ()) or ()),
                    route_lines,
                )
                # A positively mixed birth has display evidence but no trusted
                # identity ledger. Do not let it claim later clean occurrences.
                trusted_owners = [track for track in old
                                  if track.cohort_observed_at <= 0 or track.cohort_trusted]
                matching_old = [_matching_track(track) for track in old]
                complete_ownership_cache = {
                    id(candidate): _checkpoint_ownership_profile(trusted_owners, candidate)
                    for candidate in rows
                }
                mixed_candidates = {
                    id(candidate) for candidate in rows
                    if id(candidate) not in reconciled_owners
                    and (complete_ownership_cache[id(candidate)][-1]
                         or _incoherent_checkpoint_chronology(candidate))
                }

                def complete_identity_compatible(track, candidate, route_tracks=trusted_owners,
                                         cache=complete_ownership_cache,
                                         certified=reconciled_owners,
                                         untrusted=mixed_candidates):
                    certified_owner = certified.get(id(candidate))
                    if certified_owner is not None:
                        return track.track_id == certified_owner
                    if id(candidate) in untrusted:
                        return False
                    cache_key = id(candidate)
                    if cache_key not in cache:
                        cache[cache_key] = _checkpoint_ownership_profile(
                            route_tracks, candidate
                        )
                    exact_ids, covering_ids, prior_owned, _positioning, mixed_ids = \
                        cache[cache_key]
                    # Complete cardinality remains authoritative, but a
                    # candidate with positive ownership for another survivor
                    # must not be used as a fallback continuation here.
                    return not (mixed_ids or (covering_ids and track.track_id not in covering_ids))
                def complete_compatible(track, candidate):
                    return (complete_identity_compatible(track, candidate)
                            and _search_compatible(track, candidate))
                fixed_pairs, movement_pairs, pairs, cold_reserved = _complete_pair_plan(
                    matching_old,
                    rows,
                    checkpoints,
                    compatible=complete_compatible,
                    identity_compatible=complete_identity_compatible,
                    reconciled_owners=reconciled_owners,
                )
                # A complete publication can contain one newly discovered,
                # coarse downstream bracket while a held marker is already
                # searching forward.  Consume that candidate for lifecycle
                # cardinality, but defer movement until a narrow requested
                # corridor arrives.  Restrict this to an unambiguous 1:1
                # route association; competing tracks fail closed.
                hold_pairs = {
                    pair for pair in _forward_hold_pairs(matching_old, rows)
                    if complete_compatible(matching_old[pair[0]], rows[pair[1]])
                }
                if hold_pairs:
                    pairs = sorted(set(pairs) | hold_pairs)
                probe_inputs = tuple(getattr(snapshot, "rows", ()) or ())
                fragment_assignments = _certified_cohort_fragments(
                    matching_old, rows, pairs, now, probe_inputs
                )
                fragments_by_old = {}
                for new_index, old_index in fragment_assignments.items():
                    fragments_by_old.setdefault(old_index, []).append(new_index)
                cold_fragment_pairs = set()
                for old_index, fragment_indices in fragments_by_old.items():
                    base_index = dict(pairs)[old_index]
                    base = rows[base_index]
                    rows[base_index] = rebuild_estimate_from_probe_fragments(
                        base,
                        [rows[index] for index in fragment_indices],
                        probe_inputs,
                        route_lines,
                    )
                    rebuilt = rows[base_index]
                    if (rebuilt is base
                            or getattr(rebuilt, "position_authoritative", None) is False):
                        # A missing lower frontier can leave reconstruction
                        # unchanged even if the base alone was authoritative.
                        # Consuming certified fragments must still publish
                        # their evidence and poll hints without moving.
                        constituents = [rebuilt, *(rows[index] for index in fragment_indices)]
                        metadata = {
                            name: frozenset().union(*(getattr(item, name) for item in constituents))
                            for name in ("source_indices", "source_observations",
                                         "priority_indices", "exploratory_indices")
                        }
                        rows[base_index] = replace(
                            rebuilt, **metadata, position_authoritative=False,
                            checkpoint_evidence=tuple(sorted({
                                row for item in constituents for row in _checkpoint_rows(item)
                            })),
                        )
                        cold_fragment_pairs.add((old_index, base_index))
                # Fragment reconciliation may restore a real boundary. Only
                # candidates still explicitly cold are metadata-only holds.
                cold_holds = {
                    pair for pair in cold_reserved | cold_fragment_pairs
                    if getattr(rows[pair[1]], "position_authoritative", None) is False
                }
                hold_pairs |= cold_holds
                stable_gate_pairs = {
                    (old_index, new_index)
                    for old_index, new_index in pairs
                    if _stable_gate_position(old[old_index], rows[new_index]) is not None
                }
                proposed_positions = {
                    old_index: position
                    for old_index, new_index in pairs
                    if (old_index, new_index) not in hold_pairs
                    if (position := (
                        _stable_gate_position(old[old_index], rows[new_index])
                        if (old_index, new_index) in stable_gate_pairs
                        else _paired_position(old[old_index], rows[new_index],
                                              (old_index, new_index) in movement_pairs)
                    )) is not None
                }
                # A marker created before its first exact motion boundary can
                # legitimately be paired with the same physical ETA several
                # stops downstream. Ordinary motion still rejects that coarse
                # jump. When a new complete census proves mutually exclusive
                # occurrence ownership, replace the internal track atomically
                # at the observed boundary instead of attaching the fresh ETA
                # ledger to stale marker geometry.
                potential_reseeds = [
                    (old_index, new_index)
                    for old_index, new_index in pairs
                    if (old_index, new_index) in fixed_pairs
                    and (old_index, new_index) not in cold_holds
                    and old_index not in proposed_positions
                    and _first_boundary_reseed_eligible(
                        old[old_index], rows[new_index], now,
                    )
                ]
                reseed_context = (
                    _first_boundary_reseed_context(
                        old, rows, probe_inputs, complete.get(key), now,
                    )
                    if potential_reseeds else None
                )
                boundary_reseeds = {}
                if reseed_context is not None:
                    for old_index, new_index in potential_reseeds:
                        position = _certified_first_boundary_reseed_position(
                            old,
                            rows,
                            old_index,
                            new_index,
                            probe_inputs,
                            complete.get(key),
                            now,
                            context=reseed_context,
                        )
                        if position is not None:
                            boundary_reseeds[old_index] = position
                accepted_updates = _select_ordered_updates(
                    old, proposed_positions,
                    eligible_indices={
                        old_index for old_index, new_index in pairs
                        if old_index not in boundary_reseeds
                        if (
                            old_index in proposed_positions
                            or _position_order_authoritative(
                                old[old_index], rows[new_index]
                            )
                        )
                    },
                )
                used = set(fragment_assignments)
                reseeded_old = set()
                replacement_positions = {}
                for old_index, new_index in pairs:
                    track = old[old_index]
                    candidate = rows[new_index]
                    used.add(new_index)
                    if old_index in boundary_reseeds:
                        reseeded_old.add(old_index)
                        replacement_positions[new_index] = boundary_reseeds[old_index]
                        continue
                    if old_index not in accepted_updates:
                        # Complete lifecycle evidence can confirm an identity,
                        # but stale, unbracketed, or order-crossing positioning
                        # evidence must retain the last exact boundary.
                        _hold_track(track, held_positions[track.track_id])
                        if (old_index, new_index) in cold_holds:
                            track.estimate = replace(
                                track.estimate,
                                track_id=track.track_id,
                                operator_code=key[0],
                                source_indices=getattr(candidate, "source_indices", track.estimate.source_indices),
                                source_observations=getattr(candidate, "source_observations", track.estimate.source_observations),
                                checkpoint_evidence=getattr(candidate, "checkpoint_evidence", track.estimate.checkpoint_evidence),
                                priority_indices=getattr(candidate, "priority_indices", track.estimate.priority_indices),
                                exploratory_indices=getattr(candidate, "exploratory_indices", track.estimate.exploratory_indices),
                                position_authoritative=False,
                            )
                        else:
                            track.estimate = replace(candidate, track_id=track.track_id,
                                                     operator_code=key[0])
                        track.generation = generation
                        track.last_evidence_at = now
                    else:
                        track.position = proposed_positions[old_index]
                        track.estimate = replace(
                            candidate,
                            track_id=track.track_id,
                            operator_code=key[0],
                        )
                        track.position_authoritative = True
                        track.generation = generation
                        track.last_evidence_at = now
                        track.display_bracket = getattr(candidate, "bracket", None)
                        if (old_index, new_index) not in stable_gate_pairs:
                            track.boundary_observed_at = _boundary_observed_at(candidate, now)
                            track.boundary_revision = _candidate_revision(candidate)
                            track.motion_bracket = getattr(candidate, "bracket", None)
                            _commit_boundary_evidence(track, candidate)
                            track.display_bracket = getattr(candidate, "bracket", None)
                            _clear_forward_search(track)
                    _replace_track_cohort(
                        track,
                        [candidate, *(
                            rows[index]
                            for index in fragments_by_old.get(old_index, ())
                        )],
                        now,
                    )
                births = []
                for index, candidate in enumerate(rows):
                    if index in replacement_positions:
                        position = replacement_positions[index]
                    elif index in used:
                        continue
                    else:
                        position = float(candidate.position or 0.0)
                    track_id = self._next_id
                    self._next_id += 1
                    births.append(_Track(
                        track_id=track_id,
                        estimate=replace(
                            candidate, track_id=track_id, operator_code=key[0]
                        ),
                        position=position,
                        generation=generation,
                        last_evidence_at=now,
                        boundary_observed_at=_boundary_observed_at(candidate, now),
                        boundary_revision=_candidate_revision(candidate),
                        motion_bracket=getattr(candidate, "bracket", None),
                        display_bracket=getattr(candidate, "bracket", None),
                        cohort_evidence=(
                            () if id(candidate) in mixed_candidates else _checkpoint_rows(candidate)
                        ),
                        cohort_observed_at=now,
                        cohort_trusted=id(candidate) not in mixed_candidates,
                        committed_boundary_evidence=_checkpoint_rows(candidate),
                        position_authoritative=(
                            getattr(candidate, "position_authoritative", None)
                            if getattr(candidate, "position_authoritative", None) is not None
                            else (
                                not bool(getattr(candidate, "unreliable", False))
                                or _candidate_revision(candidate) is not None
                            )
                        ),
                    ))
                matched_old = {old_index for old_index, _ in pairs}
                # A complete all-stop generation is the lifecycle authority.
                # Keeping an unmatched prior track for another generation
                # renders a ghost alongside the replacement ETA instance.
                old_survivors = [
                    track for index, track in enumerate(old)
                    if index in matched_old and index not in reseeded_old
                ]
                old_survivors.sort(key=lambda track: track.position)
                merged = _merge_tracks(old_survivors, births)
                tracks.clear()
                tracks.update((track.track_id, track) for track in merged)
            elif generation is not None:
                old = sorted(tracks.values(), key=lambda track: (track.position, track.track_id))
                matching_old = [_matching_track(track) for track in old]
                trusted_owners = [track for track in old
                                  if track.cohort_observed_at <= 0 or track.cohort_trusted]
                if (
                    len(rows) != len(old)
                    or _equal_count_population_turnover(
                        old, rows, checkpoints, matching_old=matching_old,
                    )
                ):
                    # Partial positioning rows are useful evidence that the
                    # lifecycle population may have changed. This includes an
                    # equal-count swap where a refreshed checkpoint proves an
                    # old physical occurrence disappeared and the current
                    # candidates cannot cover every track one-to-one. Partial
                    # evidence is never authoritative enough to birth or
                    # retire a marker: ask the orchestrator to prioritize the
                    # complete fixed baseline and retain the request until a
                    # different atomic generation is reconciled.
                    self._lifecycle_refresh_routes.add(key)
                if not old:
                    self._bound()
                    continue
                # A publication may be re-rendered with aged/corrected ETA
                # rows. Refresh matched tracks, but never alter cardinality
                # until a newer complete generation arrives.
                ownership_cache = {}
                reservations, blocked_old, blocked_new = _same_generation_identity_plan(
                    old, rows, now,
                )
                reserved_old = dict(reservations)
                reserved_candidates = {id(rows[index]): owner for owner, index in reservations}
                blocked_candidates = {id(rows[index]) for index in blocked_new}
                old_indices = {track.track_id: index for index, track in enumerate(old)}
                if blocked_old or blocked_new:
                    self._lifecycle_refresh_routes.add(key)

                def checkpoint_ownership(candidate, route_tracks=trusted_owners,
                                         cache=ownership_cache):
                    cache_key = id(candidate)
                    if cache_key not in cache:
                        cache[cache_key] = _checkpoint_ownership_profile(
                            route_tracks, candidate
                        )
                    return cache[cache_key]

                def refresh_compatible(track, candidate, route_tracks=matching_old,
                                       identity_context=(old, old_indices, reserved_old,
                                                         reserved_candidates, blocked_old,
                                                         blocked_candidates)):
                    (real_old, old_indices, reserved_old, reserved_candidates,
                     blocked_old, blocked_candidates) = identity_context
                    old_index = old_indices[track.track_id]
                    if old_index in reserved_old or id(candidate) in reserved_candidates:
                        return (reserved_candidates.get(id(candidate)) == old_index
                                and _candidate_actionable(candidate, real_old[old_index]))
                    if old_index in blocked_old or id(candidate) in blocked_candidates:
                        return False
                    return _same_generation_actionable(
                        route_tracks, track, candidate,
                        ownership=checkpoint_ownership(candidate),
                    )

                # A positively certified mixed-owner candidate is a lifecycle
                # signal even when partial cardinality happens to be equal.
                # Keep weak/ambiguous ownership inconclusive.
                if any(checkpoint_ownership(candidate)[-1] for candidate in rows):
                    self._lifecycle_refresh_routes.add(key)

                certified_surplus_index = None
                if self._partial_birth_generations.get(key) != generation:
                    certified_surplus_index = _certified_partial_birth_index(
                        old,
                        rows,
                        tuple(getattr(snapshot, "rows", ()) or ()),
                        complete.get(key),
                        now,
                    )
                partial_birth_index = (
                    certified_surplus_index
                    if len(old) < self.max_tracks_per_route
                    else None
                )

                fresh_indices = [
                    index for index, candidate in enumerate(rows)
                    if index != certified_surplus_index
                    if _fresh_bracket_position(candidate) is not None
                    and any(
                        refresh_compatible(track, candidate)
                        for track in matching_old
                    )
                ]
                fresh_rows = [rows[index] for index in fresh_indices]
                fixed_pairs, recovery_pairs, movement_pairs = _recovery_plan(
                    matching_old,
                    fresh_rows,
                    checkpoints,
                    compatible=refresh_compatible,
                )
                pairs = _ordered_pairs(
                    matching_old, fresh_rows,
                    compatible=refresh_compatible,
                    recoveries=recovery_pairs,
                    fixed=fixed_pairs,
                )
                proposed_positions = {
                    old_index: position
                    for old_index, new_index in pairs
                    if (position := _paired_position(
                        old[old_index], fresh_rows[new_index],
                        (old_index, new_index) in movement_pairs,
                    )) is not None
                }
                # Identity reservations consume both endpoints even when a
                # boundary is stale, cold, too distant, or blocked by order.
                # Publish their current metadata in the same capacity check
                # as motion, so a held owner cannot duplicate a moving one.
                pairs = [(old_index, fresh_indices[new_index])
                         for old_index, new_index in pairs]
                pairs = sorted(set(pairs) | reservations)
                accepted_updates, birth_allowed, accepted_motion = _select_valid_partial_transaction(
                    old,
                    rows,
                    rows,
                    pairs,
                    proposed_positions,
                    metadata_indices=set(reserved_old),
                    birth_candidate=(
                        rows[partial_birth_index]
                        if partial_birth_index is not None else None
                    ),
                )
                if not birth_allowed:
                    partial_birth_index = None
                for old_index, new_index in pairs:
                    if old_index not in accepted_updates:
                        continue
                    track = old[old_index]
                    candidate = rows[new_index]
                    if old_index not in accepted_motion:
                        _refresh_held_metadata(track, candidate)
                        continue
                    _record_metadata_revisions(track, candidate)
                    track.position = proposed_positions[old_index]
                    track.estimate = replace(candidate, track_id=track.track_id,
                                             operator_code=key[0])
                    track.position_authoritative = True
                    track.last_evidence_at = now
                    track.boundary_observed_at = _boundary_observed_at(candidate, now)
                    track.boundary_revision = _candidate_revision(candidate)
                    track.motion_bracket = getattr(candidate, "bracket", None)
                    track.display_bracket = getattr(candidate, "bracket", None)
                    _commit_boundary_evidence(track, candidate)
                    _clear_forward_search(track)
                if partial_birth_index is not None:
                    candidate = rows[partial_birth_index]
                    track_id = self._next_id
                    self._next_id += 1
                    tracks[track_id] = _Track(
                        track_id=track_id,
                        estimate=replace(
                            candidate, track_id=track_id, operator_code=key[0]
                        ),
                        position=float(candidate.position or 0.0),
                        generation=generation,
                        last_evidence_at=now,
                        boundary_observed_at=_boundary_observed_at(candidate, now),
                        boundary_revision=_candidate_revision(candidate),
                        motion_bracket=getattr(candidate, "bracket", None),
                        display_bracket=getattr(candidate, "bracket", None),
                        cohort_evidence=(),
                        cohort_observed_at=now,
                        cohort_trusted=False,
                        committed_boundary_evidence=_checkpoint_rows(candidate),
                        position_authoritative=(
                            getattr(candidate, "position_authoritative", None)
                            if getattr(candidate, "position_authoritative", None) is not None
                            else (
                                not bool(getattr(candidate, "unreliable", False))
                                or _candidate_revision(candidate) is not None
                            )
                        ),
                    )
                    self._partial_birth_generations[key] = generation
                    self._routes[key] = self._sort_tracks(tracks)
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
                        self._lifecycle_refresh_routes.discard(key)
                        self._partial_birth_generations.pop(key, None)
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
            self._priority_queue.pop(key, None)
            self._priority_pending.pop(key, None)
            self._lifecycle_refresh_routes.discard(key)
            self._partial_birth_generations.pop(key, None)
        active_keys = set(self._routes)
        for state in (
            self._priority_queue,
            self._priority_pending,
        ):
            for key in list(state):
                if key not in active_keys:
                    state.pop(key, None)
        for key in list(self._terminal_indices):
            if key not in active_keys:
                self._terminal_indices.pop(key, None)
                self._priority_queue.pop(key, None)
                self._priority_pending.pop(key, None)
        self._lifecycle_refresh_routes.intersection_update(active_keys)
        self._partial_birth_generations = {
            key: generation
            for key, generation in self._partial_birth_generations.items()
            if key in active_keys
        }


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
    observations = tuple(sorted(
        (source[0], source[1], "")
        if isinstance(source, tuple) and len(source) == 2
        and isinstance(source[0], str) and isinstance(source[1], int)
        else ("", inf, repr(source))
        for source in getattr(item, "source_observations", ()) or ()
    ))
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


def _incoherent_checkpoint_chronology(estimate):
    """A downstream stop cannot precede this ladder's upstream arrival.

    Equal-stop occurrences remain separate, and equal arrival times across
    consecutive due checkpoints are not themselves evidence of contamination.
    """
    rows = sorted(_checkpoint_rows(estimate))
    return any(right[0] > left[0] and right[1] < left[1]
               for left, right in zip(rows, rows[1:], strict=False))


def _commit_boundary_evidence(track, candidate):
    """Retain bounded exact checkpoint occurrences for the motion witness."""
    track.committed_boundary_evidence = _checkpoint_rows(candidate)[:MAX_CHECKPOINT_EVIDENCE]


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


def _strict_checkpoint_rows(estimate):
    """Return an immutable ledger only when every entry is well formed."""
    raw = getattr(estimate, "checkpoint_evidence", ())
    if not isinstance(raw, tuple):
        return None
    rows = []
    for item in raw:
        if (
            not isinstance(item, tuple)
            or len(item) != 3
            or not isinstance(item[0], int)
            or isinstance(item[0], bool)
            or item[0] < 0
            or not isinstance(item[1], (int, float))
            or isinstance(item[1], bool)
            or not isfinite(item[1])
            or not isinstance(item[2], int)
            or isinstance(item[2], bool)
            or item[2] <= 0
        ):
            return None
        rows.append((item[0], float(item[1]), item[2]))
    return tuple(rows)


def _complete_checkpoint_revision_floors(complete_route, route_key):
    """Validate one atomic route's exact per-checkpoint response revisions."""
    if tuple(getattr(complete_route, "route_key", ())) != route_key:
        return None
    complete_rows = getattr(complete_route, "rows", None)
    observed_indices = getattr(complete_route, "observed_checkpoint_indices", None)
    checkpoint_revisions = getattr(complete_route, "checkpoint_revisions", None)
    if (
        not isinstance(complete_rows, tuple)
        or not isinstance(observed_indices, frozenset)
        or not isinstance(checkpoint_revisions, tuple)
        or any(
            not isinstance(index, int) or isinstance(index, bool) or index < 0
            for index in observed_indices
        )
    ):
        return None
    floors = {}
    for item in checkpoint_revisions:
        if (
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], int)
            or isinstance(item[0], bool)
            or item[0] < 0
            or not isinstance(item[1], int)
            or isinstance(item[1], bool)
            or item[1] <= 0
            or item[0] in floors
        ):
            return None
        floors[item[0]] = item[1]
    if set(floors) != set(observed_indices):
        return None
    for row in complete_rows:
        index = getattr(row, "index", None)
        revision = getattr(row, "refresh_generation", None)
        if (
            _key(row) != route_key
            or not isinstance(index, int)
            or isinstance(index, bool)
            or index < 0
            or not isinstance(revision, int)
            or isinstance(revision, bool)
            or revision <= 0
            or floors.get(index) != revision
        ):
            return None
    return floors


def _partial_birth_live_row(row, route_key):
    """Normalize one fresh live response row used by a partial-birth proof."""
    index = getattr(row, "index", None)
    revision = getattr(row, "refresh_generation", None)
    minutes = getattr(row, "minutes", None)
    signed_minutes = getattr(row, "signed_minutes", None)
    age = getattr(row, "cache_age_seconds", None)
    kind = getattr(row, "kind", None)
    arrival_value = getattr(row, "arrival_at", None)
    if (
        _key(row) != route_key
        or not isinstance(index, int)
        or isinstance(index, bool)
        or index < 0
        or not isinstance(revision, int)
        or isinstance(revision, bool)
        or revision <= 0
        or isinstance(minutes, bool)
        or isinstance(signed_minutes, bool)
        or isinstance(age, bool)
        or isinstance(arrival_value, bool)
        or all(kind != live_kind for live_kind in LIVE_PROBE_ETA_KINDS)
    ):
        return None
    try:
        minutes = float(minutes)
        signed_minutes = float(signed_minutes)
        age = float(age)
    except (TypeError, ValueError, OverflowError):
        return None
    arrival = _arrival_timestamp(arrival_value)
    if (
        arrival is None
        or not isfinite(minutes)
        or not isfinite(signed_minutes)
        or minutes < 0
        or abs(minutes - max(0.0, signed_minutes)) > 1e-6
        or not isfinite(age)
        or not 0 <= age < PARTIAL_BIRTH_FRESHNESS_SECONDS
    ):
        return None
    return index, arrival, revision, minutes, age


def _partial_birth_endpoint_row(row, route_key, index):
    """Return revision/age for a current boundary response, including empty."""
    row_index = getattr(row, "index", None)
    revision = getattr(row, "refresh_generation", None)
    age = getattr(row, "cache_age_seconds", None)
    if (
        _key(row) != route_key
        or row_index != index
        or isinstance(row_index, bool)
        or not isinstance(row_index, int)
        or row_index < 0
        or not isinstance(revision, int)
        or isinstance(revision, bool)
        or revision <= 0
        or isinstance(age, bool)
    ):
        return None
    try:
        age = float(age)
    except (TypeError, ValueError, OverflowError):
        return None
    if not isfinite(age) or not 0 <= age < PARTIAL_BIRTH_FRESHNESS_SECONDS:
        return None
    return revision, age


def _first_boundary_reseed_eligible(track, candidate, now):
    """Return whether a pair is cheap to consider before route-wide proof."""
    return (
        track.boundary_revision is None
        and getattr(track.estimate, "boundary_revision", None) is None
        and track.forward_after is None
        and track.cohort_observed_at > 0
        and track.cohort_trusted
        and 0 <= now - track.cohort_observed_at <= COHORT_EVIDENCE_TTL_SECONDS
        and not bool(getattr(candidate, "unreliable", False))
        and _candidate_actionable(candidate, track)
    )


def _first_boundary_reseed_context(
    old, candidates, probe_inputs, complete_route, now,
):
    """Validate and index one route population for bounded reseed proofs."""
    if (
        not old
        or not candidates
        or len(old) > MAX_TRACKS_PER_ROUTE
        or len(candidates) > MAX_TRACKS_PER_ROUTE
    ):
        return None
    route_key = _key(candidates[0])
    complete_floors = _complete_checkpoint_revision_floors(
        complete_route, route_key,
    )
    if complete_floors is None:
        return None

    old_nodes = {}
    revision_floors = dict(complete_floors)
    for owner_index, owner in enumerate(old):
        raw_rows = _cohort_rows(owner)
        if (
            _key(owner.estimate) != route_key
            or owner.cohort_observed_at <= 0
            or not owner.cohort_trusted
            or not 0 <= now - owner.cohort_observed_at
            <= COHORT_EVIDENCE_TTL_SECONDS
            or not raw_rows
            or len(raw_rows) > MAX_CHECKPOINT_EVIDENCE
        ):
            return None
        stops = set()
        for row_number, row in enumerate(raw_rows):
            if (
                not isinstance(row, tuple)
                or len(row) != 3
                or not isinstance(row[0], int)
                or isinstance(row[0], bool)
                or row[0] < 0
                or row[0] in stops
                or not isinstance(row[1], (int, float))
                or isinstance(row[1], bool)
                or not isfinite(row[1])
                or not isinstance(row[2], int)
                or isinstance(row[2], bool)
                or row[2] <= 0
            ):
                return None
            stops.add(row[0])
            normalized = (row[0], float(row[1]), row[2])
            old_nodes[(owner_index, row_number)] = normalized
            revision_floors[row[0]] = max(
                revision_floors.get(row[0], 0), row[2],
            )

    probe_inputs = tuple(probe_inputs)
    route_slots = {}
    route_revisions = {}
    required_probe_slots = set()
    for slot, row in enumerate(probe_inputs):
        if _key(row) != route_key:
            continue
        index = getattr(row, "index", None)
        endpoint = _partial_birth_endpoint_row(row, route_key, index)
        if (
            endpoint is None
            or endpoint[0] < revision_floors.get(index, 0)
            or (
                index in route_revisions
                and route_revisions[index] != endpoint[0]
            )
        ):
            return None
        route_revisions[index] = endpoint[0]
        route_slots.setdefault(index, []).append(slot)
        if getattr(row, "minutes", None) is not None:
            required_probe_slots.add(slot)

    seen_sources = set()
    seen_probe_slots = set()
    seen_probe_objects = set()
    candidate_slots = []
    candidate_rows = []
    for candidate in candidates:
        sources = getattr(candidate, "source_observations", None)
        strict_rows = _strict_checkpoint_rows(candidate)
        if (
            _key(candidate) != route_key
            or bool(getattr(candidate, "unreliable", False))
            or not isinstance(sources, frozenset)
            or not sources
            or strict_rows is None
            or len(strict_rows) > MAX_CHECKPOINT_EVIDENCE
        ):
            return None
        slots = []
        rows = []
        stops = set()
        for source in sorted(sources, key=repr):
            if (
                not isinstance(source, tuple)
                or len(source) != 2
                or source[0] not in {"gate", "probe"}
                or not isinstance(source[1], int)
                or isinstance(source[1], bool)
                or source[1] < 0
                or source in seen_sources
            ):
                return None
            seen_sources.add(source)
            if source[0] == "gate":
                continue
            slot = source[1]
            if (
                slot >= len(probe_inputs)
                or slot in seen_probe_slots
                or id(probe_inputs[slot]) in seen_probe_objects
            ):
                return None
            normalized = _partial_birth_live_row(probe_inputs[slot], route_key)
            if (
                normalized is None
                or normalized[0] in stops
                or normalized[2] < revision_floors.get(normalized[0], 0)
            ):
                return None
            seen_probe_slots.add(slot)
            seen_probe_objects.add(id(probe_inputs[slot]))
            stops.add(normalized[0])
            slots.append(slot)
            rows.append(normalized)
        rows.sort()
        if (
            Counter(strict_rows) != Counter(
                (index, arrival, revision)
                for index, arrival, revision, _minutes, _age in rows
            )
            or any(
                right[1] < left[1]
                for left, right in zip(rows, rows[1:], strict=False)
            )
        ):
            return None
        candidate_slots.append(tuple(slots))
        candidate_rows.append(tuple(rows))
    if seen_probe_slots != required_probe_slots:
        return None

    current_nodes = {
        (candidate_index, row_number): (index, arrival, revision)
        for candidate_index, rows in enumerate(candidate_rows)
        for row_number, (index, arrival, revision, _minutes, _age)
        in enumerate(rows)
    }

    def stop_index(nodes):
        grouped = {}
        for node, (index, arrival, revision) in nodes.items():
            grouped.setdefault(index, []).append((arrival, node, revision))
        indexed = {}
        for index, values in grouped.items():
            ordered = tuple(sorted(values))
            indexed[index] = (tuple(item[0] for item in ordered), ordered)
        return indexed

    old_by_stop = stop_index(old_nodes)

    def neighbours(current, tolerance):
        index, arrival, revision = current
        arrivals, entries = old_by_stop.get(index, ((), ()))
        lower = bisect_left(arrivals, arrival - tolerance)
        upper = bisect_right(arrivals, arrival + tolerance)
        if upper - lower > MAX_RESEED_TEMPORAL_NEIGHBOURS:
            return None
        return tuple(
            (node, (index, prior_arrival, prior_revision))
            for prior_arrival, node, prior_revision in entries[lower:upper]
            if revision >= prior_revision
        )

    exact = {}
    exact_counts = Counter()
    for current_node, current in current_nodes.items():
        nearby = neighbours(current, 0.5)
        if nearby is None:
            return None
        edges = {
            old_node for old_node, prior in nearby
            if abs(current[1] - prior[1]) <= 0.5
        }
        exact[current_node] = edges
        exact_counts.update(edges)
    reserved = {
        current_node: next(iter(edges))
        for current_node, edges in exact.items()
        if len(edges) == 1 and exact_counts[next(iter(edges))] == 1
    }
    consumed = set(reserved.values())
    graph = {}
    reverse_graph = {old_node: set() for old_node in old_nodes}
    edge_count = 0
    for current_node, current in current_nodes.items():
        if current_node in reserved:
            edges = {reserved[current_node]}
        else:
            nearby = neighbours(current, RECOVERY_ARRIVAL_TOLERANCE_SECONDS)
            if nearby is None:
                return None
            edges = {
                old_node for old_node, prior in nearby
                if old_node not in consumed and _evidence_continues(prior, current)
            }
        edge_count += len(edges)
        if edge_count > MAX_RESEED_TEMPORAL_EDGES:
            return None
        graph[current_node] = frozenset(edges)
        for old_node in edges:
            reverse_graph[old_node].add(current_node)

    # A legitimate route segment can take longer than the generic chronology
    # ceiling.  Such a segment is safe here only when both of its endpoints
    # independently continue the same trusted frozen owner.  This corroborates
    # the current source partition; it does not grant motion authority itself.
    for candidate_index, rows in enumerate(candidate_rows):
        for row_number, (left, right) in enumerate(
            zip(rows, rows[1:], strict=False),
        ):
            if (
                right[1] - left[1]
                <= COHORT_MAX_SECONDS_PER_STOP * (right[0] - left[0])
            ):
                continue
            left_edges = graph[(candidate_index, row_number)]
            right_edges = graph[(candidate_index, row_number + 1)]
            if len(left_edges) != 1 or len(right_edges) != 1:
                return None
            left_owner = next(iter(left_edges))
            right_owner = next(iter(right_edges))
            if (
                left_owner[0] != right_owner[0]
                or any(
                    node[0] != candidate_index
                    for node in reverse_graph[left_owner]
                )
                or any(
                    node[0] != candidate_index
                    for node in reverse_graph[right_owner]
                )
            ):
                return None

    return {
        "route_key": route_key,
        "probe_inputs": probe_inputs,
        "route_slots": route_slots,
        "candidate_slots": tuple(candidate_slots),
        "candidate_rows": tuple(candidate_rows),
        "old_nodes": old_nodes,
        "current_nodes": current_nodes,
        "graph": graph,
        "reverse_graph": {
            node: frozenset(edges) for node, edges in reverse_graph.items()
        },
        "reserved": reserved,
    }


def _certified_first_boundary_reseed_position(
    old, candidates, old_index, candidate_index, probe_inputs, complete_route, now,
    *, context=None,
):
    """Prove one fresh atomic replacement without claiming a long-jump identity."""
    if (
        not 0 <= old_index < len(old)
        or not 0 <= candidate_index < len(candidates)
    ):
        return None
    track = old[old_index]
    candidate = candidates[candidate_index]
    if not _first_boundary_reseed_eligible(track, candidate, now):
        return None
    if context is None:
        context = _first_boundary_reseed_context(
            old, candidates, probe_inputs, complete_route, now,
        )
    route_key = _key(candidate)
    sources = getattr(candidate, "source_observations", None)
    if (
        context is None
        or context["route_key"] != route_key
        or not isinstance(sources, frozenset)
        or not sources
        or any(
            not isinstance(source, tuple)
            or len(source) != 2
            or source[0] != "probe"
            or not isinstance(source[1], int)
            or isinstance(source[1], bool)
            for source in sources
        )
    ):
        return None
    probe_inputs = context["probe_inputs"]
    normalized = context["candidate_rows"][candidate_index]
    slots = context["candidate_slots"][candidate_index]
    if (
        not normalized
        or max(row[4] for row in normalized) - min(row[4] for row in normalized)
        > PARTIAL_BIRTH_MAX_AGE_SKEW_SECONDS
    ):
        return None

    # Exact reservations outrank temporal fallback. Every remaining plausible
    # continuation within the provider's drift window must still be exclusive
    # to this old/current pair in both directions.
    target_current_nodes = {
        node for node in context["current_nodes"] if node[0] == candidate_index
    }
    target_old_nodes = {
        node for node in context["old_nodes"] if node[0] == old_index
    }
    if (
        any(
            any(old_node[0] != old_index for old_node in context["graph"][node])
            for node in target_current_nodes
        )
        or any(
            any(
                current_node[0] != candidate_index
                for current_node in context["reverse_graph"][node]
            )
            for node in target_old_nodes
        )
        or not any(
            (old_node := context["reserved"].get(current_node)) is not None
            and old_node[0] == old_index
            and context["current_nodes"][current_node][2]
            > context["old_nodes"][old_node][2]
            for current_node in target_current_nodes
        )
    ):
        return None

    bracket = getattr(candidate, "bracket", None)
    raw_revision = getattr(candidate, "boundary_revision", None)
    raw_position = getattr(candidate, "position", None)
    raw_age = getattr(candidate, "boundary_age_seconds", None)
    if (
        not isinstance(bracket, (tuple, list))
        or len(bracket) != 2
        or any(isinstance(value, bool) for value in bracket)
        or not isinstance(raw_revision, (tuple, list))
        or len(raw_revision) != 2
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in raw_revision
        )
        or isinstance(raw_position, bool)
        or isinstance(raw_age, bool)
    ):
        return None
    try:
        lower, upper = map(float, bracket)
        position = float(raw_position)
        boundary_age = float(raw_age)
    except (TypeError, ValueError, OverflowError):
        return None
    if (
        not all(isfinite(value) for value in (lower, upper, position, boundary_age))
        or not lower.is_integer()
        or not upper.is_integer()
        or lower < 0
        or lower >= upper
        or not lower <= position <= upper
        or position - track.position <= MATCH_DISTANCE
        or lower < track.position
        or not 0 <= boundary_age < PARTIAL_BIRTH_FRESHNESS_SECONDS
    ):
        return None

    route_slots = context["route_slots"]
    lower_index, upper_index = int(lower), int(upper)
    candidate_slots_by_stop = {}
    for slot in slots:
        candidate_slots_by_stop.setdefault(int(probe_inputs[slot].index), []).append(slot)
    due_indices = sorted(row[0] for row in normalized if row[3] == 0)
    positive_indices = sorted(row[0] for row in normalized if row[3] > 0)
    if due_indices:
        expected_lower = due_indices[-1]
        expected_upper = next(
            (index for index in positive_indices if index > expected_lower),
            None,
        )
        if lower_index != expected_lower or upper_index != expected_upper:
            return None
    elif not positive_indices or upper_index != positive_indices[0]:
        return None

    upper_slots = candidate_slots_by_stop.get(upper_index, ())
    if len(upper_slots) != 1:
        return None
    upper_row = _partial_birth_live_row(probe_inputs[upper_slots[0]], route_key)
    arrival = _arrival_timestamp(getattr(candidate, "eta_arrival_at", None))
    if (
        upper_row is None
        or upper_row[2] != raw_revision[1]
        or upper_row[3] <= 0
        or arrival is None
        or abs(arrival - upper_row[1]) > 0.5
    ):
        return None
    endpoint_ages = []
    endpoint_ages_by_index = {}
    for index, revision in (
        (lower_index, raw_revision[0]),
        (upper_index, raw_revision[1]),
    ):
        endpoint_slots = route_slots.get(index, ())
        if not endpoint_slots:
            return None
        for slot in endpoint_slots:
            endpoint = _partial_birth_endpoint_row(
                probe_inputs[slot], route_key, index,
            )
            if endpoint is None or endpoint[0] != revision:
                return None
            endpoint_ages.append(endpoint[1])
            endpoint_ages_by_index.setdefault(index, []).append(endpoint[1])
            if (
                getattr(probe_inputs[slot], "minutes", None) is not None
                and _partial_birth_live_row(probe_inputs[slot], route_key) is None
            ):
                return None
    lower_slots = candidate_slots_by_stop.get(lower_index, ())
    lower_row = None
    if due_indices:
        if len(lower_slots) != 1:
            return None
        lower_row = _partial_birth_live_row(
            probe_inputs[lower_slots[0]], route_key,
        )
        if lower_row is None or lower_row[3] > 0:
            return None
    elif lower_slots:
        return None
    ages = [boundary_age, *endpoint_ages, *(row[4] for row in normalized)]
    try:
        upper_eta = float(
            getattr(probe_inputs[upper_slots[0]], "signed_minutes", None)
        )
        if due_indices:
            lower_eta = float(
                getattr(probe_inputs[lower_slots[0]], "signed_minutes", None)
            )
            if not lower_eta <= 0 < upper_eta:
                return None
            fraction = min(1.0, max(0.0, -lower_eta / (upper_eta - lower_eta)))
            expected_position = lower_index + (upper_index - lower_index) * fraction
        else:
            expected_position = upper_index - min(
                1.0, max(0.0, upper_row[3] / MINUTES_PER_STOP),
            )
        expected_boundary_age = max(
            upper_row[4], min(endpoint_ages_by_index[lower_index]),
        )
    except (TypeError, ValueError, OverflowError, ZeroDivisionError):
        return None
    if (
        not all(isfinite(value) for value in (
            upper_eta, expected_position, expected_boundary_age,
        ))
        or abs(position - expected_position) > 1e-6
        or abs(boundary_age - expected_boundary_age) > 1e-6
        or max(ages) - min(ages) > PARTIAL_BIRTH_MAX_AGE_SKEW_SECONDS
    ):
        return None
    return position


def _partial_birth_continuation_stops(track, candidate_rows):
    """Return stops with a frozen-cohort occurrence continuing in a candidate."""
    owned_by_stop = {}
    for row in _cohort_rows(track):
        owned_by_stop.setdefault(row[0], []).append(row)
    current_by_stop = {}
    for row in candidate_rows:
        current_by_stop.setdefault(row[0], []).append(row)
    continued = set()
    for index in owned_by_stop.keys() & current_by_stop.keys():
        if any(
            current[2] >= prior[2]
            and (
                abs(current[1] - prior[1]) <= 0.5
                or (
                    current[2] > prior[2]
                    and abs(current[1] - prior[1])
                    <= RECOVERY_ARRIVAL_TOLERANCE_SECONDS
                )
            )
            for prior in owned_by_stop[index]
            for current in current_by_stop[index]
        ):
            continued.add(index)
    return continued


def _unique_saturating_matching(edges, candidate_count):
    """Return the sole old-saturating matching, or None when it is ambiguous."""
    old_count = len(edges)

    def solve():
        candidate_owner = {}

        def augment(old_index, seen):
            for candidate_index in sorted(edges[old_index]):
                if candidate_index in seen:
                    continue
                seen.add(candidate_index)
                owner = candidate_owner.get(candidate_index)
                if owner is None or augment(owner, seen):
                    candidate_owner[candidate_index] = old_index
                    return True
            return False

        if sum(augment(old_index, set()) for old_index in range(old_count)) != old_count:
            return None
        return {old_index: candidate_index
                for candidate_index, old_index in candidate_owner.items()}

    matching = solve()
    if matching is None:
        return None
    unmatched = set(range(candidate_count)) - set(matching.values())
    if len(unmatched) != 1:
        return None

    # Contract each matched old--candidate edge. Every nonmatching edge then
    # points from its candidate to the candidate currently matched to that old
    # track. An alternating path out of the unmatched candidate changes which
    # candidate is surplus; an alternating cycle changes owner assignments.
    # Either proves that this is not the sole saturating matching.
    alternatives = {candidate_index: set() for candidate_index in range(candidate_count)}
    for old_index, options in enumerate(edges):
        matched = matching[old_index]
        for candidate_index in options - {matched}:
            alternatives[candidate_index].add(matched)
    unmatched_candidate = next(iter(unmatched))
    if alternatives[unmatched_candidate]:
        return None

    state = {}

    def cyclic(candidate_index):
        status = state.get(candidate_index, 0)
        if status == 1:
            return True
        if status == 2:
            return False
        state[candidate_index] = 1
        if any(cyclic(neighbour) for neighbour in alternatives[candidate_index]):
            return True
        state[candidate_index] = 2
        return False

    if any(cyclic(candidate_index) for candidate_index in range(candidate_count)):
        return None
    return matching


def _certified_partial_birth_index(
    old, candidates, probe_inputs, complete_route, now,
):
    """Prove one same-generation surplus from two complete stop censuses.

    This is intentionally narrower than complete-generation reconciliation. It
    can add one display-only marker, but it can neither retire a marker nor give
    the newborn an identity-authoritative cohort.
    """
    if not old or len(candidates) != len(old) + 1:
        return None
    if any(
        track.cohort_observed_at <= 0
        or not track.cohort_trusted
        or not 0 <= now - track.cohort_observed_at <= COHORT_EVIDENCE_TTL_SECONDS
        for track in old
    ):
        return None
    route_key = _key(candidates[0])
    if any(_key(candidate) != route_key or bool(getattr(candidate, "unreliable", False))
           for candidate in candidates):
        return None
    complete_revision_floors = _complete_checkpoint_revision_floors(
        complete_route, route_key,
    )
    if complete_revision_floors is None:
        return None

    probe_inputs = tuple(probe_inputs)
    seen_sources = set()
    seen_row_objects = set()
    slot_owner = {}
    candidate_slots = []
    candidate_rows = []
    for candidate_index, candidate in enumerate(candidates):
        sources = getattr(candidate, "source_observations", None)
        if not isinstance(sources, frozenset) or not sources:
            return None
        slots = []
        rows = []
        stops = set()
        for source in sources:
            if (
                not isinstance(source, tuple)
                or len(source) != 2
                or source[0] not in {"gate", "probe"}
                or not isinstance(source[1], int)
                or isinstance(source[1], bool)
                or source[1] < 0
                or source in seen_sources
            ):
                return None
            seen_sources.add(source)
            if source[0] == "gate":
                continue
            slot = source[1]
            if slot >= len(probe_inputs):
                return None
            if id(probe_inputs[slot]) in seen_row_objects:
                return None
            seen_row_objects.add(id(probe_inputs[slot]))
            normalized = _partial_birth_live_row(probe_inputs[slot], route_key)
            if normalized is None or normalized[0] in stops:
                return None
            stops.add(normalized[0])
            slots.append(slot)
            rows.append(normalized)
            slot_owner[slot] = candidate_index
        strict_rows = _strict_checkpoint_rows(candidate)
        if not slots or strict_rows is None or Counter(strict_rows) != Counter(
            (index, arrival, revision)
            for index, arrival, revision, _minutes, _age in rows
        ):
            return None
        rows.sort()
        if any(
            right[1] < left[1]
            or right[1] - left[1]
            > COHORT_MAX_SECONDS_PER_STOP * (right[0] - left[0])
            for left, right in zip(rows, rows[1:], strict=False)
        ):
            return None
        candidate_slots.append(tuple(slots))
        candidate_rows.append(tuple(
            (index, arrival, revision)
            for index, arrival, revision, _minutes, _age in rows
        ))

    old_revisions = {}
    for track in old:
        for index, _arrival, revision in _cohort_rows(track):
            old_revisions[index] = max(old_revisions.get(index, 0), revision)
    for index, revision in complete_revision_floors.items():
        old_revisions[index] = max(old_revisions.get(index, 0), revision)
    if any(
        revision < old_revisions.get(index, 0)
        for rows in candidate_rows
        for index, _arrival, revision in rows
    ):
        return None

    edges = []
    for track in old:
        edges.append({
            candidate_index
            for candidate_index, rows in enumerate(candidate_rows)
            if len(_partial_birth_continuation_stops(track, rows)) >= 2
        })
    matching = _unique_saturating_matching(edges, len(candidates))
    if matching is None:
        return None
    unmatched = (set(range(len(candidates))) - set(matching.values())).pop()
    if any(
        _partial_birth_continuation_stops(track, candidate_rows[unmatched])
        for track in old
    ):
        # The surplus must be wholly new. Even one plausible frozen-cohort
        # occurrence can be a split fragment still owned by an established
        # marker; the two-stop census proves cardinality, not that fragment's
        # right to a new identity.
        return None

    route_slots = {}
    for slot, row in enumerate(probe_inputs):
        if _key(row) != route_key:
            continue
        index = getattr(row, "index", None)
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            return None
        if index in old_revisions:
            revision = getattr(row, "refresh_generation", None)
            if (
                not isinstance(revision, int)
                or isinstance(revision, bool)
                or revision < old_revisions[index]
            ):
                return None
        route_slots.setdefault(index, []).append(slot)

    censuses = []
    candidate_numbers = set(range(len(candidates)))
    for index, raw_slots in sorted(route_slots.items()):
        if len(raw_slots) != len(candidates):
            continue
        normalized = [
            _partial_birth_live_row(probe_inputs[slot], route_key)
            for slot in raw_slots
        ]
        if any(row is None for row in normalized):
            continue
        owners = [slot_owner.get(slot) for slot in raw_slots]
        if set(owners) != candidate_numbers or len(set(owners)) != len(candidates):
            continue
        revisions = {row[2] for row in normalized}
        if len(revisions) != 1 or next(iter(revisions)) < old_revisions.get(index, 0):
            continue
        ranked = sorted(
            (row[1], owner, row[4])
            for row, owner in zip(normalized, owners, strict=True)
        )
        if any(right[0] - left[0] <= 0.5
               for left, right in zip(ranked, ranked[1:], strict=False)):
            continue
        censuses.append((
            index,
            tuple(owner for _arrival, owner, _age in ranked),
            tuple(age for _arrival, _owner, age in ranked),
            {owner: arrival for arrival, owner, _age in ranked},
        ))

    witnessed = False
    for census_index, left in enumerate(censuses):
        for right in censuses[census_index + 1:]:
            stop_gap = right[0] - left[0]
            ages = (*left[2], *right[2])
            if (
                stop_gap <= 0
                or left[1] != right[1]
                or max(ages) - min(ages) > PARTIAL_BIRTH_MAX_AGE_SKEW_SECONDS
            ):
                continue
            if all(
                0 <= right[3][candidate_index] - left[3][candidate_index]
                <= COHORT_MAX_SECONDS_PER_STOP * stop_gap
                for candidate_index in candidate_numbers
            ):
                witnessed = True
                break
        if witnessed:
            break
    if not witnessed:
        return None

    birth = candidates[unmatched]
    bracket = getattr(birth, "bracket", None)
    raw_revision = getattr(birth, "boundary_revision", None)
    raw_position = getattr(birth, "position", None)
    raw_age = getattr(birth, "boundary_age_seconds", None)
    if (
        not isinstance(bracket, (tuple, list))
        or len(bracket) != 2
        or any(isinstance(value, bool) for value in bracket)
        or not isinstance(raw_revision, (tuple, list))
        or len(raw_revision) != 2
        or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0
               for value in raw_revision)
        or isinstance(raw_position, bool)
        or isinstance(raw_age, bool)
    ):
        return None
    try:
        lower, upper = map(float, bracket)
        position = float(raw_position)
        boundary_age = float(raw_age)
    except (TypeError, ValueError, OverflowError):
        return None
    if (
        not all(isfinite(value) for value in (lower, upper, position, boundary_age))
        or not lower.is_integer()
        or not upper.is_integer()
        or lower < 0
        or lower >= upper
        or position <= 0
        or not lower <= position <= upper
        or not 0 <= boundary_age < PARTIAL_BIRTH_FRESHNESS_SECONDS
    ):
        return None
    lower_index, upper_index = int(lower), int(upper)
    birth_slots_by_stop = {}
    for slot in candidate_slots[unmatched]:
        birth_slots_by_stop.setdefault(int(probe_inputs[slot].index), []).append(slot)
    birth_rows = [
        _partial_birth_live_row(probe_inputs[slot], route_key)
        for slot in candidate_slots[unmatched]
    ]
    if any(row is None for row in birth_rows):
        return None
    due_indices = sorted(row[0] for row in birth_rows if row[3] == 0)
    positive_indices = sorted(row[0] for row in birth_rows if row[3] > 0)
    if due_indices:
        expected_lower = due_indices[-1]
        expected_upper = next(
            (index for index in positive_indices if index > expected_lower),
            None,
        )
        if lower_index != expected_lower or upper_index != expected_upper:
            return None
    elif not positive_indices or upper_index != positive_indices[0]:
        return None
    upper_slots = birth_slots_by_stop.get(upper_index, ())
    if len(upper_slots) != 1:
        return None
    upper_row = _partial_birth_live_row(probe_inputs[upper_slots[0]], route_key)
    if upper_row is None or upper_row[2] != raw_revision[1] or upper_row[3] <= 0:
        return None
    if any(
        (endpoint := _partial_birth_endpoint_row(
            probe_inputs[slot], route_key, upper_index,
        )) is None
        or endpoint[0] != raw_revision[1]
        for slot in route_slots.get(upper_index, ())
    ):
        return None

    lower_slots = birth_slots_by_stop.get(lower_index, ())
    if lower_slots:
        if len(lower_slots) != 1:
            return None
        lower_row = _partial_birth_live_row(probe_inputs[lower_slots[0]], route_key)
        if lower_row is None or lower_row[2] != raw_revision[0] or lower_row[3] > 0:
            return None
        if any(
            (endpoint := _partial_birth_endpoint_row(
                probe_inputs[slot], route_key, lower_index,
            )) is None
            or endpoint[0] != raw_revision[0]
            for slot in route_slots.get(lower_index, ())
        ):
            return None
    else:
        endpoint_slots = route_slots.get(lower_index, ())
        if len(endpoint_slots) != 1:
            return None
        endpoint = probe_inputs[endpoint_slots[0]]
        normalized_endpoint = _partial_birth_endpoint_row(
            endpoint, route_key, lower_index,
        )
        if (
            normalized_endpoint is None
            or normalized_endpoint[0] != raw_revision[0]
            or getattr(endpoint, "minutes", None) is not None
        ):
            return None
    return unmatched


def _checkpoint_overlap(owned, current):
    """Return bounded one-to-one physical checkpoint matches."""
    return tuple(new_number for _old_number, new_number in
                 _checkpoint_overlap_pairs(owned, current))


def _checkpoint_overlap_pairs(owned, current):
    """Return one-to-one physical old/current row-number matches."""
    matches = []
    for stop in {row[0] for row in owned} & {row[0] for row in current}:
        old_rows = sorted((arrival, number) for number, (index, arrival, _revision)
                          in enumerate(owned) if index == stop)
        new_rows = sorted((arrival, number) for number, (index, arrival, _revision)
                          in enumerate(current) if index == stop)
        old_number = new_number = 0
        while old_number < len(old_rows) and new_number < len(new_rows):
            old_arrival, old_index = old_rows[old_number]
            new_arrival, candidate_number = new_rows[new_number]
            if abs(old_arrival - new_arrival) <= 0.5:
                matches.append((old_index, candidate_number))
                old_number += 1
                new_number += 1
            elif old_arrival < new_arrival:
                old_number += 1
            else:
                new_number += 1
    return tuple(matches)


def _cohort_rows(track):
    """Return identity evidence from the last complete lifecycle publication."""
    if track.cohort_observed_at > 0.0:
        return tuple(track.cohort_evidence) if track.cohort_trusted else ()
    # Private helper tests and legacy in-memory tracks predate the dedicated
    # ledger. Their current estimate is the only available complete cohort.
    return _checkpoint_rows(track.estimate)


def _matching_track(track):
    """Expose only geometry for an untrusted display during identity matching.

    The display estimate and motion witness remain intact on the real track.
    This matching-only view prevents exact checkpoints, ETA anchors, temporal
    recovery, or source-slot tie breaks from giving its display identity weight.
    """
    if track.cohort_observed_at <= 0 or track.cohort_trusted:
        return track
    return replace(track, estimate=replace(
        track.estimate,
        checkpoint_evidence=(),
        eta_arrival_at=None,
        source_observations=frozenset(),
        source_indices=frozenset(),
    ), committed_boundary_evidence=())


def _checkpoint_arrival_range_mask(arrivals, center, tolerance):
    """Index the original subtraction predicate without rounded-cutoff drift."""
    # Widen the tolerance before addition: near cancellation, one ULP of the
    # resulting cutoff can be much smaller than subtraction's rounding error.
    # The outward-rounded bounds form a superset; validate its endpoints using
    # the original predicate. Binary correction preserves repeated-row bounds.
    expanded = nextafter(tolerance, inf)
    lower = bisect_left(arrivals, nextafter(center - expanded, -inf))
    upper = bisect_right(arrivals, nextafter(center + expanded, inf))
    if lower < upper and abs(arrivals[lower] - center) > tolerance:
        left, right = lower, upper
        while left < right:
            middle = (left + right) // 2
            value = arrivals[middle]
            if value >= center or abs(value - center) <= tolerance:
                right = middle
            else:
                left = middle + 1
        lower = left
    if lower < upper and abs(arrivals[upper - 1] - center) > tolerance:
        left, right = lower, upper
        while left < right:
            middle = (left + right) // 2
            value = arrivals[middle]
            if value <= center or abs(value - center) <= tolerance:
                left = middle + 1
            else:
                right = middle
        upper = left
    return (1 << upper) - (1 << lower)


def _indexed_checkpoint_links(owned, current, range_cache=None):
    """Compare ledgers by physical stop, retaining occurrence multiplicity."""
    if range_cache is None:
        range_cache = {}
    exact, temporal, eligible, newer_stops = set(), set(), set(), set()
    for stop in owned.keys() & current.keys():
        before, after = owned[stop], current[stop]
        # Normal route ledgers have one occurrence per physical checkpoint.
        # Avoid allocating a per-stop matching table for this common case.
        if len(before) == len(after) == 1:
            old_arrival, old_revision, _ = before[0]
            new_arrival, new_revision, row_number = after[0]
            drift = abs(new_arrival - old_arrival)
            is_exact = drift <= 0.5
            newer = new_revision > old_revision and drift <= RECOVERY_ARRIVAL_TOLERANCE_SECONDS
            if is_exact:
                exact.add(row_number)
            if newer:
                newer_stops.add(stop)
            if is_exact or newer:
                temporal.add(row_number)
                eligible.add(row_number)
            continue
        old_number = new_number = 0
        while old_number < len(before) and new_number < len(after):
            old_arrival = before[old_number][0]
            new_arrival, _revision, row_number = after[new_number]
            if abs(old_arrival - new_arrival) <= 0.5:
                exact.add(row_number)
                old_number += 1
                new_number += 1
            elif old_arrival < new_arrival:
                old_number += 1
            else:
                new_number += 1
        # Share the current response's arrival/revision indexes across all old
        # ledgers. Each old occurrence needs two contiguous arrival ranges and
        # a revision-mask intersection, never a scan of every current occurrence.
        if stop not in range_cache:
            arrivals = tuple(row[0] for row in after)
            by_revision = {}
            for number, (_arrival, revision, _row) in enumerate(after):
                by_revision[revision] = by_revision.get(revision, 0) | (1 << number)
            revisions = sorted(by_revision)
            suffix_masks = [0] * (len(revisions) + 1)
            for number in range(len(revisions) - 1, -1, -1):
                suffix_masks[number] = suffix_masks[number + 1] | by_revision[revisions[number]]
            range_cache[stop] = arrivals, revisions, suffix_masks, {}
        arrivals, revisions, suffix_masks, newer_masks = range_cache[stop]
        if (after[-1][0] - before[0][0] <= RECOVERY_ARRIVAL_TOLERANCE_SECONDS
                and before[-1][0] - after[0][0] <= RECOVERY_ARRIVAL_TOLERANCE_SECONDS
                and revisions[0] > max(row[1] for row in before)):
            # Every old/current occurrence continues every other occurrence.
            # With multiplicity on either side no row has a unique old owner.
            temporal.update(row[2] for row in after)
            newer_stops.add(stop)
            continue
        temporal_mask = unique_mask = repeated_unique_mask = 0
        for old_arrival, old_revision, _ in before:
            newer_mask = newer_masks.get(old_revision)
            if newer_mask is None:
                newer_mask = suffix_masks[bisect_right(revisions, old_revision)]
                newer_masks[old_revision] = newer_mask
            exact_mask = _checkpoint_arrival_range_mask(arrivals, old_arrival, 0.5)
            wide_mask = _checkpoint_arrival_range_mask(
                arrivals, old_arrival, RECOVERY_ARRIVAL_TOLERANCE_SECONDS,
            )
            newer = wide_mask & newer_mask
            if newer:
                newer_stops.add(stop)
            matches = exact_mask | newer
            temporal_mask |= matches
            if matches.bit_count() == 1:
                repeated_unique_mask |= unique_mask & matches
                unique_mask |= matches
        eligible_mask = unique_mask & ~repeated_unique_mask
        for number, (_arrival, _revision, row_number) in enumerate(after):
            bit = 1 << number
            if temporal_mask & bit:
                temporal.add(row_number)
            if eligible_mask & bit:
                eligible.add(row_number)
    return exact, temporal, eligible, len(newer_stops) >= 2, bool(exact)


def _forced_identity_pairs(edges, invalid):
    """Return edges common to every maximum injective census assignment."""
    edges = {index: tuple(sorted(neighbours - invalid)) for index, neighbours in edges.items()}

    def augment(owners, old_index, visited, forbidden=None):
        for new_index in edges[old_index]:
            if (old_index, new_index) == forbidden or new_index in visited:
                continue
            visited.add(new_index)
            if new_index not in owners or augment(owners, owners[new_index], visited, forbidden):
                owners[new_index] = old_index
                return True
        return False

    matching = {}
    for old_index in edges:
        augment(matching, old_index, set())

    def forced(pair):
        # Removing one edge from a maximum matching leaves a deficit of one.
        # It is forced exactly when no alternating path can repair that deficit,
        # including paths starting at a previously unmatched old identity.
        owners = dict(matching)
        del owners[pair[1]]
        matched_old = set(owners.values())
        return not any(
            augment(owners, index, set(), pair)
            for index in edges if index not in matched_old
        )

    return {
        (old_index, new_index) for new_index, old_index in matching.items()
        if new_index not in invalid and forced((old_index, new_index))
    }


def _same_generation_identity_plan(old, new, now):
    """Reserve forced census identities before considering motion eligibility.

    Strictly newer observations at two distinct physical checkpoints certify
    temporal edges. Neither position nor mutable source slots break temporal
    ambiguity. Exact one-checkpoint matching retains its existing contracts.
    """
    raw_current = [getattr(candidate, "checkpoint_evidence", ()) for candidate in new]
    raw_owned = [
        (track.cohort_evidence if track.cohort_trusted else ())
        if track.cohort_observed_at > 0
        else getattr(track.estimate, "checkpoint_evidence", ())
        for track in old
    ]
    if any(not isinstance(rows, tuple) or len(rows) > MAX_CHECKPOINT_EVIDENCE
           for rows in (*raw_current, *raw_owned)):
        return set(), set(range(len(old))), set(range(len(new)))
    current = [_strict_checkpoint_rows(candidate) for candidate in new]
    owned = [
        _strict_checkpoint_rows(replace(
            track.estimate, checkpoint_evidence=rows,
        ))
        for track, rows in zip(old, raw_owned, strict=True)
    ]
    # A malformed ledger can hide a competitor. It cannot establish uniqueness
    # by silently dropping its invalid rows.
    if any(rows is None for rows in (*current, *owned)):
        return set(), set(range(len(old))), set(range(len(new)))
    def indexed(rows):
        stops = {}
        for number, (stop, arrival, revision) in enumerate(rows):
            stops.setdefault(stop, []).append((arrival, revision, number))
        return {stop: sorted(values, key=lambda value: (value[0], value[2]))
                for stop, values in stops.items()}

    # Equal-time populations often repeat an entire ledger. Share comparison
    # work, but keep all physical owners in the masks and in the final census.
    owner_groups = {}
    for index, rows in enumerate(owned):
        owner_groups[rows] = owner_groups.get(rows, 0) | (1 << index)
    indexed_owned = {rows: indexed(rows) for rows in owner_groups}
    profiles = {}
    remaining = Counter(current)
    revision_floors = _route_metadata_revision_floors(old)
    # Revisions belong to physical stop responses, not individual vehicles.
    # A stale candidate retains identity edges only to fence its component;
    # it cannot supply matching capacity or fall through to ordinary motion.
    invalid = {index for index, candidate in enumerate(new)
               if not _metadata_nonregressing(candidate, revision_floors)}
    mixed_edges = set()
    exact_edges = {index: set() for index, rows in enumerate(owned) if rows}
    temporal_edges = {index: set() for index, rows in enumerate(owned) if rows}
    for new_index, candidate in enumerate(new):
        rows = current[new_index]
        if rows not in profiles:
            current_stops = indexed(rows)
            range_cache = {}
            links = {ledger: _indexed_checkpoint_links(stops, current_stops, range_cache)
                     for ledger, stops in indexed_owned.items()}
            exact_owners = [0] * len(rows)
            raw_owners = [0] * len(rows)
            eligible_owners = [0] * len(rows)
            for ledger, (exact, temporal, eligible, _strong, _proof) in links.items():
                mask = owner_groups[ledger]
                for row in exact:
                    exact_owners[row] |= mask
                for row in temporal:
                    raw_owners[row] |= mask
                for row in eligible:
                    eligible_owners[row] |= mask
            exclusive_stops = {}
            strong_owners = set()
            for number, (exact, raw, eligible) in enumerate(zip(
                exact_owners, raw_owners, eligible_owners, strict=True,
            )):
                if exact:
                    if exact.bit_count() == 1:
                        strong_owners.add(exact)
                elif raw.bit_count() == 1 and eligible:
                    exclusive_stops.setdefault(raw, set()).add(rows[number][0])
            strong_owners.update(mask for mask, stops in exclusive_stops.items() if len(stops) >= 2)
            mixed = sum(strong_owners) if len(strong_owners) > 1 else 0
            prior = {number for number, mask in enumerate(exact_owners) if mask}
            covering = sum(owner_groups[ledger] for ledger, link in links.items()
                           if prior and link[0] == prior)
            any_exact = 0
            for mask in exact_owners:
                any_exact |= mask
            profiles[rows] = links, covering, any_exact, mixed, bool(prior and not covering)
        links, covering, any_exact, mixed, uncovered = profiles[rows]
        if mixed or uncovered or _incoherent_checkpoint_chronology(candidate):
            invalid.add(new_index)
            mixed_edges.update(
                (index, new_index) for index in range(len(old))
                if (mixed | any_exact) & (1 << index)
            )
        for old_index, track in enumerate(old):
            if not owned[old_index] or covering and not covering & (1 << old_index):
                continue
            _exact, _temporal, _eligible, strong, exact_proof = links[owned[old_index]]
            if exact_proof:
                exact_edges[old_index].add(new_index)
            recent = (track.cohort_observed_at <= 0
                      or 0 <= now - track.cohort_observed_at <= COHORT_EVIDENCE_TTL_SECONDS)
            if recent and strong:
                temporal_edges[old_index].add(new_index)
        remaining[rows] -= 1
        if not remaining[rows]:
            del profiles[rows]
    # Compose exact ownership first: a newer response for a nearby old bus
    # cannot steal the immutable checkpoint population of its exact owner.
    # Invalid candidates cannot provide the extra matching capacity which
    # makes a neighbouring valid edge appear forced. Fence their whole component
    # in the unfiltered census, while preserving unrelated components.
    census_edges = {index: exact_edges[index] | temporal_edges[index] for index in exact_edges}
    for old_index, new_index in mixed_edges:
        census_edges.setdefault(old_index, set()).add(new_index)
    reverse = {}
    for old_index, neighbours in census_edges.items():
        for new_index in neighbours:
            reverse.setdefault(new_index, set()).add(old_index)
    invalid_old = set()
    pending = list(invalid)
    while pending:
        for old_index in reverse.get(pending.pop(), ()):
            if old_index in invalid_old:
                continue
            invalid_old.add(old_index)
            unseen = census_edges[old_index] - invalid
            invalid.update(unseen)
            pending.extend(unseen)
    reservations = _forced_identity_pairs(exact_edges, invalid)
    reserved_old = {index for index, _ in reservations}
    reserved_new = {index for _, index in reservations}
    edges = {index: neighbours - reserved_new for index, neighbours in temporal_edges.items()
             if index not in reserved_old}
    exact_candidates = {index for neighbours in exact_edges.values() for index in neighbours}
    for old_index, neighbours in edges.items():
        if exact_edges.get(old_index):
            neighbours.intersection_update(exact_edges[old_index])
        neighbours.difference_update(exact_candidates - exact_edges.get(old_index, set()))
    for old_index, new_index in mixed_edges:
        if old_index not in reserved_old and new_index not in reserved_new:
            edges.setdefault(old_index, set()).add(new_index)

    reservations.update(_forced_identity_pairs(edges, invalid))
    blocked_old = {index for index, neighbours in edges.items() if neighbours} | invalid_old
    blocked_new = {index for neighbours in edges.values() for index in neighbours} | invalid
    # A one-stop drift is not a reservation or an ambiguity certificate. It
    # retains historical short-distance fallback only outside these fences.
    blocked_old.difference_update(index for index, _ in reservations)
    blocked_new.difference_update(index for _, index in reservations)
    return reservations, blocked_old, blocked_new


def _metadata_revision_floors(track):
    floors = dict(track.metadata_revision_floors)
    for stop, _arrival, revision in (*_checkpoint_rows(track.estimate), *_cohort_rows(track)):
        floors[stop] = max(floors.get(stop, 0), revision)
    return floors


def _route_metadata_revision_floors(tracks):
    floors = {}
    for track in tracks:
        for stop, revision in _metadata_revision_floors(track).items():
            floors[stop] = max(floors.get(stop, 0), revision)
    return floors


def _metadata_nonregressing(candidate, floors):
    return all(revision >= floors.get(stop, 0)
               for stop, _arrival, revision in _checkpoint_rows(candidate))


def _record_metadata_revisions(track, candidate):
    floors = _metadata_revision_floors(track)
    for stop, _arrival, revision in _checkpoint_rows(candidate):
        floors[stop] = max(floors.get(stop, 0), revision)
    track.metadata_revision_floors = floors


def _refresh_held_metadata(track, candidate):
    """Replace one current identity bundle without renewing motion evidence."""
    metadata = {
        name: getattr(candidate, name)
        for name in (
            "source_indices", "source_observations", "checkpoint_evidence",
            "priority_indices", "exploratory_indices", "eta_arrival_at", "eta_minutes",
        )
    }
    if getattr(candidate, "position_authoritative", None) is False:
        metadata["position_authoritative"] = False
    _record_metadata_revisions(track, candidate)
    track.estimate = replace(track.estimate, **metadata)


def _evidence_continues(prior, current):
    """Whether one current occurrence can be the same physical ETA row."""
    old_stop, old_arrival, old_revision = prior
    new_stop, new_arrival, new_revision = current
    if old_stop != new_stop:
        return False
    drift = abs(new_arrival - old_arrival)
    return drift <= 0.5 or (
        new_revision > old_revision
        and drift <= RECOVERY_ARRIVAL_TOLERANCE_SECONDS
    )


def _replace_track_cohort(track, candidates, now):
    """Publish a survivor's newly reconciled, identity-only cohort."""
    evidence = [
        row
        for candidate in candidates
        for row in _checkpoint_rows(candidate)
    ]
    track.cohort_evidence = tuple(
        sorted(set(evidence))[:MAX_CHECKPOINT_EVIDENCE]
    )
    track.cohort_observed_at = now
    track.cohort_trusted = True
    track.metadata_revision_floors = {}
    for stop, _arrival, revision in track.cohort_evidence:
        track.metadata_revision_floors[stop] = max(
            track.metadata_revision_floors.get(stop, 0), revision,
        )


def _reconcile_complete_probe_ownership(old, new, now, probe_inputs, route_lines):
    """Atomically repartition certified components of a complete population.

    Occurrences are source slots on the current side and (owner, row number)
    on the historical side. Exact pairs are reserved globally first. A full
    injection must exist, and every alternative owner is tested for feasibility;
    a maximum matching alone is never treated as an ownership certificate.
    """
    if not old or not new or not probe_inputs:
        return new, {}

    def probe_only(estimate):
        sources = getattr(estimate, "source_observations", ()) or ()
        return bool(sources) and not getattr(estimate, "unreliable", False) and all(
            isinstance(source, tuple) and len(source) == 2
            and source[0] == "probe" and isinstance(source[1], int)
            and not isinstance(source[1], bool) and source[1] >= 0
            for source in sources
        )

    def coherent(rows):
        rows = sorted(rows)
        return len({row[0] for row in rows}) >= 2 and all(
            right[0] > left[0] and right[1] > left[1]
            for left, right in zip(rows, rows[1:], strict=False)
        )

    historical = {}
    for owner, track in enumerate(old):
        age = now - track.cohort_observed_at
        cohort = _cohort_rows(track)
        # Invalid ledger entries cannot become certificates through coercion.
        valid = all(
            isinstance(row, tuple) and len(row) == 3
            and isinstance(row[0], int) and not isinstance(row[0], bool)
            and row[0] >= 0 and isinstance(row[1], (int, float))
            and isfinite(row[1]) and isinstance(row[2], int)
            and not isinstance(row[2], bool) and row[2] > 0
            for row in cohort
        )
        if (probe_only(track.estimate) and valid and coherent(cohort)
                and 0 <= age <= COHORT_EVIDENCE_TTL_SECONDS):
            historical.update(((owner, number), row)
                              for number, row in enumerate(cohort))
    if not historical:
        return new, {}

    source_counts = Counter(
        source for candidate in new
        for source in getattr(candidate, "source_observations", ()) or ()
        if isinstance(source, tuple) and len(source) == 2
        and isinstance(source[0], str) and isinstance(source[1], int)
    )
    current = {}
    candidate_slots = {}
    invalid_candidates = set()
    for index, candidate in enumerate(new):
        if not probe_only(candidate):
            invalid_candidates.add(index)
        slots = []
        for source in getattr(candidate, "source_observations", ()) or ():
            if (not isinstance(source, tuple) or len(source) != 2
                    or source[0] != "probe" or isinstance(source[1], bool)
                    or not isinstance(source[1], int)
                    or not 0 <= source[1] < len(probe_inputs)):
                invalid_candidates.add(index)
                if isinstance(source, tuple) and source and source[0] == "probe":
                    # An unlocatable probe occurrence could consume any old
                    # node. No route-wide uniqueness proof is available.
                    return new, {}
                continue
            slot = source[1]
            row = probe_inputs[slot]
            arrival = _arrival_timestamp(getattr(row, "arrival_at", None))
            stop = getattr(row, "index", None)
            revision = getattr(row, "refresh_generation", None)
            if (_key(row) != _key(candidate) or not isinstance(stop, int)
                    or isinstance(stop, bool) or stop < 0
                    or not isinstance(revision, int) or isinstance(revision, bool)
                    or revision <= 0 or arrival is None):
                return new, {}
            # Even an excluded scheduled/stale occurrence occupies its raw
            # slot. Keep it in the graph so its historical node cannot become
            # available to a neighbouring candidate's temporal fallback.
            current[slot] = (stop, arrival, revision)
            slots.append(slot)
            if (isinstance(getattr(row, "minutes", None), bool)
                    or isinstance(getattr(row, "cache_age_seconds", None), bool)
                    or isinstance(getattr(row, "signed_minutes", None), bool)):
                invalid_candidates.add(index)
                continue
            try:
                minutes = float(row.minutes)
                age = float(row.cache_age_seconds)
            except (AttributeError, TypeError, ValueError, OverflowError):
                invalid_candidates.add(index)
                continue
            if (source_counts[source] != 1
                    or getattr(row, "kind", None) not in LIVE_PROBE_ETA_KINDS
                    or not isfinite(minutes)
                    or not isfinite(age) or not 0 <= age < 900):
                invalid_candidates.add(index)
        candidate_slots[index] = slots
        if Counter(current[slot] for slot in slots) != Counter(_checkpoint_rows(candidate)):
            invalid_candidates.add(index)

    # Candidate membership and all possible historical owners define components.
    # Invalid candidates still connect their valid rows, poisoning the whole
    # affected component instead of freeing their occurrences for another bus.
    exact = {slot: {node for node, prior in historical.items()
                    if prior[0] == row[0] and row[2] >= prior[2]
                    and abs(prior[1] - row[1]) <= 0.5}
             for slot, row in current.items()}
    exact_counts = Counter(node for edges in exact.values() for node in edges)
    reserved = {slot: next(iter(edges)) for slot, edges in exact.items()
                if len(edges) == 1 and exact_counts[next(iter(edges))] == 1}
    consumed = set(reserved.values())
    graph = {
        slot: ({reserved[slot]} if slot in reserved else {
            node for node, prior in historical.items()
            if node not in consumed and row[2] >= prior[2] and _evidence_continues(prior, row)
        }) for slot, row in current.items()
    }
    owner_candidates = {}
    for index, slots in candidate_slots.items():
        for slot in slots:
            for owner, _number in graph[slot]:
                owner_candidates.setdefault(owner, set()).add(index)

    def injection(slots, forced=None):
        claimed = {}
        if forced is not None:
            claimed[forced[1]] = forced[0]

        def augment(slot, seen):
            for node in sorted(graph[slot]):
                if node in seen or (forced is not None and node == forced[1]):
                    continue
                seen.add(node)
                if node not in claimed or augment(claimed[node], seen):
                    claimed[node] = slot
                    return True
            return False

        for slot in sorted(slots):
            if forced is not None and slot == forced[0]:
                continue
            if not augment(slot, set()):
                return None
        return {slot: node for node, slot in claimed.items()}

    replacements = {}
    certified = {}
    remaining = set(candidate_slots)
    while remaining:
        component = {min(remaining)}
        pending = list(component)
        while pending:
            index = pending.pop()
            for slot in candidate_slots[index]:
                for owner, _number in graph[slot]:
                    adjacent = owner_candidates[owner] - component
                    component.update(adjacent)
                    pending.extend(adjacent)
        remaining.difference_update(component)
        slots = {slot for index in component for slot in candidate_slots[index]}
        if component & invalid_candidates or not slots:
            continue
        matching = injection(slots)
        if matching is None:
            continue
        # An alternate injection assigning any row to a different owner makes
        # the entire component ambiguous (including Hall-deficient subsets).
        if any(injection(slots, (slot, node)) is not None
               for slot in slots for node in graph[slot]
               if node[0] != matching[slot][0]):
            continue
        owned_slots = {}
        for slot, (owner, _number) in matching.items():
            owned_slots.setdefault(owner, []).append(slot)
        if len(owned_slots) != len(component) or any(
            not coherent([current[slot] for slot in owned])
            for owned in owned_slots.values()
        ):
            continue
        if {frozenset(owned) for owned in owned_slots.values()} == {
            frozenset(candidate_slots[index]) for index in component
        }:
            owners_by_slots = {frozenset(owned): old[owner].track_id
                               for owner, owned in owned_slots.items()}
            certified.update({
                id(new[index]): owners_by_slots[frozenset(candidate_slots[index])]
                for index in component
            })
            continue
        rebuilt = []
        for owner, owned in sorted(owned_slots.items()):
            estimate = rebuild_estimate_from_probe_sources(
                old[owner].estimate, sorted(owned), probe_inputs, route_lines,
            )
            if estimate is None:
                break
            rebuilt.append((estimate, old[owner].track_id))
        if len(rebuilt) != len(component):
            continue
        if Counter(source for estimate, _owner in rebuilt
                   for source in estimate.source_observations) != Counter(
            source for index in component for source in new[index].source_observations
        ):
            continue
        for index, (estimate, owner_id) in zip(sorted(component), rebuilt, strict=True):
            replacements[index] = estimate
            certified[id(estimate)] = owner_id
    return sorted([replacements.get(index, candidate)
                   for index, candidate in enumerate(new)], key=_candidate_sort_key), certified


def _certified_cohort_fragments(old, new, pairs, now, probe_inputs):
    """Assign proved split fragments to survivors before lifecycle births.

    Sparse stop rotation can split one ETA ladder after a partial positioning
    update has replaced the marker estimate. The complete-generation birth
    path may consume an extra candidate only when its every physical row is an
    injective continuation of one uniquely matched survivor's retained cohort.
    This evidence affects cardinality only; marker motion still uses the base
    candidate and the normal fresh-boundary gates.
    """
    paired_old = {old_index: new_index for old_index, new_index in pairs}
    paired_new = {new_index for _old_index, new_index in pairs}
    if not paired_old or len(new) <= len(paired_new):
        return {}

    cohorts = []
    for track in old:
        age = now - track.cohort_observed_at
        cohorts.append(
            _cohort_rows(track)
            if 0.0 <= age <= COHORT_EVIDENCE_TTL_SECONDS
            else ()
        )
    candidate_rows = [_checkpoint_rows(candidate) for candidate in new]
    stop_occurrences = {}
    source_occurrences = {}
    for candidate, rows in zip(new, candidate_rows, strict=True):
        for stop, _arrival, _revision in rows:
            stop_occurrences[stop] = stop_occurrences.get(stop, 0) + 1
        for source in getattr(candidate, "source_observations", ()) or ():
            source_occurrences[source] = source_occurrences.get(source, 0) + 1

    def uniquely_probe_owned(candidate):
        sources = getattr(candidate, "source_observations", ()) or ()

        def valid_source(source):
            if (
                not isinstance(source, tuple)
                or len(source) != 2
                or source[0] != "probe"
                or isinstance(source[1], bool)
                or not isinstance(source[1], int)
                or not 0 <= source[1] < len(probe_inputs)
            ):
                return False
            row = probe_inputs[source[1]]
            return _key(row) == _key(candidate) and getattr(row, "minutes", None) is not None

        return (
            bool(sources)
            and not bool(getattr(candidate, "unreliable", False))
            and all(
                valid_source(source)
                and source_occurrences.get(source) == 1
                for source in sources
            )
        )

    cohort_index = {}
    for old_index, cohort in enumerate(cohorts):
        for evidence_index, (stop, arrival, revision) in enumerate(cohort):
            cohort_index.setdefault(stop, []).append(
                (arrival, revision, old_index, evidence_index)
            )
    cohort_arrivals = {}
    for stop, entries in cohort_index.items():
        entries.sort()
        cohort_arrivals[stop] = tuple(entry[0] for entry in entries)
    match_cache = {}

    def row_matches(row):
        cached = match_cache.get(row)
        if cached is not None:
            return cached
        stop, arrival, revision = row
        entries = cohort_index.get(stop, ())
        arrivals = cohort_arrivals.get(stop, ())
        lower = bisect_left(
            arrivals, arrival - RECOVERY_ARRIVAL_TOLERANCE_SECONDS
        )
        upper = bisect_right(
            arrivals, arrival + RECOVERY_ARRIVAL_TOLERANCE_SECONDS
        )
        owners = set()
        evidence_indices = set()
        for old_arrival, old_revision, owner, evidence_index in entries[lower:upper]:
            if not _evidence_continues(
                (stop, old_arrival, old_revision), (stop, arrival, revision)
            ):
                continue
            owners.add(owner)
            if len(owners) > 1:
                break
            evidence_indices.add(evidence_index)
        result = (frozenset(owners), frozenset(evidence_indices))
        match_cache[row] = result
        return result

    assignments = {}
    unmatched = [
        index for index in range(len(new)) if index not in paired_new
    ]
    for old_index, base_index in paired_old.items():
        cohort = cohorts[old_index]
        base = candidate_rows[base_index]
        if (
            not cohort
            or not base
            or not uniquely_probe_owned(new[base_index])
            or not _search_compatible(old[old_index], new[base_index])
            or abs(
                old[old_index].position
                - float(getattr(new[base_index], "position", 0.0) or 0.0)
            ) > MATCH_DISTANCE
        ):
            continue

        base_owned_rows = []
        base_reserved = set()
        base_conflict = False
        for row in base:
            owners, evidence_indices = row_matches(row)
            if owners and owners != {old_index}:
                base_conflict = True
                break
            if owners == {old_index}:
                base_owned_rows.append(row)
                base_reserved.update(evidence_indices)
        if base_conflict or not base_owned_rows:
            continue

        proposals = []
        for candidate_index in unmatched:
            rows = candidate_rows[candidate_index]
            if (
                not rows
                or not uniquely_probe_owned(new[candidate_index])
                or not _search_compatible(old[old_index], new[candidate_index])
            ):
                continue
            edges = []
            possible = set()
            valid = True
            for row in rows:
                owners, evidence_indices = row_matches(row)
                if owners != {old_index}:
                    valid = False
                    break
                row_edges = set(evidence_indices)
                # One historical occurrence cannot support both the survivor's
                # base and a purported extra occurrence at the same checkpoint.
                if not row_edges or row_edges & base_reserved:
                    valid = False
                    break
                edges.append(row_edges)
                possible.update(row_edges)
            if not valid:
                continue

            claimed = {}

            def augment(row_index, visited, graph=edges, owners=claimed):
                for evidence_index in sorted(graph[row_index]):
                    if evidence_index in visited:
                        continue
                    visited.add(evidence_index)
                    owner = owners.get(evidence_index)
                    if owner is None or augment(owner, visited, graph, owners):
                        owners[evidence_index] = row_index
                        return True
                return False

            if sum(
                augment(row_index, set()) for row_index in range(len(rows))
            ) == len(rows):
                proposals.append((candidate_index, rows, possible))

        # If two fragments can spend any of the same retained occurrence, list
        # order cannot decide which is the satellite. Both keep normal birth
        # semantics so equal ETA multiplicity is never collapsed.
        accepted = [
            proposal
            for proposal in proposals
            if not any(
                proposal[2] & other[2]
                for other in proposals
                if other[0] != proposal[0]
            )
        ]
        # Newly observed rows in the paired base are not historical ownership
        # proof, but they are still part of the current physical population.
        # Include them in chronology so a retained downstream fragment cannot
        # be folded across a newer departure or a reversed arrival chain.
        physical_rows = [*base, *(
            row for _candidate_index, rows, _possible in accepted for row in rows
        )]
        physical_rows.sort()
        if len({row[0] for row in physical_rows}) < 2 or len({
            row[0] for row in physical_rows
        }) != len(physical_rows):
            continue
        if any(
            right[1] <= left[1]
            or right[1] - left[1]
            > COHORT_MAX_SECONDS_PER_STOP * (right[0] - left[0])
            for left, right in zip(
                physical_rows, physical_rows[1:], strict=False
            )
        ):
            continue
        assignments.update(
            (candidate_index, old_index)
            for candidate_index, _rows, _possible in accepted
        )
    return assignments


def _missing_instance(track, index, rows):
    if all(getattr(row, "minutes", None) is None for row in rows):
        return True
    witness = (track.committed_boundary_evidence
               or _checkpoint_rows(track.estimate))
    prior = [row for row in witness if row[0] == index]
    current = []
    for row in rows:
        if getattr(row, "minutes", None) is None:
            continue
        arrival = _arrival_timestamp(getattr(row, "arrival_at", None))
        if arrival is None:
            return False
        try:
            revision = int(getattr(row, "refresh_generation", 0))
        except (TypeError, ValueError):
            revision = 0
        current.append((index, arrival, revision))
    # Without comparable timestamps, a nonempty response is inconclusive.
    if not prior or not current:
        return False
    # Find a maximum injective matching.  A complete matching means every
    # committed occurrence survived; otherwise at least one disappeared.
    edges = {
        old_index: {
            current_index for current_index, new in enumerate(current)
            if _evidence_continues(old, new)
        }
        for old_index, old in enumerate(prior)
    }
    # A timestamp drift backed by an old row revision is not proof of removal;
    # the caller's response-level revision guard makes it inconclusive.
    for old_index, old in enumerate(prior):
        if not edges[old_index] and any(
            current_index < len(current)
            and current[current_index][2] <= old[2]
            and abs(current[current_index][1] - old[1]) > 0.5
            for current_index in range(len(current))
        ):
            return False
    matched_current = set()

    def augment(old_index, seen):
        for current_index in sorted(edges[old_index]):
            if current_index in seen:
                continue
            seen.add(current_index)
            if current_index not in matched_current or augment(
                matched_current_map[current_index], seen
            ):
                matched_current.add(current_index)
                matched_current_map[current_index] = old_index
                return True
        return False

    matched_current_map = {}
    matched = sum(augment(old_index, set()) for old_index in sorted(edges))
    return matched < len(prior)


def _search_anchors(after, terminal):
    # A few forward scouts plus a distant sentinel; do not walk every stop.
    return tuple(sorted({index for index in (
        after + 1, after + 2, after + 4, after + 6,
        after + max(1, (terminal - after) // 2), terminal,
    ) if after < index <= terminal}))


def _physical_indices(values, terminal=None):
    """Return valid, distinct physical stop indices in ascending order."""
    indices = set()
    for value in values or ():
        if isinstance(value, bool):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if not isfinite(number) or not number.is_integer() or number < 0:
            continue
        index = int(number)
        if terminal is None or index <= terminal:
            indices.add(index)
    return tuple(sorted(indices))


def _next_poll_checkpoints(track, terminal):
    """Choose the smallest probe set which can refine one marker next."""
    if track.forward_after is not None:
        frontier = tuple(
            index for index in _physical_indices(track.forward_frontier, terminal)
            if index > track.forward_after
        )
        if not frontier:
            return ()
        sentinel = terminal if terminal in frontier else frontier[-1]
        # Center the representative over the remaining interior stops, not
        # over the terminus itself. On an even span, prefer the lower central
        # scout so a fresh corridor ending just before the midpoint remains
        # reachable in this poll.
        midpoint = (track.forward_after + sentinel - 1) / 2
        representative = min(
            frontier, key=lambda index: (abs(index - midpoint), index)
        )
        # The nearest response advances a just-departed marker; the midpoint
        # and terminus find a vehicle which crossed several stops between polls.
        return tuple(dict.fromkeys((frontier[0], representative, sentinel)))

    cold_hints = _physical_indices(
        getattr(track.estimate, "exploratory_indices", None), terminal
    )
    if (getattr(track.estimate, "position_authoritative", None) is False
            and cold_hints):
        priority = _physical_indices(
            getattr(track.estimate, "priority_indices", None), terminal
        )
        return tuple(dict.fromkeys((*cold_hints, *priority)))[:3]

    bracket = _physical_indices(getattr(track.estimate, "bracket", None), terminal)
    if not bracket:
        hints = _physical_indices(
            getattr(track.estimate, "exploratory_indices", None), terminal
        )
        if hints:
            priority = _physical_indices(
                getattr(track.estimate, "priority_indices", None), terminal
            )
            return tuple(dict.fromkeys((*hints, *priority)))[:3]
    if bracket:
        lower, upper = bracket[0], bracket[-1]
        exploratory = tuple(
            index for index in _physical_indices(
                getattr(track.estimate, "exploratory_indices", None), terminal
            )
            if lower < index < upper
        )[:2]
        if upper - lower > 1 and exploratory:
            # ETA-guided adjacent probes replace a coarse old boundary. One
            # projected integer still needs both established endpoints so its
            # response cannot erase the interval it was selected to refine.
            plan = (exploratory if len(exploratory) == 2 else
                    (lower, exploratory[0], upper))
        elif upper - lower > 1:
            # A binary step needs its prior lower and upper evidence alongside
            # the new midpoint. Omitting the lower lets the next presentation
            # reconstruct an older, wider baseline and repeat prior work.
            plan = (lower, (lower + upper) // 2, upper)
        else:
            plan = (lower, upper)
        # A complete candidate may be accepted for lifecycle continuity while
        # motion remains held at its prior boundary. Refresh one distinct old
        # motion upper so a wide candidate can eventually narrow, including
        # overlapping or degenerate prior brackets.
        motion = _physical_indices(track.motion_bracket, terminal)
        if upper - lower > MATCH_DISTANCE and motion:
            plan = tuple(dict.fromkeys((*plan, motion[-1])))
        return plan

    hints = _physical_indices(
        getattr(track.estimate, "priority_indices", None), terminal
    )[:2]
    if hints:
        return hints
    return (terminal,) if terminal is not None else ()


def _priority_population(recovery_plans, ordinary_plans):
    """Interleave per-track plans, with recovery plans before refinement."""
    population = []
    seen = set()
    for plans in (recovery_plans, ordinary_plans):
        groups = []
        for _track_id, values in sorted(plans, key=lambda item: item[0]):
            checkpoints = tuple(dict.fromkeys(values))
            if checkpoints:
                groups.append(checkpoints)
        for depth in range(max((len(group) for group in groups), default=0)):
            for group in groups:
                if depth < len(group) and group[depth] not in seen:
                    seen.add(group[depth])
                    population.append(group[depth])
    return tuple(population)


def _clear_forward_search(track):
    track.forward_after = None
    track.forward_revision = 0
    track.forward_started_revision = 0
    track.forward_frontier = ()
    track.forward_baselines.clear()


def _refresh_forward_search(track, checkpoints, terminal):
    if terminal is None:
        return
    if track.motion_bracket is None:
        return
    try:
        upper = float(track.motion_bracket[1])
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


def _temporal_checkpoint_score(owned, current):
    """Return one-to-one newer checkpoint continuity and timestamp drift."""
    owned_by_index = {}
    current_by_index = {}
    for index, arrival, revision in owned:
        owned_by_index.setdefault(index, []).append((arrival, revision))
    for index, arrival, revision in current:
        current_by_index.setdefault(index, []).append((arrival, revision))

    total_matches = 0
    total_drift_ms = 0
    for index in owned_by_index.keys() & current_by_index.keys():
        before = sorted(owned_by_index[index])
        after = sorted(current_by_index[index])
        # A candidate can exceptionally retain more than one occurrence at a
        # checkpoint after cluster repair. Solve that tiny ordered subproblem
        # exactly so equal-arrival multiplicity is never counted twice.
        states = [[(0, 0) for _item in range(len(after) + 1)]
                  for _item in range(len(before) + 1)]
        for old_count in range(1, len(before) + 1):
            for new_count in range(1, len(after) + 1):
                choices = [
                    states[old_count - 1][new_count],
                    states[old_count][new_count - 1],
                ]
                old_arrival, old_revision = before[old_count - 1]
                new_arrival, new_revision = after[new_count - 1]
                drift = abs(new_arrival - old_arrival)
                if (
                    new_revision > old_revision
                    and drift <= RECOVERY_ARRIVAL_TOLERANCE_SECONDS
                ):
                    matches, drift_ms = states[old_count - 1][new_count - 1]
                    choices.append((matches + 1, drift_ms + round(drift * 1000)))
                states[old_count][new_count] = max(
                    choices, key=lambda value: (value[0], -value[1])
                )
        matches, drift_ms = states[-1][-1]
        total_matches += matches
        total_drift_ms += drift_ms
    return total_matches, total_drift_ms


def _checkpoint_population_missing(old, checkpoints):
    """Return whether a fresh response lost an owned ETA occurrence.

    Compare the whole tracked population at each checkpoint so two vehicles
    with the same stop and arrival remain two occurrences. A non-empty response
    with an invalid timestamp is inconclusive; a successful empty response is
    valid absence evidence.
    """
    owned_by_index = {}
    for track in old:
        for index, arrival, revision in _checkpoint_rows(track.estimate):
            owned_by_index.setdefault(index, []).append((arrival, revision))

    for index, owned in owned_by_index.items():
        response = checkpoints.get(index)
        if response is None:
            continue
        response_revision, response_rows = response
        # Require a response newer than the entire owned population at this
        # checkpoint. This prevents a mixed old/current cache from supplying
        # false absence evidence for an occurrence it never refreshed.
        if response_revision <= max(revision for _arrival, revision in owned):
            continue
        current = []
        inconclusive = False
        for row in response_rows:
            if getattr(row, "minutes", None) is None:
                continue
            arrival = _arrival_timestamp(getattr(row, "arrival_at", None))
            if arrival is None:
                inconclusive = True
                break
            current.append(arrival)
        if inconclusive:
            continue

        before = sorted(arrival for arrival, _revision in owned)
        after = sorted(current)
        old_number = new_number = matched = 0
        while old_number < len(before) and new_number < len(after):
            drift = after[new_number] - before[old_number]
            if abs(drift) <= RECOVERY_ARRIVAL_TOLERANCE_SECONDS:
                matched += 1
                old_number += 1
                new_number += 1
            elif drift < 0:
                new_number += 1
            else:
                old_number += 1
        if matched < len(before):
            return True
    return False


def _identity_coverage_size(old, new):
    """Maximum one-to-one physical-continuity coverage of two populations."""
    def indexed(rows):
        grouped = {}
        for index, arrival, revision in rows:
            grouped.setdefault(index, []).append((arrival, revision))
        out = {}
        for index, values in grouped.items():
            values.sort()
            arrivals = tuple(arrival for arrival, _revision in values)
            revisions = tuple(revision for _arrival, revision in values)
            maxima = [revisions]
            width = 1
            while width * 2 <= len(values):
                previous = maxima[-1]
                maxima.append(tuple(
                    max(previous[offset], previous[offset + width])
                    for offset in range(len(values) - width * 2 + 1)
                ))
                width *= 2
            out[index] = (arrivals, revisions, tuple(maxima))
        return out

    def continues(owned, current):
        for index in owned.keys() & current.keys():
            old_arrivals, old_revisions, _old_maxima = owned[index]
            new_arrivals, _new_revisions, new_maxima = current[index]
            for old_arrival, old_revision in zip(
                old_arrivals, old_revisions, strict=True
            ):
                if bisect_left(new_arrivals, old_arrival - 0.5) < bisect_right(
                    new_arrivals, old_arrival + 0.5
                ):
                    return True
                lower = bisect_left(
                    new_arrivals,
                    old_arrival - RECOVERY_ARRIVAL_TOLERANCE_SECONDS,
                )
                upper = bisect_right(
                    new_arrivals,
                    old_arrival + RECOVERY_ARRIVAL_TOLERANCE_SECONDS,
                )
                if lower >= upper:
                    continue
                span = upper - lower
                level = span.bit_length() - 1
                width = 1 << level
                newest = max(
                    new_maxima[level][lower],
                    new_maxima[level][upper - width],
                )
                if newest > old_revision:
                    return True
        return False

    owned_rows = [indexed(_checkpoint_rows(track.estimate)) for track in old]
    current_rows = [indexed(_checkpoint_rows(candidate)) for candidate in new]
    edges = []
    for owned in owned_rows:
        track_edges = []
        for new_index, current in enumerate(current_rows):
            if continues(owned, current):
                track_edges.append(new_index)
        edges.append(track_edges)

    claimed = {}

    def augment(old_index, visited):
        for new_index in edges[old_index]:
            if new_index in visited:
                continue
            visited.add(new_index)
            owner = claimed.get(new_index)
            if owner is None or augment(owner, visited):
                claimed[new_index] = old_index
                return True
        return False

    return sum(
        augment(old_index, set()) for old_index in range(len(old))
    )


def _equal_count_population_turnover(old, new, checkpoints, *, matching_old=None):
    """Detect a proved departure plus uncovered replacement at equal count."""
    return (
        bool(old)
        and len(old) == len(new)
        and _checkpoint_population_missing(old, checkpoints)
        and _identity_coverage_size(old if matching_old is None else matching_old, new) < len(old)
    )


def _unique_ordered_temporal_pairs(old, new, compatible=None):
    """Return recovery edges shared by every best ordered temporal alignment."""
    positions = {}
    for old_index, track in enumerate(old):
        positions.setdefault(track.position, []).append(old_index)
    tied_old = {
        old_index for indices in positions.values() if len(indices) > 1
        for old_index in indices
    }
    edge_scores = {}
    for old_index, track in enumerate(old):
        # Stable list order is not identity evidence within an exact prior
        # position tie. The dedicated exact-checkpoint matcher proves unique
        # tied permutations; tolerant timestamps deliberately fail closed.
        if old_index in tied_old:
            continue
        owned = _checkpoint_rows(track.estimate)
        if not owned:
            continue
        for new_index, candidate in enumerate(new):
            if compatible is not None and not compatible(track, candidate):
                continue
            position = _bracket_position(track, candidate)
            if (
                position is None
                or (
                    abs(position - track.position) > MATCH_DISTANCE
                    and _long_jump_geometry(track, candidate) is None
                )
            ):
                continue
            current = _checkpoint_rows(candidate)
            matches, drift_ms = _temporal_checkpoint_score(owned, current)
            if matches:
                edge_scores[(old_index, new_index)] = (matches, drift_ms)

    # Score total checkpoint continuity first, then the number of identities
    # retained, then timestamp drift. Intersect equal optima instead of using a
    # deterministic tie-break as identity evidence.
    empty = ((0, 0, 0), frozenset())
    states = [[empty for _item in range(len(new) + 1)]
              for _item in range(len(old) + 1)]
    for old_count in range(1, len(old) + 1):
        for new_count in range(1, len(new) + 1):
            choices = [
                states[old_count - 1][new_count],
                states[old_count][new_count - 1],
            ]
            pair = (old_count - 1, new_count - 1)
            edge = edge_scores.get(pair)
            if edge is not None:
                previous_score, previous_required = states[old_count - 1][new_count - 1]
                choices.append((
                    (
                        previous_score[0] + edge[0],
                        previous_score[1] + 1,
                        previous_score[2] - edge[1],
                    ),
                    previous_required | {pair},
                ))
            maximum = max(choice[0] for choice in choices)
            optimal = [choice[1] for choice in choices if choice[0] == maximum]
            required = set(optimal[0])
            for pairs in optimal[1:]:
                required.intersection_update(pairs)
            states[old_count][new_count] = (maximum, frozenset(required))
    return set(states[-1][-1][1])


def _unique_tied_checkpoint_assignment(overlap, old_indices, new_indices):
    """Return a unique maximum perfect assignment for one prior-position tie."""

    def solve(rows, columns):
        row_maximum = []
        for row_number in rows:
            strength = max(
                (overlap[row_number][column] for column in columns),
                default=0,
            )
            winners = [
                column for column in columns
                if overlap[row_number][column] == strength
            ]
            if strength <= 0 or len(winners) != 1:
                row_maximum = []
                break
            row_maximum.append(winners[0])
        if len(row_maximum) == len(rows) and len(set(row_maximum)) == len(rows):
            return tuple(zip(rows, row_maximum, strict=True))

        # Ambiguous local evidence gets an exact bounded search. Splitting the
        # sparse overlap graph first keeps this bound independent of the total
        # number of markers tied at the same stop.
        if len(rows) > 12:
            return _unique_large_tied_assignment(overlap, rows, columns)
        states = {0: (0, (), ())}
        for row_number in rows:
            next_states = {}
            for used, value in states.items():
                for local_column, column in enumerate(columns):
                    if (
                        used & (1 << local_column)
                        or not overlap[row_number][column]
                    ):
                        continue
                    strength, first, last = value
                    item = (
                        strength + overlap[row_number][column],
                        first + ((row_number, column),),
                        last + ((row_number, column),),
                    )
                    key = used | (1 << local_column)
                    if key not in next_states or item[0] > next_states[key][0]:
                        next_states[key] = item
                    elif item[0] == next_states[key][0]:
                        next_states[key] = (
                            item[0],
                            min(item[1], next_states[key][1]),
                            max(item[2], next_states[key][2]),
                        )
            states = next_states
        result = states.get((1 << len(columns)) - 1)
        if result and result[0] and result[1] == result[2]:
            return result[1]
        return ()

    unmatched_rows = set(range(len(overlap)))
    local_pairs = []
    covered_columns = set()
    while unmatched_rows:
        rows = {min(unmatched_rows)}
        columns = set()
        while True:
            expanded_columns = columns | {
                column
                for row_number in rows
                for column, strength in enumerate(overlap[row_number])
                if strength
            }
            expanded_rows = rows | {
                row_number
                for row_number, weights in enumerate(overlap)
                if any(weights[column] for column in expanded_columns)
            }
            if expanded_rows == rows and expanded_columns == columns:
                break
            rows, columns = expanded_rows, expanded_columns
        if len(rows) != len(columns):
            return ()
        component_pairs = solve(tuple(sorted(rows)), tuple(sorted(columns)))
        if len(component_pairs) != len(rows):
            return ()
        local_pairs.extend(component_pairs)
        unmatched_rows.difference_update(rows)
        covered_columns.update(columns)
    if len(covered_columns) != len(new_indices):
        return ()
    return tuple(
        (old_indices[row_number], new_indices[column])
        for row_number, column in sorted(local_pairs)
    )


def _unique_large_tied_assignment(overlap, rows, columns):
    """Solve and prove uniqueness of a large weighted bipartite component."""
    size = len(rows)
    maximum = max(
        overlap[row][column] for row in rows for column in columns
    )
    forbidden = (maximum + 1) * (size + 1)
    costs = [
        [
            maximum - overlap[row][column]
            if overlap[row][column] else forbidden
            for column in columns
        ]
        for row in rows
    ]

    # Hungarian minimum-cost assignment, retaining its duals so uniqueness can
    # be checked in the tight-edge graph without repeated assignments.
    row_dual = [0] * (size + 1)
    column_dual = [0] * (size + 1)
    matched_row = [0] * (size + 1)
    predecessor = [0] * (size + 1)
    for row_number in range(1, size + 1):
        matched_row[0] = row_number
        column = 0
        minimum = [float("inf")] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[column] = True
            active_row = matched_row[column]
            delta = float("inf")
            next_column = 0
            for candidate_column in range(1, size + 1):
                if used[candidate_column]:
                    continue
                reduced = (
                    costs[active_row - 1][candidate_column - 1]
                    - row_dual[active_row]
                    - column_dual[candidate_column]
                )
                if reduced < minimum[candidate_column]:
                    minimum[candidate_column] = reduced
                    predecessor[candidate_column] = column
                if minimum[candidate_column] < delta:
                    delta = minimum[candidate_column]
                    next_column = candidate_column
            for candidate_column in range(size + 1):
                if used[candidate_column]:
                    row_dual[matched_row[candidate_column]] += delta
                    column_dual[candidate_column] -= delta
                else:
                    minimum[candidate_column] -= delta
            column = next_column
            if matched_row[column] == 0:
                break
        while True:
            previous_column = predecessor[column]
            matched_row[column] = matched_row[previous_column]
            column = previous_column
            if column == 0:
                break

    assignment = [-1] * size
    for column in range(1, size + 1):
        assignment[matched_row[column] - 1] = column - 1
    if any(
        overlap[rows[row_number]][columns[column]] <= 0
        for row_number, column in enumerate(assignment)
    ):
        return ()

    inverse = {column: row for row, column in enumerate(assignment)}
    adjacency = [set() for _ in range(size)]
    for row_number in range(size):
        for column in range(size):
            if column == assignment[row_number]:
                continue
            if overlap[rows[row_number]][columns[column]] <= 0:
                continue
            if (
                costs[row_number][column]
                == row_dual[row_number + 1] + column_dual[column + 1]
            ):
                adjacency[row_number].add(inverse[column])

    colors = [0] * size

    def cyclic(row_number):
        colors[row_number] = 1
        for neighbour in adjacency[row_number]:
            if colors[neighbour] == 1:
                return True
            if colors[neighbour] == 0 and cyclic(neighbour):
                return True
        colors[row_number] = 2
        return False

    if any(colors[row] == 0 and cyclic(row) for row in range(size)):
        return ()
    return tuple(
        (rows[row], columns[column])
        for row, column in enumerate(assignment)
    )


def _unique_ordered_checkpoint_pairs(old, new, compatible=None):
    """Return an exact checkpoint assignment only when its optimum is unique.

    A fast vehicle can retain an already-fetched downstream checkpoint while
    that row becomes its new positioning boundary. The unchanged response is
    strong identity evidence even though it is not a newer-response recovery
    anchor. Candidates may also absorb checkpoints from a neighbour, so select
    the sole maximum-cardinality route-ordered assignment or fail closed.
    """
    compatible = compatible or _search_compatible
    component_blocks = []
    unresolved_old = set()
    components = []
    for index, track in enumerate(old):
        if components and old[components[-1][0]].position == track.position:
            components[-1].append(index)
        else:
            components.append([index])
    for component in components:
        if len(component) < 2:
            continue
        lower_bound = old[component[0] - 1].position if component[0] else -inf
        upper_bound = (
            old[component[-1] + 1].position
            if component[-1] + 1 < len(old) else inf
        )
        candidates = [
            index for index, candidate in enumerate(new)
            if lower_bound < float(candidate.position or 0.0) < upper_bound
            and any(
                compatible(old[old_index], candidate)
                and _checkpoint_overlap(
                    _checkpoint_rows(old[old_index].estimate),
                    _checkpoint_rows(candidate),
                )
                for old_index in component
            )
        ]
        if not candidates:
            unresolved_old.update(component)
            continue
        pairs = ()
        if len(candidates) == len(component):
            overlap = [[len(_checkpoint_overlap(
                _checkpoint_rows(old[old_index].estimate),
                _checkpoint_rows(new[new_index]),
            )) if compatible(old[old_index], new[new_index]) else 0
                        for new_index in candidates] for old_index in component]
            pairs = _unique_tied_checkpoint_assignment(
                overlap, component, candidates
            )
        if not pairs:
            unresolved_old.update(component)
        component_blocks.append((
            component[0], component[-1], min(candidates), max(candidates), pairs
        ))

    # Independently solving a tied component and the whole ordered route can
    # reserve the same track or candidate twice. It can also let assignments
    # on opposite sides of two tied components cross. Accept only mutually
    # ordered component intervals, then solve each remaining route gap once.
    blocks = []
    for block in component_blocks:
        while blocks and blocks[-1][3] >= block[2]:
            previous = blocks.pop()
            block = (
                previous[0], block[1],
                min(previous[2], block[2]), max(previous[3], block[3]),
                (),
            )
        blocks.append(block)

    def gap_pairs(old_lower, old_upper, new_lower, new_upper):
        old_indices = [
            index for index in range(old_lower, old_upper)
            if index not in unresolved_old
        ]
        new_indices = list(range(new_lower, new_upper))
        return _mandatory_ordered_checkpoint_pairs(
            [old[index] for index in old_indices],
            [new[index] for index in new_indices],
            compatible,
            old_indices=old_indices,
            new_indices=new_indices,
        )

    exact = set()
    old_start = new_start = 0
    for old_lower, old_upper, new_lower, new_upper, pairs in blocks:
        exact.update(gap_pairs(old_start, old_lower, new_start, new_lower))
        exact.update(pairs)
        old_start = old_upper + 1
        new_start = new_upper + 1
    exact.update(gap_pairs(old_start, len(old), new_start, len(new)))
    return exact


def _mandatory_ordered_checkpoint_pairs(
    old, new, compatible, *, old_indices=None, new_indices=None
):
    """Return checkpoint pairs shared by every best ordered alignment."""
    old_indices = tuple(range(len(old))) if old_indices is None else tuple(old_indices)
    new_indices = tuple(range(len(new))) if new_indices is None else tuple(new_indices)
    dp = [[((0, 0), frozenset()) for _ in range(len(new) + 1)]
          for _ in range(len(old) + 1)]
    for old_count in range(1, len(old) + 1):
        for new_count in range(1, len(new) + 1):
            choices = [dp[old_count - 1][new_count],
                       dp[old_count][new_count - 1]]
            track = old[old_count - 1]
            candidate = new[new_count - 1]
            overlap = len(_checkpoint_overlap(
                _checkpoint_rows(track.estimate), _checkpoint_rows(candidate)
            ))
            if compatible(track, candidate) and overlap:
                score, required = dp[old_count - 1][new_count - 1]
                pair = (
                    old_indices[old_count - 1],
                    new_indices[new_count - 1],
                )
                choices.append((
                    (score[0] + overlap, score[1] + 1),
                    required | {pair},
                ))
            maximum = max(choice[0] for choice in choices)
            optimal = [choice[1] for choice in choices if choice[0] == maximum]
            required = set(optimal[0])
            for pairs in optimal[1:]:
                required.intersection_update(pairs)
            dp[old_count][new_count] = (maximum, frozenset(required))
    return set(dp[-1][-1][1])


def _long_jump_geometry(track, candidate):
    """Return a fresh forward corridor or nested correction for a long jump."""
    position = _bracket_position(track, candidate)
    if position is None or abs(position - track.position) <= MATCH_DISTANCE:
        return None
    try:
        lower, upper = map(float, candidate.bracket)
    except (AttributeError, TypeError, ValueError):
        return None
    nested = (
        getattr(track.estimate, "bracket", None) is not None
        and track.motion_bracket is not None
        and track.motion_bracket[0] <= lower <= upper <= track.motion_bracket[1]
    )
    # A newly fetched interior pair may sharply correct an initially sparse,
    # coarse bracket. It cannot move the bus outside the last physical search
    # interval, and the temporal assignment still has to be uniquely optimal.
    # Forward jumps retain the narrow-corridor requirement.
    if upper - lower > MATCH_DISTANCE and not nested:
        return None
    return (position, lower, upper) if position > track.position or nested else None


def _recovery_pairs(old, new, checkpoints, compatible=None):
    """Uniquely evidenced long jumps, not a wider nearest-neighbour radius."""
    possible = _unique_ordered_temporal_pairs(old, new, compatible)
    for i, track in enumerate(old):
        for j, candidate in enumerate(new):
            if compatible is not None and not compatible(track, candidate):
                continue
            jump = _long_jump_geometry(track, candidate)
            if jump is None:
                continue
            position, lower, upper = jump
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


def _recovery_plan(old, new, checkpoints, compatible=None):
    """Separate fixed identity associations from safe long-distance motion."""
    fixed = _unique_ordered_checkpoint_pairs(old, new, compatible)
    movement = _recovery_pairs(old, new, checkpoints, compatible)
    if compatible is not None:
        fixed = {
            (old_index, new_index)
            for old_index, new_index in fixed
            if compatible(old[old_index], new[new_index])
        }
        movement = {
            (old_index, new_index)
            for old_index, new_index in movement
            if compatible(old[old_index], new[new_index])
        }

    def conflicts_with_fixed(pair):
        old_index, new_index = pair
        return any(
            pair != anchor
            and (
                old_index == anchor[0]
                or new_index == anchor[1]
                or (old_index - anchor[0]) * (new_index - anchor[1]) < 0
            )
            for anchor in fixed
        )

    # Exact immutable ownership outranks a weaker refreshed or timestamp-free
    # recovery. Keep only weaker edges which fit around every fixed route-order
    # anchor; otherwise their reservations could displace the exact assignment.
    movement = {
        pair for pair in movement
        if pair in fixed or not conflicts_with_fixed(pair)
    }
    safe_exact_motion = {
        pair for pair in fixed
        if _long_jump_geometry(old[pair[0]], new[pair[1]]) is not None
    }
    return fixed, fixed | movement, movement | safe_exact_motion


def _complete_pair_plan(old, new, checkpoints, compatible=None,
                        identity_compatible=None, reconciled_owners=None):
    """Reserve unique identities before matching the remaining route order.

    Mutually unique checkpoint evidence may retain a never-position-confirmed
    timetable identity after a reliable marker passes its held coordinate.
    Explicitly cold candidates have a separate strict identity graph: their
    projections and forward-search fences are not physical position evidence.
    Historical authority still constrains the later motion acceptance step.
    """
    exact_edges = {}
    reverse_edges = {index: set() for index in range(len(new))}
    for old_index, track in enumerate(old):
        edges = set()
        for new_index, candidate in enumerate(new):
            if compatible is not None and not compatible(track, candidate):
                continue
            if _checkpoint_overlap(
                _cohort_rows(track), _checkpoint_rows(candidate)
            ):
                edges.add(new_index)
                reverse_edges[new_index].add(old_index)
        exact_edges[old_index] = edges

    tentative = {
        (old_index, next(iter(edges)))
        for old_index, edges in exact_edges.items()
        if len(edges) == 1
        if len(reverse_edges[next(iter(edges))]) == 1
        if not _position_order_authoritative(old[old_index], new[next(iter(edges))])
    }
    cold_reserved = _complete_cold_reservations(
        old, new, identity_compatible, reconciled_owners,
    )
    # Preserve the existing tentative assignment, accepting cold reservations
    # only when their endpoints agree with every tentative reservation.
    cold_reserved = {
        pair for pair in cold_reserved
        if all(pair == other or (pair[0] != other[0] and pair[1] != other[1])
               for other in tentative)
    }
    reserved = tentative | cold_reserved
    if not reserved:
        fixed, recoveries, movement = _recovery_plan(
            old, new, checkpoints, compatible=compatible
        )
        pairs = _ordered_pairs(
            old,
            new,
            compatible=compatible,
            recoveries=recoveries,
            fixed=fixed,
        )
        return fixed, movement, pairs, set()

    reserved_old = {old_index for old_index, _new_index in reserved}
    reserved_new = {new_index for _old_index, new_index in reserved}
    old_indices = [index for index in range(len(old)) if index not in reserved_old]
    new_indices = [index for index in range(len(new)) if index not in reserved_new]
    remaining_old = [old[index] for index in old_indices]
    remaining_new = [new[index] for index in new_indices]
    fixed, recoveries, movement = _recovery_plan(
        remaining_old,
        remaining_new,
        checkpoints,
        compatible=compatible,
    )

    def remap(pairs):
        return {
            (old_indices[old_index], new_indices[new_index])
            for old_index, new_index in pairs
        }

    ordered = _ordered_pairs(
        remaining_old,
        remaining_new,
        compatible=compatible,
        recoveries=recoveries,
        fixed=fixed,
    )
    return (
        remap(fixed) | reserved,
        remap(movement),
        sorted(remap(ordered) | reserved),
        cold_reserved,
    )


def _complete_cold_reservations(old, new, identity_compatible=None, reconciled_owners=None):
    """Prove cold continuity against every trusted owner and current claimant."""
    current = [_strict_checkpoint_rows(candidate) for candidate in new]
    # Dropping a malformed competitor would falsely make another claim unique.
    # Empty ledgers are valid no-evidence rows; malformed ledgers invalidate
    # this census's cold uniqueness proof, including any certified edges.
    if any(ledger is None for ledger in current):
        return set()
    reconciled_owners = reconciled_owners or {}
    edges = {}
    reverse = {index: set() for index in range(len(new))}
    for old_index, track in enumerate(old):
        if track.cohort_observed_at > 0:
            if not track.cohort_trusted:
                continue
            owned = _strict_checkpoint_rows(replace(
                track.estimate, checkpoint_evidence=track.cohort_evidence,
            ))
        else:
            owned = _strict_checkpoint_rows(track.estimate)
        if owned is None:
            return set()
        if not owned:
            continue
        edges[old_index] = set()
        for new_index, candidate in enumerate(new):
            ledger = current[new_index]
            if not ledger or (identity_compatible is not None
                              and not identity_compatible(track, candidate)):
                continue
            # Atomic raw-slot reconciliation has already proved full injective
            # ownership, including legitimate refreshed ETA drift. Keep that
            # stronger certificate alongside the strict raw exact edges.
            certified = reconciled_owners.get(id(candidate)) == track.track_id
            if certified or any(
                old_stop == new_stop and abs(old_arrival - new_arrival) <= 0.5
                and new_revision >= old_revision
                for old_stop, old_arrival, old_revision in owned
                for new_stop, new_arrival, new_revision in ledger
            ):
                edges[old_index].add(new_index)
                reverse[new_index].add(old_index)
    return {
        (old_index, new_index)
        for old_index, candidates in edges.items()
        if len(candidates) == 1
        for new_index in candidates
        if len(reverse[new_index]) == 1
        if getattr(new[new_index], "position_authoritative", None) is False
    }


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
    if track.cohort_observed_at > 0 and not track.cohort_trusted:
        return
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


def _ordered_pairs(old, new, compatible=None, recoveries=(), fixed=()):
    # Score an ordered alignment by retained cardinality first, then by ETA-
    # anchor presence continuity. ETA timestamps can drift by tens of seconds
    # between provider generations, while an unbracketed turnover candidate
    # has no timestamp at all. Letting plain distance make that mixed-anchor
    # pair look cheaper can assign the turnover candidate to a surviving
    # downstream track and then birth a second marker from its source ladder.
    # Source-row slots can shift after a departure, so overlap remains only a
    # deterministic final tie-breaker rather than an identity authority.
    recoveries = set(recoveries)
    fixed = set(fixed)
    if fixed:
        components = []
        for index, track in enumerate(old):
            if components and old[components[-1][0]].position == track.position:
                components[-1].append(index)
            else:
                components.append([index])
        for component in components:
            block = sorted(pair for pair in fixed if pair[0] in component)
            if len(block) < 2 or not any(
                left[1] > right[1]
                for left in block for right in block
                if left[0] < right[0]
            ):
                continue
            old_indices = {index for index, _new_index in block}
            new_indices = {index for _old_index, index in block}
            if old_indices != set(component) or len(new_indices) != len(component):
                continue
            old_lower, old_upper = component[0], component[-1]
            new_lower, new_upper = min(new_indices), max(new_indices)
            prefix_fixed = {
                pair for pair in fixed
                if pair[0] < old_lower and pair[1] < new_lower
            }
            suffix_fixed = {
                (old_index - old_upper - 1, new_index - new_upper - 1)
                for old_index, new_index in fixed
                if old_index > old_upper and new_index > new_upper
            }
            if fixed - set(block) != prefix_fixed | {
                (old_index + old_upper + 1, new_index + new_upper + 1)
                for old_index, new_index in suffix_fixed
            }:
                continue
            prefix_recoveries = {
                pair for pair in recoveries
                if pair[0] < old_lower and pair[1] < new_lower
            }
            suffix_recoveries = {
                (old_index - old_upper - 1, new_index - new_upper - 1)
                for old_index, new_index in recoveries
                if old_index > old_upper and new_index > new_upper
            }
            prefix = _ordered_pairs(
                old[:old_lower],
                new[:new_lower],
                compatible,
                recoveries=prefix_recoveries,
                fixed=prefix_fixed,
            )
            suffix = _ordered_pairs(
                old[old_upper + 1:],
                new[new_upper + 1:],
                compatible,
                recoveries=suffix_recoveries,
                fixed=suffix_fixed,
            )
            suffix = [
                (old_upper + 1 + old_index, new_upper + 1 + new_index)
                for old_index, new_index in suffix
            ]
            return prefix + block + suffix
    reserved_old = {i for i, _j in recoveries}
    reserved_new = {j for _i, j in recoveries}
    dp = [[(0, 0, 0, 0.0, ()) for _ in range(len(new) + 1)]
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
                        previous[0] + int(pair in fixed),
                        previous[1] + 1,
                        previous[2] + int(anchor_mismatch),
                        previous[3] + (
                            anchor_distance * 10.0 + distance * 0.01
                            if old_anchor is not None and new_anchor is not None
                            else distance
                        ),
                        previous[4] + ((0 if overlap else 1, i - 1, j - 1),),
                    )
                )
            dp[i][j] = min(
                choices,
                key=lambda item: (
                    -item[0], -item[1], item[2], item[3], item[4]
                ),
            )
    return [(item[1], item[2]) for item in dp[-1][-1][4]]


def _bracket_position(track, candidate):
    """Use only a freshly observed boundary to reposition a marker."""
    if not _candidate_actionable(candidate, track):
        return None
    return _fresh_bracket_position(candidate)


def _paired_position(track, candidate, allow_long_jump):
    """Move a paired identity only within radius or an approved recovery."""
    position = _bracket_position(track, candidate)
    if position is None:
        return None
    if abs(position - track.position) <= MATCH_DISTANCE:
        return position
    # A candidate can be rebuilt after identity assignment (for example when
    # certified sparse fragments restore its final-due/first-future boundary).
    # Never let movement approval computed for the pre-rebuild geometry bypass
    # the current candidate's narrow/nested long-jump guard.
    if allow_long_jump and _long_jump_geometry(track, candidate) is not None:
        return position
    return None


def _stable_gate_position(track, candidate):
    """Permit a narrow gate continuation without advancing motion freshness."""
    if getattr(candidate, "boundary_revision", None) is not None:
        return None
    if getattr(candidate, "unreliable", False):
        return None
    sources = getattr(candidate, "source_observations", ()) or ()
    if not any(str(source[0]).lower() == "gate" for source in sources if source):
        return None
    old_rows = _checkpoint_rows(track.estimate)
    current_rows = _checkpoint_rows(candidate)
    matched = _checkpoint_overlap_pairs(old_rows, current_rows)
    if len(matched) < 2:
        return None
    newer = sum(
        1 for old_number, current_number in matched
        if current_rows[current_number][2] > old_rows[old_number][2]
    )
    if newer < 2:
        return None
    try:
        lower, upper = map(float, candidate.bracket)
        position = float(candidate.position)
    except (AttributeError, TypeError, ValueError):
        return None
    if (not all(isfinite(value) for value in (lower, upper, position))
            or lower > upper or upper - lower > MATCH_DISTANCE
            or position <= track.position
            or position - track.position > MATCH_DISTANCE):
        return None
    return position


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


def _shares_exact_checkpoint(track, candidate):
    """Match one exact physical checkpoint/ETA/response observation."""
    owned = _checkpoint_rows(track.estimate)
    current = _checkpoint_rows(candidate)
    return bool(_checkpoint_overlap(owned, current))


def _same_positioning_owner(track, candidate):
    """Recognize a stable ETA identity across a one-sided boundary refresh."""
    revision = _candidate_revision(candidate)
    previous = _candidate_revision(track.estimate) or track.boundary_revision
    if (
        revision is None
        or previous is None
        or revision[1] < previous[1]
    ):
        return False
    try:
        old_lower, old_upper = map(float, track.estimate.bracket)
        new_lower, new_upper = map(float, candidate.bracket)
    except (AttributeError, TypeError, ValueError):
        return False
    if not all(isfinite(value) for value in (
        old_lower, old_upper, new_lower, new_upper
    )):
        return False
    if (
        old_lower > old_upper
        or new_lower > new_upper
        or abs(old_upper - new_upper) > 1e-6
    ):
        return False
    old_arrival = _arrival_timestamp(
        getattr(track.estimate, "eta_arrival_at", None)
    )
    new_arrival = _arrival_timestamp(getattr(candidate, "eta_arrival_at", None))
    if (
        old_arrival is None
        or new_arrival is None
        or abs(old_arrival - new_arrival) > 0.5
        or not new_upper.is_integer()
    ):
        return False
    upper_index = int(new_upper)
    old_rows = _checkpoint_rows(track.estimate)
    current_rows = _checkpoint_rows(candidate)
    old_boundary = any(
        index == upper_index
        and row_revision == previous[1]
        and abs(arrival - old_arrival) <= 0.5
        for index, arrival, row_revision in old_rows
    )
    current_boundary = any(
        index == upper_index
        and row_revision == revision[1]
        and abs(arrival - new_arrival) <= 0.5
        for index, arrival, row_revision in current_rows
    )
    # If the upper response itself refreshed, require another immutable row
    # from this track to survive in the rebuilt candidate. When the response
    # is unchanged, that exact upper row is already the retained checkpoint.
    endpoint_refreshed = any(
        index == upper_index and row_revision != previous[1]
        for index, _arrival, row_revision in current_rows
    )
    retained_checkpoint = any(
        old_index == new_index
        and abs(old_checkpoint_arrival - new_checkpoint_arrival) <= 0.5
        and (not endpoint_refreshed or new_index != upper_index)
        for old_index, old_checkpoint_arrival, _old_revision in old_rows
        for new_index, new_checkpoint_arrival, _new_revision in current_rows
    )
    return old_boundary and current_boundary and retained_checkpoint


def _checkpoint_ownership_profile(tracks, candidate):
    candidate_rows = _checkpoint_rows(candidate)
    owner_signatures = []
    exact_owner_sets = [set() for _ in candidate_rows]
    raw_temporal_owner_sets = [set() for _ in candidate_rows]
    temporal_eligible = {}
    for owner in tracks:
        owned = _checkpoint_rows(owner.estimate)
        exact = {
            row_number for row_number in _checkpoint_overlap(owned, candidate_rows)
        }
        for row_number in exact:
            exact_owner_sets[row_number].add(owner.track_id)
        # A refreshed checkpoint may move by several seconds.  Keep only
        # globally unambiguous one-to-one continuations, so a candidate that
        # splices two buses' newer rows fails closed instead of choosing one.
        temporal = []
        for old_number, old in enumerate(owned):
            matches = [
                new_number for new_number, new in enumerate(candidate_rows)
                if _evidence_continues(old, new)
            ]
            if len(matches) == 1:
                temporal.append((old_number, matches[0]))
            for new_number in matches:
                raw_temporal_owner_sets[new_number].add(owner.track_id)
        temporal_eligible[owner.track_id] = {
            new_number for _old_number, new_number in temporal
            if sum(match == new_number for _old, match in temporal) == 1
        }
        owner_signatures.append((owner.track_id, exact))
    # Derive only after all owners have contributed edges; this is deliberately
    # row-local so an ambiguous row cannot suppress unrelated exclusive rows.
    exclusive_rows = []
    for row_number, raw_owners in enumerate(raw_temporal_owner_sets):
        exact_owners = exact_owner_sets[row_number]
        if exact_owners:
            exclusive_rows.append(next(iter(exact_owners))
                                  if len(exact_owners) == 1 else None)
            continue
        eligible = {
            owner_id for owner_id, rows in temporal_eligible.items()
            if row_number in rows
        }
        exclusive_rows.append(next(iter(eligible))
                              if len(raw_owners) == 1 and len(eligible) == 1
                              else None)
    temporal_signatures = {
        owner_id: {row_number for row_number, value in enumerate(exclusive_rows)
                   if value == owner_id}
        for owner_id, _exact in owner_signatures
    }
    strong_owners = {
        owner_id for owner_id, rows in temporal_signatures.items()
        if any(row_number in rows and len(exact_owner_sets[row_number]) == 1
               for row_number in rows)
        or len({candidate_rows[row_number][0] for row_number in rows
                if not exact_owner_sets[row_number]}) >= 2
    }
    exclusive_owners = {
        owner_id for owner_id, signature in temporal_signatures.items()
        if owner_id in strong_owners and signature
    }
    prior_owned = {signature for _owner_id, signatures in owner_signatures
                   for signature in signatures}
    exact_owner_ids = frozenset(
        owner_id for owner_id, signatures in owner_signatures if signatures
    )
    covering_owner_ids = frozenset(
        owner_id for owner_id, signatures in owner_signatures
        if signatures == prior_owned and signatures
    )
    positioning_owner_ids = frozenset(
        owner.track_id for owner in tracks
        if _same_positioning_owner(owner, candidate)
    )
    return (
        exact_owner_ids,
        covering_owner_ids,
        prior_owned,
        positioning_owner_ids,
        frozenset(exclusive_owners) if len(exclusive_owners) > 1 else frozenset(),
    )


def _same_generation_actionable(tracks, track, candidate, ownership=None):
    """Keep stable checkpoint evidence within its prior ownership set.

    Staggered stop responses can merge checkpoint evidence from two adjacent
    buses into one candidate. That makes both prior tracks plausible owners;
    it does not make the candidate evidence for a third track. If there is no
    exact prior evidence, ordered matching retains its normal fallback.
    """
    if (track.cohort_observed_at > 0 and not track.cohort_trusted
            or not _candidate_actionable(candidate, track)):
        return False
    if ownership is None:
        ownership = _checkpoint_ownership_profile(tracks, candidate)
    (
        exact_owner_ids,
        covering_owner_ids,
        prior_owned,
        positioning_owner_ids,
        mixed_owner_ids,
    ) = ownership
    # A staggered partial response can splice distinct immutable rows from
    # adjacent vehicles.  No single prior owner covers that evidence, so hold
    # all tracks. Identical shared rows remain eligible for normal matching.
    if mixed_owner_ids:
        return False
    if prior_owned and not covering_owner_ids:
        return False
    if covering_owner_ids:
        # A unique covering owner is authoritative: a positioning match from
        # another track cannot absorb its exclusive immutable checkpoint.
        if positioning_owner_ids:
            owner_ids = positioning_owner_ids & covering_owner_ids
            if not owner_ids:
                return False
        else:
            owner_ids = covering_owner_ids
    else:
        owner_ids = positioning_owner_ids or exact_owner_ids
    return not owner_ids or track.track_id in owner_ids


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


def _position_order_authoritative(track, candidate):
    """Whether a held identity can constrain another marker's exact motion.

    A timetable-only placeholder with no retained physical boundary has no
    trustworthy route coordinate. Keep that marker fixed, but do not let its
    tentative coordinate veto a different identity's fresh boundary. Any
    current or historical position authority preserves the strict order rule.
    """
    if (track.position_authoritative or track.boundary_revision is not None
            or track.boundary_observed_at is not None):
        return True
    explicit = getattr(track.estimate, "position_authoritative", None)
    if explicit is False or getattr(candidate, "position_authoritative", None) is False:
        return False
    return (
        not bool(getattr(track.estimate, "unreliable", False))
        or not bool(getattr(candidate, "unreliable", False))
    )


def _select_ordered_updates(old, proposed_positions, eligible_indices=None):
    """Keep the largest exact-update subset that cannot reorder identities.

    Tracks tied at the prior position are one unordered component and may
    split when fresh evidence distinguishes them. Distinct prior-position
    components retain their global order. Rejected proposals keep both their
    old position and old evidence instead of relabelling an adjusted point as
    the fresh ETA-proportionate position.
    """
    if not proposed_positions:
        return set()

    eligible_indices = (
        set(range(len(old)))
        if eligible_indices is None
        else set(eligible_indices)
    )
    active = [index for index in range(len(old)) if index in eligible_indices]
    if not active:
        return set()

    components = []
    for index in active:
        if not components or old[components[-1][0]].position != old[index].position:
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


def _retain_current_checkpoint_capacity(
    old,
    current_population,
    update_candidates,
    pairs,
    accepted_updates,
):
    """Do not let a partial refresh publish one ETA occurrence twice.

    Complete generations own lifecycle cardinality. Between them, a staggered
    candidate may contain an exact occurrence still displayed by another track.
    Apply all safe swaps together, but reject any update which would *increase*
    the displayed multiplicity of a current occurrence beyond its multiplicity
    in the estimator population. Existing equal-time multiplicity is left
    untouched; this guard cannot collapse two legitimate vehicles.
    """
    accepted = set(accepted_updates)
    if not accepted:
        return accepted
    capacity = Counter(
        row
        for candidate in current_population
        for row in _checkpoint_rows(candidate)
    )
    if not capacity:
        return accepted
    candidate_by_old = {
        old_index: update_candidates[new_index]
        for old_index, new_index in pairs
        if old_index in accepted
    }
    accepted.intersection_update(candidate_by_old)
    old_counts = {
        index: Counter(_checkpoint_rows(track.estimate))
        for index, track in enumerate(old)
    }
    candidate_counts = {
        index: Counter(_checkpoint_rows(candidate))
        for index, candidate in candidate_by_old.items()
    }
    while True:
        displayed = Counter()
        for index, track in enumerate(old):
            estimate = (
                candidate_by_old[index]
                if index in accepted
                else track.estimate
            )
            displayed.update(
                row for row in _checkpoint_rows(estimate) if row in capacity
            )
        over_capacity = {
            row for row, count in displayed.items()
            if count > capacity[row]
        }
        if not over_capacity:
            return accepted
        additions = {
            index for index in accepted
            if any(
                candidate_counts[index][row] > old_counts[index][row]
                for row in over_capacity
            )
        }
        if not additions:
            # The partial update did not create this ambiguity. Preserve the
            # prior lifecycle population for the next complete generation.
            return accepted
        accepted.difference_update(additions)


def _select_valid_partial_transaction(
    old, current_population, update_candidates, pairs, proposed_positions,
    *, birth_candidate=None, metadata_indices=None,
):
    """Select a monotone, ordered partial update and optional certified birth."""
    metadata = set(metadata_indices or ())
    revision_floors = _route_metadata_revision_floors(old)
    regressing = {
        old_index for old_index, new_index in pairs
        if not _metadata_nonregressing(update_candidates[new_index], revision_floors)
    }
    metadata.difference_update(regressing)
    proposed_positions = {index: position for index, position in proposed_positions.items()
                          if index not in regressing}
    motion = _select_ordered_updates(old, proposed_positions)
    accepted = motion | metadata
    while True:
        capacity_accepted = _retain_current_checkpoint_capacity(
            old, current_population, update_candidates, pairs, accepted,
        )
        ordered_accepted = _select_ordered_updates(
            old,
            {
                index: position for index, position in proposed_positions.items()
                if index in capacity_accepted
            },
        )
        narrowed = accepted & capacity_accepted & (ordered_accepted | metadata)
        if narrowed == accepted:
            motion = ordered_accepted & accepted
            break
        accepted = narrowed

    def result(birth_allowed):
        if metadata_indices is None:
            return accepted, birth_allowed
        return accepted, birth_allowed, motion

    if birth_candidate is None:
        return result(True)
    if not _metadata_nonregressing(birth_candidate, revision_floors):
        return result(False)
    capacity = Counter(
        row
        for candidate in current_population
        for row in _checkpoint_rows(candidate)
    )
    if not capacity:
        return result(True)
    candidate_by_old = {
        old_index: update_candidates[new_index]
        for old_index, new_index in pairs
        if old_index in accepted
    }
    displayed = Counter()
    for index, track in enumerate(old):
        estimate = candidate_by_old.get(index, track.estimate)
        displayed.update(
            row for row in _checkpoint_rows(estimate) if row in capacity
        )
    birth_counts = Counter(
        row for row in _checkpoint_rows(birth_candidate) if row in capacity
    )
    return result(all(
        displayed[row] + count <= capacity[row]
        for row, count in birth_counts.items()
    ))


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
        bracket=track.display_bracket,
        boundary_revision=track.boundary_revision,
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
