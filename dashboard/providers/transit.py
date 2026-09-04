"""Transit ETA provider: KMB, Citybus, and GMB.

Mappings are ordered immutable tuples; route order in the dashboard follows the
order declared here. All functions take an injected ``HttpClient`` and are safe
to call concurrently.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Any

from dashboard.http import FetchError, HttpClient
from dashboard.models import EtaKind, EtaRow, Operator, RouteEtaGroup

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Stop/route configuration (verified 2024-09-23, see private transport-mappings.md)
# --------------------------------------------------------------------------

# KMB bound letters: the API returns "O"/"I"; tracked directions are named
# "outbound"/"inbound".
_KMB_DIR_TO_BOUND = {"o": "outbound", "i": "inbound"}

KMB_STOPS: tuple[dict[str, str], ...] = (
    {"gate": "S", "route": "91", "dest": "Diamond Hill", "stop": "B002CEF0DBC568F5"},
    {"gate": "S", "route": "91M", "dest": "Diamond Hill", "stop": "B002CEF0DBC568F5"},
    {"gate": "S", "route": "91P", "dest": "Choi Hung", "stop": "E9018F8A7E096544"},
    {"gate": "S", "route": "291P", "dest": "Mong Kok", "stop": "E9018F8A7E096544"},
    {"gate": "N", "route": "91", "dest": "Clear Water Bay", "stop": "3592A0182BF020C7"},
    {"gate": "N", "route": "91M", "dest": "Po Lam", "stop": "B3E60EE895DBBF06"},
)

CTB_STOPS: tuple[dict[str, str], ...] = (
    {"gate": "O", "route": "792M", "dest": "Sai Kung", "stop": "003130"},
    {"gate": "I", "route": "792M", "dest": "TKO", "stop": "003130"},
)

# stop_id -> tuple of (route_no, destination, gate, route_id, seq)
# For CIRCULAR routes (e.g. 104), the same route appears at MULTIPLE
# stop_seq along the loop; the far-end stop's ETAs are the bus RETURNING
# to HKUST. We keep only the departure stop (the first stop_seq in the
# route's stop sequence at this stop_id) so arrivals are never shown.
GMB_STOPS: dict[int, tuple[tuple[str, str, str, int, int], ...]] = {
    20013010: (("11", "Choi Hung", "S", 2004791, 1),),
    20012472: (("11", "Hang Hau", "N", 2004791, 2),),
    20013011: (
        ("11B", "Choi Hung", "S", 2004828, 1),  # directional variant: NOT 2004827
        ("11S", "Choi Hung", "S", 2004826, 1),
    ),
    20012474: (
        ("11M", "Hang Hau", "N", 2004825, 2),
        ("11S", "Po Lam", "N", 2004826, 2),
        ("12", "Po Lam", "N", 2004764, 1),
        ("12", "Sai Kung", "N", 2004764, 2),
    ),
    20015226: (("104", "Kwun Tong", "S", 2007200, 1),),
}

KMB_BASE = "https://data.etabus.gov.hk/v1/transport/kmb/stop-eta/{stop}"
KMB_ROUTE_ETA_BASE = "https://data.etabus.gov.hk/v1/transport/kmb/route-eta/{route}/1"
CTB_BASE = "https://rt.data.gov.hk/v2/transport/citybus/eta/CTB/{stop}/{route}"
GMB_BASE = "https://data.etagmb.gov.hk/eta/stop/{stop}"
GMB_COOLDOWN_SECONDS = 60.0
_gmb_cooldown_until = 0.0
# The provider tolerates twenty sequential GMB requests in a normal cycle in
# current observations.  Keep the live limit process-local: a 403 lowers it
# for the remainder of this process, while a restart restores the initial
# value.  The cooldown timestamp also acts as the 403 episode marker, so two
# requests already in flight cannot double-decrement the limit.
GMB_GROUPS_PER_CYCLE = 20


class _GmbGateCache:
    """Immutable last-good gate ETAs while the shared GMB origin cools down."""

    TTL_SECONDS = 900.0

    def __init__(self) -> None:
        self._stored: tuple[float, list[EtaRow]] | None = None

    def set(self, rows: list[EtaRow]) -> None:
        self._stored = (time.monotonic(), list(rows))

    def get(self) -> list[EtaRow]:
        if self._stored is None:
            return []
        stamped, rows = self._stored
        age_seconds = max(0.0, time.monotonic() - stamped)
        if age_seconds > self.TTL_SECONDS:
            self._stored = None
            return []
        return list(rows)


_gmb_gate_cache = _GmbGateCache()

# ETA network refreshes are deliberately slower than the map/render cadence.
# The dashboard may render every ten seconds, but polling the estimator feeds
# every render tick causes avoidable bursts (especially against GMB). Cached
# observations retain their source values and carry cache age between refreshes.
TRANSIT_NETWORK_REFRESH_SECONDS = 30.0


@dataclass
class _GateEtaCache:
    stored: tuple[float, list[EtaRow], datetime | None, list[str]] | None = None

    def set(
        self,
        rows: list[EtaRow],
        latest: datetime | None,
        failed: list[str],
    ) -> None:
        self.stored = (time.monotonic(), list(rows), latest, list(failed))

    def get(self) -> tuple[list[EtaRow], datetime | None, list[str]] | None:
        if self.stored is None:
            return None
        stamped, rows, latest, failed = self.stored
        return list(rows), latest, list(failed)


_gate_eta_cache = _GateEtaCache()
_gate_network_refresh_at: float | None = None
_gate_refresh_task: asyncio.Task | None = None
_probe_network_refresh_at: float | None = None
_probe_refresh_task: asyncio.Task | None = None
_gate_refresh_waiters = 0
_probe_refresh_waiters = 0


# --------------------------------------------------------------------------
# Parsing helpers
# --------------------------------------------------------------------------


def _precise_minutes_until(iso: str | None, now: datetime) -> float | None:
    if not iso:
        return None
    try:
        eta = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except ValueError:
        return None
    if eta.tzinfo is None:
        eta = eta.replace(tzinfo=UTC)
    diff = (eta - now).total_seconds() / 60
    return max(0.0, diff)


def _minutes_until(iso: str | None, now: datetime) -> int | None:
    precise = _precise_minutes_until(iso, now)
    return round(precise) if precise is not None else None


def _kmb_kind(rmk: str) -> EtaKind:
    rmk = (rmk or "").lower()
    if "scheduled" in rmk:
        return EtaKind.SCHEDULED
    if "moving slowly" in rmk or "moving slow" in rmk:
        return EtaKind.MOVING_SLOWLY
    if "delayed" in rmk:
        return EtaKind.DELAYED
    return EtaKind.REALTIME


def _ctb_kind(rmk: str) -> EtaKind:
    """Citybus exposes no scheduled/delay remarks; KMB-cycle entries are real."""
    return EtaKind.REALTIME


def _gmb_kind(remarks: str | None) -> EtaKind:
    remarks = (remarks or "").lower()
    if "scheduled" in remarks:
        return EtaKind.SCHEDULED
    if "delayed" in remarks:
        return EtaKind.DELAYED
    return EtaKind.REALTIME


# --------------------------------------------------------------------------
# Fetchers
# --------------------------------------------------------------------------


async def _fetch_many_json(
    client: HttpClient, urls: list[str], source: str
) -> dict[str, dict[str, Any]]:
    """Fetch each distinct endpoint independently so one stop cannot hide an operator."""
    unique_urls = list(dict.fromkeys(urls))
    results = await asyncio.gather(
        *(client.fetch_json(url) for url in unique_urls),
        return_exceptions=True,
    )
    fetched: dict[str, dict[str, Any]] = {}
    failures: list[Exception] = []
    for url, result in zip(unique_urls, results, strict=True):
        if isinstance(result, Exception):
            failures.append(result)
            log.warning("%s ETA request failed for %s: %s", source, url, result)
        elif isinstance(result, dict):
            fetched[url] = result
    if not fetched and failures:
        raise failures[0]
    return fetched


async def _fetch_kmb(client: HttpClient, now: datetime) -> list[EtaRow]:
    rows: list[EtaRow] = []
    urls = [KMB_BASE.format(stop=spec["stop"]) for spec in KMB_STOPS]
    responses = await _fetch_many_json(client, urls, "KMB")
    for spec in KMB_STOPS:
        data = responses.get(KMB_BASE.format(stop=spec["stop"]), {})
        entries = (data or {}).get("data", []) or []
        for entry in entries:
            if entry.get("route") != spec["route"]:
                continue
            if entry.get("service_type", 1) != 1:
                continue
            eta_iso = entry.get("eta")
            minutes = _minutes_until(eta_iso, now)
            if minutes is None and eta_iso:
                continue  # unparseable timestamp; skip rather than crash
            bound = _KMB_DIR_TO_BOUND.get(
                str(entry.get("dir") or "")[:1].lower()
            )
            rows.append(
                EtaRow(
                    route=spec["route"],
                    destination=spec["dest"],
                    gate=spec["gate"],
                    operator=Operator.KMB,
                    minutes=minutes,
                    kind=_kmb_kind(entry.get("rmk_en") or ""),
                    eta_time=_parse_iso(eta_iso),
                    source_time=_parse_iso(entry.get("data_timestamp")),
                    bound=bound,
                )
            )
    return rows


_CTB_DIR_TO_BOUND = {"O": "outbound", "I": "inbound"}


async def _fetch_citybus(client: HttpClient, now: datetime) -> list[EtaRow]:
    rows: list[EtaRow] = []
    urls = [CTB_BASE.format(stop=spec["stop"], route=spec["route"]) for spec in CTB_STOPS]
    responses = await _fetch_many_json(client, urls, "Citybus")
    for spec in CTB_STOPS:
        data = responses.get(CTB_BASE.format(stop=spec["stop"], route=spec["route"]), {})
        entries = (data or {}).get("data", []) or []
        for entry in entries:
            if entry.get("dir") != spec["gate"]:
                continue
            eta_iso = entry.get("eta")
            if not eta_iso:
                continue  # empty eta (e.g. KMB Cycle) -> no departure
            minutes = _minutes_until(eta_iso, now)
            if minutes is None:
                continue
            rows.append(
                EtaRow(
                    route=spec["route"],
                    destination=spec["dest"],
                    gate="N",  # both 792M directions load at North Gate
                    operator=Operator.CITYBUS,
                    minutes=minutes,
                    kind=_ctb_kind(entry.get("rmk_en") or ""),
                    eta_time=_parse_iso(eta_iso),
                    source_time=_parse_iso(entry.get("data_timestamp")),
                    bound=_CTB_DIR_TO_BOUND.get(str(entry.get("dir") or "")),
                )
            )
    return rows


async def _fetch_gmb(client: HttpClient, now: datetime) -> list[EtaRow]:
    if time.monotonic() < _gmb_cooldown_until:
        log.info("GMB gate polling paused during rate-limit cooldown")
        return _gmb_gate_cache.get()
    rows: list[EtaRow] = []
    urls = [GMB_BASE.format(stop=stop_id) for stop_id in GMB_STOPS]
    unique_urls = list(dict.fromkeys(urls))
    responses: dict[str, dict[str, Any]] = {}
    failures: list[Exception] = []
    for url in unique_urls:
        if time.monotonic() < _gmb_cooldown_until:
            log.info("GMB gate polling paused during rate-limit cooldown")
            return _gmb_gate_cache.get()
        try:
            result = await client.fetch_json(url)
        except FetchError as exc:
            if exc.status_code == 403:
                _record_gmb_403()
                return _gmb_gate_cache.get()
            failures.append(exc)
            log.warning("GMB ETA request failed for %s: %s", url, exc)
            continue
        except Exception as exc:  # noqa: BLE001
            failures.append(exc)
            log.warning("GMB ETA request failed for %s: %s", url, exc)
            continue
        if isinstance(result, dict):
            responses[url] = result
        else:
            failures.append(TypeError("GMB ETA response was not an object"))
    if not responses and failures:
        raise failures[0]
    for stop_id, routes in GMB_STOPS.items():
        data = responses.get(GMB_BASE.format(stop=stop_id), {})
        entries = (data or {}).get("data", []) or []
        # live schema (verified 2026-08-07):
        #   {"route_id": 2004828, "route_seq": 1, "stop_seq": 1,
        #    "enabled": true, "eta": [{"eta_seq":1, "diff":34,
        #    "timestamp":"...", "remarks_en":"Scheduled"}]}
        by_route: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
        for entry in entries:
            if not entry.get("enabled", True):
                continue
            key = (entry.get("route_id"), entry.get("route_seq"), entry.get("stop_seq"))
            by_route.setdefault(key, []).append(entry)
        for route_no, dest, gate, route_id, seq in routes:
            # For circular routes the same route/seq can appear at multiple
            # stop_seq (the bus looping back). Keep only the FIRST stop_seq —
            # the departure stop — so returning arrivals are never shown.
            matching = [
                (stop_seq, group)
                for (rid, rseq, stop_seq), group in by_route.items()
                if rid == route_id and rseq == seq
            ]
            if not matching:
                continue
            group = sorted(matching)[0][1]
            for entry in group:
                eta_list = entry.get("eta") or []
                # For CIRCULAR routes the ETA list may include both
                # departures and the loop's return arrivals. ETAs are
                # time-ordered; keep only the monotonically increasing prefix
                # (a drop like 51 -> 46 is the bus coming back around, not a
                # new departure).
                last = -1
                for eta in eta_list:
                    diff = eta.get("diff")
                    try:
                        minutes = int(diff) if diff is not None else None
                    except (TypeError, ValueError):
                        minutes = None
                    if minutes is None:
                        continue
                    if minutes < last:
                        continue  # loop return arrival — drop it
                    last = minutes
                    rows.append(
                        EtaRow(
                            route=route_no,
                            destination=dest,
                            gate=gate,
                            operator=Operator.GMB,
                            minutes=minutes,
                            kind=_gmb_kind(eta.get("remarks_en") or ""),
                            eta_time=_parse_iso(eta.get("timestamp")),
                            source_time=_parse_iso((data or {}).get("generated_timestamp")),
                            stop_seq=entry.get("stop_seq"),
                            bound=f"seq-{seq}",
                        )
                    )
    if not failures:
        _gmb_gate_cache.set(rows)
    return rows


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def _route_sort_key(route: str) -> tuple[int, str]:
    """Order routes numerically (91 < 291P, 11 < 11B < 12 < 104) so the bus
    stop list follows human intuition instead of string sorting."""
    digits = "".join(ch for ch in route if ch.isdigit())
    letters = "".join(ch for ch in route if not ch.isdigit())
    return (int(digits) if digits else 9999, letters)


def group_etas(rows: list[EtaRow]) -> list[RouteEtaGroup]:
    """Group ETAs into a human-friendly order: North gate first, then South;
    within each gate buses (KMB, Citybus) before minibuses, and routes ordered
    numerically (91, 91M, 291P / 11, 11B, 12, 104) within each operator."""
    operator_order = {Operator.KMB: 0, Operator.CITYBUS: 1, Operator.GMB: 2}

    def sort_key(row: EtaRow) -> tuple:
        return (
            row.gate,  # N before S
            operator_order[row.operator],  # buses before minibuses
            _route_sort_key(row.route),
            row.destination,
        )

    ordered = sorted(rows, key=sort_key)
    groups: list[RouteEtaGroup] = []
    current: RouteEtaGroup | None = None
    for row in ordered:
        if current is None or (
            current.route,
            current.destination,
            current.gate,
            current.operator,
            current.stop_seq,
            current.bound,
        ) != (
            row.route,
            row.destination,
            row.gate,
            row.operator,
            row.stop_seq,
            row.bound,
        ):
            current = RouteEtaGroup(
                route=row.route,
                destination=row.destination,
                gate=row.gate,
                operator=row.operator,
                stop_seq=row.stop_seq,
                bound=row.bound,
            )
            groups.append(current)
        current.rows.append(row)
    return groups


async def fetch_transit_etas(
    client: HttpClient,
) -> tuple[list[RouteEtaGroup], datetime | None, list[str]]:
    """Fetch all three operators concurrently and return grouped, ordered ETAs.

    Returns (groups, latest_source_time, failed_operators). A single operator
    failing does not discard the others; failed operator names are returned so
    the renderer can surface an error.
    """
    global _gate_network_refresh_at, _gate_refresh_task, _gate_refresh_waiters
    now_mono = time.monotonic()
    cached = _gate_eta_cache.get()
    if (cached is not None and _gate_network_refresh_at is not None
            and now_mono - _gate_network_refresh_at < TRANSIT_NETWORK_REFRESH_SECONDS):
        rows, latest, failed = cached
        return group_etas(rows), latest, failed

    if _gate_refresh_task is None or _gate_refresh_task.done():
        # Claim the refresh slot and publish the task before awaiting it. A
        # concurrent cold-start caller joins this task instead of seeing an
        # artificial empty response or launching another operator sweep.
        _gate_network_refresh_at = now_mono
        _gate_refresh_task = asyncio.create_task(_refresh_gate_etas(client))
    task = _gate_refresh_task
    _gate_refresh_waiters += 1
    waiter_registered = True
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        _gate_refresh_waiters -= 1
        waiter_registered = False
        if not task.done() and _gate_refresh_waiters == 0:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            _gate_network_refresh_at = None
        raise
    finally:
        if waiter_registered and _gate_refresh_waiters > 0:
            _gate_refresh_waiters -= 1
        if task.done() and _gate_refresh_task is task:
            _gate_refresh_task = None


async def _refresh_gate_etas(
    client: HttpClient,
) -> tuple[list[RouteEtaGroup], datetime | None, list[str]]:
    """Perform one uncached gate sweep; callers coordinate through the task."""
    now = datetime.now(UTC)
    results = await client.gather_any(
        [_fetch_kmb(client, now), _fetch_citybus(client, now), _fetch_gmb(client, now)]
    )
    operator_names = ["KMB", "Citybus", "GMB"]
    rows: list[EtaRow] = []
    latest: datetime | None = None
    failed: list[str] = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            log.warning("transit operator fetch failed: %s", result)
            failed.append(operator_names[i])
            continue
        rows.extend(result)
        for row in result:
            if row.source_time and (latest is None or row.source_time > latest):
                latest = row.source_time
    _gate_eta_cache.set(rows, latest, failed)
    return group_etas(rows), latest, failed


# --------------------------------------------------------------------------
# Probe-stop ETAs (downstream bus-position estimates)
# --------------------------------------------------------------------------

# Probe cache holds the last value per fetch group. The ceiling is deliberately
# longer than a complete sweep so early and late groups coexist, while still
# removing departed vehicles during a prolonged provider outage.
PROBE_TTL_SECONDS = 900.0
PROBE_GENERATION_TTL_SECONDS = 900.0
MAX_PROBE_ROUTE_GENERATIONS = 128
MAX_PROBE_CACHE_ENTRIES = 1024
def _wall_clock_now() -> datetime:
    return datetime.now(UTC)


_probe_wall_clock: Callable[[], datetime] = _wall_clock_now
_probe_mono_clock: Callable[[], float] = time.monotonic


@dataclass(frozen=True)
class ProbeEta:
    """One ETA observation at a probe stop along a tracked direction."""

    operator: str
    route: str
    bound: str
    stop_id: str
    index: int  # official stop-sequence index of the probe stop
    minutes: float | None
    kind: EtaKind = EtaKind.REALTIME
    # Age of this stop's last successful response at collection time. Source
    # minutes remain immutable; this metadata aligns identities across
    # staggered refreshes and controls whether a boundary may move a marker.
    cache_age_seconds: float = 0.0
    arrival_at: datetime | None = None
    observed_at: datetime | None = None
    refresh_generation: int = 0


@dataclass(frozen=True)
class ProbeRouteGeneration:
    """One atomically published, complete route observation."""

    route_key: tuple[str, str, str]
    rows: tuple[ProbeEta, ...]
    generation: int
    collected_at: datetime
    observed_checkpoint_indices: frozenset[int] = frozenset()

    def __post_init__(self) -> None:
        if not self.observed_checkpoint_indices:
            object.__setattr__(
                self,
                "observed_checkpoint_indices",
                frozenset(
                    int(row.index) for row in self.rows if hasattr(row, "index")
                ),
            )


@dataclass(frozen=True)
class _StoredProbeGeneration:
    """Private publication record, including validation and expiry metadata."""

    public: ProbeRouteGeneration
    topology_keys: frozenset[tuple[Any, ...]]
    group_keys: frozenset[str]
    published_monotonic: float


@dataclass(frozen=True)
class ProbeEtaSnapshot:
    """Complete route generations visible at one collection time."""

    routes: tuple[ProbeRouteGeneration, ...]
    collected_at: datetime
    positioning_rows: tuple[ProbeEta, ...] | None = None
    positioning_checkpoints: frozenset[tuple[str, str, str, int]] = frozenset()

    @property
    def rows(self) -> tuple[ProbeEta, ...]:
        if self.positioning_rows is not None:
            return self.positioning_rows
        return tuple(row for route in self.routes for row in route.rows)

    @property
    def complete_routes(self) -> tuple[ProbeRouteGeneration, ...]:
        return self.routes


def _probe_cache_key(probe) -> str:
    key = ":".join(str(value) for value in _probe_topology_key(probe)[:-1])
    return key


def _probe_topology_key(probe) -> tuple[Any, ...]:
    """Canonical, order-independent identity of one request-shaped probe."""
    return (
        str(getattr(probe, "operator", "")),
        str(getattr(probe, "route", "")),
        str(getattr(probe, "bound", "")),
        str(getattr(probe, "stop_id", "")),
        int(getattr(probe, "index", 0)),
        str(getattr(probe, "route_id", "")),
        str(getattr(probe, "sequence", "")),
        _fetch_group_key(probe),
    )


def _fetch_group_key(probe) -> str:
    """Group probes by the single HTTP request that serves them.

    KMB is fetched once per route (both directions), GMB once per physical
    stop (all routes), and Citybus once per (stop, route). One fetch per group
    covers every matching probe in it.
    """
    if probe.operator == "KMB":
        return f"KMB:route:{probe.route}"
    if probe.operator == "CTB":
        return f"CTB:{probe.stop_id}:{probe.route}"
    return f"{probe.operator}:{probe.stop_id}"


class ProbeEtaCache:
    """Immutable last-value cache between staggered all-stop sweep steps.

    The TTL spans normal route sweeps, preventing mid-generation splits, but
    bounds staleness when repeated failures leave an entry untouched.
    """

    def __init__(
        self,
        ttl_seconds: float = PROBE_TTL_SECONDS,
        *,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], datetime] | None = None,
        max_entries: int = MAX_PROBE_CACHE_ENTRIES,
    ) -> None:
        self._ttl = ttl_seconds
        self._clock = clock
        self._wall_clock = wall_clock or _probe_wall_clock
        self._max_entries = max(1, int(max_entries))
        self._store: dict[str, tuple[float, list[ProbeEta]]] = {}

    def _sweep(self, now: float) -> None:
        expired = [
            key for key, (stored, _rows) in self._store.items()
            if self._ttl > 0 and now - stored >= self._ttl
        ]
        for key in expired:
            self._store.pop(key, None)
        while len(self._store) > self._max_entries:
            oldest = min(self._store, key=lambda key: (self._store[key][0], key))
            self._store.pop(oldest, None)

    def get(self, key: str) -> list[ProbeEta] | None:
        now_mono = self._clock()
        self._sweep(now_mono)
        entry = self._store.get(key)
        if entry is None:
            return None
        age_seconds = max(0.0, now_mono - entry[0])
        # Cached source values are immutable.  Consumers use this age to
        # decide whether a boundary is fresh enough to move a marker.
        return [replace(eta, cache_age_seconds=age_seconds) for eta in entry[1]]

    def age_seconds(self, key: str) -> float | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        return max(0.0, self._clock() - entry[0])

    def set(self, key: str, value: list[ProbeEta]) -> None:
        now = self._clock()
        self._sweep(now)
        self._store[key] = (now, list(value))
        self._sweep(now)


_probe_cache = ProbeEtaCache()


_probe_priority_cursor = 0
_probe_background_cursor = 0
_probe_generation = 0
_probe_route_generations: dict[tuple[str, str, str], _StoredProbeGeneration] = {}
_probe_group_versions: dict[str, int] = {}
_probe_group_rows: dict[str, tuple[ProbeEta, ...]] = {}
_probe_route_published_versions: dict[tuple[str, str, str], dict[str, int]] = {}

# data.etagmb.gov.hk rate-limits bursts with 403s. GMB groups are capped per
# cycle and the whole GMB sweep backs off for a cooldown when a 403 appears;
# the probe cache keeps last-good values meanwhile. A 403 also steps the
# process-local batch limit down, with no automatic recovery until restart.
_GMB_BATCH_STEP = 3
_GMB_BATCH_FLOOR = 5


def _record_gmb_403() -> None:
    """Enter the shared cooldown and reduce the next probe batch once.

    This is deliberately synchronous: the gate and probe fetchers run on the
    same event loop, so the first 403 sets the episode marker before another
    in-flight request can handle its own 403.
    """
    global GMB_GROUPS_PER_CYCLE, _gmb_cooldown_until
    now = time.monotonic()
    if now < _gmb_cooldown_until:
        return
    old_limit = GMB_GROUPS_PER_CYCLE
    new_limit = max(_GMB_BATCH_FLOOR, old_limit - _GMB_BATCH_STEP)
    GMB_GROUPS_PER_CYCLE = new_limit
    _gmb_cooldown_until = now + GMB_COOLDOWN_SECONDS
    log.warning(
        "GMB ETA rate-limited (403); reducing probe batch limit %d -> %d "
        "and pausing gate and probe polling for %.0fs",
        old_limit,
        new_limit,
        GMB_COOLDOWN_SECONDS,
    )


async def _fetch_raw_stop_eta(client: HttpClient, probe) -> Any:
    """Fetch the raw ETA payload covering ``probe`` (one request per group)."""
    if probe.operator == "KMB":
        return await client.fetch_json(KMB_ROUTE_ETA_BASE.format(route=probe.route))
    if probe.operator == "CTB":
        return await client.fetch_json(
            CTB_BASE.format(stop=probe.stop_id, route=probe.route)
        )
    return await client.fetch_json(GMB_BASE.format(stop=probe.stop_id))


def _parse_probe_etas(probe, raw: Any, now: datetime) -> list[ProbeEta]:
    """Filter a raw stop payload down to this probe's route/direction."""
    entries = (raw or {}).get("data", []) or []
    out: list[ProbeEta] = []
    _CTB_BOUND_DIRS = {"outbound": "O", "inbound": "I"}
    if probe.operator == "KMB":
        for entry in entries:
            entry_stop = entry.get("stop") or entry.get("stop_id")
            if entry_stop is not None:
                if str(entry_stop) != str(probe.stop_id):
                    continue
            else:
                # The route-wide endpoint identifies the official occurrence
                # by one-based sequence, not by stop ID.
                try:
                    if int(entry.get("seq")) != int(probe.index) + 1:
                        continue
                except (TypeError, ValueError):
                    continue
            if entry.get("route") != probe.route:
                continue
            if entry.get("service_type", 1) != 1:
                continue
            # KMB marks bounds "o"/"i"; compare case-insensitively — the API
            # returns uppercase letters.
            entry_dir = str(entry.get("dir") or "")[:1].lower()
            if entry_dir and entry_dir != probe.bound[:1].lower():
                continue
            minutes = _precise_minutes_until(entry.get("eta"), now)
            if minutes is None:
                continue
            out.append(
                ProbeEta(
                    operator=probe.operator,
                    route=probe.route,
                    bound=probe.bound,
                    stop_id=probe.stop_id,
                    index=probe.index,
                    minutes=minutes,
                    kind=_kmb_kind(entry.get("rmk_en") or ""),
                    arrival_at=_parse_iso(entry.get("eta"))
                    or now + timedelta(minutes=minutes),
                    observed_at=_parse_iso(entry.get("data_timestamp")) or now,
                )
            )
        return out
    if probe.operator == "CTB":
        expected_dir = _CTB_BOUND_DIRS.get(probe.bound, "")
        for entry in entries:
            # Both 792M directions share stops; the dir letter keeps this
            # probe's estimates on their own direction.
            if expected_dir and entry.get("dir") != expected_dir:
                continue
            eta_iso = entry.get("eta")
            if not eta_iso:
                continue
            minutes = _precise_minutes_until(eta_iso, now)
            if minutes is None:
                continue
            out.append(
                ProbeEta(
                    operator=probe.operator,
                    route=probe.route,
                    bound=probe.bound,
                    stop_id=probe.stop_id,
                    index=probe.index,
                    minutes=minutes,
                    kind=_ctb_kind(entry.get("rmk_en") or ""),
                    arrival_at=_parse_iso(eta_iso)
                    or now + timedelta(minutes=minutes),
                    observed_at=_parse_iso(entry.get("data_timestamp")) or now,
                )
            )
        return out
    for entry in entries:
        if not entry.get("enabled", True):
            continue
        if entry.get("route_id") != probe.route_id:
            continue
        if entry.get("route_seq") != probe.sequence:
            continue
        # A physical stop can occur more than once on a circular route.  An
        # ETA entry belongs only to its official occurrence; treating every
        # matching route entry as this probe creates phantom ladders.
        entry_stop_seq = entry.get("stop_seq")
        if entry_stop_seq is not None:
            try:
                if int(entry_stop_seq) != int(probe.index) + 1:
                    continue
            except (TypeError, ValueError):
                continue
        last = -1
        for eta in entry.get("eta") or []:
            # ``route_seq`` above identifies the route variation/direction;
            # ``eta_seq`` only orders consecutive minibuses within this exact
            # route-stop response.  Keep equal timestamps as distinct rows:
            # without a public vehicle ID, a genuinely bunched pair cannot be
            # distinguished safely from a duplicated provider prediction.
            # GMB supplies an exact arrival timestamp alongside rounded
            # integer `diff`.  The timestamp keeps staggered stop probes on a
            # common clock and reduces false ladder splits at minute edges.
            minutes = _precise_minutes_until(eta.get("timestamp"), now)
            if minutes is None:
                diff = eta.get("diff")
                try:
                    minutes = float(diff) if diff is not None else None
                except (TypeError, ValueError):
                    minutes = None
            if minutes is None or minutes < last:
                continue
            last = minutes
            out.append(
                ProbeEta(
                    operator=probe.operator,
                    route=probe.route,
                    bound=probe.bound,
                    stop_id=probe.stop_id,
                    index=probe.index,
                    minutes=minutes,
                    kind=_gmb_kind(eta.get("remarks_en") or ""),
                    arrival_at=_parse_iso(eta.get("timestamp")),
                    observed_at=_parse_iso((raw or {}).get("generated_timestamp")) or now,
                )
            )
    return out


async def fetch_probe_etas(
    client: HttpClient,
    probes: Sequence[Any],
    max_per_cycle: int = 36,
    priorities=None,
) -> list[ProbeEta]:
    """Return cached probe observations, refreshing the network at most every 30s."""
    global _probe_network_refresh_at, _probe_refresh_task, _probe_refresh_waiters
    if not probes:
        return []
    now_mono = time.monotonic()
    if ((_probe_refresh_task is None or _probe_refresh_task.done())
            and _probe_network_refresh_at is not None
            and now_mono - _probe_network_refresh_at < TRANSIT_NETWORK_REFRESH_SECONDS):
        return _collect_probe_cache(probes)
    if _probe_refresh_task is None or _probe_refresh_task.done():
        _probe_network_refresh_at = now_mono
        _probe_refresh_task = asyncio.create_task(
            _refresh_probe_etas(client, probes, max_per_cycle, priorities)
        )
    task = _probe_refresh_task
    _probe_refresh_waiters += 1
    waiter_registered = True
    try:
        await asyncio.shield(task)
        # The shared refresh is parameterized by its first caller's probe
        # subset. Re-collect from this caller's keys after it completes so a
        # concurrent caller never receives another caller's marker list.
        return _collect_probe_cache(probes)
    except asyncio.CancelledError:
        _probe_refresh_waiters -= 1
        waiter_registered = False
        if not task.done() and _probe_refresh_waiters == 0:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            _probe_network_refresh_at = None
        raise
    finally:
        if waiter_registered and _probe_refresh_waiters > 0:
            _probe_refresh_waiters -= 1
        if task.done() and _probe_refresh_task is task:
            _probe_refresh_task = None


def _collect_probe_cache(probes: Sequence[Any]) -> list[ProbeEta]:
    unique: dict[str, Any] = {}
    for probe in probes:
        unique.setdefault(_probe_cache_key(probe), probe)
    return [
        eta
        for key in unique
        if (cached := _probe_cache.get(key)) is not None
        for eta in cached
    ]


async def _refresh_probe_etas(
    client: HttpClient,
    probes: Sequence[Any],
    max_per_cycle: int = 36,
    priorities=None,
) -> list[ProbeEta]:
    """Poll up to ``max_per_cycle`` fetch groups round-robin; cache results.

    Probes sharing one HTTP request (same KMB route, same physical GMB stop,
    or same Citybus stop+route) are fetched ONCE per sweep step and parsed once
    per probe, so dense probing stays cheap. Cached values are returned for
    probes outside this cycle's window. GMB groups are additionally capped
    per cycle and the GMB sweep backs off for a cooldown when the host
    rate-limits (HTTP 403), serving last-good cache meanwhile.
    """
    global _probe_priority_cursor, _probe_background_cursor, _probe_generation
    if not probes:
        return []
    unique: dict[str, Any] = {}
    for probe in probes:
        unique.setdefault(_probe_cache_key(probe), probe)

    # Fetch-group table: one raw request serves every probe in its bucket.
    groups: dict[str, list[Any]] = {}
    for key in sorted(unique):
        probe = unique[key]
        groups.setdefault(_fetch_group_key(probe), []).append(probe)

    group_keys = sorted(groups)
    active_groups = set(group_keys)
    for stale_group in set(_probe_group_versions) - active_groups:
        _probe_group_versions.pop(stale_group, None)
        _probe_group_rows.pop(stale_group, None)
    routes: dict[tuple[str, str, str], list[str]] = {}
    for key, route_probes in groups.items():
        for probe in route_probes:
            route_key = (probe.operator, probe.route, probe.bound)
            route_list = routes.setdefault(route_key, [])
            if key not in route_list:
                route_list.append(key)
    for route_key, published_groups in tuple(_probe_route_published_versions.items()):
        current_groups = set(routes.get(route_key, ()))
        if current_groups != set(published_groups):
            _probe_route_published_versions.pop(route_key, None)
            _probe_route_generations.pop(route_key, None)
    gmb_paused = time.monotonic() < _gmb_cooldown_until

    def selectable(group_key: str) -> bool:
        if not group_key.startswith("GMB:"):
            return True
        return not gmb_paused

    selectable_keys = [key for key in group_keys if selectable(key)]
    gmb_count = len(group_keys) - len(selectable_keys)
    if gmb_count and gmb_paused:
        log.info(
            "GMB probe sweep paused for %.0fs (rate-limit cooldown); "
            "serving last-good cache",
            _gmb_cooldown_until - time.monotonic(),
        )
    cycle_budget = max(0, max_per_cycle)
    # Stage individual fetch groups in a bounded round-robin. Publications are
    # assembled below only after every group has advanced since publication.
    selected_groups: list[str] = []
    selected_priority: list[str] = []
    selected_background: list[str] = []
    used_gmb = 0
    priority_groups = []
    priority_indices = priorities or {}
    for group_key, route_probes in groups.items():
        is_priority = False
        for probe in route_probes:
            wanted = {int(index) for index in priority_indices.get(
                (probe.operator, probe.route, probe.bound), ())
                      if isinstance(index, int) or str(index).isdigit()}
            if int(getattr(probe, "index", -1)) in wanted:
                is_priority = True
                break
        if is_priority:
            priority_groups.append(group_key)
    background_groups = [key for key in group_keys if key not in priority_groups]
    priority_start = _probe_priority_cursor % len(priority_groups) if priority_groups else 0
    background_start = _probe_background_cursor % len(background_groups) if background_groups else 0
    reserve = min(len(background_groups), max(1, cycle_budget // 4)) if cycle_budget else 0
    gmb_background_reserve = min(
        sum(key.startswith("GMB:") for key in background_groups),
        max(1, GMB_GROUPS_PER_CYCLE // 4),
    )
    priority_gmb_limit = max(0, GMB_GROUPS_PER_CYCLE - gmb_background_reserve)
    for key in (priority_groups[priority_start:] + priority_groups[:priority_start]):
        if key not in selectable_keys or len(selected_priority) >= max(0, cycle_budget - reserve):
            continue
        if key.startswith("GMB:") and used_gmb >= priority_gmb_limit:
            continue
        selected_priority.append(key)
        used_gmb += int(key.startswith("GMB:"))
    for key in (background_groups[background_start:] + background_groups[:background_start]):
        if key not in selectable_keys or len(selected_background) >= reserve:
            continue
        if key.startswith("GMB:") and used_gmb >= GMB_GROUPS_PER_CYCLE:
            continue
        selected_background.append(key)
        used_gmb += int(key.startswith("GMB:"))
    remaining = cycle_budget - len(selected_priority) - len(selected_background)
    for key in (background_groups[background_start:] + background_groups[:background_start]):
        if key in selected_background or key not in selectable_keys or remaining <= 0:
            continue
        if key.startswith("GMB:") and used_gmb >= GMB_GROUPS_PER_CYCLE:
            continue
        selected_background.append(key)
        used_gmb += int(key.startswith("GMB:"))
        remaining -= 1
    selected_groups = selected_priority + selected_background
    if priority_groups:
        _probe_priority_cursor = (priority_start + len(selected_priority)) % len(priority_groups)
    if background_groups:
        _probe_background_cursor = (background_start + len(selected_background)) % len(background_groups)
    if not selected_groups:
        return [
            eta
            for key in unique
            if (cached := _probe_cache.get(key)) is not None
            for eta in cached
        ]

    selected_other = [key for key in selected_groups if not key.startswith("GMB:")]
    selected_gmb = [key for key in selected_groups if key.startswith("GMB:")]
    successful: dict[str, bool] = {}
    refreshed_rows: dict[str, tuple[ProbeEta, ...]] = {}
    async def refresh_group(group_key: str) -> bool:
        global _gmb_cooldown_until
        if group_key.startswith("GMB:") and time.monotonic() < _gmb_cooldown_until:
            successful[group_key] = False
            return True
        probes_in_group = groups[group_key]
        now = _probe_wall_clock()
        if group_key.startswith("GMB:") and time.monotonic() < _gmb_cooldown_until:
            return True
        try:
            raw = await _fetch_raw_stop_eta(client, probes_in_group[0])
        except FetchError as exc:
            if exc.status_code == 403 and group_key.startswith("GMB:"):
                _record_gmb_403()
                successful[group_key] = False
                return True
            else:
                log.warning(
                    "probe ETA fetch failed for %s: %s",
                    group_key, type(exc).__name__,
                )
                successful[group_key] = False
            return False
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "probe ETA fetch failed for %s: %s", group_key, type(exc).__name__
            )
            successful[group_key] = False
            return False
        global _probe_generation
        _probe_generation += 1
        group_generation = _probe_generation
        group_rows: list[ProbeEta] = []
        for probe in probes_in_group:
            key = _probe_cache_key(probe)
            parsed = [
                replace(row, refresh_generation=group_generation)
                for row in _parse_probe_etas(probe, raw, now)
            ]
            _probe_cache.set(key, parsed)
            refreshed_rows[key] = tuple(parsed)
            group_rows.extend(parsed)
        _probe_group_versions[group_key] = group_generation
        _probe_group_rows[group_key] = tuple(group_rows)
        successful[group_key] = True
        return False

    # Non-GMB origins can overlap. GMB requests are deliberately sequential:
    # once one returns 403, stop this sweep instead of draining an already
    # queued burst into the provider during its cooldown.
    await asyncio.gather(*(refresh_group(group_key) for group_key in selected_other))
    for group_key in selected_gmb:
        if await refresh_group(group_key):
            break

    # Publish complete route generations atomically. Empty successful entries
    # are valid observations; failed or rate-limited groups retain the prior
    # complete generation.
    collected_at = _probe_wall_clock()
    published_monotonic = _probe_mono_clock()
    selected_routes = sorted({route_key for key in selected_groups for route_key in routes
                              if key in routes[route_key]})
    for route_key in selected_routes:
        route_groups = routes[route_key]
        if not all(key in _probe_group_versions for key in route_groups):
            continue
        previous = _probe_route_published_versions.get(route_key, {})
        if previous and not all(_probe_group_versions[key] > previous.get(key, 0)
                                for key in route_groups):
            continue
        route_probes = tuple(
            probe for probe in unique.values()
            if (probe.operator, probe.route, probe.bound) == route_key
        )
        rows = tuple(eta for probe in route_probes
                     for eta in (_probe_cache.get(_probe_cache_key(probe)) or ()))
        _probe_generation += 1
        public = ProbeRouteGeneration(
            route_key=route_key,
            rows=rows,
            generation=_probe_generation,
            collected_at=collected_at,
            observed_checkpoint_indices=frozenset(p.index for p in route_probes),
        )
        _probe_route_published_versions[route_key] = {
            key: _probe_group_versions[key] for key in route_groups
        }
        _probe_route_generations[route_key] = _StoredProbeGeneration(
            public=public,
            topology_keys=frozenset(_probe_topology_key(probe) for probe in route_probes),
            group_keys=frozenset(route_groups),
            published_monotonic=published_monotonic,
        )
    while len(_probe_route_generations) > MAX_PROBE_ROUTE_GENERATIONS:
        oldest = min(
            _probe_route_generations,
            key=lambda key: _probe_route_generations[key].published_monotonic,
        )
        _probe_route_generations.pop(oldest, None)

    collected: list[ProbeEta] = []
    for key in unique:
        cached = _probe_cache.get(key)
        if cached is not None:
            collected.extend(cached)
    return collected


async def fetch_probe_snapshot(
    client: HttpClient,
    probes: Sequence[Any],
    max_per_cycle: int = 36,
    priorities=None,
) -> ProbeEtaSnapshot:
    """Fetch probes and expose only atomically complete route generations."""
    await fetch_probe_etas(client, probes, max_per_cycle=max_per_cycle,
                           priorities=priorities)
    collected_at = _probe_wall_clock()
    monotonic_at = _probe_mono_clock()
    for stale_key, stale_generation in tuple(_probe_route_generations.items()):
        if (
            PROBE_GENERATION_TTL_SECONDS > 0
            and monotonic_at - stale_generation.published_monotonic
            >= PROBE_GENERATION_TTL_SECONDS
        ):
            _probe_route_generations.pop(stale_key, None)
    requested = {(p.operator, p.route, p.bound) for p in probes}
    requested_topology: dict[
        tuple[str, str, str], tuple[frozenset[tuple[Any, ...]], frozenset[str]]
    ] = {}
    for route_key in requested:
        route_probes = [p for p in probes if (p.operator, p.route, p.bound) == route_key]
        requested_topology[route_key] = (
            frozenset(_probe_topology_key(p) for p in route_probes),
            frozenset(_fetch_group_key(p) for p in route_probes),
        )
    routes: list[ProbeRouteGeneration] = []
    for key in sorted(requested):
        generation = _probe_route_generations.get(key)
        if generation is None:
            continue
        topology_keys, group_keys = requested_topology[key]
        if generation.topology_keys != topology_keys or generation.group_keys != group_keys:
            continue
        if (
            PROBE_GENERATION_TTL_SECONDS > 0
            and monotonic_at - generation.published_monotonic
            >= PROBE_GENERATION_TTL_SECONDS
        ):
            _probe_route_generations.pop(key, None)
            continue
        aged_rows = tuple(
            replace(
                row,
                minutes=row.minutes,
                cache_age_seconds=(
                    float(getattr(row, "cache_age_seconds", 0.0) or 0.0)
                    + max(0.0, (collected_at - generation.public.collected_at).total_seconds())
                ),
            )
            for row in generation.public.rows
        )
        routes.append(replace(generation.public, rows=aged_rows))
    positioning: list[ProbeEta] = []
    positioning_checkpoints: set[tuple[str, str, str, int]] = set()
    for probe in probes:
        key = _probe_cache_key(probe)
        cached = _probe_cache.get(key)
        if cached is None:
            continue
        checkpoint = (str(probe.operator), str(probe.route), str(probe.bound), int(probe.index))
        positioning_checkpoints.add(checkpoint)
        if cached:
            positioning.extend(cached)
        else:
            positioning.append(ProbeEta(
                operator=str(probe.operator), route=str(probe.route),
                bound=str(probe.bound), stop_id=str(probe.stop_id),
                index=int(probe.index), minutes=None,
                cache_age_seconds=float(_probe_cache.age_seconds(key) or 0.0),
            ))
    return ProbeEtaSnapshot(
        routes=tuple(routes), collected_at=collected_at,
        positioning_rows=tuple(positioning),
        positioning_checkpoints=frozenset(positioning_checkpoints),
    )
