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
    """Last-good gate ETAs, aged while the shared GMB origin is cooling down."""

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
        elapsed_minutes = int(age_seconds // 60.0)
        aged: list[EtaRow] = []
        for row in rows:
            if row.minutes is None:
                aged.append(row)
            elif row.minutes - elapsed_minutes >= 0:
                aged.append(replace(row, minutes=row.minutes - elapsed_minutes))
        return aged


_gmb_gate_cache = _GmbGateCache()

# ETA network refreshes are deliberately slower than the map/render cadence.
# The dashboard may render every ten seconds, but polling the estimator feeds
# every render tick causes avoidable bursts (especially against GMB). Cached
# observations are aged and served between refreshes.
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
        elapsed_minutes = int(max(0.0, time.monotonic() - stamped) // 60.0)
        aged: list[EtaRow] = []
        for row in rows:
            if row.minutes is None:
                aged.append(row)
            elif row.minutes - elapsed_minutes >= 0:
                aged.append(replace(row, minutes=row.minutes - elapsed_minutes))
        return aged, latest, list(failed)


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
    # Age of this stop's last successful response at collection time.  ETA
    # minutes are already aged to the same clock; this metadata only resolves
    # contradictory vehicle lists across staggered stop refreshes.
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

    @property
    def rows(self) -> tuple[ProbeEta, ...]:
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

    KMB/GMB ETA feeds are per-stop (all routes at once); Citybus is per
    (stop, route). One fetch per group covers every probe in it.
    """
    if probe.operator == "CTB":
        return f"CTB:{probe.stop_id}:{probe.route}"
    return f"{probe.operator}:{probe.stop_id}"


class ProbeEtaCache:
    """Last-value cache so the map can render between staggered sweeps.

    The TTL spans normal anchor sweeps, preventing mid-generation splits, but
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
        # Requests rotate across stops, so cached ETAs must count down
        # continuously between probes; otherwise a bus marker freezes and
        # then jumps a full minute.
        elapsed_minutes = age_seconds / 60.0
        if elapsed_minutes <= 0 and not any(eta.arrival_at is not None for eta in entry[1]):
            return entry[1]
        aged: list[ProbeEta] = []
        for eta in entry[1]:
            if eta.minutes is None:
                aged.append(replace(eta, cache_age_seconds=age_seconds))
                continue
            if eta.arrival_at is not None:
                # Absolute arrivals are always evaluated against the current
                # injected wall clock. Monotonic time is used only for cache
                # age and expiry, so clock injection remains testable.
                now = self._wall_clock()
                remaining = max(0.0, (eta.arrival_at - now).total_seconds() / 60.0)
            else:
                remaining = eta.minutes - elapsed_minutes
            if remaining > 0:
                aged.append(
                    replace(
                        eta,
                        minutes=remaining,
                        cache_age_seconds=age_seconds,
                    )
                )
        return aged

    def set(self, key: str, value: list[ProbeEta]) -> None:
        now = self._clock()
        self._sweep(now)
        self._store[key] = (now, value)
        self._sweep(now)


_probe_cache = ProbeEtaCache()


_probe_cursor = 0
_probe_generation = 0
_probe_route_generations: dict[tuple[str, str, str], _StoredProbeGeneration] = {}

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
        return await client.fetch_json(KMB_BASE.format(stop=probe.stop_id))
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
            _refresh_probe_etas(client, probes, max_per_cycle)
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
) -> list[ProbeEta]:
    """Poll up to ``max_per_cycle`` fetch groups round-robin; cache results.

    Probes sharing one HTTP request (same physical stop for KMB/GMB; same
    stop+route for Citybus) are fetched ONCE per sweep step and parsed once
    per probe, so dense probing stays cheap. Cached values are returned for
    probes outside this cycle's window. GMB groups are additionally capped
    per cycle and the GMB sweep backs off for a cooldown when the host
    rate-limits (HTTP 403), serving last-good cache meanwhile.
    """
    global _probe_cursor, _probe_generation
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
    routes: dict[tuple[str, str, str], list[str]] = {}
    for key, route_probes in groups.items():
        for probe in route_probes:
            route_key = (probe.operator, probe.route, probe.bound)
            route_list = routes.setdefault(route_key, [])
            if key not in route_list:
                route_list.append(key)
    route_keys = sorted(routes)
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
    # Select whole routes in deterministic round-robin order. A route is
    # publishable only when every one of its anchor groups is refreshed.
    selected_routes: list[tuple[str, str, str]] = []
    selected_groups: list[str] = []
    used_groups: set[str] = set()
    used_gmb = 0
    start = _probe_cursor % len(route_keys) if route_keys else 0
    last_considered = 0
    last_selected = -1
    for offset in range(len(route_keys)):
        last_considered = offset + 1
        route_key = route_keys[(start + offset) % len(route_keys)]
        route_groups = [key for key in routes[route_key] if key in selectable_keys]
        if len(route_groups) != len(routes[route_key]):
            continue
        extra = [key for key in route_groups if key not in used_groups]
        extra_gmb = sum(key.startswith("GMB:") for key in extra)
        if len(selected_groups) + len(extra) > cycle_budget:
            continue
        if used_gmb + extra_gmb > GMB_GROUPS_PER_CYCLE:
            continue
        selected_routes.append(route_key)
        last_selected = offset
        selected_groups.extend(extra)
        used_groups.update(extra)
        used_gmb += extra_gmb
    if route_keys:
        advance = (last_selected + 1) if last_selected >= 0 else last_considered
        _probe_cursor = (start + max(1, advance)) % len(route_keys)
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
    _probe_generation += 1
    generation = _probe_generation
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
        for probe in probes_in_group:
            key = _probe_cache_key(probe)
            parsed = [
                replace(row, refresh_generation=generation)
                for row in _parse_probe_etas(probe, raw, now)
            ]
            _probe_cache.set(key, parsed)
            refreshed_rows[key] = tuple(parsed)
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
    for route_key in selected_routes:
        route_groups = routes[route_key]
        if not all(successful.get(key, False) for key in route_groups):
            continue
        route_probes = tuple(
            probe for probe in unique.values()
            if (probe.operator, probe.route, probe.bound) == route_key
        )
        rows = tuple(
            eta for probe in route_probes
            for eta in refreshed_rows.get(_probe_cache_key(probe), ())
        )
        public = ProbeRouteGeneration(
            route_key=route_key,
            rows=rows,
            generation=generation,
            collected_at=collected_at,
        )
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
) -> ProbeEtaSnapshot:
    """Fetch probes and expose only atomically complete route generations."""
    await fetch_probe_etas(client, probes, max_per_cycle=max_per_cycle)
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
        elapsed_minutes = max(
            0.0, (collected_at - generation.public.collected_at).total_seconds() / 60.0
        )
        aged_rows = tuple(
            replace(
                row,
                minutes=(
                    max(0.0, (row.arrival_at - collected_at).total_seconds() / 60.0)
                    if row.arrival_at is not None
                    else row.minutes - elapsed_minutes
                ),
            )
            for row in generation.public.rows
            if (
                row.arrival_at is not None and row.arrival_at > collected_at
            ) or (
                row.arrival_at is None
                and row.minutes is not None
                and row.minutes - elapsed_minutes > 0
            )
        )
        routes.append(replace(generation.public, rows=aged_rows))
    return ProbeEtaSnapshot(
        routes=tuple(routes), collected_at=collected_at
    )
