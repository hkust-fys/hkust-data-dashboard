"""Transit ETA provider: KMB, Citybus, and GMB.

Mappings are ordered immutable tuples; route order in the dashboard follows the
order declared here. All functions take an injected ``HttpClient`` and are safe
to call concurrently.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
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

# ETA TTL is the provider cache window; the bot renders at its own cadence.
TRANSIT_TTL_SECONDS = 25.0


# --------------------------------------------------------------------------
# Parsing helpers
# --------------------------------------------------------------------------


def _minutes_until(iso: str | None, now: datetime) -> int | None:
    if not iso:
        return None
    try:
        eta = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except ValueError:
        return None
    if eta.tzinfo is None:
        eta = eta.replace(tzinfo=UTC)
    diff = (eta - now).total_seconds() / 60
    return max(0, round(diff))


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
    global _gmb_cooldown_until
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
                _gmb_cooldown_until = time.monotonic() + GMB_COOLDOWN_SECONDS
                log.warning(
                    "GMB ETA rate-limited (403); pausing gate and probe polling for %.0fs",
                    GMB_COOLDOWN_SECONDS,
                )
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
    now = datetime.now(UTC)
    results = await client.gather_any(
        [
            _fetch_kmb(client, now),
            _fetch_citybus(client, now),
            _fetch_gmb(client, now),
        ]
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
    return group_etas(rows), latest, failed


# --------------------------------------------------------------------------
# Probe-stop ETAs (downstream bus-position estimates)
# --------------------------------------------------------------------------

# Probe cache holds the last value per fetch group. The ceiling is deliberately
# longer than a complete sweep so early and late groups coexist, while still
# removing departed vehicles during a prolonged provider outage. Fourteen GMB
# groups per 10-second cycle cover the current 91-group set in about 70 seconds.
# The 15-minute ceiling therefore still bounds prolonged outage staleness.
PROBE_TTL_SECONDS = 900.0


@dataclass(frozen=True)
class ProbeEta:
    """One ETA observation at a probe stop along a tracked direction."""

    operator: str
    route: str
    bound: str
    stop_id: str
    index: int  # official stop-sequence index of the probe stop
    minutes: int | None
    kind: EtaKind = EtaKind.REALTIME


def _probe_cache_key(probe) -> str:
    return f"{probe.operator}:{probe.route}:{probe.bound}:{probe.stop_id}"


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

    The TTL spans several normal sweeps, preventing mid-sweep ladder splits,
    but bounds staleness when repeated failures leave an entry untouched.
    """

    def __init__(
        self,
        ttl_seconds: float = PROBE_TTL_SECONDS,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ttl = ttl_seconds
        self._clock = clock
        self._store: dict[str, tuple[float, list[ProbeEta]]] = {}

    def get(self, key: str) -> list[ProbeEta] | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        age_seconds = max(0.0, self._clock() - entry[0])
        if self._ttl > 0 and age_seconds > self._ttl:
            self._store.pop(key, None)
            return None
        # Requests rotate across stops, so cached integer ETAs must count down
        # between probes; otherwise a bus marker freezes and then jumps.
        elapsed_minutes = int(age_seconds // 60.0)
        if not elapsed_minutes:
            return entry[1]
        aged: list[ProbeEta] = []
        for eta in entry[1]:
            if eta.minutes is None:
                aged.append(eta)
                continue
            remaining = eta.minutes - elapsed_minutes
            if remaining >= 0:
                aged.append(replace(eta, minutes=remaining))
        return aged

    def set(self, key: str, value: list[ProbeEta]) -> None:
        self._store[key] = (self._clock(), value)


_probe_cache = ProbeEtaCache()


_probe_cursor = 0
_gmb_cursor = 0

# data.etagmb.gov.hk rate-limits bursts with 403s. GMB groups are capped per
# cycle and the whole GMB sweep backs off for a cooldown when a 403 appears;
# the probe cache keeps last-good values meanwhile.
GMB_GROUPS_PER_CYCLE = 14


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
            minutes = _minutes_until(entry.get("eta"), now)
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
            minutes = _minutes_until(eta_iso, now)
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
        last = -1
        for eta in entry.get("eta") or []:
            diff = eta.get("diff")
            try:
                minutes = int(diff) if diff is not None else None
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
                )
            )
    return out


async def fetch_probe_etas(
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
    global _probe_cursor, _gmb_cursor, _gmb_cooldown_until
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
    # Cap GMB groups per cycle, but rotate the cap across the complete sorted
    # set.  Slicing before round-robin would permanently starve groups after
    # the first 30. Non-GMB groups retain their independent round-robin.
    other = [key for key in selectable_keys if not key.startswith("GMB:")]
    gmb = [key for key in selectable_keys if key.startswith("GMB:")]
    cycle_budget = max(0, max_per_cycle)
    gmb_window = min(GMB_GROUPS_PER_CYCLE, cycle_budget, len(gmb))
    selected_gmb: list[str] = []
    if gmb_window:
        gmb_start = (_gmb_cursor * gmb_window) % len(gmb)
        selected_gmb = [gmb[(gmb_start + i) % len(gmb)] for i in range(gmb_window)]
        _gmb_cursor += 1
    if not other and not selected_gmb:
        return [
            eta
            for key in unique
            if (cached := _probe_cache.get(key)) is not None
            for eta in cached
        ]

    remaining = cycle_budget - len(selected_gmb)
    other_window = min(remaining, len(other))
    selected_other = []
    if other_window:
        other_start = (_probe_cursor * other_window) % len(other)
        selected_other = [other[(other_start + i) % len(other)] for i in range(other_window)]
        _probe_cursor += 1
    async def refresh_group(group_key: str) -> bool:
        global _gmb_cooldown_until
        if group_key.startswith("GMB:") and time.monotonic() < _gmb_cooldown_until:
            return True
        probes_in_group = groups[group_key]
        now = datetime.now(UTC)
        if group_key.startswith("GMB:") and time.monotonic() < _gmb_cooldown_until:
            return True
        try:
            raw = await _fetch_raw_stop_eta(client, probes_in_group[0])
        except FetchError as exc:
            if exc.status_code == 403 and group_key.startswith("GMB:"):
                _gmb_cooldown_until = time.monotonic() + GMB_COOLDOWN_SECONDS
                log.warning(
                    "GMB ETA rate-limited (403); pausing gate and probe polling for %.0fs",
                    GMB_COOLDOWN_SECONDS,
                )
                return True
            else:
                log.warning(
                    "probe ETA fetch failed for %s: %s",
                    group_key, type(exc).__name__,
                )
            return False
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "probe ETA fetch failed for %s: %s", group_key, type(exc).__name__
            )
            return False
        for probe in probes_in_group:
            key = _probe_cache_key(probe)
            _probe_cache.set(key, _parse_probe_etas(probe, raw, now))
        return False

    # Non-GMB origins can overlap. GMB requests are deliberately sequential:
    # once one returns 403, stop this sweep instead of draining an already
    # queued burst into the provider during its cooldown.
    await asyncio.gather(*(refresh_group(group_key) for group_key in selected_other))
    for group_key in selected_gmb:
        if await refresh_group(group_key):
            break

    collected: list[ProbeEta] = []
    for key in unique:
        cached = _probe_cache.get(key)
        if cached is not None:
            collected.extend(cached)
    return collected
