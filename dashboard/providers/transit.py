"""Transit ETA provider: KMB, Citybus, and GMB.

Mappings are ordered immutable tuples; route order in the dashboard follows the
order declared here. All functions take an injected ``HttpClient`` and are safe
to call concurrently.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

from dashboard.http import HttpClient
from dashboard.models import EtaKind, EtaRow, Operator, RouteEtaGroup

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Stop/route configuration (verified 2024-09-23, see private transport-mappings.md)
# --------------------------------------------------------------------------

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
                )
            )
    return rows


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
                )
            )
    return rows


async def _fetch_gmb(client: HttpClient, now: datetime) -> list[EtaRow]:
    rows: list[EtaRow] = []
    urls = [GMB_BASE.format(stop=stop_id) for stop_id in GMB_STOPS]
    responses = await _fetch_many_json(client, urls, "GMB")
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
                        )
                    )
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
        ) != (row.route, row.destination, row.gate, row.operator, row.stop_seq):
            current = RouteEtaGroup(
                route=row.route,
                destination=row.destination,
                gate=row.gate,
                operator=row.operator,
                stop_seq=row.stop_seq,
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
