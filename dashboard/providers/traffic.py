"""Transport Department traffic provider: detector speeds/volume/occupancy,
Special Traffic News, roadworks GeoJSON.

Official TD/data.gov.hk feeds and RTHK's public current traffic-news page.
Speed bands are dashboard heuristics (documented), not TD classifications.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta, timezone
from html.parser import HTMLParser
from typing import Any

from dashboard.http import CachedFetch, FetchError, HttpClient, as_datetime
from dashboard.models import (
    Roadwork,
    SpeedBand,
    TrafficCorridorStatus,
    TrafficIncident,
    TrafficObservation,
)
from dashboard.providers.traffic_news import (
    DEFAULT_RTHK_NEWS_MAX_AGE_HOURS,
    RTHK_TRAFFIC_NEWS_URL,
    filter_expired_rthk_incidents,
    is_cleared_notice,
    parse_rthk_traffic_news,
    rthk_latest_report_time,
)
from dashboard.providers.traffic_reconciliation import (
    pair_td_bilingual_pages,
    reconcile_incident_reports,
)

log = logging.getLogger(__name__)
HKT = timezone(timedelta(hours=8))

# --------------------------------------------------------------------------
# Endpoints (public, no key)
# --------------------------------------------------------------------------

DETECTOR_META_URL = (
    "https://static.data.gov.hk/td/traffic-data-strategic-major-roads/info/"
    "traffic_speed_volume_occ_info.csv"
)
DETECTOR_OBS_URL = "https://resource.data.one.gov.hk/td/traffic-detectors/rawSpeedVol-all.xml"
# Retained for offline parser compatibility only; runtime never collects it.
SPECIAL_NEWS_URL = "https://www.td.gov.hk/en/special_news/trafficnews.xml"
SPECIAL_NEWS_PAGE_URL = "https://www.td.gov.hk/en/special_news/spnews.htm"
SPECIAL_NEWS_TC_PAGE_URL = "https://www.td.gov.hk/tc/special_news/spnews.htm"
ROADWORKS_URL = (
    "https://resource.data.one.gov.hk/td/roadworks-location/get_all_the_roadworks.geojson"
)

# Refresh cadences.
OBS_TTL_SECONDS = 55.0
# Both current news pages are checked independently once per minute.
NEWS_TTL_SECONDS = 60.0
ROADWORKS_TTL_SECONDS = 15 * 60.0
META_TTL_SECONDS = 24 * 60 * 60.0

DETECTOR_META_SPEC = CachedFetch(
    DETECTOR_META_URL,
    META_TTL_SECONDS,
    cache_key="td-detector-metadata",
)
DETECTOR_OBS_SPEC = CachedFetch(
    DETECTOR_OBS_URL,
    OBS_TTL_SECONDS,
    cache_key="td-detector-observations",
)
SPECIAL_NEWS_SPEC = CachedFetch(
    SPECIAL_NEWS_URL,
    NEWS_TTL_SECONDS,
    cache_key="td-special-news",
)
SPECIAL_NEWS_PAGE_SPEC = CachedFetch(
    SPECIAL_NEWS_PAGE_URL,
    NEWS_TTL_SECONDS,
    cache_key="td-special-news-page",
)
SPECIAL_NEWS_TC_PAGE_SPEC = CachedFetch(
    SPECIAL_NEWS_TC_PAGE_URL,
    NEWS_TTL_SECONDS,
    cache_key="td-special-news-page-tc",
)
RTHK_NEWS_SPEC = CachedFetch(
    RTHK_TRAFFIC_NEWS_URL,
    NEWS_TTL_SECONDS,
    cache_key="rthk-traffic-news-page0",
)
ROADWORKS_SPEC = CachedFetch(
    ROADWORKS_URL,
    ROADWORKS_TTL_SECONDS,
    cache_key="td-roadworks",
)

# --------------------------------------------------------------------------
# Direction words that appear in TD detector descriptions ("Westbound",
# "Eastbound") and in special-news text.
# --------------------------------------------------------------------------

DIRECTION_HINTS: tuple[tuple[str, str], ...] = (
    ("eastbound", "→ E"),
    ("westbound", "← W"),
    ("northbound", "→ N"),
    ("southbound", "← S"),
    ("towards", "→"),
    ("toward", "→"),
    ("to kwun tong", "→"),
    ("to po lam", "→"),
    ("to hang hau", "→"),
    ("to sai kung", "→"),
    ("to choi hung", "→"),
    ("inbound", "→"),
    ("outbound", "←"),
    ("from kwun tong", "←"),
    ("from po lam", "←"),
)


@dataclass
class _DetectorMeta:
    detector_id: str
    description: str
    latitude: float | None
    longitude: float | None
    direction: str = ""


# --------------------------------------------------------------------------
# Detector metadata CSV
# --------------------------------------------------------------------------


def parse_detector_metadata(csv_text: str) -> dict[str, _DetectorMeta]:
    """Parse the TD detector CSV into {detector_id: meta}.

    Column names are not stable across releases (previously `detector_id`,
    now `AID_ID_Number`; description is `Road_EN`), so we locate headers by
    name (case-insensitive) rather than position. BOM is stripped.
    """
    text = csv_text.lstrip("\ufeff")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return {}
    header = [h.strip().lower() for h in _split_csv_line(lines[0])]

    def col(*names: str) -> int | None:
        for i, h in enumerate(header):
            if h in names:
                return i
        return None

    id_col = col("aid_id_number", "detector_id", "detectorid", "id", "indid")
    desc_col = col("road_en", "description", "road_description", "road", "location")
    lat_col = col("latitude", "lat")
    lon_col = col("longitude", "lon", "long")
    dir_col = col("direction")
    if id_col is None:
        log.warning("TD detector CSV has no id column; headers: %s", header)
        return {}

    meta: dict[str, _DetectorMeta] = {}
    for line in lines[1:]:
        fields = _split_csv_line(line)
        if len(fields) <= id_col:
            continue
        detector_id = fields[id_col].strip()
        if not detector_id:
            continue
        meta[detector_id] = _DetectorMeta(
            detector_id=detector_id,
            description=(
                fields[desc_col].strip() if desc_col is not None and len(fields) > desc_col else ""
            ),
            latitude=(
                _to_float(fields[lat_col])
                if lat_col is not None and len(fields) > lat_col
                else None
            ),
            longitude=(
                _to_float(fields[lon_col])
                if lon_col is not None and len(fields) > lon_col
                else None
            ),
            direction=(
                fields[dir_col].strip() if dir_col is not None and len(fields) > dir_col else ""
            ),
        )
    return meta


def _split_csv_line(line: str) -> list[str]:
    """Minimal CSV line splitter (handles quoted fields)."""
    result: list[str] = []
    current: list[str] = []
    in_quotes = False
    for ch in line:
        if ch == '"':
            in_quotes = not in_quotes
        elif ch == "," and not in_quotes:
            result.append("".join(current))
            current = []
        else:
            current.append(ch)
    result.append("".join(current))
    return result


def _to_float(value: str) -> float | None:
    try:
        return float(value.strip())
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------
# Detector observations XML
# --------------------------------------------------------------------------


def parse_detector_observations(xml_text: str) -> dict[str, dict[str, Any]]:
    """Parse rawSpeedVol-all.xml into {detector_id: aggregate lane stats}.

    Live schema (verified 2026-08-06):
      <raw_speed_volume_list><date>YYYY-MM-DD</date><periods>
        <period><period_from>HH:MM:SS</period_from><detectors>
          <detector><detector_id>AID01101</detector_id><direction>..</direction>
            <lanes><lane><lane_id>Fast Lane</lane_id><speed>70</speed>
              <occupancy>0</occupancy><volume>0</volume><s.d.>0</s.d.><valid>Y</valid>
            </lane>...</lanes></detector>...</detectors>
        </period></periods></raw_speed_volume_list>

    Lane values are aggregated: average speed of valid lanes, summed volume.
    ``capture_time`` is built from <date> + <period_from> (local time, UTC+8).
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        log.warning("TD detector XML parse failed: %s", exc)
        return {}

    date_text = _local_text(root, "date")
    out: dict[str, dict[str, Any]] = {}
    # Single pass over <period> groups: each period's from-time applies to its
    # own detectors. (The previous per-detector ancestor rescan was O(n^2) and
    # large enough to stall the event loop and block the Discord heartbeat.)
    for period in root.iter():
        if _local_name(period.tag) != "period":
            continue
        period_from = _local_text(period, "period_from")
        capture = _build_capture_time(date_text, period_from)
        for detector in period.iter():
            if _local_name(detector.tag) != "detector":
                continue
            did = _local_text(detector, "detector_id")
            if not did:
                continue

            speeds: list[float] = []
            volumes: list[int] = []
            occupancies: list[float] = []
            for lane in detector.iter():
                if _local_name(lane.tag) != "lane":
                    continue
                valid = (_local_text(lane, "valid") or "").strip().upper()
                if valid == "N":
                    continue
                speed = _to_float(_local_text(lane, "speed"))
                if speed is not None:
                    speeds.append(speed)
                volume = _to_int(_local_text(lane, "volume"))
                if volume is not None:
                    volumes.append(volume)
                occ = _to_float(_local_text(lane, "occupancy"))
                if occ is not None:
                    occupancies.append(occ)
            if not speeds and not volumes:
                continue
            out[did] = {
                "speed": round(sum(speeds) / len(speeds), 1) if speeds else None,
                "volume": sum(volumes) if volumes else None,
                "occupancy": round(sum(occupancies) / len(occupancies), 1) if occupancies else None,
                "capture_time": capture,
            }
    return out


def _local_name(tag: str) -> str:
    return tag.split("}")[-1]


def _local_text(elem: ET.Element, name: str) -> str:
    """Return the direct text of the first child whose local name matches."""
    for child in elem.iter():
        if _local_name(child.tag) == name:
            return (child.text or "").strip()
    return ""


def _build_capture_time(date_text: str, period_from: str) -> datetime | None:
    if not date_text or not period_from:
        return None
    try:
        return datetime.fromisoformat(f"{date_text}T{period_from}+08:00")
    except ValueError:
        return None


def _to_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(float(value.strip()))
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------
# Road matching (the road table comes from dashboard.providers.tracked_roads)
# --------------------------------------------------------------------------


def match_roads(text: str, roads: Any) -> list[str]:
    """Return canonical road keys whose aliases appear in ``text``.

    ``roads`` is a ``TrackedRoads`` table; passing ``None`` matches nothing so
    callers without a loaded road table stay silent rather than over-matching.
    """
    if roads is None:
        return []
    return roads.match(text)


_PARENTHETICAL = re.compile(r"\([^()]*\)")
_DIRECTIONAL_WORD = re.compile(
    r"\b(?:bound|inbound|outbound|towards?|direction|heading|clockwise|anticlockwise)\b",
    re.IGNORECASE,
)
_ROAD_DESIGNATOR = re.compile(
    r"\b(?:road|street|highway|expressway|tunnel|flyover|bridge|bypass|avenue|drive|lane)\b",
    re.IGNORECASE,
)


def _without_direction_parentheticals(text: str) -> str:
    """Remove parenthetical directions before looking for narrative road names."""

    return _PARENTHETICAL.sub(
        lambda match: " " if _DIRECTIONAL_WORD.search(match.group(0)) else match.group(0),
        text,
    )


def _road_words(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).strip()


def _road_key_phrases(key: str, roads: Any) -> set[str]:
    """Return normalized names that may identify one canonical road key."""

    values: list[object] = [key]
    aliases = getattr(roads, "aliases", None)
    if isinstance(aliases, dict):
        values.append(aliases.get(key, ""))
    display_name = getattr(roads, "display_name", None)
    if callable(display_name):
        values.append(display_name(key))
    return {words for value in values if (words := _road_words(value))}


def resolve_incident_road_keys(
    incident: TrafficIncident,
    roads: Any = None,
    *,
    prefer_refinement: bool = False,
    explicit_fallback_keys: Iterable[str] = (),
) -> list[str]:
    """Resolve a TD traffic notice to canonical tracked-road keys.

    TD's dedicated ``road``/``location`` fields are authoritative. Narrative
    road names are considered only when those fields do not name a road, with
    one conservative exception: a specifically written tracked key may refine
    an explicit parent (for example ``Lung Cheung Road flyover``). Parenthetical
    direction labels are never road evidence. ``explicit_fallback_keys`` keeps
    table-outage behavior exact-field-only rather than reopening prose matching.
    """

    explicit_fields: list[str] = []
    seen_fields: set[str] = set()
    for value in (incident.road, incident.location):
        cleaned = _without_direction_parentheticals(value).strip()
        identity = cleaned.casefold()
        if cleaned and identity not in seen_fields:
            seen_fields.add(identity)
            explicit_fields.append(cleaned)

    if roads is None:
        explicit_words = {_road_words(field) for field in explicit_fields}
        return [key for key in explicit_fallback_keys if _road_words(key) in explicit_words]

    explicit_matches: list[str] = []
    seen_keys: set[str] = set()
    for field in explicit_fields:
        for key in match_roads(field, roads):
            if key not in seen_keys:
                seen_keys.add(key)
                explicit_matches.append(key)

    explicit_parents = [
        _road_words(field) for field in explicit_fields if _ROAD_DESIGNATOR.search(field)
    ]
    narrative = _without_direction_parentheticals(" ".join((incident.title, incident.description)))
    narrative_matches = match_roads(narrative, roads)
    refinements = [
        key
        for key in narrative_matches
        if any(
            phrase.startswith(f"{parent} ")
            for parent in explicit_parents
            for phrase in _road_key_phrases(key, roads)
        )
    ]

    if explicit_matches:
        if prefer_refinement:
            return refinements
        return explicit_matches
    if explicit_parents:
        return refinements
    return narrative_matches


def _direction_from(text: str, meta_direction: str = "") -> str:
    lowered = text.lower()
    for hint, arrow in DIRECTION_HINTS:
        if hint in lowered:
            return arrow
    # fall back to the CSV Direction column (e.g. "North West")
    if meta_direction:
        return f"({meta_direction})"
    return ""


def build_corridor_statuses(
    observations: dict[str, dict[str, Any]],
    meta: dict[str, _DetectorMeta],
    roads: Any = None,
    max_observations: int = 6,
) -> list[TrafficCorridorStatus]:
    """Group detector observations by matched road and summarize."""
    groups: dict[str, list[TrafficObservation]] = {}
    for did, obs in observations.items():
        m = meta.get(did)
        if m is None:
            continue
        corridors = match_roads(m.description, roads)
        if not corridors:
            continue
        # a detector may serve multiple roads; attach to the first match
        corridor = corridors[0]
        speed = obs.get("speed")
        stale = obs.get("capture_time") is None
        band = speed_band(speed, stale=stale)
        observation = TrafficObservation(
            corridor=corridor,
            direction=_direction_from(m.description, m.direction),
            description=m.description,
            latitude=m.latitude if m.latitude is not None else 0.0,
            longitude=m.longitude if m.longitude is not None else 0.0,
            speed_kmh=speed,
            volume=obs.get("volume"),
            occupancy_pct=obs.get("occupancy"),
            capture_time=obs.get("capture_time"),
            band=band,
            stale=stale,
        )
        groups.setdefault(corridor, []).append(observation)

    statuses: list[TrafficCorridorStatus] = []
    for corridor in sorted(groups):
        obs_list = sorted(
            groups[corridor],
            key=lambda o: (o.speed_kmh is None, -(o.speed_kmh or 0)),
        )[:max_observations]
        capture = max(
            (o.capture_time for o in obs_list if o.capture_time),
            default=None,
        )
        statuses.append(
            TrafficCorridorStatus(
                name=corridor,
                direction=_direction_from(
                    obs_list[0].description,
                    obs_list[0].direction if hasattr(obs_list[0], "direction") else "",
                ),
                observations=obs_list,
                capture_time=capture,
            )
        )
    return statuses


def speed_band(speed_kmh: float | None, stale: bool = False) -> SpeedBand:
    """Dashboard speed bands (heuristics): red <20, amber 20–40, green >40,
    gray missing/stale."""
    if speed_kmh is None or stale:
        return SpeedBand.GRAY
    if speed_kmh < 20:
        return SpeedBand.RED
    if speed_kmh <= 40:
        return SpeedBand.AMBER
    return SpeedBand.GREEN


# --------------------------------------------------------------------------
# Special Traffic News (XML v2)
# --------------------------------------------------------------------------


def parse_special_news(xml_text: str) -> list[TrafficIncident]:
    """Parse current and legacy TD special traffic news XML schemas."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        log.warning("TD special news parse failed: %s", exc)
        return []

    incidents: list[TrafficIncident] = []
    items = [
        element
        for element in root.iter()
        if element.tag.split("}")[-1].lower() in {"item", "message"}
    ]
    for item in items:
        fields: dict[str, str] = {}
        for child in item:
            tag = child.tag.split("}")[-1].lower()
            fields[tag] = (child.text or "").strip()
        identifier = (
            fields.get("incident_number") or fields.get("identifier") or fields.get("id") or ""
        )
        title = fields.get("incident_heading_en") or fields.get("title") or ""
        description = (
            fields.get("content_en")
            or fields.get("incident_detail_en")
            or fields.get("description")
            or fields.get("details")
            or ""
        )
        location = fields.get("location_en") or fields.get("location") or ""
        direction = fields.get("direction_en") or fields.get("direction") or ""
        status = fields.get("incident_status_en") or fields.get("status") or ""
        if not (identifier or title):
            continue
        announcement_time = as_datetime(fields.get("announcement_date"))
        if announcement_time is not None and announcement_time.tzinfo is None:
            # TD's live feed publishes local Hong Kong wall-clock values.
            announcement_time = announcement_time.replace(tzinfo=HKT)
        incidents.append(
            TrafficIncident(
                identifier=identifier,
                title=title,
                description=_sanitize_text(description),
                road=fields.get("road") or location,
                location=location,
                direction=direction,
                status=status,
                start_time=as_datetime(fields.get("start_time") or fields.get("effective_time")),
                end_time=as_datetime(fields.get("end_time") or fields.get("expiry_time")),
                announcement_time=announcement_time,
                latitude=_to_float(fields.get("latitude") or ""),
                longitude=_to_float(fields.get("longitude") or ""),
                near_landmark=fields.get("near_landmark_en", ""),
                between_landmark=fields.get("between_landmark_en", ""),
                source="TD",
                source_url=SPECIAL_NEWS_PAGE_URL,
            )
        )
    return _dedupe_incidents(incidents)


def _validate_special_news_feed(xml_text: str) -> None:
    """Reject malformed or unrecognized TD news before it enters the cache.

    A recognized root with no ``item``/``message`` children is a legitimate
    empty feed.  Once entries exist, each must retain one of the identity/title
    fields understood by :func:`parse_special_news`; otherwise a schema change
    must be observable as a fetch failure rather than fresh "no notices" data.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise FetchError("Invalid TD special news XML") from exc

    root_name = _local_name(root.tag).casefold()
    if root_name not in {"list", "trafficnews"}:
        raise FetchError(f"Unexpected TD special news root {root_name!r}")

    identity_fields = {
        "incident_number",
        "identifier",
        "id",
        "incident_heading_en",
        "title",
    }
    items = [
        element
        for element in root.iter()
        if _local_name(element.tag).casefold() in {"item", "message"}
    ]
    for item in items:
        fields = {_local_name(child.tag).casefold() for child in item}
        if not fields.intersection(identity_fields):
            raise FetchError("Unrecognized TD special news item schema")


async def _fetch_validated_special_news(client: HttpClient, url: str) -> str:
    xml_text = await client.fetch_xml_text(url)
    _validate_special_news_feed(xml_text)
    return xml_text


class _SpecialNewsPageParser(HTMLParser):
    """Extract list-item text from TD's small, legacy-markup news page."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.found_ordered_list = False
        self.items: list[str] = []
        self.page_text: list[str] = []
        self._ordered_list_depth = 0
        self._item_parts: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        lowered = tag.casefold()
        if lowered == "ol":
            self.found_ordered_list = True
            self._ordered_list_depth += 1
        elif lowered == "li" and self._ordered_list_depth:
            self._item_parts = []
        elif lowered == "br" and self._item_parts is not None:
            self._item_parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.casefold()
        if lowered == "li" and self._item_parts is not None:
            text = _sanitize_text(" ".join(self._item_parts))
            if text:
                self.items.append(text)
            self._item_parts = None
        elif lowered == "ol" and self._ordered_list_depth:
            self._ordered_list_depth -= 1

    def handle_data(self, data: str) -> None:
        self.page_text.append(data)
        if self._item_parts is not None:
            self._item_parts.append(data)


_REOPENED = re.compile(r"\bre[\s-]?opened\s+to\s+all\s+traffic\b", re.IGNORECASE)
_BOUND_DIRECTION = re.compile(r"\(([^()]*(?:bound|bounds))\)", re.IGNORECASE)


def parse_special_news_page(html_text: str) -> list[TrafficIncident]:
    """Parse the complete official TD special-news HTML list.

    The page is authoritative for the current list.  Its page timestamp is a
    refresh time, not an event announcement time, so parsed entries deliberately
    leave ``announcement_time`` unset.
    """
    parser = _SpecialNewsPageParser()
    try:
        parser.feed(html_text)
        parser.close()
    except Exception as exc:  # HTMLParser can surface malformed entity input
        raise FetchError("Invalid TD special news HTML") from exc

    page_text = _sanitize_text(" ".join(parser.page_text)).casefold()
    if "special traffic news" not in page_text or not parser.found_ordered_list:
        raise FetchError("Unrecognized TD special news page")

    incidents: list[TrafficIncident] = []
    for text in parser.items:
        lowered = text.casefold()
        reopened = bool(_REOPENED.search(text))
        if "traffic accident" in lowered:
            title = "Traffic accident"
        elif "vehicle breakdown" in lowered:
            title = "Vehicle breakdown"
        elif "roadwork" in lowered or "works" in lowered:
            title = "Road works"
        else:
            title = "TD special traffic news"
        direction_match = _BOUND_DIRECTION.search(text)
        digest = hashlib.sha256(text.casefold().encode("utf-8")).hexdigest()[:16]
        incidents.append(
            TrafficIncident(
                identifier=f"TD-WEB-{digest}",
                title=title,
                description=text,
                road="",
                location="",
                direction=(direction_match.group(1).strip() if direction_match else ""),
                status="CLOSED" if reopened else "ACTIVE",
                source="TD",
                source_url=SPECIAL_NEWS_PAGE_URL,
            )
        )
    return incidents


async def _fetch_validated_special_news_page(client: HttpClient, url: str) -> str:
    html_text = await client.fetch_html(url, max_bytes=256 * 1024)
    parse_special_news_page(html_text)
    return html_text


async def _fetch_validated_special_news_tc_page(client: HttpClient, url: str) -> str:
    html_text = await client.fetch_html(url, max_bytes=256 * 1024)
    parser = _SpecialNewsPageParser()
    parser.feed(html_text)
    page_text = _sanitize_text(" ".join(parser.page_text))
    if "\u7279\u5225\u4ea4\u901a\u6d88\u606f" not in page_text or not parser.found_ordered_list:
        raise FetchError("Unrecognized TD Chinese special news page")
    return html_text


def special_news_page_time(html_text: str) -> datetime | None:
    """Read TD's displayed page update time, separately from our fetch time."""
    parser = _SpecialNewsPageParser()
    parser.feed(html_text)
    text = _sanitize_text(" ".join(parser.page_text))
    match = re.search(
        r"(\d{4})(?:/|\u5e74)(\d{1,2})(?:/|\u6708)(\d{1,2})(?:\u65e5)?\s+"
        r"(\d{1,2}:\d{2}:\d{2})\s+([AP]M)",
        text,
        re.I,
    )
    if match is None:
        return None
    try:
        normalized = "/".join(match.group(1, 2, 3)) + " " + " ".join(match.group(4, 5))
        return datetime.strptime(normalized.upper(), "%Y/%m/%d %I:%M:%S %p").replace(tzinfo=HKT)
    except ValueError:
        return None


async def _fetch_validated_rthk_news(client: HttpClient, url: str, roads: Any) -> str:
    html_text = await client.fetch_html(url, max_bytes=512 * 1024)
    parse_rthk_traffic_news(html_text, roads)
    return html_text


def _sanitize_text(text: str) -> str:
    """Remove control chars and collapse whitespace."""
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _dedupe_incidents(incidents: list[TrafficIncident]) -> list[TrafficIncident]:
    seen: set[tuple[str, str]] = set()
    out: list[TrafficIncident] = []
    for inc in incidents:
        key = (inc.identifier or inc.title, inc.status)
        if key in seen:
            continue
        seen.add(key)
        out.append(inc)
    return out


def _incident_event_key(incident: TrafficIncident, road_key: str) -> tuple[str, ...]:
    """Return a conservative key pairing a closure with its re-opening item."""
    text = _road_words(" ".join((incident.title, incident.description, incident.direction)))

    def detail(pattern: str) -> str:
        match = re.search(pattern, text)
        return match.group(1).strip() if match else ""

    direction = detail(r"\b((?:[a-z]+\s+){0,3}bound|both bounds)\b")
    landmark = detail(r"\bnear\s+(.+?)(?=\s+(?:which|is|was)\b|$)")
    cause = detail(r"\bdue to\s+(.+?)(?=\s+(?:is|was|which|part|the)\b|$)")
    lane = detail(r"\b((?:fast|slow|middle) lane|part of the lanes|all lanes|the lane)\b")
    original_text = " ".join((incident.title, incident.description, incident.direction))
    if not direction:
        match = re.search(r"往(.{1,12}?)方向", original_text)
        direction = match.group(1).strip() if match else ""
    if not landmark:
        match = re.search(r"近(.+?)(?=的|有|，|。|$)", original_text)
        landmark = match.group(1).strip() if match else ""
        # RTHK uses both "近港鐵站慢線的..." and "近港鐵站的慢線...".
        # Lane wording is not part of the landmark's identity.
        landmark = re.sub(r"(?:慢線|快線|中線|部分行車線|所有行車線)$", "", landmark)
    if not cause:
        for phrase in ("交通意外", "壞車", "水管緊急維修", "道路工程"):
            if phrase in original_text:
                cause = phrase
                break
    return road_key, direction, landmark, cause, lane


def _is_reopened_incident(incident: TrafficIncident) -> bool:
    return is_cleared_notice(incident)


def filter_relevant_incidents(
    incidents: list[TrafficIncident], roads: Any = None, limit: int | None = None
) -> list[TrafficIncident]:
    """Keep tracked-road updates and suppress closures already re-opened.

    TD's HTML list is newest-first and can briefly contain both a re-opening
    notice and the older closure.  A matching earlier re-opening therefore
    cancels the older closure.  The re-opening remains as a ``CLOSED`` update
    so the renderer can attribute the latest state without treating it as
    active congestion evidence.
    """
    cleared: set[tuple[str, ...]] = set()
    relevant: list[TrafficIncident] = []
    for incident in incidents:
        road_keys = resolve_incident_road_keys(incident, roads)
        if not road_keys:
            continue
        event_keys = {_incident_event_key(incident, key) for key in road_keys}
        if _is_reopened_incident(incident):
            cleared.update(event_keys)
            relevant.append(incident)
            if limit is not None and len(relevant) >= limit:
                break
            continue
        if event_keys.intersection(cleared):
            continue
        relevant.append(incident)
        if limit is not None and len(relevant) >= limit:
            break
    return relevant


# --------------------------------------------------------------------------
# Roadworks GeoJSON
# --------------------------------------------------------------------------


def parse_roadworks(geojson: dict[str, Any], roads: Any = None) -> list[Roadwork]:
    """Parse TD roadworks GeoJSON; match against our tracked roads."""
    out: list[Roadwork] = []
    features = geojson.get("features") or []
    for feature in features:
        props = feature.get("properties") or {}
        description = " ".join(
            str(props.get(k) or "") for k in ("description", "name", "location", "road")
        )
        if not match_roads(description, roads):
            continue
        identifier = str(props.get("id") or props.get("identifier") or "")
        out.append(
            Roadwork(
                identifier=identifier,
                description=_sanitize_text(description),
                road=str(props.get("road") or ""),
                start_time=as_datetime(props.get("start_date") or props.get("start_time")),
                end_time=as_datetime(props.get("end_date") or props.get("end_time")),
            )
        )
    return out


async def _collect_td_news(
    client: HttpClient, roads: Any
) -> tuple[list[TrafficIncident], list[str], dict[str, datetime]]:
    markers: list[str] = []
    times: dict[str, datetime] = {}
    raw_incidents: list[TrafficIncident] = []
    fetched_at: float | None = None
    page_updated_at: datetime | None = None
    page_text = ""
    chinese_page_text = ""
    english_result, chinese_result = await asyncio.gather(
        client._fetch_cached(  # noqa: SLF001
            SPECIAL_NEWS_PAGE_SPEC,
            lambda url: _fetch_validated_special_news_page(client, url),
        ),
        client._fetch_cached(  # noqa: SLF001
            SPECIAL_NEWS_TC_PAGE_SPEC,
            lambda url: _fetch_validated_special_news_tc_page(client, url),
        ),
        return_exceptions=True,
    )
    if isinstance(english_result, BaseException):
        log.warning("TD special news page fetch failed: %s", english_result)
        markers.append("TD traffic news page unavailable")
    else:
        stale, page_text, fetched_at = english_result
        if stale:
            markers.append("TD traffic news")
        raw_incidents = parse_special_news_page(page_text)
        page_updated_at = special_news_page_time(page_text)
        raw_incidents = [
            replace(incident, page_updated_at=page_updated_at) for incident in raw_incidents
        ]
    if isinstance(chinese_result, BaseException):
        log.warning("TD Chinese special news page fetch failed: %s", chinese_result)
    else:
        _tc_stale, chinese_page_text, _tc_fetched_at = chinese_result

    if raw_incidents and chinese_page_text:
        sidecars = pair_td_bilingual_pages(page_text, chinese_page_text, raw_incidents, roads)
        raw_incidents = [
            replace(
                incident,
                translated_description=sidecars.get(incident.identifier, ""),
            )
            for incident in raw_incidents
        ]

    incidents = filter_relevant_incidents(raw_incidents, roads)
    if fetched_at is not None:
        checked_time = datetime.fromtimestamp(fetched_at, UTC)
        times["traffic_news_checked"] = checked_time
        official_time = max(
            (incident.announcement_time for incident in incidents if incident.announcement_time),
            default=None,
        )
        times["traffic_news"] = page_updated_at or official_time or checked_time
        if page_updated_at is not None:
            times["traffic_news_updated"] = page_updated_at
    return incidents, markers, times


async def _collect_rthk_news(
    client: HttpClient,
    roads: Any,
    *,
    rthk_news_max_age_hours: float = DEFAULT_RTHK_NEWS_MAX_AGE_HOURS,
    now: datetime | None = None,
) -> tuple[list[TrafficIncident], list[str], dict[str, datetime]]:
    markers: list[str] = []
    times: dict[str, datetime] = {}
    try:
        stale, page_text, fetched_at = await client._fetch_cached(  # noqa: SLF001
            RTHK_NEWS_SPEC,
            lambda url: _fetch_validated_rthk_news(client, url, roads),
        )
        if stale:
            markers.append("RTHK traffic news")
        incidents = filter_expired_rthk_incidents(
            filter_relevant_incidents(parse_rthk_traffic_news(page_text, roads), roads),
            now=now or datetime.now(HKT),
            max_age_hours=rthk_news_max_age_hours,
        )
        times["rthk_news_checked"] = datetime.fromtimestamp(fetched_at, UTC)
        report_time = rthk_latest_report_time(page_text)
        if report_time is not None:
            times["rthk_news"] = report_time
        return incidents, markers, times
    except Exception as exc:  # noqa: BLE001
        log.warning("RTHK traffic news fetch failed: %s", exc)
        return [], ["RTHK traffic news unavailable"], {}


# --------------------------------------------------------------------------
# Public facade
# --------------------------------------------------------------------------


async def fetch_traffic_data(
    client: HttpClient,
    roads: Any = None,
    *,
    rthk_news_max_age_hours: float = DEFAULT_RTHK_NEWS_MAX_AGE_HOURS,
    now: datetime | None = None,
) -> tuple[
    list[TrafficCorridorStatus],
    list[TrafficIncident],
    list[Roadwork],
    datetime | None,
    list[str],
    dict[str, datetime],
]:
    """Fetch detectors, special news, and roadworks (metadata is cached daily).

    ``roads`` is the loaded ``TrackedRoads`` table used to match TD text; with
    ``None`` no road matches and only empty statuses/notices are returned.

    Returns ``(statuses, incidents, roadworks, detector_capture_time,
    stale_sources, source_times)``.  ``source_times`` preserves the cached
    fetch time for each independently refreshed TD feed; the detector XML's
    own capture time takes precedence when present.
    """
    stale_sources: list[str] = []
    source_times: dict[str, datetime] = {}
    # metadata: daily cache, tolerate failure
    meta: dict[str, _DetectorMeta] = {}
    try:
        stale, meta_text, _ = await client.fetch_text_cached(
            DETECTOR_META_SPEC,
            max_bytes=2 * 1024 * 1024,
        )
        if stale:
            stale_sources.append("TD detector metadata")
        meta = parse_detector_metadata(meta_text)
    except Exception as exc:  # noqa: BLE001
        log.warning("TD detector metadata fetch failed: %s", exc)

    statuses: list[TrafficCorridorStatus] = []
    capture_time: datetime | None = None
    try:
        stale, obs_text, fetched_at = await client.fetch_xml_text_cached(DETECTOR_OBS_SPEC)
        if stale:
            stale_sources.append("TD detector observations")
        obs = parse_detector_observations(obs_text)
        statuses = build_corridor_statuses(obs, meta, roads)
        capture_time = max(
            (o.capture_time for s in statuses for o in s.observations if o.capture_time),
            default=None,
        )
        source_times["detectors"] = capture_time or datetime.fromtimestamp(fetched_at, UTC)
    except Exception as exc:  # noqa: BLE001
        log.warning("TD detector observations fetch failed: %s", exc)

    td_news, rthk_news = await asyncio.gather(
        _collect_td_news(client, roads),
        _collect_rthk_news(
            client,
            roads,
            rthk_news_max_age_hours=rthk_news_max_age_hours,
            now=now,
        ),
    )
    incidents = reconcile_incident_reports(td_news[0] + rthk_news[0], roads)
    stale_sources.extend(td_news[1])
    stale_sources.extend(rthk_news[1])
    source_times.update(td_news[2])
    source_times.update(rthk_news[2])

    roadworks: list[Roadwork] = []
    try:
        stale, rw, fetched_at = await client.fetch_json_cached(ROADWORKS_SPEC)
        if stale:
            stale_sources.append("TD roadworks")
        roadworks = parse_roadworks(rw, roads)
        source_times["roadworks"] = datetime.fromtimestamp(fetched_at, UTC)
    except Exception as exc:  # noqa: BLE001
        log.warning("TD roadworks fetch failed: %s", exc)

    return statuses, incidents, roadworks, capture_time, stale_sources, source_times
