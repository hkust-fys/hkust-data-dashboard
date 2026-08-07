"""Transport Department traffic provider: detector speeds/volume/occupancy,
Special Traffic News, roadworks GeoJSON, and two CCTV JPEGs.

Official TD/data.gov.hk feeds only. RTHK scraping is explicitly out of scope.
Speed bands are dashboard heuristics (documented), not TD classifications.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from dashboard.http import HttpClient, as_datetime
from dashboard.models import (
    ImageAsset,
    Roadwork,
    SpeedBand,
    TrafficCorridorStatus,
    TrafficIncident,
    TrafficObservation,
)

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Endpoints (public, no key)
# --------------------------------------------------------------------------

DETECTOR_META_URL = (
    "https://static.data.gov.hk/td/traffic-data-strategic-major-roads/info/"
    "traffic_speed_volume_occ_info.csv"
)
DETECTOR_OBS_URL = (
    "https://resource.data.one.gov.hk/td/traffic-detectors/rawSpeedVol-all.xml"
)
SPECIAL_NEWS_URL = "https://www.td.gov.hk/en/special_news/trafficnews.xml"
ROADWORKS_URL = (
    "https://resource.data.one.gov.hk/td/roadworks-location/get_all_the_roadworks.geojson"
)

# TD CCTV snapshots moved from cctv.td.gov.hk to tdcctv.data.one.gov.hk
# (verified 2026-08-07; the old host no longer resolves).
CCTV_CAMERAS: tuple[dict[str, str], ...] = (
    {
        "id": "K627F",
        "url": "https://tdcctv.data.one.gov.hk/K627F.JPG",
        "label": "K627F — Clear Water Bay Rd near Fei Ngo Shan Rd",
        "caption": "TD CCTV K627F: point view at Clear Water Bay Road near Fei Ngo Shan Road.",
    },
    {
        "id": "AID07117",
        "url": "https://tdcctv.data.one.gov.hk/AID07117.JPG",
        "label": "AID07117 — Lung Cheung Rd near Diamond Hill",
        "caption": "TD CCTV AID07117: point view at Lung Cheung Road near Diamond Hill.",
    },
)

# Refresh cadences.
OBS_TTL_SECONDS = 55.0
CCTV_TTL_SECONDS = 115.0
NEWS_TTL_SECONDS = 295.0
ROADWORKS_TTL_SECONDS = 15 * 60.0
META_TTL_SECONDS = 24 * 60 * 60.0

# --------------------------------------------------------------------------
# Corridor aliases (dashboard heuristics, not TD classifications)
# --------------------------------------------------------------------------

CORRIDOR_ALIASES: dict[str, tuple[str, ...]] = {
    "Clear Water Bay Road": ("clear water bay", "new clear water bay"),
    "Lung Cheung Road": ("lung cheung",),
    "Hiram's Highway": ("hiram",),
    "Sai Kung Road": ("sai kung road",),
    "Hang Hau Road": ("hang hau",),
    "Ying Yip Road": ("ying yip",),
    "Tai Po Tsai Road": ("tai po tsai",),
    "University Road": ("university road",),
    "Po Lam Road": ("po lam road",),
    "Chun Ying Street": ("chun ying",),
}

# Direction words that appear in TD detector descriptions ("Westbound",
# "Eastbound") and in special-news text.
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
                fields[desc_col].strip()
                if desc_col is not None and len(fields) > desc_col
                else ""
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
                fields[dir_col].strip()
                if dir_col is not None and len(fields) > dir_col
                else ""
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
    for detector in root.iter():
        if _local_name(detector.tag) != "detector":
            continue
        did = _local_text(detector, "detector_id")
        if not did:
            continue
        # capture time: the period_from of the ancestor <period>
        period_from = ""
        for node in root.iter():
            if detector in node.iter() and node is not detector and _local_name(node.tag) == "period":
                period_from = _local_text(node, "period_from")
                break
        capture = _build_capture_time(date_text, period_from)

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
# Corridor matching
# --------------------------------------------------------------------------

def match_corridors(text: str) -> list[str]:
    """Return canonical corridor names whose aliases appear in ``text``."""
    lowered = text.lower()
    matched = []
    for canonical, aliases in CORRIDOR_ALIASES.items():
        if any(alias in lowered for alias in aliases):
            matched.append(canonical)
    return matched


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
    max_observations: int = 6,
) -> list[TrafficCorridorStatus]:
    """Group detector observations by matched corridor and summarize."""
    groups: dict[str, list[TrafficObservation]] = {}
    for did, obs in observations.items():
        m = meta.get(did)
        if m is None:
            continue
        corridors = match_corridors(m.description)
        if not corridors:
            continue
        # a detector may serve multiple corridors; attach to the first match
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
    """Parse TD special traffic news v2; dedupe by identifier/status."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        log.warning("TD special news parse failed: %s", exc)
        return []

    ns = root.tag.split("}")[0].strip("{") if "}" in root.tag else ""
    nsmap = {"td": ns} if ns else {}

    incidents: list[TrafficIncident] = []
    for item in root.findall(".//td:item", nsmap) if nsmap else root.findall(".//item"):
        fields: dict[str, str] = {}
        for child in item:
            tag = child.tag.split("}")[-1]
            fields[tag] = (child.text or "").strip()
        identifier = fields.get("identifier") or fields.get("id") or ""
        title = fields.get("title") or ""
        description = fields.get("description") or fields.get("details") or ""
        location = fields.get("location") or ""
        direction = fields.get("direction") or ""
        status = fields.get("status") or ""
        if not (identifier or title):
            continue
        incidents.append(
            TrafficIncident(
                identifier=identifier,
                title=title,
                description=_sanitize_text(description),
                road=fields.get("road") or "",
                location=location,
                direction=direction,
                status=status,
                start_time=as_datetime(fields.get("start_time") or fields.get("effective_time")),
                end_time=as_datetime(fields.get("end_time") or fields.get("expiry_time")),
            )
        )
    return _dedupe_incidents(incidents)


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


def filter_relevant_incidents(
    incidents: list[TrafficIncident], limit: int = 3
) -> list[TrafficIncident]:
    """Keep only incidents whose text matches our corridors; dedupe already done."""
    relevant = [
        inc
        for inc in incidents
        if match_corridors(f"{inc.title} {inc.description} {inc.location} {inc.road}")
    ]
    return relevant[:limit]


# --------------------------------------------------------------------------
# Roadworks GeoJSON
# --------------------------------------------------------------------------

def parse_roadworks(geojson: dict[str, Any]) -> list[Roadwork]:
    """Parse TD roadworks GeoJSON; match against our corridors."""
    out: list[Roadwork] = []
    features = geojson.get("features") or []
    for feature in features:
        props = feature.get("properties") or {}
        description = " ".join(
            str(props.get(k) or "")
            for k in ("description", "name", "location", "road")
        )
        if not match_corridors(description):
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


# --------------------------------------------------------------------------
# CCTV
# --------------------------------------------------------------------------

def _is_jpeg(data: bytes) -> bool:
    return len(data) >= 3 and data[:3] == b"\xff\xd8\xff"


async def fetch_cctv_images(client: HttpClient) -> list[ImageAsset]:
    """Fetch the two representative TD CCTV JPEGs as replaceable attachments."""
    assets: list[ImageAsset] = []
    for camera in CCTV_CAMERAS:
        try:
            data = await client.fetch_bytes(
                camera["url"], max_bytes=4 * 1024 * 1024
            )
            if not _is_jpeg(data):
                log.warning("CCTV %s returned non-JPEG content", camera["id"])
                continue
            assets.append(
                ImageAsset(
                    filename=f"{camera['id']}.jpg",
                    data=data,
                    content_type="image/jpeg",
                    label=camera["label"],
                    caption=camera["caption"],
                    source_time=client.utcnow(),
                )
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("CCTV %s fetch failed: %s", camera["id"], exc)
    return assets


# --------------------------------------------------------------------------
# Public facade
# --------------------------------------------------------------------------

async def fetch_traffic_data(
    client: HttpClient,
) -> tuple[list[TrafficCorridorStatus], list[TrafficIncident], list[Roadwork], datetime | None]:
    """Fetch detectors, special news, and roadworks (metadata is cached daily).

    Returns (statuses, incidents, roadworks, capture_time).
    """
    # metadata: daily cache, tolerate failure
    meta: dict[str, _DetectorMeta] = {}
    try:
        meta_text = await client.fetch_text(DETECTOR_META_URL, max_bytes=2 * 1024 * 1024)
        meta = parse_detector_metadata(meta_text)
    except Exception as exc:  # noqa: BLE001
        log.warning("TD detector metadata fetch failed: %s", exc)

    statuses: list[TrafficCorridorStatus] = []
    capture_time: datetime | None = None
    try:
        obs_text = await client.fetch_xml_text(DETECTOR_OBS_URL)
        obs = parse_detector_observations(obs_text)
        statuses = build_corridor_statuses(obs, meta)
        capture_time = max(
            (o.capture_time for s in statuses for o in s.observations if o.capture_time),
            default=None,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("TD detector observations fetch failed: %s", exc)

    incidents: list[TrafficIncident] = []
    try:
        news_text = await client.fetch_xml_text(SPECIAL_NEWS_URL)
        incidents = filter_relevant_incidents(parse_special_news(news_text))
    except Exception as exc:  # noqa: BLE001
        log.warning("TD special news fetch failed: %s", exc)

    roadworks: list[Roadwork] = []
    try:
        rw = await client.fetch_json(ROADWORKS_URL)
        roadworks = parse_roadworks(rw)
    except Exception as exc:  # noqa: BLE001
        log.warning("TD roadworks fetch failed: %s", exc)

    return statuses, incidents, roadworks, capture_time
