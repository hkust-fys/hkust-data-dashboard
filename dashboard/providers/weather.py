"""HKO weather provider: observations (rhrread) plus active warning signals
(warnsum + warningInfo).

Warning codes are normalized to a fixed, ordered list so the renderer can show
signals before ordinary conditions and tolerate empty/null/mismatched shapes.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from dashboard.http import CachedFetch, HttpClient, as_datetime
from dashboard.models import WeatherSnapshot, WeatherWarning

log = logging.getLogger(__name__)

RHRREAD_URL = (
    "https://data.weather.gov.hk/weatherAPI/opendata/weather.php"
    "?dataType=rhrread&lang=en"
)
WARNSUM_URL = (
    "https://data.weather.gov.hk/weatherAPI/opendata/weather.php"
    "?dataType=warnsum&lang=en"
)
WARNING_INFO_URL = (
    "https://data.weather.gov.hk/weatherAPI/opendata/weather.php"
    "?dataType=warningInfo&lang=en"
)

OBS_TTL_SECONDS = 600.0  # 10 minutes
WARN_TTL_SECONDS = 60.0  # 1 minute
WARN_INFO_TTL_SECONDS = 300.0

# Fixed, ordered list of warning codes the dashboard understands. Codes outside
# this list are still shown (as-is) but render after the known ones.
# Codes match the HKO warnsum payload keys (e.g. WHOT, TC, RAIN, THUNDER...).
KNOWN_WARNING_CODES: tuple[str, ...] = (
    "TC",      # tropical cyclone signals
    "RAIN",    # rainstorm
    "THUNDER", # thunderstorm
    "LANDSLIP",
    "FIRE",
    "WHOT",    # very hot weather warning
    "COLD",
    "MONSOON",
    "FROST",
    "TSUNAMI",
)

# Friendly display names for the codes we know.
WARNING_NAMES: dict[str, str] = {
    "TC": "Typhoon",
    "TC1": "Standby Signal No. 1",
    "TC3": "Strong Wind Signal No. 3",
    "TC8NE": "Gale Signal No. 8 NE",
    "TC8NW": "Gale Signal No. 8 NW",
    "TC8SE": "Gale Signal No. 8 SE",
    "TC8SW": "Gale Signal No. 8 SW",
    "TC9": "Gale Signal No. 9",
    "TC10": "Hurricane Signal No. 10",
    "TC8PRE": "Pre-No. 8 Special Announcement",
    "RAIN": "Rainstorm",
    "WRAINA": "Amber Rainstorm",
    "WRAINR": "Red Rainstorm",
    "WRAINB": "Black Rainstorm",
    "THUNDER": "Thunderstorm",
    "WTS": "Thunderstorm Warning",
    "LANDSLIP": "Landslip",
    "WL": "Landslip Warning",
    "FIRE": "Fire Danger",
    "WFIRER": "Red Fire Danger",
    "WFIREY": "Yellow Fire Danger",
    "WHOT": "Very Hot Weather",
    "COLD": "Cold Weather",
    "WCOLD": "Cold Weather Warning",
    "MONSOON": "Strong Monsoon",
    "WMSGNL": "Strong Monsoon Signal",
    "FROST": "Frost",
    "WFROST": "Frost Warning",
    "TSUNAMI": "Tsunami",
    "WTM": "Tsunami Warning",
}

# HKO official warning-icon base. The wxwarntoday.json "Icon" field gives
# "/images_e/<name>.gif"; active warnings use the "images/<name>.issuing.gif"
# variant. We fetch that JSON rather than hardcoding the mapping.
HKO_ICON_BASE = "https://www.hko.gov.hk/en/wxinfo/dailywx/images/{name}.issuing.gif"
WARNTODAY_URL = "https://www.hko.gov.hk/wxinfo/dailywx/wxwarntoday.json"
WARNTODAY_TTL_SECONDS = 5 * 60.0

# The Pre-No. 8 Special Announcement is HKO's ~2-hour advance notice before
# Tropical Cyclone Warning Signal No. 8. It appears as a statement in the
# warningInfo feed rather than a warnsum code, so it is surfaced as a synthetic
# warning code for the alert monitor's edge-triggered diffing.
PRE_NO8_CODE = "TC8PRE"
_PRE_NO8_PHRASES: tuple[str, ...] = (
    "pre-no. 8 special announcement",
    "pre-no 8 special announcement",
    "pre-no.8 special announcement",
    "pre-no 8",
)
_PRE_NO8_NAME = "Pre-No. 8 Special Announcement"


def _pre_no8_from_warning_info(warning_info: dict[str, Any] | None) -> WeatherWarning | None:
    """Detect the Pre-No. 8 statement in the warningInfo payload."""
    if not isinstance(warning_info, dict):
        return None
    details = warning_info.get("details")
    candidates: list[Any] = []
    if isinstance(details, dict):
        candidates.extend(details.values())
    for key in ("statement", "statements", "desc", "description"):
        value = warning_info.get(key)
        if isinstance(value, str):
            candidates.append(value)
        elif isinstance(value, list):
            candidates.extend(value)
    for candidate in candidates:
        if not isinstance(candidate, (str, dict)):
            continue
        if isinstance(candidate, str):
            text = candidate
        else:
            text = " ".join(str(v) for v in candidate.values() if isinstance(v, str))
        lowered = text.lower()
        if any(phrase in lowered for phrase in _PRE_NO8_PHRASES):
            issued = None
            if isinstance(candidate, dict):
                issued = as_datetime(candidate.get("issueTime")) or as_datetime(
                    candidate.get("updateTime")
                )
            return WeatherWarning(
                code=PRE_NO8_CODE,
                name=_PRE_NO8_NAME,
                summary=text[:200],
                icon_url="",
                issued_at=issued,
            )
    return None


def _icon_urls_from_warntoday(raw: dict[str, Any]) -> dict[str, str]:
    """Build {warning_code: active-icon-url} from the wxwarntoday JSON.

    The JSON lists currently-active warnings with an Icon field like
    "/images_e/vhot.gif"; the page JS rewrites that to the ".issuing.gif"
    variant under "images/".
    """
    urls: dict[str, str] = {}
    for entry in raw.get("WARNING_DATABASE") or []:
        if not isinstance(entry, dict):
            continue
        code = entry.get("WarningCode")
        icon = entry.get("Icon") or ""
        name = icon.rsplit("/", 1)[-1].removesuffix(".gif")
        if code and name:
            urls[code] = HKO_ICON_BASE.format(name=name)
    return urls


def _warning_icon_url(code: str, icon_map: dict[str, str] | None = None) -> str:
    if icon_map:
        return icon_map.get(code, "")
    return ""


def _latest_warning_time(warnsum: dict[str, Any]) -> datetime | None:
    """warnsum has no top-level updateTime; take the newest per-code time."""
    latest: datetime | None = None
    for payload in warnsum.values():
        if not isinstance(payload, dict):
            continue
        for key in ("updateTime", "issueTime"):
            ts = as_datetime(payload.get(key))
            if ts and (latest is None or ts > latest):
                latest = ts
    return latest


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _find_station_value(
    data: list[dict[str, Any]] | None, place: str, key: str = "value"
) -> float | None:
    """Find the first entry matching ``place``; HKO sometimes uses 'station'."""
    for entry in data or []:
        if not isinstance(entry, dict):
            continue
        name = entry.get("place") or entry.get("station") or ""
        if name == place:
            return _safe_float(entry.get(key))
    return None


def parse_observations(raw: dict[str, Any]) -> WeatherSnapshot:
    """Parse rhrread; tolerate missing sections."""
    update = as_datetime(raw.get("updateTime"))
    temp = _find_station_value(raw.get("temperature", {}).get("data"), "Sai Kung")
    rain = _find_station_value(raw.get("rainfall", {}).get("data"), "Sai Kung", "max")
    humidity = _safe_int(_find_station_value(raw.get("humidity", {}).get("data"), "Sai Kung"))
    return WeatherSnapshot(
        temperature_c=temp,
        rainfall_mm=rain,
        humidity_pct=humidity,
        station="Sai Kung",
        source_time=update,
    )


def parse_warnings(
    warnsum: dict[str, Any],
    warning_info: dict[str, Any] | None = None,
    icon_map: dict[str, str] | None = None,
) -> list[WeatherWarning]:
    """Normalize warnsum + warningInfo into ordered WeatherWarning objects."""
    info_map: dict[str, dict[str, Any]] = {}
    if isinstance(warning_info, dict):
        details = warning_info.get("details", {})
        if isinstance(details, dict):
            info_map = {
                code: (entry or {}) for code, entry in details.items() if isinstance(entry, dict)
            }

    active: list[WeatherWarning] = []
    raw_codes: set[str] = set()

    # warnsum shape: {"TC": {...}, "RAIN": {...}, ...} with issueDateTime etc.
    for code, payload in (warnsum or {}).items():
        if not isinstance(payload, dict) or not payload:
            continue
        name = WARNING_NAMES.get(code, code)
        info = info_map.get(code, {})
        summary = info.get("summary") or ""
        action = info.get("action") or ""
        # Keep the warning's original issue time distinct from later reissue
        # or provider-update timestamps shown in the embed metadata.
        issued = None
        for key in ("issueDateTime", "issueTime", "updateTime"):
            issued = as_datetime(payload.get(key))
            if issued:
                break
        active.append(
            WeatherWarning(
                code=code,
                name=name,
                summary=summary if isinstance(summary, str) else "",
                action=action if isinstance(action, str) else "",
                icon_url=_warning_icon_url(code, icon_map),
                issued_at=issued,
            )
        )
        raw_codes.add(code)

    # warningInfo may contain codes not present in warnsum (e.g. when the
    # warning has just been cancelled); only add ones warnsum missed.
    for code, payload in info_map.items():
        if code in raw_codes:
            continue
        name = WARNING_NAMES.get(code, code)
        issued = as_datetime(payload.get("issueTime")) or as_datetime(payload.get("updateTime"))
        active.append(
            WeatherWarning(
                code=code,
                name=name,
                summary=(payload.get("summary") or ""),
                action=(payload.get("action") or ""),
                icon_url=_warning_icon_url(code, icon_map),
                issued_at=issued,
            )
        )

    # Order: known codes first (in declared order), then any extra codes.
    def sort_key(w: WeatherWarning) -> tuple:
        known = KNOWN_WARNING_CODES.index(w.code) if w.code in KNOWN_WARNING_CODES else len(
            KNOWN_WARNING_CODES
        )
        return (known, w.code)

    active = sorted(active, key=sort_key)
    pre_no8 = _pre_no8_from_warning_info(warning_info)
    if pre_no8 is not None and all(w.code != PRE_NO8_CODE for w in active):
        active.append(pre_no8)
    return active


async def fetch_weather_conditions(
    client: HttpClient,
    obs_spec: CachedFetch | None = None,
    warn_spec: CachedFetch | None = None,
    warn_info_spec: CachedFetch | None = None,
    warntoday_spec: CachedFetch | None = None,
) -> tuple[WeatherSnapshot | None, list[WeatherWarning], datetime | None]:
    """Fetch observations and warnings concurrently (via the shared cache).

    Returns (snapshot, warnings, warning_source_time). Each failing source is
    skipped rather than failing the whole call.
    """
    obs_spec = obs_spec or CachedFetch(RHRREAD_URL, OBS_TTL_SECONDS, cache_key="rhrread")
    warn_spec = warn_spec or CachedFetch(WARNSUM_URL, WARN_TTL_SECONDS, cache_key="warnsum")
    warn_info_spec = warn_info_spec or CachedFetch(
        WARNING_INFO_URL, WARN_INFO_TTL_SECONDS, cache_key="warningInfo"
    )
    warntoday_spec = warntoday_spec or CachedFetch(
        WARNTODAY_URL, WARNTODAY_TTL_SECONDS, cache_key="warntoday"
    )

    snapshot: WeatherSnapshot | None = None
    warnings: list[WeatherWarning] = []
    warn_time: datetime | None = None

    try:
        _, obs_raw, _ = await client.fetch_json_cached(obs_spec)
        if obs_raw is not None:
            snapshot = parse_observations(obs_raw)
    except Exception as exc:  # noqa: BLE001
        log.warning("HKO observations fetch failed: %s", exc)

    try:
        _, warn_raw, _ = await client.fetch_json_cached(warn_spec)
    except Exception as exc:  # noqa: BLE001
        log.warning("HKO warnsum fetch failed: %s", exc)
        warn_raw = None

    if warn_raw:
        # warnsum has no top-level updateTime; take the latest per-code time.
        warn_time = _latest_warning_time(warn_raw)

    info_raw: dict[str, Any] | None = None
    if warn_raw:
        try:
            _, info_raw, _ = await client.fetch_json_cached(warn_info_spec)
        except Exception as exc:  # noqa: BLE001
            log.warning("HKO warningInfo fetch failed: %s", exc)

    icon_map: dict[str, str] = {}
    try:
        _, warntoday_raw, _ = await client.fetch_json_cached(warntoday_spec)
        icon_map = _icon_urls_from_warntoday(warntoday_raw or {})
    except Exception as exc:  # noqa: BLE001
        log.warning("HKO wxwarntoday fetch failed (icons fall back to text): %s", exc)

    if warn_raw:
        warnings = parse_warnings(warn_raw, info_raw, icon_map)

    return snapshot, warnings, warn_time
