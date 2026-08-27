"""HKO weather provider: observations (rhrread) plus active warning signals
(warnsum + warningInfo).

Warning identities, names, and icons come from HKO's live payloads so newly
introduced warning variants do not require a local display-name table.
"""

from __future__ import annotations

import asyncio
import io
import logging
from datetime import datetime
from typing import Any
from urllib.parse import urljoin, urlparse

from PIL import Image, UnidentifiedImageError

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

# wxwarntoday supplies both the authoritative display metadata and the icon
# path. Keep the host fixed while allowing only the source-provided path.
HKO_ORIGIN = "https://www.hko.gov.hk/"
WARNTODAY_URL = "https://www.hko.gov.hk/wxinfo/dailywx/wxwarntoday.json"
WARNTODAY_TTL_SECONDS = 5 * 60.0
_warning_icon_cache: dict[str, bytes] = {}
_WARNING_ICON_MAX_BYTES = 256 * 1024

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
    elif isinstance(details, list):
        candidates.extend(details)
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
            parts = [str(v) for v in candidate.values() if isinstance(v, str)]
            contents = candidate.get("contents")
            if isinstance(contents, list):
                parts.extend(
                    item if isinstance(item, str) else " ".join(
                        str(v) for v in item.values() if isinstance(v, str)
                    )
                    for item in contents
                    if isinstance(item, (str, dict))
                )
            text = " ".join(parts)
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


def _warning_metadata_from_warntoday(raw: dict[str, Any]) -> dict[str, tuple[str, str]]:
    """Build canonical-code -> (official name, official icon URL)."""
    metadata: dict[str, tuple[str, str]] = {}
    for entry in raw.get("WARNING_DATABASE") or []:
        if not isinstance(entry, dict):
            continue
        code = entry.get("WarningCode")
        source_name = str(entry.get("WarningName") or entry.get("warningName") or "").strip()
        warning_type = str(entry.get("Type") or entry.get("type") or "").strip()
        icon = str(entry.get("Icon") or entry.get("icon") or "").strip()
        if not code:
            continue
        name = _source_warning_name(warning_type, source_name, str(code))
        # Accept only an origin-relative path.  A protocol-relative value such
        # as ``//example.invalid/icon.gif`` would otherwise escape HKO through
        # ``urljoin``.
        icon_url = (
            urljoin(HKO_ORIGIN, icon)
            if icon.startswith("/") and not icon.startswith("//")
            else ""
        )
        metadata[str(code)] = (name, icon_url)
    return metadata


def _source_warning_name(warning_type: Any, source_name: Any, code: str) -> str:
    """Construct a display name from HKO's warning type/name fields."""
    warning_type = str(warning_type or "").strip()
    source_name = str(source_name or "").strip()
    if not warning_type:
        return source_name or code
    if warning_type.casefold() in {"amber", "red", "black", "yellow"}:
        return f"{warning_type} {source_name}".strip() or code
    if warning_type.casefold() in source_name.casefold():
        return source_name
    if any(word in warning_type.casefold() for word in ("signal", "warning")):
        return warning_type
    return f"{warning_type} {source_name}".strip() or code


def _icon_urls_from_warntoday(raw: dict[str, Any]) -> dict[str, str]:
    """Compatibility projection of source-driven warning metadata."""
    return {code: icon for code, (_, icon) in _warning_metadata_from_warntoday(raw).items()}


def _warning_icon_url(code: str, icon_map: dict[str, str] | None = None) -> str:
    if icon_map:
        return icon_map.get(code, "")
    return ""


def _warning_info_map(warning_info: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Normalize legacy mapping and current list-shaped warningInfo details."""
    if not isinstance(warning_info, dict):
        return {}
    details = warning_info.get("details", {})
    entries: list[tuple[str, dict[str, Any]]] = []
    if isinstance(details, dict):
        entries = [(str(key), value) for key, value in details.items() if isinstance(value, dict)]
    elif isinstance(details, list):
        for value in details:
            if not isinstance(value, dict):
                continue
            key = (
                value.get("warningStatementSubType")
                or value.get("subtype")
                or value.get("warningStatementCode")
            )
            if key:
                entries.append((str(key), value))
    result: dict[str, dict[str, Any]] = {}
    for key, value in entries:
        contents = value.get("contents")
        summary = value.get("summary") or value.get("statement") or ""
        if not summary and isinstance(contents, list):
            bits = []
            for item in contents:
                if isinstance(item, str):
                    bits.append(item)
                elif isinstance(item, dict):
                    bits.extend(str(v) for v in item.values() if isinstance(v, str))
            summary = " ".join(bits).strip()
        normalized = dict(value)
        normalized["summary"] = summary
        normalized["action"] = value.get("action") or ""
        code = str(value.get("warningStatementCode") or key)
        subtype = value.get("warningStatementSubType") or value.get("subtype")
        result[key] = normalized
        result.setdefault(code, normalized)
        if subtype:
            result[str(subtype)] = normalized
    return result


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
    warning_metadata: dict[str, tuple[str, str]] | None = None,
) -> list[WeatherWarning]:
    """Normalize warnsum + warningInfo into ordered WeatherWarning objects."""
    info_map = _warning_info_map(warning_info)

    active: list[WeatherWarning] = []
    # warnsum's outer key is a statement family; payload code is canonical.
    for family, payload in (warnsum or {}).items():
        if not isinstance(payload, dict) or not payload:
            continue
        code = str(payload.get("code") or family)
        metadata_name, source_icon = (warning_metadata or {}).get(code, ("", ""))
        live_name = _source_warning_name(
            payload.get("type") or payload.get("Type"),
            payload.get("name") or payload.get("Name"),
            code,
        )
        name = live_name if payload.get("type") or payload.get("name") else metadata_name or code
        info = info_map.get(code) or info_map.get(str(family), {})
        summary = info.get("summary") or ""
        action = info.get("action") or ""
        if isinstance(summary, str):
            stripped = summary.strip()
            if stripped.casefold().startswith(name.casefold()):
                summary = stripped[len(name) :].lstrip(" :\u2014-\u2013")
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
                icon_url=source_icon or _warning_icon_url(code, icon_map),
                issued_at=issued,
            )
        )
    # Preserve source order within the high-priority typhoon/rainstorm buckets.
    def sort_key(item: tuple[int, WeatherWarning]) -> tuple:
        index, w = item
        # Keep the most consequential families first so the renderer's single
        # primary thumbnail is normally a typhoon or rainstorm icon.
        family_rank = 0 if w.code.startswith("TC") else 1 if w.code.startswith("WRAIN") else 2
        return (family_rank, index)

    pre_no8 = _pre_no8_from_warning_info(warning_info)
    if pre_no8 is not None and all(w.code != PRE_NO8_CODE for w in active):
        active.append(pre_no8)
    active = [w for _, w in sorted(enumerate(active), key=sort_key)]
    return active


async def _fetch_warning_icons(
    client: HttpClient, warnings: list[WeatherWarning]
) -> None:
    """Attach cached official icon bytes for the renderer's composite image."""
    async def load(warning: WeatherWarning) -> None:
        parsed = urlparse(warning.icon_url)
        if parsed.scheme != "https" or parsed.hostname != "www.hko.gov.hk":
            return
        cached = _warning_icon_cache.get(warning.icon_url)
        if cached is not None:
            warning.icon_data = cached
            return
        try:
            data = await client.fetch_bytes(warning.icon_url, max_bytes=_WARNING_ICON_MAX_BYTES)
        except Exception as exc:  # noqa: BLE001
            log.warning("HKO warning icon fetch failed for %s: %s", warning.code, exc)
            return
        try:
            normalized = _normalize_warning_icon(data)
        except (OSError, UnidentifiedImageError, Image.DecompressionBombError):
            log.warning("HKO warning icon response was not a valid image for %s", warning.code)
            return
        if normalized is None:
            log.warning("HKO warning icon response had no usable frame for %s", warning.code)
            return
        _warning_icon_cache[warning.icon_url] = normalized
        warning.icon_data = normalized

    await asyncio.gather(*(load(warning) for warning in warnings))


def _normalize_warning_icon(data: bytes) -> bytes | None:
    """Return one deterministic, static PNG frame from an HKO icon.

    HKO publishes blinking GIFs whose blank frame must not become the
    dashboard's thumbnail.  Score every frame by the amount of visible
    foreground (alpha coverage and contrast against its corner background),
    retaining the earliest frame on ties for deterministic output.
    """
    with Image.open(io.BytesIO(data)) as source:
        best: tuple[int, int, Image.Image] | None = None
        frame_count = getattr(source, "n_frames", 1)
        for index in range(frame_count):
            source.seek(index)
            frame = source.convert("RGBA")
            pixels = frame.load()
            background = pixels[0, 0]
            foreground = 0
            for pixel in frame.getdata():
                if pixel[3] and (pixel[:3] != background[:3] or pixel[3] != background[3]):
                    foreground += 1
            # Alpha coverage breaks ties for flat-colour icons; foreground
            # contrast is the primary signal for blink/blank frames.
            score = foreground * 2 + sum(1 for pixel in frame.getdata() if pixel[3])
            candidate = (score, -index, frame.copy())
            if best is None or candidate[:2] > best[:2]:
                best = candidate
        if best is None:
            return None
        output = io.BytesIO()
        best[2].save(output, format="PNG", optimize=True)
    normalized = output.getvalue()
    return normalized if normalized and len(normalized) <= _WARNING_ICON_MAX_BYTES else None


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

    warning_metadata: dict[str, tuple[str, str]] = {}
    try:
        _, warntoday_raw, _ = await client.fetch_json_cached(warntoday_spec)
        warning_metadata = _warning_metadata_from_warntoday(warntoday_raw or {})
    except Exception as exc:  # noqa: BLE001
        log.warning("HKO wxwarntoday fetch failed (icons fall back to text): %s", exc)

    if warn_raw:
        warnings = parse_warnings(warn_raw, info_raw, warning_metadata=warning_metadata)
        await _fetch_warning_icons(client, warnings)

    return snapshot, warnings, warn_time
