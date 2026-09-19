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
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin, urlparse

from PIL import Image, UnidentifiedImageError

from dashboard.http import CachedFetch, FetchError, HttpClient, as_datetime, safe_endpoint
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

# wxwarntoday supplies authoritative display metadata; the warning details
# page supplies the static PNG catalog used for icons.
WARNTODAY_URL = "https://www.hko.gov.hk/wxinfo/dailywx/wxwarntoday.json"
WARNTODAY_TTL_SECONDS = 5 * 60.0
WARNING_DETAILS_URL = "https://www.hko.gov.hk/en/wservice/warning/details.htm"
WARNING_DETAILS_TTL_SECONDS = 24 * 60 * 60.0
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


def _normalize_warning_label(value: Any) -> str:
    return "".join(char for char in str(value or "").casefold() if char.isalnum())


def _is_static_warning_png_url(url: str) -> bool:
    parsed = urlparse(url)
    return (
        parsed.scheme == "https"
        and parsed.hostname == "www.hko.gov.hk"
        and not parsed.query
        and not parsed.fragment
        and parsed.path.startswith("/en/textonly/img/warn/images/")
        and parsed.path.casefold().endswith(".png")
    )


class _WarningCatalogParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.entries: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.casefold() != "img":
            return
        values = dict(attrs)
        alt, src = values.get("alt"), values.get("src")
        if not alt or not src:
            return
        url = urljoin(WARNING_DETAILS_URL, src)
        if _is_static_warning_png_url(url):
            self.entries.append((_normalize_warning_label(alt), url))


def _warning_catalog_from_html(html: str) -> dict[str, str]:
    """Return unambiguous normalized alt labels to official static PNG URLs."""
    parser = _WarningCatalogParser()
    parser.feed(html)
    grouped: dict[str, set[str]] = {}
    for label, url in parser.entries:
        if label:
            grouped.setdefault(label, set()).add(url)
    catalog = {
        label: next(iter(urls))
        for label, urls in grouped.items()
        if len(urls) == 1
    }
    return catalog


def _require_warning_catalog(html: str) -> None:
    if not _warning_catalog_from_html(html):
        raise FetchError("HKO warning details contained no usable static PNG catalog")


def _warning_static_icon(
    entry: dict[str, Any],
    catalog: dict[str, str] | None,
) -> str:
    if not catalog:
        return ""
    names = (
        entry.get("WarningName"),
        entry.get("warningName"),
        _source_warning_name(
            entry.get("Type") or entry.get("type"),
            entry.get("WarningName") or entry.get("warningName"),
            "",
        ),
    )
    for name in names:
        icon = catalog.get(_normalize_warning_label(name))
        if icon and _is_static_warning_png_url(icon):
            return icon
    return ""


def _warning_metadata_from_warntoday(
    raw: dict[str, Any],
    catalog: dict[str, str] | None = None,
) -> dict[str, tuple[str, str]]:
    """Build canonical-code -> (official name, official icon URL)."""
    if not isinstance(raw, dict) or not isinstance(raw.get("WARNING_DATABASE"), list):
        return {}
    metadata: dict[str, tuple[str, str]] = {}
    for entry in raw.get("WARNING_DATABASE") or []:
        if not isinstance(entry, dict):
            continue
        code = entry.get("WarningCode")
        source_name = str(
            entry.get("WarningName") or entry.get("warningName") or ""
        ).strip()
        warning_type = str(entry.get("Type") or entry.get("type") or "").strip()
        if not code:
            continue
        name = _source_warning_name(warning_type, source_name, str(code))
        icon_url = _warning_static_icon(entry, catalog)
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
        icon = icon_map.get(code)
        if icon is not None and _is_static_warning_png_url(icon):
            return icon
        # HKO's live warnsum code can identify a family while wxwarntoday
        # appends a subtype (for example WMSGNL_MONSOON).  Use that fallback
        # only when the metadata key is unambiguous; a guessed icon is worse
        # than no icon when several variants are published.
        candidates = [value for key, value in icon_map.items() if key.startswith(f"{code}_")]
        if len(candidates) == 1 and _is_static_warning_png_url(candidates[0]):
            return candidates[0]
    return ""


def _warning_metadata_entry(
    code: str, warning_metadata: dict[str, tuple[str, str]] | None
) -> tuple[str, str]:
    """Resolve exact HKO metadata, or one unambiguous code-subtype entry."""
    if not warning_metadata:
        return "", ""
    exact = warning_metadata.get(code)
    if exact is not None:
        return exact
    candidates = [
        value for key, value in warning_metadata.items() if key.startswith(f"{code}_")
    ]
    return candidates[0] if len(candidates) == 1 else ("", "")


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
    warning_catalog: dict[str, str] | None = None,
) -> list[WeatherWarning]:
    """Normalize warnsum + warningInfo into ordered WeatherWarning objects."""
    info_map = _warning_info_map(warning_info)

    active: list[WeatherWarning] = []
    # warnsum's outer key is a statement family; payload code is canonical.
    for family, payload in (warnsum or {}).items():
        if not isinstance(payload, dict) or not payload:
            continue
        code = str(payload.get("code") or family)
        metadata_name, source_icon = _warning_metadata_entry(code, warning_metadata)
        live_name = _source_warning_name(
            payload.get("type") or payload.get("Type"),
            payload.get("name") or payload.get("Name"),
            code,
        )
        name = live_name if payload.get("type") or payload.get("name") else metadata_name or code
        # warnsum also carries official names. An unavailable wxwarntoday
        # response must not disable icons present in the validated catalog.
        if not source_icon and warning_catalog:
            source_icon = warning_catalog.get(_normalize_warning_label(name), "")
            if not _is_static_warning_png_url(source_icon):
                source_icon = ""
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
        if (
            not _is_static_warning_png_url(warning.icon_url)
        ):
            log.warning(
                "HKO warning icon unavailable code=%s reason=no_static_png_metadata", warning.code,
            )
            return
        cached = _warning_icon_cache.get(warning.icon_url)
        if cached is not None:
            warning.icon_data = cached
            return
        try:
            data = await client.fetch_bytes(warning.icon_url, max_bytes=_WARNING_ICON_MAX_BYTES)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "HKO warning icon fetch failed code=%s endpoint=%s error=%s: %s",
                warning.code, safe_endpoint(warning.icon_url), type(exc).__name__, exc,
            )
            return
        try:
            normalized = _normalize_warning_icon(data)
        except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
            log.warning(
                "HKO warning icon rejected code=%s endpoint=%s bytes=%d error=%s",
                warning.code, safe_endpoint(warning.icon_url), len(data), type(exc).__name__,
            )
            return
        if normalized is None:
            log.warning(
                "HKO warning icon rejected code=%s endpoint=%s bytes=%d reason=no_usable_static_png",
                warning.code, safe_endpoint(warning.icon_url), len(data),
            )
            return
        _warning_icon_cache[warning.icon_url] = normalized
        warning.icon_data = normalized
        log.info(
            "HKO warning icon cached code=%s endpoint=%s bytes=%d",
            warning.code, safe_endpoint(warning.icon_url), len(normalized),
        )

    await asyncio.gather(*(load(warning) for warning in warnings))


def _normalize_warning_icon(data: bytes) -> bytes | None:
    """Validate and return one deterministic static PNG from the HKO catalog.

    The catalog is static PNG-only; reject animated or other source formats.
    """
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return None
    with Image.open(io.BytesIO(data)) as source:
        if getattr(source, "n_frames", 1) != 1:
            return None
        frame = source.convert("RGBA")
        output = io.BytesIO()
        frame.save(output, format="PNG", optimize=True)
    normalized = output.getvalue()
    return normalized if normalized and len(normalized) <= _WARNING_ICON_MAX_BYTES else None


async def fetch_weather_conditions(
    client: HttpClient,
    obs_spec: CachedFetch | None = None,
    warn_spec: CachedFetch | None = None,
    warn_info_spec: CachedFetch | None = None,
    warntoday_spec: CachedFetch | None = None,
    details_spec: CachedFetch | None = None,
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
    details_spec = details_spec or CachedFetch(
        WARNING_DETAILS_URL, WARNING_DETAILS_TTL_SECONDS, cache_key="warning-details"
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
    metadata_result, catalog_result = await asyncio.gather(
        client.fetch_json_cached(warntoday_spec),
        client.fetch_html_cached(details_spec, validator=_require_warning_catalog),
        return_exceptions=True,
    )
    warntoday_raw: dict[str, Any] = {}
    if isinstance(metadata_result, Exception):
        log.warning("HKO wxwarntoday fetch failed (using warnsum names for icons): %s", metadata_result)
    else:
        _, warntoday_raw, _ = metadata_result
    catalog: dict[str, str] = {}
    if isinstance(catalog_result, Exception):
        log.warning("HKO warning details fetch failed (icons unavailable): %s", catalog_result)
    else:
        _, details_html, _ = catalog_result
        catalog = _warning_catalog_from_html(details_html)
    warning_metadata = _warning_metadata_from_warntoday(warntoday_raw, catalog)

    if warn_raw:
        warnings = parse_warnings(
            warn_raw, info_raw, warning_metadata=warning_metadata, warning_catalog=catalog
        )
        await _fetch_warning_icons(client, warnings)

    return snapshot, warnings, warn_time
