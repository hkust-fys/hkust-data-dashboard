"""Official bilingual Hong Kong road names from LandsD Road Centreline.

The provider downloads only the distinct English/Traditional-Chinese name
pairs from the public CSDI FeatureServer.  It is independent of route geometry
and Overpass so a failure in either source cannot remove traffic-news aliases.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import time
import unicodedata
from dataclasses import dataclass
from urllib.parse import urlencode

from dashboard.http import HttpClient

log = logging.getLogger(__name__)

ROAD_NAMES_URL = (
    "https://portal.csdi.gov.hk/server/rest/services/common/"
    "landsd_rcd_1637310758814_80061/FeatureServer/0/query"
)
ROAD_NAMES_CACHE_NAME = "road-names.json"
ROAD_NAMES_CACHE_VERSION = 1
ROAD_NAMES_TTL_SECONDS = 24 * 3600.0
ROAD_NAMES_STALE_SECONDS = 90 * 24 * 3600.0
ROAD_NAMES_FAILURE_COOLDOWN_SECONDS = 30 * 60.0
ROAD_NAMES_PAGE_SIZE = 3000
ROAD_NAMES_MAX_PAGES = 4
ROAD_NAMES_MAX_BYTES = 2 * 1024 * 1024
ROAD_NAMES_MAX_ENTRIES = ROAD_NAMES_PAGE_SIZE * ROAD_NAMES_MAX_PAGES
ROAD_NAMES_MAX_ALIASES = 16
ROAD_NAMES_MAX_LENGTH = 255

FALLBACK_ROAD_NAMES: dict[str, tuple[str, ...]] = {
    "clear water bay road": ("清水灣道",),
    "new clear water bay road": ("新清水灣道",),
}

_WHITESPACE = re.compile(r"\s+")
_APOSTROPHES = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u02bc": "'",
        "\uff07": "'",
    }
)


def _clean_name(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).translate(_APOSTROPHES)
    return _WHITESPACE.sub(" ", normalized).strip()


def normalize_name(text: str) -> str:
    """Return the exact-match key shared by OSM and official English names."""

    return _clean_name(str(text)).casefold()


@dataclass(frozen=True)
class _CacheRecord:
    names: dict[str, tuple[str, ...]]
    fetched_at: float


_memory_last_good: dict[str, _CacheRecord] = {}
_refresh_tasks: dict[str, asyncio.Task[_CacheRecord]] = {}
_refresh_retry_after: dict[str, float] = {}
_startup_attempted: set[str] = set()
_refresh_shutdown = False


def _cache_file(cache_dir: str) -> str:
    return os.path.join(cache_dir, "maps", ROAD_NAMES_CACHE_NAME)


def _record_age(record: _CacheRecord) -> float:
    return max(0.0, time.time() - record.fetched_at)


def _valid_pair(english: object, chinese: object) -> tuple[str, str] | None:
    if not isinstance(english, str) or not isinstance(chinese, str):
        return None
    english_key = normalize_name(english)
    chinese_name = _clean_name(chinese)
    if (
        not english_key
        or not chinese_name
        or english_key == "-99"
        or chinese_name == "-99"
        or len(english_key) > ROAD_NAMES_MAX_LENGTH
        or len(chinese_name) > ROAD_NAMES_MAX_LENGTH
    ):
        return None
    return english_key, chinese_name


def _validate_names(raw_names: object) -> dict[str, tuple[str, ...]] | None:
    if not isinstance(raw_names, dict) or not raw_names:
        return None
    if len(raw_names) > ROAD_NAMES_MAX_ENTRIES:
        return None
    names: dict[str, tuple[str, ...]] = {}
    for raw_english, raw_chinese in raw_names.items():
        if not isinstance(raw_english, str) or not isinstance(raw_chinese, list):
            return None
        english_key = normalize_name(raw_english)
        if not english_key or english_key != raw_english or not raw_chinese:
            return None
        if len(raw_chinese) > ROAD_NAMES_MAX_ALIASES:
            return None
        chinese_names: list[str] = []
        for value in raw_chinese:
            pair = _valid_pair(english_key, value)
            if pair is None:
                return None
            chinese_name = pair[1]
            if chinese_name not in chinese_names:
                chinese_names.append(chinese_name)
        if not chinese_names:
            return None
        names[english_key] = tuple(sorted(chinese_names))
    return names or None


def _load_disk_cache(cache_dir: str) -> _CacheRecord | None:
    memory = _memory_last_good.get(cache_dir)
    if memory is not None:
        if _record_age(memory) <= ROAD_NAMES_STALE_SECONDS:
            return memory
        _memory_last_good.pop(cache_dir, None)

    try:
        with open(_cache_file(cache_dir), encoding="utf-8") as file:
            raw = json.load(file)
        if not isinstance(raw, dict) or raw.get("version") != ROAD_NAMES_CACHE_VERSION:
            return None
        fetched_at = float(raw.get("fetched_at"))
        if not math.isfinite(fetched_at) or fetched_at <= 0:
            return None
        record = _CacheRecord(
            names=_validate_names(raw.get("names")) or {},
            fetched_at=fetched_at,
        )
        if not record.names or _record_age(record) > ROAD_NAMES_STALE_SECONDS:
            return None
        _memory_last_good[cache_dir] = record
        return record
    except (OSError, ValueError, TypeError):
        return None


def _save_disk_cache(record: _CacheRecord, cache_dir: str) -> None:
    if not record.names:
        return
    payload = {
        "version": ROAD_NAMES_CACHE_VERSION,
        "fetched_at": record.fetched_at,
        "names": {key: list(values) for key, values in record.names.items()},
    }
    try:
        path = _cache_file(cache_dir)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        temporary = f"{path}.tmp"
        with open(temporary, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, separators=(",", ":"))
        os.replace(temporary, path)
        _memory_last_good[cache_dir] = record
    except OSError as exc:
        # The fetched table remains usable in memory even if persistence fails.
        _memory_last_good[cache_dir] = record
        log.warning("road-name cache write failed: %s", exc)


def _page_url(offset: int) -> str:
    query = urlencode(
        {
            "f": "json",
            "where": (
                "ENGLISHSTREETNAME IS NOT NULL AND "
                "CHINESESTREETNAME IS NOT NULL"
            ),
            "outFields": "ENGLISHSTREETNAME,CHINESESTREETNAME",
            "returnGeometry": "false",
            "returnDistinctValues": "true",
            "orderByFields": "ENGLISHSTREETNAME,CHINESESTREETNAME",
            "resultRecordCount": str(ROAD_NAMES_PAGE_SIZE),
            "resultOffset": str(offset),
        }
    )
    return f"{ROAD_NAMES_URL}?{query}"


def _parse_page(raw: object) -> tuple[list[tuple[str, str]], bool]:
    if not isinstance(raw, dict) or raw.get("error") is not None:
        raise ValueError("invalid CSDI road-name response")
    features = raw.get("features")
    if not isinstance(features, list) or len(features) > ROAD_NAMES_PAGE_SIZE:
        raise ValueError("invalid CSDI road-name feature list")
    pairs: list[tuple[str, str]] = []
    for feature in features:
        if not isinstance(feature, dict) or not isinstance(feature.get("attributes"), dict):
            raise ValueError("invalid CSDI road-name feature")
        attributes = feature["attributes"]
        if "ENGLISHSTREETNAME" not in attributes or "CHINESESTREETNAME" not in attributes:
            raise ValueError("missing CSDI road-name fields")
        pair = _valid_pair(
            attributes["ENGLISHSTREETNAME"],
            attributes["CHINESESTREETNAME"],
        )
        if pair is not None:
            pairs.append(pair)
    exceeded = raw.get("exceededTransferLimit", False)
    if not isinstance(exceeded, bool):
        raise ValueError("invalid CSDI pagination marker")
    if exceeded and not features:
        raise ValueError("empty partial CSDI road-name page")
    return pairs, exceeded


async def _download_names(client: HttpClient) -> dict[str, tuple[str, ...]]:
    aliases: dict[str, set[str]] = {}
    offset = 0
    complete = False
    for _page_number in range(ROAD_NAMES_MAX_PAGES):
        raw = await client.fetch_json(
            _page_url(offset),
            headers=None,
            max_bytes=ROAD_NAMES_MAX_BYTES,
        )
        pairs, exceeded = _parse_page(raw)
        for english_key, chinese_name in pairs:
            aliases.setdefault(english_key, set()).add(chinese_name)
        features = raw["features"]
        offset += len(features)
        if not exceeded:
            complete = True
            break
    if not complete or not aliases:
        raise ValueError("incomplete CSDI road-name dataset")
    return {key: tuple(sorted(values)) for key, values in sorted(aliases.items())}


async def _refresh_names(client: HttpClient, cache_dir: str) -> _CacheRecord:
    names = await _download_names(client)
    record = _CacheRecord(names=names, fetched_at=time.time())
    _save_disk_cache(record, cache_dir)
    _refresh_retry_after.pop(cache_dir, None)
    return record


def _finish_refresh(task: asyncio.Task[_CacheRecord], cache_dir: str) -> None:
    _refresh_tasks.pop(cache_dir, None)
    if _refresh_shutdown or task.cancelled():
        return
    try:
        task.result()
    except Exception as exc:  # noqa: BLE001
        _refresh_retry_after[cache_dir] = (
            time.monotonic() + ROAD_NAMES_FAILURE_COOLDOWN_SECONDS
        )
        log.warning("road-name refresh failed: %s", type(exc).__name__)


def _start_refresh(client: HttpClient, cache_dir: str) -> asyncio.Task[_CacheRecord]:
    task = _refresh_tasks.get(cache_dir)
    if task is None:
        task = asyncio.create_task(_refresh_names(client, cache_dir))
        _refresh_tasks[cache_dir] = task
        task.add_done_callback(
            lambda done, cache_dir=cache_dir: _finish_refresh(done, cache_dir)
        )
    return task


def _fallback() -> dict[str, tuple[str, ...]]:
    return dict(FALLBACK_ROAD_NAMES)


async def fetch_road_names(
    client: HttpClient,
    *,
    cache_dir: str,
    wait_for_refresh: bool = False,
) -> dict[str, tuple[str, ...]]:
    """Return fresh or retained official road names and refresh independently.

    The default never waits for an expired or missing table.  Callers that
    explicitly opt in may await the shared refresh, primarily for startup and
    live-source checks.
    """

    global _refresh_shutdown
    _refresh_shutdown = False
    cached = _load_disk_cache(cache_dir)
    task = _refresh_tasks.get(cache_dir)
    first_process_fetch = cache_dir not in _startup_attempted
    if first_process_fetch:
        _startup_attempted.add(cache_dir)
    retry_after = _refresh_retry_after.get(cache_dir)
    if task is None and not first_process_fetch and retry_after is not None:
        if time.monotonic() < retry_after:
            return dict(cached.names) if cached is not None else _fallback()
        task = _start_refresh(client, cache_dir)
    elif task is None and (
        first_process_fetch
        or cached is None
        or _record_age(cached) > ROAD_NAMES_TTL_SECONDS
    ):
        task = _start_refresh(client, cache_dir)

    if task is None:
        return dict(cached.names) if cached is not None else _fallback()
    if not wait_for_refresh:
        return dict(cached.names) if cached is not None else _fallback()
    try:
        refreshed = await asyncio.shield(task)
        return dict(refreshed.names)
    except Exception as exc:  # noqa: BLE001
        _refresh_retry_after[cache_dir] = (
            time.monotonic() + ROAD_NAMES_FAILURE_COOLDOWN_SECONDS
        )
        log.warning("road-name refresh failed: %s", type(exc).__name__)
        return dict(cached.names) if cached is not None else _fallback()


async def shutdown_background_refreshes() -> None:
    """Cancel and drain refreshes before the shared HTTP session is closed."""

    global _refresh_shutdown
    _refresh_shutdown = True
    tasks = list(_refresh_tasks.values())
    for task in tasks:
        if not task.done():
            task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    _refresh_tasks.clear()
    _refresh_retry_after.clear()
    _memory_last_good.clear()
    _startup_attempted.clear()


__all__ = ["fetch_road_names", "normalize_name", "shutdown_background_refreshes"]
