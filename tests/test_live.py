"""Opt-in live smoke tests against public endpoints (not run in CI).

Run explicitly with:  python -m pytest -m live
"""

import aiohttp
import pytest

from dashboard.http import HttpClient
from dashboard.providers import transit
from dashboard.providers.traffic import (
    DETECTOR_META_URL,
    DETECTOR_OBS_URL,
    SPECIAL_NEWS_URL,
    parse_detector_metadata,
    parse_detector_observations,
    parse_special_news,
)
from dashboard.providers.weather import (
    RHRREAD_URL,
    WARNSUM_URL,
    fetch_weather_conditions,
)

pytestmark = pytest.mark.live


@pytest.mark.asyncio
async def test_live_transit_etas_parse():
    async with aiohttp.ClientSession() as session:
        client = HttpClient(session)
        groups, latest, failed = await transit.fetch_transit_etas(client)
        assert isinstance(groups, list)
        assert latest is not None
        assert isinstance(failed, list)


@pytest.mark.asyncio
async def test_live_hko_weather_parses():
    async with aiohttp.ClientSession() as session:
        client = HttpClient(session)
        snapshot, warnings, warn_time = await fetch_weather_conditions(client)
        # Sai Kung station should have a temperature
        assert snapshot is not None
        assert snapshot.temperature_c is not None


@pytest.mark.asyncio
async def test_live_td_detectors_parse():
    async with aiohttp.ClientSession() as session:
        client = HttpClient(session)
        meta_text = await client.fetch_text(DETECTOR_META_URL)
        meta = parse_detector_metadata(meta_text)
        assert meta, "detector metadata should parse"
        obs_text = await client.fetch_xml_text(DETECTOR_OBS_URL)
        obs = parse_detector_observations(obs_text)
        assert obs, "detector observations should parse"
        # at least one matched corridor has fresh data
        from dashboard.providers.traffic import build_corridor_statuses

        statuses = build_corridor_statuses(obs, meta)
        assert any(st.observations for st in statuses)


@pytest.mark.asyncio
async def test_live_td_special_news_parses():
    async with aiohttp.ClientSession() as session:
        client = HttpClient(session)
        text = await client.fetch_xml_text(SPECIAL_NEWS_URL)
        incidents = parse_special_news(text)
        assert isinstance(incidents, list)


@pytest.mark.asyncio
async def test_live_hko_warnsum_parses():
    async with aiohttp.ClientSession() as session:
        client = HttpClient(session)
        raw = await client.fetch_json(WARNSUM_URL)
        assert isinstance(raw, dict)
        # warnsum has no top-level updateTime; codes carry their own issueTime
        for code, payload in raw.items():
            assert isinstance(payload, dict)
            assert payload.get("code") == code


@pytest.mark.asyncio
async def test_live_hko_rhrread_parses():
    async with aiohttp.ClientSession() as session:
        client = HttpClient(session)
        raw = await client.fetch_json(RHRREAD_URL)
        assert "temperature" in raw
