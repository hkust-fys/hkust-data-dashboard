"""Traffic provider tests: parsing, corridor matching, and source caching."""

from datetime import UTC, datetime

import pytest

from dashboard.models import SpeedBand
from dashboard.providers.traffic import (
    DETECTOR_META_SPEC,
    DETECTOR_META_URL,
    DETECTOR_OBS_SPEC,
    DETECTOR_OBS_URL,
    ROADWORKS_SPEC,
    ROADWORKS_URL,
    SPECIAL_NEWS_SPEC,
    SPECIAL_NEWS_URL,
    _sanitize_text,
    build_corridor_statuses,
    fetch_traffic_data,
    filter_relevant_incidents,
    match_corridors,
    parse_detector_metadata,
    parse_detector_observations,
    parse_roadworks,
    parse_special_news,
    speed_band,
)
from tests.fixtures import sample_data as s


def test_parse_detector_metadata_handles_aliased_headers():
    # live CSV uses AID_ID_Number; also accept legacy detector_id header
    meta = parse_detector_metadata(s.DETECTOR_CSV)
    assert "AID1001" in meta
    assert meta["AID1001"].description == "Clear Water Bay Road near Fei Ngo Shan Road - Eastbound"
    assert meta["AID1001"].latitude == 22.337
    assert meta["AID1001"].direction == "East"

    legacy = s.DETECTOR_CSV.replace("AID_ID_Number,District", "detector_id,district")
    meta_legacy = parse_detector_metadata(legacy)
    assert "AID1001" in meta_legacy


def test_parse_detector_metadata_strips_bom():
    meta = parse_detector_metadata("\ufeff" + s.DETECTOR_CSV)
    assert "AID1001" in meta


def test_parse_detector_metadata_missing_id_column():
    assert parse_detector_metadata("foo,bar\n1,2\n") == {}
    assert parse_detector_metadata("") == {}


def test_parse_detector_observations():
    obs = parse_detector_observations(s.DETECTOR_XML)
    # AID1001: two lanes, avg speed (15+18)/2 = 16.5, sum volume 18, avg occupancy 42.5
    assert obs["AID1001"]["speed"] == 16.5
    assert obs["AID1001"]["volume"] == 18
    assert obs["AID1001"]["occupancy"] == 42.5
    assert obs["AID1001"]["capture_time"] is not None
    assert "AID1004" in obs


def test_parse_detector_observations_skips_invalid_lanes():
    xml = s.DETECTOR_XML.replace("<valid>Y</valid>", "<valid>N</valid>", 1)
    obs = parse_detector_observations(xml)
    assert "AID1001" in obs  # still has the second valid lane


def test_parse_detector_observations_bad_xml():
    assert parse_detector_observations("not xml at all") == {}


def test_match_corridors_aliases():
    assert match_corridors("Clear Water Bay Road accident") == ["Clear Water Bay Road"]
    assert match_corridors("New Clear Water Bay Road works") == ["Clear Water Bay Road"]
    assert match_corridors("Lung Cheung Road") == ["Lung Cheung Road"]
    assert match_corridors("Hiram's Highway") == ["Hiram's Highway"]
    assert match_corridors("Po Lam Road closure") == ["Po Lam Road"]
    assert match_corridors("Nathan Road") == []


def test_speed_bands():
    assert speed_band(10) == SpeedBand.RED
    assert speed_band(20) == SpeedBand.AMBER
    assert speed_band(40) == SpeedBand.AMBER
    assert speed_band(41) == SpeedBand.GREEN
    assert speed_band(None) == SpeedBand.GRAY
    assert speed_band(10, stale=True) == SpeedBand.GRAY


def test_build_corridor_statuses_groups_and_orders():
    meta = parse_detector_metadata(s.DETECTOR_CSV)
    obs = parse_detector_observations(s.DETECTOR_XML)
    statuses = build_corridor_statuses(obs, meta)
    names = [st.name for st in statuses]
    assert "Clear Water Bay Road" in names
    assert "Lung Cheung Road" in names
    # unrelated road excluded
    assert "Nathan Road" not in names
    cwb = [st for st in statuses if st.name == "Clear Water Bay Road"][0]
    assert cwb.observations[0].band == SpeedBand.RED
    assert cwb.direction != ""  # from the "Eastbound" description hint


def test_build_corridor_statuses_empty():
    assert build_corridor_statuses({}, {}) == []


def test_parse_special_news_and_filter_relevant():
    incidents = parse_special_news(s.SPECIAL_NEWS_XML)
    assert len(incidents) == 3
    relevant = filter_relevant_incidents(incidents)
    assert [i.identifier for i in relevant] == ["TN-1", "TN-2"]
    assert len(relevant) <= 3


def test_parse_special_news_dedupes():
    doubled = s.SPECIAL_NEWS_XML.replace("</trafficNews>", s.SPECIAL_NEWS_XML.split("<trafficNews>")[-1])
    incidents = parse_special_news(doubled)
    ids = [i.identifier for i in incidents]
    assert len(ids) == len(set(ids))


def test_parse_special_news_bad_xml():
    assert parse_special_news("garbage") == []


def test_sanitize_text():
    assert _sanitize_text("a\x00b\n c ") == "a b c"


def test_parse_roadworks_matches_corridors():
    rw = parse_roadworks(s.ROADWORKS_JSON)
    assert len(rw) == 1
    assert rw[0].identifier == "RW-1"
    assert "Hang Hau Road" in rw[0].description


def test_jpeg_validation():
    # bus-stop frame validation uses the same JPEG sniff
    from dashboard.providers.cameras import _is_jpeg

    assert _is_jpeg(s.jpeg_bytes())
    assert not _is_jpeg(b"not a jpeg")
    assert not _is_jpeg(b"")


def _traffic_client(monkeypatch):
    from dashboard.http import FetchError, HttpClient

    client = HttpClient(object(), retry_attempts=1)
    calls = {url: 0 for url in (
        DETECTOR_META_URL,
        DETECTOR_OBS_URL,
        SPECIAL_NEWS_URL,
        ROADWORKS_URL,
    )}
    state = {"fail": False}

    def record(url):
        calls[url] += 1
        if state["fail"]:
            raise FetchError("offline")

    async def fetch_text(url, _headers=None, _max_bytes=None):
        record(url)
        return s.DETECTOR_CSV

    async def fetch_xml_text(url, _headers=None, _max_bytes=None):
        record(url)
        return s.DETECTOR_XML if url == DETECTOR_OBS_URL else s.SPECIAL_NEWS_XML

    async def fetch_json(url, _headers=None, _max_bytes=None):
        record(url)
        return s.ROADWORKS_JSON

    monkeypatch.setattr(client, "fetch_text", fetch_text)
    monkeypatch.setattr(client, "fetch_xml_text", fetch_xml_text)
    monkeypatch.setattr(client, "fetch_json", fetch_json)
    return client, calls, state


@pytest.mark.asyncio
async def test_fetch_traffic_data_honors_all_source_ttls(monkeypatch):
    client, calls, _ = _traffic_client(monkeypatch)
    first = await fetch_traffic_data(client)
    second = await fetch_traffic_data(client)

    assert sum(calls.values()) == 4
    assert all(count == 1 for count in calls.values())
    assert second == first
    assert first[0]
    assert first[3] is not None
    assert first[5]["detectors"] == first[3]
    assert first[5]["traffic_news"] == datetime.fromtimestamp(
        client.cache._store[SPECIAL_NEWS_SPEC.key()].fetched_at, UTC  # noqa: SLF001
    )
    assert first[5]["roadworks"] == datetime.fromtimestamp(
        client.cache._store[ROADWORKS_SPEC.key()].fetched_at, UTC  # noqa: SLF001
    )

    assert DETECTOR_META_SPEC.ttl == 24 * 60 * 60
    assert DETECTOR_OBS_SPEC.ttl == 55
    assert SPECIAL_NEWS_SPEC.ttl == 295
    assert ROADWORKS_SPEC.ttl == 15 * 60


@pytest.mark.asyncio
async def test_fetch_traffic_data_uses_expired_values_on_source_errors(monkeypatch):
    client, calls, state = _traffic_client(monkeypatch)
    first = await fetch_traffic_data(client)

    for entry in client.cache._store.values():  # noqa: SLF001
        entry.fetched_at = 0
    state["fail"] = True
    stale = await fetch_traffic_data(client)

    assert sum(calls.values()) == 8
    assert all(count == 2 for count in calls.values())
    assert stale[:4] == first[:4]
    assert stale[4] == [
        "TD detector metadata",
        "TD detector observations",
        "TD traffic news",
        "TD roadworks",
    ]
    assert stale[3] == first[3]
    assert stale[5] == {
        "detectors": first[3],
        "traffic_news": datetime.fromtimestamp(0, UTC),
        "roadworks": datetime.fromtimestamp(0, UTC),
    }
