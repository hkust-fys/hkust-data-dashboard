"""Traffic provider tests: parsing, road matching, and source caching."""

from datetime import UTC, datetime

import pytest

from dashboard.models import SpeedBand, TrafficIncident
from dashboard.providers.tracked_roads import TrackedRoads, fallback_roads
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
    match_roads,
    parse_detector_metadata,
    parse_detector_observations,
    parse_roadworks,
    parse_special_news,
    resolve_incident_road_keys,
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


def test_match_roads_aliases():
    roads = fallback_roads()
    assert match_roads("Clear Water Bay Road accident", roads) == ["clear water bay road"]
    assert match_roads("New Clear Water Bay Road works", roads) == ["new clear water bay road"]
    assert match_roads("Lung Cheung Road", roads) == ["lung cheung road"]
    assert match_roads("Hiram's Highway", roads) == ["hiram's highway"]
    assert match_roads("Po Lam Road closure", roads) == ["po lam road"]
    assert match_roads("Nathan Road", roads) == []
    # no table loaded: matches nothing rather than over-matching
    assert match_roads("Clear Water Bay Road", None) == []


def _notice_roads() -> TrackedRoads:
    names = {
        "tseung kwan o tunnel": "Tseung Kwan O Tunnel",
        "tseung kwan o tunnel road": "Tseung Kwan O Tunnel Road",
    }
    return TrackedRoads(
        display_names=names,
        aliases={key: key for key in names},
        road_routes={key: ("12",) for key in names},
    )


def _tko_road_notice() -> TrafficIncident:
    return TrafficIncident(
        identifier="tko-road-reopened",
        title="Road Incident",
        description=(
            "The fast lane of Tseung Kwan O Road (Tseung Kwan O Tunnel bound) "
            "near Hing Tin Estate which was closed due to traffic accident is re-opened "
            "to all traffic."
        ),
        road="Tseung Kwan O Road",
        location="Tseung Kwan O Road",
        direction="",
        status="active",
    )


def test_incident_road_resolution_rejects_directional_parenthetical_for_untracked_road():
    roads = _notice_roads()
    incident = _tko_road_notice()

    assert resolve_incident_road_keys(incident, roads) == []
    assert filter_relevant_incidents([incident], roads) == []


def test_incident_road_resolution_ignores_parenthetical_direction_without_explicit_road():
    roads = _notice_roads()
    incident = _tko_road_notice()
    incident.road = ""
    incident.location = "Hing Tin Estate"

    assert resolve_incident_road_keys(incident, roads) == []

    incident.description = "A lane on Tseung Kwan O Tunnel is closed."
    assert resolve_incident_road_keys(incident, roads) == ["tseung kwan o tunnel"]


def test_incident_road_resolution_explicit_fields_outrank_unrelated_narrative():
    roads = TrackedRoads(
        display_names={"clear water bay road": "Clear Water Bay Road"},
        aliases={"clear water bay road": "clear water bay road"},
    )
    incident = _tko_road_notice()
    incident.description += " Traffic remains normal on Clear Water Bay Road."

    assert resolve_incident_road_keys(incident, roads) == []

    incident.road = "Clear Water Bay Road"
    incident.location = "Hing Tin Estate"
    assert resolve_incident_road_keys(incident, roads) == ["clear water bay road"]


def test_incident_road_resolution_keeps_explicit_road_and_strict_subroad_refinement():
    names = {
        "lung cheung road": "Lung Cheung Road",
        "lung cheung road flyover": "Lung Cheung Road flyover",
    }
    roads = TrackedRoads(
        display_names=names,
        aliases={key: key for key in names},
        road_routes={key: ("91",) for key in names},
    )
    incident = TrafficIncident(
        identifier="lung-cheung-flyover",
        title="Traffic incident",
        description="Lung Cheung Road flyover is closed",
        road="Lung Cheung Road",
        location="Choi Hung Estate",
        direction="Mong Kok-bound",
        status="active",
        near_landmark="Choi Hung Estate",
    )

    assert resolve_incident_road_keys(incident, roads) == ["lung cheung road"]
    assert resolve_incident_road_keys(incident, roads, prefer_refinement=True) == [
        "lung cheung road flyover"
    ]
    subroad_only = TrackedRoads(
        display_names={"lung cheung road flyover": "Lung Cheung Road flyover"},
        aliases={"lung cheung road flyover": "lung cheung road flyover"},
    )
    assert resolve_incident_road_keys(incident, subroad_only) == [
        "lung cheung road flyover"
    ]


def test_speed_bands():
    assert speed_band(10) == SpeedBand.RED
    assert speed_band(20) == SpeedBand.AMBER
    assert speed_band(40) == SpeedBand.AMBER
    assert speed_band(41) == SpeedBand.GREEN
    assert speed_band(None) == SpeedBand.GRAY
    assert speed_band(10, stale=True) == SpeedBand.GRAY


def test_build_corridor_statuses_groups_and_orders():
    roads = fallback_roads()
    meta = parse_detector_metadata(s.DETECTOR_CSV)
    obs = parse_detector_observations(s.DETECTOR_XML)
    statuses = build_corridor_statuses(obs, meta, roads)
    names = [st.name for st in statuses]
    assert "clear water bay road" in names
    # unrelated road excluded
    assert "nathan road" not in names
    cwb = [st for st in statuses if st.name == "clear water bay road"][0]
    assert cwb.observations[0].band == SpeedBand.RED
    assert cwb.direction != ""  # from the "Eastbound" description hint


def test_build_corridor_statuses_empty():
    assert build_corridor_statuses({}, {}) == []


def test_parse_special_news_and_filter_relevant():
    roads = fallback_roads()
    incidents = parse_special_news(s.SPECIAL_NEWS_XML)
    assert len(incidents) == 3
    relevant = filter_relevant_incidents(incidents, roads)
    assert [i.identifier for i in relevant] == ["TN-1", "TN-2"]
    assert len(relevant) <= 3


def test_parse_special_news_live_uppercase_schema_and_hkt_announcement_time():
    incidents = parse_special_news(s.SPECIAL_NEWS_LIVE_XML)

    assert len(incidents) == 1
    incident = incidents[0]
    assert incident.identifier == "IN-26-00001"
    assert incident.title == "Road Incident"
    assert incident.description == "One lane near Fei Ngo Shan Road is closed."
    assert incident.road == "Clear Water Bay Road"
    assert incident.location == "Clear Water Bay Road"
    assert incident.direction == "Kowloon"
    assert incident.status == "UPDATED"
    assert incident.announcement_time == datetime.fromisoformat(
        "2026-08-13T17:23:00+08:00"
    )


def test_parse_special_news_keeps_td_coordinate_and_landmarks():
    xml = """<list><message>
      <INCIDENT_NUMBER>I-1</INCIDENT_NUMBER><INCIDENT_HEADING_EN>Closure</INCIDENT_HEADING_EN>
      <LATITUDE>22.3274</LATITUDE><LONGITUDE>114.2332</LONGITUDE>
      <NEAR_LANDMARK_EN>HKUST</NEAR_LANDMARK_EN>
      <BETWEEN_LANDMARK_EN>Gate A and Gate B</BETWEEN_LANDMARK_EN>
    </message></list>"""
    incident = parse_special_news(xml)[0]
    assert incident.latitude == 22.3274
    assert incident.longitude == 114.2332
    assert incident.near_landmark == "HKUST"
    assert incident.between_landmark == "Gate A and Gate B"


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
    roads = fallback_roads()
    rw = parse_roadworks(s.ROADWORKS_JSON, roads)
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
    first = await fetch_traffic_data(client, fallback_roads())
    second = await fetch_traffic_data(client, fallback_roads())

    assert sum(calls.values()) == 4
    assert all(count == 1 for count in calls.values())
    assert second == first
    assert first[0]
    assert first[3] is not None
    assert first[5]["detectors"] == first[3]
    assert first[5]["traffic_news"] == datetime.fromisoformat(
        "2026-08-13T16:45:00+08:00"
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
    first = await fetch_traffic_data(client, fallback_roads())

    for entry in client.cache._store.values():  # noqa: SLF001
        entry.fetched_at = 0
    state["fail"] = True
    stale = await fetch_traffic_data(client, fallback_roads())

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
        "traffic_news": datetime.fromisoformat("2026-08-13T16:45:00+08:00"),
        "roadworks": datetime.fromtimestamp(0, UTC),
    }
