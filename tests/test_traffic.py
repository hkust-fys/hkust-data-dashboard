"""Traffic provider tests: parsing, road matching, and source caching."""

from datetime import UTC, datetime, timedelta

import pytest

from dashboard.models import SpeedBand, TrafficIncident
from dashboard.providers.tracked_roads import TrackedRoads, fallback_roads, with_official_names
from dashboard.providers.traffic import (
    DETECTOR_META_SPEC,
    DETECTOR_META_URL,
    DETECTOR_OBS_SPEC,
    DETECTOR_OBS_URL,
    ROADWORKS_SPEC,
    ROADWORKS_URL,
    RTHK_NEWS_SPEC,
    SPECIAL_NEWS_PAGE_SPEC,
    SPECIAL_NEWS_PAGE_URL,
    SPECIAL_NEWS_SPEC,
    SPECIAL_NEWS_TC_PAGE_SPEC,
    SPECIAL_NEWS_TC_PAGE_URL,
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
    parse_special_news_page,
    resolve_incident_road_keys,
    special_news_page_time,
    speed_band,
)
from dashboard.providers.traffic_news import (
    RTHK_TRAFFIC_NEWS_URL,
    filter_expired_rthk_incidents,
    parse_rthk_traffic_news,
    rthk_latest_report_time,
)
from tests.fixtures import sample_data as s

TD_NEWS_PAGE_HTML = """<html><body>
<p>2026/9/16 04:45:05 PM</p><h1>Special Traffic News</h1><ol>
<li>Due to traffic accident, part of the lanes of Clear Water Bay Road
 (Sai Kung bound) near Ngau Chi Wan Municipal Building is closed to all traffic.<br>
 Only remaining lanes are available to motorists. Traffic is busy now.</li>
<li>Roadworks on Lung Cheung Road near Diamond Hill.</li>
<li>Unrelated notice on Nathan Road.</li>
</ol></body></html>"""

TD_NEWS_TC_PAGE_HTML = """<html><body>
<p>2026\u5e749\u670816\u65e5 04:45:05 PM</p><h1>\u7279\u5225\u4ea4\u901a\u6d88\u606f</h1><ol>
<li>\u56e0\u4ea4\u901a\u610f\u5916\uff0c\u6e05\u6c34\u7063\u9053(\u5f80\u897f\u8ca2\u65b9\u5411)\u8fd1\u725b\u6c60\u7063\u5e02\u653f\u5927\u5ec8\u7684\u90e8\u4efd\u884c\u8eca\u7dda\u73fe\u5df2\u5c01\u9589\u3002</li>
<li>\u9f8d\u7fd4\u9053\u8fd1\u947d\u77f3\u5c71\u7684\u9053\u8def\u5de5\u7a0b\u3002</li>
<li>\u5f4c\u6566\u9053\u7684\u5176\u4ed6\u6d88\u606f\u3002</li>
</ol></body></html>"""

EMPTY_RTHK_PAGE_HTML = """<html><body><div class="articles">
<h1>交通消息</h1></div></body></html>"""


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
    assert resolve_incident_road_keys(incident, subroad_only) == ["lung cheung road flyover"]


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


def test_relevant_incident_filter_has_no_source_display_cap():
    roads = fallback_roads()
    names = [
        "Clear Water Bay Road",
        "New Clear Water Bay Road",
        "Lung Cheung Road",
        "Hang Hau Road",
        "Wan Po Road",
    ]
    incidents = [
        TrafficIncident(
            identifier=f"notice-{index}",
            title="Traffic notice",
            description=f"Lane closure on {name}",
            road=name,
            location=name,
            direction="",
            status="ACTIVE",
        )
        for index, name in enumerate(names)
    ]

    assert filter_relevant_incidents(incidents, roads) == incidents
    assert filter_relevant_incidents(incidents, roads, limit=3) == incidents[:3]


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
    assert incident.announcement_time == datetime.fromisoformat("2026-08-13T17:23:00+08:00")


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
    doubled = s.SPECIAL_NEWS_XML.replace(
        "</trafficNews>", s.SPECIAL_NEWS_XML.split("<trafficNews>")[-1]
    )
    incidents = parse_special_news(doubled)
    ids = [i.identifier for i in incidents]
    assert len(ids) == len(set(ids))


def test_parse_special_news_bad_xml():
    assert parse_special_news("garbage") == []


def test_parse_td_special_news_page_preserves_markup_text_and_matches_routes():
    roads = TrackedRoads(
        display_names={"clear water bay road": "Clear Water Bay Road"},
        aliases={"clear water bay road": "clear water bay road"},
        road_routes={"clear water bay road": ("91", "91M")},
    )
    incidents = parse_special_news_page(
        """<html><body><strong>Special Traffic News</strong><ol>
        <li>Due to traffic accident, part of the lanes of <b>Clear Water Bay Road</b>
        (Sai Kung bound) near Ngau Chi Wan &amp; Municipal Building is closed.<br>
        Traffic is busy now.</li></ol></body></html>"""
    )

    assert len(incidents) == 1
    assert incidents[0].description == (
        "Due to traffic accident, part of the lanes of Clear Water Bay Road "
        "(Sai Kung bound) near Ngau Chi Wan & Municipal Building is closed. "
        "Traffic is busy now."
    )
    keys = resolve_incident_road_keys(incidents[0], roads)
    assert keys == ["clear water bay road"]
    assert roads.routes_for_keys(keys) == ["91", "91M"]
    assert incidents[0].source == "TD"
    assert incidents[0].announcement_time is None


def test_td_page_reopening_suppresses_older_matching_closure_only():
    roads = fallback_roads()
    incidents = parse_special_news_page(
        """<html><body>Special Traffic News<ol>
        <li>Part of the lanes of Clear Water Bay Road (Sai Kung bound) near
        Ngau Chi Wan Municipal Building which was closed due to traffic accident
        is re-opened to all traffic.</li>
        <li>Due to traffic accident, part of the lanes of Clear Water Bay Road
        (Sai Kung bound) near Ngau Chi Wan Municipal Building is closed.</li>
        <li>Due to vehicle breakdown, the slow lane of Clear Water Bay Road
        (Kowloon bound) near Fei Ngo Shan Road is closed.</li>
        </ol></body></html>"""
    )
    relevant = filter_relevant_incidents(incidents, roads, limit=10)

    assert [item.status for item in relevant] == ["CLOSED", "ACTIVE"]
    assert "re-opened" in relevant[0].description
    assert "Fei Ngo Shan Road" in relevant[1].description


def test_parse_rthk_current_page_keeps_latest_cleared_cwb_update():
    html = """<html><body><div class="articles"><h1>交通消息</h1>
    <ul class="dec"><li class="inner">較早前清水灣道往西貢方向，近牛池灣街市的
    交通意外已清理，龍尾：牛池灣村遊樂場。<div class="date">
    2026-09-16 HKT 15:45</div></li></ul>
    <ul class="dec"><li class="inner">清水灣道往西貢方向，近牛池灣街市有交通意外，
    部分行車線封閉，一帶車多。<div class="date">2026-09-16 HKT 15:17</div>
    </li></ul></div></body></html>"""
    parsed = parse_rthk_traffic_news(html, fallback_roads())
    relevant = filter_relevant_incidents(parsed, fallback_roads(), limit=10)

    assert len(parsed) == 2
    assert len(relevant) == 1
    assert relevant[0].status == "CLOSED"
    assert relevant[0].source == "RTHK"
    assert relevant[0].road == "Clear Water Bay Road"
    assert relevant[0].announcement_time == datetime.fromisoformat("2026-09-16T15:45:00+08:00")
    assert "已清理" in relevant[0].description


def test_parse_rthk_distinguishes_new_clear_water_bay_road():
    html = """<html><body><div class="articles">交通消息
    <ul class="dec"><li class="inner">新清水灣道往西貢方向有交通意外。
    <div class="date">2026-09-16 HKT 16:00</div></li></ul>
    </div></body></html>"""
    incident = parse_rthk_traffic_news(html, fallback_roads())[0]

    assert incident.road == "New Clear Water Bay Road"
    assert resolve_incident_road_keys(incident, fallback_roads()) == ["new clear water bay road"]


def test_filter_expired_rthk_incidents_preserves_td_and_fresh_related_reports():
    now = datetime.fromisoformat("2026-09-16T17:00:00+08:00")
    expired = TrafficIncident(
        "rthk-old", "RTHK traffic update", "old report", "Clear Water Bay Road", "", "", "ACTIVE",
        source="RTHK", announcement_time=now - timedelta(hours=3, minutes=1),
    )
    fresh = TrafficIncident(
        "rthk-fresh", "RTHK traffic update", "fresh report", "Clear Water Bay Road", "", "", "ACTIVE",
        source="RTHK", announcement_time=now - timedelta(hours=3),
    )
    missing_time = TrafficIncident(
        "rthk-unknown", "RTHK traffic update", "unknown time", "Clear Water Bay Road", "", "", "ACTIVE",
        source="RTHK",
    )
    td = TrafficIncident(
        "td-current", "Traffic accident", "TD active report", "Clear Water Bay Road", "", "", "ACTIVE",
        source="TD", page_updated_at=now, related_reports=(expired, fresh, missing_time),
        reconciliation_key="traffic-cwb",
    )

    retained = filter_expired_rthk_incidents([expired, td], now=now, max_age_hours=3)

    assert [incident.identifier for incident in retained] == ["td-current"]
    assert [report.identifier for report in retained[0].related_reports] == [
        "rthk-fresh", "rthk-unknown",
    ]
    assert retained[0].reconciliation_key == "traffic-cwb"
    assert retained[0].page_updated_at == now


@pytest.mark.parametrize("max_age_hours", [0, float("nan"), float("inf"), float("-inf")])
def test_filter_expired_rthk_incidents_rejects_invalid_max_age(max_age_hours):
    incident = TrafficIncident(
        "rthk", "RTHK traffic update", "report", "Clear Water Bay Road", "", "", "ACTIVE",
        source="RTHK", announcement_time=datetime.fromisoformat("2026-09-16T16:00:00+08:00"),
    )

    with pytest.raises(ValueError):
        filter_expired_rthk_incidents(
            [incident],
            now=datetime.fromisoformat("2026-09-16T17:00:00+08:00"),
            max_age_hours=max_age_hours,
        )


def test_rthk_matches_other_bus_roads_and_retains_multiple_named_roads():
    roads = with_official_names(fallback_roads(), {"lung cheung road": ("龍翔道",)})
    html = """<div class="articles">交通消息<ul class="dec">
    <li class="inner">龍翔道近新清水灣道有交通意外，一帶車多。
    <div class="date">2026-09-16 HKT 16:00</div></li></ul></div>"""
    incident = parse_rthk_traffic_news(html, roads)[0]
    assert resolve_incident_road_keys(incident, roads) == [
        "lung cheung road",
        "new clear water bay road",
    ]
    assert filter_relevant_incidents([incident], roads) == [incident]


def test_rthk_clearance_matches_lane_before_or_after_landmark_particle():
    roads = with_official_names(fallback_roads(), {"lung cheung road": ("龍翔道",)})
    html = """<div class="articles">交通消息<ul class="dec">
    <li class="inner">較早前龍翔道往荃灣方向，近黃大仙港鐵站慢線的交通意外已清理，交通回復正常。
    <div class="date">2026-09-16 HKT 16:02</div></li>
    <li class="inner">龍翔道往荃灣方向，近黃大仙港鐵站的慢線有交通意外，龍尾：星河明居。
    <div class="date">2026-09-16 HKT 15:42</div></li></ul></div>"""
    incidents = filter_relevant_incidents(parse_rthk_traffic_news(html, roads), roads)
    assert len(incidents) == 1
    assert incidents[0].is_cleared


def test_td_source_page_time_and_rthk_latest_report_are_not_fetch_times():
    assert special_news_page_time("<p>2026/9/16 04:45:05 PM</p>") == datetime.fromisoformat(
        "2026-09-16T16:45:05+08:00"
    )
    assert special_news_page_time("<p>No timestamp supplied</p>") is None
    assert special_news_page_time("<p>2026/99/16 04:45:05 PM</p>") is None
    html = """<div class="articles">交通消息<ul class="dec">
    <li class="inner">彌敦道有交通消息。<div class="date">2026-09-16 HKT 16:50</div></li>
    <li class="inner">清水灣道有交通消息。<div class="date">2026-09-16 HKT 16:20</div></li>
    </ul></div>"""
    assert rthk_latest_report_time(html) == datetime.fromisoformat("2026-09-16T16:50:00+08:00")
    assert len(parse_rthk_traffic_news(html, fallback_roads())) == 1


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
    calls = {
        url: 0
        for url in (
            DETECTOR_META_URL,
            DETECTOR_OBS_URL,
            SPECIAL_NEWS_PAGE_URL,
            SPECIAL_NEWS_TC_PAGE_URL,
            SPECIAL_NEWS_URL,
            RTHK_TRAFFIC_NEWS_URL,
            ROADWORKS_URL,
        )
    }
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

    async def fetch_html(url, _headers=None, _max_bytes=None, **_kwargs):
        record(url)
        if url == SPECIAL_NEWS_PAGE_URL:
            return TD_NEWS_PAGE_HTML
        if url == SPECIAL_NEWS_TC_PAGE_URL:
            return TD_NEWS_TC_PAGE_HTML
        return EMPTY_RTHK_PAGE_HTML

    async def fetch_json(url, _headers=None, _max_bytes=None):
        record(url)
        return s.ROADWORKS_JSON

    monkeypatch.setattr(client, "fetch_text", fetch_text)
    monkeypatch.setattr(client, "fetch_xml_text", fetch_xml_text)
    monkeypatch.setattr(client, "fetch_html", fetch_html)
    monkeypatch.setattr(client, "fetch_json", fetch_json)
    return client, calls, state


@pytest.mark.asyncio
async def test_fetch_traffic_data_honors_all_source_ttls(monkeypatch):
    client, calls, _ = _traffic_client(monkeypatch)
    first = await fetch_traffic_data(client, fallback_roads())
    second = await fetch_traffic_data(client, fallback_roads())

    assert sum(calls.values()) == 6
    assert calls[SPECIAL_NEWS_URL] == 0
    assert all(
        calls[url] == 1
        for url in (
            DETECTOR_META_URL,
            DETECTOR_OBS_URL,
            SPECIAL_NEWS_PAGE_URL,
            SPECIAL_NEWS_TC_PAGE_URL,
            RTHK_TRAFFIC_NEWS_URL,
            ROADWORKS_URL,
        )
    )
    assert second == first
    assert first[0]
    assert any("Ngau Chi Wan Municipal Building" in item.description for item in first[1])
    assert first[3] is not None
    assert first[5]["detectors"] == first[3]
    td_checked = datetime.fromtimestamp(
        client.cache._store[SPECIAL_NEWS_PAGE_SPEC.key()].fetched_at,
        UTC,  # noqa: SLF001
    )
    page_time = datetime.fromisoformat("2026-09-16T16:45:05+08:00")
    assert first[5]["traffic_news"] == page_time
    assert first[5]["traffic_news_checked"] == td_checked
    assert first[5]["traffic_news_updated"] == page_time
    assert all(item.page_updated_at == page_time for item in first[1] if item.source == "TD")
    assert all(item.announcement_time is None for item in first[1] if item.source == "TD")
    assert any(item.translated_description for item in first[1] if item.source == "TD")
    assert "rthk_news_checked" in first[5]
    assert first[5]["roadworks"] == datetime.fromtimestamp(
        client.cache._store[ROADWORKS_SPEC.key()].fetched_at,
        UTC,  # noqa: SLF001
    )

    assert DETECTOR_META_SPEC.ttl == 24 * 60 * 60
    assert DETECTOR_OBS_SPEC.ttl == 55
    assert SPECIAL_NEWS_SPEC.ttl == 60
    assert SPECIAL_NEWS_PAGE_SPEC.ttl == 60
    assert SPECIAL_NEWS_TC_PAGE_SPEC.ttl == 60
    assert RTHK_NEWS_SPEC.ttl == 60
    assert ROADWORKS_SPEC.ttl == 15 * 60


@pytest.mark.asyncio
async def test_fetch_traffic_data_uses_expired_values_on_source_errors(monkeypatch):
    client, calls, state = _traffic_client(monkeypatch)
    first = await fetch_traffic_data(client, fallback_roads())

    for entry in client.cache._store.values():  # noqa: SLF001
        entry.fetched_at = 0
    state["fail"] = True
    stale = await fetch_traffic_data(client, fallback_roads())

    assert sum(calls.values()) == 12
    assert calls[SPECIAL_NEWS_URL] == 0
    assert all(
        calls[url] == 2
        for url in (
            DETECTOR_META_URL,
            DETECTOR_OBS_URL,
            SPECIAL_NEWS_PAGE_URL,
            SPECIAL_NEWS_TC_PAGE_URL,
            RTHK_TRAFFIC_NEWS_URL,
            ROADWORKS_URL,
        )
    )
    assert stale[:4] == first[:4]
    assert stale[4] == [
        "TD detector metadata",
        "TD detector observations",
        "TD traffic news",
        "RTHK traffic news",
        "TD roadworks",
    ]
    assert stale[3] == first[3]
    assert stale[5] == {
        "detectors": first[3],
        "traffic_news": first[5]["traffic_news"],
        "traffic_news_checked": datetime.fromtimestamp(0, UTC),
        "traffic_news_updated": first[5]["traffic_news_updated"],
        "rthk_news_checked": datetime.fromtimestamp(0, UTC),
        "roadworks": datetime.fromtimestamp(0, UTC),
    }


@pytest.mark.asyncio
async def test_fetch_traffic_data_distinguishes_valid_empty_news_feed(monkeypatch):
    client, _, _ = _traffic_client(monkeypatch)
    original_fetch_html = client.fetch_html

    async def fetch_html(url, headers=None, max_bytes=None):
        if url == SPECIAL_NEWS_PAGE_URL:
            return "<html><body>Special Traffic News<ol></ol></body></html>"
        return await original_fetch_html(url, headers, max_bytes)

    monkeypatch.setattr(client, "fetch_html", fetch_html)
    result = await fetch_traffic_data(client, fallback_roads())

    assert result[1] == []
    assert "TD traffic news page unavailable" not in result[4]
    checked = datetime.fromtimestamp(
        client.cache._store[SPECIAL_NEWS_PAGE_SPEC.key()].fetched_at,
        UTC,  # noqa: SLF001
    )
    assert result[5]["traffic_news"] == checked
    assert result[5]["traffic_news_checked"] == checked


@pytest.mark.asyncio
async def test_failed_td_page_does_not_use_incomplete_xml_as_current_news(monkeypatch):
    client, calls, _ = _traffic_client(monkeypatch)
    original_fetch_html = client.fetch_html
    original_fetch_xml_text = client.fetch_xml_text

    async def fetch_html(url, headers=None, max_bytes=None):
        if url == SPECIAL_NEWS_PAGE_URL:
            return "<html><body>upstream error</body></html>"
        return await original_fetch_html(url, headers, max_bytes)

    async def fetch_xml_text(url, headers=None, max_bytes=None):
        if url == SPECIAL_NEWS_URL:
            return "<list />"
        return await original_fetch_xml_text(url, headers, max_bytes)

    monkeypatch.setattr(client, "fetch_html", fetch_html)
    monkeypatch.setattr(client, "fetch_xml_text", fetch_xml_text)
    result = await fetch_traffic_data(client, fallback_roads())

    assert result[1] == []
    assert "TD traffic news page unavailable" in result[4]
    assert not any("XML" in source for source in result[4])
    assert "traffic_news_checked" not in result[5]
    assert SPECIAL_NEWS_PAGE_SPEC.key() not in client.cache._store  # noqa: SLF001
    assert SPECIAL_NEWS_SPEC.key() not in client.cache._store  # noqa: SLF001
    assert calls[SPECIAL_NEWS_URL] == 0


@pytest.mark.asyncio
async def test_fetch_traffic_data_uses_stale_td_page_after_invalid_refresh(monkeypatch):
    client, _, _ = _traffic_client(monkeypatch)
    first = await fetch_traffic_data(client, fallback_roads())
    original_fetch_html = client.fetch_html

    async def fetch_html(url, headers=None, max_bytes=None):
        if url == SPECIAL_NEWS_PAGE_URL:
            return "<html><body>changed schema</body></html>"
        return await original_fetch_html(url, headers, max_bytes)

    monkeypatch.setattr(client, "fetch_html", fetch_html)
    client.cache._store[SPECIAL_NEWS_PAGE_SPEC.key()].fetched_at = 0  # noqa: SLF001
    result = await fetch_traffic_data(client, fallback_roads())

    assert result[1] == first[1]
    assert "TD traffic news" in result[4]
    assert "TD traffic news page unavailable" not in result[4]
    assert SPECIAL_NEWS_SPEC.key() not in client.cache._store  # noqa: SLF001
    assert result[5]["traffic_news_checked"] == datetime.fromtimestamp(0, UTC)


@pytest.mark.asyncio
async def test_fetch_traffic_data_uses_stale_rthk_page_after_invalid_refresh(monkeypatch):
    client, _, _ = _traffic_client(monkeypatch)
    original_fetch_html = client.fetch_html
    rthk_page = """<html><body><div class="articles">交通消息
    <ul class="dec"><li class="inner">清水灣道往西貢方向近牛池灣街市有交通意外。
    <div class="date">2026-09-16 HKT 15:17</div></li></ul>
    </div></body></html>"""
    state = {"invalid": False}

    async def fetch_html(url, headers=None, max_bytes=None):
        if url == RTHK_TRAFFIC_NEWS_URL:
            return "<html>changed schema</html>" if state["invalid"] else rthk_page
        return await original_fetch_html(url, headers, max_bytes)

    monkeypatch.setattr(client, "fetch_html", fetch_html)
    now = datetime.fromisoformat("2026-09-16T16:00:00+08:00")
    first = await fetch_traffic_data(client, fallback_roads(), now=now)
    state["invalid"] = True
    client.cache._store[RTHK_NEWS_SPEC.key()].fetched_at = 0  # noqa: SLF001
    result = await fetch_traffic_data(client, fallback_roads(), now=now)

    first_rthk = [item for item in first[1] if item.source == "RTHK"]
    result_rthk = [item for item in result[1] if item.source == "RTHK"]
    assert result_rthk == first_rthk
    assert "RTHK traffic news" in result[4]
    assert "RTHK traffic news unavailable" not in result[4]
    assert result[5]["rthk_news_checked"] == datetime.fromtimestamp(0, UTC)


@pytest.mark.asyncio
async def test_fetch_traffic_data_expires_old_rthk_cached_reports(monkeypatch):
    client, _, _ = _traffic_client(monkeypatch)
    original_fetch_html = client.fetch_html
    rthk_page = """<html><body><div class="articles"><h1>\u4ea4\u901a\u6d88\u606f</h1>
    <ul class="dec"><li class="inner">\u6e05\u6c34\u7063\u9053\u5f80\u897f\u8ca2\u65b9\u5411\u8fd1\u725b\u6c60\u7063\u8857\u5e02\u6709\u4ea4\u901a\u610f\u5916\u3002
    <div class="date">2026-09-16 HKT 14:00</div></li></ul>
    </div></body></html>"""
    state = {"invalid": False}

    async def fetch_html(url, headers=None, max_bytes=None):
        if url == RTHK_TRAFFIC_NEWS_URL:
            return "<html>changed schema</html>" if state["invalid"] else rthk_page
        return await original_fetch_html(url, headers, max_bytes)

    monkeypatch.setattr(client, "fetch_html", fetch_html)
    first = await fetch_traffic_data(
        client,
        fallback_roads(),
        now=datetime.fromisoformat("2026-09-16T16:30:00+08:00"),
    )
    assert any(incident.source == "RTHK" for incident in first[1])

    state["invalid"] = True
    client.cache._store[RTHK_NEWS_SPEC.key()].fetched_at = 0  # noqa: SLF001
    result = await fetch_traffic_data(
        client,
        fallback_roads(),
        now=datetime.fromisoformat("2026-09-16T17:30:00+08:00"),
    )

    assert not any(incident.source == "RTHK" for incident in result[1])
    assert any(incident.source == "TD" for incident in result[1])
    assert "RTHK traffic news" in result[4]
    assert result[5]["rthk_news"] == datetime.fromisoformat("2026-09-16T14:00:00+08:00")
    assert result[5]["rthk_news_checked"] == datetime.fromtimestamp(0, UTC)
