"""Strict bilingual and cross-source traffic-report reconciliation tests."""

from dataclasses import replace
from datetime import datetime

from dashboard.models import TrafficIncident
from dashboard.providers.tracked_roads import TrackedRoads
from dashboard.providers.traffic import parse_special_news_page, special_news_page_time
from dashboard.providers.traffic_reconciliation import (
    incident_details,
    pair_td_bilingual_pages,
    reconcile_incident_reports,
)

PAGE_TIME = datetime.fromisoformat("2026-09-16T16:56:38+08:00")


def _roads() -> TrackedRoads:
    return TrackedRoads(
        display_names={
            "lung cheung road": "Lung Cheung Road",
            "kwun tong bypass": "Kwun Tong Bypass",
            "clear water bay road": "Clear Water Bay Road",
            "new clear water bay road": "New Clear Water Bay Road",
        },
        aliases={
            "lung cheung road": "lung cheung road",
            "kwun tong bypass": "kwun tong bypass",
            "clear water bay road": "clear water bay road",
            "new clear water bay road": "new clear water bay road",
        },
        name_aliases={
            "lung cheung road": ("\u9f8d\u7fd4\u9053",),
            "kwun tong bypass": ("\u89c0\u5858\u7e5e\u9053",),
            "clear water bay road": ("\u6e05\u6c34\u7063\u9053",),
            "new clear water bay road": ("\u65b0\u6e05\u6c34\u7063\u9053",),
        },
    )


ENGLISH_PAGE = """<html><body><p>2026/9/16 04:56:38 PM</p>
<h1>Special Traffic News</h1><ol>
<li>The slow lane of Lung Cheung Road (Tsuen Wan bound) near Wong Tai Sin MTR
Station which was closed due to traffic accident is re-opened to all traffic.</li>
<li>The slow lane of Kwun Tong Bypass (Mong Kok bound) near Kwun Tong Ferry Pier
which was closed due to vehicle breakdown is re-opened to all traffic.</li>
<li>Part of the lanes of Clear Water Bay Road (Sai Kung bound) near Ngau Chi Wan
Municipal Building which was closed due to traffic accident is re-opened to all traffic.</li>
<li>Administrative notice unrelated to tracked roads.</li>
</ol></body></html>"""

CHINESE_PAGE = """<html><body><p>2026\u5e749\u670816\u65e5 04:56:38 PM</p>
<h1>\u7279\u5225\u4ea4\u901a\u6d88\u606f</h1><ol>
<li>\u8f03\u65e9\u524d\u56e0\u4ea4\u901a\u610f\u5916\u800c\u5c01\u9589\u7684\u9f8d\u7fd4\u9053(\u5f80\u8343\u7063\u65b9\u5411)\u8fd1\u9ec3\u5927\u4ed9\u6e2f\u9435\u7ad9\u7684\u6162\u7dda\u73fe\u5df2\u89e3\u5c01\u3002</li>
<li>\u8f03\u65e9\u524d\u56e0\u8eca\u8f1b\u6545\u969c\u800c\u5c01\u9589\u7684\u89c0\u5858\u7e5e\u9053(\u5f80\u65fa\u89d2\u65b9\u5411)\u8fd1\u89c0\u5858\u78bc\u982d\u7684\u6162\u7dda\u73fe\u5df2\u89e3\u5c01\u3002</li>
<li>\u8f03\u65e9\u524d\u56e0\u4ea4\u901a\u610f\u5916\u800c\u5c01\u9589\u7684\u6e05\u6c34\u7063\u9053(\u5f80\u897f\u8ca2\u65b9\u5411)\u8fd1\u725b\u6c60\u7063\u5e02\u653f\u5927\u5ec8\u7684\u90e8\u4efd\u884c\u8eca\u7dda\u73fe\u5df2\u89e3\u5c01\u3002</li>
<li>\u8207\u8ddf\u8e64\u9053\u8def\u7121\u95dc\u7684\u884c\u653f\u6d88\u606f\u3002</li>
</ol></body></html>"""


def _enriched_td() -> list[TrafficIncident]:
    reports = parse_special_news_page(ENGLISH_PAGE)
    sidecars = pair_td_bilingual_pages(ENGLISH_PAGE, CHINESE_PAGE, reports, _roads())
    return [
        replace(
            report,
            page_updated_at=PAGE_TIME,
            translated_description=sidecars.get(report.identifier, ""),
        )
        for report in reports[:3]
    ]


def _rthk(text: str, minute: str, identifier: str) -> TrafficIncident:
    return TrafficIncident(
        identifier=identifier,
        title="RTHK traffic update",
        description=text,
        road="",
        location="",
        direction="",
        status="CLOSED",
        announcement_time=datetime.fromisoformat(f"2026-09-16T{minute}:00+08:00"),
        source="RTHK",
    )


def test_groups_real_lung_and_breakdown_synonym_but_not_landmark_conflict():
    td_lung, td_kwun, td_cwb = _enriched_td()
    rthk_lung = _rthk(
        "\u8f03\u65e9\u524d\u9f8d\u7fd4\u9053\u5f80\u8343\u7063\u65b9\u5411\uff0c\u8fd1\u9ec3\u5927\u4ed9\u6e2f\u9435\u7ad9\u6162\u7dda\u7684\u4ea4\u901a\u610f\u5916\u5df2\u6e05\u7406\uff0c\u4ea4\u901a\u56de\u5fa9\u6b63\u5e38\u3002",
        "16:02",
        "rthk-lung",
    )
    rthk_kwun = _rthk(
        "\u8f03\u65e9\u524d\u89c0\u5858\u7e5e\u9053\u5f80\u65fa\u89d2\u65b9\u5411\uff0c\u8fd1\u89c0\u5858\u78bc\u982d\u6162\u7dda\u7684\u58de\u8eca\u5df2\u6e05\u7406\uff0c\u4ea4\u901a\u56de\u5fa9\u6b63\u5e38\u3002",
        "15:48",
        "rthk-kwun",
    )
    rthk_cwb = _rthk(
        "\u8f03\u65e9\u524d\u6e05\u6c34\u7063\u9053\u5f80\u897f\u8ca2\u65b9\u5411\uff0c\u8fd1\u725b\u6c60\u7063\u8857\u5e02\u7684\u4ea4\u901a\u610f\u5916\u5df2\u6e05\u7406\u3002",
        "15:45",
        "rthk-cwb",
    )

    result = reconcile_incident_reports(
        [td_lung, td_kwun, td_cwb, rthk_lung, rthk_kwun, rthk_cwb], _roads()
    )

    assert [report.identifier for report in result] == [
        td_lung.identifier,
        td_kwun.identifier,
        td_cwb.identifier,
        rthk_cwb.identifier,
    ]
    assert td_lung.description == result[0].description
    assert result[0].related_reports[0].description == rthk_lung.description
    assert result[1].related_reports[0].description == rthk_kwun.description
    assert result[0].reconciliation_key == result[0].related_reports[0].reconciliation_key
    assert not result[2].related_reports
    assert not result[3].related_reports
    assert (
        incident_details(td_cwb, _roads()).landmark == "\u725b\u6c60\u7063\u5e02\u653f\u5927\u5ec8"
    )
    assert incident_details(rthk_cwb, _roads()).landmark == "\u725b\u6c60\u7063\u8857\u5e02"


def test_groups_known_tseng_lan_shue_landmark_variant_with_strict_signature():
    english = """<html><body><p>2026/9/16 04:33:00 PM</p>
    <h1>Special Traffic News</h1><ol><li>Due to traffic accident , the slow lane of
    Clear Water Bay Road (Sai Kung bound) near Tseng Lan Shue is closed to all traffic.
    Only remaining lane is available to motorists. Traffic is busy now.</li></ol></body></html>"""
    chinese = """<html><body><p>2026\u5e749\u670816\u65e5 04:33:00 PM</p>
    <h1>\u7279\u5225\u4ea4\u901a\u6d88\u606f</h1><ol><li>\u56e0\u4ea4\u901a\u610f\u5916\uff0c\u6e05\u6c34\u7063\u9053(\u5f80\u897f\u8ca2\u65b9\u5411)\u8fd1\u4e95\u6b04\u6a39\u7684\u6162\u7dda\u73fe\u5df2\u5c01\u9589\u3002
    \u99d5\u99db\u4eba\u58eb\u53ea\u53ef\u4f7f\u7528\u9918\u4e0b\u884c\u8eca\u7dda\u884c\u8eca\u3002\u73fe\u6642\u4e0a\u5740\u4ea4\u901a\u7e41\u5fd9\u3002</li></ol></body></html>"""
    parsed_td = parse_special_news_page(english)
    sidecars = pair_td_bilingual_pages(english, chinese, parsed_td, _roads())
    td = replace(
        parsed_td[0],
        page_updated_at=datetime.fromisoformat("2026-09-16T16:33:00+08:00"),
        translated_description=sidecars[parsed_td[0].identifier],
    )
    rthk = TrafficIncident(
        "rthk-tseng-lan-shue",
        "RTHK traffic update",
        "\u6e05\u6c34\u7063\u9053\u5f80\u897f\u8ca2\u65b9\u5411\uff0c\u8fd1\u4e95\u6b04\u6751\u6709\u4ea4\u901a\u610f\u5916\uff0c\u9f8d\u5c3e\uff1a\u5f69\u96f2\u90a8\u3002",
        "",
        "",
        "",
        "ACTIVE",
        announcement_time=datetime.fromisoformat("2026-09-16T16:00:00+08:00"),
        source="RTHK",
    )

    td_details = incident_details(td, _roads())
    rthk_details = incident_details(rthk, _roads())
    result = reconcile_incident_reports([td, rthk], _roads())

    assert td.title == "Traffic accident"
    assert td_details == rthk_details
    assert td_details.landmark == "tseng lan shue"
    assert td_details.direction == "\u897f\u8ca2"
    assert td_details.cause == "traffic-accident"
    assert len(result) == 1
    assert result[0].identifier == td.identifier
    assert result[0].related_reports == (replace(rthk, reconciliation_key=result[0].reconciliation_key),)
    assert result[0].reconciliation_key

    opposite_direction = replace(
        rthk,
        identifier="rthk-tseng-lan-shue-kowloon-bound",
        description=(
            "\u6e05\u6c34\u7063\u9053\u5f80\u4e5d\u9f8d\u65b9\u5411\uff0c\u8fd1\u4e95\u6b04\u6751\u6709\u4ea4\u901a\u610f\u5916\uff0c\u9f8d\u5c3e\uff1a\u5f69\u96f2\u90a8\u3002"
        ),
    )
    opposite_state = replace(
        rthk,
        identifier="rthk-tseng-lan-shue-cleared",
        description=(
            "\u6e05\u6c34\u7063\u9053\u5f80\u897f\u8ca2\u65b9\u5411\uff0c\u8fd1\u4e95\u6b04\u6751\u7684\u4ea4\u901a\u610f\u5916\u5df2\u6e05\u7406\u3002"
        ),
        status="CLOSED",
    )
    for guarded in (opposite_direction, opposite_state):
        guarded_result = reconcile_incident_reports([td, guarded], _roads())
        assert len(guarded_result) == 2
        assert all(not report.related_reports for report in guarded_result)

    duplicate = replace(rthk, identifier="rthk-tseng-lan-shue-duplicate")
    ambiguous = reconcile_incident_reports([td, rthk, duplicate], _roads())
    assert len(ambiguous) == 3
    assert all(not report.related_reports for report in ambiguous)


def test_bilingual_pairing_fails_closed_across_page_transition():
    reports = parse_special_news_page(ENGLISH_PAGE)
    later = CHINESE_PAGE.replace("04:56:38", "04:57:38")
    assert pair_td_bilingual_pages(ENGLISH_PAGE, later, reports, _roads()) == {}
    fewer = CHINESE_PAGE.replace(
        "<li>\u8207\u8ddf\u8e64\u9053\u8def\u7121\u95dc\u7684\u884c\u653f\u6d88\u606f\u3002</li>",
        "",
    )
    assert pair_td_bilingual_pages(ENGLISH_PAGE, fewer, reports, _roads()) == {}


def test_untracked_bilingual_item_does_not_block_proven_tracked_sidecars():
    reports = parse_special_news_page(ENGLISH_PAGE)
    sidecars = pair_td_bilingual_pages(ENGLISH_PAGE, CHINESE_PAGE, reports, _roads())
    assert set(sidecars) == {report.identifier for report in reports[:3]}
    assert reports[3].identifier not in sidecars


def test_bilingual_pairing_rejects_reordered_duplicate_event_signatures():
    english = """<p>2026/9/16 04:56:38 PM</p><h1>Special Traffic News</h1><ol>
    <li>Due to traffic accident, Clear Water Bay Road (Sai Kung bound) near
    Landmark A is closed.</li>
    <li>Due to traffic accident, Clear Water Bay Road (Kowloon bound) near
    Landmark B is closed.</li></ol>"""
    # The Chinese locale has the same coarse road/cause/state signatures but
    # reverses the two distinct events. No direction translation is attempted.
    chinese = """<p>2026\u5e749\u670816\u65e5 04:56:38 PM</p><h1>\u7279\u5225\u4ea4\u901a\u6d88\u606f</h1><ol>
    <li>\u56e0\u4ea4\u901a\u610f\u5916\uff0c\u6e05\u6c34\u7063\u9053(\u5f80\u4e5d\u9f8d\u65b9\u5411)\u8fd1\u5730\u6a19B\u73fe\u5df2\u5c01\u9589\u3002</li>
    <li>\u56e0\u4ea4\u901a\u610f\u5916\uff0c\u6e05\u6c34\u7063\u9053(\u5f80\u897f\u8ca2\u65b9\u5411)\u8fd1\u5730\u6a19A\u73fe\u5df2\u5c01\u9589\u3002</li>
    </ol>"""
    reports = parse_special_news_page(english)

    assert pair_td_bilingual_pages(english, chinese, reports, _roads()) == {}


def test_active_clearance_conflict_and_ambiguous_candidates_stay_separate():
    td = _enriched_td()[0]
    cleared = _rthk(
        "\u9f8d\u7fd4\u9053\u5f80\u8343\u7063\u65b9\u5411\uff0c\u8fd1\u9ec3\u5927\u4ed9\u6e2f\u9435\u7ad9\u7684\u4ea4\u901a\u610f\u5916\u5df2\u6e05\u7406\u3002",
        "16:02",
        "cleared",
    )
    active_td = replace(
        td,
        status="ACTIVE",
        translated_description=(
            "\u9f8d\u7fd4\u9053(\u5f80\u8343\u7063\u65b9\u5411)\u8fd1\u9ec3\u5927\u4ed9\u6e2f\u9435\u7ad9"
            "\u7684\u6162\u7dda\u6709\u4ea4\u901a\u610f\u5916\u3002"
        ),
    )
    conflict = reconcile_incident_reports([active_td, cleared], _roads())
    assert len(conflict) == 2
    assert all(not item.related_reports for item in conflict)

    duplicate = replace(cleared, identifier="cleared-again")
    ambiguous = reconcile_incident_reports([td, cleared, duplicate], _roads())
    assert len(ambiguous) == 3
    assert all(not item.related_reports for item in ambiguous)


def test_page_update_time_is_not_incident_announcement_time():
    reports = _enriched_td()
    assert special_news_page_time(CHINESE_PAGE) == PAGE_TIME
    assert all(report.page_updated_at == PAGE_TIME for report in reports)
    assert all(report.announcement_time is None for report in reports)


def test_report_outside_bounded_time_window_stays_separate():
    td = _enriched_td()[0]
    old_rthk = _rthk(
        "\u9f8d\u7fd4\u9053\u5f80\u8343\u7063\u65b9\u5411\uff0c\u8fd1\u9ec3\u5927\u4ed9\u6e2f\u9435\u7ad9"
        "\u7684\u4ea4\u901a\u610f\u5916\u5df2\u6e05\u7406\u3002",
        "12:00",
        "old-rthk",
    )

    result = reconcile_incident_reports([td, old_rthk], _roads())

    assert len(result) == 2
    assert all(not item.related_reports for item in result)


def test_new_clear_water_bay_road_stays_distinct_in_signature():
    report = TrafficIncident(
        "new-cwb",
        "RTHK traffic update",
        "\u65b0\u6e05\u6c34\u7063\u9053\u5f80\u897f\u8ca2\u65b9\u5411\uff0c\u8fd1\u5f69\u96f2\u90a8\u6709\u4ea4\u901a\u610f\u5916\u3002",
        "",
        "",
        "",
        "ACTIVE",
        source="RTHK",
    )
    details = incident_details(report, _roads())
    assert details.road_keys == ("new clear water bay road",)
    assert details.landmark == "\u5f69\u96f2\u90a8"
