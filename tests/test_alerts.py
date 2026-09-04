"""Alert monitor tests: transition detection, escalation wording, pings."""

from dashboard import road_policy
from dashboard.alerts import (
    CRITICAL_WARNING_CODES,
    ROAD_ALERT_COOLDOWN_SECONDS,
    AlertMonitor,
    _family_of,
)
from dashboard.models import (
    Roadwork,
    SpeedBand,
    TrafficCorridorStatus,
    TrafficIncident,
    TrafficObservation,
    WeatherWarning,
)
from tests.fixtures import sample_data as s


def _warning(code: str) -> WeatherWarning:
    names = {
        "TC1": "Standby Signal No. 1", "TC8NE": "Gale Signal No. 8 NE",
        "WRAINA": "Amber Rainstorm", "WRAINR": "Red Rainstorm",
        "WRAINB": "Black Rainstorm", "TC8PRE": "Pre-No. 8 Special Announcement",
        "WHOT": "Very Hot Weather",
    }
    return WeatherWarning(code=code, name=names.get(code, code))


def _status(name: str, band: SpeedBand, speed: float | None = None) -> TrafficCorridorStatus:
    return TrafficCorridorStatus(
        name=name,
        direction="",
        observations=[
            TrafficObservation(
                corridor=name, direction="", description=name,
                latitude=22.33, longitude=114.22,
                speed_kmh=speed, volume=1, occupancy_pct=0.1,
                capture_time=s.utc(), band=band,
            )
        ],
        capture_time=s.utc(),
    )


def test_first_update_seeds_state_no_events():
    mon = AlertMonitor()
    events = mon.update([_warning("WHOT")], [])
    assert events == []


def test_typhoon_hoist_posts_noncritical_event():
    mon = AlertMonitor()
    mon.update([], [])  # seed
    events = mon.update([_warning("TC1")], [])
    assert len(events) == 1
    assert not events[0].critical  # T1 is not critical
    assert "Standby Signal No. 1" in events[0].text
    assert "hoisted" in events[0].text


def test_typhoon_escalation_reads_as_upgrade():
    mon = AlertMonitor()
    mon.update([_warning("TC1")], [])  # seed
    events = mon.update([_warning("TC8NE")], [])
    texts = [e.text for e in events]
    assert any("upgraded to" in t and "Gale Signal No. 8 NE" in t for t in texts)
    assert any("downgraded from" in t and "Standby Signal No. 1" in t for t in texts)
    # the T8 hoist is critical (pings)
    critical = [e for e in events if e.critical]
    assert any("Gale Signal No. 8 NE" in e.text for e in critical)


def test_rainstorm_escalation_amber_to_red():
    mon = AlertMonitor()
    mon.update([_warning("WRAINA")], [])
    events = mon.update([_warning("WRAINR")], [])
    texts = [e.text for e in events]
    assert any("upgraded to" in t and "Red Rainstorm" in t for t in texts)
    # red is not critical; black is
    assert all(not e.critical for e in events)


def test_black_rainstorm_posts_without_role_ping():
    mon = AlertMonitor()
    mon.update([], [])
    events = mon.update([_warning("WRAINB")], [])
    assert events and events[0].critical
    # Weather is prominent but never pings the role.
    msg = mon.ping_for(events[0], role_id=123456789)
    assert "<@&123456789>" not in msg


def test_cancel_posts_event():
    mon = AlertMonitor()
    mon.update([_warning("WRAINB")], [])
    events = mon.update([], [])
    assert len(events) == 1
    assert "cancelled" in events[0].text
    assert events[0].critical  # black rainstorm removal is prominent (no ping)
    assert mon.ping_for(events[0], role_id=123456789) == events[0].text


def test_cancel_uses_last_source_name():
    mon = AlertMonitor()
    mon.update([WeatherWarning(code="WRAINA", name="Amber Rainstorm Warning Signal")], [])
    events = mon.update([], [])
    assert events and "Amber Rainstorm Warning Signal" in events[0].text


def test_pre_no8_announcement_is_critical_code():
    assert "TC8PRE" in CRITICAL_WARNING_CODES
    mon = AlertMonitor()
    mon.update([], [])
    events = mon.update([_warning("TC8PRE")], [])
    assert events and events[0].critical
    assert "Pre-No. 8 Special Announcement" in events[0].text
    assert "<@&123456789>" not in mon.ping_for(events[0], role_id=123456789)


def test_congestion_requires_two_consecutive_red_updates():
    mon = AlertMonitor()
    mon.update([], [])
    slow = _status("Clear Water Bay Road", SpeedBand.RED, speed=10)
    assert mon.update([], [slow]) == []  # first red: not yet
    events = mon.update([], [slow])  # second consecutive red: alert
    assert len(events) == 1
    assert events[0].ping
    assert "Heavy congestion" in events[0].text
    assert "<@&123456789>" in mon.ping_for(events[0], role_id=123456789)


def test_congestion_streak_resets_on_nonconsecutive_red():
    mon = AlertMonitor()
    mon.update([], [])
    slow = _status("Clear Water Bay Road", SpeedBand.RED, speed=10)
    clear = _status("Clear Water Bay Road", SpeedBand.GREEN, speed=60)
    assert mon.update([], [slow]) == []
    assert mon.update([], [clear]) == []  # streak resets
    assert mon.update([], [slow]) == []  # needs two in a row again
    events = mon.update([], [slow])
    assert len(events) == 1 and events[0].ping


def test_congestion_cooldown_suppresses_repeat_ping():
    mon = AlertMonitor()
    mon.update([], [])
    slow = _status("Clear Water Bay Road", SpeedBand.RED, speed=10)
    mon.update([], [slow])
    first = mon.update([], [slow])
    assert len(first) == 1
    # stays red: active, no repeat
    assert mon.update([], [slow]) == []
    # clears, then jams again within the cooldown: thread update is retained,
    # but the repeated role ping is suppressed.
    clear = _status("Clear Water Bay Road", SpeedBand.GREEN, speed=60)
    cleared = mon.update([], [clear])
    assert len(cleared) == 1 and "easing" in cleared[0].text
    assert mon.update([], [slow]) == []
    repeated = mon.update([], [slow])
    assert len(repeated) == 1 and not repeated[0].ping


def test_one_hour_road_alert_cooldown_is_shared_across_sources(monkeypatch):
    from dashboard.providers.tracked_roads import fallback_roads

    clock = [1000.0]
    monkeypatch.setattr("dashboard.alerts.time.monotonic", lambda: clock[0])
    mon = AlertMonitor(roads=fallback_roads())
    incident = TrafficIncident(
        "td-cwb", "Incident on Clear Water Bay Road", "", "Clear Water Bay Road",
        "", "", "active",
    )
    slow = _status("Clear Water Bay Road", SpeedBand.RED, speed=10)
    clear = _status("Clear Water Bay Road", SpeedBand.GREEN, speed=60)
    mon.update([], [])

    news = mon.update([], [], [incident])
    assert len(news) == 1 and news[0].ping
    mon.update([], [slow], [incident])
    congestion = mon.update([], [slow], [incident])
    assert len(congestion) == 1 and "Heavy congestion" in congestion[0].text
    assert not congestion[0].ping  # traffic news shadows detector ping on same road

    mon.update([], [clear], [])
    clock[0] += ROAD_ALERT_COOLDOWN_SECONDS + 1
    mon.update([], [slow], [])
    later = mon.update([], [slow], [])
    assert len(later) == 1 and later[0].ping

    reverse = AlertMonitor(roads=fallback_roads())
    new_cwb_slow = _status("New Clear Water Bay Road", SpeedBand.RED, speed=9)
    new_cwb_news = TrafficIncident(
        "td-new-cwb", "Incident on New Clear Water Bay Road", "",
        "New Clear Water Bay Road", "", "", "active",
    )
    reverse.update([], [])
    reverse.update([], [new_cwb_slow])
    detector = reverse.update([], [new_cwb_slow])
    assert len(detector) == 1 and detector[0].ping
    news_after_detector = reverse.update([], [new_cwb_slow], [new_cwb_news])
    assert len(news_after_detector) == 1 and not news_after_detector[0].ping


def test_roadwork_does_not_consume_road_alert_cooldown():
    from dashboard.providers.tracked_roads import fallback_roads

    mon = AlertMonitor(roads=fallback_roads())
    roadwork = Roadwork("rw-1", "Works", "Clear Water Bay Road")
    incident = TrafficIncident(
        "td-cwb", "Incident on Clear Water Bay Road", "", "Clear Water Bay Road",
        "", "", "active",
    )
    mon.update([], [], [], [])
    assert mon.update([], [], [], [roadwork])
    news = mon.update([], [], [incident], [roadwork])
    assert len(news) == 1 and news[0].ping


def test_congestion_alerts_watch_only_two_clear_water_bay_roads():
    mon = AlertMonitor()
    other = _status("Lung Cheung Road", SpeedBand.RED, speed=10)
    mon.update([], [])
    mon.update([], [other])
    assert mon.update([], [other]) == []


def test_alerts_read_the_central_important_road_policy(monkeypatch):
    monkeypatch.setattr(
        road_policy, "IMPORTANT_ROAD_KEYS", frozenset({"lung cheung road"})
    )
    mon = AlertMonitor()
    mon.update([], [])
    important = _status("Lung Cheung Road", SpeedBand.RED, speed=10)
    ordinary = _status("Clear Water Bay Road", SpeedBand.RED, speed=10)

    mon.update([], [important, ordinary])
    events = mon.update([], [important, ordinary])

    assert len(events) == 1
    assert "Lung Cheung Road" in events[0].text
    assert events[0].ping


def test_congestion_alert_names_affected_routes():
    from dashboard.providers.route_geometry import RouteLine
    from dashboard.providers.tracked_roads import build_tracked_roads

    line = RouteLine("91", "KMB", "outbound")
    roads = build_tracked_roads(
        [line], [["Clear Water Bay Road", "Lung Cheung Road"]]
    )
    mon = AlertMonitor(roads=roads)
    mon.update([], [])
    slow = _status("clear water bay road", SpeedBand.RED, speed=8)
    mon.update([], [slow])
    events = mon.update([], [slow])
    assert events and "affects: 91" in events[0].text


def test_congestion_clears_when_band_leaves_red():
    mon = AlertMonitor()
    mon.update([], [])
    slow = _status("Clear Water Bay Road", SpeedBand.RED, speed=10)
    mon.update([], [slow])
    mon.update([], [slow])
    clear = _status("Clear Water Bay Road", SpeedBand.GREEN, speed=60)
    events = mon.update([], [clear])
    assert len(events) == 1
    assert "easing" in events[0].text
    assert not events[0].ping


def test_relevant_td_traffic_notice_posts_and_clears():
    mon = AlertMonitor()
    incident = TrafficIncident(
        identifier="td-1",
        title="Lane closure on Clear Water Bay Road",
        description="One lane closed",
        road="Clear Water Bay Road",
        location="HKUST approach",
        direction="eastbound",
        status="active",
    )
    mon.update([], [], [])
    events = mon.update([], [], [incident])
    assert len(events) == 1
    assert "Lane closure on Clear Water Bay Road" in events[0].text
    assert events[0].ping

    events = mon.update([], [], [])
    assert len(events) == 1
    assert "TD traffic notice cleared" in events[0].text


def test_traffic_notice_thread_messages_preserve_full_source_within_limits():
    mon = AlertMonitor()
    source = "first " + ("x" * 4_100) + " last"
    incident = TrafficIncident(
        identifier="td-long",
        title="Lane closure",
        description=source,
        road="Clear Water Bay Road",
        location="HKUST approach",
        direction="eastbound",
        status="active",
    )
    mon.update([], [], [])

    event = mon.update([], [], [incident])[0]
    messages = mon.messages_for(event, role_id=123)

    assert len(messages) == 3
    assert all(len(message) <= 2_000 for message in messages)
    assert source in "".join(messages)
    assert "<@&123>" in messages[0]


def test_priority_road_traffic_news_pings_only_when_new():
    from dashboard.providers.tracked_roads import fallback_roads

    for road in ("Clear Water Bay Road", "New Clear Water Bay Road"):
        mon = AlertMonitor(roads=fallback_roads())
        incident = TrafficIncident(
            identifier=f"td-{road}",
            title=f"Lane closure on {road}",
            description="One lane closed",
            road=road,
            location="",
            direction="eastbound",
            status="active",
        )
        mon.update([], [], [])

        posted = mon.update([], [], [incident])
        assert len(posted) == 1 and posted[0].ping
        assert "<@&123456789>" in mon.ping_for(posted[0], role_id=123456789)

        cleared = mon.update([], [], [])
        assert len(cleared) == 1 and not cleared[0].ping
        assert "<@&123456789>" not in mon.ping_for(cleared[0], role_id=123456789)


def test_direction_only_parenthetical_does_not_feed_incident_alert_keys(monkeypatch):
    from dashboard.providers.tracked_roads import TrackedRoads

    names = {
        "tseung kwan o tunnel": "Tseung Kwan O Tunnel",
        "tseung kwan o tunnel road": "Tseung Kwan O Tunnel Road",
    }
    roads = TrackedRoads(
        display_names=names,
        aliases={key: key for key in names},
        road_routes={key: ("12",) for key in names},
    )
    incident = TrafficIncident(
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
    monkeypatch.setattr(road_policy, "IMPORTANT_ROAD_KEYS", frozenset(names))

    assert AlertMonitor(roads=roads)._incident_road_keys(incident) == set()


def test_other_traffic_news_and_roadworks_do_not_ping():
    from dashboard.providers.tracked_roads import fallback_roads

    mon = AlertMonitor(roads=fallback_roads())
    mon.update([], [], [], [])
    incident = TrafficIncident(
        identifier="td-lung-cheung",
        title="Lane closure on Lung Cheung Road",
        description="One lane closed",
        road="Lung Cheung Road",
        location="",
        direction="eastbound",
        status="active",
    )
    roadwork = Roadwork("rw-cwb", "Works on Clear Water Bay Road", "Clear Water Bay Road")

    events = mon.update([], [], [incident], [roadwork])

    assert len(events) == 2
    assert all(not event.ping for event in events)


def test_relevant_roadwork_posts_once_and_clears_with_description():
    mon = AlertMonitor()
    roadwork = Roadwork(
        identifier="rw-1",
        description="Lane closure near HKUST",
        road="Clear Water Bay Road",
    )
    mon.update([], [], [], [])

    events = mon.update([], [], [], [roadwork])
    assert len(events) == 1
    assert "TD roadworks" in events[0].text
    assert "Lane closure near HKUST" in events[0].text
    assert "Clear Water Bay Road" in events[0].text
    assert not events[0].critical

    assert mon.update([], [], [], [roadwork]) == []

    events = mon.update([], [], [], [])
    assert len(events) == 1
    assert "TD roadworks cleared" in events[0].text
    assert "Lane closure near HKUST" in events[0].text


def test_first_update_seeds_roadwork_without_flood():
    mon = AlertMonitor()
    roadwork = Roadwork("rw-1", "Lane closure", "University Road")

    assert mon.update([], [], [], [roadwork]) == []
    assert mon.update([], [], [], [roadwork]) == []


def test_no_events_when_nothing_changes():
    mon = AlertMonitor()
    mon.update([_warning("WHOT")], [])
    events = mon.update([_warning("WHOT")], [])
    assert events == []


def test_family_grouping():
    assert _family_of("TC1") == "TC"
    assert _family_of("TC8NE") == "TC"
    assert _family_of("WRAINA") == "WRAIN"
    assert _family_of("WRAINB") == "WRAIN"
    assert _family_of("WHOT") == "WHOT"


def test_critical_codes_cover_t8_and_black():
    assert "TC8NE" in CRITICAL_WARNING_CODES
    assert "TC9" in CRITICAL_WARNING_CODES
    assert "TC10" in CRITICAL_WARNING_CODES
    assert "WRAINB" in CRITICAL_WARNING_CODES
    assert "TC1" not in CRITICAL_WARNING_CODES
    assert "WRAINA" not in CRITICAL_WARNING_CODES
    assert "WRAINR" not in CRITICAL_WARNING_CODES
