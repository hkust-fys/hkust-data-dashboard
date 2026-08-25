"""Alert monitor tests: transition detection, escalation wording, pings."""

from dashboard.alerts import (
    CRITICAL_WARNING_CODES,
    AlertMonitor,
    _family_of,
    _warning_name,
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
    return WeatherWarning(code=code, name=_warning_name(code))


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
    # clears, then jams again within the cooldown: no new ping
    clear = _status("Clear Water Bay Road", SpeedBand.GREEN, speed=60)
    cleared = mon.update([], [clear])
    assert len(cleared) == 1 and "easing" in cleared[0].text
    assert mon.update([], [slow]) == []
    assert mon.update([], [slow]) == []


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

    events = mon.update([], [], [])
    assert len(events) == 1
    assert "TD traffic notice cleared" in events[0].text


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
