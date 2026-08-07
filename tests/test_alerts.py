"""Alert monitor tests: transition detection, escalation wording, pings."""

from dashboard.alerts import (
    CRITICAL_WARNING_CODES,
    AlertMonitor,
    _family_of,
    _warning_name,
)
from dashboard.models import SpeedBand, TrafficCorridorStatus, TrafficObservation, WeatherWarning
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


def test_black_rainstorm_pings():
    mon = AlertMonitor()
    mon.update([], [])
    events = mon.update([_warning("WRAINB")], [])
    assert events and events[0].critical
    # ping rendering with a role
    msg = mon.ping_for(events[0], role_id=123456789)
    assert "<@&123456789>" in msg


def test_cancel_posts_event():
    mon = AlertMonitor()
    mon.update([_warning("WRAINB")], [])
    events = mon.update([], [])
    assert len(events) == 1
    assert "cancelled" in events[0].text
    assert events[0].critical  # black rainstorm removal also pings


def test_heavy_congestion_appears_and_clears():
    mon = AlertMonitor()
    mon.update([], [])
    slow = _status("Clear Water Bay Road", SpeedBand.RED, speed=10)
    events = mon.update([], [slow])
    assert len(events) == 1
    assert events[0].critical
    assert "heavy congestion" in events[0].text

    clear = _status("Clear Water Bay Road", SpeedBand.GREEN, speed=60)
    events = mon.update([], [clear])
    assert len(events) == 1
    assert "cleared" in events[0].text


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
