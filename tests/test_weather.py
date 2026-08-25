"""HKO weather provider tests: warning-code normalization, tolerance of empty
objects/null fields, and observation parsing."""

from datetime import datetime

from dashboard.providers.weather import (
    KNOWN_WARNING_CODES,
    parse_observations,
    parse_warnings,
)
from tests.fixtures import sample_data as s


def test_parse_observations_sai_kung():
    snap = parse_observations(s.hko_rhrread())
    assert snap.temperature_c == 28.5
    assert snap.rainfall_mm == 0.0
    assert snap.humidity_pct == 71
    assert snap.source_time is not None


def test_parse_observations_tolerates_missing_sections():
    snap = parse_observations({})
    assert snap.temperature_c is None
    assert snap.rainfall_mm is None
    assert snap.humidity_pct is None


def test_parse_observations_ignores_wrong_station():
    raw = {
        "temperature": {"data": [{"place": "Wong Tai Sin", "value": 31.0}]},
        "rainfall": {"data": [{"place": "Wong Tai Sin", "max": 3.0}]},
        "humidity": {"data": [{"place": "Wong Tai Sin", "value": 60}]},
    }
    snap = parse_observations(raw)
    assert snap.temperature_c is None
    assert snap.rainfall_mm is None
    assert snap.humidity_pct is None


def test_parse_warnings_normalizes_codes_in_order():
    warnings = parse_warnings(s.hko_warnsum(("RAIN", "TC")), s.hko_warning_info())
    codes = [w.code for w in warnings]
    # known-code order: TC before RAIN regardless of input order
    assert codes == ["TC", "RAIN"]
    tc = warnings[0]
    assert tc.name == "Typhoon"
    assert tc.summary == "Tropical Cyclone Warning"
    assert tc.action == "Stay indoors"


def test_parse_warnings_empty_objects():
    assert parse_warnings({}) == []
    assert parse_warnings({"TC": {}}) == []
    assert parse_warnings(None) == []


def test_parse_warnings_unknown_codes_sorted_last():
    warnings = parse_warnings({"TC": {"code": "TC"}, "XYZ": {"code": "XYZ"}}, None)
    codes = [w.code for w in warnings]
    assert codes == ["TC", "XYZ"]
    assert KNOWN_WARNING_CODES[0] == "TC"


def test_parse_warnings_null_fields_tolerated():
    warnings = parse_warnings(
        {"HOT": {"code": "HOT"}},
        {"details": {"HOT": {"summary": None, "action": None}}},
    )
    assert warnings[0].summary == ""
    assert warnings[0].action == ""


def test_pre_no8_statement_becomes_synthetic_warning():
    from dashboard.providers.weather import PRE_NO8_CODE

    info = {
        "details": {
            "TC": {
                "code": "TC",
                "statement": (
                    "The Hong Kong Observatory issues the Pre-No. 8 Special "
                    "Announcement at 6:15 p.m."
                ),
                "issueTime": "2026-08-21T18:15:00+08:00",
            }
        }
    }
    warnings = parse_warnings({"TC": {"code": "TC"}}, info)
    codes = [w.code for w in warnings]
    assert PRE_NO8_CODE in codes
    pre = next(w for w in warnings if w.code == PRE_NO8_CODE)
    assert pre.name == "Pre-No. 8 Special Announcement"
    assert pre.issued_at == datetime.fromisoformat("2026-08-21T18:15:00+08:00")


def test_no_pre_no8_without_the_statement():
    warnings = parse_warnings(
        {"TC": {"code": "TC"}},
        {"details": {"TC": {"code": "TC", "summary": "Gale Signal No. 8 NE in force"}}},
    )
    assert [w.code for w in warnings] == ["TC"]


def test_parse_warning_issued_time_does_not_use_later_update_time():
    warnings = parse_warnings(
        {
            "WHOT": {
                "code": "WHOT",
                "issueTime": "2026-08-05T06:45:00+08:00",
                "updateTime": "2026-08-13T06:45:00+08:00",
            }
        }
    )
    assert warnings[0].issued_at == datetime.fromisoformat("2026-08-05T06:45:00+08:00")
